"""Уточнение требований: агент-уточнитель и учёт ответов автора (цикл уточнения).

1. Clarifier превращает подтверждённые находки в короткий список вопросов автору: один вопрос — одно решение,
   с приоритетом, адресатом, вариантами ответа и допущением на случай, если ответа не будет.
2. Вопросы пишутся в файл-шаблон <отчёт>_questions.md. Автор вписывает ответы и запускает агента с --answers.
3. Resolver решает, какие находки ответ закрыл, какие нет; закрытые уходят из вердикта, по открытым — новый вопрос.
Код гарантирует: каждый вопрос привязан к реальной находке, каждый критичный дефект покрыт вопросом.
"""

from __future__ import annotations

import re
from dataclasses import replace
from difflib import SequenceMatcher

from reqcheck import prompts
from reqcheck.agents import _list, _s, numbered_document
from reqcheck.grounding import Grounder, normalize
from reqcheck.llm import JsonLLM
from reqcheck.models import CRITICAL, MAJOR, OK, SEV_RANK, VIOLATION, Document, Finding, Question, RequirementAssessment

PRIORITIES = ("blocker", "important", "clarify")
PRIORITY_RU = {"blocker": "блокирует разработку", "important": "нужно до тест-дизайна", "clarify": "можно уточнить по ходу"}
PRIORITY_ICON = {"blocker": "🔴", "important": "🟠", "clarify": "🟡"}
MAX_QUESTIONS = 20


SCATTER_GAP = 5  # строк: критичные дефекты дальше друг от друга — разные места документа


def scattered_blocker(q: Question, fmap: dict[str, Finding]) -> bool:
    """Вопрос связывает два и больше критичных дефекта, между строками которых больше SCATTER_GAP строк."""
    spots = []
    for fid in q.fids:
        f = fmap.get(fid)
        if f is not None and f.severity == CRITICAL:
            spots.append({x.line for x in f.quotes if x.line})
    spots = [s for s in spots if s]
    return any(min(abs(a - b) for a in s1 for b in s2) > SCATTER_GAP
               for i, s1 in enumerate(spots) for s2 in spots[i + 1:])


def _lines(fids: list[str], fmap: dict[str, Finding]) -> list[int]:
    out: list[int] = []
    for fid in fids:
        for q in fmap[fid].quotes if fid in fmap else []:
            if q.line and q.line not in out:
                out.append(q.line)
    return sorted(out)


DEFAULT_ASSUMPTION = {
    "blocker": "До ответа требование считается неготовым: разработку этой части не начинаем.",
    "important": "До ответа в тест-кейсах ожидаемый результат помечается как «требует уточнения».",
    "clarify": "До ответа берём формулировку документа как есть и фиксируем это в протоколе ревью.",
}


def complete_question(q: Question, fmap: dict[str, Finding]) -> Question:
    """Гарантия кодом: у каждого вопроса есть варианты ответа и допущение на случай, если ответа не будет.
    Если модель их не дала (или вопрос построен из находки правила), варианты берутся из исправления находки."""
    if not q.options:
        fix = next((fmap[x].suggestion for x in q.fids if x in fmap and fmap[x].suggestion), "")
        if fix:
            q.options = [f"Принять исправление: {fix[:250].rstrip(' .;')}",
                         "Оставить как есть: пояснить, почему это не дефект"]
    if not q.assumption:
        q.assumption = DEFAULT_ASSUMPTION.get(q.priority, DEFAULT_ASSUMPTION["important"])
    return q


def _number(questions: list[Question]) -> list[Question]:
    rank = {p: i for i, p in enumerate(PRIORITIES)}
    questions.sort(key=lambda q: rank.get(q.priority, 9))
    blockers = [q for q in questions if q.priority == "blocker"]
    rest = [q for q in questions if q.priority != "blocker"]
    kept = blockers + rest[:max(0, MAX_QUESTIONS - len(blockers))]  # блокирующие не режем никогда
    for i, q in enumerate(kept, start=1):
        q.qid = f"Q-{i:02d}"
    return kept


_STOP_STEMS = {"строк", "котор", "какой", "какая", "какие", "каков", "долже", "должн", "докум", "требо", "почем",
               "нужно", "можно", "также", "други", "этого", "чтобы", "сейча", "имени", "именн"}


def _stems(text: str) -> set[str]:
    """Грубые основы слов (первые 5 букв слов от 5 букв) — для сравнения тем двух вопросов."""
    return {w[:5] for w in re.findall(r"[a-zа-яё]{5,}", text.lower())} - _STOP_STEMS


def merge_contained(questions: list[Question], fmap: dict[str, Finding]) -> list[Question]:
    """Вопрос, все строки которого уже разбирает другой вопрос того же или более высокого приоритета, сливается в него.

    Модель иногда задаёт про одно место два вопроса: «почему GET вместо POST (строки 52, 103, 104, 120)» и «какую
    операцию выполняет метод (строка 103)». Автору это одно решение. Находки второго вопроса переходят к первому,
    так что каждая по-прежнему покрыта вопросом; порядок остальных вопросов не меняется.

    Одних общих строк мало: в строке может быть несколько разных дефектов («метод GET» и «идемпотентность» в одном
    пункте интеграции). Поэтому вопросы ещё должны говорить об одном и том же — не меньше двух общих основ слов."""
    rank = {p: i for i, p in enumerate(PRIORITIES)}
    merged: set[int] = set()
    for i, q in enumerate(questions):
        lines = set(q.lines)
        if not lines:
            continue
        host = next((h for j, h in enumerate(questions)
                     if j != i and j not in merged and lines <= set(h.lines)
                     and rank.get(h.priority, 9) <= rank.get(q.priority, 9)
                     and (len(h.lines) > len(lines) or j < i)
                     and len(_stems(q.text) & _stems(h.text)) >= 2), None)
        if host is None:
            continue
        host.fids += [f for f in q.fids if f not in host.fids]
        host.lines = _lines(host.fids, fmap)
        merged.add(i)
    return [q for i, q in enumerate(questions) if i not in merged]


# Технический вопрос (HTTP-коды, методы, URL, протоколы) решает не бизнес-заказчик, а архитектор
# (номера HTTP-кодов отдельно не ищем: «строка 126» выглядела бы как код ответа)
_TECH_TOPIC = re.compile(r"\bhttp\b|\brest\b|\bapi\b|эндпоинт|\burl\b|код\w* ответа|статус\w*[- ]?код"
                         r"|\b(?:get|post|put|patch|delete)\b|протокол|таймаут|\bjson\b|\bquery\b", re.I)


def route_addressee(q: Question) -> Question:
    if q.addressee.lower().startswith("бизнес") and _TECH_TOPIC.search(q.text):
        q.addressee = "Архитектор"
    return q


def questions_from_findings(findings: list[Finding]) -> list[Question]:
    """Запасной вариант без LLM: вопросы из полей question критичных и существенных находок."""
    fmap = {f.fid: f for f in findings}
    by_text: dict[str, Question] = {}
    for f in sorted(findings, key=lambda f: SEV_RANK.get(f.severity, 9)):
        if f.severity not in (CRITICAL, MAJOR):
            continue
        # У находки без черновика вопроса (например, от правила) вопрос строится из её заголовка —
        # так гарантия «каждый критичный дефект покрыт вопросом» не зависит от того, заполнила ли модель поле
        text = f.question or f"Как устранить: «{f.title}»? Какое значение или поведение считать верным?"
        key = normalize(text)[:80]
        if key in by_text:
            by_text[key].fids.append(f.fid)
            continue
        by_text[key] = Question(qid="", text=text, priority="blocker" if f.severity == CRITICAL else "important",
                                addressee=f.addressee or "Аналитик", why=f.explanation[:300], fids=[f.fid],
                                source="findings")
    qs = list(by_text.values())
    for q in qs:
        q.lines = _lines(q.fids, fmap)
        complete_question(q, fmap)
    return _number(qs)


def with_minor_batch(questions: list[Question], findings: list[Finding]) -> list[Question]:
    """Мелкие находки без вопроса — одним сводным вопросом, чтобы автор мог ответить и на них.

    Отдельный вопрос на каждую мелочь превратил бы список в шум, а без вопроса такие находки нельзя закрыть
    ответом. Строится кодом после Clarifier: промпт модели и кэш не меняются."""
    covered = {fid for q in questions for fid in q.fids}
    minor = [f for f in findings if f.severity not in (CRITICAL, MAJOR) and f.fid not in covered]
    if not minor:
        return questions
    fmap = {f.fid: f for f in findings}
    fids = [f.fid for f in minor]
    shown = ", ".join(fids[:12]) + (f" и ещё {len(fids) - 12}" if len(fids) > 12 else "")
    batch = Question(
        qid="", text=f"Мелкие замечания ({len(fids)}: {shown}): принять предложенные исправления или оставить формулировки документа?",
        priority="clarify", addressee="Аналитик",
        why="Мелкие неточности формулировок копятся и замедляют тест-дизайн; ответ позволяет закрыть их одним решением.",
        options=["Принять все предложенные исправления", "Принять, кроме перечисленных номеров находок",
                 "Оставить формулировки документа как есть"],
        assumption="Команда применяет предложенные исправления и фиксирует это в протоколе ревью.",
        fids=fids, lines=_lines(fids, fmap), source="findings")
    return _number([*questions, batch])


class Clarifier:
    """Агент-уточнитель: минимальный набор вопросов, ответы на которые снимают найденные дефекты."""

    def __init__(self, llm: JsonLLM, doc: Document):
        self.llm, self.doc = llm, doc

    def run(self, findings: list[Finding]) -> list[Question]:
        cand = [f for f in findings if f.severity in (CRITICAL, MAJOR) or f.question]
        if not cand:
            return []
        listing = "\n\n".join(
            f"{f.fid} [{f.severity}, требования: {', '.join(f.rids) or '—'}] {f.title}\n"
            f"Пояснение: {f.explanation}\n"
            f"Цитаты: " + " / ".join(f"«{q.text}» (строка {q.line})" for q in f.quotes)
            + (f"\nЧерновик вопроса ревьюера: {f.question}" if f.question else "")
            for f in cand
        )
        data = self.llm.generate_json(
            "clarifier", prompts.BASE_SYSTEM,
            f"{prompts.CLARIFIER}\nНаходки:\n{listing}\n\nВесь документ (номер строки: текст):\n{numbered_document(self.doc)}")
        fmap = {f.fid: f for f in findings}
        out: list[Question] = []
        for item in _list(data.get("questions") if isinstance(data, dict) else data):
            if not isinstance(item, dict):
                continue
            text = _s(item.get("question"), 600)
            fids = list(dict.fromkeys(x for x in (_s(v, 12) for v in _list(item.get("fids"))) if x in fmap))
            if not text or not fids:  # вопрос без привязки к реальной находке не принимаем
                continue
            pr = _s(item.get("priority"), 12).lower()
            if any(fmap[x].severity == CRITICAL for x in fids):
                pr = "blocker"  # критичный дефект всегда блокирует — модель не может понизить
            elif pr not in PRIORITIES:
                pr = "important"
            out.append(Question(
                qid="", text=text, priority=pr, addressee=_s(item.get("addressee"), 60) or "Аналитик",
                why=_s(item.get("why"), 400), options=[_s(o, 300).rstrip(" .;") for o in _list(item.get("options")) if _s(o)][:4],
                assumption=_s(item.get("assumption"), 400), fids=fids, lines=_lines(fids, fmap),
            ))
        # Блокирующий вопрос — про одно место документа: вопрос, который связал критичные дефекты из разных мест,
        # автор не закроет одним ответом. Такой вопрос не берём, его находки получают отдельные вопросы ниже.
        out = [q for q in out if not scattered_blocker(q, fmap)]
        # Каждый критичный дефект должен быть покрыт вопросом: недостающие добираем из находок
        covered = {fid for q in out for fid in q.fids}
        missing = [f for f in findings if f.severity in (CRITICAL, MAJOR) and f.fid not in covered]
        out += questions_from_findings(missing)
        out = merge_contained(out, fmap)
        return _number([route_addressee(complete_question(q, fmap)) for q in out])


# --- Файл ответов ---------------------------------------------------------------------------------
def render_questions_file(questions: list[Question], title: str, input_path: str, report_path: str, self_path: str,
                          sha256: str = "") -> str:
    L = [f"# Вопросы агента автору требований: {title}", ""]
    if sha256:
        L += [f"Документ: `{input_path}` (SHA-256 {sha256[:16]}…)", ""]
    L += [
        "Как ответить: впишите ответ после «Ответ:» (можно в несколько строк), сохраните файл и запустите повторно:",
        "",
        "```",
        f"python main.py --input \"{input_path}\" --output \"{report_path}\" --answers \"{self_path}\"",
        "```",
        "",
        "Пустые ответы агент пропускает. Ответы — данные для агента: он решит, какие дефекты они закрывают, "
        "и пересчитает вердикт.",
        "",
    ]
    for q in questions:
        L.append(f"## {q.qid} · {PRIORITY_RU.get(q.priority, q.priority)} · Кому: {q.addressee}")
        L.append("")
        L.append(f"Вопрос: {q.text}")
        if q.why:
            L.append(f"Почему важно: {q.why}")
        if q.options:
            L.append("Варианты: " + "; ".join(f"({i}) {o}" for i, o in enumerate(q.options, start=1)))
        if q.assumption:
            L.append(f"Если ответа не будет: {q.assumption}")
        refs = ", ".join(q.fids)
        if q.lines:
            refs += " (строки: " + ", ".join(map(str, q.lines)) + ")"
        L.append(f"Находки: {refs}")
        L.append(f"Ответ: {q.answer}".rstrip())
        L.append("")
    return "\n".join(L) + "\n"


_BLOCK = re.compile(r"^##\s+(Q-\d+(?:\.\d+)*)\b[^\n]*\n(.*?)(?=^##\s+Q-|\Z)", re.M | re.S)


def parse_answers(text: str) -> list[dict]:
    """Разобрать файл ответов: [{qid, question, answer}] только для непустых ответов."""
    out = []
    for m in _BLOCK.finditer(text or ""):
        body = m.group(2)
        qm = re.search(r"^Вопрос:\s*(.+)$", body, re.M)
        am = re.search(r"^Ответ:[ \t]*(.*)\Z", body, re.M | re.S)
        answer = (am.group(1) if am else "").strip()
        if answer:
            out.append({"qid": m.group(1), "question": qm.group(1).strip() if qm else "", "answer": answer[:2000]})
    return out


_SHA = re.compile(r"SHA-256\s+([0-9a-f]{8,64})", re.I)


def answers_sha(text: str) -> str:
    """Начало SHA-256 документа, к которому писались ответы (из шапки файла вопросов), или пустая строка."""
    m = _SHA.search((text or "")[:2000])
    return m.group(1).lower() if m else ""


def match_answers(answers: list[dict], questions: list[Question]) -> list[tuple[Question | None, dict]]:
    """Сопоставить ответы с вопросами текущего прогона: по номеру, если текст совпадает, иначе по тексту вопроса.

    Каждому вопросу — не больше одного ответа: второй ответ, похожий на уже занятый вопрос, остаётся без пары
    и уходит в Resolver как ответ на вопрос прошлого прогона (ничего не теряется и не перезаписывается)."""
    by_id = {q.qid: q for q in questions}
    used: set[str] = set()
    pairs = []
    for a in answers:
        sim = lambda x: SequenceMatcher(None, normalize(x.text), normalize(a["question"])).ratio()  # noqa: E731, B023 — вызывается в той же итерации
        q = by_id.get(a["qid"])
        if q is None or q.qid in used or (a["question"] and sim(q) < 0.6):
            free = [x for x in questions if x.qid not in used]
            best = max(free, key=sim, default=None)
            q = best if best is not None and a["question"] and sim(best) >= 0.6 else None
        if q:
            used.add(q.qid)
        pairs.append((q, a))
    return pairs


_NUM = re.compile(r"\d(?:[\d  ]*\d)?(?:[.,]\d+)?")


def _numbers(text: str) -> set[str]:
    return {re.sub(r"[\s ]", "", m.group(0)).replace(",", ".") for m in _NUM.finditer(text or "")}


def wording_matches_answer(wording: str, chosen: list[str], answer: str) -> bool:
    """Проверка кодом: значения, выбранные автором (дословно из ответа), должны быть в новой формулировке."""
    ans_norm, wording_norm = normalize(answer), normalize(wording)
    real = [c for c in chosen if c and normalize(c) in ans_norm]  # выдуманные «выбранные значения» не считаются
    need = set().union(*(_numbers(c) for c in real)) if real else set()
    if not need <= _numbers(wording):
        return False
    # Выбор без чисел («POST», «через SMS»): большинство слов выбранного варианта должно быть в формулировке
    for c in real:
        if _numbers(c):
            continue
        words = re.findall(r"\w{3,}", normalize(c))
        if words and sum(1 for w in words if w in wording_norm) * 2 < len(words) + 1:
            return False
    return True


def _strip_tags(text: str) -> str:
    """Ответ автора не может «закрыть» свой блок данных в промпте и выйти на уровень инструкций."""
    return re.sub(r"<\s*/?\s*ответ\s+автора\s*>", " ", text, flags=re.I)


def self_contradictions(data, answer_by_q: dict[str, str]) -> dict[str, tuple[str, str]]:
    """Ответы, которые модель назвала противоречащими самим себе: {qid: (часть 1, часть 2)}.

    Принимаются только пары, где обе части — разные фрагменты именно этого ответа (сверка кодом):
    выдуманная моделью «цитата» не может открыть находку, закрытую честным ответом."""
    out: dict[str, tuple[str, str]] = {}
    for c in _list(data.get("self_contradictions") if isinstance(data, dict) else None):
        if not isinstance(c, dict):
            continue
        qid, a, b = _s(c.get("qid"), 24), _s(c.get("part_a"), 300), _s(c.get("part_b"), 300)
        answer = normalize(answer_by_q.get(qid, ""))
        na, nb = normalize(a).strip(" .,;:«»\""), normalize(b).strip(" .,;:«»\"")
        if len(na) >= 5 and len(nb) >= 5 and na != nb and na not in nb and nb not in na \
                and na in answer and nb in answer:
            out.setdefault(qid, (a.strip(), b.strip()))
    return out


# Автор сам меняет документ — расхождение ответа с текстом документа не противоречие, а правка
_REPLACES = re.compile(r"замен|исключ|удал|убра|исправ|вместо|отмен", re.I)


def answer_conflicts(data, answer_by_q: dict[str, str], doc: Document | None) -> dict[str, tuple[str, str, str]]:
    """Конфликты, которые назвал агент «Проверка ответов» и подтвердил код: {qid: (часть ответа, вторая часть, где)}.

    part_a обязана дословно быть в этом ответе; part_b — в нём же (self), в другом ответе (other_answer)
    или в документе (document; не принимается, если автор в ответе сам заменяет или исключает старое)."""
    out: dict[str, tuple[str, str, str]] = {}
    grounder = Grounder(doc) if doc is not None else None
    norm = {k: normalize(v) for k, v in answer_by_q.items()}

    def clean(x: str) -> str:
        return normalize(x).strip(" .,;:«»\"")

    for c in _list(data.get("conflicts") if isinstance(data, dict) else None):
        if not isinstance(c, dict):
            continue
        qid, kind = _s(c.get("qid"), 24), _s(c.get("kind"), 20).lower()
        a, b = _s(c.get("part_a"), 300), _s(c.get("part_b"), 300)
        na, nb = clean(a), clean(b)
        if qid not in norm or qid in out or len(na) < 5 or len(nb) < 5 or na not in norm[qid]:
            continue
        if kind == "self":
            if na != nb and na not in nb and nb not in na and nb in norm[qid]:
                out[qid] = (a.strip(), b.strip(), "сам себе")
        elif kind == "other_answer":
            other = _s(c.get("other_qid"), 24)
            if other != qid and other in norm and nb in norm[other]:
                out[qid] = (a.strip(), b.strip(), f"ответу на {other}")
        elif kind == "document" and grounder is not None and na not in nb and nb not in na                 and not _REPLACES.search(answer_by_q[qid]):
            loc = grounder.locate(b)
            if loc.found and loc.line:
                out[qid] = (a.strip(), b.strip(), f"документу (строка {loc.line})")
    return out


class Resolver:
    """Учёт ответов автора: какие находки ответ закрыл, какие частично, какие остались (с новым вопросом)."""

    def __init__(self, llm: JsonLLM, doc: Document):
        self.llm, self.doc = llm, doc
        self.conflicts: dict[str, tuple[str, str, str]] = {}  # подтверждённые кодом конфликты ответов (агент проверки)

    def check_answers(self, pairs: list[tuple[Question | None, dict]]) -> dict[str, tuple[str, str, str]]:
        """Отдельный запрос «Проверка ответов». Сбой не мешает учёту ответов: тогда конфликтов просто нет."""
        qa = "\n\n".join(f"{self._key(q, a)}. Вопрос: {q.text if q else a['question']}\n"
                          f"<ответ автора>\n{_strip_tags(a['answer'])}\n</ответ автора>" for q, a in pairs)
        try:
            data = self.llm.generate_json(
                "answer_check", prompts.BASE_SYSTEM,
                f"{prompts.ANSWER_CHECK}\nВопросы и ответы автора:\n{qa}"
                f"\n\nВесь документ (номер строки: текст):\n{numbered_document(self.doc)}")
        except Exception:  # noqa: BLE001 — проверка дополнительная, основной учёт ответов идёт без неё
            return {}
        return answer_conflicts(data, {self._key(q, a): a["answer"] for q, a in pairs}, self.doc)

    @staticmethod
    def _key(q: Question | None, a: dict) -> str:
        # Ответ на вопрос прошлого прогона может носить тот же номер, что и вопрос этого прогона, — разводим их
        return q.qid if q else f"прошлый-{a['qid']}"

    def run(self, pairs: list[tuple[Question | None, dict]], findings: list[Finding]) -> list[dict]:
        fmap = {f.fid: f for f in findings}
        linked = {fid for q, _ in pairs if q for fid in q.fids}
        unmatched = any(q is None for q, _ in pairs)
        scope = [f for f in findings if f.fid in linked or (unmatched and f.severity in (CRITICAL, MAJOR))]
        if not scope:
            return []
        self.conflicts = self.check_answers(pairs)
        effects = self._ask(pairs, scope, fmap)
        # Модель могла пропустить находку отвеченного вопроса (так бывает с лёгкими моделями на длинном списке).
        # Переспрашиваем один раз, только по пропущенным находкам: короткий запрос модель отрабатывает полностью.
        missing = self._missing(pairs, effects, fmap)
        if missing:
            retry = [(replace(q, fids=missing[q.qid]), a) for q, a in pairs if q and q.qid in missing]
            try:
                effects += self._ask(retry, [fmap[fid] for fids in missing.values() for fid in fids], fmap)
            except Exception:  # noqa: BLE001 — первый ответ уже есть; сбой переспроса не должен его терять
                pass
        # Что осталось без решения, молча открытым не оставляем: автор ответил, а в отчёте нет следа.
        # Помечаем «без решения», находка остаётся в вердикте.
        effects += [{"fid": fid, "qid": qid, "status": "undecided", "comment": "", "revised_wording": "", "follow_up": ""}
                    for qid, fids in self._missing(pairs, effects, fmap).items() for fid in fids]
        return effects

    @staticmethod
    def _missing(pairs: list[tuple[Question | None, dict]], effects: list[dict], fmap: dict) -> dict[str, list[str]]:
        """Находки отвеченных вопросов этого прогона, по которым модель не вернула решения: {qid: [fid…]}."""
        decided = {(e["qid"], e["fid"]) for e in effects}
        out: dict[str, list[str]] = {}
        for q, _ in pairs:
            if q:
                fids = [fid for fid in q.fids if fid in fmap and (q.qid, fid) not in decided]
                if fids:
                    out[q.qid] = fids
        return out

    def _ask(self, pairs: list[tuple[Question | None, dict]], scope: list[Finding], fmap: dict) -> list[dict]:
        """Один запрос к модели и проверка её решений кодом."""
        qa = "\n\n".join(
            f"{self._key(q, a)}. Вопрос: {q.text if q else a['question']}\n"
            f"Связанные находки: {', '.join(q.fids) if q else 'определи сам по смыслу'}\n"
            f"<ответ автора>\n{_strip_tags(a['answer'])}\n</ответ автора>"
            for q, a in pairs
        )
        listing = "\n\n".join(
            f"{f.fid} [{f.severity}] {f.title}\nПояснение: {f.explanation}\n"
            f"Цитаты: " + " / ".join(f"«{q.text}» (строка {q.line})" for q in f.quotes)
            for f in scope
        )
        data = self.llm.generate_json(
            "resolver", prompts.BASE_SYSTEM,
            f"{prompts.RESOLVER}\nВопросы и ответы автора:\n{qa}\n\nНаходки:\n{listing}"
            f"\n\nВесь документ (номер строки: текст):\n{numbered_document(self.doc)}")
        valid_q = {self._key(q, a) for q, a in pairs}
        answer_by_q = {self._key(q, a): a["answer"] for q, a in pairs}
        orig_qid = {self._key(q, a): (q.qid if q else a["qid"]) for q, a in pairs}
        # Ответ на вопрос этого прогона может решать только находки этого вопроса; ответ на вопрос
        # прошлого прогона (без пары) — любую находку из списка, переданного модели
        allowed = {q.qid: set(q.fids) for q, _ in pairs if q}
        clash = {k: (a, b, "сам себе") for k, (a, b) in self_contradictions(data, answer_by_q).items()}
        for k, v in self.conflicts.items():
            clash.setdefault(k, v)
        asked: set[str] = set()
        effects = []
        for e in _list(data.get("effects") if isinstance(data, dict) else data):
            if not isinstance(e, dict):
                continue
            fid, qid = _s(e.get("fid"), 12), _s(e.get("qid"), 24)
            status = _s(e.get("status"), 20).lower()
            if status not in ("resolved", "partial", "open") and _s(e.get("comment")):
                # Нестандартный статус (например, «revised_wording»): решение есть, но закрывать по нему нельзя —
                # считаем «частично снято», находка остаётся в вердикте, комментарий модели виден в отчёте
                status = "partial"
            if fid not in fmap or qid not in valid_q or status not in ("resolved", "partial", "open"):
                continue
            if qid in allowed and fid not in allowed[qid]:
                continue  # модель приписала ответу находку, о которой вопрос не спрашивал
            if qid not in allowed and fid not in {f.fid for f in scope}:
                continue
            comment = _s(e.get("comment"), 500)
            if status == "resolved" and not comment:
                status = "partial"  # закрыть дефект без обоснования нельзя
            wording = _s(e.get("revised_wording"), 800)
            if wording and not wording_matches_answer(wording, [_s(v, 100) for v in _list(e.get("chosen_values"))],
                                                      answer_by_q.get(qid, "")):
                wording = ""  # формулировка модели расходится с выбором автора — не показываем её как исправление
                comment += " Новую формулировку модель предложила с расхождением с ответом автора, она отклонена кодом."
            follow_up = _s(e.get("follow_up"), 600)
            if qid in clash and status in ("resolved", "partial"):
                # Ответ сам себе противоречит (обе части найдены в ответе дословно): закрывать нельзя,
                # иначе модель молча выбрала бы одно из значений за автора. Уточнение — одно на ответ.
                a_part, b_part, where = clash[qid]
                status, wording = "open", ""
                comment = f"ответ противоречит {where}: «{a_part}» и «{b_part}»"
                follow_up = "" if qid in asked else (
                    f"Ответ противоречит {where}: «{a_part}» и «{b_part}». Какое из утверждений верно?")
            if status == "open" and follow_up:
                asked.add(qid)
            # В отчёт и файл ответов — исходный номер вопроса
            effects.append({"fid": fid, "qid": orig_qid.get(qid, qid), "status": status, "comment": comment,
                            "revised_wording": wording, "follow_up": follow_up})
        return effects


def apply_effects(effects: list[dict], findings: list[Finding], questions: list[Question]) -> tuple[set[str], list[Question]]:
    """Применить решения Resolver к находкам (код, не модель).

    Решения по одной находке от разных ответов сводятся вместе: находка закрыта, только если ни один ответ
    не оставил её открытой. По открытым — уточняющий вопрос (Q-xx.1). Возвращает (закрытые fid, уточнения)."""
    fmap = {f.fid: f for f in findings}
    qmap = {q.qid: q for q in questions}
    by_fid: dict[str, list[dict]] = {}
    for e in effects:
        by_fid.setdefault(e["fid"], []).append(e)
    resolved: set[str] = set()
    followups: list[Question] = []
    for fid, items in by_fid.items():
        f = fmap[fid]
        notes = []
        for e in items:
            if e["status"] == "resolved":
                notes.append(f"закрыто ответом {e['qid']}: {e['comment']}")
            elif e["status"] == "undecided":
                notes.append(f"по ответу {e['qid']} Resolver решения не вынес, проверьте вручную")
            elif e["status"] == "partial":
                notes.append(f"частично снято ответом {e['qid']}: {e['comment']}")
            else:
                notes.append(f"ответ {e['qid']} не снимает дефект: {e['comment']}")
                if e["follow_up"]:
                    base = qmap.get(e["qid"])
                    n = sum(1 for x in followups if x.qid.startswith(e["qid"] + ".")) + 1
                    followups.append(Question(
                        qid=f"{e['qid']}.{n}", text=e["follow_up"],
                        priority=base.priority if base else ("blocker" if f.severity == CRITICAL else "important"),
                        addressee=base.addressee if base else (f.addressee or "Аналитик"), fids=[f.fid],
                        lines=[q.line for q in f.quotes if q.line], source="resolver"))
        f.resolution = "; ".join(notes)
        # «Без решения» не мешает закрытию: если другой ответ находку закрыл, а модель промолчала — она закрыта
        if {e["status"] for e in items} - {"undecided"} == {"resolved"}:
            wording = next((e["revised_wording"] for e in items if e["revised_wording"]), "")
            if wording:
                f.suggestion = wording
            resolved.add(fid)
    for q in questions:
        if q.answer:
            closed = sum(1 for fid in q.fids if fid in resolved)
            q.outcome = f"закрыто находок: {closed} из {len(q.fids)}"
    return resolved, followups


def close_marks(resolved: list[Finding], remaining: list[Finding],
                assessments: dict[str, RequirementAssessment]) -> None:
    """Находка закрыта ответом — её критерий у требования больше не нарушен (если по нему нет других открытых находок).
    Иначе таблица оценок и балл противоречили бы вердикту, пересчитанному после ответов."""
    still = {(rid, f.criterion) for f in remaining for rid in f.rids}
    for f in resolved:
        for rid in f.rids:
            a = assessments.get(rid)
            if a is None or (rid, f.criterion) in still:
                continue
            for m in a.marks:
                if m.criterion == f.criterion and m.status == VIOLATION:
                    m.status, m.comment = OK, f"закрыто ответом автора ({f.resolution[:120]})"


def carry_over(unmatched: list[dict], followups: list[Question]) -> list[Question]:
    """Ответы на вопросы, которых нет в этом прогоне (например, уточнения прошлого прогона), не теряем:
    они попадают в отчёт и обратно в файл ответов, который перезаписывается рядом с отчётом."""
    by_qid = {q.qid: q for q in followups}
    out = []
    for a in unmatched:
        if a["qid"] in by_qid:
            by_qid[a["qid"]].answer = a["answer"]  # то же уточнение снова задано — сохраняем ответ при нём
            continue
        out.append(Question(qid=a["qid"], text=a["question"] or "(вопрос из прошлого прогона)", priority="clarify",
                            answer=a["answer"], source="answers_file",
                            outcome="вопрос из прошлого прогона: ответ передан агенту учёта ответов"))
    return out
