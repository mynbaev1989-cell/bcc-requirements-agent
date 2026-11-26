"""Детерминированные проверки (работают без LLM и служат префильтром).

Ничего не знают о конкретном кейсе: только общие признаки плохих требований —
размытые слова, заглушки, пустые разделы, незаполненные чек-листы и таблицы,
смешение единиц измерения, отсутствие версии, дубли идентификаторов.
"""

from __future__ import annotations

import re
from collections import Counter

from reqcheck.models import MAJOR, MINOR, Block, ChecklistItem, Document, Finding, Quote, Requirement
from reqcheck.segmenter import FENCE, HEADING_MD

# Размытые («слабые») слова и обороты — признак неоднозначности/непроверяемости (29148 п. 5.2.5, IEEE 830 п. 4.3.2)
WEAK_PATTERNS: list[tuple[str, str]] = [
    (r"\bбыстр(о|ый|ая|ое|ые)\b", "быстро"),
    (r"\bудобн\w*", "удобно"),
    (r"\bоптимальн\w*", "оптимально"),
    (r"\bэффективн\w*", "эффективно"),
    (r"\bпо возможности\b", "по возможности"),
    (r"\bпри необходимости\b", "при необходимости"),
    (r"\bв случае необходимости\b", "в случае необходимости"),
    (r"\bи т\.\s?д\.?", "и т. д."),
    (r"\bи т\.\s?п\.?", "и т. п."),
    (r"\bи тому подобное\b", "и тому подобное"),
    (r"\bи друг(ие|их)\b", "и другие"),
    (r"\bдостаточн\w*", "достаточно"),
    (r"\bсвоевременн\w*", "своевременно"),
    (r"\bинтуитивн\w*", "интуитивно"),
    (r"\bгибк(ий|о|ая|ое|ие)\b", "гибкий"),
    (r"\bнадёжн\w*|\bнадежн\w*", "надёжно"),
    (r"\bкак правило\b", "как правило"),
    (r"\bобычно\b", "обычно"),
    (r"\bпримерно\b|\bоколо\b|\bприблизительно\b", "примерно"),
    (r"\bминимальн\w* (время|задержк)\w*", "минимальное время"),
    (r"\bмаксимально\b", "максимально"),
    (r"\bсовременн\w*", "современный"),
    (r"\bпростой\b|\bпросто\b", "просто"),
    (r"\bи/или\b", "и/или"),
    (r"\b(TBD|TBA|TODO)\b", "TBD"),
    (r"\buser[- ]friendly\b|\bas appropriate\b|\betc\.?", "размытый оборот (EN)"),
]

# Текст, обращённый к ИИ-проверяющему (prompt injection): документ — данные, но такие строки стоит показать человеку.
# Модели это не обходит до конца, поэтому защита многослойная: правила и сверка цитат от модели не зависят.
INJECTION_RE = re.compile(
    r"(игнорир\w*|забудь\w*|не\s+учитыва\w*)\s+(все\s+|всех\s+)?(предыдущ|прошл|выше|системн)\w*\s+(инструкц|указани|правил|промпт)"
    r"|ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts?)"
    r"|\b(ты|вы)\s+[—–-]?\s*(язык\w*\s+модел\w*|ии|нейросеть|ai|llm)\b"
    r"|\b(system\s+prompt|системн\w+\s+промпт)"
    r"|\bне\s+(сообщай|сообщайте|указывай|указывайте|находи|ищи)\w*\s+(о\s+)?(дефект|ошибк|нарушени|замечани)"
    r"|\b(дефектов|ошибок|замечаний)\s+(в\s+(этом|данном)\s+документе\s+)?нет\b.{0,40}(ответь|верни|сообщи|выведи)",
    re.IGNORECASE,
)


def injection_lines(doc: Document) -> list[int]:
    """Номера строк, похожих на инструкции для ИИ внутри документа требований."""
    return [i for i, line in enumerate(doc.lines, start=1) if INJECTION_RE.search(line)]


PLACEHOLDER = re.compile(
    # Только целые фразы-заглушки: «(раздел будет заполнен позднее)», «В разработке», «TBD».
    # Обычный текст вроде «Будет заполнена анкета клиента…» сюда не попадает.
    r"^\s*\(?\s*(заполня(ется|ются)\b.*|tbd|tba|todo|будет (определено|уточнено|описано).*"
    r"|(раздел|пункт|информация|содержание|данные)?\s*(будет|находится)?\s*(в\s+)?"
    r"(заполнен[оаы]?|добавлен[оаы]?|дописан[оаы]?|разработке|подготовке|доработке)"
    r"(\s+(позднее|позже|в следующей версии))?"
    r"|уточнить.*|уточняется|n/?a|—|-|\.\.\.)\s*\)?\.?\s*$",
    re.I,
)

# Единицы одной величины: основная и дробная
UNIT_GROUPS = {
    "деньги": {
        "основная": re.compile(r"\bтенге\b|₸|\bKZT\b|\bруб(\.|лей|ли|ль)?\b|\bUSD\b|\$|\bдоллар\w*", re.I),
        # Только формы денежных единиц: «центр», «центральный», «Копенгаген» сюда не попадают
        "дробная": re.compile(r"\bтиын(а|ов|ы|ах)?\b|\bкопе(ек|йк[аиуеой]\w*)\b|\bцент(а|ов|ы|ах)?\b", re.I),
    },
    "время": {
        "основная": re.compile(r"\b(секунд\w*|сек\.?|минут\w*|мин\.?|час(а|ов)?)\b", re.I),
        "дробная": re.compile(r"\b(миллисекунд\w*|мс)\b", re.I),
    },
}

VERSION_RE = re.compile(r"(верси[яи](?:\s+документа)?|version|ред\.|редакци[яи])\s*[:№#]?\s*v?\d+(\.\d+)*", re.I)
DATE_RE = re.compile(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
YES_NO_CELL = re.compile(r"^\s*(да|нет)(\s*/\s*(да|нет|не требуется|н/п))+\s*$", re.I)


class RuleEngine:
    def __init__(self, doc: Document, blocks: list[Block]):
        self.doc = doc
        self.blocks = blocks
        self._n = 0

    def _fid(self) -> str:
        self._n += 1
        return f"F-R{self._n:03d}"

    def _finding(self, **kw) -> Finding:
        kw.setdefault("source", "rule")
        kw.setdefault("verified", True)
        return Finding(fid=self._fid(), **kw)

    # --- Проверки отдельных требований ---------------------------------------------------------
    def weak_words(self, reqs: list[Requirement]) -> list[Finding]:
        out: list[Finding] = []
        for r in reqs:
            hits: list[tuple[str, int]] = []
            for ln in range(r.line_start, r.line_end + 1):
                line = self.doc.lines[ln - 1] if 0 < ln <= len(self.doc.lines) else ""
                for pattern, label in WEAK_PATTERNS:
                    for m in re.finditer(pattern, line, re.I):
                        hits.append((m.group(0), ln))
                        _ = label
            seen = set()
            for word, ln in hits:
                key = word.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(self._finding(
                    scope="requirement", criterion="unambiguous", severity=MINOR, rids=[r.rid],
                    title=f"Размытая формулировка «{word}»",
                    explanation=f"Слово «{word}» допускает разное толкование и не задаёт измеримого критерия.",
                    quotes=[Quote(word, ln, True)],
                    suggestion="Заменить на измеримое значение или явное условие.",
                    question=f"Что именно означает «{word}» в {r.source_id or r.rid}? Укажите измеримое значение.",
                    addressee="Аналитик",
                ))
        return out

    def duplicate_ids(self, reqs: list[Requirement]) -> list[Finding]:
        counts = Counter(r.source_id for r in reqs if not r.source_id_assigned)
        out = []
        for sid, cnt in counts.items():
            if cnt > 1:
                same = [r for r in reqs if r.source_id == sid]
                # Несколько атомарных требований из одного пункта — это не дубль
                if len({r.line_start for r in same}) == 1:
                    continue
                out.append(self._finding(
                    scope="set", criterion="set_traceable", severity=MAJOR, rids=[r.rid for r in same],
                    title=f"Идентификатор {sid} встречается {cnt} раз",
                    explanation="Неуникальный идентификатор ломает трассируемость и ссылки на требование.",
                    quotes=[Quote(sid, r.line_start, True) for r in same],
                    suggestion="Присвоить уникальные идентификаторы.", question=f"Какой из пунктов {sid} считать основным?",
                    addressee="Аналитик",
                ))
        return out

    def missing_ids(self, reqs: list[Requirement]) -> list[Finding]:
        """Одна сводная находка вместо десятков «нет идентификатора» по каждому пункту."""
        without = [r for r in reqs if r.source_id_assigned]
        if not without:
            return []
        first = without[0]
        listed = ", ".join(r.rid for r in without[:30]) + ("…" if len(without) > 30 else "")
        return [self._finding(
            scope="set", criterion="set_traceable", severity=MINOR if len(without) < len(reqs) else MAJOR,
            rids=[r.rid for r in without],
            title=f"{len(without)} из {len(reqs)} требований без идентификатора в документе",
            explanation=f"На требования без номера нельзя однозначно сослаться в задачах, тестах и при изменениях: {listed}.",
            quotes=[Quote(first.quote[:120], first.line_start, True)],
            suggestion="Присвоить всем требованиям, критериям приёмки и правилам API уникальные идентификаторы (например НФТ-1, ДАН-1, API-1, КП-1).",
            question="Можно ли пронумеровать все требования единым образом?", addressee="Аналитик",
        )]

    # --- Проверки набора -----------------------------------------------------------------------
    def placeholders(self) -> list[Finding]:
        out = []
        for b in self.blocks:
            if b.kind in ("paragraph", "list_item") and PLACEHOLDER.match(b.text.strip("* ")):
                out.append(self._finding(
                    scope="set", criterion="set_complete", severity=MAJOR,
                    title=f"Раздел «{b.section.split(' › ')[-1]}» не заполнен",
                    explanation=f"Вместо содержания стоит заглушка «{b.text}».",
                    quotes=[Quote(b.text, b.line_start, True)],
                    suggestion="Заполнить раздел до передачи в разработку или явно указать, что он не требуется.",
                    question=f"Кто и когда заполнит раздел «{b.section.split(' › ')[-1]}»?", addressee="Аналитик / QA",
                ))
        return out

    def empty_sections(self) -> list[Finding]:
        """Заголовок, за которым сразу идёт заголовок того же или более высокого уровня."""
        out = []
        heading = HEADING_MD
        heads, in_code = [], False
        for i, line in enumerate(self.doc.lines):
            if FENCE.match(line):
                in_code = not in_code  # «# комментарий» внутри блока кода — не заголовок
                continue
            if not in_code and (m := heading.match(line)):
                heads.append((i, len(m.group(1)), m.group(2)))
        for k, (i, lvl, title) in enumerate(heads):
            nxt = heads[k + 1] if k + 1 < len(heads) else None
            end = nxt[0] if nxt else len(self.doc.lines)
            body = [ln for ln in self.doc.lines[i + 1:end] if ln.strip() and not re.match(r"^\s*([-*_])(\s*\1){2,}\s*$", ln)]
            if not body and (nxt is None or nxt[1] <= lvl):
                out.append(self._finding(
                    scope="set", criterion="set_complete", severity=MAJOR,
                    title=f"Пустой раздел «{title}»", explanation="У раздела нет содержания.",
                    quotes=[Quote(title, i + 1, True)], suggestion="Заполнить раздел или удалить заголовок.",
                    question=f"Что должно быть в разделе «{title}»?", addressee="Аналитик",
                ))
        return out

    def empty_tables(self) -> list[Finding]:
        """Строки таблиц, где все ячейки пустые (например, незаполненный блок согласующих)."""
        out = []
        for b in self.blocks:
            if b.kind == "table_row" and b.cells is not None and all(not c.strip() for c in b.cells):
                sec = b.section.split(" › ")[-1]
                out.append(self._finding(
                    scope="set", criterion="modifiable", severity=MINOR,
                    title=f"Таблица «{sec}» не заполнена",
                    explanation="Строка таблицы пустая: нет данных об ответственных/значениях.",
                    quotes=[Quote(sec, self._heading_line(sec) or b.line_start, True)],
                    suggestion="Заполнить таблицу (ответственные, подписи, даты).",
                    question=f"Кто согласует документ (таблица «{sec}»)?", addressee="Владелец продукта",
                ))
        return out

    def _heading_line(self, title: str) -> int | None:
        for i, line in enumerate(self.doc.lines):
            if line.lstrip("# ").strip() == title:
                return i + 1
        return None

    def _passport_anchor(self) -> int:
        """Строка паспорта документа (первая таблица в начале документа), куда ожидается версия.

        Строка «Статус» внутри паспорта предпочтительна; таблицы полей API и прочие таблицы ниже не берём.
        """
        rows: list[int] = []
        for i, line in enumerate(self.doc.lines[:40]):
            if line.lstrip().startswith("|"):
                rows.append(i + 1)
            elif rows:
                break  # первая таблица закончилась
        data_rows = [n for n in rows if not re.match(r"^\s*\|[\s:|-]+\|?\s*$", self.doc.lines[n - 1])][1:]  # без шапки
        status = [n for n in data_rows if re.search(r"^\s*\|\s*(статус|status)\s*\|", self.doc.lines[n - 1], re.I)]
        if status:
            return status[0]
        return data_rows[-1] if data_rows else 1

    def missing_version(self) -> list[Finding]:
        head = "\n".join(self.doc.lines[:40])
        tech = re.compile(r"\b(tls|ssl|api|http|протокол\w*|библиотек\w*|java|python|ios|android|ос|sdk|openapi)\W{0,3}$", re.I)
        doc_versions = [mt for mt in VERSION_RE.finditer(self.doc.text)
                        if not tech.search(self.doc.text[max(0, mt.start() - 20):mt.start()].rstrip())]
        if doc_versions or (DATE_RE.search(head) and re.search(r"верси[яи]\s+документа|^\s*\|?\s*(верси[яи]|version)\b", head, re.I | re.M)):
            return []
        # Якорь — место, где версию ждут: строка «версия», иначе паспорт документа (строка «Статус»), иначе начало
        doc_version = re.compile(r"документ[^.\n]{0,20}верси|верси[яи]\s*(документа|/)|^\s*\|?\s*(версия|version)\b", re.I)
        anchor_line = next((i + 1 for i, line in enumerate(self.doc.lines) if doc_version.search(line)), None)
        if anchor_line is None:
            anchor_line = self._passport_anchor()
        anchor = self.doc.lines[anchor_line - 1].strip()[:120] or self.doc.title
        return [self._finding(
            scope="set", criterion="modifiable", severity=MAJOR,
            title="Не указаны версия и дата документа",
            explanation="Без версии нельзя понять, какую редакцию согласовали и проверяли (управление изменениями).",
            quotes=[Quote(anchor, anchor_line, True)],
            suggestion="Добавить в шапку версию, дату и историю изменений.",
            question="Какая версия и дата у этой редакции документа?", addressee="Аналитик",
        )]

    def mixed_units(self) -> list[Finding]:
        out = []
        for group, pats in UNIT_GROUPS.items():
            major = [(i + 1, m.group(0)) for i, line in enumerate(self.doc.lines) if (m := pats["основная"].search(line))]
            minor = [(i + 1, m.group(0)) for i, line in enumerate(self.doc.lines) if (m := pats["дробная"].search(line))]
            if major and minor:
                (l1, w1), (l2, w2) = major[0], minor[0]
                # если на строке с дробной единицей есть и основная — берём её как пару
                for ln, w in major:
                    if ln == l2:
                        l1, w1 = ln, w
                        break
                out.append(self._finding(
                    scope="set", criterion="set_consistent", severity=MINOR,
                    title=f"Смешение единиц ({group}): «{w1}» и «{w2}»",
                    explanation=f"В документе одновременно используются «{w1}» и «{w2}» — риск ошибки пересчёта и сравнения.",
                    quotes=[Quote(w1, l1, True), Quote(w2, l2, True)],
                    suggestion="Зафиксировать единую единицу хранения и правила пересчёта/округления.",
                    question=f"В каких единицах хранить и сравнивать значения ({group})?", addressee="Аналитик / архитектор",
                ))
        return out

    # --- Чек-листы ревью (для приложения B) ------------------------------------------------------
    def find_checklists(self) -> list[ChecklistItem]:
        """Строки таблиц с незаполненными вариантами «Да / Нет» — чек-лист ревью из самого документа."""
        items = []
        for b in self.blocks:
            if b.kind != "table_row" or not b.cells:
                continue
            yes_no = [c for c in b.cells if YES_NO_CELL.match(c)]
            if yes_no and b.cells[0].strip() and not YES_NO_CELL.match(b.cells[0]):
                items.append(ChecklistItem(question=b.cells[0].strip(), line=b.line_start))
        return items

    def unfilled_checklist(self, items: list[ChecklistItem]) -> list[Finding]:
        if not items:
            return []
        return [self._finding(
            scope="set", criterion="set_validatable", severity=MAJOR,
            title=f"Чек-лист ревью не заполнен (пунктов: {len(items)})",
            explanation="Отметки «Да/Нет» не проставлены: нет подтверждения, что ревью требований проводилось.",
            quotes=[Quote(items[0].question, items[0].line, True)],
            suggestion="Провести ревью и заполнить чек-лист (черновик — в приложении B отчёта).",
            question="Кто проводит ревью по чек-листу и когда?", addressee="Аналитик / QA",
        )]

    def run_set_rules(self) -> tuple[list[Finding], list[ChecklistItem]]:
        checklist = self.find_checklists()
        findings = (self.placeholders() + self.empty_sections() + self.empty_tables() + self.missing_version()
                    + self.mixed_units() + self.unfilled_checklist(checklist))
        return findings, checklist
