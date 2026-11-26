"""Роли агента поверх детерминированного оркестратора.

Extractor → Reviewer → Consistency → Critic → Analyst (чек-лист, трассировка, тесты).
Каждая роль — один или несколько вызовов LLM с JSON-ответом. Ответ модели не доверяется слепо:
коды критериев нормализуются, неизвестные поля игнорируются, цитаты проверяет Grounder.
"""

from __future__ import annotations

import json
import os
from difflib import SequenceMatcher
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from reqcheck import criteria, prompts
from reqcheck.grounding import Grounder, normalize
from reqcheck.llm import JsonLLM, LLMError
from reqcheck.models import (
    MAJOR, NA, OK, SEV_RANK, SEVERITIES, VIOLATION, Block, ChecklistItem, CriterionMark, Document, Finding,
    Requirement, RequirementAssessment, TestDraft, TraceRow,
)

EXTRACT_CHUNK_BLOCKS = 120
REVIEW_BATCH = int(os.environ.get("REQCHECK_REVIEW_BATCH", "8") or 8)  # больше пакет — меньше запросов к LLM
CRITIC_BATCH = 40
KINDS = {"functional", "nonfunctional", "data", "integration", "ui", "api", "acceptance", "other"}


def _s(value: Any, limit: int = 1200) -> str:
    """Безопасно привести поле ответа модели к строке ограниченной длины."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    return value.strip()[:limit]


def _list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    return [value]


_NUMBERED_CACHE: dict[tuple[str, int], str] = {}


def numbered_document(doc: Document, max_chars: int = 180_000) -> str:
    """Документ с номерами строк для промптов. Строится один раз на документ (его шлют 7+ агентов)."""
    key = (doc.sha256 or str(id(doc)), max_chars)
    if key not in _NUMBERED_CACHE:
        _NUMBERED_CACHE[key] = _number_lines(doc, max_chars)
    return _NUMBERED_CACHE[key]


def visible_lines(doc: Document, max_chars: int = 180_000) -> int:
    """Сколько строк документа целиком помещается в промпт (остальное модели не видят)."""
    size = 0
    for i, line in enumerate(doc.lines, start=1):
        size += len(f"{i:>4}: {line}") + 1
        if size > max_chars:
            return i - 1
    return len(doc.lines)


def _number_lines(doc: Document, max_chars: int) -> str:
    out, size = [], 0
    for i, line in enumerate(doc.lines, start=1):
        row = f"{i:>4}: {line}"
        size += len(row) + 1
        if size > max_chars:
            out.append("… (документ обрезан по лимиту контекста)")
            break
        out.append(row)
    return "\n".join(out)


# --------------------------------------------------------------------------------------------------
class Extractor:
    def __init__(self, llm: JsonLLM, doc: Document, blocks: list[Block]):
        self.llm, self.doc, self.blocks = llm, doc, blocks
        self.by_id = {b.bid: b for b in blocks}

    def run(self) -> list[Requirement]:
        found: list[tuple[int, Requirement]] = []
        for start in range(0, len(self.blocks), EXTRACT_CHUNK_BLOCKS):
            chunk = self.blocks[start:start + EXTRACT_CHUNK_BLOCKS]
            listing = "\n".join(
                f"{b.bid} [{b.kind}, строки {b.line_start}-{b.line_end}, раздел: {b.section}]\n{b.text}" for b in chunk
            )
            data = self.llm.generate_json("extractor", prompts.BASE_SYSTEM, prompts.EXTRACTOR + listing)
            for item in _list(data.get("requirements") if isinstance(data, dict) else data):
                if not isinstance(item, dict):
                    continue
                req = self._to_requirement(item)
                if req:
                    found.append((req.line_start, req))
        found.sort(key=lambda x: x[0])
        return [r for _, r in found]

    def _to_requirement(self, item: dict) -> Requirement | None:
        bids = [b for b in (_s(x, 20) for x in _list(item.get("block_ids"))) if b in self.by_id]
        if not bids:
            return None
        blocks = [self.by_id[b] for b in bids]
        block_text = " ".join(b.text for b in blocks)
        quote = _s(item.get("quote"), 600)
        # Цитата должна быть внутри указанного блока; иначе берём начало блока — ID и строки остаются достоверными
        if not quote or normalize(quote).strip(" .") not in normalize(block_text):
            quote = blocks[0].text[:300]
        sid_doc = blocks[0].source_id
        sid_llm = _s(item.get("source_id"), 30) or None
        if sid_llm and normalize(sid_llm) not in normalize(block_text):
            sid_llm = None  # идентификатор, которого нет в тексте, не принимаем
        source_id = sid_doc or sid_llm
        kind = _s(item.get("kind"), 20).lower()
        return Requirement(
            rid="", source_id=source_id or "", source_id_assigned=source_id is None,
            summary=_s(item.get("summary"), 200) or blocks[0].text[:120],
            quote=quote, line_start=min(b.line_start for b in blocks), line_end=max(b.line_end for b in blocks),
            section=blocks[0].section, kind=kind if kind in KINDS else "other", block_ids=bids,
        )


def requirements_from_blocks(blocks: list[Block]) -> list[Requirement]:
    """Режим без LLM: каждый блок-кандидат — одно требование."""
    from reqcheck.segmenter import is_requirement_candidate

    from reqcheck.rules import PLACEHOLDER
    from reqcheck.segmenter import EXPLICIT_ID, _strip_md

    reqs = []
    for b in blocks:
        if not is_requirement_candidate(b) or PLACEHOLDER.match(b.text.strip("* ")):
            continue
        header = " ".join(b.table_header or []).lower()
        kind = "acceptance" if ("ожидаем" in header or "результат" in header) else (
            "api" if "parameter" in header or "параметр" in header else "other")
        clean = _strip_md(EXPLICIT_ID.sub("", b.text, count=1)) if b.source_id else _strip_md(b.text)
        reqs.append(Requirement(
            rid="", source_id=b.source_id or "", source_id_assigned=b.source_id is None,
            summary=(clean[:110] + "…") if len(clean) > 110 else clean,
            quote=_strip_md(b.text)[:300], line_start=b.line_start, line_end=b.line_end, section=b.section,
            kind=kind, block_ids=[b.bid],
        ))
    return reqs


def number_requirements(reqs: list[Requirement]) -> None:
    for i, r in enumerate(reqs, start=1):
        r.rid = f"R-{i:03d}"
        if r.source_id_assigned or not r.source_id:
            r.source_id_assigned = True
            r.source_id = ""


# --------------------------------------------------------------------------------------------------
def _finding_from(item: dict, scope: str, grounder: Grounder, default_rids: list[str],
                  line_of: dict[str, int] | None = None) -> Finding | None:
    if not isinstance(item, dict):
        return None
    sev = _s(item.get("severity"), 20).lower()
    rids = [_s(r, 10) for r in _list(item.get("rids"))] or default_rids
    # Повторяющаяся цитата привязывается к вхождению рядом с требованиями находки, а не к первому в документе
    near_rids = default_rids if scope == "requirement" else rids
    near = tuple(sorted({line_of[r] for r in near_rids if line_of and r in line_of}))
    quotes = [grounder.locate(_s(q, 600), near) for q in _list(item.get("quotes")) if _s(q)]
    title = _s(item.get("title"), 200)
    if not title:
        return None
    return Finding(
        fid="", scope=scope, criterion=criteria.normalize_code(_s(item.get("criterion"), 40), scope),
        severity=sev if sev in SEVERITIES else MAJOR, title=title, explanation=_s(item.get("explanation")),
        rids=rids, quotes=quotes, example=_s(item.get("example")), suggestion=_s(item.get("suggestion")),
        question=_s(item.get("question"), 600), addressee=_s(item.get("addressee"), 60), source="llm",
        verified=any(q.found for q in quotes),
    )


class Reviewer:
    def __init__(self, llm: JsonLLM, doc: Document, grounder: Grounder):
        self.llm, self.doc, self.grounder = llm, doc, grounder
        self.errors: list[str] = []

    def run(self, reqs: list[Requirement], blocks_by_id: dict[str, Block]) -> tuple[dict[str, RequirementAssessment], list[Finding]]:
        context = numbered_document(self.doc)
        assessments: dict[str, RequirementAssessment] = {}
        findings: list[Finding] = []
        known = {r.rid for r in reqs}
        line_of = {r.rid: r.line_start for r in reqs}
        batches = [reqs[i:i + REVIEW_BATCH] for i in range(0, len(reqs), REVIEW_BATCH)]
        shared = {b for b, n in Counter(b for r in reqs for b in set(r.block_ids)).items() if n > 1}

        def wording(r: Requirement) -> str:
            # Обычно модель видит весь текст требования. Только если Extractor выделил из одного абзаца несколько
            # требований, у каждого показываем свою формулировку, а абзац — отдельно как контекст: иначе у соседних
            # требований одинаковый текст, и модель принимает их за дубли или за одно «неатомарное» требование.
            block = " ".join(blocks_by_id[b].text for b in r.block_ids if b in blocks_by_id)
            own = r.quote.strip()
            if not own or not shared.intersection(r.block_ids) or normalize(own) == normalize(block):
                return block or own
            # Указание только у таких требований: общий промпт не меняется, остальные пакеты берутся из кэша
            return (f"Формулировка: {own}\nАбзац целиком: {block}\n(Из этого абзаца выделено несколько требований. "
                    "Оценивай только формулировку выше: соседние требования абзаца — не её дубли и не делают её неатомарной.)")

        def ask(batch: list[Requirement]) -> Any:
            listing = "\n\n".join(
                f"{r.rid} (в документе: {r.source_id or 'без номера'}; строки {r.line_start}-{r.line_end}; "
                f"раздел: {r.section}; вид: {r.kind})\n" + wording(r)
                for r in batch
            )
            prompt = f"{prompts.REVIEWER}\n\nТребования для оценки:\n{listing}\n\nВесь документ (номер строки: текст):\n{context}"
            return self.llm.generate_json("reviewer", prompts.BASE_SYSTEM, prompt)

        # Пачки независимы — отправляем параллельно (REQCHECK_WORKERS, по умолчанию 3); порядок результатов сохраняется
        workers = max(1, min(int(os.environ.get("REQCHECK_WORKERS", "3") or 1), len(batches) or 1))
        answers: list[Any] = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(ask, b): i for i, b in enumerate(batches)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    answers[i] = fut.result()
                except LLMError as exc:
                    self.errors.append(str(exc))
                    for r in batches[i]:
                        assessments[r.rid] = RequirementAssessment(
                            r.rid, [CriterionMark(c, NA, "не оценено: LLM не ответила") for c in criteria.REQ_CODES])
        if batches and all(a is None for a in answers):
            raise LLMError(self.errors[0] if self.errors else "Reviewer не получил ответов")
        for data in answers:
            if data is None:
                continue
            for item in _list(data.get("assessments") if isinstance(data, dict) else data):
                if not isinstance(item, dict):
                    continue
                rid = _s(item.get("rid"), 10)
                if rid not in known:
                    continue
                marks: dict[str, CriterionMark] = {}
                for m in _list(item.get("marks")):
                    if not isinstance(m, dict):
                        continue
                    code = criteria.normalize_code(_s(m.get("criterion"), 40), "requirement")
                    status = _s(m.get("status"), 12).lower()
                    if code in criteria.REQ_CODES:
                        marks[code] = CriterionMark(code, status if status in (OK, VIOLATION, NA) else NA, _s(m.get("comment"), 400))
                for code in criteria.REQ_CODES:
                    marks.setdefault(code, CriterionMark(code, NA, "модель не дала оценку"))
                assessments[rid] = RequirementAssessment(rid, [marks[c] for c in criteria.REQ_CODES])
                for f in _list(item.get("findings")):
                    fnd = _finding_from(f, "requirement", self.grounder, [rid], line_of)
                    if fnd:
                        fnd.rids = [rid]
                        findings.append(fnd)
        return assessments, findings


class ConsistencyReviewer:
    def __init__(self, llm: JsonLLM, doc: Document, grounder: Grounder):
        self.llm, self.doc, self.grounder = llm, doc, grounder
        self.gap_error = ""
        self.consistency_error = ""

    def run(self, reqs: list[Requirement]) -> tuple[list[CriterionMark], list[Finding]]:
        listing = "\n".join(f"{r.rid} | {r.source_id or '—'} | {r.kind} | строки {r.line_start}-{r.line_end} | {r.summary}" for r in reqs)
        prompt = (f"{prompts.CONSISTENCY}\n\nСписок требований (ID | номер в документе | вид | строки | кратко):\n{listing}"
                  f"\n\nВесь документ (номер строки: текст):\n{numbered_document(self.doc)}")
        # Противоречия и пропуски — независимые проходы по одному документу: запускаем одновременно
        with ThreadPoolExecutor(max_workers=2) as pool:
            gaps_future = pool.submit(self.llm.generate_json, "gaps", prompts.BASE_SYSTEM,
                                      prompt.replace(prompts.CONSISTENCY, prompts.GAPS, 1))
            try:
                data = self.llm.generate_json("consistency", prompts.BASE_SYSTEM, prompt)
            except LLMError as exc:  # сбой проверки противоречий не отменяет найденные пропуски
                data, self.consistency_error = None, str(exc)
            try:
                gaps = gaps_future.result()
            except LLMError as exc:  # сбой поиска пропусков не отменяет найденные противоречия
                gaps, self.gap_error = None, str(exc)
        if self.consistency_error and self.gap_error:
            raise LLMError(self.consistency_error)
        line_of = {r.rid: r.line_start for r in reqs}
        if not isinstance(data, dict):
            return [], self._collect(gaps, line_of)
        marks: dict[str, CriterionMark] = {}
        for m in _list(data.get("set_marks")):
            if isinstance(m, dict):
                code = criteria.normalize_code(_s(m.get("criterion"), 40), "set")
                status = _s(m.get("status"), 12).lower()
                if code in criteria.SET_CODES:
                    marks[code] = CriterionMark(code, status if status in (OK, VIOLATION, NA) else NA, _s(m.get("comment"), 600))
        for code in criteria.SET_CODES:
            marks.setdefault(code, CriterionMark(code, NA, "модель не дала оценку"))
        findings = self._collect(data, line_of) + self._collect(gaps, line_of)
        return [marks[c] for c in criteria.SET_CODES], findings

    def _collect(self, data: Any, line_of: dict[str, int]) -> list[Finding]:
        out = []
        if not isinstance(data, dict):
            return out
        for f in _list(data.get("findings")):
            fnd = _finding_from(f, "set", self.grounder, [], line_of)
            if fnd:
                fnd.rids = [r for r in fnd.rids if r in line_of]
                out.append(fnd)
        return out


class Critic:
    """Второй независимый проход: подтверждает, понижает или отклоняет критичные и существенные находки."""

    def __init__(self, llm: JsonLLM, doc: Document):
        self.llm, self.doc = llm, doc

    def run(self, findings: list[Finding]) -> tuple[list[Finding], list[Finding]]:
        targets = [f for f in findings if f.source == "llm" and f.severity in ("critical", "major")]
        if not targets:
            return findings, []
        rejected: set[str] = set()
        context = numbered_document(self.doc)
        for start in range(0, len(targets), CRITIC_BATCH):
            batch = targets[start:start + CRITIC_BATCH]
            listing = "\n\n".join(
                f"{f.fid} [{f.severity}, {criteria.name(f.criterion)}, scope={f.scope}, требования: {', '.join(f.rids) or '—'}] "
                f"{f.title}\nПояснение: {f.explanation}\n"
                f"Цитаты: " + " / ".join(f"«{q.text}» (строка {q.line})" for q in f.quotes)
                + (f"\nПример: {f.example}" if f.example else "")
                for f in batch
            )
            data = self.llm.generate_json("critic", prompts.BASE_SYSTEM,
                                          f"{prompts.CRITIC}{listing}\n\nВесь документ (номер строки: текст):\n{context}")
            by_id = {f.fid: f for f in batch}
            verdicts = [v for v in _list(data.get("verdicts") if isinstance(data, dict) else data) if isinstance(v, dict)]
            # Сначала отклонения и понижения, потом дубли: иначе дубль сливается в находку, которую критик
            # отклонит ниже по списку, и пропадают обе
            verdicts.sort(key=lambda v: _s(v.get("verdict"), 12).lower() == "duplicate")
            for v in verdicts:
                f = by_id.get(_s(v.get("fid"), 12))
                if not f:
                    continue
                verdict = _s(v.get("verdict"), 12).lower()
                reason = _s(v.get("reason"), 400)
                if verdict == "reject":
                    rejected.add(f.fid)
                    f.critic = f"отклонено критиком: {reason}"
                elif verdict == "duplicate":
                    other = _s(v.get("duplicate_of"), 12)
                    if other in by_id and other != f.fid and other not in rejected:
                        rejected.add(f.fid)
                        f.critic = f"дубль {other}: {reason}"
                        main = by_id[other]
                        main.rids = list(dict.fromkeys(main.rids + f.rids))
                elif verdict == "downgrade":
                    sev = _s(v.get("severity"), 12).lower()
                    if sev in SEVERITIES and SEVERITIES.index(sev) > SEVERITIES.index(f.severity):
                        f.severity = sev
                    f.critic = f"понижено критиком: {reason}"
                elif verdict == "confirm":
                    f.critic = "подтверждено критиком"
        kept = [f for f in findings if f.fid not in rejected]
        dropped = [f for f in findings if f.fid in rejected]
        return kept, dropped


class Analyst:
    def __init__(self, llm: JsonLLM, doc: Document, grounder: Grounder):
        self.llm, self.doc, self.grounder = llm, doc, grounder

    def checklist_and_trace(self, reqs: list[Requirement], checklist: list[ChecklistItem]) -> list[TraceRow]:
        listing = "\n".join(f"{r.rid} | {r.source_id or '—'} | {r.kind} | {r.summary}" for r in reqs)
        cl = "\n".join(f"{i}. {c.question}" for i, c in enumerate(checklist, start=1)) or "(чек-листа в документе нет — верни пустой checklist)"
        prompt = (f"{prompts.ANALYST_CHECKLIST}\n\nПункты чек-листа:\n{cl}\n\nТребования (ID | номер | вид | кратко):\n{listing}"
                  f"\n\nВесь документ (номер строки: текст):\n{numbered_document(self.doc)}")
        data = self.llm.generate_json("analyst_checklist", prompts.BASE_SYSTEM, prompt)
        if not isinstance(data, dict):
            return []
        for item in _list(data.get("checklist")):
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index")) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(checklist):
                answer = _s(item.get("answer"), 30)
                if answer not in ("Да", "Нет", "Не требуется", "Недостаточно данных"):
                    answer = "Недостаточно данных"
                checklist[idx].answer = answer
                checklist[idx].justification = _s(item.get("justification"), 600)
                checklist[idx].quotes = [q for q in (self.grounder.locate(_s(x, 400)) for x in _list(item.get("quotes"))) if q.found]
        known = {r.rid for r in reqs}
        rows = []
        for t in _list(data.get("trace")):
            if isinstance(t, dict) and _s(t.get("rid"), 10) in known:
                rows.append(TraceRow(
                    rid=_s(t.get("rid"), 10),
                    acceptance=[x for x in (_s(a, 10) for a in _list(t.get("acceptance"))) if x in known],
                    api=[x for x in (_s(a, 10) for a in _list(t.get("api"))) if x in known],
                    note=_s(t.get("note"), 300),
                ))
        return rows

    def tests(self, reqs: list[Requirement]) -> list[TestDraft]:
        listing = "\n".join(f"{r.rid} | {r.source_id or '—'} | {r.kind} | {r.summary}" for r in reqs)
        prompt = (f"{prompts.ANALYST_TESTS}\n\nТребования (ID | номер | вид | кратко):\n{listing}"
                  f"\n\nВесь документ (номер строки: текст):\n{numbered_document(self.doc)}")
        data = self.llm.generate_json("analyst_tests", prompts.BASE_SYSTEM, prompt)
        known = {r.rid for r in reqs}
        out = []
        for t in _list(data.get("tests") if isinstance(data, dict) else data)[:15]:
            if isinstance(t, dict) and _s(t.get("title")):
                rid = _s(t.get("rid"), 10)
                out.append(TestDraft(rid=rid if rid in known else "", title=_s(t.get("title"), 200),
                                     given=_s(t.get("given"), 500), when=_s(t.get("when"), 500), then=_s(t.get("then"), 600)))
        return out


# --------------------------------------------------------------------------------------------------
SEVERITY_PENALTY = {"critical": 35, "major": 20, "minor": 8}


def score_requirement(assessment: RequirementAssessment | None, findings: list[Finding]) -> tuple[int, str]:
    penalty = sum(SEVERITY_PENALTY.get(f.severity, 10) for f in findings)
    if assessment:
        with_finding = {f.criterion for f in findings}
        penalty += 5 * sum(1 for m in assessment.marks if m.status == VIOLATION and m.criterion not in with_finding)
    score = max(0, 100 - penalty)
    light = "green" if score >= 80 else ("yellow" if score >= 50 else "red")
    return score, light


DEDUPE_TITLE_SIMILARITY = 0.6  # заголовки «одного и того же» дефекта


def dedupe(findings: list[Finding]) -> list[Finding]:
    """Убрать дубли: одинаковый критерий + те же требования + одна и та же первая цитата/заголовок."""
    # Ключ без rids: экстрактор иногда режет один блок на несколько требований с одной цитатой,
    # и ревьюер повторяет один дефект на каждом. Такие находки сливаем: rids объединяем, берём старшую критичность.
    rank = SEV_RANK
    seen: dict[tuple, list[Finding]] = {}
    titles: dict[int, str] = {}
    out = []
    for f in findings:
        q = normalize(f.quotes[0].text)[:60] if f.quotes else ""
        key = (f.scope, f.criterion, q or normalize(f.title)[:60])
        if q == "" and f.scope == "requirement":
            key = key + (tuple(sorted(f.rids)),)
        title = normalize(f.title)
        main = next((g for g in seen.get(key, [])
                     if SequenceMatcher(None, titles[id(g)], title).ratio() >= DEDUPE_TITLE_SIMILARITY
                     or set(g.rids) == set(f.rids)), None)
        titles[id(f)] = title
        if main is None:
            seen.setdefault(key, []).append(f)
            out.append(f)
            continue
        main.rids = sorted(set(main.rids) | set(f.rids))
        if rank.get(f.severity, 3) < rank.get(main.severity, 3):
            main.severity = f.severity
    return out
