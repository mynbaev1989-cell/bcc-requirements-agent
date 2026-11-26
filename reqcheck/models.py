"""Структуры данных агента. Всё, что попадает в отчёт, проходит через эти классы."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Серьёзность дефекта
CRITICAL = "critical"
MAJOR = "major"
MINOR = "minor"
SEVERITIES = (CRITICAL, MAJOR, MINOR)
SEV_RANK = {CRITICAL: 0, MAJOR: 1, MINOR: 2}  # порядок сортировки: сначала критичные
SEVERITY_RU = {CRITICAL: "критичный", MAJOR: "существенный", MINOR: "незначительный"}

# Статус оценки по критерию
OK = "ok"
VIOLATION = "violation"
NA = "na"


@dataclass
class Diagnostic:
    """Сообщение о проблеме при чтении файла или в работе агента (вместо аварийного завершения)."""

    level: str  # error | warning | info
    code: str  # машинный код, например PDF_NO_TEXT
    message: str  # что случилось, по-русски
    hint: str = ""  # что сделать пользователю


@dataclass
class Document:
    path: str
    name: str  # имя файла
    title: str  # название документа (первый заголовок или имя файла)
    fmt: str  # md | txt | pdf | docx
    text: str
    lines: list[str]
    sha256: str
    size_bytes: int
    pages: int | None = None
    line_pages: list[int] | None = None  # номер страницы для каждой строки (PDF)


@dataclass
class Block:
    """Структурный фрагмент документа: абзац, пункт списка, строка таблицы, код."""

    bid: str  # B-001
    kind: str  # paragraph | list_item | table_row | code
    text: str
    line_start: int  # с 1
    line_end: int
    section: str  # путь раздела: «Часть A › Функциональные требования»
    source_id: str | None = None  # явный идентификатор из документа (Ф-1, FR-2…)
    table_header: list[str] | None = None
    cells: list[str] | None = None


@dataclass
class Requirement:
    rid: str  # R-001 — идентификатор в отчёте
    source_id: str  # идентификатор в документе или присвоенный агентом
    source_id_assigned: bool  # True, если в документе номера не было
    summary: str  # краткая формулировка
    quote: str  # дословная цитата
    line_start: int
    line_end: int
    section: str
    kind: str = "other"  # functional | nonfunctional | data | integration | acceptance | api | ui | other
    block_ids: list[str] = field(default_factory=list)


@dataclass
class Quote:
    text: str
    line: int | None = None  # номер строки в исходнике, если найдена
    found: bool = False


@dataclass
class Finding:
    fid: str
    scope: str  # requirement | set
    criterion: str  # код критерия из criteria.py
    severity: str
    title: str
    explanation: str
    rids: list[str] = field(default_factory=list)
    quotes: list[Quote] = field(default_factory=list)
    example: str = ""  # конкретный сценарий, на котором видно дефект
    suggestion: str = ""  # исправленная формулировка
    question: str = ""  # вопрос автору требований
    addressee: str = ""  # кому адресован вопрос
    source: str = "llm"  # llm | rule
    verified: bool = False  # все цитаты найдены в исходнике
    critic: str = ""  # вердикт агента-критика
    resolution: str = ""  # как на находку повлиял ответ автора (цикл уточнения)
    grouped: bool = False  # сводная находка по нескольким требованиям (agents.collapse_singular)


def per_requirement(f: Finding) -> bool:
    """Находка про конкретные требования: ставит ❌ их критерию и штрафует их балл.

    Сводная находка показывается в дефектах набора, но по смыслу — про каждое из своих требований."""
    return f.scope == "requirement" or f.grouped


@dataclass
class CriterionMark:
    criterion: str
    status: str  # ok | violation | na
    comment: str = ""


@dataclass
class RequirementAssessment:
    rid: str
    marks: list[CriterionMark] = field(default_factory=list)
    score: int = 100  # 0–100, считается кодом
    light: str = "green"  # green | yellow | red


@dataclass
class ChecklistItem:
    question: str
    line: int
    answer: str = "Недостаточно данных"  # Да | Нет | Не требуется | Недостаточно данных
    justification: str = ""
    quotes: list[Quote] = field(default_factory=list)


@dataclass
class TestDraft:
    rid: str
    title: str
    given: str
    when: str
    then: str


@dataclass
class TraceRow:
    rid: str
    acceptance: list[str] = field(default_factory=list)  # ID требований-критериев приёмки
    api: list[str] = field(default_factory=list)  # ID требований API / поля
    note: str = ""


@dataclass
class Question:
    """Уточняющий вопрос автору требований (агент-уточнитель)."""
    qid: str
    text: str
    priority: str = "important"  # blocker | important | clarify
    addressee: str = "Аналитик"
    why: str = ""  # риск, если не ответить
    options: list[str] = field(default_factory=list)  # варианты ответа (закрытый вопрос)
    assumption: str = ""  # допущение агента, если ответа не будет
    fids: list[str] = field(default_factory=list)  # находки, которые закрывает ответ
    lines: list[int] = field(default_factory=list)
    source: str = "llm"  # llm | findings (детерминированный запасной вариант)
    answer: str = ""  # ответ автора (из файла --answers)
    outcome: str = ""  # итог учёта ответа


@dataclass
class RunResult:
    document: Document | None
    passport: dict[str, Any] = field(default_factory=dict)
    blocks: list[Block] = field(default_factory=list)
    requirements: list[Requirement] = field(default_factory=list)
    assessments: dict[str, RequirementAssessment] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    set_marks: list[CriterionMark] = field(default_factory=list)
    checklist: list[ChecklistItem] = field(default_factory=list)
    tests: list[TestDraft] = field(default_factory=list)
    trace: list[TraceRow] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    resolved: list[Finding] = field(default_factory=list)  # находки, закрытые ответами автора
    diagnostics: list[Diagnostic] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    document_score: int | None = None
    verdict_dev: str = ""
    verdict_test: str = ""
    priorities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("blocks", None)  # служебная нарезка, в отчёт не нужна
        if data.get("document"):
            # текст документа в JSON не дублируем
            data["document"].pop("text", None)
            data["document"].pop("lines", None)
            data["document"].pop("line_pages", None)
        return data
