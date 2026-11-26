"""Детерминированная нарезка документа на структурные блоки.

Блок — абзац, пункт списка, строка таблицы или блок кода с номерами строк и путём раздела.
Явные идентификаторы требований (Ф-1, FR-2, REQ-10, НФТ-3, 4.2.1 …) извлекаются кодом, без LLM.
"""

from __future__ import annotations

import re

from reqcheck.models import Block, Document

HEADING_MD = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
HEADING_NUM = re.compile(r"^\s*(\d+(?:\.\d+){0,4})\.?\s+([A-ZА-ЯЁ][^.!?]{1,80})$")
HEADING_PART = re.compile(r"^\s*(Часть|Раздел|Глава|Part|Section)\s+[\wА-Яа-яЁё]{1,4}[.:]?\s+\S.{0,80}$", re.I)
LIST_ITEM = re.compile(r"^(\s*)([*\-+•]|\d+[.)])\s+(.*)$")
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
HRULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")

# Явный идентификатор требования в начале пункта: **Ф-1.**, FR-2:, REQ_10), НФТ-3, BR-01, US-5
EXPLICIT_ID = re.compile(
    r"^\s*(?:\*\*|__)?\s*([A-Za-zА-Яа-яЁё]{1,8}[-_–]\d{1,4}(?:\.\d{1,3})*)\s*[.:)]?\s*(?:\*\*|__)?\s*[.:)\-–—]?\s+"
)
# Нумерованный пункт вида «4.2.1» в начале строки. Десятичное число с единицей («1.5 часа», «2.5 млн») — не номер;
# единица — целым словом, иначе «4.2 Минимальная сумма», «1.2 Разработка», «2.1 XML-интерфейс» теряли бы номер
_UNIT = (r"(?:сек(?:унд\w*)?|мс|мин(?:ут\w*)?|ч|час(?:а|ов)?|сут(?:ок|ки)?|дн(?:я|ей)?|д(?:ень|ня|ней)|"
         r"тенге|руб(?:л\w*)?|млн|млрд|тыс(?:яч\w*)?|кб|мб|гб|раз(?:а)?|x|×)\.?(?!\w)")
NUMBERED_ID = re.compile(r"^\s*(\d+\.\d+(?:\.\d+){0,3})\.?\s+(?!\s*(?:%|₸|" + _UNIT + r"))", re.I)
# Начало нового пункта в PDF/TXT без маркеров списка: «Ф-2. …», «FR-3: …». Знак после ID обязателен, иначе
# перенос строки вида «…согласно\nФТ-2 сумма…» резал бы абзац на ссылке
ITEM_START = re.compile(r"^\s*(?:\*\*|__)?\s*[A-Za-zА-Яа-яЁё]{1,8}[-_–]\d{1,4}(?:\.\d{1,3})*\s*(?:\*\*|__)?\s*[.:)]\s+")


def _strip_md(text: str) -> str:
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return text.strip()


def split_cells(row: str) -> list[str]:
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def extract_source_id(text: str) -> str | None:
    m = EXPLICIT_ID.match(text)
    if m:
        return m.group(1).replace("–", "-")
    m = NUMBERED_ID.match(text)
    if m:
        return m.group(1)
    return None


def segment(doc: Document) -> list[Block]:
    """Разбить документ на блоки. Работает для MD; для TXT/PDF/DOCX распознаёт нумерованные заголовки."""
    lines = doc.lines
    blocks: list[Block] = []
    section_stack: list[tuple[int, str]] = []  # (уровень, заголовок)
    is_md = doc.fmt in ("md", "docx")

    def section_path() -> str:
        return " › ".join(t for _, t in section_stack) or "(без раздела)"

    def push_heading(level: int, title: str) -> None:
        while section_stack and section_stack[-1][0] >= level:
            section_stack.pop()
        section_stack.append((level, _strip_md(title)))

    def add(kind: str, text: str, start: int, end: int, header: list[str] | None = None,
            cells: list[str] | None = None) -> None:
        clean = text.strip()
        if not clean:
            return
        blocks.append(Block(
            bid=f"B-{len(blocks) + 1:03d}", kind=kind, text=clean, line_start=start, line_end=end,
            section=section_path(), source_id=extract_source_id(clean) if kind != "table_row" else None,
            table_header=header, cells=cells,
        ))

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        lineno = i + 1
        if not line.strip() or HRULE.match(line):
            i += 1
            continue
        # Блок кода
        if FENCE.match(line):
            fence = FENCE.match(line).group(1)  # type: ignore[union-attr]
            j = i + 1
            while j < n and not lines[j].strip().startswith(fence):
                j += 1
            add("code", "\n".join(lines[i + 1:j]), lineno + 1, min(j, n))
            i = j + 1
            continue
        # Заголовки
        m = HEADING_MD.match(line)
        if m:
            push_heading(len(m.group(1)), m.group(2))
            i += 1
            continue
        if not is_md:
            m2 = HEADING_NUM.match(line)
            if m2 and len(line.strip()) <= 90 and not line.rstrip().endswith((".", ";", ",")):
                push_heading(m2.group(1).count(".") + 1, line.strip())
                i += 1
                continue
            if HEADING_PART.match(line) and len(line.strip()) <= 90:
                push_heading(1, line.strip())
                i += 1
                continue
        # Таблица
        if TABLE_ROW.match(line):
            header: list[str] | None = None
            j = i
            rows: list[tuple[int, str]] = []
            while j < n and TABLE_ROW.match(lines[j]):
                rows.append((j + 1, lines[j]))
                j += 1
            if len(rows) >= 2 and TABLE_SEP.match(rows[1][1]):
                header = [_strip_md(c) for c in split_cells(rows[0][1])]
                data_rows = rows[2:]
            else:
                data_rows = rows
            for ln, row in data_rows:
                if TABLE_SEP.match(row):
                    continue
                cells = [_strip_md(c) for c in split_cells(row)]
                if header and len(header) == len(cells):
                    text = "; ".join(f"{h}: {c}" if h else c for h, c in zip(header, cells))
                else:
                    text = " | ".join(cells)
                add("table_row", text, ln, ln, header, cells)
            i = j
            continue
        # Пункт списка (с продолжением на следующих строках)
        m = LIST_ITEM.match(line)
        if m:
            indent = len(m.group(1))
            parts = [m.group(3)]
            j = i + 1
            while j < n and lines[j].strip() and not LIST_ITEM.match(lines[j]) and not TABLE_ROW.match(lines[j]) \
                    and not HEADING_MD.match(lines[j]) and (len(lines[j]) - len(lines[j].lstrip()) > indent
                                                             or not (is_md or ITEM_START.match(lines[j]))):
                parts.append(lines[j].strip())
                j += 1
            add("list_item", " ".join(parts), lineno, j)
            i = j
            continue
        # Абзац
        j = i
        parts = []
        while j < n and lines[j].strip() and not LIST_ITEM.match(lines[j]) and not TABLE_ROW.match(lines[j]) \
                and not HEADING_MD.match(lines[j]) and not FENCE.match(lines[j]):
            if not is_md and j > i and (HEADING_NUM.match(lines[j]) or ITEM_START.match(lines[j])):
                break
            parts.append(lines[j].strip())
            j += 1
        if not parts:  # защита от зацикливания на неожиданной строке
            parts = [line.strip()]
            j = i + 1
        add("paragraph", " ".join(parts), lineno, j)
        i = j
    return blocks


# Разделы, которые обычно не содержат требований (паспорт документа, согласующие)
_NON_REQ_SECTION = re.compile(r"общие сведения|блок согласующ|согласовани|лист изменений|история изменений|содержание|оглавление", re.I)
_REQ_WORDS = re.compile(
    r"\b(долж|обязан|необходимо|требует|обеспечива|поддержива|позволя|разреш|запрещ|не допуска|получает|"
    r"отправля|сохраня|возвраща|проверя|исключа|определя|хранит|передаёт|передает|показыва|задаёт|задает|"
    r"shall|must|should|will)\w*",
    re.I,
)


def is_requirement_candidate(block: Block) -> bool:
    """Эвристика для режима без LLM: похож ли блок на требование."""
    if block.kind == "code":
        return False
    if _NON_REQ_SECTION.search(block.section):
        return False
    if block.source_id:
        return True
    if block.kind == "table_row":
        header = " ".join(block.table_header or []).lower()
        if "ожидаем" in header or "результат" in header:  # критерии приёмки
            return True
        return bool(re.search(r"parameter|параметр", header))
    if block.kind == "list_item" and len(block.text) >= 15 and not _CHECKLIST_SECTION.search(block.section):
        return True
    if re.search(r"требован|правил|логик", block.section, re.I) and len(block.text) >= 25:
        return True
    return bool(_REQ_WORDS.search(block.text)) and len(block.text) >= 25


_CHECKLIST_SECTION = re.compile(r"ревью|чек-?лист|review|checklist", re.I)
