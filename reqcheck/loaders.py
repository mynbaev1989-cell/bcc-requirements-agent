"""Чтение входного документа (MD, TXT, PDF, DOCX) с диагностикой вместо аварийного завершения.

Тип файла определяется по содержимому (сигнатуре), а не только по расширению.
Любая проблема превращается в LoadError с понятным Diagnostic, который попадает в отчёт.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import threading
import zipfile
from pathlib import Path
from typing import TypeVar
from collections.abc import Callable

from reqcheck.models import Diagnostic, Document

T = TypeVar("T")

# Предупреждения pypdf о битой структуре («EOF marker not found») не печатаем в консоль:
# проблема и так попадает в диагностику отчёта понятным сообщением.
logging.getLogger("pypdf").setLevel(logging.ERROR)

DEFAULT_MAX_BYTES = 20 * 1024 * 1024  # 20 МБ
DEFAULT_MAX_CHARS = 400_000  # ~100 страниц текста
DEFAULT_MAX_PAGES = 300
PARSE_TIMEOUT_SEC = 60  # защита от «зависающих» PDF
MIN_MEANINGFUL_CHARS = 20


class LoadError(Exception):
    """Файл нельзя проверить. Содержит диагностику для отчёта."""

    def __init__(self, diagnostic: Diagnostic):
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


def _fail(code: str, message: str, hint: str = "") -> LoadError:
    return LoadError(Diagnostic("error", code, message, hint))


def run_with_timeout(fn: Callable[[], T], timeout: float, what: str) -> T:
    """Выполнить fn в фоновом daemon-потоке; по таймауту — LoadError, а не зависание агента."""
    box: dict[str, object] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 — передаём наружу любую ошибку разбора
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise _fail(
            "PARSE_TIMEOUT",
            f"Разбор файла ({what}) не завершился за {timeout:.0f} с — файл, вероятно, повреждён.",
            "Пересохраните документ (например, «Печать → PDF» или экспорт в DOCX) и запустите проверку снова.",
        )
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box["value"]  # type: ignore[return-value]


def detect_format(raw: bytes, suffix: str) -> str:
    """Определить формат по сигнатуре. Возвращает pdf | docx | zip | text | binary | empty."""
    if not raw.strip(b" \t\r\n\x00"):
        return "empty"
    head = raw[:1024]
    if b"%PDF-" in head:
        return "pdf"
    if raw[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                if "word/document.xml" in zf.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            return "binary"
        return "zip"
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or raw[:3] == b"\xef\xbb\xbf":
        return "text"
    sample = raw[:65536]
    if b"\x00" in sample:
        return "binary"
    # Доля управляющих символов (кроме \t \n \r \f) — признак бинарных данных
    ctrl = sum(1 for b in sample if b < 32 and b not in (9, 10, 12, 13))
    if ctrl / max(len(sample), 1) > 0.02:
        return "binary"
    if suffix == ".pdf":
        # Расширение PDF, но сигнатуры нет — это не PDF
        return "binary"
    return "text"


def decode_text(raw: bytes) -> tuple[str, str | None]:
    """Декодировать текст: UTF-8 (с BOM), UTF-16 по BOM, затем CP1251. Возвращает (текст, предупреждение)."""
    if raw[:3] == b"\xef\xbb\xbf":
        return raw[3:].decode("utf-8", errors="replace"), None
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace"), None
    try:
        return raw.decode("utf-8"), None
    except UnicodeDecodeError:
        pass
    try:
        text = raw.decode("cp1251")
        return text, "Файл не в UTF-8; прочитан как Windows-1251. Проверьте, что кириллица в отчёте отображается верно."
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), "Часть символов не удалось декодировать и они заменены на «�»."


def _read_pdf(raw: bytes, max_pages: int) -> tuple[str, list[int], int, list[Diagnostic]]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    diags: list[Diagnostic] = []
    try:
        reader = PdfReader(io.BytesIO(raw), strict=False)
    except (PdfReadError, ValueError, OSError, KeyError, TypeError) as exc:
        raise _fail("PDF_CORRUPTED", f"PDF повреждён и не читается: {exc}", "Пересохраните PDF из исходного редактора.") from exc
    if reader.is_encrypted:
        try:
            ok = reader.decrypt("")
        except Exception:  # noqa: BLE001
            ok = 0
        if not ok:
            raise _fail(
                "PDF_ENCRYPTED",
                "PDF защищён паролем, текст недоступен.",
                "Снимите защиту (сохраните копию без пароля) и запустите проверку снова.",
            )
    try:
        pages = reader.pages
        n_pages = len(pages)
    except Exception as exc:  # noqa: BLE001
        raise _fail("PDF_CORRUPTED", f"Не удалось прочитать структуру страниц PDF: {exc}", "Пересохраните PDF.") from exc
    if n_pages == 0:
        raise _fail("PDF_NO_PAGES", "В PDF нет ни одной страницы.", "Проверьте, что выгружен правильный файл.")
    if n_pages > max_pages:
        diags.append(
            Diagnostic("warning", "PDF_TRUNCATED", f"В PDF {n_pages} стр.; проверены первые {max_pages}.", "Разделите документ на части.")
        )
    lines: list[str] = []
    line_pages: list[int] = []
    bad_pages: list[int] = []
    for i, page in enumerate(pages[:max_pages], start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 — одна битая страница не должна ронять весь прогон
            bad_pages.append(i)
            continue
        for line in page_text.splitlines():
            lines.append(line.rstrip())
            line_pages.append(i)
    if bad_pages:
        diags.append(
            Diagnostic(
                "warning",
                "PDF_BAD_PAGES",
                f"Не удалось извлечь текст со страниц: {', '.join(map(str, bad_pages))}.",
                "Проверьте эти страницы вручную или пересохраните PDF.",
            )
        )
    text = "\n".join(lines)
    if len(re.sub(r"\s", "", text)) < MIN_MEANINGFUL_CHARS:
        raise _fail(
            "PDF_NO_TEXT",
            "В PDF нет текстового слоя (похоже на скан или изображения) — проверять нечего.",
            "Сохраните документ как текстовый PDF из редактора или пришлите DOCX/MD. Для скана нужно предварительное распознавание (OCR).",
        )
    return text, line_pages, n_pages, diags


class _DocxNumbering:
    """Автонумерация Word (w:numPr): номер пункта не хранится в тексте, его считает редактор.

    Без неё «1.2.», «ФТ-3.» из многоуровневого списка пропадают: агент решил бы, что у требований нет
    идентификаторов. Считаем номера как Word: счётчики общие для всех w:num одного w:abstractNum (список
    продолжается), w:startOverride начинает уровень заново; шаблон lvlText («%1.%2.», «ФТ-%1.»)."""

    LETTERS = {"lowerLetter": "abcdefghijklmnopqrstuvwxyz", "upperLetter": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
               "russianLower": "абвгдежзиклмнопрстуфхцчшщэюя", "russianUpper": "АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЭЮЯ"}
    ROMAN = ((1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
             (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"))
    MAX_LEVELS = 9  # уровни списка Word: 0–8

    def __init__(self, document) -> None:  # noqa: ANN001 — docx.Document
        self.levels: dict[str, dict[int, tuple[str, str, int]]] = {}  # numId → ilvl → (numFmt, lvlText, start)
        self.abstract_of: dict[str, str] = {}  # numId → abstractNumId: по нему общие счётчики
        self.overrides: dict[str, dict[int, int]] = {}  # numId → ilvl → startOverride
        self.counters: dict[str, list[int]] = {}
        self.pending: dict[str, dict[int, int]] = {}  # abstractNumId → ilvl → с какого номера начать заново
        self.started: set[str] = set()
        try:
            root = document.part.numbering_part.element
        except (AttributeError, KeyError, NotImplementedError):
            return
        try:
            self._parse(root)
        except (ValueError, TypeError):  # битый numbering.xml: читаем документ без автонумерации, а не падаем
            self.levels, self.abstract_of, self.overrides = {}, {}, {}

    def _parse(self, root) -> None:  # noqa: ANN001
        from docx.oxml.ns import qn
        val = qn("w:val")

        def num_val(el, default: int) -> int:  # noqa: ANN001
            return int(el.get(val)) if el is not None else default

        abstract: dict[str, dict[int, tuple[str, str, int]]] = {}
        for an in root.findall(qn("w:abstractNum")):
            lv = {}
            for lvl in an.findall(qn("w:lvl")):
                fmt, text = lvl.find(qn("w:numFmt")), lvl.find(qn("w:lvlText"))
                ilvl = int(lvl.get(qn("w:ilvl"), 0))
                if 0 <= ilvl < self.MAX_LEVELS:  # уровень вне 0–8 Word не показывает: игнорируем, а не падаем
                    lv[ilvl] = (fmt.get(val) if fmt is not None else "decimal",
                                text.get(val) if text is not None else "",
                                num_val(lvl.find(qn("w:start")), 1))
            abstract[an.get(qn("w:abstractNumId"))] = lv
        for num in root.findall(qn("w:num")):
            ref = num.find(qn("w:abstractNumId"))
            if ref is None or ref.get(val) not in abstract:
                continue
            num_id = num.get(qn("w:numId"))
            self.levels[num_id], self.abstract_of[num_id] = abstract[ref.get(val)], ref.get(val)
            for ov in num.findall(qn("w:lvlOverride")):
                so = ov.find(qn("w:startOverride"))
                if so is not None:
                    self.overrides.setdefault(num_id, {})[int(ov.get(qn("w:ilvl"), 0))] = num_val(so, 1)

    def _num_pr(self, para):  # noqa: ANN001, ANN202 — абзац, затем цепочка стилей
        p_pr = para._p.pPr
        if p_pr is not None and p_pr.numPr is not None:
            return p_pr.numPr
        style = para.style
        while style is not None:
            s_pr = style.element.pPr
            if s_pr is not None and s_pr.numPr is not None:
                return s_pr.numPr
            style = style.base_style
        return None

    def label(self, para) -> str | None:  # noqa: ANN001
        """Номер пункта («1.2.», «ФТ-3.»); «» — маркированный список; None — абзац не в списке."""
        num_pr = self._num_pr(para)
        if num_pr is None or num_pr.numId is None:
            return None
        num_id = str(num_pr.numId.val)
        try:
            ilvl = int(num_pr.ilvl.val) if num_pr.ilvl is not None else 0
        except (ValueError, TypeError):  # битый w:ilvl: абзац читаем без номера
            return None
        levels = self.levels.get(num_id)
        if not levels or ilvl not in levels:
            return None
        fmt, text, _ = levels[ilvl]
        if fmt in ("bullet", "none") or not text:
            return ""
        key = self.abstract_of[num_id]
        cnt = self.counters.setdefault(key, [0] * self.MAX_LEVELS)
        pending = self.pending.setdefault(key, {})
        if num_id not in self.started:  # первый абзац этого w:num: его startOverride начинает уровни заново
            self.started.add(num_id)
            pending.update(self.overrides.get(num_id, {}))
        for k in range(ilvl + 1):
            if k in pending:  # startOverride, в том числе у родителя, если список начат сразу с вложенного пункта
                cnt[k] = pending.pop(k)
            elif cnt[k] == 0:  # пропущенный верхний уровень начинается со своего start
                cnt[k] = levels.get(k, ("", "", 1))[2]
            elif k == ilvl:
                cnt[k] += 1
        for k in range(ilvl + 1, self.MAX_LEVELS):
            cnt[k] = 0  # новый пункт уровня сбрасывает вложенные
        return re.sub(r"%([1-9])", lambda m: self._fmt(levels.get(int(m.group(1)) - 1, ("decimal",))[0],
                                                       cnt[int(m.group(1)) - 1]), text)

    def _fmt(self, fmt: str, n: int) -> str:
        if n <= 0:
            return str(n)
        if fmt in ("lowerRoman", "upperRoman"):
            out, rest = "", n
            for value, sym in self.ROMAN:
                while rest >= value:
                    out, rest = out + sym, rest - value
            return out.upper() if fmt == "upperRoman" else out
        if fmt == "decimalZero":
            return f"{n:02d}"
        letters = self.LETTERS.get(fmt)
        return letters[(n - 1) % len(letters)] if letters else str(n)


def _read_docx(raw: bytes) -> str:
    import docx  # python-docx

    try:
        document = docx.Document(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise _fail("DOCX_CORRUPTED", f"DOCX повреждён: {exc}", "Откройте файл в Word и сохраните заново.") from exc
    out: list[str] = []
    numbering = _DocxNumbering(document)
    # Абзацы и таблицы в порядке следования в документе
    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = docx.text.paragraph.Paragraph(child, document)
            text = para.text.strip()
            style = (para.style.name if para.style is not None else "") or ""
            m = re.match(r"(?:Heading|Заголовок)\s*(\d)", style)
            label = numbering.label(para) if text else None  # номер из автонумерации Word
            if m and text:
                out.append("#" * int(m.group(1)) + " " + (f"{label} " if label else "") + text)
            elif text:
                if label:
                    out.append(f"{label} {text}")  # «1.2. Система должна…», «ФТ-3. …» — номер виден сегментатору
                else:
                    prefix = "* " if label == "" or "List" in style or "Список" in style else ""
                    out.append(prefix + text)
            out.append("")
        elif tag == "tbl":
            table = docx.table.Table(child, document)
            for r_idx, row in enumerate(table.rows):
                cells = [c.text.replace("\n", " ").strip() for c in row.cells]
                out.append("| " + " | ".join(cells) + " |")
                if r_idx == 0:
                    out.append("|" + "---|" * len(cells))
            out.append("")
    return "\n".join(out)


def _guess_title(lines: list[str], fallback: str) -> str:
    for line in lines[:60]:
        m = re.match(r"^\s*#\s+(.+?)\s*#*\s*$", line)
        if m:
            return m.group(1).strip()
    for line in lines[:10]:
        s = line.strip()
        if 3 <= len(s) <= 150:
            return s
    return fallback


def load_document(
    path: str | os.PathLike[str],
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> tuple[Document, list[Diagnostic]]:
    """Прочитать документ. Возвращает (Document, предупреждения) или бросает LoadError."""
    p = Path(path)
    diags: list[Diagnostic] = []
    if not p.exists():
        raise _fail("FILE_NOT_FOUND", f"Файл не найден: {p}", "Проверьте путь к документу требований.")
    if p.is_dir():
        raise _fail("IS_DIRECTORY", f"Указана папка, а не файл: {p}", "Передайте путь к файлу документа.")
    try:
        size = p.stat().st_size
    except OSError as exc:
        raise _fail("FILE_UNREADABLE", f"Нет доступа к файлу: {exc}", "Проверьте права доступа.") from exc
    if size > max_bytes:
        raise _fail(
            "FILE_TOO_LARGE",
            f"Файл слишком большой: {size / 1048576:.1f} МБ (лимит {max_bytes / 1048576:.0f} МБ).",
            "Разделите документ на части или увеличьте лимит флагом --max-size-mb.",
        )
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise _fail("FILE_UNREADABLE", f"Не удалось прочитать файл: {exc}", "Проверьте, что файл не открыт эксклюзивно другой программой.") from exc

    suffix = p.suffix.lower()
    kind = detect_format(raw, suffix)
    line_pages: list[int] | None = None
    n_pages: int | None = None

    if kind == "empty":
        raise _fail("FILE_EMPTY", "Файл пустой — в нём нет текста для проверки.", "Проверьте, что выгружен правильный файл.")
    if kind == "binary":
        what = "PDF" if suffix == ".pdf" else "текст"
        raise _fail(
            "FILE_BINARY",
            f"Содержимое файла не похоже ни на {what}, ни на другой поддерживаемый формат (повреждённые или бинарные данные).",
            "Поддерживаются MD, TXT, PDF с текстовым слоем и DOCX. Пересохраните документ в одном из них.",
        )
    if kind == "zip":
        raise _fail("UNSUPPORTED_ZIP", "Это ZIP-архив, а не документ DOCX.", "Распакуйте архив и передайте сам документ.")

    if kind == "pdf":
        text, line_pages, n_pages, pdf_diags = run_with_timeout(lambda: _read_pdf(raw, max_pages), PARSE_TIMEOUT_SEC, "PDF")
        diags.extend(pdf_diags)
        fmt = "pdf"
    elif kind == "docx":
        text = run_with_timeout(lambda: _read_docx(raw), PARSE_TIMEOUT_SEC, "DOCX")
        fmt = "docx"
    else:
        text, warning = decode_text(raw)
        if warning:
            diags.append(Diagnostic("warning", "ENCODING", warning))
        fmt = "md" if suffix in (".md", ".markdown") else "txt"
        if suffix in (".pdf", ".docx", ".doc"):
            diags.append(
                Diagnostic(
                    "warning",
                    "EXTENSION_MISMATCH",
                    f"Расширение «{suffix}», но внутри обычный текст — файл прочитан как текст.",
                    "Проверьте, что выгружен правильный файл.",
                )
            )

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("﻿", "")
    if len(re.sub(r"\s", "", text)) < MIN_MEANINGFUL_CHARS:
        raise _fail("NO_TEXT", "В документе нет распознаваемого текста (только пробелы или пустые строки).", "Проверьте содержимое файла.")
    if len(text) > max_chars:
        diags.append(
            Diagnostic(
                "warning",
                "TEXT_TRUNCATED",
                f"Документ длиннее {max_chars} символов; проверена только первая часть.",
                "Разделите документ на части и проверьте каждую.",
            )
        )
        text = text[:max_chars]
        if line_pages is not None:
            line_pages = line_pages[: text.count("\n") + 1]
    lines = text.split("\n")
    doc = Document(
        path=str(p),
        name=p.name,
        title=_guess_title(lines, p.stem),
        fmt=fmt,
        text=text,
        lines=lines,
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=size,
        pages=n_pages,
        line_pages=line_pages,
    )
    return doc, diags
