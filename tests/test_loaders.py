"""Чтение файлов: поддерживаемые форматы и повреждённые/пустые входы дают диагностику, а не исключение наружу."""

import pytest

from conftest import make_blank_pdf, make_encrypted_pdf, make_text_pdf
from reqcheck.loaders import LoadError, detect_format, load_document
from reqcheck.segmenter import segment


def test_markdown_ok(sample_md):
    doc, diags = load_document(sample_md)
    assert doc.fmt == "md"
    assert doc.title == "Тестовый документ требований"
    assert "FR-1" in doc.text
    assert not diags


def test_cp1251_text_is_decoded(tmp_path):
    p = tmp_path / "win.txt"
    p.write_bytes("Система должна сохранять лимит перевода клиента.".encode("cp1251"))
    doc, diags = load_document(p)
    assert "лимит" in doc.text
    assert any(d.code == "ENCODING" for d in diags)


def test_text_pdf(tmp_path):
    p = tmp_path / "req.pdf"
    make_text_pdf(p, "FR-1. Система должна отправлять SMS после оплаты.\nFR-2. Лимит от 100 до 50 000 тенге.")
    doc, _ = load_document(p)
    assert doc.fmt == "pdf"
    assert doc.pages == 1
    assert "SMS" in doc.text
    assert doc.line_pages and doc.line_pages[0] == 1


def test_docx(tmp_path):
    import docx

    d = docx.Document()
    d.add_heading("Требования", level=1)
    d.add_paragraph("FR-1. Система должна хранить историю операций 5 лет.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Параметр", "Значение"
    t.cell(1, 0).text, t.cell(1, 1).text = "amount", "1–100"
    p = tmp_path / "req.docx"
    d.save(p)
    doc, _ = load_document(p)
    assert doc.fmt == "docx"
    assert "# Требования" in doc.text
    assert "| amount | 1–100 |" in doc.text


@pytest.mark.parametrize(
    "name,content,code",
    [
        ("empty.md", b"", "FILE_EMPTY"),
        ("zeros.md", b"\x00" * 2048, "FILE_EMPTY"),
        ("spaces.md", b"   \n\n\t \n", "FILE_EMPTY"),
        ("short.md", "# A\n\nок".encode(), "NO_TEXT"),
        ("garbage.pdf", bytes(range(256)) * 40, "FILE_BINARY"),
        ("noise.md", b"\x01\x02\x03\x04binary\x05\x06" * 300, "FILE_BINARY"),
        ("fake.docx", b"PK\x03\x04this is not a zip at all" * 10, "FILE_BINARY"),
    ],
)
def test_broken_inputs_raise_diagnostic(tmp_path, name, content, code):
    p = tmp_path / name
    p.write_bytes(content)
    with pytest.raises(LoadError) as err:
        load_document(p)
    assert err.value.diagnostic.code == code
    assert err.value.diagnostic.message  # понятное сообщение на русском


def test_truncated_pdf(tmp_path):
    p = tmp_path / "full.pdf"
    make_text_pdf(p, "FR-1. Система должна отправлять SMS после оплаты.")
    raw = p.read_bytes()
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(raw[: len(raw) // 3])
    with pytest.raises(LoadError) as err:
        load_document(broken)
    assert err.value.diagnostic.code in {"PDF_CORRUPTED", "PDF_NO_TEXT", "PDF_NO_PAGES"}


def test_scanned_pdf_without_text(tmp_path):
    p = tmp_path / "scan.pdf"
    make_blank_pdf(p)
    with pytest.raises(LoadError) as err:
        load_document(p)
    assert err.value.diagnostic.code == "PDF_NO_TEXT"
    assert "OCR" in err.value.diagnostic.hint


def test_encrypted_pdf(tmp_path):
    p = tmp_path / "secret.pdf"
    make_encrypted_pdf(p, "FR-1. Система должна отправлять SMS.")
    with pytest.raises(LoadError) as err:
        load_document(p)
    assert err.value.diagnostic.code == "PDF_ENCRYPTED"


def test_missing_file_and_directory(tmp_path):
    with pytest.raises(LoadError) as err:
        load_document(tmp_path / "nope.md")
    assert err.value.diagnostic.code == "FILE_NOT_FOUND"
    with pytest.raises(LoadError) as err:
        load_document(tmp_path)
    assert err.value.diagnostic.code == "IS_DIRECTORY"


def test_size_limit(tmp_path):
    p = tmp_path / "big.md"
    p.write_text("Система должна работать. " * 5000, encoding="utf-8")
    with pytest.raises(LoadError) as err:
        load_document(p, max_bytes=1000)
    assert err.value.diagnostic.code == "FILE_TOO_LARGE"


def test_text_with_pdf_extension_is_read_with_warning(tmp_path):
    p = tmp_path / "actually_text.pdf"
    p.write_text("FR-1. Система должна сохранять лимит перевода клиента.", encoding="utf-8")
    assert detect_format(p.read_bytes(), ".pdf") == "binary"  # без сигнатуры %PDF это не PDF


def test_docx_auto_numbering_keeps_requirement_ids(tmp_path):
    import docx
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    d = docx.Document()
    d.add_heading("Требования", level=1)
    paras = [d.add_paragraph(t, style="List Number") for t in
             ("Система должна сохранять порог.", "Порог задаётся в тенге.", "Клиент видит итог.")]
    # Шаблон уровней как в банковских ТЗ: «ФТ-1.» и вложенный «ФТ-1.1.»
    num_id = d.styles["List Number"].element.pPr.numPr.numId.val
    root = d.part.numbering_part.element
    num = next(n for n in root.findall(qn("w:num")) if n.get(qn("w:numId")) == str(num_id))
    abs_id = num.find(qn("w:abstractNumId")).get(qn("w:val"))
    an = next(a for a in root.findall(qn("w:abstractNum")) if a.get(qn("w:abstractNumId")) == abs_id)
    if len(an.findall(qn("w:lvl"))) < 2:  # в шаблоне python-docx список одноуровневый — добавляем второй уровень
        import copy
        lvl1 = copy.deepcopy(an.find(qn("w:lvl")))
        lvl1.set(qn("w:ilvl"), "1")
        an.find(qn("w:lvl")).addnext(lvl1)
    for lvl in an.findall(qn("w:lvl")):
        ilvl = int(lvl.get(qn("w:ilvl")))
        lvl.find(qn("w:numFmt")).set(qn("w:val"), "decimal")
        lvl.find(qn("w:lvlText")).set(qn("w:val"), "ФТ-%1." if ilvl == 0 else "ФТ-%1.%2.")
    p_pr = paras[1]._p.get_or_add_pPr()  # второй пункт — вложенный уровень
    num_pr = OxmlElement("w:numPr")
    il, nid = OxmlElement("w:ilvl"), OxmlElement("w:numId")
    il.set(qn("w:val"), "1")
    nid.set(qn("w:val"), str(num_id))
    num_pr.append(il)
    num_pr.append(nid)
    p_pr.append(num_pr)
    path = tmp_path / "tz.docx"
    d.save(path)
    doc, _ = load_document(path)
    lines = [ln for ln in doc.lines if ln.strip()]
    assert "ФТ-1. Система должна сохранять порог." in lines
    assert "ФТ-1.1. Порог задаётся в тенге." in lines
    assert "ФТ-2. Клиент видит итог." in lines
    blocks = segment(doc)
    assert {b.source_id for b in blocks if b.source_id} >= {"ФТ-1", "ФТ-2"}


def _docx_list(tmp_path, texts, fmt="decimal", lvl_text="%1.", start=""):
    """DOCX со списком «List Number»; возвращает (document, paragraphs, root numbering, numId, abstractNumId)."""
    import docx
    from docx.oxml.ns import qn
    d = docx.Document()
    d.add_heading("Требования", level=1)
    paras = [d.add_paragraph(t, style="List Number") for t in texts]
    num_id = str(d.styles["List Number"].element.pPr.numPr.numId.val)
    root = d.part.numbering_part.element
    num = next(n for n in root.findall(qn("w:num")) if n.get(qn("w:numId")) == num_id)
    abs_id = num.find(qn("w:abstractNumId")).get(qn("w:val"))
    an = next(a for a in root.findall(qn("w:abstractNum")) if a.get(qn("w:abstractNumId")) == abs_id)
    lvl = an.find(qn("w:lvl"))
    lvl.find(qn("w:numFmt")).set(qn("w:val"), fmt)
    lvl.find(qn("w:lvlText")).set(qn("w:val"), lvl_text)
    if start != "":
        lvl.find(qn("w:start")).set(qn("w:val"), start)
    return d, paras, root, num_id, abs_id


def _add_num(root, num_id, abs_id, start_override=None):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    num = OxmlElement("w:num")
    num.set(qn("w:numId"), num_id)
    ref = OxmlElement("w:abstractNumId")
    ref.set(qn("w:val"), abs_id)
    num.append(ref)
    if start_override is not None:
        ov, so = OxmlElement("w:lvlOverride"), OxmlElement("w:startOverride")
        ov.set(qn("w:ilvl"), "0")
        so.set(qn("w:val"), str(start_override))
        ov.append(so)
        num.append(ov)
    root.append(num)


def _use_num(para, num_id):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    num_pr = OxmlElement("w:numPr")
    il, nid = OxmlElement("w:ilvl"), OxmlElement("w:numId")
    il.set(qn("w:val"), "0")
    nid.set(qn("w:val"), num_id)
    num_pr.append(il)
    num_pr.append(nid)
    para._p.get_or_add_pPr().append(num_pr)


def test_docx_numbering_continues_across_nums_and_restarts_on_override(tmp_path):
    d, paras, root, num_id, abs_id = _docx_list(tmp_path, ("Первый.", "Второй.", "Третий.", "Четвёртый."),
                                                lvl_text="4.%1.")
    _add_num(root, "901", abs_id)  # тот же список другим w:num — Word продолжает нумерацию
    _add_num(root, "902", abs_id, start_override=1)  # явный перезапуск
    _use_num(paras[1], "901")
    _use_num(paras[2], "902")
    _use_num(paras[3], "902")
    path = tmp_path / "tz.docx"
    d.save(path)
    lines = [ln for ln in load_document(path)[0].lines if ln.strip()]
    assert lines[1:] == ["4.1. Первый.", "4.2. Второй.", "4.1. Третий.", "4.2. Четвёртый."]


def test_docx_roman_numbering_and_broken_numbering_part(tmp_path):
    d, *_ = _docx_list(tmp_path, ("Цели.", "Требования."), fmt="upperRoman", lvl_text="%1.")
    d.save(tmp_path / "roman.docx")
    assert [ln for ln in load_document(tmp_path / "roman.docx")[0].lines if ln.strip()][1:] == ["I. Цели.", "II. Требования."]
    d, *_ = _docx_list(tmp_path, ("Система должна сохранять порог.",), start="x")  # битый w:start
    d.save(tmp_path / "broken.docx")
    doc, _ = load_document(tmp_path / "broken.docx")  # документ читается, просто без автонумерации
    assert "Система должна сохранять порог." in " ".join(doc.lines)


def test_docx_restart_from_nested_item_and_bad_level(tmp_path):
    import copy
    from docx.oxml.ns import qn
    d, paras, root, num_id, abs_id = _docx_list(tmp_path, ("Первый.", "Второй.", "Вложенный.", "Глубокий."))
    an = next(a for a in root.findall(qn("w:abstractNum")) if a.get(qn("w:abstractNumId")) == abs_id)
    lvls = {lv.get(qn("w:ilvl")): lv for lv in an.findall(qn("w:lvl"))}
    if "1" not in lvls:
        lvls["1"] = copy.deepcopy(lvls["0"])
        lvls["1"].set(qn("w:ilvl"), "1")
        an.append(lvls["1"])
    lvls["1"].find(qn("w:numFmt")).set(qn("w:val"), "decimal")
    lvls["1"].find(qn("w:lvlText")).set(qn("w:val"), "%1.%2.")
    bad = copy.deepcopy(lvls["0"])  # уровень 9 — в Word такого нет
    bad.set(qn("w:ilvl"), "9")
    an.append(bad)
    _add_num(root, "902", abs_id, start_override=1)
    _use_num(paras[2], "902")
    _use_num(paras[3], "902")
    paras[2]._p.pPr.findall(qn("w:numPr"))[-1].find(qn("w:ilvl")).set(qn("w:val"), "1")
    paras[3]._p.pPr.findall(qn("w:numPr"))[-1].find(qn("w:ilvl")).set(qn("w:val"), "9")
    d.save(tmp_path / "nested.docx")
    lines = [ln for ln in load_document(tmp_path / "nested.docx")[0].lines if ln.strip()]
    assert lines[1:] == ["1. Первый.", "2. Второй.", "1.1. Вложенный.", "* Глубокий."]  # битый уровень — без номера, но без падения
