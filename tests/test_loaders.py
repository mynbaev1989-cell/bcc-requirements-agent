"""Чтение файлов: поддерживаемые форматы и повреждённые/пустые входы дают диагностику, а не исключение наружу."""

import pytest

from conftest import make_blank_pdf, make_encrypted_pdf, make_text_pdf
from reqcheck.loaders import LoadError, detect_format, load_document


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
