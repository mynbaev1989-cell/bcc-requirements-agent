"""Документ кейса (input/requirements.md) в виде PDF: явные ID и число требований в offline-режиме."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from reqcheck.loaders import load_document
from reqcheck.pipeline import Options, run

ROOT = Path(__file__).resolve().parents[1]
CASE_MD = ROOT / "input" / "requirements.md"
FONT = ROOT / "reqcheck" / "fonts" / "DejaVuSans.ttf"  # тот же шрифт, что в report_pdf.py


def _md_to_plain(md: str, bullets: bool = True) -> str:
    """Как выглядел бы документ, распечатанный в PDF: без Markdown-разметки (**, `, #).
    bullets=True: маркер списка «* » печатается как «• »; False: маркер пропадает (PDF без маркеров)."""
    out = []
    for line in md.splitlines():
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^\*\s+", "• " if bullets else "", line)
        line = line.replace("**", "").replace("`", "")
        out.append(line)
    return "\n".join(out)


def _make_pdf(path: Path, text: str) -> None:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_font("DejaVu", "", str(FONT))
    pdf.add_page()
    pdf.set_font("DejaVu", size=10)
    for line in text.split("\n"):
        pdf.multi_cell(0, 5, line or " ", new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


@pytest.fixture(scope="module")
def case_runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("case")
    if not CASE_MD.exists():  # input/ не в git: документ кейса скачивается командой из README (в CI — отдельным шагом)
        pytest.skip("нет input/requirements.md — скачайте документ кейса (см. README)")
    md_text = CASE_MD.read_bytes().decode("utf-8")
    md_path = d / "case.md"
    md_path.write_bytes(md_text.encode("utf-8"))
    pdf_path = d / "case.pdf"  # с маркерами списка «•»
    _make_pdf(pdf_path, _md_to_plain(md_text, bullets=True))
    pdf_nb_path = d / "case_nobullets.pdf"  # маркеры не напечатаны
    _make_pdf(pdf_nb_path, _md_to_plain(md_text, bullets=False))
    return {
        "pdf_path": pdf_path,
        "md": run(str(md_path), Options(offline=True)),
        "pdf": run(str(pdf_path), Options(offline=True)),
        "pdf_nb": run(str(pdf_nb_path), Options(offline=True)),
    }


def test_pdf_loads_as_pdf(case_runs):
    doc, _ = load_document(case_runs["pdf_path"])
    assert doc.fmt.lower() == "pdf"
    assert "Ф-1" in doc.text and "Уведомление о расходах" in doc.text  # кириллица прочиталась


@pytest.mark.parametrize("sid", ["Ф-1", "Ф-2", "Ф-3", "Ф-4"])
def test_explicit_ids_extracted_from_pdf(case_runs, sid):
    md_ids = {r.source_id for r in case_runs["md"].requirements}
    pdf_ids = {r.source_id for r in case_runs["pdf"].requirements}
    assert sid in md_ids, "ID должен находиться и в .md"
    assert sid in pdf_ids, f"{sid} не извлечён из PDF"


# Допуск по числу требований. Измерено: md=31, pdf=36 (+5). Причина не в потере требований: в PDF нет
# Markdown-структуры, поэтому заголовок сливается с абзацем («Нефункциональные требования Сервис ...»),
# строки таблицы чек-листа превращаются в абзацы с «|», а описания кодов ответа 200/400/401/403/500
# выделяются отдельными пунктами. Критерии приёмки из таблицы: md даёт 3 строки, pdf — 1 (таблица не распознана).
MAX_COUNT_DIFF = 5


def test_requirement_count_pdf_vs_md(case_runs):
    n_md = len(case_runs["md"].requirements)
    n_pdf = len(case_runs["pdf"].requirements)
    print(f"требований: md={n_md}, pdf={n_pdf}")
    assert abs(n_pdf - n_md) <= MAX_COUNT_DIFF, f"md={n_md}, pdf={n_pdf}"


# Регрессия: без маркеров списка строки «Ф-1. …»…«Ф-4. …» склеивались в один абзац, и явные ID терялись
@pytest.mark.parametrize("sid", ["Ф-1", "Ф-2", "Ф-3", "Ф-4"])
def test_explicit_ids_from_pdf_without_bullets(case_runs, sid):
    assert sid in {r.source_id for r in case_runs["pdf_nb"].requirements}


@pytest.mark.parametrize("text, expected", [
    ("Функциональные требования\nФ-1. Система должна слать push.\nФ-2. Сумма не более 1 млн.", ["Ф-1", "Ф-2"]),
    ("FR-3: Логин по SMS.\nFR-4) Пароль 8 символов.", ["FR-3", "FR-4"]),
    # перенос строки перед ссылкой без знака после ID — это продолжение абзаца, а не новый пункт
    ("Ф-5. Лимит задаётся согласно\nФТ-2 сумма в тенге.", ["Ф-5"]),
], ids=["заголовок-и-пункты", "латиница-двоеточие-скобка", "ссылка-на-переносе"])
def test_txt_items_without_bullets_split_by_explicit_id(tmp_path, text, expected):
    from reqcheck.loaders import load_document
    from reqcheck.segmenter import segment

    p = tmp_path / "doc.txt"
    p.write_bytes(text.encode("utf-8"))
    doc, _ = load_document(str(p))
    blocks = segment(doc)
    assert [b.source_id for b in blocks if b.source_id] == expected
