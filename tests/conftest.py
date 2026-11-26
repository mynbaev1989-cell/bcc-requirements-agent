import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLE_MD = """# Тестовый документ требований

Версия 1.0 от 01.11.2026

## Функциональные требования

* **FR-1.** Система должна отправлять SMS клиенту быстро после оплаты.
* **FR-2.** Лимит перевода — от 100 до 50 000 тенге включительно.
* **FR-3.** Лимит перевода — не более 30 000 тенге.

## Требования к данным

Хранить сумму в тиынах и лимит в тенге.

## Тестирование

заполняются тестировщиками

## Критерии приёмки

| Исходные условия | Действие | Ожидаемый результат |
| :--- | :--- | :--- |
| Лимит 50 000 | Перевести 40 000 | Перевод выполнен |

## Ревью

| Проверка | Проверено |
| :--- | :--- |
| Требование может быть протестировано | Да / Нет |
"""


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch, tmp_path):
    """Тесты никогда не ходят в настоящую модель и не читают .env."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REQCHECK_VERSION", "test")
    monkeypatch.setenv("REQCHECK_NO_DOTENV", "1")


@pytest.fixture
def sample_md(tmp_path) -> Path:
    p = tmp_path / "req.md"
    p.write_text(SAMPLE_MD, encoding="utf-8")
    return p


def make_text_pdf(path: Path, text: str) -> None:
    from fpdf import FPDF

    font = ROOT / "reqcheck" / "fonts" / "DejaVuSans.ttf"
    pdf = FPDF()
    pdf.add_font("DejaVu", "", str(font))
    pdf.add_page()
    pdf.set_font("DejaVu", size=11)
    for line in text.split("\n"):
        pdf.multi_cell(0, 6, line or " ", new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


def make_blank_pdf(path: Path) -> None:
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(width=595, height=842)
    with open(path, "wb") as fh:
        w.write(fh)


def make_encrypted_pdf(path: Path, text: str) -> None:
    from pypdf import PdfReader, PdfWriter

    plain = path.with_name("plain_" + path.name)
    make_text_pdf(plain, text)
    w = PdfWriter(clone_from=PdfReader(str(plain)))
    w.encrypt(user_password="secret", owner_password="owner")
    with open(path, "wb") as fh:
        w.write(fh)
