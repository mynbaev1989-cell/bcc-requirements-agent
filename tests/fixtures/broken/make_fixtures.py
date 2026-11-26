"""Генерация «враждебных» входных файлов для проверки устойчивости reqcheck.

Запуск:  .venv\\Scripts\\python tests\\fixtures\\broken\\make_fixtures.py
Файлы создаются рядом со скриптом. long_line.md (~2 МБ) не хранится в git — только создаётся этим скриптом.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

from fpdf import FPDF
from pypdf import PdfReader, PdfWriter

HERE = Path(__file__).resolve().parent
FONT = HERE.parents[2] / "reqcheck" / "fonts" / "DejaVuSans.ttf"

REQ_TEXT = [
    "Требования к системе",
    "ФТ-1. Система должна позволять клиенту создать заявку на перевод.",
    "ФТ-2. Система должна быстро отправлять уведомление клиенту.",
    "НФТ-1. Время ответа не должно превышать 3 секунд.",
    "ФТ-3. Клиент может отменить заявку до подтверждения.",
]


def text_pdf(path: Path, lines: list[str]) -> None:
    """Обычный текстовый PDF с кириллицей."""
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", "", str(FONT))
    pdf.set_font("DejaVu", size=12)
    for ln in lines * 20:  # повторяем, чтобы файл был достаточно большим для обрезки
        pdf.multi_cell(0, 8, ln, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


def main() -> None:
    rnd = random.Random(42)

    # Пустой файл и нулевые байты
    (HERE / "empty.md").write_bytes(b"")
    (HERE / "zeros.md").write_bytes(b"\x00" * 2048)

    # PDF из случайных байтов
    (HERE / "random.pdf").write_bytes(bytes(rnd.getrandbits(8) for _ in range(4096)))

    # Текстовый PDF, обрезанный на трети
    full = HERE / "_full.pdf"
    text_pdf(full, REQ_TEXT)
    data = full.read_bytes()
    (HERE / "truncated.pdf").write_bytes(data[: len(data) // 3])
    full.unlink()

    # PDF с пустой страницей без текста (имитация скана)
    w = PdfWriter()
    w.add_blank_page(width=595, height=842)
    with open(HERE / "scan.pdf", "wb") as f:
        w.write(f)

    # PDF с паролем
    src = HERE / "_plain.pdf"
    text_pdf(src, REQ_TEXT)
    w = PdfWriter(clone_from=PdfReader(str(src)))
    w.encrypt(user_password="secret", owner_password="owner")
    with open(HERE / "encrypted.pdf", "wb") as f:
        w.write(f)
    src.unlink()

    # Текст с расширением .docx
    (HERE / "fake.docx").write_text("\n".join(REQ_TEXT), encoding="utf-8")

    # Кодировка Windows-1251
    (HERE / "cp1251.md").write_bytes("\n".join(REQ_TEXT).encode("cp1251"))

    # Только пробелы и переводы строк
    (HERE / "whitespace.md").write_text(" \n  \n\t\n" * 50, encoding="utf-8")

    # 200 строк художественного текста без требований
    story = [
        f"Вечер {i} выдался тихим, и старый рыбак долго смотрел на море, вспоминая лето."
        for i in range(1, 201)
    ]
    (HERE / "no_requirements.md").write_text("\n".join(story), encoding="utf-8")

    # Одна строка ~2 МБ без переводов строк
    (HERE / "long_line.md").write_text("Система должна работать быстро. " * 36000, encoding="utf-8")

    print("Создано:", ", ".join(sorted(p.name for p in HERE.iterdir() if p.suffix in (".md", ".pdf", ".docx"))))


if __name__ == "__main__":
    main()
