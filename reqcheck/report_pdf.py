"""PDF-версия отчёта: рендер нашего же Markdown (заголовки, абзацы, списки, таблицы, цитаты, код).

Шрифт DejaVu (кириллица) поставляется вместе с агентом в reqcheck/fonts — PDF одинаков на любой ОС.
"""

from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF
from fpdf.fonts import FontFace

FONT_DIR = Path(__file__).resolve().parent / "fonts"
MAX_CELL_CHARS = 700

EMOJI = {
    "🔴": "●", "🟠": "◑", "🟡": "○", "🟢": "✓", "✅": "✓", "❌": "✗", "➖": "–", "⚠️": "!", "⚠": "!", "ℹ️": "i", "ℹ": "i",
}
LIGHT_COLOR = {"🔴": (192, 32, 32), "🟠": (214, 110, 0), "🟡": (170, 140, 0), "🟢": (30, 140, 60)}


def _plain(text: str) -> str:
    for k, v in EMOJI.items():
        text = text.replace(k, v)
    text = text.replace("\\|", "|").replace("\\\\", "\\")
    # Символы вне BMP (редкие эмодзи) DejaVu не содержит
    return "".join(ch if ord(ch) < 0x10000 else "?" for ch in text)


def _md_inline(text: str) -> str:
    """Оставляем только **жирный** — его понимает fpdf2; прочие маркеры экранируем."""
    text = _plain(text)
    text = text.replace("__", "_​_").replace("--", "-​-")
    return text


class ReportPDF(FPDF):
    def __init__(self) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.add_font("DejaVu", "", str(FONT_DIR / "DejaVuSans.ttf"))
        self.add_font("DejaVu", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
        self.add_font("DejaVu", "I", str(FONT_DIR / "DejaVuSans-Oblique.ttf"))
        self.add_font("DejaVu", "BI", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(14, 14, 14)
        self.set_font("DejaVu", size=9)

    def footer(self) -> None:
        self.set_y(-11)
        self.set_font("DejaVu", size=7)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, f"reqcheck — отчёт о проверке требований · стр. {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def _col_widths(rows: list[list[str]], total: float, font_size: float) -> list[float]:
    """Ширина колонок: минимум — самое длинное слово (чтобы слова не рвались), остаток — по средней длине текста."""
    n = max(len(r) for r in rows)
    char_mm = font_size * 0.19
    mins, avgs = [0.0] * n, [0.0] * n
    for i in range(n):
        col = [r[i] for r in rows if i < len(r)]
        longest_word = max((len(w) for c in col for w in c.split()), default=1)
        mins[i] = min(longest_word, 16) * char_mm + 2.5
        avgs[i] = sum(min(len(c), 400) for c in col) / max(len(col), 1) + 1
    rest = total - sum(mins)
    if rest <= 0:
        s = sum(mins)
        return [total * w / s for w in mins]
    s = sum(avgs)
    return [mins[i] + rest * avgs[i] / s for i in range(n)]


def render_pdf(markdown_text: str, out_path: str) -> None:
    pdf = ReportPDF()
    pdf.add_page()
    width = pdf.w - pdf.l_margin - pdf.r_margin
    lines = markdown_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            pdf.ln(1.5)
            i += 1
            continue
        if stripped.startswith("```"):
            j = i + 1
            code: list[str] = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                code.append(lines[j])
                j += 1
            pdf.set_fill_color(244, 244, 244)
            pdf.set_font("DejaVu", size=8)
            pdf.multi_cell(width, 4.2, _plain("\n".join(code)), fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("DejaVu", size=9)
            pdf.ln(1)
            i = j + 1
            continue
        if stripped.startswith("|"):
            j = i
            rows: list[list[str]] = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                raw = lines[j].strip()
                if not re.match(r"^\|\s*:?-{2,}", raw):
                    cells = re.split(r"(?<!\\)\|", raw.strip("|"))
                    rows.append([_md_inline(c.strip())[:MAX_CELL_CHARS] for c in cells])
                j += 1
            if rows:
                ncol = max(len(r) for r in rows)
                rows = [r + [""] * (ncol - len(r)) for r in rows]
                size = 7 if ncol >= 5 else 8
                pdf.set_font("DejaVu", size=size)
                with pdf.table(col_widths=_col_widths(rows, width, size), text_align="LEFT", line_height=size * 0.5,
                               markdown=True, first_row_as_headings=True,
                               headings_style=FontFace(emphasis="BOLD", fill_color=(230, 236, 245)),
                               padding=1) as table:
                    for r in rows:
                        row = table.row()
                        for c in r:
                            row.cell(c)
                pdf.set_font("DejaVu", size=9)
                pdf.ln(2)
            i = j
            continue
        if stripped == "---":
            pdf.ln(2)
            pdf.set_draw_color(180, 180, 180)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3)
            i += 1
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            text = m.group(2)
            color = next((c for k, c in LIGHT_COLOR.items() if k in text), (20, 40, 80))
            size = {1: 15, 2: 12.5, 3: 10}[level]
            if level == 2 and pdf.get_y() > pdf.h - 60:
                pdf.add_page()
            pdf.ln(3 if level < 3 else 2)
            pdf.set_font("DejaVu", "B", size)
            pdf.set_text_color(*color)
            pdf.multi_cell(width, size * 0.5, _plain(text), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("DejaVu", size=9)
            pdf.ln(1)
            i += 1
            continue
        if stripped.startswith(">"):
            pdf.set_text_color(70, 70, 70)
            pdf.set_font("DejaVu", "I", 8.5)
            pdf.set_x(pdf.l_margin + 3)
            pdf.multi_cell(width - 3, 4.2, _plain(stripped.lstrip("> ")), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("DejaVu", size=9)
            pdf.set_text_color(0, 0, 0)
            i += 1
            continue
        lm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if lm:
            depth = len(lm.group(1)) // 2
            bullet = "•" if lm.group(2) in "-*" else lm.group(2)
            indent = 3 + depth * 5
            pdf.set_x(pdf.l_margin + indent)
            pdf.multi_cell(width - indent, 4.4, f"{bullet} " + _md_inline(lm.group(3)), markdown=True,
                           new_x="LMARGIN", new_y="NEXT")
            i += 1
            continue
        pdf.multi_cell(width, 4.4, _md_inline(stripped), markdown=True, new_x="LMARGIN", new_y="NEXT")
        i += 1
    pdf.output(out_path)
