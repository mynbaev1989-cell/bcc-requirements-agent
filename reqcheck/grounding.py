"""Привязка цитат к исходному тексту (защита от галлюцинаций LLM).

Каждая цитата из ответа модели ищется в документе: сначала точное совпадение после нормализации,
затем нечёткое (difflib) по окну строк. Не найденная цитата помечается found=False, а находка,
у которой нет ни одной подтверждённой цитаты, в отчёт не попадает (счётчик в паспорте).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from reqcheck.models import Document, Quote

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_QUOTES = dict.fromkeys(map(ord, "«»„“”‟\"'’‘`"), "")


def normalize(text: str) -> str:
    t = text.replace(" ", " ").replace(" ", " ").replace("ё", "е").replace("Ё", "Е")
    t = t.translate(_DASHES).translate(_QUOTES)
    t = re.sub(r"(\*\*|__|\*|~~)", "", t)
    t = t.replace("|", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


class Grounder:
    """Индекс документа для быстрого поиска цитат."""

    FUZZY_THRESHOLD = 0.86
    MIN_FUZZY_LEN = 12
    ELLIPSIS_GAP = 400  # «…» в цитате пропускает не больше ~400 символов (несколько строк)

    def __init__(self, doc: Document):
        self.doc = doc
        self.norm_lines = [normalize(line) for line in doc.lines]
        # Склеенный нормализованный текст + отображение позиции → номер строки
        parts: list[str] = []
        self._offsets: list[int] = []
        pos = 0
        for i, nl in enumerate(self.norm_lines):
            self._offsets.append(pos)
            parts.append(nl)
            pos += len(nl) + 1
        self.joined = " ".join(parts)

    def _line_at(self, pos: int) -> int:
        lo, hi = 0, len(self._offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._offsets[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    def locate(self, quote: str, near: tuple[int, ...] = ()) -> Quote:
        """Найти цитату в документе (точно, по частям с многоточием, нечётко). Результат кэшируется.

        near — строки требований, к которым относится цитата: если цитата встречается в документе
        несколько раз («ФТ-4», «30 000 тенге»), берётся вхождение, ближайшее к ним, а не первое.
        """
        memo = self.__dict__.setdefault("_memo", {})
        key = (quote, near)
        if key not in memo:
            memo[key] = self._locate(quote, near)
        found = memo[key]
        return Quote(text=quote, line=found.line, found=found.found)

    def _locate(self, quote: str, near: tuple[int, ...] = ()) -> Quote:
        q = normalize(quote).strip(" .;:,")
        if len(q) < 3:
            return Quote(text=quote, found=False)
        idx = self.joined.find(q)
        if idx >= 0:
            lines = [self._line_at(idx)]
            while near and (idx := self.joined.find(q, idx + 1)) >= 0:
                lines.append(self._line_at(idx))
            best = min(lines, key=lambda ln: (min(abs(ln - n) for n in near), ln)) if near else lines[0]
            return Quote(text=quote, line=best, found=True)
        # Цитата с многоточием: ищем по частям
        if "..." in q or "…" in q:
            pieces = [p.strip() for p in re.split(r"\.\.\.|…", q) if len(p.strip()) >= 6]
            # Части ищем по порядку и рядом друг с другом: иначе цитату можно склеить из фраз разных разделов
            start = self.joined.find(pieces[0]) if pieces else -1
            while start >= 0:
                pos, ok = start + len(pieces[0]), True
                for p in pieces[1:]:
                    nxt = self.joined.find(p, pos)
                    if nxt < 0 or nxt - pos > self.ELLIPSIS_GAP:
                        ok = False
                        break
                    pos = nxt + len(p)
                if ok:
                    return Quote(text=quote, line=self._line_at(start), found=True)
                start = self.joined.find(pieces[0], start + 1)
        if len(q) < self.MIN_FUZZY_LEN:
            return Quote(text=quote, found=False)
        best_ratio, best_line = 0.0, None
        # Префильтр: окно должно содержать хотя бы половину «длинных» слов цитаты (иначе O(N) SequenceMatcher дорог)
        keywords = [w for w in re.findall(r"\w{5,}", q)][:12]
        need = max(1, len(keywords) // 2)
        for i in range(len(self.norm_lines)):
            near = " ".join(self.norm_lines[i:i + 3])
            if keywords and sum(1 for w in keywords if w in near) < need:
                continue
            window = self.norm_lines[i]
            for extra in range(0, 3):
                if extra:
                    if i + extra >= len(self.norm_lines):
                        break
                    window = window + " " + self.norm_lines[i + extra]
                if not window:
                    continue
                if len(window) >= len(q):
                    # Совпавшие куски должны лежать рядом: делим на длину участка окна, который они покрывают.
                    # Иначе цитата, собранная из слов соседних строк, набрала бы высокий балл.
                    sm = SequenceMatcher(None, q, window, autojunk=False)
                    blocks = [b for b in sm.get_matching_blocks() if b.size]
                    matched = sum(b.size for b in blocks)
                    span = blocks[-1].b + blocks[-1].size - blocks[0].b if blocks else 0
                    ratio = matched / max(len(q), span)
                else:
                    ratio = SequenceMatcher(None, q, window, autojunk=False).ratio()
                if ratio > best_ratio:
                    best_ratio, best_line = ratio, i + 1
        if best_ratio >= self.FUZZY_THRESHOLD and best_line:
            return Quote(text=quote, line=best_line, found=True)
        return Quote(text=quote, found=False)

    def page_of(self, line: int | None) -> int | None:
        if line is None or not self.doc.line_pages:
            return None
        idx = line - 1
        if 0 <= idx < len(self.doc.line_pages):
            return self.doc.line_pages[idx]
        return None
