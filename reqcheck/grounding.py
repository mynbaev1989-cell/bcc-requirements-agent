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


_NUM = re.compile(r"\d(?:[\d ]*\d)?(?:[.,]\d+)?")
_NEG = re.compile(r"(?<!\w)(?:не|ни|нет|без)(?!\w)")


def _numbers(text: str) -> list[str]:
    return sorted(m.group(0).replace(" ", "").replace(",", ".") for m in _NUM.finditer(text))


def same_facts(quote: str, source: str) -> bool:
    """Нечёткое совпадение не должно менять смысл: числа цитаты есть в исходном тексте, отрицаний не больше.

    «…в течение 90 секунд» вместо «30 секунд» или добавленное «не» похожи по символам, но это другой текст."""
    have = _numbers(source)
    for n in _numbers(quote):
        if n not in have:
            return False
        have.remove(n)
    return len(_NEG.findall(quote)) <= len(_NEG.findall(source))


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
        for nl in self.norm_lines:
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
            hits: list[int] = []
            while start >= 0:
                pos, ok = start + len(pieces[0]), True
                for p in pieces[1:]:
                    nxt = self.joined.find(p, pos)
                    if nxt < 0 or nxt - pos > self.ELLIPSIS_GAP:
                        ok = False
                        break
                    pos = nxt + len(p)
                if ok:
                    hits.append(self._line_at(start))
                    if not near:
                        break
                start = self.joined.find(pieces[0], start + 1)
            if hits:  # как и у точной цитаты: из нескольких вхождений — ближайшее к требованию
                best = min(hits, key=lambda ln: (min(abs(ln - n) for n in near), ln)) if near else hits[0]
                return Quote(text=quote, line=best, found=True)
        if len(q) < self.MIN_FUZZY_LEN:
            return Quote(text=quote, found=False)
        best_ratio, best_line, best_window = 0.0, None, ""
        # Префильтр: окно должно содержать хотя бы половину «длинных» слов цитаты (иначе O(N) SequenceMatcher дорог)
        keywords = [w for w in re.findall(r"\w{5,}", q)][:12]
        need = max(1, len(keywords) // 2)
        for i in range(len(self.norm_lines)):
            nearby = " ".join(self.norm_lines[i:i + 3])
            if keywords and sum(1 for w in keywords if w in nearby) < need:
                continue
            window = self.norm_lines[i]
            for extra in range(0, 3):
                if extra:
                    if i + extra >= len(self.norm_lines):
                        break
                    window = window + " " + self.norm_lines[i + extra]
                if not window:
                    continue
                sm = SequenceMatcher(None, q, window, autojunk=False)
                blocks = [b for b in sm.get_matching_blocks() if b.size]
                if len(window) >= len(q):
                    # Совпавшие куски должны лежать рядом: делим на длину участка окна, который они покрывают.
                    # Иначе цитата, собранная из слов соседних строк, набрала бы высокий балл.
                    matched = sum(b.size for b in blocks)
                    span = blocks[-1].b + blocks[-1].size - blocks[0].b if blocks else 0
                    ratio = matched / max(len(q), span)
                else:
                    ratio = sm.ratio()
                if ratio > best_ratio:
                    # Строка — та, где начинается совпадение, а не первая строка окна: окно «пустая/чужая строка +
                    # нужная» набирает тот же балл и иначе сдвигало бы ссылку на строку выше
                    best_ratio, best_line = ratio, self._window_line(i, extra, blocks[0].b if blocks else 0)
                    best_window = window
        if best_ratio >= self.FUZZY_THRESHOLD and best_line and same_facts(q, best_window):
            return Quote(text=quote, line=best_line, found=True)
        return Quote(text=quote, found=False)

    def _window_line(self, i: int, extra: int, offset: int) -> int:
        """Номер строки (с 1) для позиции offset в окне из строк i..i+extra, склеенных пробелом."""
        pos = 0
        for k in range(extra + 1):
            pos += len(self.norm_lines[i + k]) + 1
            if offset < pos:
                return i + k + 1
        return i + extra + 1

    def page_of(self, line: int | None) -> int | None:
        if line is None or not self.doc.line_pages:
            return None
        idx = line - 1
        if 0 <= idx < len(self.doc.line_pages):
            return self.doc.line_pages[idx]
        return None
