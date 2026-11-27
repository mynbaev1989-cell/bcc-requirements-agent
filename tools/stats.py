"""Доверительные интервалы для метрик замера: bootstrap по документам (стандартная библиотека, без numpy).

Документы, а не дефекты — единица повторной выборки: дефекты одного документа зависимы (один стиль, один автор),
и выборка по дефектам дала бы ложно узкий интервал. Документов всего 7, поэтому интервалы широкие — так и есть.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

N_RESAMPLES = 10_000
SEED = 0


def bootstrap_ci(docs: Sequence[dict], stat: Callable[[Sequence[dict]], float],
                 n: int = N_RESAMPLES, seed: int = SEED, level: float = 0.95) -> tuple[float, float]:
    """Перцентильный интервал статистики stat по повторным выборкам документов с возвращением."""
    if not docs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    k = len(docs)
    values = sorted(stat([docs[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    lo = values[int((1 - level) / 2 * n)]
    hi = values[min(int((1 + level) / 2 * n), n - 1)]
    return (round(lo, 3), round(hi, 3))


def recall(docs: Sequence[dict]) -> float:
    return sum(d["found"] for d in docs) / max(sum(d["gold_defects"] for d in docs), 1)


def precision(docs: Sequence[dict]) -> float:
    """Средняя по документам — как заголовочная цифра точности."""
    return sum(d["precision"] for d in docs) / max(len(docs), 1)


def severity_exact(docs: Sequence[dict]) -> float:
    return sum(d["severity_exact_n"] for d in docs) / max(sum(d["found"] for d in docs), 1)


METRICS = {"recall": recall, "precision": precision, "severity_exact": severity_exact}


def intervals(docs: Sequence[dict]) -> dict[str, dict[str, float]]:
    """{метрика: {"value": …, "lo": …, "hi": …}} для полноты, точности и критичности."""
    out = {}
    for name, fn in METRICS.items():
        lo, hi = bootstrap_ci(docs, fn)
        out[name] = {"value": round(fn(docs), 3), "lo": lo, "hi": hi}
    return out
