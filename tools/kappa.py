"""κ Коэна по критичности: разметка ↔ второй разметчик ↔ агент.

Запуск:  python tools/kappa.py eval/kappa/labels2.json
labels2.json — оценки второго разметчика по документу и id дефекта ({"<документ>": {"D1": "major", ...}}),
агент — поле severity_pairs полного варианта в eval/ablation.json (пересчитывается tools/ablation.py из кэша).
Печатает невзвешенную и линейно взвешенную κ (уровни упорядочены: minor < major < critical) и долю
точных совпадений для каждой пары на общих дефектах.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ("minor", "major", "critical")


def cohen_kappa(a: list[str], b: list[str], weighted: bool = False) -> float:
    """κ Коэна для двух равных списков меток из LEVELS; weighted — линейные веса по расстоянию уровней."""
    n = len(a)
    if n == 0 or n != len(b):
        return 0.0
    k = len(LEVELS)
    idx = {lv: i for i, lv in enumerate(LEVELS)}

    def w(i: int, j: int) -> float:
        return 1 - abs(i - j) / (k - 1) if weighted else float(i == j)

    obs = sum(w(idx[x], idx[y]) for x, y in zip(a, b)) / n
    pa = [sum(1 for x in a if x == lv) / n for lv in LEVELS]
    pb = [sum(1 for y in b if y == lv) / n for lv in LEVELS]
    exp = sum(pa[i] * pb[j] * w(i, j) for i in range(k) for j in range(k))
    return round((obs - exp) / (1 - exp), 3) if exp < 1 else 1.0


def pair_stats(a: list[str], b: list[str]) -> dict:
    return {"n": len(a), "exact": round(sum(x == y for x, y in zip(a, b)) / max(len(a), 1), 3),
            "kappa": cohen_kappa(a, b), "kappa_linear": cohen_kappa(a, b, weighted=True)}


def load(labels2_path: Path, ablation_path: Path = ROOT / "eval" / "ablation.json") -> dict[tuple[str, str], dict]:
    """{(документ, id дефекта): {"gold": …, "agent": … или None, "second": …}}."""
    second = json.loads(labels2_path.read_text(encoding="utf-8"))
    full = json.loads(ablation_path.read_text(encoding="utf-8"))[0]
    agent = {(r["document"], p["defect"]): p for r in full["docs"] for p in r.get("severity_pairs", [])}
    out = {}
    for doc, marks in second.items():
        for did, rec in marks.items():
            key = (doc, did)
            out[key] = {"gold": rec["gold"], "second": rec["second"],
                        "agent": agent[key]["agent"] if key in agent else None}
    return out


def report(rows: dict) -> dict[str, dict]:
    vals = list(rows.values())
    with_agent = [v for v in vals if v["agent"]]
    return {
        "разметка ↔ второй разметчик": pair_stats([v["gold"] for v in vals], [v["second"] for v in vals]),
        "разметка ↔ агент": pair_stats([v["gold"] for v in with_agent], [v["agent"] for v in with_agent]),
        "второй разметчик ↔ агент": pair_stats([v["second"] for v in with_agent], [v["agent"] for v in with_agent]),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 1
    res = report(load(Path(argv[1])))
    for name, st in res.items():
        print(f"{name}: n={st['n']}, точно {st['exact']:.0%}, κ={st['kappa']:.2f}, κ линейная={st['kappa_linear']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
