"""κ Коэна (tools/kappa.py) и пары критичности в eval_score."""

import json

from tools.kappa import cohen_kappa, load, pair_stats, report


def test_kappa_known_values():
    a = ["major", "major", "minor", "critical"]
    assert cohen_kappa(a, a) == 1.0
    # Классический пример: наблюдаемое 0,5 при ожидаемом 0,5 → κ = 0
    assert cohen_kappa(["major", "minor", "major", "minor"], ["major", "major", "minor", "minor"]) == 0.0
    assert cohen_kappa([], []) == 0.0


def test_linear_weights_credit_adjacent_levels():
    a = ["minor", "major", "critical", "major"]
    near = ["major", "critical", "major", "minor"]  # все расхождения на один уровень
    assert cohen_kappa(a, near, weighted=True) > cohen_kappa(a, near)
    assert pair_stats(a, a)["exact"] == 1.0


def test_load_and_report(tmp_path):
    abl = [{"docs": [{"document": "d.md", "severity_pairs": [
        {"defect": "D1", "gold": "major", "agent": "critical"},
        {"defect": "D2", "gold": "minor", "agent": "minor"}]}]}]
    (tmp_path / "ablation.json").write_text(json.dumps(abl), encoding="utf-8")
    l2 = {"d.md": {"D1": {"gold": "major", "second": "major"}, "D2": {"gold": "minor", "second": "minor"},
                   "D3": {"gold": "critical", "second": "major"}}}
    (tmp_path / "l2.json").write_text(json.dumps(l2), encoding="utf-8")
    rows = load(tmp_path / "l2.json", tmp_path / "ablation.json")
    assert rows[("d.md", "D3")]["agent"] is None  # агент дефект не нашёл
    res = report(rows)
    assert res["разметка ↔ второй разметчик"]["n"] == 3
    assert res["разметка ↔ агент"]["n"] == 2 and res["разметка ↔ агент"]["exact"] == 0.5
