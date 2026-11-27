"""Bootstrap-интервалы метрик замера (tools/stats.py)."""

from tools.stats import bootstrap_ci, intervals, precision, recall, severity_exact

DOCS = [
    {"found": 14, "gold_defects": 14, "precision": 0.93, "severity_exact_n": 5},
    {"found": 13, "gold_defects": 13, "precision": 0.81, "severity_exact_n": 9},
    {"found": 14, "gold_defects": 14, "precision": 0.88, "severity_exact_n": 5},
    {"found": 15, "gold_defects": 15, "precision": 0.82, "severity_exact_n": 8},
    {"found": 16, "gold_defects": 16, "precision": 0.96, "severity_exact_n": 8},
    {"found": 14, "gold_defects": 16, "precision": 0.80, "severity_exact_n": 6},
    {"found": 16, "gold_defects": 16, "precision": 0.85, "severity_exact_n": 6},
]


def test_point_values_match_headline_formulas():
    assert recall(DOCS) == 102 / 104
    assert abs(precision(DOCS) - sum(d["precision"] for d in DOCS) / 7) < 1e-12
    assert severity_exact(DOCS) == 47 / 102


def test_interval_contains_value_and_is_reproducible():
    ci = intervals(DOCS)
    for v in ci.values():
        assert v["lo"] <= v["value"] <= v["hi"]
    assert intervals(DOCS) == ci  # фиксированный seed — тот же результат


def test_constant_metric_has_zero_width():
    same = [{"found": 1, "gold_defects": 1, "precision": 0.9, "severity_exact_n": 1}] * 5
    assert bootstrap_ci(same, precision) == (0.9, 0.9)
    assert bootstrap_ci([], precision) == (0.0, 0.0)
