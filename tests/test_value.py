"""Тесты value.py: оценка эффекта и сценарии."""


def test_avoided_rework_is_discounted_by_measured_precision(monkeypatch):
    from reqcheck.value import estimate
    monkeypatch.delenv("REQCHECK_PRECISION", raising=False)
    v = estimate(40, 30, n_critical=4, agent_seconds=300, n_major=6)
    assert v.precision == 0.85 and v.avoided_rework_hours == round((4 + 0.5 * 6) * 0.85 * 1 * 6, 1)
    monkeypatch.setenv("REQCHECK_PRECISION", "1")
    assert estimate(40, 30, 4, 300, n_major=6).avoided_rework_hours == 42.0


def test_value_formula_shows_configured_major_weight(monkeypatch):
    from reqcheck.value import estimate
    monkeypatch.setenv("REQCHECK_MAJOR_WEIGHT", "0.3")
    v = estimate(10, 5, 1, 60, n_major=4)
    assert v.major_weight == 0.3 and v.avoided_rework_hours == round((1 + 0.3 * 4) * 0.85 * 1 * 6, 1)


def test_value_scenarios_conservative_and_time_only(monkeypatch):
    from reqcheck.value import estimate
    for k in ("REQCHECK_ESCALATION", "REQCHECK_PRECISION", "REQCHECK_CONSERVATIVE_ESCALATION",
              "REQCHECK_CONSERVATIVE_PRECISION", "REQCHECK_MAJOR_WEIGHT"):
        monkeypatch.delenv(k, raising=False)
    v = estimate(45, 29, 4, 240, n_major=20)
    base, cons, time_only = v.scenarios
    assert base["roi"] == v.roi and base["avoided_rework_hours"] == v.avoided_rework_hours
    assert cons["escalation"] == 3 and cons["precision"] == 0.7
    assert cons["avoided_rework_hours"] == round(14 * 0.7 * 2, 1)
    assert time_only["avoided_rework_hours"] == 0 and time_only["benefit_kzt"] == v.saved_kzt
    assert time_only["roi"] < cons["roi"] < base["roi"]
    # осторожная точность не выше измеренной
    monkeypatch.setenv("REQCHECK_PRECISION", "0.5")
    assert estimate(45, 29, 4, 240, n_major=20).scenarios[1]["precision"] == 0.5
