"""Калибровка критичности кодом (reqcheck/severity.py)."""

from reqcheck.models import Finding
from reqcheck.severity import NOTE, calibrate, needs_downgrade


def _f(criterion, severity="critical", title="Не описано поведение при сбое", explanation="", critic=""):
    return Finding("F", "requirement", criterion, severity, title, explanation, critic=critic)


def test_vague_and_incomplete_critical_downgraded():
    for crit in ("verifiable", "unambiguous", "complete", "set_complete"):
        f = _f(crit)
        assert calibrate([f]) == [f] and f.severity == "major" and f.critic == NOTE


def test_conflicts_and_other_criteria_untouched():
    keep = [_f("complete", title="Противоречие сроков в ФТ-3 и АК-2"),
            _f("verifiable", explanation="значение расходится с таблицей API"),
            _f("correct", title="Метод GET меняет состояние"),
            _f("set_consistent"),
            _f("complete", severity="major"), _f("unambiguous", severity="minor")]
    assert calibrate(keep) == []
    assert [f.severity for f in keep] == ["critical"] * 4 + ["major", "minor"]


def test_critic_note_preserved_and_never_upgrades():
    f = _f("complete", critic="подтверждено")
    calibrate([f])
    assert f.critic.startswith("подтверждено; ") and NOTE in f.critic
    assert not needs_downgrade(_f("complete", severity="major"))
