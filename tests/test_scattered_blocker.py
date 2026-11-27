"""Блокирующий вопрос не связывает критичные дефекты из разных мест документа."""
from reqcheck.clarify import SCATTER_GAP, scattered_blocker
from reqcheck.models import CRITICAL, MAJOR, Finding, Question, Quote


def _f(fid: str, sev: str, *lines: int) -> Finding:
    return Finding(fid=fid, scope="requirement", criterion="unambiguous", severity=sev, title=fid,
                   explanation="", quotes=[Quote(text="x", line=n, found=True) for n in lines])


def test_two_critical_far_apart_is_scattered():
    fmap = {"F-1": _f("F-1", CRITICAL, 10), "F-2": _f("F-2", CRITICAL, 10 + SCATTER_GAP + 1)}
    assert scattered_blocker(Question(qid="Q-01", text="?", fids=["F-1", "F-2"]), fmap)


def test_critical_close_together_is_not_scattered():
    fmap = {"F-1": _f("F-1", CRITICAL, 39), "F-2": _f("F-2", CRITICAL, 40, 80)}
    assert not scattered_blocker(Question(qid="Q-01", text="?", fids=["F-1", "F-2"]), fmap)


def test_major_far_away_does_not_count():
    fmap = {"F-1": _f("F-1", CRITICAL, 10), "F-2": _f("F-2", MAJOR, 90)}
    assert not scattered_blocker(Question(qid="Q-01", text="?", fids=["F-1", "F-2", "F-404"]), fmap)
