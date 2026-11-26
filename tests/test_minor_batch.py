"""Сводный вопрос по мелким находкам без вопроса (with_minor_batch): каждая находка отчёта может быть закрыта ответом."""

from reqcheck.clarify import MAX_QUESTIONS, with_minor_batch
from reqcheck.models import Finding, Question, Quote


def _f(fid, sev, line):
    return Finding(fid=fid, scope="requirement", criterion="unambiguous", severity=sev, title=f"t{fid}",
                   explanation="e", quotes=[Quote(text="x", line=line, found=True)])


def _q(qid, fids, pr="important"):
    return Question(qid=qid, text=f"Вопрос {qid}?", priority=pr, fids=fids)


def test_uncovered_minor_findings_get_one_batch_question():
    findings = [_f("F-001", "critical", 3), _f("F-002", "minor", 7), _f("F-003", "minor", 9), _f("F-004", "minor", 12)]
    qs = with_minor_batch([_q("Q-01", ["F-001"], "blocker"), _q("Q-02", ["F-004"])], findings)
    assert [q.qid for q in qs] == ["Q-01", "Q-02", "Q-03"]  # номера прежних вопросов не сдвигаются
    batch = qs[-1]
    assert batch.fids == ["F-002", "F-003"] and batch.priority == "clarify"
    assert batch.options and batch.assumption and batch.lines == [7, 9]
    assert "F-002" in batch.text and batch.text.endswith("?")


def test_no_batch_when_every_finding_is_covered():
    findings = [_f("F-001", "major", 3), _f("F-002", "minor", 7)]
    qs = [_q("Q-01", ["F-001", "F-002"])]
    assert with_minor_batch(qs, findings) is qs


def test_major_findings_never_go_to_batch():
    findings = [_f("F-001", "major", 3), _f("F-002", "minor", 7)]
    qs = with_minor_batch([], findings)
    assert len(qs) == 1 and qs[0].fids == ["F-002"]


def test_long_list_is_shortened_in_text_but_kept_in_fids():
    findings = [_f(f"F-{i:03d}", "minor", i) for i in range(1, 16)]
    batch = with_minor_batch([], findings)[0]
    assert len(batch.fids) == 15 and "и ещё 3" in batch.text


def test_batch_never_pushes_out_blockers():
    findings = [_f(f"F-{i:03d}", "critical", i) for i in range(1, MAX_QUESTIONS + 1)] + [_f("F-099", "minor", 99)]
    blockers = [_q(f"Q-{i:02d}", [f"F-{i:03d}"], "blocker") for i in range(1, MAX_QUESTIONS + 1)]
    qs = with_minor_batch(blockers, findings)
    assert sum(q.priority == "blocker" for q in qs) == MAX_QUESTIONS
