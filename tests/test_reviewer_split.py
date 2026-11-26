"""Reviewer: деление пополам при сбое переспроса/первичного пакета (один раз, без рекурсии)."""

import threading

from reqcheck import agents
from reqcheck.agents import Reviewer
from reqcheck.grounding import Grounder
from reqcheck.llm import FakeLLM, LLMError
from reqcheck.models import Requirement

from helpers import _doc


def _reqs(n):
    return [Requirement(rid=f"R-{i:03d}", source_id="", source_id_assigned=True, summary="", quote=f"Требование {i}",
                        line_start=3, line_end=3, section="ТЗ", block_ids=[]) for i in range(1, n + 1)]


def _ok(rid):
    return {"rid": rid, "marks": [{"criterion": "verifiable", "status": "ok"}], "findings": []}


def _st(assessment):
    return next(m.status for m in assessment.marks if m.criterion == "verifiable")


def _rids_in(prompt, reqs):
    return [r.rid for r in reqs if f"{r.rid} (" in prompt]


class Scripted(FakeLLM):
    """Первый запрос с данным набором требований — по сценарию first, повторный (переспрос) — по retry_for."""

    def __init__(self, reqs, first, retry):
        super().__init__()
        self.reqs, self.first, self.retry = reqs, first, retry
        self.seen: set[tuple] = set()
        self.lock = threading.Lock()

    def generate_json(self, role, system, prompt):
        self.prompts.append((role, prompt))
        self.stats.add(role, calls=1)
        rids = tuple(_rids_in(prompt, self.reqs))
        with self.lock:
            again = rids in self.seen
            self.seen.add(rids)
        return (self.retry if again else self.first)(rids)


def _run(llm, reqs, tmp_path):
    doc = _doc(tmp_path, "# ТЗ\n\nТребование 1.\n")
    rv = Reviewer(llm, doc, Grounder(doc))
    a, f = rv.run(reqs, {})
    return rv, a


def test_retry_failure_splits_in_halves(tmp_path, monkeypatch):
    monkeypatch.setattr(agents, "REVIEW_BATCH", 4)
    reqs = _reqs(8)
    full = tuple(r.rid for r in reqs[4:])

    def first(rids):  # второй пакет модель пропускает целиком
        return {"assessments": [] if rids == full else [_ok(r) for r in rids]}

    def retry(rids):
        if rids == full:
            raise LLMError("сбой переспроса")
        return {"assessments": [_ok(r) for r in rids]}

    llm = Scripted(reqs, first, retry)
    rv, a = _run(llm, reqs, tmp_path)
    assert rv.skipped == []
    assert all(a[r.rid].marks[0].comment != "не оценено: модель пропустила требование" for r in reqs)
    assert len(llm.prompts) == 2 + 1 + 2  # 2 первичных, переспрос, 2 половины


def test_half_failure_goes_to_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(agents, "REVIEW_BATCH", 4)
    reqs = _reqs(8)
    full = tuple(r.rid for r in reqs[4:])
    bad_half = ("R-007", "R-008")

    def first(rids):
        return {"assessments": [] if rids == full else [_ok(r) for r in rids]}

    def retry(rids):
        if rids == full or rids == bad_half:
            raise LLMError("сбой")
        return {"assessments": [_ok(r) for r in rids]}

    # половина R-007/R-008 запрашивается впервые -> попадает в first, поэтому задаём сбой и там
    def first2(rids):
        if rids == bad_half:
            raise LLMError("сбой половины")
        return first(rids)

    llm = Scripted(reqs, first2, retry)
    rv, a = _run(llm, reqs, tmp_path)
    assert rv.skipped == ["R-007", "R-008"]
    assert len(llm.prompts) == 5  # без дальнейшей рекурсии
    assert "сбой половины" in " ".join(rv.errors)
    assert "R-005" in a and _st(a["R-005"]) == "ok"


def test_quota_error_no_split(tmp_path, monkeypatch):
    monkeypatch.setattr(agents, "REVIEW_BATCH", 4)
    reqs = _reqs(8)
    full = tuple(r.rid for r in reqs[4:])

    def first(rids):
        return {"assessments": [] if rids == full else [_ok(r) for r in rids]}

    def retry(rids):
        raise LLMError("Исчерпана дневная квота бесплатного ключа")

    llm = Scripted(reqs, first, retry)
    rv, _ = _run(llm, reqs, tmp_path)
    assert len(llm.prompts) == 3  # 2 первичных + 1 переспрос, половин нет
    assert rv.skipped == [r.rid for r in reqs[4:]]


def test_primary_batch_failure_splits(tmp_path, monkeypatch):
    monkeypatch.setattr(agents, "REVIEW_BATCH", 4)
    reqs = _reqs(4)
    full = tuple(r.rid for r in reqs)

    def first(rids):
        if rids == full:
            raise LLMError("сбой пакета")
        return {"assessments": [_ok(r) for r in rids]}

    llm = Scripted(reqs, first, first)
    rv, a = _run(llm, reqs, tmp_path)
    assert all(_st(a[r.rid]) == "ok" for r in reqs)
    assert len(llm.prompts) == 3


def test_primary_quota_failure_no_split_and_raises(tmp_path):
    import pytest
    reqs = _reqs(4)

    def boom(rids):
        raise LLMError("Исчерпан лимит вызовов LLM на прогон (5).")

    llm = Scripted(reqs, boom, boom)
    with pytest.raises(LLMError):
        _run(llm, reqs, tmp_path)
    assert len(llm.prompts) == 1


def test_success_path_prompts_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(agents, "REVIEW_BATCH", 4)
    reqs = _reqs(8)
    llm = FakeLLM({"reviewer": lambda p: {"assessments": [_ok(r.rid) for r in reqs if f"{r.rid} (" in p]}})
    doc = _doc(tmp_path, "# ТЗ\n\nТребование 1.\n")
    rv = Reviewer(llm, doc, Grounder(doc))
    rv.run(reqs, {})
    ctx = agents.numbered_document(doc)

    def expect(batch):
        listing = "\n\n".join(
            f"{r.rid} (в документе: без номера; строки 3-3; раздел: ТЗ; вид: other)\n{r.quote.strip()}" for r in batch)
        return f"{agents.prompts.REVIEWER}\n\nТребования для оценки:\n{listing}\n\nВесь документ (номер строки: текст):\n{ctx}"

    assert sorted(p for _, p in llm.prompts) == sorted([expect(reqs[:4]), expect(reqs[4:])])
    assert rv.skipped == [] and rv.errors == []
