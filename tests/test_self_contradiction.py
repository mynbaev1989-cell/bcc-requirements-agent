"""Ответ автора, противоречащий сам себе, не закрывает находку: модель называет обе части, код сверяет их с ответом."""

from helpers import _doc
from reqcheck.clarify import Resolver, apply_effects, self_contradictions
from reqcheck.llm import FakeLLM
from reqcheck.models import Finding, Question, Quote

ANSWER = "Отправлять push сразу при сохранении порога, но push отправляется только при новой покупке выше порога."
A, B = "Отправлять push сразу при сохранении порога", "push отправляется только при новой покупке выше порога"


def _case(tmp_path):
    doc = _doc(tmp_path, "# ТЗ\n\nPush при превышении порога.\n\nПорог задаёт клиент.\n")
    f1 = Finding(fid="F-001", scope="requirement", criterion="unambiguous", severity="critical", title="Когда push",
                 explanation="", quotes=[Quote("Push при превышении порога", 3, True)])
    f2 = Finding(fid="F-002", scope="requirement", criterion="complete", severity="major", title="Порог",
                 explanation="", quotes=[Quote("Порог задаёт клиент", 5, True)])
    q = Question(qid="Q-01", text="Когда отправлять push?", priority="blocker", fids=["F-001", "F-002"])
    return doc, [f1, f2], q


def _resolve(tmp_path, clashes):
    doc, findings, q = _case(tmp_path)
    reply = {"self_contradictions": clashes, "effects": [
        {"fid": "F-001", "qid": "Q-01", "status": "resolved", "comment": "правило задано"},
        {"fid": "F-002", "qid": "Q-01", "status": "resolved", "comment": "правило задано"}]}
    effects = Resolver(FakeLLM({"resolver": reply}), doc).run(
        [(q, {"qid": "Q-01", "question": q.text, "answer": ANSWER})], findings)
    return effects, findings, q


def test_grounded_self_contradiction_keeps_findings_open_with_one_followup(tmp_path):
    effects, findings, q = _resolve(tmp_path, [{"qid": "Q-01", "part_a": A, "part_b": B}])
    assert {e["status"] for e in effects} == {"open"}
    assert all("противоречит сам себе" in e["comment"] for e in effects)
    resolved, followups = apply_effects(effects, findings, [q])
    assert resolved == set()
    assert [f.qid for f in followups] == ["Q-01.1"]  # одно уточнение на ответ, а не на каждую находку
    assert A in followups[0].text and B in followups[0].text


def test_invented_part_does_not_reopen_honest_answer(tmp_path):
    effects, _, _ = _resolve(tmp_path, [{"qid": "Q-01", "part_a": A, "part_b": "push не отправляется никогда"}])
    assert {e["status"] for e in effects} == {"resolved"}


def test_no_field_changes_nothing(tmp_path):
    effects, _, _ = _resolve(tmp_path, [])
    assert {e["status"] for e in effects} == {"resolved"}


def test_parts_must_differ_and_belong_to_that_answer():
    answers = {"Q-01": ANSWER, "Q-02": "Лимит 50 000 тенге."}
    bad = [
        {"qid": "Q-01", "part_a": A, "part_b": A},                      # одна и та же часть
        {"qid": "Q-01", "part_a": A, "part_b": "Отправлять push"},      # часть внутри другой
        {"qid": "Q-02", "part_a": A, "part_b": B},                      # части из чужого ответа
        {"qid": "Q-03", "part_a": A, "part_b": B},                      # нет такого ответа
        {"qid": "Q-01", "part_a": "", "part_b": B},
        "мусор",
    ]
    assert self_contradictions({"self_contradictions": bad}, answers) == {}
    assert self_contradictions(None, answers) == {}
    ok = self_contradictions({"self_contradictions": [{"qid": "Q-01", "part_a": f"«{A}»", "part_b": B.upper()}]}, answers)
    assert list(ok) == ["Q-01"]
