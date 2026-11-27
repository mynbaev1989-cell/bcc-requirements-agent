"""Агент «Проверка ответов»: конфликт внутри ответа, между ответами или с документом не даёт закрыть находку.

Модель только называет конфликт; код сверяет обе цитаты дословно (ответ — с текстом ответа, документ — через Grounder).
"""

from helpers import _doc
from reqcheck.clarify import Resolver, answer_conflicts
from reqcheck.llm import FakeLLM
from reqcheck.models import Finding, Question, Quote

DOC = "# ТЗ\n\nСрок рассмотрения заявки — 5 рабочих дней.\n\nСтавка по вкладу 10% годовых.\n"


def _case(tmp_path):
    doc = _doc(tmp_path, DOC)
    f1 = Finding(fid="F-001", scope="requirement", criterion="unambiguous", severity="critical", title="Срок",
                 explanation="", quotes=[Quote("Срок рассмотрения заявки — 5 рабочих дней", 3, True)])
    f2 = Finding(fid="F-002", scope="requirement", criterion="complete", severity="major", title="Ставка",
                 explanation="", quotes=[Quote("Ставка по вкладу 10% годовых", 5, True)])
    q1 = Question(qid="Q-01", text="Какой срок?", priority="blocker", fids=["F-001"])
    q2 = Question(qid="Q-02", text="Какая ставка?", priority="important", fids=["F-002"])
    return doc, [f1, f2], [q1, q2]


def _run(tmp_path, answers, conflicts, fail=False):
    doc, findings, qs = _case(tmp_path)
    resolver = {"effects": [{"fid": "F-001", "qid": "Q-01", "status": "resolved", "comment": "срок задан"},
                            {"fid": "F-002", "qid": "Q-02", "status": "resolved", "comment": "ставка задана"}]}
    llm = FakeLLM({"resolver": resolver, "answer_check": {"conflicts": conflicts}},
                  fail_roles={"answer_check"} if fail else None)
    pairs = [(q, {"qid": q.qid, "question": q.text, "answer": a}) for q, a in zip(qs, answers)]
    effects = Resolver(llm, doc).run(pairs, findings)
    return {e["fid"]: e for e in effects}, llm


ANS = ["Срок 3 рабочих дня, а для повторной заявки срок 4 рабочих дня.", "Ставка 12% годовых, срок 7 рабочих дней."]


def test_self_conflict_keeps_finding_open(tmp_path):
    out, _ = _run(tmp_path, ANS, [{"qid": "Q-01", "kind": "self", "part_a": "Срок 3 рабочих дня",
                                   "part_b": "для повторной заявки срок 4 рабочих дня"}])
    assert out["F-001"]["status"] == "open" and "сам себе" in out["F-001"]["comment"]
    assert "Какое из утверждений верно" in out["F-001"]["follow_up"]
    assert out["F-002"]["status"] == "resolved"


def test_conflict_with_other_answer(tmp_path):
    out, _ = _run(tmp_path, ANS, [{"qid": "Q-02", "kind": "other_answer", "part_a": "срок 7 рабочих дней",
                                   "part_b": "Срок 3 рабочих дня", "other_qid": "Q-01"}])
    assert out["F-002"]["status"] == "open" and "ответу на Q-01" in out["F-002"]["comment"]
    assert out["F-001"]["status"] == "resolved"


def test_conflict_with_document_line(tmp_path):
    out, _ = _run(tmp_path, ANS, [{"qid": "Q-02", "kind": "document", "part_a": "Ставка 12% годовых",
                                   "part_b": "Ставка по вкладу 10% годовых", "line": 5}])
    assert out["F-002"]["status"] == "open" and "документу (строка 5)" in out["F-002"]["comment"]


def test_document_conflict_ignored_when_author_replaces_text(tmp_path):
    answers = [ANS[0], "Заменить ставку: 12% годовых вместо прежней."]
    out, _ = _run(tmp_path, answers, [{"qid": "Q-02", "kind": "document", "part_a": "12% годовых",
                                       "part_b": "Ставка по вкладу 10% годовых", "line": 5}])
    assert out["F-002"]["status"] == "resolved"


def test_invented_quotes_do_not_open_finding(tmp_path):
    out, _ = _run(tmp_path, ANS, [
        {"qid": "Q-01", "kind": "self", "part_a": "Срок 2 рабочих дня", "part_b": "срок 4 рабочих дня"},  # нет в ответе
        {"qid": "Q-02", "kind": "other_answer", "part_a": "Ставка 12% годовых", "part_b": "Срок 9 рабочих дней",
         "other_qid": "Q-01"},  # второй части нет в другом ответе
        {"qid": "Q-02", "kind": "document", "part_a": "Ставка 12% годовых", "part_b": "Ставка 15% в месяц"},  # нет в документе
    ])
    assert out["F-001"]["status"] == "resolved" and out["F-002"]["status"] == "resolved"


def test_check_failure_does_not_block_resolver(tmp_path):
    out, llm = _run(tmp_path, ANS, [], fail=True)
    assert out["F-001"]["status"] == "resolved" and out["F-002"]["status"] == "resolved"
    assert [r for r, _ in llm.prompts] == ["answer_check", "resolver"]


def test_answer_check_prompt_carries_answers_and_document(tmp_path):
    _, llm = _run(tmp_path, ANS, [])
    prompt = dict(llm.prompts)["answer_check"]
    assert "Срок 3 рабочих дня" in prompt and "Ставка по вкладу 10% годовых" in prompt
    assert "<ответ автора>" in prompt


def test_answer_conflicts_rejects_bad_shapes(tmp_path):
    doc, _, _ = _case(tmp_path)
    by_q = {"Q-01": ANS[0]}
    assert answer_conflicts(None, by_q, doc) == {}
    assert answer_conflicts({"conflicts": ["x", {"qid": "Q-09", "kind": "self"}]}, by_q, doc) == {}
    # одинаковые части — не конфликт; неизвестный вид — не принимается
    same = {"qid": "Q-01", "kind": "self", "part_a": "Срок 3 рабочих дня", "part_b": "Срок 3 рабочих дня"}
    odd = {"qid": "Q-01", "kind": "magic", "part_a": "Срок 3 рабочих дня", "part_b": "срок 4 рабочих дня"}
    assert answer_conflicts({"conflicts": [same, odd]}, by_q, doc) == {}
    assert answer_conflicts({"conflicts": [{**odd, "kind": "document"}]}, by_q, None) == {}
