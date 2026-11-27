"""Проверка ответа автора кодом, без модели (kept_conflict).

Вопрос о противоречии, автор выбрал значение одной стороны и прямо оставил вторую («как есть», «не трогаем»):
после ответа документ всё ещё противоречив, закрывать нельзя. Молчит на честных ответах: замена старого значения,
сужение области, «это другое» (автор спорит с находкой).
"""

from helpers import _doc
from reqcheck.clarify import Resolver, kept_conflict
from reqcheck.llm import FakeLLM
from reqcheck.models import Finding, Question, Quote


def _contradiction():
    return Finding(fid="F-001", scope="set", criterion="consistent", severity="critical", title="Срок хранения",
                   explanation="", quotes=[Quote("Черновик хранится 30 суток", 3, True),
                                           Quote("Черновик удаляется через 45 суток", 5, True)])


# --- kept_conflict ---------------------------------------------------------------------------------

def test_kept_conflict_when_other_side_left_as_is():
    hit = kept_conflict("Берём 45 суток. Строку про 30 суток оставьте как есть.", _contradiction())
    assert hit is not None
    chosen, kept = hit
    assert "45" in chosen and "30" in kept


def test_kept_conflict_with_do_not_touch_marker():
    assert kept_conflict("Срок 45 суток, пункт с 30 сутками не трогаем.", _contradiction())


def test_no_conflict_when_other_side_replaced():
    assert kept_conflict("Берём 45 суток, строку про 30 суток исправить на 45.", _contradiction()) is None
    assert kept_conflict("Берём 45 суток вместо 30, остальное как есть.", _contradiction()) is None


def test_no_conflict_when_only_chosen_side_kept():
    # «как есть» относится к выбранной стороне, вторую сторону автор не упоминает
    assert kept_conflict("45 суток оставляем как есть, это верное значение.", _contradiction()) is None


def test_no_conflict_for_scope_narrowing_without_keep_marker():
    f = Finding(fid="F-002", scope="set", criterion="consistent", severity="major", title="Код ошибки",
                explanation="", quotes=[Quote("возвращается 403", 3, True), Quote("возвращается 404", 5, True)])
    assert kept_conflict("Для карт возвращаем 404. Код 403 остаётся только для запрета по роли.", f) is None


def test_no_conflict_for_single_side_finding():
    f = Finding(fid="F-003", scope="requirement", criterion="verifiable", severity="major", title="Срок",
                explanation="", quotes=[Quote("Черновик хранится 30 суток", 3, True)])
    assert kept_conflict("30 суток оставить как есть, 45 не нужно.", f) is None


def test_kept_conflict_parts_without_punctuation_and_split_by_lines():
    hit = kept_conflict("Принимаем 45 суток,\n30 суток оставить как есть.", _contradiction())
    assert hit == ("Принимаем 45 суток", "30 суток оставить как есть")


def test_no_conflict_when_author_says_it_is_another_matter():
    assert kept_conflict("Принимаем 45 суток. Срок 30 суток для другого процесса, не трогаем.", _contradiction()) is None


# --- в Resolver ------------------------------------------------------------------------------------

def _resolve(tmp_path, answer):
    doc = _doc(tmp_path, "# ТЗ\n\nЧерновик хранится 30 суток\n\nЧерновик удаляется через 45 суток\n")
    q = Question(qid="Q-01", text="Сколько хранить черновик?", priority="blocker", fids=["F-001"])
    reply = {"effects": [{"fid": "F-001", "qid": "Q-01", "status": "resolved", "comment": "срок выбран"}]}
    llm = FakeLLM({"resolver": reply})
    return Resolver(llm, doc).run([(q, {"qid": "Q-01", "question": q.text, "answer": answer})], [_contradiction()])


def test_resolver_keeps_finding_open_when_other_side_kept(tmp_path):
    [e] = _resolve(tmp_path, "Берём 45 суток. Строку про 30 суток оставьте как есть.")
    assert e["status"] == "open" and "оставляет" in e["comment"]
    assert "30 суток" in e["follow_up"]


def test_resolver_closes_answer_with_two_different_quantities(tmp_path):
    # «срок рассмотрения 5 дней, срок исполнения 10 дней» — две величины, а не противоречие: код не вмешивается
    [e] = _resolve(tmp_path, "Храним 45 суток. Срок рассмотрения заявки 5 дней, срок исполнения 10 дней.")
    assert e["status"] == "resolved"


def test_one_follow_up_per_answer_even_after_retry(tmp_path):
    doc = _doc(tmp_path, "# ТЗ\n\nЧерновик хранится 30 суток\n\nЧерновик удаляется через 45 суток\n")
    f1, f2 = _contradiction(), _contradiction()
    f2 = Finding(fid="F-002", scope=f1.scope, criterion=f1.criterion, severity=f1.severity, title=f1.title,
                 explanation="", quotes=f1.quotes)
    q = Question(qid="Q-01", text="Сколько хранить?", priority="blocker", fids=["F-001", "F-002"])
    calls = []

    def reply(prompt):  # первый запрос решает только F-001, переспрос — F-002
        calls.append(1)
        fid = "F-001" if len(calls) == 1 else "F-002"
        return {"effects": [{"fid": fid, "qid": "Q-01", "status": "resolved", "comment": "выбрано"}]}

    effects = Resolver(FakeLLM({"resolver": reply}), doc).run(
        [(q, {"qid": "Q-01", "question": q.text, "answer": "Берём 45 суток. Строку про 30 суток оставьте как есть."})],
        [f1, f2])
    assert len(calls) == 2 and all(e["status"] == "open" for e in effects)
    assert sum(bool(e["follow_up"]) for e in effects) == 1


def test_resolver_closes_on_honest_answer(tmp_path):
    [e] = _resolve(tmp_path, "Берём 45 суток, строку про 30 суток исправить на 45.")
    assert e["status"] == "resolved"


def test_reviewer_stops_only_on_http_codes_as_numbers():
    from reqcheck.agents import Reviewer
    r = Reviewer.__new__(Reviewer)  # нужен только разбор текста ошибки
    r.llm = FakeLLM()
    r._last_error = "Таймаут: ответ через 14003 мс, 1400 строк"
    assert not r._no_more_requests()
    r._last_error = "HTTP 403 Forbidden"
    assert r._no_more_requests()
