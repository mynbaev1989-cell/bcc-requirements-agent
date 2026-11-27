"""Проверки ответа автора кодом, без модели.

1. kept_conflict: вопрос о противоречии, автор выбрал значение одной стороны и прямо оставил вторую
   («как есть», «не трогаем») — после ответа документ всё ещё противоречив, закрывать нельзя.
2. value_clash: в одном ответе одна и та же величина с двумя разными значениями.
Обе проверки должны молчать на честных ответах: диапазон, разные условия, замена старого значения.
"""

from helpers import _doc
from reqcheck.clarify import Resolver, kept_conflict, value_clash
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


# --- value_clash -----------------------------------------------------------------------------------

def test_value_clash_same_quantity_two_values():
    hit = value_clash("Платёж уменьшаем, срок остаётся 12 месяцев. Оставшийся срок при этом сокращается до 8 месяцев.")
    assert hit is not None and "12 месяцев" in hit[0] and "8 месяцев" in hit[1]


def test_value_clash_across_clauses():
    assert value_clash("Срок выпуска — не более 5 рабочих дней для всех каналов; оператору сообщать срок до 7 рабочих дней.")


def test_no_clash_for_range():
    assert value_clash("Сумма покупки от 15 000 до 1 500 000 тенге, сумма считается без комиссии.") is None


def test_no_clash_for_different_conditions():
    assert value_clash("Срок 2 рабочих дня для физических лиц, срок 5 рабочих дней для юридических лиц.") is None


def test_no_clash_for_different_periods_or_quantities():
    assert value_clash("Лимит 50 000 тенге в день и лимит 500 000 тенге в месяц.") is None
    assert value_clash("Ставка 12% годовых, комиссия за обслуживание 1% от суммы.") is None
    assert value_clash("Платёж списывается 25 числа, напоминание за 3 дня до списания.") is None


def test_no_clash_when_values_equal():
    assert value_clash("Срок 5 рабочих дней. Повторный срок тоже 5 рабочих дней.") is None


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


def test_resolver_keeps_finding_open_on_value_clash(tmp_path):
    [e] = _resolve(tmp_path, "Храним 45 суток, после удаления черновик хранится в архиве ещё 30 суток.")
    assert e["status"] == "open" and "противоречит сам себе" in e["comment"]


def test_resolver_closes_on_honest_answer(tmp_path):
    [e] = _resolve(tmp_path, "Берём 45 суток, строку про 30 суток исправить на 45.")
    assert e["status"] == "resolved"
