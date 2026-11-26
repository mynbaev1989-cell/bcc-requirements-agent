"""Новые детерминированные правила: «лимит без периода» и «поле в примере не совпадает с описанием API»."""

from pathlib import Path

import pytest
from helpers import _doc

from reqcheck.loaders import load_document
from reqcheck.rules import RuleEngine
from reqcheck.segmenter import segment

ROOT = Path(__file__).parent.parent


def _eng(tmp_path, text):
    doc = _doc(tmp_path, text)
    return doc, RuleEngine(doc, segment(doc))


# ---------- правило 1: лимит без периода ----------

def test_limit_without_period_fires(tmp_path):
    doc, eng = _eng(tmp_path, "# Док\n\n- Лимит снятия наличных по новой карте составляет 100 000 тенге.")
    (f,) = eng.limit_without_period()
    assert f.severity == "major" and f.criterion == "complete" and f.source == "rule"
    assert f.quotes[0].line == 3 and "За какой период" in f.question


@pytest.mark.parametrize("line", [
    "Лимит снятия составляет 100 000 тенге в сутки.",
    "Максимальная сумма одного перевода составляет 500 000 тенге.",
    "Лимит на одну операцию — 50 000 тенге.",
    "Максимальная сумма платежа 30 000 тенге.",
    "Разовый лимит 10 000 тенге.",
    "Лимит по одному реестру составляет 1 000 000 тенге.",
    "Не более 200 000 тенге за период действия акции.",
    "Лимит 100 000 тенге в течение 30 дней.",
    "Комиссия 1% от суммы платежа, но не более 500 тенге.",
    "Желаемый лимит от 100 000 до 3 000 000 тенге.",
    "Не более 3 попыток ввода кода.",          # решение: не денежный лимит — не срабатываем
    "Лимит ожидания ответа 5 секунд.",         # длительность, не лимит суммы
])
def test_limit_with_scope_or_not_money_is_silent(tmp_path, line):
    _, eng = _eng(tmp_path, "# Док\n\n- " + line)
    assert eng.limit_without_period() == []


def test_limit_in_api_table_is_silent(tmp_path):
    text = ("# API\n\n| Параметр | Тип | Описание |\n|---|---|---|\n"
            "| items | array | Не более 10 000 записей в одном запросе |\n"
            "| sum | integer | Максимальная сумма 100 000 тенге |\n")
    _, eng = _eng(tmp_path, text)
    assert eng.limit_without_period() == []


# ---------- правило 2: пример JSON против таблицы ----------

TABLE = "| Поле | Тип |\n|---|---|\n| deposit_id | string |\n| amount | integer |\n| term_months | integer |\n| rate_percent | decimal |\n"


def _example(body):
    return "# API\n\n" + TABLE + "\nПример:\n\n```json\n" + body + "\n```\n"


def test_example_mismatch_fires(tmp_path):
    _, eng = _eng(tmp_path, _example('{"deposit_id": "D", "amount": 1, "term_months": 12, "interest_rate": 12.0}'))
    (f,) = eng.api_example_mismatch()
    assert f.criterion == "set_consistent" and f.severity == "major"
    assert [q.text for q in f.quotes] == ['"interest_rate"', "rate_percent"]


def test_example_matches_table_is_silent(tmp_path):
    _, eng = _eng(tmp_path, _example('{"deposit_id": "D", "amount": 1, "term_months": 12, "rate_percent": 12.0}'))
    assert eng.api_example_mismatch() == []


def test_example_subset_of_table_is_silent(tmp_path):
    # в примере нет лишних ключей — частичный пример допустим
    _, eng = _eng(tmp_path, _example('{"deposit_id": "D", "amount": 1}'))
    assert eng.api_example_mismatch() == []


def test_example_of_other_object_is_silent(tmp_path):
    # пересечение < 50%: пример запроса/другого объекта, а не этой таблицы
    _, eng = _eng(tmp_path, _example('{"token": "t", "client": "c", "amount": 1}'))
    assert eng.api_example_mismatch() == []


def test_inline_json_line_and_no_table(tmp_path):
    _, eng = _eng(tmp_path, "# API\n\n" + TABLE + '\n{"deposit_id": "D", "amount": 1, "term_months": 2, "rate": 1}\n')
    assert len(eng.api_example_mismatch()) == 1
    _, eng2 = _eng(tmp_path, '# API\n\n{"a": 1, "b": 2}\n')
    assert eng2.api_example_mismatch() == []


def test_registered_in_run_set_rules(tmp_path):
    _, eng = _eng(tmp_path, "# Док\n\n- Лимит по карте 100 000 тенге.\n\n" + TABLE + '\n{"deposit_id": "D", "amount": 1, "term_months": 2, "rate": 1}\n')
    titles = " | ".join(f.title for f in eng.run_set_rules()[0])
    assert "Лимит без периода" in titles and "не совпадает с описанием API" in titles


# ---------- регрессия на размеченных документах ----------

def _run(rel):
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"нет {rel}")
    doc, _ = load_document(path)
    return doc, RuleEngine(doc, segment(doc))


def test_blind_card_block_limit_line_63():
    _, eng = _run("eval_blind/blind_card_block.md")
    lines = [f.quotes[0].line for f in eng.limit_without_period()]
    assert lines == [63]  # REQ-16; REQ-15 («в сутки») не затронут


def test_holdout_deposit_example_mismatch():
    _, eng = _run("eval_holdout/holdout_deposit.md")
    (f,) = eng.api_example_mismatch()
    assert [q.text for q in f.quotes] == ['"interest_rate"', "rate_percent"] and f.quotes[0].line == 91


@pytest.mark.parametrize("rel", ["eval/autopay_utilities.md", "eval/credit_card_application.md", "eval/p2p_phone.md",
                                 "eval_blind/blind_payroll.md", "eval_holdout/holdout_installments.md"])
def test_no_limit_findings_on_other_labeled_docs(rel):
    _, eng = _run(rel)
    assert eng.limit_without_period() == []


def test_limit_of_single_transfer_is_scoped():
    """«Лимит перевода — не более 30 000 тенге» — лимит одного перевода: область задана, находки нет."""
    from reqcheck.rules import LIMIT_SCOPE_RE
    assert LIMIT_SCOPE_RE.search("FR-3. Лимит перевода — не более 30 000 тенге.")
    assert not LIMIT_SCOPE_RE.search("Лимит снятия наличных по новой карте составляет 100 000 тенге.")
