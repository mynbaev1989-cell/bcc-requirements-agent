"""Правило «ограничение за период и правило, которое может его обойти» (RuleEngine.limit_bypass)."""

from reqcheck.loaders import load_document
from reqcheck.rules import RuleEngine
from reqcheck.segmenter import segment


def _hits(tmp_path, text):
    p = tmp_path / "doc.md"
    p.write_text(text, encoding="utf-8")
    doc = load_document(str(p))[0]
    return RuleEngine(doc, segment(doc)).limit_bypass()


def test_limit_and_bypass_with_common_subject(tmp_path):
    text = ("# Вход\n\n"
            "* **Т-1.** Клиент может запросить SMS-код не более 3 раз в сутки.\n"
            "* **Т-2.** После смены номера телефона SMS-код можно запросить снова.\n")
    hits = _hits(tmp_path, text)
    assert len(hits) == 1
    f = hits[0]
    assert [q.line for q in f.quotes] == [3, 4] and f.severity == "major" and f.criterion == "set_consistent"
    assert "строки 4" in f.question or "строка 4" in f.explanation


def test_no_hit_without_common_subject_or_without_bypass(tmp_path):
    assert _hits(tmp_path, "* Выписка формируется не более 1 раза в месяц.\n* Пароль можно сменить снова в любой момент.\n") == []
    assert _hits(tmp_path, "* Не более 5 переводов в сутки.\n* Перевод подтверждается кодом.\n") == []
    # Обе части в одном предложении — это уточнение правила, а не два правила
    assert _hits(tmp_path, "* Не более 2 выписок в месяц, повторно выписку можно заказать в отделении.\n") == []


def test_rule_finding_merged_into_llm_contradiction_on_same_lines():
    from reqcheck.agents import overlap_duplicates
    from reqcheck.models import Finding, Quote

    llm = Finding("F-1", "set", "set_consistent", "critical", "Противоречие лимита уведомлений", "",
                  quotes=[Quote("a", line=39, found=True), Quote("b", line=40, found=True)])
    rule = Finding("F-2", "set", "set_consistent", "major", "Ограничение за период и правило, которое может его обойти", "",
                   source="rule", quotes=[Quote("a", line=39, found=True), Quote("b", line=40, found=True)])
    other = Finding("F-3", "set", "set_consistent", "major", "Противоречие сроков", "",
                    quotes=[Quote("c", line=80, found=True)])
    assert overlap_duplicates([llm, rule, other]) == [(rule, llm)]
    assert overlap_duplicates([rule, other]) == []  # LLM этого не нашла — находка правила остаётся
