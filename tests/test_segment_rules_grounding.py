"""Нарезка, детерминированные правила и привязка цитат."""

from reqcheck.grounding import Grounder
from reqcheck.loaders import load_document
from reqcheck.rules import RuleEngine
from reqcheck.segmenter import extract_source_id, segment


def test_explicit_ids():
    assert extract_source_id("**Ф-1.** Держатель карты задаёт порог") == "Ф-1"
    assert extract_source_id("FR-12: The system shall") == "FR-12"
    assert extract_source_id("4.2.1 Система должна") == "4.2.1"
    assert extract_source_id("Система должна хранить данные") is None


def test_segment_structure(sample_md):
    doc, _ = load_document(sample_md)
    blocks = segment(doc)
    ids = [b.source_id for b in blocks if b.source_id]
    assert ids == ["FR-1", "FR-2", "FR-3"]
    fr1 = next(b for b in blocks if b.source_id == "FR-1")
    assert fr1.line_start == 7
    assert fr1.section.endswith("Функциональные требования")
    rows = [b for b in blocks if b.kind == "table_row"]
    assert rows[0].table_header == ["Исходные условия", "Действие", "Ожидаемый результат"]


def test_rules_find_generic_defects(sample_md):
    doc, _ = load_document(sample_md)
    blocks = segment(doc)
    engine = RuleEngine(doc, blocks)
    findings, checklist = engine.run_set_rules()
    titles = " | ".join(f.title for f in findings)
    assert "не заполнен" in titles  # «заполняются тестировщиками»
    assert "Смешение единиц" in titles  # тенге и тиыны
    assert "Не указаны версия" not in titles  # версия есть
    assert len(checklist) == 1 and checklist[0].question.startswith("Требование может быть")
    # каждая находка правил ссылается на строку документа
    assert all(q.line for f in findings for q in f.quotes)


def test_weak_words(sample_md):
    from reqcheck.agents import number_requirements, requirements_from_blocks

    doc, _ = load_document(sample_md)
    blocks = segment(doc)
    reqs = requirements_from_blocks(blocks)
    number_requirements(reqs)
    weak = RuleEngine(doc, blocks).weak_words(reqs)
    assert any("быстро" in f.title for f in weak)


def test_grounding_exact_fuzzy_and_hallucination(sample_md):
    doc, _ = load_document(sample_md)
    g = Grounder(doc)
    exact = g.locate("Лимит перевода — от 100 до 50 000 тенге включительно")
    assert exact.found and exact.line == 8
    # кавычки, ё/е, тире и markdown-звёздочки не мешают
    fuzzy = g.locate("**FR-1.** Система должна отправлять SMS клиенту быстро после оплаты")
    assert fuzzy.found and fuzzy.line == 7
    fake = g.locate("Система должна поддерживать оплату криптовалютой через блокчейн")
    assert not fake.found
