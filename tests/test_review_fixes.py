"""Регрессионные тесты на исправления по ревью кода: правила, сверка цитат, Critic, Resolver."""

from reqcheck.agents import Critic, visible_lines
from reqcheck.clarify import Resolver, wording_matches_answer
from reqcheck.grounding import Grounder
from reqcheck.llm import FakeLLM
from reqcheck.loaders import load_document
from reqcheck.models import Finding, Question, Quote
from reqcheck.rules import RuleEngine
from reqcheck.segmenter import segment


def _doc(tmp_path, text):
    path = tmp_path / "doc.md"
    # Нейтральная строка в конце: загрузчик отклоняет документы почти без текста
    path.write_text(text + "\n\nПримечание: тестовый документ для проверки правил агента.\n", encoding="utf-8")
    doc, _ = load_document(path)
    return doc


def test_mixed_units_ignores_words_like_centre(tmp_path):
    doc = _doc(tmp_path, "# ТЗ\n\nЛимит 50 000 тенге.\n\nДанные хранятся в процессинговом центре и контакт-центре.\n")
    assert RuleEngine(doc, segment(doc)).mixed_units() == []
    doc2 = _doc(tmp_path, "# ТЗ\n\nЛимит 50 000 тенге.\n\nСумма хранится в тиынах.\n")
    assert len(RuleEngine(doc2, segment(doc2)).mixed_units()) == 1


def test_empty_sections_ignore_comments_in_code_blocks(tmp_path):
    doc = _doc(tmp_path, "# ТЗ\n\n## API\n\n```bash\n# пример запроса\ncurl https://example\n```\n\n## Данные\n\nПоле amount.\n")
    titles = [f.title for f in RuleEngine(doc, segment(doc)).empty_sections()]
    assert titles == []


def test_weak_word_question_uses_rid_without_source_id(tmp_path):
    from reqcheck.models import Requirement
    doc = _doc(tmp_path, "# ТЗ\n\nСистема должна быстро отправлять SMS.\n")
    req = Requirement(rid="R-001", source_id="", source_id_assigned=True, summary="", quote="", line_start=3, line_end=3,
                      section="ТЗ")
    (f,) = RuleEngine(doc, segment(doc)).weak_words([req])
    assert "в R-001?" in f.question


def test_scattered_words_are_not_a_verified_quote(tmp_path):
    doc = _doc(tmp_path, "Лимит перевода задаёт банк.\nПорог не более 30 000 тенге.\nСписание в сутки один раз.\n")
    g = Grounder(doc)
    assert not g.locate("лимит перевода не более 1 000 000 тенге в сутки").found
    assert g.locate("порог не более 30 000 тенге").found


def test_repeated_quote_is_anchored_near_its_requirement(tmp_path):
    doc = _doc(tmp_path, "Содержание: ФТ-4 лимиты\n\n\n\n\nФТ-4. Лимит 30 000 тенге.\n")
    g = Grounder(doc)
    assert g.locate("ФТ-4").line == 1  # без подсказки — первое вхождение
    assert g.locate("ФТ-4", (6,)).line == 6


def test_critic_reject_after_duplicate_keeps_the_duplicate(tmp_path):
    doc = _doc(tmp_path, "# ТЗ\n\nЛимит 30 000 тенге.\n")
    f1 = Finding(fid="T-001", scope="set", criterion="set_consistent", severity="critical", title="A", explanation="",
                 rids=["R-001"], source="llm", verified=True)
    f2 = Finding(fid="T-002", scope="set", criterion="set_consistent", severity="critical", title="B", explanation="",
                 rids=["R-002"], source="llm", verified=True)
    llm = FakeLLM({"critic": {"verdicts": [{"fid": "T-002", "verdict": "duplicate", "duplicate_of": "T-001"},
                                           {"fid": "T-001", "verdict": "reject", "reason": "нет дефекта"}]}})
    kept, dropped = Critic(llm, doc).run([f1, f2])
    assert [f.fid for f in kept] == ["T-002"] and [f.fid for f in dropped] == ["T-001"]


def test_resolver_cannot_close_finding_of_another_question(tmp_path):
    doc = _doc(tmp_path, "# ТЗ\n\nЛимит 500 000 тенге.\n\nGET /pay списывает деньги.\n")
    limit = Finding(fid="F-001", scope="set", criterion="set_consistent", severity="critical", title="Лимит",
                    explanation="", quotes=[Quote("Лимит 500 000 тенге", 3, True)])
    api = Finding(fid="F-002", scope="set", criterion="correct", severity="critical", title="GET списывает",
                  explanation="", quotes=[Quote("GET /pay", 5, True)])
    q = Question(qid="Q-01", text="Какой лимит?", priority="blocker", fids=["F-001"])
    llm = FakeLLM({"resolver": {"effects": [
        {"fid": "F-001", "qid": "Q-01", "status": "resolved", "comment": "лимит выбран"},
        {"fid": "F-002", "qid": "Q-01", "status": "resolved", "comment": "заодно"}]}})
    effects = Resolver(llm, doc).run([(q, {"qid": "Q-01", "question": "Какой лимит?", "answer": "1 000 000"})],
                                     [limit, api])
    assert [e["fid"] for e in effects] == ["F-001"]


def test_wording_check_covers_text_choices():
    answer = "Изменить метод на POST."
    assert not wording_matches_answer("Списание выполняется методом GET.", ["Изменить метод на POST"], answer)
    assert wording_matches_answer("Списание выполняется методом POST.", ["Изменить метод на POST"], answer)


def test_visible_lines_reports_truncation(tmp_path):
    doc = _doc(tmp_path, "\n".join("строка " + "x" * 90 for _ in range(3000)))
    assert visible_lines(doc) < len(doc.lines)
    assert visible_lines(doc, max_chars=10**9) == len(doc.lines)
