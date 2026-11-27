"""Тесты clarify.py: Resolver, apply_effects, parse_answers, проверка формулировки."""

import pytest
from helpers import _doc
from reqcheck.clarify import Resolver, apply_effects, wording_matches_answer
from reqcheck.llm import FakeLLM, LLMError
from reqcheck.models import Finding, Question, Quote


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


def test_finding_skipped_by_resolver_is_marked_not_silently_open(tmp_path):
    doc = _doc(tmp_path, "# ТЗ\n\nОтвет 403 при запрете.\n\nДля чужой карты 404.\n")
    f1 = Finding(fid="F-001", scope="set", criterion="set_consistent", severity="major", title="403 или 404",
                 explanation="", quotes=[Quote("Ответ 403 при запрете", 3, True)])
    f2 = Finding(fid="F-002", scope="requirement", criterion="singular", severity="minor", title="Два условия",
                 explanation="", quotes=[Quote("Для чужой карты 404", 5, True)])
    q = Question(qid="Q-01", text="403 или 404?", priority="blocker", fids=["F-001", "F-002"])
    llm = FakeLLM({"resolver": {"effects": [{"fid": "F-002", "qid": "Q-01", "status": "resolved", "comment": "404"}]}})
    effects = Resolver(llm, doc).run([(q, {"qid": "Q-01", "question": "403 или 404?", "answer": "404"})], [f1, f2])
    assert {e["fid"]: e["status"] for e in effects} == {"F-002": "resolved", "F-001": "undecided"}
    resolved, followups = apply_effects(effects, [f1, f2], [q])
    assert resolved == {"F-002"} and not followups
    assert "не вынес" in f1.resolution
    assert [r for r, _ in llm.prompts] == ["answer_check", "resolver", "resolver"]  # пропуск переспрошен один раз


def test_resolver_asks_again_only_about_skipped_findings(tmp_path):
    doc = _doc(tmp_path, "# ТЗ\n\nОтвет 403 при запрете.\n\nДля чужой карты 404.\n")
    f1 = Finding(fid="F-001", scope="set", criterion="set_consistent", severity="major", title="403 или 404",
                 explanation="", quotes=[Quote("Ответ 403 при запрете", 3, True)])
    f2 = Finding(fid="F-002", scope="requirement", criterion="singular", severity="minor", title="Два условия",
                 explanation="", quotes=[Quote("Для чужой карты 404", 5, True)])
    q = Question(qid="Q-01", text="403 или 404?", priority="blocker", fids=["F-001", "F-002"])

    def answer(prompt):
        if "Связанные находки: F-001, F-002" in prompt:  # первый запрос: модель пропускает F-001
            return {"effects": [{"fid": "F-002", "qid": "Q-01", "status": "resolved", "comment": "404"}]}
        assert "Связанные находки: F-001\n" in prompt and "F-002 [" not in prompt  # переспрос — только о пропущенной
        return {"effects": [{"fid": "F-001", "qid": "Q-01", "status": "resolved", "comment": "выбран 404"}]}

    effects = Resolver(FakeLLM({"resolver": answer}), doc).run(
        [(q, {"qid": "Q-01", "question": "403 или 404?", "answer": "404"})], [f1, f2])
    assert {e["fid"]: e["status"] for e in effects} == {"F-002": "resolved", "F-001": "resolved"}


def test_failed_resolver_retry_keeps_first_answer(tmp_path):
    doc = _doc(tmp_path, "# ТЗ\n\nОтвет 403 при запрете.\n\nДля чужой карты 404.\n")
    f1 = Finding(fid="F-001", scope="set", criterion="set_consistent", severity="major", title="403 или 404",
                 explanation="", quotes=[Quote("Ответ 403 при запрете", 3, True)])
    f2 = Finding(fid="F-002", scope="requirement", criterion="singular", severity="minor", title="Два условия",
                 explanation="", quotes=[Quote("Для чужой карты 404", 5, True)])
    q = Question(qid="Q-01", text="403 или 404?", priority="blocker", fids=["F-001", "F-002"])
    calls = []

    def answer(prompt):
        calls.append(1)
        if len(calls) > 1:
            raise LLMError("квота")
        return {"effects": [{"fid": "F-002", "qid": "Q-01", "status": "resolved", "comment": "404"}]}

    effects = Resolver(FakeLLM({"resolver": answer}), doc).run(
        [(q, {"qid": "Q-01", "question": "403 или 404?", "answer": "404"})], [f1, f2])
    assert {e["fid"]: e["status"] for e in effects} == {"F-002": "resolved", "F-001": "undecided"}


def test_wording_check_covers_text_choices():
    answer = "Изменить метод на POST."
    assert not wording_matches_answer("Списание выполняется методом GET.", ["Изменить метод на POST"], answer)
    assert wording_matches_answer("Списание выполняется методом POST.", ["Изменить метод на POST"], answer)


def test_undecided_does_not_block_closure_by_another_answer():
    f = Finding(fid="F-001", scope="requirement", criterion="correct", severity="major", title="400 или 200",
                explanation="", rids=["R-001"])
    q1 = Question(qid="Q-01", text="?", priority="blocker", fids=["F-001"], answer="400")
    q2 = Question(qid="Q-02", text="?", priority="important", fids=["F-001"], answer="да")
    effects = [{"fid": "F-001", "qid": "Q-01", "status": "resolved", "comment": "400", "revised_wording": "", "follow_up": ""},
               {"fid": "F-001", "qid": "Q-02", "status": "undecided", "comment": "", "revised_wording": "", "follow_up": ""}]
    resolved, _ = apply_effects(effects, [f], [q1, q2])
    assert resolved == {"F-001"}


def test_nonstandard_resolver_status_counts_as_partial(tmp_path):
    doc = _doc(tmp_path, "# ТЗ\n\nGET /save сохраняет порог.\n")
    f = Finding(fid="F-001", scope="requirement", criterion="correct", severity="critical", title="GET для сохранения",
                explanation="", quotes=[Quote("GET /save сохраняет порог", 3, True)])
    q = Question(qid="Q-01", text="GET или POST?", priority="blocker", fids=["F-001"])
    llm = FakeLLM({"resolver": {"effects": [{"fid": "F-001", "qid": "Q-01", "status": "revised_wording",
                                             "comment": "автор выбрал POST"}]}})
    effects = Resolver(llm, doc).run([(q, {"qid": "Q-01", "question": "GET или POST?", "answer": "POST"})], [f])
    assert [(e["fid"], e["status"]) for e in effects] == [("F-001", "partial")]
    assert [r for r, _ in llm.prompts] == ["answer_check", "resolver"]  # решение есть — переспрашивать не нужно


def test_answers_parse_nested_followup_ids():
    """Уточнение к уточнению (Q-01.1.1) не склеивается с Q-01.1."""
    from reqcheck.clarify import parse_answers
    text = "## Q-01.1 · x\n\nВопрос: a\nОтвет: b\n\n## Q-01.1.1 · y\n\nВопрос: c\nОтвет: d\n"
    assert [a["qid"] for a in parse_answers(text)] == ["Q-01.1", "Q-01.1.1"]


def _f(fid, line, severity="major"):
    return Finding(fid=fid, scope="requirement", criterion="correct", severity=severity, title=fid, explanation="",
                   quotes=[Quote("x", line, True)])


def test_question_about_same_place_merges_into_broader_one():
    # Регрессия по кейсу: «какую операцию выполняет метод (строка 103)» повторял «почему GET вместо POST (103, 104)»
    from reqcheck.clarify import merge_contained

    fmap = {f.fid: f for f in (_f("F-1", 103, "critical"), _f("F-2", 104, "critical"), _f("F-3", 103))}
    broad = Question("", "Почему для сохранения настроек метод GET, а не POST, и где метод чтения?", "blocker",
                     fids=["F-1", "F-2"], lines=[103, 104])
    narrow = Question("", "Какую операцию выполняет метод — чтение или сохранение порога?", "important",
                      fids=["F-3"], lines=[103])
    out = merge_contained([broad, narrow], fmap)
    assert out == [broad] and broad.fids == ["F-1", "F-2", "F-3"]


@pytest.mark.parametrize("text, priority, lines", [
    # тот же пункт, но другой дефект — вопрос остаётся (строка 52: метод API и идемпотентность)
    ("Подтвердите, что повторная доставка события процессинга не шлёт второй push", "important", [103]),
    # вопрос шире хозяина по строкам — не сливается
    ("Почему метод сохранения GET и где метод чтения?", "important", [103, 200]),
    # более высокий приоритет не сливается в более низкий
    ("Какую операцию выполняет метод сохранения — чтение?", "blocker", [103]),
], ids=["другая-тема", "строки-шире", "приоритет-выше"])
def test_question_is_kept_when_not_a_duplicate(text, priority, lines):
    from reqcheck.clarify import merge_contained

    fmap = {f.fid: f for f in (_f("F-1", 103), _f("F-2", 104), _f("F-3", 103))}
    host = Question("", "Почему для сохранения метод GET, а не POST, и где метод чтения?", "important",
                    fids=["F-1", "F-2"], lines=[103, 104])
    other = Question("", text, priority, fids=["F-3"], lines=lines)
    assert len(merge_contained([host, other], fmap)) == 2


@pytest.mark.parametrize("text, addressee, expected", [
    ("Какой HTTP-статус возвращать при обращении к чужой карте?", "Бизнес-заказчик", "Архитектор"),
    ("Возвращать ошибку с кодом ответа 200 или 400?", "Бизнес-заказчик", "Архитектор"),
    ("Должен ли push уходить сразу при сохранении настройки (строка 126)?", "Бизнес-заказчик", "Бизнес-заказчик"),
    ("Какая дата определяет месяц покупки (строка 404)?", "Бизнес-заказчик", "Бизнес-заказчик"),
    ("Какой HTTP-метод использовать?", "Аналитик", "Аналитик"),  # меняем только бизнес-адресата
], ids=["http-статус", "код-ответа", "номер-строки-не-код", "строка-404", "не-бизнес"])
def test_technical_question_goes_to_architect(text, addressee, expected):
    from reqcheck.clarify import route_addressee

    assert route_addressee(Question("", text, addressee=addressee)).addressee == expected
