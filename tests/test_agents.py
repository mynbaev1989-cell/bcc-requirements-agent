"""Тесты agents.py: Extractor, Reviewer, Critic, слияние и сведение находок, _own_id."""

from helpers import _doc
from reqcheck.agents import Critic, visible_lines
from reqcheck.grounding import Grounder
from reqcheck.llm import FakeLLM, LLMError
from reqcheck.models import Finding, Quote
from reqcheck.segmenter import segment


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


def test_visible_lines_reports_truncation(tmp_path):
    doc = _doc(tmp_path, "\n".join("строка " + "x" * 90 for _ in range(3000)))
    assert visible_lines(doc) < len(doc.lines)
    assert visible_lines(doc, max_chars=10**9) == len(doc.lines)


def test_reviewer_sees_own_wording_when_paragraph_is_split(tmp_path):
    from reqcheck.agents import Reviewer
    from reqcheck.models import Block, Requirement
    text = "Открыть депозит может только клиент с ЭЦП. Клиент без ЭЦП получает отказ 403."
    doc = _doc(tmp_path, "# ТЗ\n\n" + text + "\n")
    block = Block(bid="B-001", kind="paragraph", text=text, line_start=3, line_end=3, section="ТЗ")
    reqs = [Requirement(rid=f"R-00{i}", source_id="", source_id_assigned=True, summary="", quote=q, line_start=3,
                        line_end=3, section="ТЗ", block_ids=["B-001"])
            for i, q in enumerate(text.replace(". ", ".|").split("|"), start=1)]
    llm = FakeLLM({"reviewer": {"assessments": []}})
    Reviewer(llm, doc, Grounder(doc)).run(reqs, {"B-001": block})
    listing = llm.prompts[0][1].split("Весь документ")[0]
    assert "Формулировка: Открыть депозит может только клиент с ЭЦП." in listing
    assert "Формулировка: Клиент без ЭЦП получает отказ 403." in listing


def test_reviewer_sees_full_text_of_long_single_requirement(tmp_path):
    from reqcheck.agents import Reviewer
    from reqcheck.models import Block, Requirement
    text = "ФТ-5. Система списывает комиссию. " + "Подробное условие списания. " * 20 + "Лимит 50 000 тенге."
    doc = _doc(tmp_path, "# ТЗ\n\n" + text + "\n")
    block = Block(bid="B-001", kind="paragraph", text=text, line_start=3, line_end=3, section="ТЗ")
    req = Requirement(rid="R-001", source_id="ФТ-5", source_id_assigned=False, summary="", quote=text[:60],
                      line_start=3, line_end=3, section="ТЗ", block_ids=["B-001"])
    llm = FakeLLM({"reviewer": {"assessments": []}})
    Reviewer(llm, doc, Grounder(doc)).run([req], {"B-001": block})
    listing = llm.prompts[0][1].split("Весь документ")[0]
    assert "Формулировка:" not in listing and "Лимит 50 000 тенге." in listing


def test_set_finding_repeating_requirement_finding_is_merged(tmp_path):
    from reqcheck.agents import overlap_duplicates
    get_req = Finding(fid="T-1", scope="requirement", criterion="appropriate", severity="critical",
                      title="Использование HTTP-метода GET для мутирующей операции сохранения настройки", explanation="",
                      rids=["R-025"], quotes=[Quote("Заменить настройку", 120, True)], source="llm")
    get_set = Finding(fid="T-2", scope="set", criterion="set_consistent", severity="major",
                      title="Использование метода GET для операции сохранения данных", explanation="",
                      rids=["R-017", "R-025"], quotes=[Quote("REST type | GET", 104, True), Quote("Заменить", 120, True)],
                      source="llm")
    other_set = Finding(fid="T-3", scope="set", criterion="set_complete", severity="major",
                        title="Не описан таймаут push-сервиса", explanation="", rids=["R-025"],
                        quotes=[Quote("Заменить настройку", 120, True)], source="llm")
    rule_set = Finding(fid="T-4", scope="set", criterion="set_validatable", severity="major",
                       title="Использование GET для сохранения", explanation="", rids=["R-025"],
                       quotes=[Quote("x", 120, True)], source="rule")
    pairs = overlap_duplicates([get_req, get_set, other_set, rule_set])
    assert pairs == [(get_set, get_req)]  # другой дефект на той же строке и находка правила не сливаются


def test_set_finding_with_requirement_criterion_keeps_set_scoring():
    """Признак сводной находки — явный: обычная находка по набору с кодом требования баллы требований не трогает."""
    from reqcheck.agents import sync_marks
    from reqcheck.models import RequirementAssessment, CriterionMark, per_requirement
    f = Finding(fid="T-1", scope="set", criterion="necessary", severity="major", title="Дубль R-001 и R-002",
                explanation="", rids=["R-001", "R-002"], source="llm")
    asm = {"R-001": RequirementAssessment("R-001", [CriterionMark("necessary", "ok")])}
    sync_marks([f], asm)
    assert not per_requirement(f) and asm["R-001"].marks[0].status == "ok"


def test_single_minor_singular_finding_is_left_as_is():
    from reqcheck.agents import collapse_singular
    f = Finding(fid="T-1", scope="requirement", criterion="singular", severity="minor", title="t", explanation="",
                rids=["R-001"])
    assert collapse_singular([f]) == ([f], 0)


def test_reviewer_asks_again_about_skipped_requirements(tmp_path):
    from reqcheck.agents import Reviewer
    from reqcheck.models import Requirement
    doc = _doc(tmp_path, "# ТЗ\n\nФ-1. Система должна сохранять порог.\n\nФ-2. Порог задаётся в тенге.\n")
    reqs = [Requirement(rid="R-001", source_id="Ф-1", source_id_assigned=False, summary="a", quote="Ф-1. Система",
                        line_start=3, line_end=3, section="", kind="functional", block_ids=[]),
            Requirement(rid="R-002", source_id="Ф-2", source_id_assigned=False, summary="b", quote="Ф-2. Порог",
                        line_start=5, line_end=5, section="", kind="functional", block_ids=[])]
    calls = []

    def answer(prompt):
        calls.append(1)
        rid = "R-001" if len(calls) == 1 else "R-002"  # первый ответ пропускает R-002, переспрос его оценивает
        return {"assessments": [{"rid": rid, "marks": [{"criterion": "verifiable", "status": "ok"}], "findings": []}]}

    rv = Reviewer(FakeLLM({"reviewer": answer}), doc, Grounder(doc))
    assessments, _ = rv.run(reqs, {})
    assert set(assessments) == {"R-001", "R-002"} and len(calls) == 2 and not rv.skipped


def test_critic_keeps_verdicts_of_batches_that_answered(tmp_path):
    from reqcheck.agents import CRITIC_BATCH
    doc = _doc(tmp_path, "# ТЗ\n\nСистема должна сохранять порог.\n")
    fs = [Finding(fid=f"T-{i:03d}", scope="set", criterion="set_consistent", severity="major", title=f"Находка {i}",
                  explanation="", source="llm", verified=True) for i in range(1, CRITIC_BATCH + 6)]
    calls = []

    def answer(prompt):
        calls.append(1)
        if len(calls) > 1:
            raise LLMError("503")
        return {"verdicts": [{"fid": "T-001", "verdict": "reject", "reason": "нет дефекта"}]}

    critic = Critic(FakeLLM({"critic": answer}), doc)
    kept, dropped = critic.run(fs)
    assert [f.fid for f in dropped] == ["T-001"] and len(kept) == CRITIC_BATCH + 4 and critic.errors


def test_extractor_rejects_llm_id_that_is_a_reference(tmp_path):
    """ID из середины текста или из чужой ячейки — ссылка на другой пункт, а не номер блока (иначе ложный дубль ID)."""
    from reqcheck.agents import Extractor
    doc = _doc(tmp_path, "# ТЗ\n\n**ФТ-3.** Система должна отправлять SMS.\n\n"
                         "Проверка выполнения ФТ-3 проводится на тестовом стенде банка.\n\n"
                         "Требование ФТ-4: система должна хранить историю переводов.\n\n"
                         "| ID | Требование |\n|---|---|\n| ФТ-5 | Система должна вести журнал, см. ФТ-3 |\n")
    blocks = segment(doc)
    ref = next(b.bid for b in blocks if b.text.startswith("Проверка"))
    own = next(b.bid for b in blocks if b.text.startswith("Требование ФТ-4"))
    row = next(b.bid for b in blocks if b.kind == "table_row")
    items = [{"block_ids": [ref], "source_id": "ФТ-3"}, {"block_ids": [own], "source_id": "ФТ-4"},
             {"block_ids": [row], "source_id": "ФТ-5"}, {"block_ids": [row], "source_id": "ФТ-3"}]
    ex = Extractor(FakeLLM(), doc, blocks)
    got = [ex._to_requirement(i) for i in items]
    assert [(r.source_id, r.source_id_assigned) for r in got] == [("", True), ("ФТ-4", False), ("ФТ-5", False), ("", True)]


def test_own_id_accepts_labels_and_cells_rejects_references_and_prefixes():
    from reqcheck.agents import _own_id
    from reqcheck.models import Block

    def para(text):
        return Block(bid="B-001", kind="paragraph", text=text, line_start=1, line_end=1, section="ТЗ")

    assert _own_id("ФТ-6", para("Требование ФТ-6 Система должна хранить журнал."))
    assert _own_id("ФТ-2", para("ФТ-2: сумма перевода не более 1 млн."))
    assert not _own_id("ФТ-2", para("Согласно ФТ-2: сумма перевода не более 1 млн."))  # ссылка
    assert not _own_id("ФТ-4", para("ФТ-41 Система должна хранить журнал."))  # другой номер
    assert not _own_id("4.2", para("4.2.1 Система должна хранить журнал."))
    assert not _own_id("ФТ-3", para("Проверка выполнения ФТ-3 проводится на стенде."))
    row = Block(bid="B-002", kind="table_row", text="ID: ФТ-7 (обяз.)", line_start=2, line_end=2, section="ТЗ",
                cells=["ФТ-7 (обязательное)", "Система должна вести журнал, см. ФТ-3"])
    assert _own_id("ФТ-7", row) and not _own_id("ФТ-3", row)


def test_critic_duplicate_keeps_quotes_and_fake_downgrade_is_not_counted(tmp_path):
    doc = _doc(tmp_path, "# ТЗ\n\nЛимит 30 000 тенге.\n\nЛимит 50 000 тенге.\n")
    f1 = Finding(fid="T-001", scope="set", criterion="set_consistent", severity="critical", title="A", explanation="",
                 rids=["R-001"], quotes=[Quote("Лимит 30 000 тенге", 3, True)], source="llm", verified=True)
    f2 = Finding(fid="T-002", scope="set", criterion="set_consistent", severity="critical", title="B", explanation="",
                 rids=["R-002"], quotes=[Quote("Лимит 50 000 тенге", 5, True)], source="llm", verified=True)
    f3 = Finding(fid="T-003", scope="set", criterion="set_complete", severity="major", title="C", explanation="",
                 rids=["R-001"], quotes=[Quote("Лимит 30 000 тенге", 3, True)], source="llm", verified=True)
    llm = FakeLLM({"critic": {"verdicts": [{"fid": "T-002", "verdict": "duplicate", "duplicate_of": "T-001"},
                                           {"fid": "T-003", "verdict": "downgrade", "severity": "critical"}]}})
    kept, _ = Critic(llm, doc).run([f1, f2, f3])
    assert [q.line for q in kept[0].quotes] == [3, 5] and kept[0].rids == ["R-001", "R-002"]
    assert kept[1].severity == "major" and not kept[1].critic.startswith("понижено")


def test_reviewer_ignores_assessment_of_requirement_from_another_batch(tmp_path, monkeypatch):
    from reqcheck import agents
    from reqcheck.models import Requirement
    monkeypatch.setattr(agents, "REVIEW_BATCH", 1)
    doc = _doc(tmp_path, "# ТЗ\n\nСистема должна отправлять SMS.\n\nСистема должна хранить журнал.\n")
    reqs = [Requirement(rid=f"R-00{i}", source_id="", source_id_assigned=True, summary="", quote="", line_start=l,
                        line_end=l, section="ТЗ") for i, l in ((1, 3), (2, 5))]

    class PerBatch(FakeLLM):  # пакет R-002 ошибочно оценивает ещё и R-001 — нарушением
        def generate_json(self, role, system, prompt):
            ok = lambda rid: {"rid": rid, "marks": [{"criterion": "verifiable", "status": "ok"}]}  # noqa: E731
            if "R-002 (" in prompt:
                return {"assessments": [ok("R-002"), {"rid": "R-001", "marks": [{"criterion": "verifiable", "status": "violation"}],
                                                      "findings": [{"title": "Чужая", "quotes": ["отправлять SMS"]}]}]}
            return {"assessments": [ok("R-001")]}

    assessments, findings = agents.Reviewer(PerBatch(), doc, Grounder(doc)).run(reqs, {})
    mark = next(m for m in assessments["R-001"].marks if m.criterion == "verifiable")
    assert mark.status == "ok" and findings == []


def test_critic_reject_without_reason_keeps_finding(tmp_path):
    # ADR 0004: отклонение без записанной причины не принимается
    doc = _doc(tmp_path, "# ТЗ\n\nЛимит 30 000 тенге.\n")
    f = Finding(fid="T-001", scope="set", criterion="set_consistent", severity="critical", title="A", explanation="",
                rids=["R-001"], source="llm", verified=True)
    kept, dropped = Critic(FakeLLM({"critic": {"verdicts": [{"fid": "T-001", "verdict": "reject"}]}}), doc).run([f])
    assert [x.fid for x in kept] == ["T-001"] and not dropped and "без причины" in kept[0].critic


def test_reviewer_reports_requirements_with_missing_marks(tmp_path):
    from reqcheck import criteria
    from reqcheck.agents import Reviewer
    doc = _doc(tmp_path, "# ТЗ\n\nФТ-1. Система должна сохранять порог.\n\nФТ-2. Система должна читать порог.\n")
    blocks = segment(doc)
    from reqcheck.models import Requirement
    reqs = [Requirement(rid=f"R-00{i}", source_id="", source_id_assigned=True, summary="", quote=b.text,
                        line_start=b.line_start, line_end=b.line_end, section="ТЗ", block_ids=[b.bid])
            for i, b in enumerate([b for b in blocks if b.kind == "paragraph"], start=1)]
    full = [{"criterion": c, "status": "ok"} for c in criteria.REQ_CODES]
    llm = FakeLLM({"reviewer": {"assessments": [{"rid": "R-001", "marks": full},
                                                {"rid": "R-002", "marks": [{"criterion": "verifiable", "status": "ok"}]}]}})
    rv = Reviewer(llm, doc, Grounder(doc))
    rv.run(reqs, {b.bid: b for b in blocks})
    assert rv.partial == ["R-002"]
