"""Оркестратор: параллельные стадии применяются в порядке плана, сбой LLM → диагностика + запасной вариант."""

import time

from reqcheck.llm import LLMError
from reqcheck.models import Finding, Quote, RunResult
from reqcheck.orchestrator import Options, Orchestrator, RunContext, Step
from reqcheck.steps import build_plan


class _Write(Step):
    """Пишет свою метку в общий список; задержка в run() меняет порядок завершения потоков."""

    def __init__(self, label: str, delay: float = 0.0, uses_llm: bool = False):
        self.name = self.agent = label
        self.delay, self.uses_llm = delay, uses_llm

    def run(self, ctx):
        time.sleep(self.delay)
        return self.name

    def apply(self, ctx, out):
        ctx.metrics.setdefault("order", []).append(out)


class _Broken(_Write):
    def run(self, ctx):
        raise LLMError("квота исчерпана")

    def fallback(self, ctx):
        return "запасной"


def _ctx(llm=object()):
    return RunContext(input_path="x.md", opts=Options(), result=RunResult(document=None), llm=llm)


def test_parallel_stage_applies_in_plan_order():
    ctx = _ctx()
    Orchestrator([_Write("A"), [_Write("B", 0.2), _Write("C", 0.0)], _Write("D")]).run(ctx)
    assert ctx.metrics["order"] == ["A", "B", "C", "D"]  # C закончил раньше B, но применён после
    assert [j["stage"] for j in ctx.metrics["journal"]] == ["1", "2.1", "2.2", "3"]


def test_auxiliary_step_failure_does_not_mark_check_incomplete():
    step = _Broken("Вопросы автору", uses_llm=True)
    step.finds_defects = False
    ctx = _ctx()
    Orchestrator([step]).run(ctx)
    assert [d.code for d in ctx.result.diagnostics] == ["LLM_AUX_FAILED"]  # не входит в INCOMPLETE_CODES


def test_llm_error_gives_diagnostic_and_fallback():
    ctx = _ctx()
    Orchestrator([_Broken("Шаг X", uses_llm=True)]).run(ctx)
    assert ctx.metrics["order"] == ["запасной"]
    assert [d.code for d in ctx.result.diagnostics] == ["LLM_STEP_FAILED"]
    assert ctx.metrics["journal"][0]["status"] == "сбой → запасной вариант"


def test_no_llm_uses_fallback_and_skips_llm_only_steps():
    ctx = _ctx(llm=None)
    Orchestrator([_Broken("С запасным", uses_llm=True), _Write("Только LLM", uses_llm=True), _Write("Код")]).run(ctx)
    assert ctx.metrics["order"] == ["запасной", "Код"]
    assert not ctx.result.diagnostics
    assert [j["status"] for j in ctx.metrics["journal"]] == ["без LLM → запасной вариант", "ok"]


def test_stop_ends_run():
    class _Stop(_Write):
        def apply(self, ctx, out):
            ctx.stop = True

    ctx = _ctx()
    Orchestrator([_Stop("стоп"), _Write("после")]).run(ctx)
    assert "order" not in ctx.metrics


def test_plan_review_parallel_set_check_after():
    plan = build_plan()
    i = next(k for k, s in enumerate(plan) if isinstance(s, list))
    assert {s.agent for s in plan[i]} == {"Reviewer, параллельные пакеты (LLM)", "Analyst (LLM)"}
    # проверка набора — отдельной стадией сразу после: меньше запросов в первую минуту на бесплатном тарифе
    assert plan[i + 1].agent == "ConsistencyReviewer + GapFinder (LLM)"


def test_stage_numbers_have_no_gaps():
    class _Off(_Write):
        def enabled(self, ctx):
            return False

    ctx = _ctx(llm=None)
    Orchestrator([_Write("A"), _Off("выкл"), _Write("Только LLM", uses_llm=True), [_Write("B"), _Write("C")]]).run(ctx)
    assert [j["stage"] for j in ctx.metrics["journal"]] == ["1", "2.1", "2.2"]


def test_stats_are_thread_safe():
    from concurrent.futures import ThreadPoolExecutor

    from reqcheck.llm import LLMStats

    st = LLMStats()
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: [st.add("reviewer", calls=1, input_tokens=2) for _ in range(500)], range(8)))
    assert st.calls == 4000 and st.input_tokens == 8000
    assert st.by_role["reviewer"]["calls"] == 4000


def test_merge_step_keeps_one_finding_with_both_quotes(sample_md):
    from reqcheck.orchestrator import Options, RunContext
    from reqcheck.models import RunResult
    from reqcheck.steps import MergeOverlapStep
    a = Finding(fid="T-1", scope="requirement", criterion="complete", severity="major", title="Отсутствует URL для среды Prod",
                explanation="", rids=["R-020"], quotes=[Quote("Prod: —", 108, True)], source="llm")
    b = Finding(fid="T-2", scope="set", criterion="set_complete", severity="critical", title="Отсутствие Prod URL для API",
                explanation="", rids=["R-020"], quotes=[Quote("Request URL", 108, True)], source="llm")
    ctx = RunContext(input_path=str(sample_md), opts=Options(), result=RunResult(document=None))
    ctx.findings = [a, b]
    step = MergeOverlapStep()
    step.apply(ctx, step.run(ctx))
    assert ctx.findings == [a] and a.severity == "critical" and len(a.quotes) == 2
    assert ctx.metrics["merged_overlap"] == 1


def test_merged_finding_marks_criterion_violated_at_added_requirement(sample_md):
    from reqcheck.orchestrator import Options, RunContext
    from reqcheck.models import RunResult, RequirementAssessment, CriterionMark
    from reqcheck.steps import MergeOverlapStep, NumberStep
    main = Finding(fid="T-1", scope="requirement", criterion="correct", severity="major", title="403 или 404 для чужой карты",
                   explanation="", rids=["R-043"], quotes=[Quote("403", 175, True)], source="llm")
    dup = Finding(fid="T-2", scope="set", criterion="set_consistent", severity="major", title="Противоречие 403 и 404 карты",
                  explanation="", rids=["R-043", "R-045"], quotes=[Quote("403", 175, True), Quote("404", 189, True)], source="llm")
    ctx = RunContext(input_path=str(sample_md), opts=Options(), result=RunResult(document=None))
    ctx.result.assessments = {"R-043": RequirementAssessment("R-043", [CriterionMark("correct", "violation")]),
                              "R-045": RequirementAssessment("R-045", [CriterionMark("correct", "ok")])}
    ctx.findings = [main, dup]
    for step in (MergeOverlapStep(), NumberStep()):  # оценки сверяются с окончательным составом находок
        step.apply(ctx, step.run(ctx))
    assert ctx.findings == [main] and main.rids == ["R-043", "R-045"]
    assert ctx.result.assessments["R-045"].marks[0].status == "violation"


def test_minor_singular_findings_collapse_into_one_but_keep_marks_and_scores(sample_md):
    from reqcheck.orchestrator import Options, RunContext
    from reqcheck.models import RunResult, RequirementAssessment, CriterionMark, Requirement
    from reqcheck.pipeline import _finalize
    from reqcheck.steps import MergeOverlapStep, NumberStep

    def single(fid, rid, line, sev="minor"):
        return Finding(fid=fid, scope="requirement", criterion="singular", severity=sev, title=f"Два действия в {rid}",
                       explanation="", rids=[rid], quotes=[Quote(f"текст {rid}", line, True)], suggestion=f"Разделить {rid}",
                       source="llm")
    a, b, c = single("T-1", "R-001", 10), single("T-2", "R-002", 20), single("T-3", "R-003", 30, sev="major")
    b.question = "Разделить ли R-002?"
    other = Finding(fid="T-4", scope="requirement", criterion="correct", severity="major", title="Противоречие",
                    explanation="", rids=["R-001"], quotes=[Quote("x", 10, True)], source="llm")
    ctx = RunContext(input_path=str(sample_md), opts=Options(), result=RunResult(document=None))
    ctx.reqs = [Requirement(rid=f"R-00{i}", source_id="", source_id_assigned=True, summary="", quote="", line_start=i * 10,
                            line_end=i * 10, section="ТЗ") for i in (1, 2, 3)]
    ctx.result.assessments = {r.rid: RequirementAssessment(r.rid, [CriterionMark("singular", "ok")]) for r in ctx.reqs}
    ctx.findings = [a, b, c, other]
    merge = MergeOverlapStep()
    for step in (merge, NumberStep()):
        step.apply(ctx, step.run(ctx))
    assert merge.summary([]).endswith("сведено в одну: 2")  # журнал оркестратора показывает сведение
    merged = [f for f in ctx.findings if f.scope == "set"]
    assert len(ctx.findings) == 3 and c in ctx.findings and other in ctx.findings  # существенная не сводится
    (m,) = merged
    assert m.criterion == "singular" and m.rids == ["R-001", "R-002"] and [q.line for q in m.quotes] == [10, 20]
    assert m.grouped and "R-002: Разделить R-002" in m.suggestion and ctx.metrics["merged_singular"] == 2
    assert "R-001, R-002" in m.question  # вопрос автору не теряется: один общий вместо мелких
    assert ctx.result.assessments["R-002"].marks[0].status == "violation"
    # Балл требования тот же, что с отдельной находкой: −8 за незначительный дефект
    ctx.result.requirements = ctx.reqs
    _finalize(ctx.result)
    assert ctx.result.assessments["R-002"].score == 92
    from reqcheck.report_md import render_markdown
    ctx.result.findings = ctx.findings
    assert f"См. сводную находку **{m.fid}**" in render_markdown(ctx.result)
