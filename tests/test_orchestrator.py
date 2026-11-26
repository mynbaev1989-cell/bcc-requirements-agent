"""Оркестратор: параллельные стадии применяются в порядке плана, сбой LLM → диагностика + запасной вариант."""

import time

from reqcheck.llm import LLMError
from reqcheck.models import RunResult
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
