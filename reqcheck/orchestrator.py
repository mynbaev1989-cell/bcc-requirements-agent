"""Движок мультиагентного прогона: шаг (Step), общий контекст (RunContext) и оркестратор (Orchestrator).

Каждый шаг — один агент с одной задачей (агент-LLM или детерминированный код). План прогона — список стадий;
шаги одной стадии не зависят друг от друга и выполняются параллельно. Оркестратор:
- выполняет `Step.run` (в потоке, только чтение контекста) и затем `Step.apply` строго в порядке плана —
  отчёт детерминирован, даже если агенты отвечают в разном порядке;
- если LLM нет или шаг упал с LLMError, пишет диагностику и включает запасной вариант шага (`Step.fallback`, код);
- ведёт журнал (стадия, агент, время, статус, итог) — он попадает в Приложение E отчёта.
Ошибки, не связанные с LLM, не глушатся: их ловит `pipeline.run` и превращает в диагностику INTERNAL_ERROR.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from reqcheck.llm import JsonLLM, LLMError
from reqcheck.loaders import DEFAULT_MAX_BYTES
from reqcheck.models import Block, ChecklistItem, Diagnostic, Document, Finding, Requirement, RunResult


@dataclass
class Options:
    model: str | None = None
    offline: bool = False
    formats: tuple[str, ...] = ("md", "pdf", "json")
    max_bytes: int = DEFAULT_MAX_BYTES
    timeout: float = 240.0
    use_cache: bool = True
    cache_dir: str = ".reqcheck_cache"
    critic: bool = True
    answers: str | None = None  # файл с ответами автора на вопросы агента (цикл уточнения)


@dataclass
class RunContext:
    """Общая «доска» прогона: шаги читают её в run() и пишут в неё только в apply()."""
    input_path: str
    opts: Options
    result: RunResult
    llm: JsonLLM | None = None
    doc: Document | None = None
    blocks: list[Block] = field(default_factory=list)
    grounder: Any = None
    rules: Any = None
    checklist: list[ChecklistItem] = field(default_factory=list)
    set_rule_findings: list[Finding] = field(default_factory=list)
    reqs: list[Requirement] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    answer_pairs: list = field(default_factory=list)  # (Question | None, ответ) из файла --answers
    stop: bool = False  # дальше идти некуда (например, документ не прочитан)

    @property
    def metrics(self) -> dict[str, Any]:
        return self.result.metrics

    def diag(self, level: str, code: str, message: str, hint: str = "") -> None:
        self.result.diagnostics.append(Diagnostic(level, code, message, hint))


class Step:
    """Один агент прогона. Подкласс задаёт run/apply и, если нужно, запасной вариант без LLM."""
    name = ""  # шаг в журнале и в диагностике
    agent = ""  # кто выполняет: «Reviewer (LLM)», «RuleEngine (код)»
    fallback_agent = ""  # кто выполняет запасной вариант
    roles: tuple[str, ...] = ()  # роли LLM (для таблицы нагрузки по ролям)
    uses_llm = False
    # Ищет ли шаг дефекты. Сбой такого шага делает смысловую проверку неполной (вердикт не «Да»); сбой
    # вспомогательного (вопросы автору, тесты, чек-лист, критик, учёт ответов) на полноту поиска не влияет
    finds_defects = True
    failure_hint = "Отчёт дополнен детерминированными проверками; повторите запуск позже."

    def enabled(self, ctx: RunContext) -> bool:
        return True

    def run(self, ctx: RunContext) -> Any:  # может выполняться в потоке: контекст только читаем
        raise NotImplementedError

    def apply(self, ctx: RunContext, out: Any) -> None:  # всегда в основном потоке, в порядке плана
        pass

    def fallback(self, ctx: RunContext) -> Any:  # результат без LLM; None — шаг просто пропускается
        return None

    def summary(self, out: Any) -> str:
        return ""

    def status(self, out: Any) -> str:  # статус успешно выполненного шага для журнала
        return "ok"


class Orchestrator:
    def __init__(self, plan: list[Step | list[Step]]):
        self.plan = plan

    def run(self, ctx: RunContext) -> None:
        journal = ctx.metrics.setdefault("journal", [])
        n = 0  # номер стадии в журнале: пропущенные стадии не занимают номер
        for stage in self.plan:
            if ctx.stop:
                break
            steps = [s for s in (stage if isinstance(stage, list) else [stage]) if s.enabled(ctx)]
            if not steps:
                continue
            if len(steps) == 1:
                outcomes = [self._execute(steps[0], ctx)]
            else:
                with ThreadPoolExecutor(max_workers=len(steps)) as pool:
                    outcomes = list(pool.map(lambda s: self._execute(s, ctx), steps))
            entries = []
            for step, (status, out, seconds, error) in zip(steps, outcomes):
                agent = step.agent
                if status == "failed":
                    code = "LLM_STEP_FAILED" if step.finds_defects else "LLM_AUX_FAILED"
                    ctx.diag("warning", code, f"Шаг «{step.name}» не выполнен: {error}", step.failure_hint)
                if status in ("failed", "no_llm"):
                    out = step.fallback(ctx)
                    if out is None and status == "no_llm":
                        continue  # шагу без LLM нечего делать — в журнал не пишем
                    agent = step.fallback_agent or f"{step.agent} → запасной вариант"
                if out is not None:
                    step.apply(ctx, out)
                entries.append({
                    "step": step.name, "agent": agent, "seconds": seconds,
                    "status": {"ok": step.status(out), "failed": "сбой → запасной вариант",
                               "no_llm": "без LLM → запасной вариант"}[status],
                    "out": self._summary(step, out) if out is not None else (error or "")[:120],
                    "roles": list(step.roles),
                })
            if entries:
                n += 1
                for k, e in enumerate(entries, start=1):
                    journal.append({"stage": f"{n}" if len(entries) == 1 else f"{n}.{k}", **e})

    @staticmethod
    def _execute(step: Step, ctx: RunContext) -> tuple[str, Any, float, str]:
        if step.uses_llm and ctx.llm is None:
            return "no_llm", None, 0.0, ""
        t0 = time.perf_counter()
        try:
            out = step.run(ctx)
            return "ok", out, round(time.perf_counter() - t0, 2), ""
        except LLMError as exc:
            return "failed", None, round(time.perf_counter() - t0, 2), str(exc)

    @staticmethod
    def _summary(step: Step, out: Any) -> str:
        try:
            return step.summary(out)
        except Exception:  # noqa: BLE001 — сводка для журнала не должна ронять прогон
            return ""
