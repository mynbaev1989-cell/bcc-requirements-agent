"""Шаги прогона — по одному агенту на шаг — и план прогона.

План (стадии по порядку; шаги в квадратных скобках идут параллельно):
 1  Loader                      чтение документа (код)
 2  Segmenter + RuleEngine      нарезка на блоки и правила набора (код)
 3  Подключение модели          GeminiClient или режим без LLM (код)
 4  Extractor                   выделение требований (LLM; запасной вариант — эвристика по структуре)
 5  [Reviewer | Analyst: чек-лист | Analyst: тесты]   (LLM, параллельно)
 6  ConsistencyReviewer + GapFinder   противоречия и пропуски (LLM; внутри два запроса параллельно)
 7  Grounder + dedupe           отсев находок без подтверждённой цитаты, слияние дублей (код)
 8  Critic                      перепроверка находок (LLM)
 9  Нумерация находок           порядок отчёта, F-ID (код)
10  Clarifier                   вопросы автору (LLM; запасной вариант — вопросы из находок кодом)
11  Чтение ответов              файл --answers (код)
12  Resolver                    учёт ответов автора (LLM)

Проверка набора идёт отдельной стадией после Reviewer: так в первую минуту уходит меньше запросов
и бесплатный тариф (лимит запросов в минуту) реже отвечает 429.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from reqcheck.agents import (
    Analyst, ConsistencyReviewer, Critic, Extractor, Reviewer, dedupe, number_requirements, requirements_from_blocks,
)
from reqcheck.clarify import (
    Clarifier, Resolver, apply_effects, carry_over, match_answers, parse_answers, questions_from_findings,
)
from reqcheck.grounding import Grounder
from reqcheck.llm import GeminiClient, LLMUnavailable
from reqcheck.loaders import LoadError, load_document
from reqcheck.models import SEV_RANK
from reqcheck.orchestrator import RunContext, Step
from reqcheck.rules import RuleEngine
from reqcheck.segmenter import segment


# ---------- код ----------

class LoadStep(Step):
    name, agent = "Чтение документа", "Loader (код)"

    def run(self, ctx: RunContext) -> Any:
        try:
            return load_document(ctx.input_path, max_bytes=ctx.opts.max_bytes)
        except LoadError as err:
            return err

    def apply(self, ctx: RunContext, out: Any) -> None:
        if isinstance(out, LoadError):
            ctx.result.diagnostics.append(out.diagnostic)
            ctx.result.passport["mode"] = "проверка не выполнена: документ не прочитан"
            ctx.stop = True
            return
        doc, diags = out
        ctx.doc = ctx.result.document = doc
        ctx.result.diagnostics.extend(diags)
        ctx.result.passport.update({
            "document_title": doc.title, "file": doc.name, "format": doc.fmt.upper(), "sha256": doc.sha256,
            "size_bytes": doc.size_bytes, "lines": len(doc.lines), "pages": doc.pages})

    def status(self, out: Any) -> str:
        return "ошибка" if isinstance(out, LoadError) else "ok"

    def summary(self, out: Any) -> str:
        if isinstance(out, LoadError):
            return out.diagnostic.code
        doc = out[0]
        return f"{doc.fmt.upper()}, строк {len(doc.lines)}"


class SegmentRulesStep(Step):
    name, agent = "Нарезка и правила набора", "Segmenter + RuleEngine (код)"

    def run(self, ctx: RunContext) -> Any:
        blocks = segment(ctx.doc)
        rules = RuleEngine(ctx.doc, blocks)
        set_rule_findings, checklist = rules.run_set_rules()
        return blocks, rules, set_rule_findings, checklist

    def apply(self, ctx: RunContext, out: Any) -> None:
        ctx.blocks, ctx.rules, ctx.set_rule_findings, ctx.checklist = out
        ctx.result.blocks = ctx.blocks
        ctx.result.checklist = ctx.checklist
        ctx.grounder = Grounder(ctx.doc)

    def summary(self, out: Any) -> str:
        return f"блоков {len(out[0])}, находок правил {len(out[2])}"


class ConnectLLMStep(Step):
    name, agent = "Подключение модели", "GeminiClient (код)"

    def run(self, ctx: RunContext) -> Any:
        if ctx.llm is not None or ctx.opts.offline:  # модель передана снаружи (тесты) или режим без LLM
            return ctx.llm, None
        try:
            return GeminiClient(model=ctx.opts.model, timeout=ctx.opts.timeout,
                                cache_dir=ctx.opts.cache_dir if ctx.opts.use_cache else None), None
        except LLMUnavailable as exc:
            return None, exc

    def apply(self, ctx: RunContext, out: Any) -> None:
        ctx.llm, err = out
        if err is not None:
            ctx.diag("warning", "LLM_UNAVAILABLE", f"LLM недоступна: {err}",
                     "Отчёт построен только детерминированными правилами. Задайте GEMINI_API_KEY для полной проверки.")
        if ctx.opts.offline:
            ctx.diag("info", "OFFLINE", "Запуск с --offline: использованы только детерминированные правила.")
        p = ctx.result.passport
        p["model"] = ctx.llm.model if ctx.llm else "не использовалась"
        p["mode"] = "LLM + детерминированные правила" if ctx.llm else "только детерминированные правила (без LLM)"

    def summary(self, out: Any) -> str:
        return out[0].model if out[0] else "без LLM"


# ---------- LLM-агенты ----------

class ExtractStep(Step):
    name, agent, roles, uses_llm = "Выделение требований", "Extractor (LLM)", ("extractor",), True
    fallback_agent = "Эвристика по структуре документа (код)"

    def run(self, ctx: RunContext) -> Any:
        reqs = Extractor(ctx.llm, ctx.doc, ctx.blocks).run()
        if not reqs:  # модель не выделила ни одного требования — берём эвристику
            return "llm_empty", requirements_from_blocks(ctx.blocks)
        return "llm", reqs

    def fallback(self, ctx: RunContext) -> Any:
        return "heuristic", requirements_from_blocks(ctx.blocks)

    def apply(self, ctx: RunContext, out: Any) -> None:
        how, reqs = out
        if how == "llm_empty":
            ctx.diag("warning", "EXTRACTOR_EMPTY", "Модель не выделила ни одного требования; применена эвристика.")
        ctx.result.passport["segmentation"] = (
            "структура документа + LLM (Extractor), цитаты сверены с исходником" if how == "llm"
            else "эвристика по структуре документа")
        number_requirements(reqs)
        ctx.reqs = ctx.result.requirements = reqs
        if not reqs:
            ctx.diag("error", "NO_REQUIREMENTS", "В документе не найдено ни одного требования.",
                     "Проверьте, что передан документ с требованиями (ФТ, НФТ, критерии приёмки), а не другой текст.")
        # Правила по отдельным требованиям (код) — сразу после выделения
        rules = ctx.rules
        ctx.findings += rules.weak_words(reqs) + rules.duplicate_ids(reqs) + rules.missing_ids(reqs)
        ctx.findings += ctx.set_rule_findings

    def summary(self, out: Any) -> str:
        prefix = "модель вернула 0 → эвристика: " if out[0] == "llm_empty" else ""
        return f"{prefix}требований {len(out[1])}"


class _NeedsRequirements(Step):
    uses_llm = True

    def enabled(self, ctx: RunContext) -> bool:
        return bool(ctx.reqs)


class ReviewStep(_NeedsRequirements):
    name, agent, roles = "Оценка каждого требования", "Reviewer, параллельные пакеты (LLM)", ("reviewer",)

    def run(self, ctx: RunContext) -> Any:
        reviewer = Reviewer(ctx.llm, ctx.doc, ctx.grounder)
        assessments, req_findings = reviewer.run(ctx.reqs, {b.bid: b for b in ctx.blocks})
        return assessments, req_findings, list(reviewer.errors)

    def apply(self, ctx: RunContext, out: Any) -> None:
        assessments, req_findings, errors = out
        for err in errors:
            ctx.diag("warning", "LLM_STEP_FAILED", f"Часть требований не оценена моделью: {err}",
                     "Отмечены в разделе 3 как «не оценено»; повторите запуск позже.")
        ctx.result.assessments = assessments
        # «Нет идентификатора / не по шаблону» по каждому пункту сводит правило missing_ids — убираем шум
        noise = [f for f in req_findings if f.criterion in ("traceable", "conforming") and f.severity == "minor"]
        ctx.metrics["merged_id_findings"] = len(noise)
        noise_ids = {id(f) for f in noise}
        ctx.findings += [f for f in req_findings if id(f) not in noise_ids]

    def summary(self, out: Any) -> str:
        return f"оценено {len(out[0])}, находок {len(out[1])}"


class SetCheckStep(_NeedsRequirements):
    name, agent = "Проверка набора: противоречия и пропуски", "ConsistencyReviewer + GapFinder (LLM)"
    roles = ("consistency", "gaps")

    def __init__(self) -> None:
        self._cons: ConsistencyReviewer | None = None

    def run(self, ctx: RunContext) -> Any:
        self._cons = ConsistencyReviewer(ctx.llm, ctx.doc, ctx.grounder)
        marks, findings = self._cons.run(ctx.reqs)
        return marks, findings, self._cons.consistency_error, self._cons.gap_error

    def fallback(self, ctx: RunContext) -> Any:
        # Упали оба прохода: общая ошибка шага уже в диагностике, добавляем причину сбоя поиска пропусков
        if self._cons is not None and self._cons.gap_error:
            ctx.diag("warning", "LLM_STEP_FAILED", f"Шаг «поиск пропусков» не выполнен: {self._cons.gap_error}")
        return None  # оценки набора останутся «не оценено»

    def apply(self, ctx: RunContext, out: Any) -> None:
        marks, findings, consistency_error, gap_error = out
        ctx.result.set_marks = marks
        ctx.findings += findings
        if consistency_error:
            ctx.diag("warning", "LLM_STEP_FAILED", f"Шаг «проверка противоречий» не выполнен: {consistency_error}",
                     "Найденные пропуски в отчёте есть; оценки набора — «не оценено». Повторите запуск позже.")
        if gap_error:
            ctx.diag("warning", "LLM_STEP_FAILED", f"Шаг «поиск пропусков» не выполнен: {gap_error}")

    def summary(self, out: Any) -> str:
        return f"находок {len(out[1])}"


class AnalystChecklistStep(_NeedsRequirements):
    name, agent, roles = "Чек-лист и трассировка", "Analyst (LLM)", ("analyst_checklist",)

    def run(self, ctx: RunContext) -> Any:
        checklist = copy.deepcopy(ctx.checklist)  # агент заполняет ответы; общий контекст меняем только в apply()
        trace = Analyst(ctx.llm, ctx.doc, ctx.grounder).checklist_and_trace(ctx.reqs, checklist)
        return trace or [], checklist

    def apply(self, ctx: RunContext, out: Any) -> None:
        ctx.result.trace, ctx.checklist = out
        ctx.result.checklist = ctx.checklist

    def summary(self, out: Any) -> str:
        return f"строк трассировки {len(out[0])}"


class AnalystTestsStep(_NeedsRequirements):
    name, agent, roles = "Черновики тест-кейсов", "Analyst (LLM)", ("analyst_tests",)

    def run(self, ctx: RunContext) -> Any:
        return Analyst(ctx.llm, ctx.doc, ctx.grounder).tests(ctx.reqs)

    def apply(self, ctx: RunContext, out: Any) -> None:
        ctx.result.tests = out or []

    def summary(self, out: Any) -> str:
        return f"тест-кейсов {len(out)}"


class GroundStep(Step):
    """Анти-галлюцинации: находка LLM без подтверждённой цитаты из документа в отчёт не попадает."""
    name, agent = "Сверка цитат и дедупликация", "Grounder + dedupe (код)"

    def run(self, ctx: RunContext) -> Any:
        grounded = [f for f in ctx.findings if not (f.source == "llm" and not f.verified)]
        return grounded, len(ctx.findings) - len(grounded)

    def apply(self, ctx: RunContext, out: Any) -> None:
        grounded, dropped = out
        for f in grounded:
            f.quotes = [q for q in f.quotes if q.found] or f.quotes  # в отчёт — только найденные в документе цитаты
        ctx.findings = dedupe(grounded)
        self._merged = len(grounded) - len(ctx.findings)
        ctx.metrics["dropped_ungrounded"] += dropped
        for i, f in enumerate(ctx.findings, start=1):
            f.fid = f"T-{i:03d}"  # временные ID для Critic

    def summary(self, out: Any) -> str:
        return f"отброшено без цитаты {out[1]}, слито дублей {getattr(self, '_merged', 0)}"


class CriticStep(Step):
    name, agent, roles, uses_llm = "Перепроверка находок", "Critic (LLM)", ("critic",), True

    def enabled(self, ctx: RunContext) -> bool:
        return ctx.opts.critic and bool(ctx.findings)

    def run(self, ctx: RunContext) -> Any:
        return Critic(ctx.llm, ctx.doc).run(ctx.findings)

    def apply(self, ctx: RunContext, out: Any) -> None:
        ctx.findings, dropped = out
        m = ctx.metrics
        m["merged_duplicates"] = sum(1 for f in dropped if f.critic.startswith("дубль"))
        m["dropped_by_critic"] = len(dropped) - m["merged_duplicates"]
        m["dropped_by_critic_titles"] = [f"{f.title} — {f.critic}" for f in dropped]
        m["downgraded_by_critic"] = sum(1 for f in ctx.findings if f.critic.startswith("понижено"))

    def summary(self, out: Any) -> str:
        return f"оставлено {len(out[0])}, отклонено/слито {len(out[1])}"


class NumberStep(Step):
    """Порядок отчёта: сначала находки по требованиям (в порядке требований, затем по важности), потом по набору."""
    name, agent = "Нумерация находок", "Оркестратор (код)"

    def run(self, ctx: RunContext) -> Any:
        order = {r.rid: i for i, r in enumerate(ctx.reqs)}
        req_f = sorted([f for f in ctx.findings if f.scope == "requirement"],
                       key=lambda f: (order.get(f.rids[0], 10**6) if f.rids else 10**6, SEV_RANK.get(f.severity, 9)))
        set_f = sorted([f for f in ctx.findings if f.scope != "requirement"], key=lambda f: SEV_RANK.get(f.severity, 9))
        return req_f + set_f

    def apply(self, ctx: RunContext, out: Any) -> None:
        for i, f in enumerate(out, start=1):
            f.fid = f"F-{i:03d}"
        ctx.findings = ctx.result.findings = out

    def summary(self, out: Any) -> str:
        return f"находок {len(out)}"


# ---------- уточнение требований ----------

class ClarifyStep(Step):
    name, agent, roles, uses_llm = "Вопросы автору", "Clarifier (LLM)", ("clarifier",), True
    fallback_agent = "Clarifier (запасной вариант: код)"

    def run(self, ctx: RunContext) -> Any:
        if not ctx.findings:
            return []
        return Clarifier(ctx.llm, ctx.doc).run(ctx.findings)

    def fallback(self, ctx: RunContext) -> Any:
        return questions_from_findings(ctx.findings)

    def apply(self, ctx: RunContext, out: Any) -> None:
        ctx.result.questions = out

    def summary(self, out: Any) -> str:
        return f"вопросов {len(out)}"


class ReadAnswersStep(Step):
    name, agent = "Чтение ответов автора", "Loader (код)"

    def enabled(self, ctx: RunContext) -> bool:
        return bool(ctx.opts.answers)

    def run(self, ctx: RunContext) -> Any:
        try:
            return parse_answers(Path(ctx.opts.answers).read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError) as exc:
            return exc

    def apply(self, ctx: RunContext, out: Any) -> None:
        ctx.result.passport["answers_file"] = str(ctx.opts.answers)
        if isinstance(out, Exception):
            ctx.diag("warning", "ANSWERS_UNREADABLE", f"Файл ответов не прочитан: {out}",
                     "Проверьте путь и кодировку (UTF-8) файла, переданного в --answers.")
            return
        if not out:
            ctx.diag("info", "ANSWERS_EMPTY", "В файле ответов нет заполненных ответов.",
                     "Впишите ответы после «Ответ:» в файле *_questions.md.")
            return
        pairs = match_answers(out, ctx.result.questions)
        for q, a in pairs:
            if q:
                q.answer = a["answer"]
        ctx.answer_pairs = pairs
        ctx.metrics["answers_used"] = len(out)
        ctx.metrics["answers_unmatched"] = [a["qid"] for q, a in pairs if q is None]

    def status(self, out: Any) -> str:
        return "ошибка" if isinstance(out, Exception) else "ok"

    def summary(self, out: Any) -> str:
        return str(out)[:120] if isinstance(out, Exception) else f"ответов {len(out)}"


class ResolveAnswersStep(Step):
    name, agent, roles, uses_llm = "Учёт ответов автора", "Resolver (LLM)", ("resolver",), True
    fallback_agent = "Resolver → ответы не применены"

    def enabled(self, ctx: RunContext) -> bool:
        return bool(ctx.answer_pairs)

    def run(self, ctx: RunContext) -> Any:
        return Resolver(ctx.llm, ctx.doc).run(ctx.answer_pairs, ctx.findings)

    def fallback(self, ctx: RunContext) -> Any:
        if ctx.llm is None:  # без LLM ответы только сохраняем, решения не принимаем
            ctx.diag("warning", "ANSWERS_NOT_APPLIED", "Ответы автора прочитаны, но не учтены: нужна LLM.",
                     "Запустите без --offline с ключом GEMINI_API_KEY.")
            ctx.result.questions = ctx.result.questions + carry_over(self._unmatched(ctx), [])
            return None
        return []  # Resolver упал: ни одна находка не закрыта, ответы сохраняются

    def apply(self, ctx: RunContext, out: Any) -> None:
        resolved, followups = apply_effects(out, ctx.findings, ctx.result.questions)
        ctx.result.resolved = [f for f in ctx.findings if f.fid in resolved]
        ctx.findings = ctx.result.findings = [f for f in ctx.findings if f.fid not in resolved]
        ctx.result.questions = ctx.result.questions + followups + carry_over(self._unmatched(ctx), followups)
        ctx.metrics["resolved_by_answers"] = len(resolved)

    @staticmethod
    def _unmatched(ctx: RunContext) -> list[dict]:
        return [a for q, a in ctx.answer_pairs if q is None]

    def summary(self, out: Any) -> str:
        return f"решений по находкам {len(out)}"


def build_plan() -> list[Step | list[Step]]:
    """План прогона. Список внутри списка — независимые шаги, которые выполняются параллельно."""
    return [
        LoadStep(),
        SegmentRulesStep(),
        ConnectLLMStep(),
        ExtractStep(),
        [ReviewStep(), AnalystChecklistStep(), AnalystTestsStep()],
        SetCheckStep(),
        GroundStep(),
        CriticStep(),
        NumberStep(),
        ClarifyStep(),
        ReadAnswersStep(),
        ResolveAnswersStep(),
    ]
