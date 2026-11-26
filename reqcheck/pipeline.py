"""Точка входа прогона: план агентов (steps.py) → оркестратор (orchestrator.py) → вердикт и метрики → отчёт.

Главный принцип: прогон всегда заканчивается отчётом. Любая ошибка превращается в диагностику
в отчёте (Приложение E), а не в traceback.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

from reqcheck import __version__, criteria
from reqcheck.agents import score_requirement, sync_marks
from reqcheck.llm import JsonLLM
from reqcheck.models import CRITICAL, MAJOR, NA, SEV_RANK, CriterionMark, Diagnostic, Finding, RequirementAssessment, RunResult, per_requirement
from reqcheck.orchestrator import Options, Orchestrator, RunContext
from reqcheck.steps import build_plan
from reqcheck.value import estimate

__all__ = ["Options", "agent_version", "run", "write_outputs"]


# Диагностики, после которых смысловая проверка считается неполной (вердикт не может быть «Да»)
INCOMPLETE_CODES = ("LLM_UNAVAILABLE", "OFFLINE", "LLM_STEP_FAILED", "REVIEW_INCOMPLETE", "DOC_TRUNCATED")


def agent_version() -> str:
    """Версия агента: __version__ + git describe (тег/коммит), если доступно."""
    env = os.environ.get("REQCHECK_VERSION")
    if env:
        return env
    here = Path(__file__).resolve().parent
    try:
        out = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], cwd=here, capture_output=True,
                             text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return f"{__version__} ({out.stdout.strip()})"
    except (OSError, subprocess.SubprocessError):
        pass
    return f"{__version__} (git недоступен)"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run(input_path: str, options: Options | None = None, llm: JsonLLM | None = None) -> RunResult:
    opts = options or Options()
    t0 = time.perf_counter()
    result = RunResult(document=None)
    result.passport = {
        "document_title": Path(input_path).name,
        "file": str(input_path),
        "input_path": str(input_path),
        "run_at": _now_iso(),
        "agent_version": agent_version(),
        "model": "—",
        "mode": "—",
        "standards": "ISO/IEC/IEEE 29148:2018 (п. 5.2.5, 5.2.6); IEEE Std 830-1998 (п. 4.3)",
    }
    m = result.metrics
    m.update({"dropped_ungrounded": 0, "dropped_by_critic": 0, "downgraded_by_critic": 0, "llm_calls": 0})
    try:
        _run_inner(input_path, opts, llm, result)
    except Exception as exc:  # noqa: BLE001 — последний рубеж: отчёт с диагностикой вместо падения
        result.diagnostics.append(Diagnostic(
            "error", "INTERNAL_ERROR", f"Внутренняя ошибка агента: {type(exc).__name__}: {exc}",
            "Сообщите разработчикам; трассировка сохранена в JSON-отчёте.",
        ))
        m["traceback"] = traceback.format_exc(limit=8)
    m["duration_sec"] = round(time.perf_counter() - t0, 2)
    _finalize(result)
    return result


def _run_inner(input_path: str, opts: Options, llm: JsonLLM | None, result: RunResult) -> None:
    ctx = RunContext(input_path=input_path, opts=opts, result=result, llm=llm)
    Orchestrator(build_plan()).run(ctx)
    if ctx.llm:
        _llm_metrics(ctx.llm, result)


# Роли в порядке плана. Словарь статистики заполняется в порядке первых запросов, а параллельные шаги
# обращаются к модели в произвольном порядке: без сортировки таблица нагрузки в отчёте менялась бы от прогона к прогону
ROLE_ORDER = ("extractor", "reviewer", "analyst_checklist", "analyst_tests", "consistency", "gaps", "critic",
              "clarifier", "resolver")


def _role_rank(role: str) -> tuple[int, str]:
    return (ROLE_ORDER.index(role) if role in ROLE_ORDER else len(ROLE_ORDER), role)


def _llm_metrics(llm: JsonLLM, result: RunResult) -> None:
    """Нагрузка на LLM: вызовы, кэш, токены, роли; смена модели из-за квоты — в паспорт и диагностику."""
    st = llm.stats
    result.metrics.update({
        "llm_calls": st.calls, "llm_cache_hits": st.cache_hits, "llm_retries": st.retries,
        "llm_failures": st.failures, "input_tokens": st.input_tokens, "output_tokens": st.output_tokens,
        "thinking_tokens": st.thinking_tokens, "llm_seconds": round(st.seconds, 1),
        "llm_by_role": {k: {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                        for k, v in sorted(st.by_role.items(), key=lambda kv: _role_rank(kv[0]))}})
    p = result.passport
    if len(st.models_used) > 1:
        p["model"] = " + ".join(st.models_used) + " (переключение из-за квоты/доступности)"
    elif st.model_version and st.model_version != llm.model:
        p["model"] = f"{llm.model} → {st.model_version}"
    for e in st.errors:
        if "переключаюсь" in e:
            result.diagnostics.append(Diagnostic("info", "LLM_FALLBACK", e, "Модель сменилась во время прогона."))


def _finalize(result: RunResult) -> None:
    """Оценки, вердикт, приоритеты, экономия — считаются кодом, не моделью."""
    reqs = result.requirements
    findings = result.findings
    by_rid: dict[str, list[Finding]] = {}
    for f in filter(per_requirement, findings):  # сводная находка штрафует каждое своё требование
        for rid in f.rids:
            by_rid.setdefault(rid, []).append(f)
    # Оценивал ли требования Reviewer — до того, как ниже появятся отметки по находкам правил
    llm_assessed = any(m.status != NA for a in result.assessments.values() for m in a.marks)
    for r in reqs:
        a = result.assessments.get(r.rid)
        if a is None:
            a = RequirementAssessment(r.rid, [CriterionMark(c, NA, "не оценено (LLM недоступна)") for c in criteria.REQ_CODES])
            result.assessments[r.rid] = a
            # Reviewer не ответил, но находка правил у требования есть — её критерий нарушен, а не «не оценён»
            sync_marks(by_rid.get(r.rid, []), {r.rid: a})
        a.score, a.light = score_requirement(a, by_rid.get(r.rid, []))
    if not result.set_marks:
        result.set_marks = [CriterionMark(c, NA, "не оценено (LLM недоступна)") for c in criteria.SET_CODES]

    n_crit = sum(1 for f in findings if f.severity == CRITICAL)
    n_major = sum(1 for f in findings if f.severity == MAJOR)
    if reqs and llm_assessed:
        mean = sum(result.assessments[r.rid].score for r in reqs) / len(reqs)
        set_pen = sum({"critical": 8, "major": 4, "minor": 1}.get(f.severity, 2) for f in findings
                      if not per_requirement(f))
        result.document_score = max(0, round(mean - set_pen))

    load_failed = result.document is None or not reqs
    if load_failed:
        result.verdict_dev = "Нет — документ не проверен (см. диагностику в Приложении E)."
        result.verdict_test = "Нет — документ не проверен."
    else:
        # Смысловая проверка неполная: LLM не было, модель не оценила требования или упал смысловой шаг.
        # Тогда «Да» не даём: отсутствие дефектов у правил не значит, что их нет.
        incomplete = not llm_assessed or any(d.code in INCOMPLETE_CODES for d in result.diagnostics)
        if n_crit:
            result.verdict_dev = f"Нет — найдено критичных дефектов: {n_crit}. Начинать разработку до их устранения нельзя."
        elif n_major:
            result.verdict_dev = f"Условно — критичных дефектов нет, но есть существенные ({n_major}); разработку можно начинать после их уточнения."
        elif incomplete:
            result.verdict_dev = ("Условно — критичных и существенных дефектов не найдено, но смысловая проверка "
                                  "выполнена не полностью; повторите запуск с доступной LLM.")
        else:
            result.verdict_dev = "Да — критичных и существенных дефектов не найдено."
        unverifiable = [f for f in findings if f.criterion in ("verifiable", "set_validatable") and f.severity in (CRITICAL, MAJOR)]
        contradictions = [f for f in findings if f.criterion in ("set_consistent", "correct") and f.severity == CRITICAL]
        if contradictions or any(f.severity == CRITICAL and f.criterion == "verifiable" for f in findings):
            result.verdict_test = ("Нет — есть противоречивые или непроверяемые требования: по ним нельзя однозначно "
                                   "сформулировать ожидаемый результат теста.")
        elif unverifiable or n_crit or n_major or incomplete:  # критичный дефект не может дать вердикт лучше существенного
            result.verdict_test = "Условно — тест-дизайн можно начинать, но часть ожидаемых результатов требует уточнения."
        else:
            result.verdict_test = "Да — требования проверяемы."
        if incomplete:
            result.verdict_dev += " (Оценка неполная: смысловая проверка LLM выполнена не полностью, см. Приложение E.)"

    ranked = sorted(findings, key=lambda f: (SEV_RANK.get(f.severity, 9),
                                             0 if f.scope != "requirement" else 1, -len(f.rids)))
    result.priorities = [f.fid for f in ranked[:7] if f.severity in (CRITICAL, MAJOR)]

    agent_sec = result.metrics.get("duration_sec", 0.0)
    mt = result.metrics
    # Находка, закрытая ответом автора, — тоже дефект, найденный до разработки: в ценности её учитываем
    found = findings + result.resolved
    result.metrics["value"] = estimate(len(reqs), len(found), sum(1 for f in found if f.severity == CRITICAL), agent_sec,
                                       n_major=sum(1 for f in found if f.severity == MAJOR),
                                       tokens_in=mt.get("input_tokens", 0),
                                       tokens_out=mt.get("output_tokens", 0) + mt.get("thinking_tokens", 0)).as_dict()
    result.metrics.update({"requirements": len(reqs), "findings": len(findings), "critical": n_crit, "major": n_major,
                           "minor": sum(1 for f in findings if f.severity == "minor")})
    result.passport["duration_sec"] = agent_sec
    open_q = [q for q in result.questions if not q.answer]
    result.metrics.update({"questions": len(result.questions),
                           "questions_blocker_open": sum(1 for q in open_q if q.priority == "blocker")})


def _resolved(path: str) -> Path | None:
    try:
        return Path(path).resolve() if path else None
    except (OSError, ValueError):
        return None


def write_outputs(result: RunResult, output_path: str, formats: tuple[str, ...]) -> tuple[list[str], list[Diagnostic]]:
    """Записать отчёт. Формат основного файла — по расширению; остальные — рядом с тем же именем."""
    from reqcheck.report_md import render_markdown
    from reqcheck.report_pdf import render_pdf

    out = Path(output_path)
    written: list[str] = []
    problems: list[Diagnostic] = []
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # путь отчёта ведёт внутрь файла или нет прав: код выхода 2, а не трассировка
        return [], [Diagnostic("error", "WRITE_FAILED", f"Не удалось создать папку для отчёта: {exc}",
                               "Укажите другой путь отчёта.")]
    source = _resolved(result.passport.get("input_path", ""))

    def overwrites_input(target: Path) -> bool:
        # «tz.md tz.pdf» положил бы Markdown-версию отчёта в tz.md — поверх проверяемого документа
        if source is None or _resolved(str(target)) != source:
            return False
        problems.append(Diagnostic("warning", "WRITE_SKIPPED", f"{target.name} не записан: это входной документ.",
                                   "Укажите для отчёта другое имя файла."))
        return True

    stem = out.with_suffix("")
    main_ext = out.suffix.lower().lstrip(".") or "md"
    wanted = list(dict.fromkeys([main_ext] + list(formats)))
    if result.questions:
        result.passport["report_questions_file"] = Path(f"{stem}_questions.md").name
    md_text = render_markdown(result)
    for fmt in wanted:
        target = out if fmt == main_ext and out.suffix else stem.with_suffix("." + fmt)
        if overwrites_input(target):
            continue
        try:
            if fmt in ("md", "markdown", "txt"):
                target.write_text(md_text, encoding="utf-8")
            elif fmt == "json":
                target.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            elif fmt == "pdf":
                render_pdf(md_text, str(target))
            else:
                problems.append(Diagnostic("warning", "UNKNOWN_FORMAT", f"Неизвестный формат отчёта «{fmt}» пропущен."))
                continue
            written.append(str(target))
        except Exception as exc:  # noqa: BLE001 — один формат не должен мешать остальным
            problems.append(Diagnostic("warning", "WRITE_FAILED", f"Не удалось записать {target.name}: {exc}"))
    if result.questions and not overwrites_input(Path(f"{stem}_questions.md")):
        # Шаблон для ответов автора: ответы уже учтённые в этом прогоне сохраняются в файле
        from reqcheck.clarify import render_questions_file
        target = Path(f"{stem}_questions.md")
        p = result.passport
        try:
            target.write_text(render_questions_file(result.questions, p.get("document_title", ""), p.get("input_path", ""),
                                                    str(out), str(target), p.get("sha256", "")), encoding="utf-8")
            written.append(str(target))
        except OSError as exc:
            problems.append(Diagnostic("warning", "WRITE_FAILED", f"Не удалось записать {target.name}: {exc}"))
    return written, problems
