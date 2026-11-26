"""Отчёт в Markdown. Разделы 1–6 — строго в порядке раздела 6 ТЗ; дополнения — приложениями A–E."""

from __future__ import annotations

from collections import Counter, defaultdict

from reqcheck import criteria
from reqcheck.clarify import PRIORITY_ICON, PRIORITY_RU
from reqcheck.models import CRITICAL, NA, OK, SEVERITY_RU, VIOLATION, Finding, Quote, RunResult

LIGHT = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
SEV_ICON = {"critical": "🔴", "major": "🟠", "minor": "🟡"}
KIND_RU = {"functional": "функциональное", "nonfunctional": "нефункциональное", "data": "данные",
           "integration": "интеграция", "ui": "UI", "api": "API", "acceptance": "критерий приёмки", "other": "прочее"}

REQUIRED_SECTIONS = [
    "## 1. Паспорт прогона",
    "## 2. Как агент нарезал требования",
    "## 3. Проверка каждого требования",
    "## 4. Проверка всего набора требований",
    "## 5. Сводка",
    "## 6. Итог",
]


def cell(text: object) -> str:
    s = "" if text is None else str(text)
    return s.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").strip()


def _where(result: RunResult, line: int | None) -> str:
    if line is None:
        return "место не определено"
    doc = result.document
    if doc and doc.line_pages and 0 < line <= len(doc.line_pages):
        return f"стр. {doc.line_pages[line - 1]}, строка {line}"
    return f"строка {line}"


def _quote(result: RunResult, q: Quote) -> str:
    return f"«{q.text.strip()}» ({_where(result, q.line)})"


def _finding_md(result: RunResult, f: Finding, rid_map: dict) -> list[str]:
    head = (f"- **{f.fid}** · {SEV_ICON.get(f.severity, '')} {SEVERITY_RU.get(f.severity, f.severity)} · "
            f"{criteria.name(f.criterion)} ({criteria.source(f.criterion)}) — **{f.title}**")
    out = [head]
    if f.scope != "requirement" and f.rids:
        refs = ", ".join(f"{r} ({rid_map[r].source_id})" if r in rid_map and rid_map[r].source_id else r for r in f.rids)
        out.append(f"  - Требования: {refs}")
    if f.quotes:
        out.append("  - Где: " + "; ".join(_quote(result, q) for q in f.quotes))
    if f.explanation:
        out.append(f"  - Почему это дефект: {f.explanation}")
    if f.example:
        out.append(f"  - Пример: {f.example}")
    if f.suggestion:
        out.append(f"  - Как исправить: {f.suggestion}")
    if f.question:
        out.append(f"  - Вопрос автору{f' ({f.addressee})' if f.addressee else ''}: {f.question}")
    if f.resolution:
        out.append(f"  - Ответ автора: {f.resolution}")
    src = "правило агента" if f.source == "rule" else "LLM"
    out.append(f"  - Источник: {src}" + (f"; {f.critic}" if f.critic else ""))
    return out


def render_markdown(result: RunResult) -> str:
    doc = result.document
    p = result.passport
    reqs = result.requirements
    rid_map = {r.rid: r for r in reqs}
    findings = result.findings
    L: list[str] = []
    a = L.append

    a(f"# Отчёт о проверке требований: {p.get('document_title', '')}")
    a("")
    a("> Отчёт сформирован автоматически агентом reqcheck по входному документу. Ручных правок нет. "
      "Каждое нарушение ссылается на место в документе (номер строки; для PDF — страница и строка); "
      "цитаты сверены с исходным текстом кодом агента.")
    a("")

    # 1. Паспорт ------------------------------------------------------------------------------------
    a(REQUIRED_SECTIONS[0])
    a("")
    a("| Параметр | Значение |")
    a("| :--- | :--- |")
    hits, calls = result.metrics.get("llm_cache_hits", 0), result.metrics.get("llm_calls", 0)
    rows = [
        ("Название документа", p.get("document_title")),
        ("Файл", p.get("file")),
        ("Формат / размер", f"{p.get('format', '—')} / {p.get('size_bytes', '—')} байт, строк: {p.get('lines', '—')}"
         + (f", страниц: {p['pages']}" if p.get("pages") else "")),
        ("SHA-256 входного файла", p.get("sha256", "—")),
        ("Дата и время прогона", p.get("run_at")),
        ("Модель", p.get("model")),
        ("Версия агента", p.get("agent_version")),
        ("Режим", p.get("mode")),
        ("Нарезка требований", p.get("segmentation", "—")),
        ("Длительность прогона", f"{p.get('duration_sec', '—')} с" + (
            f" (ответов модели из кэша: {hits} из {hits + calls}; время и токены — только новых запросов)" if hits else "")),
        ("Вызовы LLM", f"{calls} (из кэша: {hits}, "
                       f"повторов: {result.metrics.get('llm_retries', 0)})"),
        ("Токены (вход / выход)", f"{result.metrics.get('input_tokens', 0)} / {result.metrics.get('output_tokens', 0)}"),
        ("Критерии", p.get("standards")),
    ]
    if p.get("answers_file"):
        rows.append(("Ответы автора", f"{p['answers_file']}: учтено ответов {result.metrics.get('answers_used', 0)}, "
                                      f"закрыто находок {result.metrics.get('resolved_by_answers', 0)}"))
    for k, v in rows:
        a(f"| {cell(k)} | {cell(v)} |")
    a("")

    # 2. Нарезка ------------------------------------------------------------------------------------
    a(REQUIRED_SECTIONS[1])
    a("")
    if reqs:
        assigned = sum(1 for r in reqs if r.source_id_assigned)
        a(f"Выделено требований: **{len(reqs)}**. Идентификатор в отчёте присваивает агент (R-001…). "
          f"Если в документе у требования нет номера ({assigned} шт.), в колонке «ID в документе» стоит «нет», "
          "а место фиксируется строкой и дословной цитатой.")
        a("")
        a("| ID в отчёте | ID в документе | Строки | Раздел | Вид | Краткая формулировка | Цитата |")
        a("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for r in reqs:
            lines = f"{r.line_start}" if r.line_start == r.line_end else f"{r.line_start}–{r.line_end}"
            if doc and doc.line_pages and 0 < r.line_start <= len(doc.line_pages):
                lines = f"стр. {doc.line_pages[r.line_start - 1]}, {lines}"
            sec = r.section.split(" › ")[-1]
            quote = r.quote if len(r.quote) <= 220 else r.quote[:217] + "…"
            a(f"| {r.rid} | {cell(r.source_id or 'нет')} | {lines} | {cell(sec)} | {KIND_RU.get(r.kind, r.kind)} | "
              f"{cell(r.summary)} | «{cell(quote)}» |")
    else:
        a("Требования не выделены — см. диагностику в Приложении E.")
    a("")

    # 3. Проверка каждого требования ----------------------------------------------------------------
    a(REQUIRED_SECTIONS[2])
    a("")
    a("Оценка по 10 критериям ISO/IEC/IEEE 29148 п. 5.2.5 и IEEE 830 п. 4.3: ✅ соблюдено, ❌ нарушено, ➖ не применимо / не оценено. "
      "Балл 0–100 считается кодом: −35 за критичный, −20 за существенный, −8 за незначительный дефект.")
    a("")
    by_rid: dict[str, list[Finding]] = defaultdict(list)
    grouped_by_rid: dict[str, list[Finding]] = defaultdict(list)  # сводные находки: у требования — только ссылка
    for f in findings:
        if f.scope == "requirement" or f.grouped:
            for rid in f.rids:
                (grouped_by_rid if f.grouped else by_rid)[rid].append(f)
    for r in reqs:
        asm = result.assessments.get(r.rid)
        score = f"{LIGHT.get(asm.light, '')} {asm.score}/100" if asm else ""
        a(f"### {r.rid} · {r.source_id or 'без номера'} · {score}")
        a("")
        a(f"> {' '.join(r.quote.split())} ({_where(result, r.line_start)})")
        a("")
        if asm:
            ok = [criteria.name(m.criterion) for m in asm.marks if m.status == OK]
            bad = [m for m in asm.marks if m.status == VIOLATION]
            na = [criteria.name(m.criterion) for m in asm.marks if m.status == NA]
            if bad:
                a("❌ **Нарушено:** " + "; ".join(f"{criteria.name(m.criterion)}" + (f" — {m.comment}" if m.comment else "") for m in bad))
                a("")
            if ok:
                a("✅ Соблюдено: " + ", ".join(ok))
                a("")
            if na:
                a("➖ Не применимо / не оценено: " + ", ".join(na))
                a("")
        fs = by_rid.get(r.rid, [])
        if fs:
            a("**Дефекты:**")
            a("")
            for f in fs:
                L.extend(_finding_md(result, f, rid_map))
            a("")
        elif not grouped_by_rid.get(r.rid):
            a("Дефектов не найдено.")
            a("")
        for f in grouped_by_rid.get(r.rid, []):
            a(f"См. сводную находку **{f.fid}** в дефектах набора: {f.title}.")
            a("")

    # 4. Проверка набора ----------------------------------------------------------------------------
    a(REQUIRED_SECTIONS[3])
    a("")
    a("Оценка набора по ISO/IEC/IEEE 29148 п. 5.2.6 и IEEE 830 п. 4.3:")
    a("")
    a("| Критерий | Источник | Оценка | Комментарий |")
    a("| :--- | :--- | :--- | :--- |")
    status_ru = {OK: "✅ соблюдено", VIOLATION: "❌ нарушено", NA: "➖ не оценено"}
    for m in result.set_marks:
        a(f"| {criteria.name(m.criterion)} | {criteria.source(m.criterion)} | {status_ru.get(m.status, m.status)} | {cell(m.comment)} |")
    a("")
    set_f = [f for f in findings if f.scope != "requirement"]
    a(f"### Дефекты набора ({len(set_f)})")
    a("")
    if set_f:
        for f in set_f:
            L.extend(_finding_md(result, f, rid_map))
    else:
        a("Дефектов уровня набора не найдено.")
    a("")

    # 5. Сводка -------------------------------------------------------------------------------------
    a(REQUIRED_SECTIONS[4])
    a("")
    sev = Counter(f.severity for f in findings)
    kinds = Counter(KIND_RU.get(r.kind, r.kind) for r in reqs)
    a(f"- Требований: **{len(reqs)}**" + (f" ({', '.join(f'{k}: {v}' for k, v in kinds.most_common())})" if kinds else ""))
    viol_reqs = sum(1 for r in reqs if by_rid.get(r.rid))
    a(f"- Нарушений (находок): **{len(findings)}** — 🔴 критичных: {sev.get('critical', 0)}, "
      f"🟠 существенных: {sev.get('major', 0)}, 🟡 незначительных: {sev.get('minor', 0)}")
    a(f"- Требований с нарушениями: {viol_reqs} из {len(reqs)}")
    if result.document_score is not None:
        light = "🟢" if result.document_score >= 80 else ("🟡" if result.document_score >= 50 else "🔴")
        a(f"- Готовность документа: {light} **{result.document_score}/100**")
    else:
        a("- Готовность документа: не рассчитана (смысловая проверка LLM не выполнялась)")
    a("")
    by_crit = Counter(f.criterion for f in findings)
    if by_crit:
        a("| Критерий | Нарушений |")
        a("| :--- | ---: |")
        for code, cnt in by_crit.most_common():
            a(f"| {criteria.name(code)} | {cnt} |")
        a("")
    a("### Список критичных дефектов")
    a("")
    crit = [f for f in findings if f.severity == CRITICAL]
    if crit:
        groups: dict[str, list[Finding]] = defaultdict(list)
        for f in crit:
            groups[criteria.CRITICAL_GROUP_BY_CRITERION.get(f.criterion, "прочие")].append(f)
        for g in ("непроверяемые", "противоречивые", "неполные", "прочие"):
            if groups.get(g):
                a(f"**{g.capitalize()}:**")
                a("")
                for f in groups[g]:
                    where = _where(result, f.quotes[0].line) if f.quotes else ""
                    refs = ", ".join(f.rids)
                    a(f"- {f.fid} ({refs + '; ' if refs else ''}{where}) — {f.title}")
                a("")
    else:
        a("Критичных дефектов не найдено.")
        a("")

    # 6. Итог ---------------------------------------------------------------------------------------
    a(REQUIRED_SECTIONS[5])
    a("")
    a(f"- **Можно ли начинать разработку:** {result.verdict_dev}")
    a(f"- **Можно ли начинать тестирование:** {result.verdict_test}")
    if result.questions:
        open_q = [q for q in result.questions if not q.answer]
        blockers = [q.qid for q in open_q if q.priority == "blocker"]
        a(f"- **Вопросы автору:** {len(result.questions)}, без ответа {len(open_q)}"
          + (f"; блокируют разработку: {', '.join(blockers)}" if blockers else "") + " (Приложение A).")
    if result.resolved:
        before = result.metrics.get("before_answers") or {}
        delta = ""
        if before:
            after = {sev: sum(1 for f in findings if f.severity == sev) for sev in ("critical", "major")}
            delta = (f"; критичных было {before.get('critical', 0)} → стало {after['critical']}, "
                     f"существенных {before.get('major', 0)} → {after['major']}")
        a(f"- **Учтены ответы автора:** закрыто находок {len(result.resolved)}{delta}. "
          "Закрытые находки не входят в вердикт (Приложение A).")
    a("")
    if result.priorities:
        a("**Что исправить в первую очередь:**")
        a("")
        fmap = {f.fid: f for f in findings}
        for i, fid in enumerate(result.priorities, start=1):
            f = fmap[fid]
            where = _where(result, f.quotes[0].line) if f.quotes else ""
            fix = f" Исправление: {f.suggestion}" if f.suggestion else ""
            a(f"{i}. **{f.fid}** — {f.title} ({where}).{fix}")
        a("")

    # Приложения -----------------------------------------------------------------------------------
    a("---")
    a("")
    a("## Приложение A. Уточняющие вопросы автору требований")
    a("")
    qs = result.questions
    if qs:
        cnt = Counter(q.priority for q in qs)
        covered = {fid for q in qs for fid in q.fids}
        a(f"Агент-уточнитель свёл находки в **{len(qs)}** вопросов: {PRIORITY_ICON['blocker']} блокируют разработку — "
          f"{cnt.get('blocker', 0)}, {PRIORITY_ICON['important']} нужны до тест-дизайна — {cnt.get('important', 0)}, "
          f"{PRIORITY_ICON['clarify']} можно уточнить по ходу — {cnt.get('clarify', 0)}. Ответы закрывают {len(covered)} находок. "
          "Один вопрос — одно решение автора; у каждого указаны варианты ответа и допущение на случай, если ответа не будет.")
        a("")
        for q in qs:
            refs = ", ".join(q.fids)
            if q.lines:
                refs += " (" + ", ".join(_where(result, ln) for ln in q.lines[:4]) + ")"
            a(f"**{q.qid}** · {PRIORITY_ICON.get(q.priority, '')} {PRIORITY_RU.get(q.priority, q.priority)} · Кому: {q.addressee}")
            a("")
            a(f"- Вопрос: {q.text}")
            if q.why:
                a(f"- Почему важно: {q.why}")
            if q.options:
                a("- Варианты ответа: " + "; ".join(f"({i}) {o}" for i, o in enumerate(q.options, start=1)))
            if q.assumption:
                a(f"- Если ответа не будет: {q.assumption}")
            a(f"- Находки: {refs}")
            if q.answer:
                a(f"- **Ответ автора:** {q.answer}" + (f" → {q.outcome}" if q.outcome else ""))
            a("")
        report_name = p.get("report_questions_file", "<отчёт>_questions.md")
        a(f"Чтобы ответить, заполните строки «Ответ:» в файле `{report_name}` (лежит рядом с отчётом) и запустите агента "
          "повторно с `--answers`: он решит, какие дефекты ответы закрывают, и пересчитает вердикт.")
        a("")
        if result.resolved:
            a("### Находки, закрытые ответами автора")
            a("")
            a("| Находка | Что было | Решение агента | Новая формулировка |")
            a("| :--- | :--- | :--- | :--- |")
            for f in result.resolved:
                a(f"| {f.fid} | {cell(f.title)} | {cell(f.resolution)} | {cell(f.suggestion or '—')} |")
            a("")
        open_q = [q for q in qs if not q.answer]
        if open_q:
            a("**Черновик письма автору:**")
            a("")
            a("```text")
            a(f"Тема: Вопросы по документу «{p.get('document_title', '')}» перед началом разработки")
            a("")
            a("Добрый день! По результатам ревью требований нужны уточнения:")
            by_addr: dict[str, list] = defaultdict(list)
            for q in open_q:
                by_addr[q.addressee].append(q)
            for addr, items in by_addr.items():
                a("")
                a(f"{addr}:")
                for q in items:
                    opts = (" Варианты: " + "; ".join(q.options) + ".") if q.options else ""
                    a(f"- {q.qid}. {q.text}{opts}")
            a("")
            a("Без ответов на вопросы уровня «блокирует разработку» работы по затронутым требованиям не начинаем; "
              "по остальным примем указанные в отчёте допущения.")
            a("```")
    else:
        a("Вопросов нет.")
    a("")

    a("## Приложение B. Чек-лист ревью из документа (заполнен агентом)")
    a("")
    if result.checklist:
        a("Агент нашёл в документе таблицу-чек-лист с незаполненными вариантами ответа и заполнил её по тексту документа. "
          "Это черновик для ревьюера, а не подпись согласующего.")
        a("")
        a("| Пункт чек-листа (строка) | Ответ агента | Обоснование | Доказательство |")
        a("| :--- | :--- | :--- | :--- |")
        for c in result.checklist:
            ev = "; ".join(_quote(result, q) for q in c.quotes) or "—"
            a(f"| {cell(c.question)} (строка {c.line}) | {cell(c.answer)} | {cell(c.justification or '—')} | {cell(ev)} |")
    else:
        a("Чек-листа ревью с вариантами «Да/Нет» в документе не найдено.")
    a("")

    a("## Приложение C. Черновики недостающих критериев приёмки (Дано / Когда / Тогда)")
    a("")
    if result.tests:
        for t in result.tests:
            ref = f"{t.rid} ({rid_map[t.rid].source_id})" if t.rid in rid_map and rid_map[t.rid].source_id else (t.rid or "—")
            a(f"**{t.title}** — к {ref}")
            a("")
            a(f"- Дано: {t.given}")
            a(f"- Когда: {t.when}")
            a(f"- Тогда: {t.then}")
            a("")
    else:
        a("Не сформированы (нужна LLM).")
        a("")

    a("## Приложение D. Матрица трассируемости")
    a("")
    if result.trace:
        a("| Требование | Критерии приёмки | API / интеграция | Вывод |")
        a("| :--- | :--- | :--- | :--- |")
        for t in result.trace:
            r = rid_map.get(t.rid)
            name = f"{t.rid} ({r.source_id})" if r and r.source_id else t.rid
            a(f"| {name} | {', '.join(t.acceptance) or '❌ нет'} | {', '.join(t.api) or '—'} | {cell(t.note)} |")
        no_acc = sum(1 for t in result.trace if not t.acceptance)
        a("")
        a(f"Требований без критерия приёмки: **{no_acc}** из {len(result.trace)}.")
    else:
        a("Не построена (нужна LLM).")
    a("")

    a("## Приложение E. Диагностика и метрики прогона")
    a("")
    if result.diagnostics:
        a("| Уровень | Код | Сообщение | Что сделать |")
        a("| :--- | :--- | :--- | :--- |")
        lvl = {"error": "❌ ошибка", "warning": "⚠️ предупреждение", "info": "ℹ️ инфо"}
        for d in result.diagnostics:
            a(f"| {lvl.get(d.level, d.level)} | {d.code} | {cell(d.message)} | {cell(d.hint)} |")
    else:
        a("Проблем при чтении и обработке документа нет.")
    a("")
    m = result.metrics
    a("**Контроль качества находок:**")
    a("")
    a(f"- Отброшено находок LLM без подтверждённой цитаты в документе: {m.get('dropped_ungrounded', 0)}")
    a(f"- Отклонено агентом-критиком как ошибочные: {m.get('dropped_by_critic', 0)}; объединено дублей: "
      f"{m.get('merged_duplicates', 0)}; понижена серьёзность: {m.get('downgraded_by_critic', 0)}")
    if m.get("merged_id_findings"):
        a(f"- Замечаний «нет идентификатора» по отдельным пунктам сведено в одну находку: {m['merged_id_findings']}")
    for t in m.get("dropped_by_critic_titles", [])[:10]:
        a(f"  - {t}")
    if m.get("merged_overlap"):
        a(f"- Повторов одного дефекта от разных агентов (по требованию и по набору) слито кодом: {m['merged_overlap']}")
        for t in m.get("merged_overlap_titles", [])[:10]:
            a(f"  - {t}")
    if m.get("merged_singular"):
        a(f"- Незначительных замечаний «Единичность» по отдельным пунктам сведено в одну находку: {m['merged_singular']}")
    a("")
    journal = m.get("journal") or []
    if journal:
        a("**Журнал агентов (оркестратор):** каждый шаг — отдельный агент с одной задачей; LLM-шаги проверяются кодом "
          "(цитаты, ID, приоритеты), при сбое LLM включается детерминированный запасной вариант. "
          "Стадия вида «5.2» — шаги одной стадии выполнялись параллельно.")
        a("")
        a("| Стадия | Шаг | Агент | Время, с | Статус | Результат |")
        a("| ---: | :--- | :--- | ---: | :--- | :--- |")
        for i, j in enumerate(journal, start=1):
            a(f"| {j.get('stage', i)} | {cell(j['step'])} | {cell(j['agent'])} | {j['seconds']} | {cell(j['status'])} "
              f"| {cell(j['out'])} |")
        a("")
    by_role = m.get("llm_by_role") or {}
    if by_role:
        a("**Нагрузка на LLM по ролям:**")
        a("")
        a("| Роль | Вызовы | Из кэша | Сбои | Токены вход | Токены выход | Размышления | Время, с |")
        a("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for role, r in by_role.items():
            a(f"| {role} | {r.get('calls', 0)} | {r.get('cache_hits', 0)} | {r.get('failures', 0)} | {r.get('input_tokens', 0)} | "
              f"{r.get('output_tokens', 0)} | {r.get('thinking_tokens', 0)} | {r.get('seconds', 0)} |")
        a("")
    v = m.get("value")
    if v:
        sp = lambda x: f"{x:,.0f}".replace(",", " ")  # noqa: E731
        a("**Бизнес-ценность прогона (оценка):**")
        a("")
        a("| Показатель | Значение |")
        a("| :--- | ---: |")
        a(f"| Ручное ревью (допущение: {v['manual_base_min']:.0f} мин + {v['manual_min_per_requirement']:.0f} мин × требование) | {v['manual_hours']} ч |")
        a(f"| Работа агента (замер) | {v['agent_hours'] * 60:.1f} мин |")
        a(f"| Проверка отчёта человеком ({v['check_min_per_finding']} мин × находка) | {v['check_hours']} ч |")
        a(f"| Экономия времени ревью на документ | **{v['saved_hours']} ч ≈ {sp(v['saved_kzt'])} ₸** при ставке {sp(v['rate_kzt_per_hour'])} ₸/ч |")
        if v.get("avoided_rework_hours") is not None:
            prec = f"{v.get('precision', 1):g}".replace(".", ",")
            a(f"| Предотвращённые переделки: (критичных {v['critical_found']} + {v.get('major_found', 0)} существенных × 0,5) × "
              f"точность находок {prec} × {v.get('fix_hours_at_requirements', 1):g} ч × ({v.get('escalation_factor', 7):g} − 1) | "
              f"**{v.get('avoided_rework_hours', 0)} ч ≈ {sp(v.get('avoided_rework_kzt', 0))} ₸** |")
        a(f"| Стоимость LLM на прогон ({v.get('tokens_total', 0)} токенов; {v.get('llm_price_note', '')}) | {v.get('llm_cost_kzt', 0)} ₸ |")
        if v.get("roi") is not None:
            a(f"| Отдача: выгода / (LLM + время проверки отчёта) | **×{v['roi']}** |")
        a("")
        a("Переделки: по данным NASA (Stecklein et al., 2004/2010), дефект требований, найденный на этапе кодирования, "
          "обходится в 7–16 раз дороже, на интеграции и тестах — в 21–78 раз; агент берёт нижнюю границу (×7) "
          "и умножает на точность находок, измеренную на размеченных документах: не каждая находка — настоящий дефект. "
          "Все параметры — допущения, их можно задать переменными окружения (см. README).")
        a("")
    return "\n".join(L) + "\n"
