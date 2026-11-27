"""Экспорт вопросов автору в CSV для импорта в Jira (System → External System Import → CSV).

Одна строка — один вопрос: задача на того, кто принимает решение. Колонки сопоставляются в мастере импорта Jira:
Summary, Issue Type, Priority, Labels, Description. Исполнителя Jira ждёт логином, поэтому роль адресата
(«Архитектор», «Бизнес-заказчик») идёт в метку и в описание, а не в поле Assignee.
Кодировка UTF-8 с BOM: файл открывается и в Excel без «кракозябр».
"""

from __future__ import annotations

import csv
import io

from reqcheck.models import RunResult

# Приоритет вопроса → приоритет Jira (стандартная схема Highest…Lowest)
JIRA_PRIORITY = {"blocker": "Highest", "important": "High", "clarify": "Medium"}
SUMMARY_MAX = 250  # Jira ограничивает Summary 255 символами


def _label(text: str) -> str:
    """Метка Jira не может содержать пробелы."""
    return "_".join((text or "Аналитик").split())


def _cell(value: str) -> str:
    """Защита от формульной инъекции: ячейку, которая начинается с = + - @ (или табуляции/CR), Excel выполнит
    как формулу. Текст приходит из документа и ответа модели, поэтому такой ячейке ставится ведущий апостроф."""
    return "'" + value if value[:1] in ("=", "+", "-", "@", "\t", "\r") else value


def render_jira_csv(result: RunResult) -> str:
    """CSV с вопросами автору; пустая строка, если вопросов нет."""
    if not result.questions:
        return ""
    fmap = {f.fid: f for f in result.findings}
    title = result.passport.get("document_title", "") or "документ требований"
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["Summary", "Issue Type", "Priority", "Labels", "Labels", "Description"])
    for q in result.questions:
        text = " ".join((q.text or "").split())
        summary = f"[{q.qid}] {text}"
        if len(summary) > SUMMARY_MAX:
            summary = summary[:SUMMARY_MAX - 1] + "…"
        lines = [f"Документ: {title}", f"Кому: {q.addressee or 'Аналитик'}", "", f"Вопрос: {text}"]
        if q.why:
            lines.append(f"Почему важно: {q.why}")
        if q.options:
            lines.append("Варианты: " + "; ".join(f"({i}) {o}" for i, o in enumerate(q.options, 1)))
        if q.assumption:
            lines.append(f"Если ответа не будет: {q.assumption}")
        for fid in q.fids:
            f = fmap.get(fid)
            if f is None:
                continue
            where = ", ".join(str(x.line) for x in f.quotes if x.line)
            lines.append(f"{fid} ({f.severity}): {f.title}" + (f" — строки {where}" if where else ""))
            for x in f.quotes[:2]:
                lines.append(f"  «{x.text}»")
        w.writerow([_cell(c) for c in (summary, "Task", JIRA_PRIORITY.get(q.priority, "Medium"), "reqcheck",
                                        _label(q.addressee), "\n".join(lines))])
    return "﻿" + buf.getvalue()
