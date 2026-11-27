"""Тесты интерактивного HTML-отчёта: содержимое, экранирование, запись рядом с другими форматами, формат ответов."""

import re

from reqcheck.clarify import parse_answers
from reqcheck.models import Finding, Question, Quote
from reqcheck.pipeline import Options, run, write_outputs
from reqcheck.report_html import render_html


def _result(sample_md):
    from test_pipeline import _fake
    result = run(str(sample_md), Options(), llm=_fake(sample_md))
    # Вопросы и находка с вредным текстом задаём явно: от фикстуры LLM их наличие не зависит
    result.questions = [Question(qid="Q-01", text="Какой срок SMS?", priority="blocker", addressee="Аналитик",
                                 why="Нельзя проверить", options=["30 секунд", "5 минут"], assumption="30 секунд",
                                 fids=["F-001"], lines=[3])]
    result.findings.append(Finding(fid="F-XSS", scope="set", criterion="complete", severity="minor",
                                   title="<script>alert(1)</script>", explanation="</script><img src=x onerror=alert(2)>",
                                   quotes=[Quote(text="<b>x</b>", line=1, found=True)]))
    return result


def test_html_has_all_ids(sample_md):
    result = _result(sample_md)
    out = render_html(result)
    assert out.lower().startswith("<!doctype html>")
    for f in result.findings:
        assert f.fid in out
    for q in result.questions:
        assert q.qid in out


def test_html_escapes_model_text(sample_md):
    out = render_html(_result(sample_md))
    assert "<script>alert" not in out and "<img src=x" not in out
    assert out.count("</script>") == 1  # только свой закрывающий тег: данные — литерал внутри кода


def test_html_has_no_external_resources(sample_md):
    out = render_html(_result(sample_md))
    assert not re.findall(r"https?://", out)


def test_write_outputs_writes_html(sample_md, tmp_path):
    result = _result(sample_md)
    written, problems = write_outputs(result, str(tmp_path / "out" / "r.md"), ("md", "html"))
    html_file = tmp_path / "out" / "r.html"
    assert str(html_file) in written and html_file.read_text(encoding="utf-8").lower().startswith("<!doctype html>")
    assert not [p for p in problems if p.code == "UNKNOWN_FORMAT"]


def test_answers_in_html_format_are_parsed():
    # Тот же формат, что собирает JS: «Вариант N: текст», затем свой текст ответа
    text = ("# Вопросы агента автору требований: ТЗ\n\n"
            "## Q-01 · блокирует разработку · Кому: Аналитик\n\nВопрос: Какой срок SMS?\n"
            "Варианты: (1) 30 секунд; (2) 5 минут\nНаходки: F-001\n"
            "Ответ: Вариант 1: 30 секунд\nНо не более минуты при сбоях\n\n"
            "## Q-02 · можно уточнить по ходу · Кому: Аналитик\n\nВопрос: Другое?\nНаходки: F-002\nОтвет:\n")
    got = parse_answers(text)
    assert [a["qid"] for a in got] == ["Q-01"]
    assert got[0]["answer"].startswith("Вариант 1: 30 секунд") and "не более минуты" in got[0]["answer"]


def test_findings_go_from_critical_to_minor(sample_md):
    result = _result(sample_md)
    result.findings.insert(0, Finding(fid="F-MIN", scope="set", criterion="complete", severity="minor", title="m",
                                      explanation=""))
    result.findings.append(Finding(fid="F-CRIT", scope="set", criterion="correct", severity="critical", title="c",
                                   explanation=""))
    out = render_html(result)
    assert out.index('"F-CRIT"') < out.index('"F-MIN"')


def test_placeholder_in_title_is_not_substituted(sample_md):
    result = _result(sample_md)
    result.passport["document_title"] = "ТЗ {{DATA}} {{TITLE}}"
    out = render_html(result)
    title = out[out.index("<title>"):out.index("</title>")]
    assert "{" not in title.replace("{{DATA}}", "").replace("{{TITLE}}", "") and len(title) < 300


def test_html_marks_requirements_in_set_conflict(sample_md):
    from reqcheck.report_html import _data
    result = _result(sample_md)
    conflict = next(f for f in result.findings if f.scope == "set" and f.severity == "critical" and len(f.rids) >= 2)
    reqs = {r["rid"]: r for r in _data(result)["requirements"]}
    assert all(conflict.fid in reqs[rid]["in_set"] for rid in conflict.rids)
