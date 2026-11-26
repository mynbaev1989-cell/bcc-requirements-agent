"""Тесты отчётов: report_md и write_outputs."""


def test_report_is_never_written_over_the_input_document(tmp_path):
    from reqcheck.pipeline import Options, run, write_outputs
    src = tmp_path / "tz.md"
    text = "# ТЗ\n\n**Ф-1.** Система должна отправлять SMS клиенту в течение 30 секунд.\n"
    src.write_text(text, encoding="utf-8")
    result = run(str(src), Options(offline=True))
    written, problems = write_outputs(result, str(tmp_path / "tz.pdf"), ("md", "pdf"))
    assert src.read_text(encoding="utf-8") == text  # «tz.md tz.pdf» не затирает документ
    assert [p.code for p in problems] == ["WRITE_SKIPPED"] and str(tmp_path / "tz.pdf") in written
    blocker = tmp_path / "file.txt"
    blocker.write_text("x", encoding="utf-8")
    written, problems = write_outputs(result, str(blocker / "r.md"), ("md",))  # папка отчёта внутри файла
    assert written == [] and problems[0].code == "WRITE_FAILED"


def test_report_prints_sensitivity_table(sample_md):
    from reqcheck.pipeline import Options, run
    from reqcheck.report_md import render_markdown
    from tests.test_pipeline import _fake

    md = render_markdown(run(str(sample_md), Options(), llm=_fake(sample_md)))
    assert "Чувствительность к допущениям" in md
    for name in ("| основной |", "| осторожный | ×3 | 0,7 |", "| только время ревью | — | — | 0 ч |"):
        assert name in md


def test_llm_role_table_follows_plan_order_not_call_order():
    # Регрессия: порядок ролей в «нагрузке на LLM» зависел от того, какой параллельный шаг первым спросил модель
    from reqcheck.llm import FakeLLM
    from reqcheck.models import RunResult
    from reqcheck.pipeline import _llm_metrics

    llm = FakeLLM({})
    for role in ("critic", "analyst_tests", "reviewer", "extractor", "custom"):
        llm.stats.add(role, calls=1)
    result = RunResult(document=None)
    _llm_metrics(llm, result)
    assert list(result.metrics["llm_by_role"]) == ["extractor", "reviewer", "analyst_tests", "critic", "custom"]
