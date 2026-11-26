"""Инварианты конвейера: вопросы к критичным дефектам, вердикт при неполной проверке, цитаты, баллы,
порядок разделов, независимость от порядка завершения агентов и от PYTHONHASHSEED."""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from reqcheck.agents import SEVERITY_PENALTY, score_requirement
from reqcheck.llm import FakeLLM
from reqcheck.models import CRITICAL, VIOLATION, per_requirement
from reqcheck.pipeline import INCOMPLETE_CODES, Options, run, write_outputs
from reqcheck.report_md import REQUIRED_SECTIONS
from test_pipeline import _fake, _with_clarify

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"


def _variant(sample_md, name):
    """Несколько вариантов подставной модели (имя варианта = параметр теста)."""
    if name == "plain":
        return _fake(sample_md)
    if name == "clarify":
        return _with_clarify(sample_md)
    if name == "no_critic":
        return _fake(sample_md, fail_roles={"critic"})
    if name == "no_reviewer":
        return _fake(sample_md, fail_roles={"reviewer"})
    if name == "no_consistency":
        return _fake(sample_md, fail_roles={"consistency"})
    raise KeyError(name)


VARIANTS = ["plain", "clarify", "no_critic", "no_reviewer", "no_consistency"]


@pytest.fixture(params=VARIANTS)
def result(request, sample_md):
    return run(str(sample_md), Options(), llm=_variant(sample_md, request.param))


# ---------- критичные дефекты ↔ вопросы Clarifier'а ----------

def test_every_critical_finding_has_a_question(result):
    covered = {fid for q in result.questions for fid in q.fids}
    for f in result.findings:
        if f.severity == CRITICAL:
            assert f.fid in covered, f"критичная находка {f.fid} «{f.title}» без вопроса автору"


def test_questions_reference_existing_findings_only(result):
    known = {f.fid for f in result.findings} | {f.fid for f in result.resolved}
    for q in result.questions:
        assert q.fids and set(q.fids) <= known


def test_question_linked_to_critical_is_blocker(result):
    crit = {f.fid for f in result.findings if f.severity == CRITICAL}
    for q in result.questions:
        if set(q.fids) & crit:
            assert q.priority == "blocker"


@pytest.mark.parametrize("offline", [True, False])
def test_critical_coverage_in_rule_mode(sample_md, offline):
    res = run(str(sample_md), Options(offline=offline))  # без ключа и с --offline: запасной вариант вопросов
    covered = {fid for q in res.questions for fid in q.fids}
    assert all(f.fid in covered for f in res.findings if f.severity == CRITICAL)


# ---------- вердикт и неполная проверка ----------

@pytest.mark.parametrize("fail_roles", [
    {"reviewer"}, {"consistency"}, {"reviewer", "consistency"}, {"critic"},
    {"reviewer", "consistency", "critic", "analyst_checklist", "analyst_tests", "gaps"},
])
def test_verdict_is_not_yes_when_check_incomplete(sample_md, fail_roles):
    llm = _fake(sample_md, fail_roles=fail_roles)
    # даже если модель «ничего не нашла», «Да» давать нельзя
    llm.responses["reviewer"] = {"assessments": []}
    llm.responses["consistency"] = {"set_marks": [], "findings": []}
    llm.responses["gaps"] = {"findings": []}
    res = run(str(sample_md), Options(), llm=llm)
    assert any(d.code in INCOMPLETE_CODES for d in res.diagnostics) or fail_roles == {"critic"}
    if any(d.code in INCOMPLETE_CODES for d in res.diagnostics):
        assert not res.verdict_dev.startswith("Да")
        assert not res.verdict_test.startswith("Да")


@pytest.mark.parametrize("kwargs", [{"offline": True}, {}])
def test_verdict_is_not_yes_without_llm(sample_md, kwargs):
    res = run(str(sample_md), Options(**kwargs))
    assert not res.verdict_dev.startswith("Да")
    assert not res.verdict_test.startswith("Да")
    assert "неполная" in res.verdict_dev or res.verdict_dev.startswith("Нет")


def test_verdict_consistent_with_finding_counts(result):
    n_crit = sum(1 for f in result.findings if f.severity == CRITICAL)
    n_major = sum(1 for f in result.findings if f.severity == "major")
    if n_crit:
        assert result.verdict_dev.startswith("Нет")
    elif n_major:
        assert result.verdict_dev.startswith("Условно")
    assert result.metrics["critical"] == n_crit and result.metrics["major"] == n_major


# ---------- цитаты (Grounder) ----------

def test_ungrounded_finding_never_reaches_report(sample_md, tmp_path, result):
    assert all(f.title != "Выдуманная находка" for f in result.findings)
    out = tmp_path / "r.md"
    write_outputs(result, str(out), ("md", "json"))
    assert "Выдуманная находка" not in out.read_text(encoding="utf-8")
    assert "Выдуманная находка" not in out.with_suffix(".json").read_text(encoding="utf-8")


def test_llm_findings_have_verified_quotes_present_in_source(sample_md, result):
    text = sample_md.read_text(encoding="utf-8")
    lines = text.split("\n")
    for f in result.findings:
        if f.source != "llm":
            continue
        assert f.verified and f.quotes
        for q in f.quotes:
            assert q.found and q.line
            norm = re.sub(r"\s+", " ", q.text).strip().lower()
            assert norm in re.sub(r"\s+", " ", " ".join(lines[q.line - 1:q.line + 2])).lower() or norm in re.sub(r"\s+", " ", text).lower()


def test_dropped_ungrounded_counter_matches(sample_md):
    res = run(str(sample_md), Options(), llm=_fake(sample_md))
    assert res.metrics["dropped_ungrounded"] == 1


# ---------- балл и отметки ----------

def test_scores_match_marks_and_findings(result):
    by_rid = {}
    for f in filter(per_requirement, result.findings):
        for rid in f.rids:
            by_rid.setdefault(rid, []).append(f)
    for r in result.requirements:
        a = result.assessments[r.rid]
        assert (a.score, a.light) == score_requirement(a, by_rid.get(r.rid, []))
        assert 0 <= a.score <= 100
        assert a.light == ("green" if a.score >= 80 else "yellow" if a.score >= 50 else "red")


def test_requirement_with_critical_finding_is_not_green(result):
    for f in result.findings:
        if per_requirement(f) and f.severity == CRITICAL:
            for rid in f.rids:
                assert result.assessments[rid].score <= 100 - SEVERITY_PENALTY["critical"]
                assert result.assessments[rid].light != "green"


def test_finding_criterion_is_marked_violation(result):
    # Регрессия (вариант no_reviewer): Reviewer упал, _finalize создавал отметки «не оценено», и критерий с находкой
    # правил оставался «na» вместо «нарушен»
    for f in result.findings:
        if per_requirement(f):
            for rid in f.rids:
                marks = [m for m in result.assessments[rid].marks if m.criterion == f.criterion]
                assert marks and all(m.status == VIOLATION for m in marks)


def test_document_score_bounds_and_monotonic(sample_md):
    res = run(str(sample_md), Options(), llm=_fake(sample_md))
    assert res.document_score is not None and 0 <= res.document_score <= 100
    clean = _fake(sample_md)
    clean.responses["reviewer"] = {"assessments": []}
    clean.responses["consistency"] = {"set_marks": [], "findings": []}
    better = run(str(sample_md), Options(), llm=clean)
    assert (better.document_score or 0) >= res.document_score


def test_score_in_report_equals_score_in_json(sample_md, tmp_path):
    res = run(str(sample_md), Options(), llm=_fake(sample_md))
    write_outputs(res, str(tmp_path / "r.md"), ("md", "json"))
    data = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    md = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert data["document_score"] == res.document_score
    for rid, a in data["assessments"].items():
        assert a["score"] == res.assessments[rid].score
        assert str(a["score"]) in md


# ---------- строки и порядок разделов ----------

def test_every_finding_in_report_has_line_number(sample_md, tmp_path, result):
    for f in result.findings:
        assert f.quotes and all(q.line for q in f.quotes), f"у находки «{f.title}» нет номера строки"
    out = tmp_path / "r.md"
    write_outputs(result, str(out), ("md",))
    md = out.read_text(encoding="utf-8")
    for f in result.findings:
        for q in f.quotes:
            assert f"строка {q.line}" in md or f"строки {q.line}" in md or f"{q.line}" in md


def test_sections_in_order_1_to_6(sample_md, tmp_path, result):
    out = tmp_path / "r.md"
    write_outputs(result, str(out), ("md",))
    md = out.read_text(encoding="utf-8")
    pos = [md.index(s) for s in REQUIRED_SECTIONS]
    assert pos == sorted(pos) and len(set(pos)) == 6
    assert all(md.count(s) == 1 for s in REQUIRED_SECTIONS)


@pytest.mark.parametrize("offline", [True, False])
def test_sections_present_without_llm(sample_md, tmp_path, offline):
    res = run(str(sample_md), Options(offline=offline))
    out = tmp_path / "r.md"
    write_outputs(res, str(out), ("md",))
    md = out.read_text(encoding="utf-8")
    pos = [md.index(s) for s in REQUIRED_SECTIONS]
    assert pos == sorted(pos)


def test_finding_ids_unique_and_requirement_ids_sequential(result):
    fids = [f.fid for f in result.findings]
    assert len(fids) == len(set(fids))
    assert [r.rid for r in result.requirements] == [f"R-{i:03d}" for i in range(1, len(result.requirements) + 1)]


# ---------- независимость от порядка завершения агентов и от PYTHONHASHSEED ----------

class DelayLLM(FakeLLM):
    """FakeLLM с задержкой по ролям: меняет порядок завершения параллельных агентов."""

    def __init__(self, base: FakeLLM, delays: dict[str, float]):
        super().__init__(base.responses, fail_roles=base.fail_roles)
        self.delays = delays

    def generate_json(self, role, system, prompt):
        time.sleep(self.delays.get(role, 0.0))
        return super().generate_json(role, system, prompt)


_VOLATILE = re.compile(r"(sec|run_at|duration|time|traceback|elapsed)", re.I)


def normalize(obj):
    """Убрать из отчёта дату, время и длительности (всё, что законно меняется между прогонами)."""
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in obj.items() if not _VOLATILE.search(k)}
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    return obj


def normalize_report(data: dict) -> dict:
    """normalize + убрать оценку ценности: она считается от длительности прогона (экономия часов)."""
    data = normalize(data)
    data.get("metrics", {}).pop("value", None)
    return data


def _dump(res):
    return json.dumps(normalize_report(json.loads(json.dumps(res.to_dict(), default=str))), ensure_ascii=False,
                      sort_keys=True, indent=1)


ROLES = ["extractor", "reviewer", "consistency", "gaps", "critic", "analyst_checklist", "analyst_tests", "clarifier"]
DELAY_PROFILES = {
    "none": {},
    "forward": {r: 0.01 * i for i, r in enumerate(ROLES)},
    "reverse": {r: 0.01 * (len(ROLES) - i) for i, r in enumerate(ROLES)},
    "reviewer_slow": {"reviewer": 0.15},
    "consistency_slow": {"consistency": 0.15, "gaps": 0.05},
}


@pytest.mark.parametrize("profile", [p for p in DELAY_PROFILES if p != "none"])
def test_result_independent_of_agent_completion_order(sample_md, profile):
    base = _dump(run(str(sample_md), Options(), llm=DelayLLM(_with_clarify(sample_md), DELAY_PROFILES["none"])))
    other = _dump(run(str(sample_md), Options(), llm=DelayLLM(_with_clarify(sample_md), DELAY_PROFILES[profile])))
    assert base == other


def test_result_stable_across_repeated_runs(sample_md):
    assert _dump(run(str(sample_md), Options(), llm=_fake(sample_md))) == _dump(run(str(sample_md), Options(), llm=_fake(sample_md)))


def _clean_env(seed: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
    env.update({"REQCHECK_NO_DOTENV": "1", "PYTHONUTF8": "1", "REQCHECK_VERSION": "test", "PYTHONHASHSEED": seed})
    return env


_SCRIPT = """
import json, sys
sys.path.insert(0, {root!r}); sys.path.insert(0, {tests!r})
from test_pipeline import _with_clarify
from reqcheck.pipeline import Options, run
res = run(sys.argv[1], Options(), llm=_with_clarify(sys.argv[1]))
sys.stdout.buffer.write(json.dumps(res.to_dict(), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
"""


def _subprocess_json(args, env, cwd):
    proc = subprocess.run(args, env=env, cwd=cwd, capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")[-2000:]
    return proc.stdout


@pytest.mark.parametrize("seeds", [("0", "12345"), ("1", "987654321")])
def test_json_report_independent_of_hashseed_cli(sample_md, tmp_path, seeds):
    dumps = []
    for i, seed in enumerate(seeds):
        out = tmp_path / f"seed{i}" / "r.md"
        _subprocess_json([sys.executable, "-m", "reqcheck", str(sample_md), str(out), "--offline", "--formats", "json"],
                         _clean_env(seed), ROOT)
        dumps.append(json.dumps(normalize_report(json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))),
                                ensure_ascii=False, sort_keys=True))
    assert dumps[0] == dumps[1]


@pytest.mark.parametrize("seeds", [("0", "424242")])
def test_fake_llm_result_independent_of_hashseed(sample_md, tmp_path, seeds):
    script = tmp_path / "run_fake.py"
    script.write_text(_SCRIPT.format(root=str(ROOT), tests=str(TESTS)), encoding="utf-8")
    dumps = []
    for seed in seeds:
        raw = _subprocess_json([sys.executable, str(script), str(sample_md)], _clean_env(seed), tmp_path)
        dumps.append(json.dumps(normalize_report(json.loads(raw.decode("utf-8"))), ensure_ascii=False, sort_keys=True))
    assert dumps[0] == dumps[1]
