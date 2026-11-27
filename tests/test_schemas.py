"""Схемы ответов моделей (reqcheck/schemas.py): режим «только наблюдение»."""

import copy
import json
from pathlib import Path

import pytest

from reqcheck.llm import GeminiClient
from reqcheck.llm_openai import OpenAICompatClient
from reqcheck.schemas import SCHEMAS, violations

_FINDING = {"criterion": "verifiable", "severity": "major", "title": "t", "explanation": "e", "rids": ["R-001"],
            "quotes": ["q"], "example": "x", "suggestion": "s", "question": "?", "addressee": "Аналитик"}
_MARK = {"criterion": "necessary", "status": "ok", "comment": ""}

# Корректный пример по каждой роли
VALID = {
    "extractor": {"requirements": [{"block_ids": ["B-001"], "source_id": None, "quote": "q", "summary": "s",
                                    "kind": "functional"}]},
    "reviewer": {"assessments": [{"rid": "R-001", "marks": [_MARK], "findings": [_FINDING]}]},
    "consistency": {"set_marks": [_MARK], "findings": [_FINDING]},
    "gaps": {"findings": [_FINDING]},
    "critic": {"verdicts": [{"fid": "T-001", "verdict": "confirm", "severity": "critical", "duplicate_of": None,
                             "reason": "r"}]},
    "analyst_checklist": {"checklist": [{"index": 1, "answer": "Нет", "justification": "j", "quotes": ["q"]}],
                          "trace": [{"rid": "R-001", "acceptance": [], "api": [], "note": "n"}]},
    "analyst_tests": {"tests": [{"rid": "R-001", "title": "t", "given": "g", "when": "w", "then": "t"}]},
    "clarifier": {"questions": [{"priority": "blocker", "addressee": "Аналитик", "question": "?", "why": "w",
                                 "options": ["a"], "assumption": "as", "fids": ["F-001"]}]},
    "resolver": {"self_contradictions": [{"qid": "Q-01", "part_a": "a", "part_b": "b"}],
                 "effects": [{"qid": "Q-01", "fid": "F-001", "status": "resolved", "comment": "c",
                              "revised_wording": "w", "chosen_values": ["1"], "follow_up": ""}]},
}

# Порча: (роль, функция порчи, подстрока ожидаемого пути в сообщении)
BREAKS = [
    ("extractor", lambda d: d["requirements"][0].pop("block_ids"), "requirements[0].block_ids"),
    ("extractor", lambda d: d.pop("requirements"), "requirements"),
    ("reviewer", lambda d: d["assessments"][0].pop("rid"), "assessments[0].rid"),
    ("reviewer", lambda d: d["assessments"][0]["findings"][0].update(severity="huge"), "findings[0].severity"),
    ("reviewer", lambda d: d["assessments"][0]["marks"][0].update(status="maybe"), "marks[0].status"),
    ("consistency", lambda d: d["findings"][0].update(rids="R-001"), "findings[0].rids"),
    ("consistency", lambda d: d["findings"][0].pop("title"), "findings[0].title"),
    ("gaps", lambda d: d["findings"][0].update(severity="blocker"), "findings[0].severity"),
    ("gaps", lambda d: d.update(findings={"a": 1}), "findings"),
    ("critic", lambda d: d["verdicts"][0].update(verdict="maybe"), "verdicts[0].verdict"),
    ("critic", lambda d: d["verdicts"][0].pop("fid"), "verdicts[0].fid"),
    ("analyst_checklist", lambda d: d["checklist"][0].pop("index"), "checklist[0].index"),
    ("analyst_checklist", lambda d: d["trace"][0].update(acceptance="R-1"), "trace[0].acceptance"),
    ("analyst_tests", lambda d: d["tests"][0].pop("title"), "tests[0].title"),
    ("analyst_tests", lambda d: d["tests"][0].update(then=5), "tests[0].then"),
    ("clarifier", lambda d: d["questions"][0].pop("fids"), "questions[0].fids"),
    ("clarifier", lambda d: d["questions"][0].update(question=None), "questions[0].question"),
    ("resolver", lambda d: d["effects"][0].update(status="revised_wording"), "effects[0].status"),
    ("resolver", lambda d: d["effects"][0].pop("fid"), "effects[0].fid"),
    ("resolver", lambda d: d.pop("effects"), "effects"),
]


def test_every_role_has_schema_and_example():
    assert set(VALID) == set(SCHEMAS)


@pytest.mark.parametrize("role", sorted(VALID))
def test_valid_example_passes(role):
    assert violations(role, VALID[role]) == []


@pytest.mark.parametrize("role,mutate,path", BREAKS)
def test_broken_example_is_reported(role, mutate, path):
    data = copy.deepcopy(VALID[role])
    mutate(data)
    found = violations(role, data)
    assert found and any(path in m for m in found), found


def test_every_role_detects_wrong_top_level_type():
    for role in SCHEMAS:
        if role == "gaps":  # голый список для gaps допустим (так отвечает модель, код его не использует)
            continue
        assert violations(role, ["x"]), role


def test_message_format_and_extra_fields():
    data = copy.deepcopy(VALID["gaps"])
    data["findings"][0]["severity"] = "huge"
    data["findings"][0]["unknown_field"] = 1  # лишние поля допустимы
    data["extra"] = True
    assert violations("gaps", data) == [
        "findings[0].severity: ожидалось одно из ['critical', 'major', 'minor'], получено 'huge'"]


def test_unknown_role_is_empty():
    assert violations("nope", {"anything": 1}) == []
    assert violations("", None) == []


def test_bool_is_not_integer():
    data = copy.deepcopy(VALID["analyst_checklist"])
    data["checklist"][0]["index"] = True
    assert violations("analyst_checklist", data)


# --- встраивание в клиентов -------------------------------------------------------------------------
BAD = {"findings": [{"title": "t", "severity": "huge"}, {"severity": "major"}, {"title": "u", "severity": "x"}]}


def _gemini(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return GeminiClient(model="model-a", retries=0, cache_dir=str(tmp_path))


def _openai(tmp_path):
    return OpenAICompatClient(base_url="http://localhost:11434/v1", model="m", cache_dir=str(tmp_path))


@pytest.mark.parametrize("kind", ["gemini", "openai"])
def test_cache_hit_returns_same_data_and_notes_violations(tmp_path, monkeypatch, kind):
    llm = _gemini(tmp_path, monkeypatch) if kind == "gemini" else _openai(tmp_path)
    before = copy.deepcopy(BAD)
    llm._cache_path("sys\n\nprompt").write_text(
        json.dumps({"role": "gaps", "model": llm.model, "parsed": BAD}), encoding="utf-8")
    assert llm.generate_json("gaps", "sys", "prompt") == before
    assert llm.stats.schema_violations == {"gaps": 3}
    assert len(llm.stats.schema_samples["gaps"]) == 2
    assert llm.stats.schema_samples["gaps"][0].startswith("findings[0].severity")
    assert llm.stats.errors == [] and llm.stats.failures == 0


@pytest.mark.parametrize("kind", ["gemini", "openai"])
def test_valid_cache_hit_leaves_stats_clean(tmp_path, monkeypatch, kind):
    llm = _gemini(tmp_path, monkeypatch) if kind == "gemini" else _openai(tmp_path)
    llm._cache_path("sys\n\nprompt").write_text(
        json.dumps({"role": "gaps", "model": llm.model, "parsed": VALID["gaps"]}), encoding="utf-8")
    assert llm.generate_json("gaps", "sys", "prompt") == VALID["gaps"]
    assert llm.stats.schema_violations == {} and llm.stats.schema_samples == {}


def test_network_answer_notes_violations_openai(tmp_path, monkeypatch):
    llm = _openai(tmp_path)
    monkeypatch.setattr(llm, "_post", lambda system, prompt: (json.dumps(BAD), {}))
    assert llm.generate_json("gaps", "sys", "prompt") == BAD
    assert llm.stats.schema_violations == {"gaps": 3}
    assert llm.stats.calls == 1  # повторных запросов нет


def test_network_answer_notes_violations_gemini(tmp_path, monkeypatch):
    from types import SimpleNamespace
    llm = _gemini(tmp_path, monkeypatch)
    n = []

    class Models:
        @staticmethod
        def generate_content(model, contents, config):
            n.append(1)
            return SimpleNamespace(text=json.dumps(BAD), usage_metadata=None, model_version=model,
                                   candidates=[SimpleNamespace(finish_reason="STOP")])

    llm._client = SimpleNamespace(models=Models)
    assert llm.generate_json("gaps", "sys", "prompt") == BAD
    assert llm.stats.schema_violations == {"gaps": 3}
    assert len(n) == 1


def test_unknown_role_in_client_is_ignored(tmp_path):
    llm = _openai(tmp_path)
    llm._cache_path("sys\n\nprompt").write_text(
        json.dumps({"role": "x", "model": "m", "parsed": 5}), encoding="utf-8")
    assert llm.generate_json("x", "sys", "prompt") == 5
    assert llm.stats.schema_violations == {}


# --- закоммиченный кэш: схема не строже кода -------------------------------------------------------
CACHE = Path(__file__).resolve().parents[1] / "eval" / "llm_cache"


def test_committed_cache_passes_schemas():
    files = sorted(CACHE.glob("*.json"))
    assert files, "нет файлов кэша"
    bad = {}
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        found = violations(rec["role"], rec["parsed"])
        if found:
            bad[f.name] = found[:2]
    assert bad == {}
