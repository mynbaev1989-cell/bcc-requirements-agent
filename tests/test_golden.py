"""Золотые эталоны отчёта: фиксированный документ + фиксированные ответы FakeLLM → report.md / report.json.

Обновить эталоны: REQCHECK_UPDATE_GOLDEN=1 python -m pytest tests/test_golden.py
В обычном прогоне эталоны только сравниваются.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import types
from pathlib import Path

import pytest
from conftest import SAMPLE_MD
from test_pipeline import _fake

from reqcheck.pipeline import Options, run, write_outputs
from reqcheck.report_md import REQUIRED_SECTIONS

GOLDEN = Path(__file__).parent / "golden"
UPDATE = os.environ.get("REQCHECK_UPDATE_GOLDEN") == "1"


# ---------- нормализация изменчивого ----------

def _norm_text(s: str, tmp: Path) -> str:
    """Единые переводы строк, без абсолютных путей и временных папок."""
    s = s.replace("\r\n", "\n")
    for variant in {str(tmp), tmp.as_posix(), str(tmp).replace("\\", "\\\\")}:
        s = s.replace(variant, "<TMP>")
    s = re.sub(r"<TMP>[\/]+", "<TMP>/", s)
    return s


def normalize_md(md: str, tmp: Path) -> str:
    md = _norm_text(md, tmp)
    md = re.sub(r"(\| Дата и время прогона \| ).*?( \|)$", r"\1<RUN_AT>\2", md, flags=re.M)
    md = re.sub(r"(\| Версия агента \| ).*?( \|)$", r"\1<VERSION>\2", md, flags=re.M)
    md = re.sub(r"(\| Длительность прогона \| ).*?( \|)$", r"\1<DURATION>\2", md, flags=re.M)
    md = re.sub(r"(\| Токены \(вход / выход\) \| ).*?( \|)$", r"\1<TOKENS>\2", md, flags=re.M)
    return md


_VOLATILE_KEYS = {"run_at": "<RUN_AT>", "agent_version": "<VERSION>"}
_ZERO_KEYS = {"seconds", "duration_sec", "elapsed", "duration"}


def _walk(o, tmp: Path):
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if k in _VOLATILE_KEYS:
                out[k] = _VOLATILE_KEYS[k]
            elif k in _ZERO_KEYS and isinstance(v, (int, float)):
                out[k] = 0
            elif k in ("path", "input_path"):
                out[k] = "<PATH>"
            else:
                out[k] = _walk(v, tmp)
        return out
    if isinstance(o, list):
        return [_walk(x, tmp) for x in o]
    if isinstance(o, str):
        return _norm_text(o, tmp)
    return o


def normalize_json(text: str, tmp: Path) -> str:
    # sort_keys: порядок ключей не зависит от хода выполнения
    return json.dumps(_walk(json.loads(text), tmp), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


# ---------- прогон ----------

@pytest.fixture
def produced(tmp_path, monkeypatch):
    # Время прогона входит в оценку экономии (saved_hours, отдача): фиксируем часы конвейера, иначе эталон плавает
    monkeypatch.setattr("reqcheck.pipeline.time", types.SimpleNamespace(perf_counter=lambda: 0.0))
    src = tmp_path / "req.md"
    src.write_bytes(SAMPLE_MD.encode("utf-8"))  # байты, а не write_text: иначе на Windows \n -> \r\n и другой SHA-256
    result = run(str(src), Options(), llm=_fake(src))
    out = tmp_path / "out" / "report.md"
    _, problems = write_outputs(result, str(out), ("md", "json"))
    assert not problems
    md = out.read_text(encoding="utf-8")
    js = out.with_suffix(".json").read_text(encoding="utf-8")
    return result, normalize_md(md, tmp_path), normalize_json(js, tmp_path)


def _compare(name: str, actual: str):
    path = GOLDEN / name
    if UPDATE:
        GOLDEN.mkdir(exist_ok=True)
        path.write_bytes(actual.encode("utf-8"))  # всегда LF
        return
    assert path.exists(), f"нет эталона {name}: запустите с REQCHECK_UPDATE_GOLDEN=1"
    expected = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    if expected != actual:
        diff = list(difflib.unified_diff(expected.splitlines(), actual.splitlines(), "эталон", "получено", lineterm=""))
        pytest.fail(f"{name} расходится с эталоном:\n" + "\n".join(diff[:40]))


def test_golden_report_md(produced):
    _compare("report.md", produced[1])


def test_golden_report_json(produced):
    _compare("report.json", produced[2])


# ---------- структурные проверки ----------

def test_sections_order(produced):
    md = produced[1]
    positions = [md.index(s) for s in REQUIRED_SECTIONS]
    assert positions == sorted(positions)
    assert len(REQUIRED_SECTIONS) == 6
    # заголовки разделов не дублируются
    assert all(md.count(s) == 1 for s in REQUIRED_SECTIONS)


@pytest.mark.parametrize("label", ["Название документа", "Дата и время прогона", "Модель", "Версия агента"])
def test_passport_required_fields(produced, label):
    md = produced[1]
    m = re.search(rf"^\| {re.escape(label)} \| (.*) \|$", md, flags=re.M)
    assert m, f"в паспорте нет поля «{label}»"
    assert m.group(1).strip() not in ("", "None", "—")


def test_every_violation_has_a_row(produced):
    result, md, _ = produced
    assert result.findings
    for f in result.findings:
        assert f.fid in md, f"нарушение {f.fid} не попало в отчёт"
        assert re.search(rf"\*\*{re.escape(f.fid)}\*\*", md), f"у {f.fid} нет строки-заголовка"
