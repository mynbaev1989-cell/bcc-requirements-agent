"""Пример ворот CI (examples/ci_gate.yml): скрипт падает при открытых критичных дефектах и пропускает документ без них."""

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _gate_code() -> str:
    y = (ROOT / "examples" / "ci_gate.yml").read_text(encoding="utf-8")
    body = re.search(r"python - <<'EOF'\n(.*?)\n\s*EOF", y, re.S).group(1)
    return textwrap.dedent(body)


def _run_gate(tmp_path: Path, report: dict) -> subprocess.CompletedProcess:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "requirements_report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return subprocess.run([sys.executable, "-c", _gate_code()], cwd=tmp_path, capture_output=True,
                          text=True, encoding="utf-8")


def test_gate_fails_on_critical(tmp_path):
    report = {"verdict_dev": "Нет", "findings": [
        {"fid": "F-001", "severity": "critical", "title": "Противоречие", "quotes": [{"line": 12}, {"line": 40}]},
        {"fid": "F-002", "severity": "major", "title": "Размыто", "quotes": [{"line": 5}]}]}
    p = _run_gate(tmp_path, report)
    assert p.returncode == 1
    assert "::error file=docs/requirements.md,line=12::F-001" in p.stdout
    assert "F-002" not in p.stdout


def test_gate_passes_without_critical(tmp_path):
    report = {"verdict_dev": "Условно", "findings": [{"fid": "F-001", "severity": "major", "title": "x", "quotes": []}]}
    assert _run_gate(tmp_path, report).returncode == 0


def test_gate_reads_real_report_shape():
    """Поля, которые читают ворота, есть в настоящем JSON-отчёте агента."""
    r = json.loads((ROOT / "reports" / "requirements_report.json").read_text(encoding="utf-8"))
    assert "verdict_dev" in r
    assert all({"fid", "severity", "title", "quotes"} <= set(f) for f in r["findings"])
