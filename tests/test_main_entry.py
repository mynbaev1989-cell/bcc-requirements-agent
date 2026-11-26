"""Команда из формы сдачи: python main.py --input <документ> --output <отчёт>; старая позиционная форма тоже работает."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = "# ТЗ\n\nВерсия документа: 1.0\n\n## Требования\n\n- FR-1. Система должна отвечать быстро.\n"


def _run(args, cwd):
    env = {**os.environ, "REQCHECK_NO_DOTENV": "1", "PYTHONIOENCODING": "utf-8"}
    env.pop("GEMINI_API_KEY", None)
    env.pop("REQCHECK_LLM_BASE_URL", None)
    return subprocess.run([sys.executable, str(ROOT / "main.py"), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, encoding="utf-8")


def _doc(tmp_path):
    (tmp_path / "req.md").write_text(DOC, encoding="utf-8")
    return tmp_path / "req.md"


def test_form_command_writes_report(tmp_path):
    doc = _doc(tmp_path)
    p = _run(["--input", str(doc), "--output", str(tmp_path / "out" / "report.md"), "--offline"], tmp_path)
    assert p.returncode == 0, p.stderr
    for name in ("report.md", "report.pdf", "report.json"):
        assert (tmp_path / "out" / name).exists()


def test_short_options(tmp_path):
    doc = _doc(tmp_path)
    p = _run(["-i", str(doc), "-o", str(tmp_path / "r.md"), "--offline", "--formats", "md"], tmp_path)
    assert p.returncode == 0, p.stderr
    assert (tmp_path / "r.md").exists()


def test_positional_form_still_works(tmp_path):
    doc = _doc(tmp_path)
    p = _run([str(doc), str(tmp_path / "r.md"), "--offline", "--formats", "md"], tmp_path)
    assert p.returncode == 0, p.stderr


def test_missing_paths_is_usage_error(tmp_path):
    p = _run(["--input", str(_doc(tmp_path))], tmp_path)
    assert p.returncode == 3
    assert "--output" in p.stderr


def test_conflicting_paths_is_usage_error(tmp_path):
    doc = _doc(tmp_path)
    p = _run([str(doc), str(tmp_path / "a.md"), "--output", str(tmp_path / "b.md"), "--offline"], tmp_path)
    assert p.returncode == 3
    assert not (tmp_path / "a.md").exists() and not (tmp_path / "b.md").exists()


def test_same_path_twice_is_fine(tmp_path):
    doc = _doc(tmp_path)
    out = str(tmp_path / "r.md")
    p = _run([str(doc), out, "--output", out, "--offline", "--formats", "md"], tmp_path)
    assert p.returncode == 0, p.stderr


def test_help_shows_form_command(tmp_path):
    p = _run(["--help"], tmp_path)
    assert p.returncode == 0
    assert "main.py" in p.stdout and "--input" in p.stdout and "--output" in p.stdout


def test_questions_file_suggests_form_command(tmp_path):
    """Файл вопросов автору подсказывает повторный запуск той же командой, что в форме сдачи."""
    doc = tmp_path / "req.md"
    extra = "- FR-2. Лимит снятия наличных составляет 100 000 тенге.\n\n## Нагрузка\n\nTBD\n"
    doc.write_text(DOC + extra, encoding="utf-8")
    p = _run(["--input", str(doc), "--output", str(tmp_path / "r.md"), "--offline", "--formats", "md"], tmp_path)
    assert p.returncode == 0, p.stderr
    q = (tmp_path / "r_questions.md").read_text(encoding="utf-8")
    assert "python main.py --input" in q and "--answers" in q
