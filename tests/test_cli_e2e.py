"""Сквозные тесты CLI: коды выхода, пути с кириллицей/пробелами, форматы, защита входного файла, subprocess."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from reqcheck.cli import main
from reqcheck.report_md import REQUIRED_SECTIONS
from test_pipeline import _fake

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def with_fake_llm(monkeypatch, sample_md):
    """Подменяет создание GeminiClient на подставную модель: CLI идёт по пути «с LLM» без сети."""
    monkeypatch.setattr("reqcheck.steps.GeminiClient", lambda **kw: _fake(sample_md))


def _clean_env() -> dict:
    """Окружение subprocess: без ключей, без .env, UTF-8."""
    env = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
    env.update({"REQCHECK_NO_DOTENV": "1", "PYTHONUTF8": "1", "REQCHECK_VERSION": "test"})
    return env


# ---------- коды выхода ----------

def test_exit_0_offline(sample_md, tmp_path, capsys):
    assert main([str(sample_md), str(tmp_path / "r.md"), "--offline"]) == 0
    assert "Отчёт:" in capsys.readouterr().out


def test_exit_0_with_fake_llm(sample_md, tmp_path, with_fake_llm, capsys):
    assert main([str(sample_md), str(tmp_path / "r.md")]) == 0
    assert "fake-model" in (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "Разработка: Нет" in capsys.readouterr().out  # вердикт напечатан в консоль


def test_exit_0_without_key_falls_back_to_rules(sample_md, tmp_path):
    # ключа нет, --offline не задан: LLM_UNAVAILABLE, но проверка правилами выполнена
    assert main([str(sample_md), str(tmp_path / "r.md")]) == 0


@pytest.mark.parametrize("content,name", [
    (b"%PDF-1.7\n" + b"\x00garbage" * 50, "broken.pdf"),
    (b"", "empty.md"),
    (b"PK\x03\x04not a docx", "broken.docx"),
])
def test_exit_1_unreadable_input_still_writes_report(tmp_path, content, name):
    bad = tmp_path / name
    bad.write_bytes(content)
    out = tmp_path / "report.md"
    assert main([str(bad), str(out), "--offline"]) == 1
    md = out.read_text(encoding="utf-8")
    assert "Нет — документ не проверен" in md
    assert all(s in md for s in REQUIRED_SECTIONS)


def test_exit_2_output_parent_is_a_file(sample_md, tmp_path, capsys):
    blocker = tmp_path / "file.txt"
    blocker.write_text("x", encoding="utf-8")
    assert main([str(sample_md), str(blocker / "sub" / "r.md"), "--offline"]) == 2
    out = capsys.readouterr()
    assert "отчёт не записан" in out.err
    assert "[error]" in out.out


def test_only_target_is_input_is_skipped_but_questions_written(sample_md):
    # единственный формат совпадает с входным файлом: он не тронут, остаётся файл вопросов (код 0)
    before = sample_md.read_bytes()
    assert main([str(sample_md), str(sample_md), "--offline", "--formats", "md"]) == 0
    assert sample_md.read_bytes() == before


@pytest.mark.parametrize("argv", [
    [],
    ["only_input.md"],
    ["a.md", "b.md", "--no-such-flag"],
    ["a.md", "b.md", "--timeout", "abc"],
    ["a.md", "b.md", "--max-size-mb", "x"],
])
def test_exit_3_bad_arguments(argv):
    assert main(argv) == 3


def test_help_and_version_exit_0(capsys):
    assert main(["--version"]) == 0
    assert main(["--help"]) == 0
    assert "reqcheck" in capsys.readouterr().out


# ---------- несуществующий вход, неподдерживаемое расширение ----------

def test_missing_input_file(tmp_path):
    out = tmp_path / "r.md"
    assert main([str(tmp_path / "нет такого.md"), str(out), "--offline"]) == 1
    assert "Нет — документ не проверен" in out.read_text(encoding="utf-8")


@pytest.mark.parametrize("ext", [".exe", ".bin", ".xyz"])
def test_unsupported_binary_input(tmp_path, ext):
    # формат определяется по сигнатуре; двоичный файл с любым расширением не проверяется
    doc = tmp_path / f"doc{ext}"
    doc.write_bytes(b"MZ\x90\x00\x03\x00" * 40)
    out = tmp_path / "r.md"
    assert main([str(doc), str(out), "--offline"]) == 1
    assert "Нет — документ не проверен" in out.read_text(encoding="utf-8")


@pytest.mark.parametrize("ext", [".xyz", ".csv", ""])
def test_unknown_extension_with_text_is_read_as_text(tmp_path, ext):
    doc = tmp_path / f"doc{ext}"
    doc.write_text("FR-1. Система должна отправлять SMS клиенту.", encoding="utf-8")
    assert main([str(doc), str(tmp_path / "r.md"), "--offline"]) == 0


def test_too_large_input(sample_md, tmp_path):
    # Нижний предел лимита в CLI — 0,001 МБ (~1 КБ); файл заведомо больше на любой ОС (на Linux sample_md короче 1 КБ)
    big = tmp_path / "big.md"
    big.write_bytes(sample_md.read_bytes() + b"\n" + "Текст требования. ".encode("utf-8") * 200)
    out = tmp_path / "r.md"
    assert main([str(big), str(out), "--offline", "--max-size-mb", "0.0001"]) == 1
    assert out.exists()


# ---------- кириллица и пробелы в путях ----------

@pytest.mark.parametrize("in_name,out_dir,out_name", [
    ("Требования ТЗ.md", "Папка отчётов", "Отчёт 1.md"),
    ("документ.md", "out", "report.md"),
    ("my doc.md", "с пробелами и кириллицей/вложенная папка", "итог.md"),
])
def test_cyrillic_and_spaces_in_paths(sample_md, tmp_path, in_name, out_dir, out_name):
    src = tmp_path / in_name
    src.write_bytes(sample_md.read_bytes())
    out = tmp_path / out_dir / out_name
    assert main([str(src), str(out), "--offline"]) == 0
    stem = out.with_suffix("")
    for ext in (".md", ".pdf", ".json"):
        assert stem.with_suffix(ext).exists(), ext
    assert json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))["passport"]["file"].endswith(in_name)


# ---------- --formats ----------

@pytest.mark.parametrize("formats", ["md", "pdf", "json", "md,pdf", "md,json", "pdf,json", "md,pdf,json", "JSON, PDF", "json,json"])
def test_formats_for_md_output(sample_md, tmp_path, formats):
    out = tmp_path / "rep" / "r.md"
    assert main([str(sample_md), str(out), "--offline", "--formats", formats]) == 0
    expected = {"md"} | {f.strip().lower() for f in formats.split(",")}
    present = {p.suffix.lstrip(".") for p in out.parent.glob("r.*") if not p.stem.endswith("_questions")}
    assert present == expected
    if "pdf" in expected:
        assert out.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
    if "json" in expected:
        assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["metrics"]["requirements"] > 0


@pytest.mark.parametrize("formats", ["md", "json", "md,json"])
def test_pdf_main_output_is_always_written(sample_md, tmp_path, formats):
    out = tmp_path / "r.pdf"
    assert main([str(sample_md), str(out), "--offline", "--formats", formats]) == 0
    assert out.read_bytes().startswith(b"%PDF")
    expected = {"pdf"} | set(formats.split(","))
    present = {p.suffix.lstrip(".") for p in tmp_path.glob("r.*")}
    assert present == expected


def test_unknown_format_is_skipped_with_warning(sample_md, tmp_path, capsys):
    out = tmp_path / "r.md"
    assert main([str(sample_md), str(out), "--offline", "--formats", "md,docx"]) == 0
    assert "docx" in capsys.readouterr().out  # предупреждение UNKNOWN_FORMAT, не падение
    assert not out.with_suffix(".docx").exists()


def test_empty_formats_still_writes_main_file(sample_md, tmp_path):
    out = tmp_path / "r.md"
    assert main([str(sample_md), str(out), "--offline", "--formats", ""]) == 0
    assert out.exists() and not out.with_suffix(".pdf").exists()


# ---------- --answers ----------

def test_answers_missing_file_is_diagnostic_not_crash(sample_md, tmp_path, capsys):
    out = tmp_path / "r.md"
    assert main([str(sample_md), str(out), "--offline", "--answers", str(tmp_path / "нет файла.md")]) == 0
    assert "ANSWERS_UNREADABLE" in json.dumps(json.loads(out.with_suffix(".json").read_text(encoding="utf-8")),
                                             ensure_ascii=False)


def test_answers_roundtrip_through_cli(sample_md, tmp_path, with_fake_llm):
    out = tmp_path / "Мои отчёты" / "r.md"
    assert main([str(sample_md), str(out), "--formats", "md"]) == 0
    qfile = out.parent / "r_questions.md"
    assert qfile.exists()  # вопросы автору записаны рядом с отчётом
    qfile.write_text(qfile.read_text(encoding="utf-8").replace("\nОтвет:", "\nОтвет: Лимит 50 000.", 1), encoding="utf-8")
    out2 = tmp_path / "r2.md"
    assert main([str(sample_md), str(out2), "--formats", "md", "--answers", str(qfile)]) == 0
    assert out2.exists()


def test_answers_file_in_cyrillic_dir(sample_md, tmp_path):
    d = tmp_path / "ответы автора"
    d.mkdir()
    a = d / "вопросы и ответы.md"
    a.write_text("## Q-01 · x\n\nВопрос: Какой лимит?\nОтвет: 50 000\n", encoding="utf-8")
    assert main([str(sample_md), str(tmp_path / "r.md"), "--offline", "--answers", str(a)]) == 0


# ---------- защита входного файла и сбои записи ----------

def test_report_never_overwrites_input(sample_md, capsys):
    before = sample_md.read_bytes()
    out = sample_md.with_suffix(".pdf")  # «tz.md tz.pdf»: Markdown-версия легла бы на сам tz.md
    assert main([str(sample_md), str(out), "--offline"]) == 0
    assert sample_md.read_bytes() == before
    assert out.read_bytes().startswith(b"%PDF")
    assert "входной документ" in capsys.readouterr().out  # WRITE_SKIPPED показан пользователю


def test_write_skipped_via_other_spelling(sample_md, tmp_path):
    # тот же файл, записанный иначе (sub/..): resolve() должен его узнать
    before = sample_md.read_bytes()
    (tmp_path / "sub").mkdir()
    assert main([str(sample_md), str(tmp_path / "sub" / ".." / "req.pdf"), "--offline"]) == 0
    assert sample_md.read_bytes() == before


def test_write_failed_for_one_format_does_not_block_others(sample_md, tmp_path, capsys):
    out = tmp_path / "r.md"
    (tmp_path / "r.pdf").mkdir()  # на месте PDF лежит папка: запись PDF упадёт
    assert main([str(sample_md), str(out), "--offline"]) == 0  # md и json записаны → код 0
    assert out.exists() and out.with_suffix(".json").exists()
    assert "Не удалось записать r.pdf" in capsys.readouterr().out


def test_write_failed_all_formats_gives_exit_2(sample_md, tmp_path):
    for name in ("r.md", "r.pdf", "r.json", "r.html", "r_questions.md"):  # на месте каждого файла лежит папка
        (tmp_path / name).mkdir()
    assert main([str(sample_md), str(tmp_path / "r.md"), "--offline"]) == 2


def test_output_dir_is_created(sample_md, tmp_path):
    out = tmp_path / "a" / "б" / "в г" / "r.md"
    assert main([str(sample_md), str(out), "--offline", "--formats", "md"]) == 0
    assert out.exists()


# ---------- subprocess ----------

def test_subprocess_offline_cyrillic_paths(sample_md, tmp_path):
    src = tmp_path / "Требования к системе.md"
    src.write_bytes(sample_md.read_bytes())
    out = tmp_path / "Отчёты ЕНУ" / "отчёт.md"
    proc = subprocess.run([sys.executable, "-m", "reqcheck", str(src), str(out), "--offline"],
                          cwd=ROOT, env=_clean_env(), capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert "Отчёт:" in proc.stdout.decode("utf-8")
    md = out.read_text(encoding="utf-8")
    assert all(s in md for s in REQUIRED_SECTIONS)
    assert out.with_suffix(".json").exists() and out.with_suffix(".pdf").exists()


@pytest.mark.parametrize("args,code", [([], 3), (["--version"], 0)])
def test_subprocess_exit_codes(args, code):
    proc = subprocess.run([sys.executable, "-m", "reqcheck", *args], cwd=ROOT, env=_clean_env(),
                          capture_output=True, timeout=60)
    assert proc.returncode == code
