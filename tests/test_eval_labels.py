"""Разметка замера точности корректна: иначе цифры recall/precision в README считаются по битым эталонам."""

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("eval", "eval_holdout", "eval_blind")
LABELS = sorted(p for f in FOLDERS for p in (ROOT / f).glob("*.labels.json"))
TYPES = {"ambiguity", "unverifiable", "contradiction", "incomplete", "non_singular", "design_in_req",
         "units", "empty_section", "no_version", "api", "security", "regulatory"}


def test_all_eval_folders_have_labels():
    assert {p.parent.name for p in LABELS} == set(FOLDERS)


def test_ablation_measures_every_labelled_folder():
    # tools/ablation.py не импортируем: он при импорте ставит GEMINI_API_KEY и повлиял бы на другие тесты
    src = (ROOT / "tools" / "ablation.py").read_text(encoding="utf-8")
    line = next(x for x in src.splitlines() if x.startswith("FOLDERS = "))
    assert set(ast.literal_eval(line.split("=", 1)[1].strip())) == set(FOLDERS)


@pytest.mark.parametrize("path", LABELS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_labels_are_consistent_with_document(path):
    labels = json.loads(path.read_text(encoding="utf-8"))
    doc = (path.parent / labels["document"]).read_text(encoding="utf-8")
    lines = doc.splitlines()
    ids = [d["id"] for d in labels["defects"]]
    assert len(ids) == len(set(ids)), "id дефектов повторяются"
    for d in labels["defects"]:
        assert d["type"] in TYPES, d["id"]
        assert d["severity"] in ("critical", "major", "minor"), d["id"]
        assert d["quote"] in doc, f"{d['id']}: цитаты нет в документе"
        assert d["quote"] in lines[d["line"] - 1], f"{d['id']}: строка {d['line']} не содержит цитату"
        if d.get("related_quote"):
            assert d["related_quote"] in doc, f"{d['id']}: related_quote нет в документе"
    for q in labels.get("clean_requirements", []):
        assert q in doc, f"чистого требования нет в документе: {q}"
