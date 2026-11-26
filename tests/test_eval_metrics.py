"""Новые метрики замера точности (tools/eval_score.py): критичность, вопросы, extra.json.

tools.ablation здесь не импортируется: при импорте он ставит GEMINI_API_KEY."""

import json

from test_pipeline import _fake
from reqcheck.pipeline import Options
from tools.eval_score import load_extra, question_quality, score_document

# Эталон на документе из test_pipeline: FR-1 (в агенте critical), FR-3 (critical), третий дефект на несуществующей строке
LABELS = {
    "defects": [
        {"quote": "отправлять SMS клиенту быстро", "severity": "major", "type": "vague"},  # агент: critical -> разница 1
        {"quote": "не более 30 000 тенге", "severity": "critical", "type": "contradiction"},  # точно
        {"line": 500, "quote": "нет такого текста", "severity": "minor", "type": "other"},  # агент не нашёл
    ],
    "clean_requirements": [],
}


def _score(sample_md, extra=None):
    return score_document(sample_md, LABELS, Options(formats=()), llm=_fake(sample_md), extra=extra)


def test_new_fields_present_and_consistent(sample_md):
    row = _score(sample_md, extra={})
    assert row["found"] == 2 and row["gold_defects"] == 3
    assert row["severe_gold"] == 2  # critical/major среди найденных
    assert row["severe_covered_by_questions"] <= row["severe_gold"]
    assert 0 <= row["severe_question_coverage"] <= 1
    assert row["severity_exact_n"] == 1 and row["severity_within_one_n"] == 2
    assert row["severity_exact"] == 0.5 and row["severity_within_one"] == 1.0
    assert row["false_critical"] >= 0
    qq = row["question_quality"]
    assert qq["n"] == row["questions"]
    assert set(qq) >= {"with_addressee", "with_options", "with_assumption", "with_line", "duplicate_pairs"}
    assert row["precision_with_extra"] == row["precision"]  # без extra совпадает с precision


def test_extra_json_read_and_precision_not_lower(sample_md, tmp_path):
    base = _score(sample_md, extra={})
    # Находки агента, не совпавшие с эталоном, объявляем реальными дефектами через extra
    extra = {"document": sample_md.name,
             "extra_defects": [{"quote": u["quote"], "explanation": "подтверждено вручную"} for u in base["unmatched"] if u["quote"]]}
    row = _score(sample_md, extra=extra)
    assert row["precision"] == base["precision"]  # исходная метрика не меняется
    assert row["precision_with_extra"] >= row["precision"]
    assert row["extra_defects"] == len(extra["extra_defects"])


def test_load_extra(tmp_path):
    md = tmp_path / "doc.md"
    assert load_extra(md) == {}
    (tmp_path / "doc.extra.json").write_text(json.dumps({"extra_defects": [{"quote": "x", "line": 3}]}), encoding="utf-8")
    assert load_extra(md)["extra_defects"][0]["line"] == 3
    (tmp_path / "doc.extra.json").write_text("не json", encoding="utf-8")
    assert load_extra(md) == {}


def test_question_quality_duplicates():
    from reqcheck.models import Question

    qs = [Question("Q1", "Что на строке 5?", addressee="Аналитик", options=["да", "нет"], lines=[5, 6]),
          Question("Q2", "Ещё", addressee="", assumption="примем 10", lines=[6, 5]),
          Question("Q3", "Третий", lines=[9])]
    q = question_quality(qs)
    assert q["n"] == 3 and q["duplicate_pairs"] == 1
    assert q["with_options"] == round(1 / 3, 3) and q["with_assumption"] == round(1 / 3, 3)
    assert q["with_addressee"] == round(2 / 3, 3)
    assert question_quality([])["n"] == 0
