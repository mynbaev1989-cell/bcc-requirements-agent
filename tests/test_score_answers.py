"""Подсчёт слепого набора ответов: конкретный должен закрыть находку, уклончивый и противоречивый — нет."""
import json

from tools.score_answers import closed_count, score_folder


def test_closed_count_parses_outcome():
    assert closed_count("закрыто находок: 2 из 3") == 2
    assert closed_count("") is None


def test_score_folder(tmp_path):
    key = {"document": "d", "answers": [
        {"qid": "Q-01", "type": "concrete", "should_close": True},
        {"qid": "Q-02", "type": "evasive", "should_close": False},
        {"qid": "Q-03", "type": "contradictory", "should_close": False},
        {"qid": "Q-04", "type": "concrete", "should_close": True}]}
    report = {"questions": [{"qid": "Q-01", "outcome": "закрыто находок: 1 из 1"},
                            {"qid": "Q-02", "outcome": "закрыто находок: 0 из 2"},
                            {"qid": "Q-03", "outcome": "закрыто находок: 1 из 1"}]}  # Q-04 пропал из отчёта
    (tmp_path / "d_answers_key.json").write_text(json.dumps(key), encoding="utf-8")
    (tmp_path / "d.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    res = score_folder(tmp_path)
    assert res["correct"] == 2 and res["answers"] == 4
    assert res["by_type"] == {"concrete": [1, 2], "evasive": [1, 1], "contradictory": [0, 1]}
