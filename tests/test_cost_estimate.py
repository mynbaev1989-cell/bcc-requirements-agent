"""Оценка объёма запросов по кэшу (tools/cost_estimate.py)."""

from pathlib import Path

from tools.cost_estimate import measure, tokens

ROOT = Path(__file__).resolve().parents[1]


def test_tokens_range():
    assert tokens(12000) == (3000, 4000)


def test_measure_on_committed_cache(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "cache-only")  # ключ не используется: клиент без сети
    r = measure(ROOT / "eval" / "p2p_phone.md", "gemini-3.5-flash-lite", ROOT / "eval" / "llm_cache")
    assert r["cache_misses"] == 0
    assert 8 <= r["requests"] <= 20
    assert r["prompt_chars"] > r["output_chars"] > 0
