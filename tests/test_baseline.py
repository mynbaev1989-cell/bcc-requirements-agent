"""Базовая линия «один промпт» воспроизводится из закоммиченного кэша, без сети и без ключа."""
import json
import sys
from pathlib import Path

from tools import baseline_single_prompt as bl

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_replays_from_committed_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "cache-only")  # ключ не используется: все ответы в кэше
    out = tmp_path / "b.json"
    monkeypatch.setattr(sys, "argv", ["bl", "--cache", str(ROOT / "eval" / "llm_cache_baseline"), "--out", str(out)])
    assert bl.main() == 0
    tot = json.loads(out.read_text(encoding="utf-8"))["total"]
    committed = json.loads((ROOT / "eval" / "baseline_single_prompt.json").read_text(encoding="utf-8"))["total"]
    assert tot == committed and tot["gold_defects"] == 104
