"""Ворота качества в CI: закоммиченный кэш ответов модели и пороги, ниже которых CI падает."""

import json
import re
from pathlib import Path

from tools.eval_score import GATE, gate_failures

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "eval" / "llm_cache"
GOOD = {"found": 102, "precision": 0.861, "false_critical": 0, "cache_misses": 0}


def test_current_result_passes_gate():
    assert gate_failures(GOOD) == []


def test_each_threshold_fails_gate():
    assert gate_failures({**GOOD, "found": GATE["found"] - 1})
    assert gate_failures({**GOOD, "precision": GATE["precision"] - 0.001})
    assert gate_failures({**GOOD, "false_critical": 1})
    msg = gate_failures({**GOOD, "cache_misses": 1})
    assert msg and "промах" in msg[0]  # пропавший файл кэша не проходит молча


def test_committed_cache_is_clean():
    """В репозиторий попадают только ответы модели: без ключей, путей и посторонних полей."""
    files = sorted(CACHE.glob("*.json"))
    assert 50 <= len(files) <= 300 and sum(f.stat().st_size for f in files) < 5_000_000
    secret = re.compile(r"AIza[0-9A-Za-z_-]{20,}|[A-Za-z]:\\|Users[\\/]|scratchpad|/home/")
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert not secret.search(text), f.name
        data = json.loads(text)
        assert set(data) == {"role", "model", "parsed"}, f.name
        assert re.fullmatch(r"[0-9a-f]{64}\.json", f.name)
