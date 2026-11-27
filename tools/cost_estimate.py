"""Оценка объёма запросов к LLM на документ — по закоммиченному кэшу, без сети и без ключа.

Запуск:  python tools/cost_estimate.py [--cache eval/llm_cache] [--model gemini-3.5-flash-lite]
Полный конвейер прогоняется на 7 размеченных документах только на кэше; для каждого ответа из кэша
считаются символы промпта (system + prompt) и символы ответа (JSON). Токены — ОЦЕНКА: точного счётчика
токенов без API нет, берём 3–4 символа на токен (русский текст с разметкой). Цену за токен этот скрипт
не знает — её и дату берите со страницы цен провайдера (см. README, раздел «Бизнес-ценность»).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GEMINI_API_KEY", "cache-only")  # ключ не используется: запросов в сеть нет

from reqcheck.llm import GeminiClient  # noqa: E402
from reqcheck.pipeline import Options, run  # noqa: E402
from tools.ablation import FOLDERS, _NoNetwork  # noqa: E402

CHARS_PER_TOKEN = (3.0, 4.0)


class _MeasuringClient(GeminiClient):
    """Клиент только на кэше, который считает символы промптов и ответов, взятых из кэша."""

    def _from_cache(self, role, payload):
        hit, parsed = super()._from_cache(role, payload)
        if hit:
            self.calls.append((role, len(payload), len(json.dumps(parsed, ensure_ascii=False))))
        else:
            self.misses += 1
        return hit, parsed


def measure(md: Path, model: str, cache: Path) -> dict:
    llm = _MeasuringClient(model=model, retries=0, cache_dir=str(cache))
    llm.calls, llm.misses = [], 0
    llm._client = _NoNetwork()
    run(str(md), Options(formats=()), llm=llm)
    return {"document": md.name, "requests": len(llm.calls), "cache_misses": llm.misses,
            "prompt_chars": sum(c[1] for c in llm.calls), "output_chars": sum(c[2] for c in llm.calls)}


def tokens(chars: float) -> tuple[int, int]:
    """Диапазон токенов для числа символов: (при 4 символах на токен, при 3)."""
    return round(chars / CHARS_PER_TOKEN[1]), round(chars / CHARS_PER_TOKEN[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "eval" / "llm_cache"))
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    args = ap.parse_args()
    rows = []
    for folder in FOLDERS:
        for lf in sorted((ROOT / folder).glob("*.labels.json")):
            labels = json.loads(lf.read_text(encoding="utf-8"))
            md = lf.with_name(labels.get("document") or lf.name.replace(".labels.json", ".md"))
            rows.append(measure(md, args.model, Path(args.cache)))
            r = rows[-1]
            print(f"{r['document']}: запросов {r['requests']}, промахов {r['cache_misses']}, "
                  f"промпт {r['prompt_chars']:,} симв., ответ {r['output_chars']:,} симв.".replace(",", " "))
    n = len(rows)
    req = sum(r["requests"] for r in rows) / n
    pin = sum(r["prompt_chars"] for r in rows) / n
    pout = sum(r["output_chars"] for r in rows) / n
    print(f"\nВ среднем на документ: {req:.1f} запросов; вход ≈ {tokens(pin)[0]:,}–{tokens(pin)[1]:,} токенов, "
          f"выход ≈ {tokens(pout)[0]:,}–{tokens(pout)[1]:,} токенов (оценка: 3–4 символа на токен)".replace(",", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
