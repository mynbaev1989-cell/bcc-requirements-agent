"""Аблация: вклад каждого агента в точность. Работает только на кэше LLM, в сеть не ходит и квоту не тратит.

Запуск (после `python tools/eval_score.py eval` и `... eval_holdout`, которые наполнили кэш):
    python tools/ablation.py [--model gemini-3.5-flash-lite]

Варианты прогона на тех же размеченных документах:
- full          — полный конвейер;
- no_critic     — без перепроверки находок (Critic);
- no_grounding  — без критика и без сверки цитат: находки LLM с цитатой, которой нет в документе, остаются.
                  Сравнивать с no_critic: так вклад сверки не смешивается с промахами кэша критика
                  (больше находок → другой промпт критика → промах кэша);
- rules_only    — только детерминированные правила, без LLM (`--offline`).
Промах кэша = сбой LLM без запроса; такой шаг уходит в запасной вариант, и это видно в колонке «промахов кэша».
Итог печатается таблицей Markdown и сохраняется в eval/ablation.json.
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

from reqcheck import steps  # noqa: E402
from reqcheck.llm import GeminiClient  # noqa: E402
from reqcheck.pipeline import Options  # noqa: E402
from tools.eval_score import score_document  # noqa: E402

FOLDERS = ("eval", "eval_holdout")


class _NoNetwork:
    """Подменный клиент Gemini: любой запрос мимо кэша сразу падает."""

    class models:  # noqa: N801
        @staticmethod
        def generate_content(**_):
            raise RuntimeError("промах кэша: в режиме аблации сеть отключена")


def _cached_llm(model: str) -> GeminiClient:
    llm = GeminiClient(model=model, retries=0, cache_dir=str(ROOT / ".reqcheck_cache"))
    llm._client = _NoNetwork()
    return llm


def _keep_ungrounded(self, ctx):  # подмена GroundStep.run: ничего не отбрасывать
    return list(ctx.findings), 0


def run_variant(name: str, model: str) -> dict:
    original = steps.GroundStep.run
    if name == "no_grounding":
        steps.GroundStep.run = _keep_ungrounded
    try:
        rows = []
        for folder in FOLDERS:
            for lf in sorted((ROOT / folder).glob("*.labels.json")):
                labels = json.loads(lf.read_text(encoding="utf-8"))
                md = lf.with_name(labels.get("document") or lf.name.replace(".labels.json", ".md"))
                opts = Options(formats=(), critic=name not in ("no_critic", "no_grounding"), offline=name == "rules_only")
                llm = None if name == "rules_only" else _cached_llm(model)
                row = score_document(md, labels, opts, llm=llm)
                row["cache_misses"] = llm.stats.failures if llm else 0
                rows.append(row)
    finally:
        steps.GroundStep.run = original
    gold = sum(r["gold_defects"] for r in rows)
    return {
        "variant": name,
        "found": sum(r["found"] for r in rows), "gold": gold,
        "findings": sum(r["agent_findings"] for r in rows),
        "precision": round(sum(r["precision"] for r in rows) / len(rows), 3),
        "false_major_on_clean": sum(r["false_major_on_clean"] for r in rows),
        "cache_misses": sum(r["cache_misses"] for r in rows),
        "dropped_ungrounded": sum(r.get("dropped_ungrounded", 0) for r in rows),
        "merged_by_critic": sum(r.get("merged_by_critic", 0) for r in rows),
        "rejected_by_critic": sum(r.get("rejected_by_critic", 0) for r in rows),
        "docs": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    args = ap.parse_args()
    results = [run_variant(v, args.model) for v in ("full", "no_critic", "no_grounding", "rules_only")]
    print("| Вариант | Найдено дефектов | Находок | Точность | Ложные major на чистых | Отсеяно сверкой | "
          "Слито критиком | Отклонено критиком | Промахов кэша |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        print(f"| {r['variant']} | {r['found']}/{r['gold']} | {r['findings']} | {r['precision']:.0%} | "
              f"{r['false_major_on_clean']} | {r['dropped_ungrounded']} | {r['merged_by_critic']} | "
              f"{r['rejected_by_critic']} | {r['cache_misses']} |")
    out = ROOT / "eval" / "ablation.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Результаты: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
