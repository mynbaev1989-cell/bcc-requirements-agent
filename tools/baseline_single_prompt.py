"""Базовая линия «один промпт»: тот же документ целиком в ту же модель одним запросом — как если вставить ТЗ в чат.

Отвечает на вопрос «чем reqcheck лучше, чем вставить ТЗ в ChatGPT» цифрой: находки базовой линии сверяются
с той же разметкой (`*.labels.json`, `*.extra.json`) по тому же правилу, что и находки агента в tools/eval_score.py
(совпадение строк ±1). Модель, температура и документ с номерами строк — те же, что у агентов reqcheck.

    python tools/baseline_single_prompt.py                     # живой прогон (7 запросов) или ответы из кэша
    python tools/baseline_single_prompt.py --cache eval/llm_cache_baseline   # без сети, из закоммиченного кэша

Строка находки — строка её цитаты, найденной в документе кодом (Grounder); если цитаты в документе нет,
берётся номер строки, который назвала модель, и находка считается непроверенной (доля таких — отдельная метрика).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reqcheck.agents import numbered_document  # noqa: E402
from reqcheck.grounding import Grounder  # noqa: E402
from reqcheck.llm import GeminiClient  # noqa: E402
from reqcheck.loaders import load_document  # noqa: E402
from tools.eval_score import _lines_of, load_extra  # noqa: E402

FOLDERS = ("eval", "eval_holdout", "eval_blind")

# Обычный «хороший» промпт, который напишет аналитик: роль, критерии из ТЗ кейса (§6), формат ответа.
# Примеров из размеченных документов в нём нет.
SYSTEM = (
    "Ты опытный аналитик-ревьюер требований в банке. Проверь документ требований до начала разработки. "
    "Найди дефекты: неоднозначные и непроверяемые формулировки, неполноту (не описаны ошибки, граничные случаи, "
    "роли, статусы), противоречия между требованиями, критериями приёмки, схемами и спецификацией API, "
    "нереализуемые или некорректные требования, нарушения безопасности и регуляторики, пустые разделы. "
    "Критичность: critical — противоречие или ошибка, из-за которой разработка пойдёт неверно; major — неполнота или "
    "неоднозначность, которую нужно закрыть до тест-дизайна; minor — оформление. "
    "Для каждого дефекта дай дословную цитату из документа и номер строки (строки пронумерованы), "
    "и вопрос автору требований. Ответь только JSON: "
    '{"findings": [{"title": "...", "severity": "critical|major|minor", "quote": "...", "line": 0, '
    '"explanation": "...", "question": "..."}]}'
)


def documents() -> list[Path]:
    return [p for folder in FOLDERS for p in sorted((ROOT / folder).glob("*.md"))
            if p.with_suffix(".labels.json").exists()]


def score(md: Path, findings: list[dict]) -> dict:
    doc, _ = load_document(md)
    g = Grounder(doc)
    labels = json.loads(md.with_suffix(".labels.json").read_text(encoding="utf-8"))
    gold = [_lines_of(d, g) for d in labels.get("defects", [])]
    extra = [_lines_of(e, g) for e in load_extra(md).get("extra_defects", []) if isinstance(e, dict)]

    def near(a: set[int], b: set[int]) -> bool:
        return any(abs(x - y) <= 1 for x in a for y in b)

    items, grounded = [], 0
    for f in findings:
        if not isinstance(f, dict):
            continue
        loc = g.locate(str(f.get("quote") or "")) if f.get("quote") else None
        if loc is not None and loc.found and loc.line:
            lines, grounded = {loc.line}, grounded + 1
        elif isinstance(f.get("line"), int) and f["line"] > 0:
            lines = {f["line"]}
        else:
            lines = set()
        items.append((str(f.get("severity") or "").lower(), lines, bool(str(f.get("question") or "").strip())))
    n = len(items)
    found = sum(1 for dl in gold if any(near(dl, fl) for _, fl, _ in items))
    matched = sum(1 for _, fl, _ in items if any(near(fl, dl) for dl in gold))
    matched_x = sum(1 for _, fl, _ in items if any(near(fl, dl) for dl in gold + extra))
    false_crit = [fl for s, fl, _ in items if s == "critical" and not any(near(fl, dl) for dl in gold)]
    return {
        "document": md.name, "gold_defects": len(gold), "found": found, "findings": n,
        "precision": round(matched / n, 3) if n else 0.0,
        "precision_with_extra": round(matched_x / n, 3) if n else 0.0,
        "false_critical": len(false_crit),
        "false_critical_unconfirmed": sum(1 for fl in false_crit if not any(near(fl, el) for el in extra)),
        "quotes_found_in_document": round(grounded / n, 3) if n else 0.0,
        "with_question": round(sum(1 for *_, q in items if q) / n, 3) if n else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--cache", type=Path, default=ROOT / ".reqcheck_cache", help="папка кэша ответов модели")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "baseline_single_prompt.json")
    args = ap.parse_args()
    llm = GeminiClient(model=args.model, cache_dir=str(args.cache))
    rows = []
    for md in documents():
        doc, _ = load_document(md)
        data = llm.generate_json("baseline", SYSTEM, numbered_document(doc))
        found = data.get("findings", []) if isinstance(data, dict) else data if isinstance(data, list) else []
        rows.append(score(md, found if isinstance(found, list) else []))
    tot = {k: sum(r[k] for r in rows) for k in ("gold_defects", "found", "findings", "false_critical",
                                                "false_critical_unconfirmed")}
    for k in ("precision", "precision_with_extra", "quotes_found_in_document", "with_question"):
        tot[k] = round(sum(r[k] for r in rows) / len(rows), 3)
    out = {"model": args.model, "requests": len(rows), "total": tot, "docs": rows}
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Один промпт ({args.model}): найдено {tot['found']}/{tot['gold_defects']}, находок {tot['findings']}, "
          f"точность {tot['precision']:.0%} (с extra {tot['precision_with_extra']:.0%}), "
          f"ложных critical {tot['false_critical']} (вне extra {tot['false_critical_unconfirmed']}), "
          f"цитаты найдены в документе {tot['quotes_found_in_document']:.0%}, с вопросом {tot['with_question']:.0%}")
    for r in rows:
        print(f"  {r['document']}: {r['found']}/{r['gold_defects']}, находок {r['findings']}, "
              f"точность {r['precision']:.0%}, ложных critical {r['false_critical']}")
    print(f"Результаты: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
