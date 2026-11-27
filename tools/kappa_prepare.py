"""Подготовка обезличенных пунктов для второго разметчика критичности (κ Коэна, tools/kappa.py).

Запуск:  python tools/kappa_prepare.py <папка_пунктов> <папка_ключа>
В <папка_пунктов>/items.json — 104 эталонных дефекта без критичности, без имени документа и без id разметки:
цитата, связанная цитата, пояснение, тип и строки документа вокруг цитаты. В <папка_ключа>/key.json —
соответствие K001… → документ, id дефекта, критичность по разметке. Разметчику дают только items.json
и eval/kappa/guideline.md; ключ хранится отдельно, чтобы его нельзя было подсмотреть.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("eval", "eval_holdout", "eval_blind")
CONTEXT = 2  # строк до и после цитаты


def _context(lines: list[str], quote: str, line: int | None) -> str:
    idx = next((i for i, s in enumerate(lines) if quote and quote[:40] in s), None)
    if idx is None and isinstance(line, int) and 0 < line <= len(lines):
        idx = line - 1
    if idx is None:
        return ""
    return "\n".join(lines[max(idx - CONTEXT, 0): idx + CONTEXT + 1])


def collect() -> list[dict]:
    items = []
    for folder in FOLDERS:
        for lf in sorted((ROOT / folder).glob("*.labels.json")):
            labels = json.loads(lf.read_text(encoding="utf-8"))
            md = lf.with_name(labels.get("document") or lf.name.replace(".labels.json", ".md"))
            lines = md.read_text(encoding="utf-8").splitlines()
            for d in labels.get("defects", []):
                items.append({
                    "document": md.name, "defect": d.get("id"), "gold": d.get("severity", "major"),
                    "type": d.get("type", ""), "quote": d.get("quote") or "",
                    "related_quote": d.get("related_quote") or "", "explanation": d.get("explanation") or "",
                    "context": _context(lines, d.get("quote") or "", d.get("line")),
                })
    return items


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 1
    items_dir, key_dir = Path(argv[1]), Path(argv[2])
    items = collect()
    random.Random(0).shuffle(items)  # порядок не выдаёт документ
    public, key = [], {}
    for n, it in enumerate(items, 1):
        kid = f"K{n:03d}"
        key[kid] = {"document": it["document"], "defect": it["defect"], "gold": it["gold"]}
        public.append({"id": kid, **{k: it[k] for k in ("type", "quote", "related_quote", "explanation", "context")}})
    items_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)
    (items_dir / "items.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
    (key_dir / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Пунктов: {len(public)}; items.json → {items_dir}, key.json → {key_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
