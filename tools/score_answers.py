"""Подсчёт: верно ли Resolver обработал ответы автора, которые писал не автор агента (слепой набор).

Для каждого документа в папке нужны: `<имя>_answers_key.json` (тип ответа и should_close — что должен решить агент;
пишет тот, кто отвечал) и `<имя>.json` — отчёт агента, собранный с `--answers <имя>_answers.md`.

Правило: конкретный ответ (should_close: true) обработан верно, если закрыл хотя бы одну находку своего вопроса;
уклончивый и противоречивый (should_close: false) — если не закрыл ни одной. Итог по вопросу берётся из отчёта
(поле outcome «закрыто находок: X из Y», его пишет код после решения Resolver).

    python tools/score_answers.py reports/independent_answers/blind2
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TYPES = ("concrete", "evasive", "contradictory")
TYPE_RU = {"concrete": "Конкретный", "evasive": "Уклончивый", "contradictory": "Противоречивый"}


def closed_count(outcome: str) -> int | None:
    m = re.search(r"закрыто находок:\s*(\d+)\s*из\s*\d+", outcome or "")
    return int(m.group(1)) if m else None


def score_folder(folder: Path) -> dict:
    docs = []
    for key_path in sorted(folder.glob("*_answers_key.json")):
        name = key_path.name[: -len("_answers_key.json")]
        key = json.loads(key_path.read_text(encoding="utf-8"))
        report_path = folder / f"{name}.json"
        if not report_path.exists():
            raise SystemExit(f"Нет отчёта {report_path}: соберите его с --answers {name}_answers.md")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        qmap = {q["qid"]: q for q in report.get("questions", [])}
        rows = []
        for a in key.get("answers", []):
            q = qmap.get(a["qid"])
            closed = closed_count(q.get("outcome", "")) if q else None
            ok = closed is not None and ((closed > 0) if a["should_close"] else (closed == 0))
            rows.append({"qid": a["qid"], "type": a["type"], "should_close": a["should_close"],
                         "closed": closed, "correct": ok})
        docs.append({"document": name, "answers": rows})
    total = {t: [sum(r["correct"] for d in docs for r in d["answers"] if r["type"] == t),
                 sum(1 for d in docs for r in d["answers"] if r["type"] == t)] for t in TYPES}
    return {"docs": docs, "by_type": total,
            "correct": sum(v[0] for v in total.values()), "answers": sum(v[1] for v in total.values())}


def main() -> int:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reports/independent_answers/blind2")
    res = score_folder(folder)
    (folder / "score.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for t in TYPES:
        ok, n = res["by_type"][t]
        print(f"{TYPE_RU[t]}: {ok} из {n}")
    print(f"Всего: {res['correct']} из {res['answers']}")
    for d in res["docs"]:
        bad = [f"{r['qid']} ({r['type']}, закрыто {r['closed']})" for r in d["answers"] if not r["correct"]]
        print(f"  {d['document']}: ошибки — {', '.join(bad) or 'нет'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
