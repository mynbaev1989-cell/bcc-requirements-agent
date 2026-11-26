"""Замер точности агента на размеченных документах (eval/*.labels.json).

Запуск:  python tools/eval_score.py eval
Для каждого документа eval/<имя>.md агент запускается, его находки сопоставляются с разметкой:
находка агента «попала» в эталонный дефект, если хотя бы одна её цитата лежит в пределах ±1 строки
от строки дефекта (или совпадает текст цитаты). Считаются recall (доля найденных эталонных дефектов),
precision (доля находок агента, совпавших с разметкой) и ложные срабатывания на «чистых» требованиях.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reqcheck.grounding import Grounder, normalize  # noqa: E402
from reqcheck.pipeline import Options, run  # noqa: E402


def _lines_of(defect: dict, grounder: Grounder) -> set[int]:
    lines = set()
    for key in ("quote", "related_quote"):
        q = defect.get(key)
        if q:
            loc = grounder.locate(q)
            if loc.found and loc.line:
                lines.add(loc.line)
    if isinstance(defect.get("line"), int):
        lines.add(defect["line"])
    return lines


def score_document(md_path: Path, labels: dict, opts: Options, llm=None) -> dict:
    result = run(str(md_path), opts, llm=llm)
    if result.document is None:
        return {"document": md_path.name, "error": [d.message for d in result.diagnostics]}
    g = Grounder(result.document)
    defects = labels.get("defects", [])
    gold = [(d, _lines_of(d, g)) for d in defects]
    agent = [(f, {q.line for q in f.quotes if q.line}) for f in result.findings]

    def near(a: set[int], b: set[int]) -> bool:
        return any(abs(x - y) <= 1 for x in a for y in b)

    hit_gold = [d for d, dl in gold if any(near(dl, fl) for _, fl in agent)]
    matched_agent = [f for f, fl in agent if any(near(fl, dl) for _, dl in gold)]
    clean_lines = set()
    for q in labels.get("clean_requirements", []):
        loc = g.locate(q)
        if loc.found and loc.line:
            clean_lines.add(loc.line)
    false_on_clean = [f for f, fl in agent if fl & clean_lines and f.severity in ("critical", "major")]
    by_type: dict[str, list[int]] = {}
    for d, dl in gold:
        t = d.get("type", "other")
        by_type.setdefault(t, [0, 0])
        by_type[t][1] += 1
        if d in hit_gold:
            by_type[t][0] += 1
    # Уточнение: доля найденных эталонных дефектов, по которым агент задал автору вопрос
    asked = {fid for q in result.questions for fid in q.fids}
    gold_asked = [d for d, dl in gold if any(near(dl, fl) and f.fid in asked for f, fl in agent)]
    return {
        "document": md_path.name,
        "gold_defects": len(gold),
        "found": len(hit_gold),
        "recall": round(len(hit_gold) / max(len(gold), 1), 3),
        "agent_findings": len(agent),
        "precision": round(len(matched_agent) / max(len(agent), 1), 3),
        "false_major_on_clean": len(false_on_clean),
        "questions": len(result.questions),
        "gold_covered_by_questions": len(gold_asked),
        "question_coverage": round(len(gold_asked) / max(len(gold), 1), 3),
        "by_type": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_type.items())},
        "missed": [d.get("explanation") or d.get("quote") for d in defects if d not in hit_gold],
        "model": result.passport.get("model"),
        "dropped_ungrounded": result.metrics.get("dropped_ungrounded", 0),
        "merged_by_critic": result.metrics.get("merged_duplicates", 0),
        "rejected_by_critic": result.metrics.get("dropped_by_critic", 0),
    }


def main(argv: list[str]) -> int:
    folder = Path(argv[1] if len(argv) > 1 else ROOT / "eval")
    files = sorted(folder.glob("*.labels.json"))
    if not files:
        print(f"Нет файлов разметки в {folder}")
        return 1
    rows = []
    for lf in files:
        labels = json.loads(lf.read_text(encoding="utf-8"))
        md = lf.with_name(labels.get("document") or lf.name.replace(".labels.json", ".md"))
        rows.append(score_document(md, labels, Options(formats=())))
        print(json.dumps(rows[-1], ensure_ascii=False, indent=2))
    ok = [r for r in rows if "recall" in r]
    if ok:
        tot_gold = sum(r["gold_defects"] for r in ok)
        tot_found = sum(r["found"] for r in ok)
        print(f"\nИТОГО: recall {tot_found}/{tot_gold} = {tot_found / max(tot_gold, 1):.0%}; "
              f"средняя precision {sum(r['precision'] for r in ok) / len(ok):.0%}; "
              f"ложных major/critical на чистых требованиях: {sum(r['false_major_on_clean'] for r in ok)}; "
              f"эталонных дефектов, по которым задан вопрос автору: "
              f"{sum(r['gold_covered_by_questions'] for r in ok)}/{tot_gold}; "
              f"вопросов всего: {sum(r['questions'] for r in ok)}")
    out = folder / "eval_results.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Результаты: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
