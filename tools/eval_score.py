"""Замер точности агента на размеченных документах (eval/*.labels.json).

Запуск:  python tools/eval_score.py eval
Для каждого документа eval/<имя>.md агент запускается, его находки сопоставляются с разметкой:
находка агента «попала» в эталонный дефект, если хотя бы одна её цитата лежит в пределах ±1 строки
от строки дефекта (или совпадает текст цитаты). Считаются recall (доля найденных эталонных дефектов),
precision (доля находок агента, совпавших с разметкой) и ложные срабатывания на «чистых» требованиях.

Дополнительно: критичность (точно / ±1 уровень), ложные critical, покрытие вопросами critical/major дефектов,
качество вопросов и precision_with_extra (учитывает <имя>.extra.json — дефекты, подтверждённые вручную).
"""

from __future__ import annotations

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reqcheck.grounding import Grounder, normalize  # noqa: E402
from reqcheck.pipeline import Options, run  # noqa: E402

SEV_RANK = {"critical": 2, "major": 1, "minor": 0}
SEVERE = ("critical", "major")
_LINE_IN_TEXT = re.compile(r"(строк\w*|стр\.?|line)\s*№?\s*\d", re.IGNORECASE)


# Ворота качества (tools/ablation.py --gate, шаг CI): текущий результат полного конвейера на 7 документах —
# 100 из 104, средняя точность 85,8%. Ложным critical ворота считают только находку, которой нет ни в эталоне,
# ни в подтверждённых вручную дефектах <имя>.extra.json: реальный дефект вне разметки — не ложная тревога. Промах кэша недопустим:
# без ответа модели шаг тихо уходит в запасной вариант, и метрики почти не меняются (проверено: 86,0% вместо 86,1%).
GATE = {"found": 100, "precision": 0.85, "false_critical_unconfirmed": 0, "cache_misses": 0}


def gate_failures(r: dict) -> list[str]:
    """Нарушенные пороги ворот для итога варианта full (пустой список — ворота пройдены)."""
    out = []
    if r["found"] < GATE["found"]:
        out.append(f"найдено {r['found']} < {GATE['found']}")
    if r["precision"] < GATE["precision"]:
        out.append(f"точность {r['precision']:.1%} < {GATE['precision']:.0%}")
    if r["false_critical_unconfirmed"] > GATE["false_critical_unconfirmed"]:
        out.append(f"ложных critical вне эталона и extra.json: {r['false_critical_unconfirmed']}")
    if r["cache_misses"] > GATE["cache_misses"]:
        out.append(f"промахов кэша {r['cache_misses']}: ответа модели нет в кэше, шаг ушёл в запасной вариант")
    return out


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


def load_extra(md_path: Path) -> dict:
    """Необязательный файл <имя>.extra.json рядом с документом: реальные дефекты вне эталона."""
    p = md_path.parent / f"{md_path.stem}.extra.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def question_quality(questions: list) -> dict:
    """Качество вопросов: доли с адресатом, вариантами, допущением, номером строки; число пар-дублей."""
    n = len(questions)

    def share(k: int) -> float:
        return round(k / n, 3) if n else 0.0

    groups: dict[frozenset, int] = {}
    for q in questions:
        if q.lines:
            key = frozenset(q.lines)
            groups[key] = groups.get(key, 0) + 1
    blockers = [q for q in questions if q.priority == "blocker"]
    texts = [normalize(q.text or "") for q in questions]
    return {
        "n": n,
        "with_addressee": share(sum(1 for q in questions if (q.addressee or "").strip())),
        "with_options": share(sum(1 for q in questions if q.options)),
        "with_assumption": share(sum(1 for q in questions if (q.assumption or "").strip())),
        "with_line": share(sum(1 for q in questions if q.lines or _LINE_IN_TEXT.search(q.text or ""))),
        "duplicate_pairs": sum(c * (c - 1) // 2 for c in groups.values()),  # одинаковый набор строк
        # Вопрос-блокер без вариантов ответа автор закрывает дольше: ему приходится формулировать решение самому
        "blockers": len(blockers),
        "blockers_with_options": round(sum(1 for q in blockers if q.options) / len(blockers), 3) if blockers else 0.0,
        # Адресат не «Аналитик» (значение по умолчанию): вопрос направлен тому, кто реально решает
        "routed_addressee": share(sum(1 for q in questions if (q.addressee or "").strip() not in ("", "Аналитик"))),
        # Почти одинаковые формулировки (сходство текста > 0,8) — дубль, даже если строки разные
        "text_duplicate_pairs": sum(1 for i in range(n) for j in range(i + 1, n)
                                    if SequenceMatcher(None, texts[i], texts[j]).ratio() > 0.8),
    }


def _closest(defect: dict, dl: set[int], matches: list):
    """Находка, которая по смыслу про этот дефект: ближайшая по строке, при равенстве — с самой похожей цитатой."""
    gq = normalize(defect.get("quote") or "")

    def key(item):
        f, fl = item
        sim = max((SequenceMatcher(None, gq, normalize(q.text)).ratio() for q in f.quotes), default=0.0)
        return (-min(abs(x - y) for x in fl for y in dl), sim)

    return max(matches, key=key)[0]


def score_document(md_path: Path, labels: dict, opts: Options, llm=None, extra: dict | None = None) -> dict:
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
    # Строже: сводная находка (collapse_singular) считается по пунктам — каждая её цитата отдельно,
    # иначе одно совпавшее замечание закрыло бы несовпавшие соседние
    items = [{q.line} for f, fl in agent if f.grouped for q in f.quotes if q.line]
    items += [fl for f, fl in agent if not f.grouped]
    matched_items = [il for il in items if any(near(il, dl) for _, dl in gold)]
    clean_lines = set()
    for q in labels.get("clean_requirements", []):
        loc = g.locate(q)
        if loc.found and loc.line:
            clean_lines.add(loc.line)
    false_on_clean = [f for f, fl in agent if fl & clean_lines and f.severity in ("critical", "major")]
    by_type: dict[str, list[int]] = {}
    for d, _ in gold:
        t = d.get("type", "other")
        by_type.setdefault(t, [0, 0])
        by_type[t][1] += 1
        if d in hit_gold:
            by_type[t][0] += 1
    # Уточнение: доля найденных эталонных дефектов, по которым агент задал автору вопрос
    asked = {fid for q in result.questions for fid in q.fids}
    gold_asked = [d for d, dl in gold if any(near(dl, fl) and f.fid in asked for f, fl in agent)]

    # Критичность: для найденного эталонного дефекта берём максимальную критичность среди совпавших находок
    sev_exact = sev_one = severe_gold = severe_asked = sev_closest = 0
    severity_pairs = []  # (id дефекта, критичность по разметке, по агенту) — для κ Коэна (tools/kappa.py)
    for d, dl in gold:
        matches = [f for f, fl in agent if near(dl, fl)]
        if not matches:
            continue
        gold_sev = d.get("severity", "major")
        if gold_sev in SEVERE:
            severe_gold += 1
            if any(f.fid in asked for f in matches):
                severe_asked += 1
        agent_rank = max(SEV_RANK.get(f.severity, 0) for f in matches)
        # Строже к агенту — старшая критичность среди всех находок рядом (так считалась заголовочная цифра).
        # Точнее по смыслу — критичность ближайшей находки (строка, затем сходство цитаты с эталонной):
        # у половины дефектов рядом есть находки про соседний дефект с другой критичностью.
        best = _closest(d, dl, [(f, fl) for f, fl in agent if near(dl, fl)])
        severity_pairs.append({"defect": d.get("id"), "gold": gold_sev,
                               "agent": next(k for k, v in SEV_RANK.items() if v == agent_rank),
                               "agent_closest": best.severity})
        sev_closest += best.severity == gold_sev
        diff = abs(agent_rank - SEV_RANK.get(gold_sev, 1))
        sev_exact += diff == 0
        sev_one += diff <= 1
    false_critical = [f for f, fl in agent if f.severity == "critical" and not any(near(fl, dl) for _, dl in gold)]

    # Дефекты, подтверждённые вручную (extra): precision_with_extra
    if extra is None:
        extra = load_extra(md_path)
    extra_lines = [_lines_of(e, g) for e in extra.get("extra_defects", []) if isinstance(e, dict)]
    matched_extra = [f for f, fl in agent
                     if any(near(fl, dl) for _, dl in gold) or any(near(fl, el) for el in extra_lines)]
    # Ложный critical, которого нет и среди подтверждённых дефектов вне разметки (по нему считают ворота CI)
    false_critical_unconfirmed = [f for f, fl in agent if any(f is x for x in false_critical)
                                  and not any(near(fl, el) for el in extra_lines)]
    unmatched = [{"fid": f.fid, "severity": f.severity, "lines": sorted(fl), "title": f.title,
                  "quote": f.quotes[0].text if f.quotes else ""}
                 for f, fl in agent if not any(near(fl, dl) for _, dl in gold)]
    return {
        "document": md_path.name,
        "gold_defects": len(gold),
        "found": len(hit_gold),
        "recall": round(len(hit_gold) / max(len(gold), 1), 3),
        "agent_findings": len(agent),
        "precision": round(len(matched_agent) / max(len(agent), 1), 3),
        "precision_with_extra": round(len(matched_extra) / max(len(agent), 1), 3),
        "extra_defects": len(extra_lines),
        "precision_items": round(len(matched_items) / max(len(items), 1), 3),
        "false_major_on_clean": len(false_on_clean),
        "questions": len(result.questions),
        "questions_per_found_defect": round(len(result.questions) / max(len(hit_gold), 1), 3),
        "gold_covered_by_questions": len(gold_asked),
        "question_coverage": round(len(gold_asked) / max(len(gold), 1), 3),
        "severe_gold": severe_gold,
        "severe_covered_by_questions": severe_asked,
        "severe_question_coverage": round(severe_asked / max(severe_gold, 1), 3),
        "severity_exact": round(sev_exact / max(len(hit_gold), 1), 3),
        "severity_within_one": round(sev_one / max(len(hit_gold), 1), 3),
        "severity_exact_n": sev_exact,
        "severity_closest_n": sev_closest,
        "severity_within_one_n": sev_one,
        "false_critical": len(false_critical),
        "false_critical_unconfirmed": len(false_critical_unconfirmed),
        "severity_pairs": severity_pairs,
        "question_quality": question_quality(result.questions),
        "by_type": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_type.items())},
        "missed": [d.get("explanation") or d.get("quote") for d in defects if d not in hit_gold],
        "unmatched": unmatched,
        "model": result.passport.get("model"),
        "dropped_ungrounded": result.metrics.get("dropped_ungrounded", 0),
        "merged_by_critic": result.metrics.get("merged_duplicates", 0),
        "merged_overlap": result.metrics.get("merged_overlap", 0),
        "merged_singular": result.metrics.get("merged_singular", 0),
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
              f"средняя precision {sum(r['precision'] for r in ok) / len(ok):.0%} "
              f"(сводные находки по пунктам: {sum(r['precision_items'] for r in ok) / len(ok):.0%}); "
              f"ложных major/critical на чистых требованиях: {sum(r['false_major_on_clean'] for r in ok)}; "
              f"эталонных дефектов, по которым задан вопрос автору: "
              f"{sum(r['gold_covered_by_questions'] for r in ok)}/{tot_gold}; "
              f"вопросов всего: {sum(r['questions'] for r in ok)}")
        sev_gold = sum(r["severe_gold"] for r in ok)
        qn = sum(r["question_quality"]["n"] for r in ok)

        def qshare(key: str) -> str:
            return f"{sum(r['question_quality'][key] * r['question_quality']['n'] for r in ok) / max(qn, 1):.0%}"

        print(f"Критичность: точно {sum(r['severity_exact_n'] for r in ok)}/{tot_found} "
              f"(по ближайшей находке {sum(r['severity_closest_n'] for r in ok)}/{tot_found}), "
              f"±1 уровень {sum(r['severity_within_one_n'] for r in ok)}/{tot_found}; "
              f"ложных critical: {sum(r['false_critical'] for r in ok)}; "
              f"critical/major дефектов, покрытых вопросом: {sum(r['severe_covered_by_questions'] for r in ok)}/{sev_gold}; "
              f"precision с учётом extra: {sum(r['precision_with_extra'] for r in ok) / len(ok):.0%}")
        print(f"Качество вопросов ({qn}): адресат {qshare('with_addressee')}, варианты {qshare('with_options')}, "
              f"допущение {qshare('with_assumption')}, строка {qshare('with_line')}; "
              f"пар-дублей: {sum(r['question_quality']['duplicate_pairs'] for r in ok)}")
        bl = sum(r["question_quality"]["blockers"] for r in ok)
        bl_opt = sum(r["question_quality"]["blockers_with_options"] * r["question_quality"]["blockers"] for r in ok)
        print(f"Уточнение: вопросов на найденный дефект {qn / max(tot_found, 1):.2f}; блокеров с вариантами "
              f"{bl_opt / max(bl, 1):.0%} ({round(bl_opt)}/{bl}); адресат не «Аналитик» {qshare('routed_addressee')}; "
              f"почти одинаковых формулировок: {sum(r['question_quality']['text_duplicate_pairs'] for r in ok)}")
    out = folder / "eval_results.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Результаты: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
