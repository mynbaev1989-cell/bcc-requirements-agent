"""Аблация: вклад каждого агента в точность. Работает только на кэше LLM, в сеть не ходит и квоту не тратит.

Запуск (после `python tools/eval_score.py eval`, `... eval_holdout` и `... eval_blind`, которые наполнили кэш):
    python tools/ablation.py [--model gemini-3.5-flash-lite]

Варианты прогона на тех же размеченных документах:
- full          — полный конвейер;
- no_critic     — без перепроверки находок (Critic);
- no_grounding  — без критика и без сверки цитат: находки LLM с цитатой, которой нет в документе, остаются.
                  Сравнивать с no_critic: так вклад сверки не смешивается с промахами кэша критика
                  (больше находок → другой промпт критика → промах кэша);
- no_critic_no_consistency — no_critic + ConsistencyReviewer без прохода «противоречия» (проход «пропуски» остаётся);
- no_critic_no_gaps        — no_critic + ConsistencyReviewer без прохода «пропуски» (остаётся «противоречия»);
                  оба сравнивать с no_critic;
- rules_only    — только детерминированные правила, без LLM (`--offline`).
Промах кэша = сбой LLM без запроса; такой шаг уходит в запасной вариант, и это видно в колонке «промахов кэша».
Итог печатается таблицей Markdown и сохраняется в eval/ablation.json.

Ворота качества (CI): ответы модели, которые использует полный конвейер на 7 документах, лежат в репозитории
(eval/llm_cache/). Тогда точность пересчитывается в CI без сети и без ключа:
    python tools/ablation.py --cache eval/llm_cache --gate
падает (код 1), если найдено меньше GATE["found"] дефектов, средняя по документам точность ниже GATE["precision"],
есть ложные critical вне эталона и *.extra.json или хоть один промах кэша (иначе пропавший файл тихо ушёл бы в запасной вариант).
Обновить закоммиченный кэш после живого перезамера: python tools/ablation.py --export-cache eval/llm_cache
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GEMINI_API_KEY", "cache-only")  # ключ не используется: запросов в сеть нет

from reqcheck import agents, steps  # noqa: E402
from reqcheck.llm import GeminiClient, LLMError  # noqa: E402
from reqcheck.pipeline import Options  # noqa: E402
from tools.eval_score import GATE, gate_failures, score_document  # noqa: E402,F401
from tools.stats import intervals  # noqa: E402

FOLDERS = ("eval", "eval_holdout", "eval_blind")
CACHE_DIR = ROOT / ".reqcheck_cache"


class _NoNetwork:
    """Подменный клиент Gemini: любой запрос мимо кэша сразу падает."""

    class models:  # noqa: N801
        @staticmethod
        def generate_content(**_):
            raise RuntimeError("промах кэша: в режиме аблации сеть отключена")


class _RecordingClient(GeminiClient):
    """Клиент только на кэше, который запоминает файлы кэша, давшие ответ (для --export-cache)."""

    used: set[Path]

    def _from_cache(self, role, payload):
        hit, parsed = super()._from_cache(role, payload)
        if hit:
            for model in self._chain:
                path = self._cache_path(payload, model)
                if path and path.exists():
                    self.used.add(path)
                    break
        return hit, parsed


def _cached_llm(model: str) -> GeminiClient:
    llm = _RecordingClient(model=model, retries=0, cache_dir=str(CACHE_DIR))
    llm.used = set()
    llm._client = _NoNetwork()
    _USED.append(llm.used)
    return llm


_USED: list[set[Path]] = []


def _keep_ungrounded(self, ctx):  # подмена GroundStep.run: ничего не отбрасывать
    return list(ctx.findings), 0


def _consistency_without(role: str):
    """Подмена ConsistencyReviewer.run: проход `role` («consistency» или «gaps») получает сбой LLM.
    Сбой одного прохода ревьюер штатно переживает: берёт результат второго (оценки набора при сбое «противоречий» пустые)."""
    original = agents.ConsistencyReviewer.run

    def run(self, reqs):
        real = self.llm.generate_json

        def generate_json(r, *a, **kw):
            if r == role:
                raise LLMError(f"проход «{role}» выключен в аблации")
            return real(r, *a, **kw)

        self.llm.generate_json = generate_json
        try:
            return original(self, reqs)
        finally:
            del self.llm.generate_json  # снова виден метод класса

    return run


def _list_unmatched(model: str) -> None:
    """Печатает находки агента, не совпавшие с эталоном (для ручной разметки extra.json). Только кэш."""
    for folder in FOLDERS:
        for lf in sorted((ROOT / folder).glob("*.labels.json")):
            labels = json.loads(lf.read_text(encoding="utf-8"))
            md = lf.with_name(labels.get("document") or lf.name.replace(".labels.json", ".md"))
            row = score_document(md, labels, Options(formats=()), llm=_cached_llm(model))
            print(f"\n## {folder}/{md.name}")
            for u in row.get("unmatched", []):
                print(f"{u['fid']} | {u['severity']} | строки {u['lines']} | {u['title']} | «{u['quote'][:120]}»")


def run_variant(name: str, model: str) -> dict:
    original = steps.GroundStep.run
    original_cons = agents.ConsistencyReviewer.run
    if name == "no_grounding":
        steps.GroundStep.run = _keep_ungrounded
    elif name == "no_critic_no_consistency":
        agents.ConsistencyReviewer.run = _consistency_without("consistency")
    elif name == "no_critic_no_gaps":
        agents.ConsistencyReviewer.run = _consistency_without("gaps")
    try:
        rows = []
        for folder in FOLDERS:
            for lf in sorted((ROOT / folder).glob("*.labels.json")):
                labels = json.loads(lf.read_text(encoding="utf-8"))
                md = lf.with_name(labels.get("document") or lf.name.replace(".labels.json", ".md"))
                opts = Options(formats=(), critic=name == "full", offline=name == "rules_only")
                llm = None if name == "rules_only" else _cached_llm(model)
                row = score_document(md, labels, opts, llm=llm)
                row["cache_misses"] = llm.stats.failures if llm else 0
                rows.append(row)
    finally:
        steps.GroundStep.run = original
        agents.ConsistencyReviewer.run = original_cons
    gold = sum(r["gold_defects"] for r in rows)
    return {
        "variant": name,
        "found": sum(r["found"] for r in rows), "gold": gold,
        "findings": sum(r["agent_findings"] for r in rows),
        "precision": round(sum(r["precision"] for r in rows) / len(rows), 3),
        "precision_items": round(sum(r.get("precision_items", r["precision"]) for r in rows) / len(rows), 3),
        "false_major_on_clean": sum(r["false_major_on_clean"] for r in rows),
        "severe_covered": sum(r["severe_covered_by_questions"] for r in rows),
        "severe_gold": sum(r["severe_gold"] for r in rows),
        "severe_question_coverage": round(sum(r["severe_covered_by_questions"] for r in rows) / max(sum(r["severe_gold"] for r in rows), 1), 3),
        "false_critical": sum(r["false_critical"] for r in rows),
        "false_critical_unconfirmed": sum(r.get("false_critical_unconfirmed", r["false_critical"]) for r in rows),
        "severity_exact": round(sum(r["severity_exact_n"] for r in rows) / max(sum(r["found"] for r in rows), 1), 3),

        "severity_closest": round(sum(r["severity_closest_n"] for r in rows) / max(sum(r["found"] for r in rows), 1), 3),
        "severity_within_one": round(sum(r["severity_within_one_n"] for r in rows) / max(sum(r["found"] for r in rows), 1), 3),
        "precision_with_extra": round(sum(r["precision_with_extra"] for r in rows) / len(rows), 3),
        "cache_misses": sum(r["cache_misses"] for r in rows),
        "dropped_ungrounded": sum(r.get("dropped_ungrounded", 0) for r in rows),
        "merged_by_critic": sum(r.get("merged_by_critic", 0) for r in rows),
        "rejected_by_critic": sum(r.get("rejected_by_critic", 0) for r in rows),
        # слито кодом: повторы между агентами + находки «Единичность», ушедшие в сводную (N находок → 1)
        "merged_overlap": sum(r.get("merged_overlap", 0) + max(r.get("merged_singular", 0) - 1, 0) for r in rows),
        "docs": rows,
    }


def export_cache(model: str, dest: Path) -> int:
    """Скопировать в dest файлы кэша, которые использует полный конвейер (только они нужны воротам)."""
    import shutil

    _USED.clear()
    r = run_variant("full", model)
    if r["cache_misses"]:
        print(f"Промахов кэша {r['cache_misses']}: сначала живой перезамер (tools/eval_score.py), потом экспорт.")
        return 1
    used = set().union(*_USED) if _USED else set()
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.json"):
        if re.fullmatch(r"[0-9a-f]{64}", old.stem) and old not in {dest / p.name for p in used}:
            old.unlink()  # чужие json в папке назначения не трогаем
    for p in sorted(used):
        shutil.copyfile(p, dest / p.name)
    size = sum(f.stat().st_size for f in dest.glob("*.json"))
    print(f"Скопировано файлов кэша: {len(used)}, {size / 1e6:.1f} МБ → {dest}")
    return 0


def main() -> int:
    global CACHE_DIR
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--list-unmatched", action="store_true", help="напечатать находки агента, не совпавшие с эталоном")
    ap.add_argument("--cache", type=Path, default=None, help="папка кэша ответов модели (по умолчанию .reqcheck_cache)")
    ap.add_argument("--gate", action="store_true", help="только полный конвейер; код 1, если метрики ниже порогов")
    ap.add_argument("--export-cache", type=Path, default=None, metavar="DIR",
                    help="скопировать в DIR файлы кэша, нужные полному конвейеру")
    args = ap.parse_args()
    if args.cache:
        CACHE_DIR = args.cache.resolve()
    if args.export_cache:
        return export_cache(args.model, args.export_cache.resolve())
    if args.gate:
        r = run_variant("full", args.model)
        print(f"Ворота качества: найдено {r['found']}/{r['gold']}, точность {r['precision']:.1%}, "
              f"ложных critical {r['false_critical']} (вне extra.json {r['false_critical_unconfirmed']}), промахов кэша {r['cache_misses']}, "
              f"критичность точно {r['severity_exact']:.0%}")
        bad = gate_failures(r)
        for b in bad:
            print(f"  ✗ {b}")
        print("Ворота пройдены." if not bad else "Ворота НЕ пройдены.")
        return 1 if bad else 0
    if args.list_unmatched:
        _list_unmatched(args.model)
        return 0
    variants = ("full", "no_critic", "no_grounding", "no_critic_no_consistency", "no_critic_no_gaps", "rules_only")
    results = [run_variant(v, args.model) for v in variants]
    print("| Вариант | Найдено дефектов | Находок | Точность | Точность по пунктам | Ложные major на чистых | Отсеяно сверкой | "
          "Слито критиком | Отклонено критиком | Слито кодом | Промахов кэша | Вопросами покрыто critical/major | "
          "Ложные critical | Критичность точно | Критичность ±1 | Точность с extra |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        print(f"| {r['variant']} | {r['found']}/{r['gold']} | {r['findings']} | {r['precision']:.0%} | {r['precision_items']:.0%} | "
              f"{r['false_major_on_clean']} | {r['dropped_ungrounded']} | {r['merged_by_critic']} | "
              f"{r['rejected_by_critic']} | {r['merged_overlap']} | {r['cache_misses']} | "
              f"{r['severe_covered']}/{r['severe_gold']} | {r['false_critical']} | {r['severity_exact']:.0%} | "
              f"{r['severity_within_one']:.0%} | {r['precision_with_extra']:.0%} |")
    full = results[0]
    full["ci95"] = intervals(full["docs"])
    ci = full["ci95"]
    print("95%-интервалы полного конвейера (bootstrap по 7 документам, 10 000 повторов): "
          + "; ".join(f"{k} {v['value']:.0%} [{v['lo']:.0%}–{v['hi']:.0%}]" for k, v in ci.items()))
    out = ROOT / "eval" / "ablation.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Результаты: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
