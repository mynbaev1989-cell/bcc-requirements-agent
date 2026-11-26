"""Командная строка: python -m reqcheck <документ> <отчёт> [опции].

Коды выхода: 0 — проверка выполнена; 1 — документ не удалось проверить (отчёт с диагностикой записан);
2 — не удалось записать отчёт; 3 — неверные аргументы.
"""

from __future__ import annotations

import argparse
import sys

from reqcheck import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m reqcheck",
        description="ИИ-агент статической проверки документа требований (ISO/IEC/IEEE 29148, IEEE 830). "
                    "Пишет отчёт на русском в Markdown, PDF и JSON.",
    )
    p.add_argument("input", help="путь к документу требований (.md, .txt, .pdf, .docx)")
    p.add_argument("output", help="путь к файлу отчёта (.md или .pdf); рядом лягут остальные форматы")
    p.add_argument("--model", help="модель (Gemini: GEMINI_MODEL или gemini-3.5-flash; при REQCHECK_LLM_BASE_URL: REQCHECK_LLM_MODEL)")
    p.add_argument("--offline", action="store_true", help="без LLM: только детерминированные правила")
    p.add_argument("--formats", default="md,pdf,json", help="форматы отчёта через запятую (по умолчанию md,pdf,json)")
    p.add_argument("--max-size-mb", type=float, default=20.0, help="лимит размера входного файла, МБ (по умолчанию 20)")
    p.add_argument("--timeout", type=float, default=240.0, help="таймаут одного запроса к LLM, с (по умолчанию 240)")
    p.add_argument("--no-cache", action="store_true", help="не использовать кэш ответов LLM")
    p.add_argument("--no-critic", action="store_true", help="отключить перепроверку находок агентом-критиком")
    p.add_argument("--answers", help="файл с ответами автора на вопросы агента (<отчёт>_questions.md); "
                                     "агент учтёт ответы и пересчитает вердикт")
    p.add_argument("--version", action="version", version=f"reqcheck {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    # Консоль Windows: печатаем UTF-8 без падений на кириллице
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 3

    from reqcheck.pipeline import Options, run, write_outputs

    formats = tuple(f.strip().lower() for f in args.formats.split(",") if f.strip())
    opts = Options(model=args.model, offline=args.offline, formats=formats,
                   max_bytes=int(max(args.max_size_mb, 0.001) * 1024 * 1024), timeout=max(args.timeout, 5.0),
                   use_cache=not args.no_cache, critic=not args.no_critic, answers=args.answers)
    print(f"reqcheck {__version__}: проверяю «{args.input}»…")
    result = run(args.input, opts)
    written, problems = write_outputs(result, args.output, formats)
    for d in result.diagnostics + problems:
        print(f"  [{d.level}] {d.message}" + (f" → {d.hint}" if d.hint else ""))
    if not written:
        print("Ошибка: отчёт не записан.", file=sys.stderr)
        return 2
    m = result.metrics
    print(f"Требований: {m.get('requirements', 0)}; находок: {m.get('findings', 0)} "
          f"(критичных {m.get('critical', 0)}, существенных {m.get('major', 0)}); время: {m.get('duration_sec', 0)} с")
    print(f"Разработка: {result.verdict_dev}")
    if result.questions:
        print(f"Вопросов автору: {m.get('questions', 0)} (открытых блокирующих: {m.get('questions_blocker_open', 0)})"
              + (f"; учтено ответов: {m.get('answers_used', 0)}, закрыто находок: {m.get('resolved_by_answers', 0)}"
                 if m.get("answers_used") else ""))
    print("Отчёт: " + ", ".join(written))
    failed = result.document is None or not result.requirements
    return 1 if failed else 0
