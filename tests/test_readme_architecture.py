"""Таблица агентов и схема в README описывают тот же конвейер, что выполняет код (build_plan)."""

import re
from pathlib import Path

from reqcheck.steps import build_plan

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _stages():
    """Стадии плана: список шагов каждой стадии (параллельные шаги — одна стадия)."""
    return [s if isinstance(s, list) else [s] for s in build_plan()]


def _table():
    """Строки таблицы «Стадия | Агент | …» из раздела «Как устроен агент»: [(номера стадий, ячейка агента)]."""
    section = README.split("## Как устроен агент", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        m = re.match(r"\|\s*(\d+)(?:[–-](\d+))?\s*\|\s*([^|]+?)\s*\|", line)
        if m:
            first, last = int(m.group(1)), int(m.group(2) or m.group(1))
            rows.append((list(range(first, last + 1)), m.group(3)))
    return rows


def _key(text: str) -> str:
    return re.split(r"[\s,(+·]", text.strip(), maxsplit=1)[0]


def test_table_covers_every_stage_in_order():
    numbers = [n for nums, _ in _table() for n in nums]
    assert numbers == list(range(1, len(_stages()) + 1))


def test_each_row_names_an_agent_of_its_stage():
    stages = _stages()
    for nums, cell in _table():
        steps = [s for n in nums for s in stages[n - 1]]
        keys = {_key(s.agent) for s in steps} | {_key(s.name) for s in steps}
        assert any(k in cell for k in keys), f"стадия {nums}: «{cell}» не совпадает с {keys}"
        uses_llm = any("(LLM)" in s.agent for s in steps)
        assert ("LLM" in cell) == uses_llm, f"стадия {nums}: пометка LLM/код расходится с кодом"


def test_parallel_stage_is_marked_parallel():
    for nums, cell in _table():
        if any(len(_stages()[n - 1]) > 1 for n in nums):
            assert "параллельно" in cell


def test_diagram_mentions_every_llm_agent():
    svg = (ROOT / "assets" / "readme" / "pipeline.svg").read_text(encoding="utf-8")
    agents = {_key(s.agent) for st in _stages() for s in st if "(LLM)" in s.agent}
    agents |= {"ConsistencyReviewer", "GapFinder"}  # два агента одного шага проверки набора
    missing = [a for a in sorted(agents) if a not in svg]
    assert not missing, f"на схеме нет агентов: {missing}"
