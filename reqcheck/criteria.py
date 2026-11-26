"""Справочник критериев качества.

Источники (раздел 5 ТЗ кейса):
* ISO/IEC/IEEE 29148:2018, п. 5.2.5 — характеристики отдельного требования;
* ISO/IEC/IEEE 29148:2018, п. 5.2.6 — характеристики набора требований;
* IEEE Std 830-1998, п. 4.3 — характеристики хорошей спецификации (4.3.1–4.3.8).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Criterion:
    code: str
    name_ru: str
    source: str
    question: str  # что проверяем — используется в промптах и в отчёте
    critical_category: str | None = None  # группа критичных дефектов для сводки (§6 ТЗ)


# Критерии для каждого требования
REQUIREMENT_CRITERIA: list[Criterion] = [
    Criterion("necessary", "Необходимость", "29148 п. 5.2.5 (Necessary)",
              "Требование нужно: без него не будет выполнена цель или ограничение; не дублирует другое."),
    Criterion("appropriate", "Уместность (без лишних проектных решений)", "29148 п. 5.2.5 (Appropriate)",
              "Уровень детализации соответствует уровню документа; нет навязанных решений по реализации без обоснования."),
    Criterion("unambiguous", "Однозначность", "29148 п. 5.2.5 (Unambiguous); IEEE 830 п. 4.3.2",
              "Формулировку можно понять только одним способом; нет размытых слов и неопределённых терминов."),
    Criterion("complete", "Полнота", "29148 п. 5.2.5 (Complete); IEEE 830 п. 4.3.3",
              "Описаны все условия, ответы на корректный и некорректный ввод, граничные случаи; нет «TBD».",
              critical_category="неполные"),
    Criterion("singular", "Единичность (атомарность)", "29148 п. 5.2.5 (Singular)",
              "Пункт содержит одно требование, а не несколько, связанных «и»/«а также»."),
    Criterion("feasible", "Осуществимость", "29148 п. 5.2.5 (Feasible)",
              "Требование реализуемо при известных ограничениях системы и окружения."),
    Criterion("verifiable", "Проверяемость", "29148 п. 5.2.5 (Verifiable); IEEE 830 п. 4.3.6",
              "Есть объективный способ проверить выполнение: измеримые значения, условия, ожидаемый результат.",
              critical_category="непроверяемые"),
    Criterion("correct", "Корректность", "29148 п. 5.2.5 (Correct); IEEE 830 п. 4.3.1",
              "Требование точно отражает потребность; не противоречит цели, другим требованиям и фактам."),
    Criterion("conforming", "Соответствие шаблону", "29148 п. 5.2.5 (Conforming)",
              "Требование оформлено по принятому шаблону/стилю документа, имеет идентификатор."),
    Criterion("traceable", "Трассируемость", "IEEE 830 п. 4.3.8",
              "У требования есть уникальный идентификатор и понятен источник; на него можно сослаться."),
]

# Критерии для набора требований целиком
SET_CRITERIA: list[Criterion] = [
    Criterion("set_complete", "Полнота набора", "29148 п. 5.2.6 (Complete); IEEE 830 п. 4.3.3",
              "Набор покрывает все нужные функции, данные, интерфейсы, ошибки, НФТ; нет пустых разделов и «TBD».",
              critical_category="неполные"),
    Criterion("set_consistent", "Согласованность", "29148 п. 5.2.6 (Consistent); IEEE 830 п. 4.3.4",
              "Требования не противоречат друг другу, критериям приёмки и связанным документам; единые термины и единицы.",
              critical_category="противоречивые"),
    Criterion("set_feasible", "Осуществимость набора", "29148 п. 5.2.6 (Feasible)",
              "Набор реализуем целиком в рамках ограничений (сроки, архитектура, интеграции)."),
    Criterion("set_comprehensible", "Понятность", "29148 п. 5.2.6 (Comprehensible)",
              "Понятно, что ожидается от системы и как она связана с окружением; термины определены."),
    Criterion("set_validatable", "Возможность валидации", "29148 п. 5.2.6 (Able to be validated)",
              "Можно доказать, что набор решает бизнес-цель: есть критерии приёмки и измеримые цели."),
    Criterion("ranked", "Ранжированность", "IEEE 830 п. 4.3.5",
              "У требований указана важность и/или стабильность (приоритеты)."),
    Criterion("modifiable", "Модифицируемость", "IEEE 830 п. 4.3.7",
              "Структура позволяет вносить изменения: версия, оглавление, нет дублирования."),
    Criterion("set_traceable", "Трассируемость набора", "IEEE 830 п. 4.3.8",
              "Требования связаны с целями, критериями приёмки, дочерними документами и интерфейсами."),
]

ALL = {c.code: c for c in REQUIREMENT_CRITERIA + SET_CRITERIA}
REQ_CODES = [c.code for c in REQUIREMENT_CRITERIA]
SET_CODES = [c.code for c in SET_CRITERIA]

# Отображение нарушений в группы критичных дефектов из §6 ТЗ («непроверяемые, противоречивые, неполные»)
CRITICAL_GROUP_BY_CRITERION = {
    "verifiable": "непроверяемые",
    "set_consistent": "противоречивые",
    "correct": "противоречивые",
    "complete": "неполные",
    "set_complete": "неполные",
}


def name(code: str) -> str:
    c = ALL.get(code)
    return c.name_ru if c else code


def source(code: str) -> str:
    c = ALL.get(code)
    return c.source if c else ""


def normalize_code(raw: str | None, scope: str) -> str:
    """Привести код критерия от LLM к известному; неизвестные — к ближайшему по смыслу."""
    code = (raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if code in ALL:
        if scope == "set" and code in REQ_CODES:
            mapped = {"complete": "set_complete", "correct": "set_consistent", "feasible": "set_feasible",
                      "traceable": "set_traceable", "unambiguous": "set_comprehensible"}.get(code)
            return mapped or code
        return code
    aliases = {
        "consistent": "set_consistent", "consistency": "set_consistent", "contradiction": "set_consistent",
        "completeness": "complete" if scope == "requirement" else "set_complete",
        "ambiguity": "unambiguous", "ambiguous": "unambiguous",
        "verifiability": "verifiable", "testable": "verifiable", "unverifiable": "verifiable",
        "atomic": "singular", "singularity": "singular",
        "design": "appropriate", "implementation_free": "appropriate",
        "comprehensible": "set_comprehensible", "validatable": "set_validatable",
        "able_to_be_validated": "set_validatable", "ranked_for_importance": "ranked",
    }
    if code in aliases:
        return aliases[code]
    return "set_consistent" if scope == "set" else "unambiguous"
