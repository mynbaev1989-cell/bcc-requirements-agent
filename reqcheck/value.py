"""Оценка экономии времени и денег (измеримая бизнес-ценность).

Все параметры — явные допущения, их можно переопределить переменными окружения.
Фактическое время работы агента измеряется, а не предполагается.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class ValueEstimate:
    rate_kzt_per_hour: float
    manual_base_min: float
    manual_min_per_requirement: float
    check_min_per_finding: float
    manual_hours: float
    agent_hours: float
    check_hours: float
    saved_hours: float
    saved_kzt: float
    critical_found: int
    major_found: int = 0
    fix_hours_at_requirements: float = 0.0
    escalation_factor: float = 0.0
    precision: float = 1.0
    avoided_rework_hours: float = 0.0
    avoided_rework_kzt: float = 0.0
    tokens_total: int = 0
    llm_cost_kzt: float = 0.0
    llm_price_note: str = ""
    roi: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def estimate(n_requirements: int, n_findings: int, n_critical: int, agent_seconds: float,
             n_major: int = 0, tokens_in: int = 0, tokens_out: int = 0) -> ValueEstimate:
    rate = _env_float("REQCHECK_RATE_KZT", 5000.0)  # ставка аналитика/QA, ₸ в час (допущение)
    base = _env_float("REQCHECK_MANUAL_BASE_MIN", 30.0)  # прочитать документ целиком, оформить отчёт
    per_req = _env_float("REQCHECK_MANUAL_MIN_PER_REQ", 6.0)  # проверить требование по 10 критериям и записать замечание
    per_finding = _env_float("REQCHECK_CHECK_MIN_PER_FINDING", 1.5)  # человек проверяет находку агента
    manual_h = (base + per_req * n_requirements) / 60
    agent_h = agent_seconds / 3600
    check_h = per_finding * n_findings / 60
    saved_h = max(0.0, manual_h - agent_h - check_h)

    # Переделки, которых удалось избежать: дефект требований, найденный до разработки, правится за fix_h часов;
    # тот же дефект, найденный на этапе разработки, стоит в escalation раз дороже (NASA, 2010: 7–16× на этапе
    # кодирования, 21–78× на интеграции и тестах). Берём нижнюю границу этапа кодирования и только критичные/существенные.
    fix_h = _env_float("REQCHECK_FIX_HOURS", 1.0)
    esc = _env_float("REQCHECK_ESCALATION", 7.0)
    weight_major = _env_float("REQCHECK_MAJOR_WEIGHT", 0.5)  # существенный дефект доходит до кода не всегда
    # Не каждая находка — настоящий дефект: умножаем на измеренную точность находок (все размеченные документы, сводные находки по пунктам: 85%)
    precision = min(max(_env_float("REQCHECK_PRECISION", 0.85), 0.0), 1.0)
    avoided_h = (n_critical + weight_major * n_major) * precision * fix_h * max(esc - 1, 0)

    # Стоимость прогона LLM: токены × цена (по умолчанию 0 — бесплатный тариф AI Studio)
    price_in = _env_float("REQCHECK_PRICE_IN_USD_PER_M", 0.0)
    price_out = _env_float("REQCHECK_PRICE_OUT_USD_PER_M", 0.0)
    usd_kzt = _env_float("REQCHECK_USD_KZT", 500.0)
    llm_cost = (tokens_in * price_in + tokens_out * price_out) / 1_000_000 * usd_kzt
    note = ("бесплатный тариф (цена токенов не задана)" if not (price_in or price_out)
            else f"${price_in}/${price_out} за 1 млн токенов вход/выход, курс {usd_kzt:g} ₸/$")
    agent_cost = llm_cost + check_h * rate  # агент + время человека на проверку отчёта
    benefit = saved_h * rate + avoided_h * rate
    roi = round(benefit / agent_cost, 1) if agent_cost > 0 else None
    return ValueEstimate(rate, base, per_req, per_finding, round(manual_h, 2), round(agent_h, 3), round(check_h, 2),
                         round(saved_h, 2), round(saved_h * rate), n_critical, n_major, fix_h, esc, precision, round(avoided_h, 1),
                         round(avoided_h * rate), tokens_in + tokens_out, round(llm_cost, 1), note, roi)
