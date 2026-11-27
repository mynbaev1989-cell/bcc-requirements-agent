"""Калибровка критичности кодом: шкала «критично» закреплена за конфликтами, а не за любой неясностью.

Замер (eval/README.md, «Калибровка критичности кодом»): агент завышал критичность — два независимых разметчика
ставили «существенный» там, где агент ставил «критичный», в основном у находок о непроверяемой формулировке,
неполноте, TBD и пропущенном разделе. По шкале Critic (prompts.CRITIC) такая неясность — major: её закрывает
уточнение без смены логики. Модель эту границу не держит, поэтому её держит код.

Правило: находка «критичный» по критериям «проверяемость», «однозначность», «полнота», «полнота набора»
понижается до «существенный», если в её заголовке и пояснении нет признаков конфликта двух утверждений
(противоречие, несоответствие, расхождение, конфликт). Противоречия, ошибки корректности, безопасность
и регуляторика не трогаются. Повысить критичность код не может.
"""

from __future__ import annotations

import re

from reqcheck.models import CRITICAL, MAJOR, Finding

CAPPED_CRITERIA = frozenset({"verifiable", "unambiguous", "complete", "set_complete"})
# «расходит/расхожд», а не «расход»: «расходы» — частое слово в банковских ТЗ
CONFLICT = re.compile(r"противореч|несоответств|расхожд|расходит|расходят|конфликт", re.IGNORECASE)
NOTE = "критичность понижена кодом до «существенный»: неясность без конфликта утверждений закрывается уточнением"


def needs_downgrade(f: Finding) -> bool:
    return (f.severity == CRITICAL and f.criterion in CAPPED_CRITERIA
            and not CONFLICT.search(f"{f.title} {f.explanation}"))


def calibrate(findings: list[Finding]) -> list[Finding]:
    """Понижает подходящие находки на месте; возвращает список пониженных."""
    changed = [f for f in findings if needs_downgrade(f)]
    for f in changed:
        f.severity = MAJOR
        f.critic = f"{f.critic}; {NOTE}" if f.critic else NOTE
    return changed
