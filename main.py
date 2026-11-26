"""Запуск агента одной командой из корня репозитория:

    python main.py --input input/requirements.md --output reports/requirements_report.md

Рядом с отчётом лягут PDF, JSON и файл вопросов автору (<отчёт>_questions.md). Все опции: python main.py --help.
То же самое: python -m reqcheck input/requirements.md reports/requirements_report.md
"""

import sys

from reqcheck.cli import main

if __name__ == "__main__":
    sys.exit(main())
