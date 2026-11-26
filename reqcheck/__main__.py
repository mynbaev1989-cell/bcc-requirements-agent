"""Точка входа: python -m reqcheck <документ> <отчёт>."""

import sys

from reqcheck.cli import main

if __name__ == "__main__":
    sys.exit(main())
