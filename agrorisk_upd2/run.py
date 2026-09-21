#!/usr/bin/env python3
"""
run.py — запуск одной командой из корня проекта:

    python run.py                 # спросит место, дальше всё само
    python run.py Акколь          # место прямо в команде
    python run.py "51.17 71.45"   # или координаты

Файл нужен, чтобы не набирать путь `src/main.py` и не настраивать PYTHONPATH.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
