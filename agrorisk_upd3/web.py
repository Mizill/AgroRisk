#!/usr/bin/env python3
"""
web.py — веб-интерфейс одной командой:

    python web.py

Открывает браузер на http://127.0.0.1:8000/ . Сервер локальный, на
стандартной библиотеке Python: ставить ничего дополнительно не нужно.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from webapp import serve  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(prog="web.py", description="Веб-интерфейс Агрориска")
    parser.add_argument("--port", type=int, default=8000, help="порт (по умолчанию 8000)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="адрес; 0.0.0.0 — открыть доступ из локальной сети")
    parser.add_argument("--no-browser", action="store_true", help="не открывать браузер")
    args = parser.parse_args()
    return serve(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    raise SystemExit(main())
