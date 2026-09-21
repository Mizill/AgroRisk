"""
webapp.py
=========
Локальный веб-интерфейс поверх того же конвейера.

Сервер построен на стандартной библиотеке (`http.server`), поэтому никаких
новых зависимостей не появляется: Flask, FastAPI и прочее не нужны.
Страница `web/index.html` полностью автономна — ни одного обращения к
внешним CDN, так что интерфейс открывается и без интернета (данные,
разумеется, без сети не придут).

API:
    GET  /                      страница
    GET  /api/status            версия, найденная модель, прошлая точка
    GET  /api/places?q=...      подсказки по названию (геокодер Open-Meteo)
    POST /api/run               запустить расчёт → {"job": "..."}
    GET  /api/job?id=...        ход расчёта и результат
    GET  /api/report.csv?id=... выгрузка отчёта

Расчёт идёт в отдельном потоке, а страница опрашивает `/api/job` и
показывает те же шаги, что печатает консольная версия.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from data_sources import OpenMeteoClient, OpenMeteoError
from location import find_model, load_settings, resolve_place, save_settings
from main import VERSION, AnalysisParams, run_analysis

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
INDEX_PATH = os.path.join(WEB_DIR, "index.html")
MAX_JOBS = 12


# ---------------------------------------------------------------------------
# Преобразование результата в JSON
# ---------------------------------------------------------------------------
def _json_safe(value):
    """NaN, numpy-числа и даты → то, что переживает json.dumps."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if (math.isnan(number) or math.isinf(number)) else round(number, 4)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    return value


def report_to_payload(report: pd.DataFrame) -> list:
    """Декадная таблица → список словарей, готовых к отправке в браузер."""
    return [{key: _json_safe(value) for key, value in row.items()}
            for row in report.to_dict("records")]


# ---------------------------------------------------------------------------
# Задания
# ---------------------------------------------------------------------------
class Job:
    """Один расчёт: журнал шагов, результат или ошибка."""

    def __init__(self, job_id: str):
        self.id = job_id
        self.state = "running"          # running | done | error
        self.log: list = []
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.report: Optional[pd.DataFrame] = None
        self.lock = threading.Lock()

    def append(self, message: str):
        with self.lock:
            self.log.append(message)

    def snapshot(self) -> dict:
        with self.lock:
            return {"id": self.id, "state": self.state, "log": list(self.log),
                    "result": self.result, "error": self.error}


class AgroRiskApp:
    """
    Логика веб-оболочки без HTTP-обвязки: так её можно проверить
    в тестах напрямую, подставив оффлайн-клиент.
    """

    def __init__(self, client_factory=None, model_path: Optional[str] = None,
                 end_date: Optional[dt.date] = None, remember: bool = True):
        self.client_factory = client_factory or (lambda: OpenMeteoClient())
        self.model_path = model_path if model_path is not None else find_model(None)
        self.end_date = end_date
        self.remember = remember
        self.jobs: dict = {}
        self.lock = threading.Lock()

    # -- справочные вызовы ------------------------------------------------
    def status(self) -> dict:
        settings = load_settings() if self.remember else {}
        return {
            "version": VERSION,
            "model": os.path.basename(self.model_path) if self.model_path else None,
            "model_path": self.model_path,
            "last_place": {
                "query": settings.get("title") or "",
                "lat": settings.get("lat"), "lon": settings.get("lon"),
            } if settings.get("lat") is not None else None,
        }

    def places(self, query: str) -> list:
        if not (query or "").strip():
            return []
        hits = self.client_factory().search_place(query, count=6)
        return [{"name": hit.get("name"), "admin1": hit.get("admin1"),
                 "country": hit.get("country"),
                 "lat": _json_safe(hit.get("latitude")),
                 "lon": _json_safe(hit.get("longitude"))}
                for hit in hits]

    # -- расчёт -----------------------------------------------------------
    def start(self, request: dict) -> str:
        job = Job(uuid.uuid4().hex[:12])
        with self.lock:
            self.jobs[job.id] = job
            for stale in list(self.jobs)[:-MAX_JOBS]:
                self.jobs.pop(stale, None)
        threading.Thread(target=self._work, args=(job, request), daemon=True).start()
        return job.id

    def job(self, job_id: str) -> Optional[dict]:
        job = self.jobs.get(job_id)
        return job.snapshot() if job else None

    def csv(self, job_id: str) -> Optional[bytes]:
        job = self.jobs.get(job_id)
        if job is None or job.report is None:
            return None
        return job.report.to_csv(index=False).encode("utf-8-sig")

    def _work(self, job: Job, request: dict):
        try:
            client = self.client_factory()
            job.append("Ищу точку расчёта...")
            place = self._resolve(client, request)
            job.append(f"Точка: {place.describe()}")

            use_model = not bool(request.get("no_model"))
            model_path = self.model_path if use_model else None
            if use_model and not model_path:
                job.append("Файл pinn_model.pt не найден — считаю без нейросети.")

            farm = (request.get("farm") or "").strip() or place.title or "Хозяйство"
            params = AnalysisParams(
                lat=place.lat, lon=place.lon, farm=farm,
                history_days=_as_int(request.get("history_days"), 60, 10, 3650),
                calibration_days=_as_int(request.get("calibration_days"), 730, 60, 7300),
                climatology_years=_as_int(request.get("climatology_years"), 15, 1, 40),
                forecast_days=_as_int(request.get("forecast_days"), 10, 0, 16),
                model_path=model_path,
                backend=request.get("backend") if request.get("backend") in
                ("auto", "torch", "numpy") else "auto",
                pinn_calibration=request.get("pinn_calibration") if
                request.get("pinn_calibration") in ("climatology", "sigmoid", "raw")
                else "climatology",
                first_snow_doy=_as_int(request.get("first_snow_doy"), 300, 150, 366),
                sukhovey_min_hours=_as_int(request.get("sukhovey_min_hours"), 3, 1, 24),
                cover=_as_float(request.get("cover"), None, 0.0, 1.0),
                end_date=self.end_date,
            )

            report = run_analysis(client, params, verbose=False,
                                  on_progress=job.append)
            job.report = report

            decades = report_to_payload(report)
            latest = decades[-1] if decades else None
            job.result = {
                "place": {"title": place.title, "lat": place.lat, "lon": place.lon},
                "farm": farm,
                "model": os.path.basename(model_path) if model_path else None,
                "generated_at": dt.datetime.now().strftime("%d.%m.%Y %H:%M"),
                "decades": decades,
                "latest": latest,
                "forecast_days": params.forecast_days,
            }
            job.state = "done"
            job.append("Готово.")

            if self.remember:
                save_settings({"lat": place.lat, "lon": place.lon,
                               "title": place.title, "farm": farm})
        except OpenMeteoError as exc:
            job.state, job.error = "error", f"Open-Meteo не ответил: {exc}"
        except (ValueError, KeyError, RuntimeError) as exc:
            job.state, job.error = "error", str(exc)
        except Exception as exc:                                   # noqa: BLE001
            job.state, job.error = "error", f"{type(exc).__name__}: {exc}"

    def _resolve(self, client, request: dict):
        from location import Place
        lat, lon = _as_float(request.get("lat")), _as_float(request.get("lon"))
        if lat is not None and lon is not None:
            return Place(lat=lat, lon=lon, title=(request.get("query") or "").strip())
        query = (request.get("query") or "").strip()
        if not query:
            raise ValueError("Укажите населённый пункт или координаты.")
        return resolve_place(client, query)


def _as_int(value, default=None, low=None, high=None):
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return default
    if low is not None:
        number = max(low, number)
    if high is not None:
        number = min(high, number)
    return number


def _as_float(value, default=None, low=None, high=None):
    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    if low is not None:
        number = max(low, number)
    if high is not None:
        number = min(high, number)
    return number


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    server_version = f"AgroRisk/{VERSION}"
    app: AgroRiskApp = None            # подставляется в create_server

    def log_message(self, *args):      # без шума в консоли
        pass

    # -- ответы -----------------------------------------------------------
    def _send(self, code: int, body: bytes, content_type: str,
              extra_headers: Optional[dict] = None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload, code: int = 200):
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    # -- маршруты ---------------------------------------------------------
    def do_GET(self):
        route = urlparse(self.path)
        query = parse_qs(route.query)

        if route.path in ("/", "/index.html"):
            try:
                with open(INDEX_PATH, "rb") as fh:
                    body = fh.read()
            except OSError:
                return self._send(500, b"web/index.html not found", "text/plain; charset=utf-8")
            return self._send(200, body, "text/html; charset=utf-8")

        if route.path == "/api/status":
            return self._json(self.app.status())

        if route.path == "/api/places":
            try:
                return self._json({"places": self.app.places(query.get("q", [""])[0])})
            except OpenMeteoError as exc:
                return self._json({"places": [], "error": str(exc)}, 200)

        if route.path == "/api/job":
            snapshot = self.app.job(query.get("id", [""])[0])
            return self._json(snapshot) if snapshot else \
                self._json({"error": "Расчёт не найден."}, 404)

        if route.path == "/api/report.csv":
            job_id = query.get("id", [""])[0]
            body = self.app.csv(job_id)
            if body is None:
                return self._json({"error": "Отчёт ещё не готов."}, 404)
            return self._send(200, body, "text/csv; charset=utf-8",
                              {"Content-Disposition":
                               f'attachment; filename="risk_report_{job_id}.csv"'})

        return self._json({"error": "Нет такого адреса."}, 404)

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        route = urlparse(self.path)
        if route.path != "/api/run":
            return self._json({"error": "Нет такого адреса."}, 404)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError
        except (ValueError, TypeError):
            return self._json({"error": "Не разобрать запрос."}, 400)
        return self._json({"job": self.app.start(payload)})


def create_server(app: AgroRiskApp, host: str = "127.0.0.1", port: int = 8000):
    """Собирает сервер. `port=0` — свободный порт (используется в тестах)."""
    handler = type("BoundHandler", (_Handler,), {"app": app})
    return ThreadingHTTPServer((host, port), handler)


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True,
          app: Optional[AgroRiskApp] = None) -> int:
    """Запускает сервер и, если нужно, открывает браузер."""
    app = app or AgroRiskApp()
    for attempt in range(10):
        try:
            server = create_server(app, host, port + attempt)
            break
        except OSError:
            continue
    else:
        print(f"Порты {port}-{port + 9} заняты. Укажите свободный ключом --port.")
        return 1

    address = f"http://{host}:{server.server_address[1]}/"
    print(f"Агрориск {VERSION} — интерфейс открыт: {address}")
    print(f"Модель: {app.model_path or 'не найдена, расчёт без нейросети'}")
    print("Остановить — Ctrl+C")
    if open_browser:
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")
    finally:
        server.server_close()
    return 0
