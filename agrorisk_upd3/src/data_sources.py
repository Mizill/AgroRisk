"""
data_sources.py
================
ЕДИНСТВЕННЫЙ источник данных в проекте — Open-Meteo (https://open-meteo.com).

ERA5-Land/Copernicus CDS и Kazhydromet удалены полностью: ключи не нужны,
внешних регистраций не требуется, все 13 признаков нейросети закрываются
одним API.

Что откуда берётся:

  * почасовой архив (archive-api.open-meteo.com) — температура, относительная
    влажность, осадки, снег, ветер, порывы и ОБЪЁМНАЯ ВЛАЖНОСТЬ ПОЧВЫ
    (soil_moisture_0_to_7cm). Именно влажность почвы раньше требовала ERA5-Land;
    архив Open-Meteo построен на ERA5/ERA5-Land, поэтому тот же параметр
    доступен без ключа;
  * суточный архив за N лет — только осадки и средняя температура, нужны для
    климатической нормы по декадам (лёгкий запрос);
  * почасовой прогноз (api.open-meteo.com) — чтобы досчитать текущую
    и следующую декаду вперёд;
  * геокодер (geocoding-api.open-meteo.com) — поиск координат по названию
    населённого пункта, чтобы их не приходилось вводить руками.

Особенности, которые здесь учтены:

  1. Единицы задаются явно (`wind_speed_unit=ms`). По умолчанию Open-Meteo
     отдаёт ветер в км/ч — старый код считал их метрами в секунду и завышал
     скорость ветра в 3.6 раза.
  2. Архив отстаёт от реального времени примерно на 5 суток, поэтому
     `archive_end_date()` берёт дату с запасом.
  3. Набор доступных переменных у архива и прогноза различается
     (например, влажность почвы в прогнозе разбита на слои 0-1/1-3/3-9 см).
     Клиент умеет отбрасывать переменную, которую API не принял, и повторять
     запрос — см. `_request_with_fallback`.
"""
from __future__ import annotations

import datetime as dt
import time
from typing import Iterable, Optional, Sequence

ARCHIVE_LAG_DAYS = 6


class OpenMeteoError(RuntimeError):
    """Ошибка обращения к Open-Meteo (сеть, лимит, неверные параметры)."""


def archive_end_date(today: Optional[dt.date] = None) -> dt.date:
    """Последняя дата, которая гарантированно есть в архиве."""
    today = today or dt.date.today()
    return today - dt.timedelta(days=ARCHIVE_LAG_DAYS)


class OpenMeteoClient:
    """
    Клиент Open-Meteo. Ключ не нужен.

    `session` — любой объект с методом `.get(url, params=..., timeout=...)`
    (по умолчанию `requests.Session`). Через него в тестах подставляется
    оффлайн-заглушка, поэтому весь конвейер проверяется без интернета.
    """

    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

    # Почасовые переменные архива: закрывают признаки 5-12 нейросети.
    HOURLY_ARCHIVE = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "snowfall",
        "wind_speed_10m",
        "wind_gusts_10m",
        "soil_moisture_0_to_7cm",
    ]

    # В прогнозе влажность почвы разбита на другие слои — сводим к 0-9 см.
    HOURLY_FORECAST = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "snowfall",
        "wind_speed_10m",
        "wind_gusts_10m",
        "soil_moisture_0_to_1cm",
        "soil_moisture_1_to_3cm",
        "soil_moisture_3_to_9cm",
    ]

    # Для климатической нормы хватает двух суточных переменных.
    DAILY_CLIMATOLOGY = ["precipitation_sum", "temperature_2m_mean"]

    BASE_PARAMS = {
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }

    def __init__(self, timeout: int = 90, retries: int = 3,
                 retry_pause: float = 2.0, session=None):
        self.timeout = timeout
        self.retries = retries
        self.retry_pause = retry_pause
        if session is None:
            import requests
            session = requests.Session()
        self.session = session

    # ------------------------------------------------------------------
    # транспорт
    # ------------------------------------------------------------------
    def _get_json(self, url: str, params: dict) -> dict:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except Exception as exc:                      # сетевой сбой
                last_err = exc
                if attempt < self.retries:
                    time.sleep(self.retry_pause * attempt)
                    continue
                raise OpenMeteoError(f"Не удалось обратиться к {url}: {exc}") from exc

            status = getattr(resp, "status_code", 200)
            try:
                payload = resp.json()
            except Exception:
                payload = {}

            if status == 200 and not payload.get("error"):
                return payload

            reason = payload.get("reason") or getattr(resp, "text", "")
            # 429/5xx — временные, их имеет смысл повторить
            if status in (429, 500, 502, 503, 504) and attempt < self.retries:
                last_err = OpenMeteoError(f"HTTP {status}: {reason}")
                time.sleep(self.retry_pause * attempt)
                continue
            raise OpenMeteoError(f"Open-Meteo вернул ошибку (HTTP {status}): {reason}")

        raise OpenMeteoError(str(last_err))

    def _request_with_fallback(self, url: str, params: dict,
                               key: str, variables: Sequence[str]) -> dict:
        """
        Делает запрос и, если API не принял какую-то переменную, убирает её
        из списка и повторяет. Возвращает ответ; отсутствующие переменные
        просто не появятся в ответе (дальше по конвейеру станут NaN).
        """
        wanted = list(variables)
        for _ in range(len(wanted) + 1):
            if not wanted:
                raise OpenMeteoError(
                    f"Open-Meteo не принял ни одной переменной из набора '{key}'.")
            attempt_params = dict(params)
            attempt_params[key] = ",".join(wanted)
            try:
                return self._get_json(url, attempt_params)
            except OpenMeteoError as exc:
                bad = [v for v in wanted if v in str(exc)]
                if not bad:
                    raise
                wanted = [v for v in wanted if v not in bad]
        raise OpenMeteoError(f"Не удалось подобрать набор переменных '{key}'.")

    # ------------------------------------------------------------------
    # публичные методы
    # ------------------------------------------------------------------
    def get_hourly_history(self, lat: float, lon: float,
                           start: dt.date, end: dt.date,
                           variables: Optional[Iterable[str]] = None) -> dict:
        """Почасовой архив за период [start, end] включительно."""
        params = dict(self.BASE_PARAMS,
                      latitude=lat, longitude=lon,
                      start_date=start.isoformat(), end_date=end.isoformat())
        return self._request_with_fallback(
            self.ARCHIVE_URL, params, "hourly",
            list(variables) if variables else self.HOURLY_ARCHIVE)

    def get_hourly_forecast(self, lat: float, lon: float, days: int = 10,
                            variables: Optional[Iterable[str]] = None) -> dict:
        """Почасовой прогноз на ближайшие дни (максимум 16)."""
        params = dict(self.BASE_PARAMS,
                      latitude=lat, longitude=lon,
                      forecast_days=max(1, min(int(days), 16)),
                      past_days=0)
        return self._request_with_fallback(
            self.FORECAST_URL, params, "hourly",
            list(variables) if variables else self.HOURLY_FORECAST)

    def search_place(self, name: str, count: int = 5,
                     language: str = "ru") -> list:
        """
        Поиск населённого пункта по названию (геокодер Open-Meteo, ключ не нужен).
        Возвращает список словарей с name, latitude, longitude, country, admin1.
        Нужен, чтобы не вводить координаты вручную.
        """
        name = (name or "").strip()
        if not name:
            return []
        payload = self._get_json(self.GEOCODING_URL, {
            "name": name, "count": max(1, min(int(count), 20)),
            "language": language, "format": "json",
        })
        results = payload.get("results") or []
        return [r for r in results
                if r.get("latitude") is not None and r.get("longitude") is not None]

    def get_daily_history(self, lat: float, lon: float,
                          start: dt.date, end: dt.date,
                          variables: Optional[Iterable[str]] = None) -> dict:
        """Суточный архив — используется для климатической нормы."""
        params = dict(self.BASE_PARAMS,
                      latitude=lat, longitude=lon,
                      start_date=start.isoformat(), end_date=end.isoformat())
        return self._request_with_fallback(
            self.ARCHIVE_URL, params, "daily",
            list(variables) if variables else self.DAILY_CLIMATOLOGY)

    def get_climatology(self, lat: float, lon: float,
                        reference_years: int = 15,
                        end: Optional[dt.date] = None) -> dict:
        """Суточный ряд за N полных лет для расчёта декадных норм."""
        end = end or archive_end_date()
        start = dt.date(end.year - int(reference_years), 1, 1)
        return self.get_daily_history(lat, lon, start, end)
