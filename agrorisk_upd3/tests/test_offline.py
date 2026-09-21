"""
tests/test_offline.py
=====================
Сквозная проверка всего конвейера БЕЗ интернета.

Вместо сетевого клиента подставляется `FakeSession`, которая отдаёт
синтетический, но реалистичный ответ Open-Meteo (годовой ход температуры,
осадков, ветра, влажности и влажности почвы для севера Казахстана).
Нейросеть при этом берётся настоящая — из `pinn_model.pt`.

Запуск:
    python tests/test_offline.py
    (или `pytest tests/test_offline.py`, если pytest установлен)
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import quote

import numpy as np
import pandas as pd

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from data_sources import OpenMeteoClient, OpenMeteoError          # noqa: E402
from decadal_risk import (                                         # noqa: E402
    decade_label, decade_length, decade_of, hourly_json_to_df, hourly_to_daily,
    sukhovey_subindex, early_snow_subindex,
)
from features import (                                             # noqa: E402
    FeatureConfig, build_feature_frame, feature_matrix, r_factor_cap,
    rainfall_erosivity,
)
from location import (                                              # noqa: E402
    Place, find_model, load_settings, parse_coordinates, place_from_settings,
    resolve_place, save_settings,
)
from main import (                                                 # noqa: E402
    AnalysisParams, parse_args, resolve_location, run_analysis,
)
from webapp import (                                                # noqa: E402
    AgroRiskApp, create_server, report_to_payload,
)
from pinn_model import (                                           # noqa: E402
    INPUT_DIM, INPUT_FEATURES, NumpyAgroPINN, TORCH_AVAILABLE, calibrate_outputs,
    load_model, load_raw_checkpoint,
)

MODEL_PATH = os.path.join(ROOT, "pinn_model.pt")
END_DATE = dt.date(2025, 9, 15)


# ===========================================================================
# Синтетический Open-Meteo
# ===========================================================================
def _seasonal(doy: int, mean: float, amplitude: float, peak_doy: int = 200) -> float:
    return mean + amplitude * math.cos(2 * math.pi * (doy - peak_doy) / 365.25)


def _hourly_payload(start: dt.date, end: dt.date, variables, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    times, data = [], {v: [] for v in variables}
    day = start
    while day <= end:
        doy = day.timetuple().tm_yday
        t_day = _seasonal(doy, 3.0, 19.0)                  # -16 °C зимой, +22 °C летом
        rh_day = _seasonal(doy, 65.0, -15.0)               # летом суше
        sm_day = max(0.05, _seasonal(doy, 0.24, 0.06, peak_doy=100))
        wet = rng.random() < (0.18 + 0.12 * math.sin(math.pi * doy / 365.25))
        day_precip = float(rng.gamma(1.6, 4.0)) if wet else 0.0
        heat_wave = 6.0 if (185 <= doy <= 215 and rng.random() < 0.35) else 0.0
        for hour in range(24):
            diurnal = 7.0 * math.sin(2 * math.pi * (hour - 9) / 24)
            temp = t_day + diurnal + heat_wave + float(rng.normal(0, 1.2))
            rh = float(np.clip(rh_day - 0.9 * diurnal - 2.2 * heat_wave
                               + rng.normal(0, 6), 8, 100))
            wind = float(np.clip(rng.gamma(2.2, 1.5) + 0.4 * heat_wave, 0, 28))
            precip = day_precip / 6.0 if 6 <= hour < 12 else 0.0
            times.append(f"{day.isoformat()}T{hour:02d}:00")
            for name in variables:
                if name == "temperature_2m":
                    data[name].append(round(temp, 1))
                elif name == "relative_humidity_2m":
                    data[name].append(round(rh, 1))
                elif name == "precipitation":
                    data[name].append(round(precip, 2))
                elif name == "rain":
                    data[name].append(round(precip if temp > 1 else 0.0, 2))
                elif name == "snowfall":
                    data[name].append(round(precip * 0.7 if temp <= 1 else 0.0, 2))
                elif name == "wind_speed_10m":
                    data[name].append(round(wind, 1))
                elif name == "wind_gusts_10m":
                    data[name].append(round(wind * 1.7, 1))
                elif name.startswith("soil_moisture"):
                    data[name].append(round(float(np.clip(sm_day + rng.normal(0, 0.012),
                                                          0.02, 0.48)), 3))
                else:
                    data[name].append(0.0)
        day += dt.timedelta(days=1)
    return {"latitude": 51.17, "longitude": 71.45, "hourly": {"time": times, **data}}


def _daily_payload(start: dt.date, end: dt.date, variables, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    times, data = [], {v: [] for v in variables}
    day = start
    while day <= end:
        doy = day.timetuple().tm_yday
        wet = rng.random() < (0.18 + 0.12 * math.sin(math.pi * doy / 365.25))
        times.append(day.isoformat())
        for name in variables:
            if name == "precipitation_sum":
                data[name].append(round(float(rng.gamma(1.6, 4.0)) if wet else 0.0, 2))
            elif name == "temperature_2m_mean":
                data[name].append(round(_seasonal(doy, 3.0, 19.0) + float(rng.normal(0, 2.5)), 1))
            else:
                data[name].append(0.0)
        day += dt.timedelta(days=1)
    return {"latitude": 51.17, "longitude": 71.45, "daily": {"time": times, **data}}


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload, self.status_code = payload, status_code
        self.text = str(payload)[:200]

    def json(self):
        return self._payload


class FakeSession:
    """Заглушка `requests.Session`, отдающая синтетические ответы Open-Meteo."""

    def __init__(self, reject: tuple = ()):
        self.reject = set(reject)      # переменные, которые «не поддерживает» API
        self.calls = []

    def get(self, url, params=None, timeout=None):
        params = params or {}
        self.calls.append((url, params))

        if "hourly" in params:
            requested = params["hourly"].split(",")
            bad = [v for v in requested if v in self.reject]
            if bad:
                return _Response({"error": True,
                                  "reason": f"Cannot initialize WeatherVariable "
                                            f"from invalid String value {bad[0]}"}, 400)
            if "start_date" in params:
                start = dt.date.fromisoformat(params["start_date"])
                end = dt.date.fromisoformat(params["end_date"])
            else:
                start = END_DATE + dt.timedelta(days=1)
                end = start + dt.timedelta(days=int(params.get("forecast_days", 7)) - 1)
            return _Response(_hourly_payload(start, end, requested))

        if "name" in params:
            query = params["name"].strip().lower()
            if query.startswith("акколь"):
                return _Response({"results": [
                    {"name": "Акколь", "latitude": 51.99, "longitude": 70.94,
                     "country": "Казахстан", "admin1": "Акмолинская область"},
                    {"name": "Акколь", "latitude": 45.00, "longitude": 75.18,
                     "country": "Казахстан", "admin1": "Жамбылская область"},
                ]})
            if query.startswith("астана"):
                return _Response({"results": [
                    {"name": "Астана", "latitude": 51.16, "longitude": 71.47,
                     "country": "Казахстан"}]})
            return _Response({"generationtime_ms": 0.1})

        if "daily" in params:
            requested = params["daily"].split(",")
            start = dt.date.fromisoformat(params["start_date"])
            end = dt.date.fromisoformat(params["end_date"])
            return _Response(_daily_payload(start, end, requested))

        return _Response({"error": True, "reason": "unexpected request"}, 400)


# ===========================================================================
# Проверки
# ===========================================================================
def test_decade_helpers():
    assert decade_of(dt.date(2025, 5, 3)) == 1
    assert decade_of(dt.date(2025, 5, 20)) == 2
    assert decade_of(dt.date(2025, 5, 21)) == 3
    assert decade_label(dt.date(2025, 5, 21)) == "2025-05-Д3"
    assert decade_length(dt.date(2025, 2, 21)) == 8      # невисокосный февраль
    assert decade_length(dt.date(2024, 2, 21)) == 9      # високосный
    assert decade_length(dt.date(2025, 12, 21)) == 11
    assert decade_length(dt.date(2025, 5, 1)) == 10


def test_checkpoint_loads_without_torch():
    ck = load_raw_checkpoint(MODEL_PATH)
    state = ck["model_state"]
    assert state["network.0.weight"].shape == (64, INPUT_DIM)
    assert state["network.16.weight"].shape == (1, 32)
    assert abs(float(ck["K_factor"]) - 0.3) < 1e-9
    assert abs(float(ck["max_rusle"]) - 4.286635875701904) < 1e-6
    assert len(INPUT_FEATURES) == INPUT_DIM == 13


def test_numpy_backend_forward():
    model, physics = load_model(MODEL_PATH, backend="numpy")
    assert isinstance(model, NumpyAgroPINN)
    assert physics["backend"] == "numpy"
    X = np.tile(np.linspace(0.1, 0.9, INPUT_DIM), (8, 1))
    X[:, 0] = np.linspace(0, 30, 8)          # R_factor в физических единицах
    out = model.forward(X)
    assert out.shape == (8,)
    assert np.all(np.isfinite(out))
    # батч и поэлементный прогон должны совпадать (BatchNorm в режиме eval)
    one_by_one = np.array([model.forward(X[i:i + 1])[0] for i in range(8)])
    assert np.allclose(out, one_by_one, atol=1e-9)


def test_torch_and_numpy_agree():
    if not TORCH_AVAILABLE:
        return "torch не установлен — сравнение бэкендов пропущено"
    numpy_model, _ = load_model(MODEL_PATH, backend="numpy")
    torch_model, _ = load_model(MODEL_PATH, backend="torch")
    X = np.random.default_rng(3).random((16, INPUT_DIM)) * np.linspace(1, 30, INPUT_DIM)
    assert np.allclose(numpy_model.forward(X), torch_model.forward(X), atol=1e-3)


def test_rainfall_erosivity_and_cap():
    assert rainfall_erosivity([2, 3, 4]) == 0.0                  # морось не эрозионна
    strong = rainfall_erosivity([25.0])
    split = rainfall_erosivity([12.5, 12.5])
    assert strong > split                                        # ливень > двух дождей
    physics = {"K_factor": 0.3, "LS_factor": 1.1907239757543053,
               "P_factor": 0.8, "max_rusle": 4.286635875701904}
    assert 25.0 < r_factor_cap(physics) < 35.0


def test_sukhovey_and_snow_subindices():
    assert sukhovey_subindex(0, 10, 3.0, 70.0) == 0.0     # тихо и влажно
    assert sukhovey_subindex(10, 10, 12.0, 5.0) > 90.0    # все сутки суховейные
    assert sukhovey_subindex(5, 10, 5.0, 40.0) == 30.0    # половина суток, без надбавок
    # сильный порыв в одни сутки не должен поднимать индекс всей декады
    assert sukhovey_subindex(0, 10, 3.5, 55.0) == 0.0
    early = early_snow_subindex(dt.date(2025, 9, 21), 4.0, -5.0, 300)
    late = early_snow_subindex(dt.date(2025, 11, 1), 4.0, -5.0, 300)
    assert early > late                                          # ранний снег опаснее
    assert early_snow_subindex(dt.date(2025, 5, 1), 0.0, 3.0, 300) == 0.0


def test_hourly_aggregation_and_units():
    payload = _hourly_payload(dt.date(2025, 7, 1), dt.date(2025, 7, 20),
                              OpenMeteoClient.HOURLY_ARCHIVE)
    daily = hourly_to_daily(hourly_json_to_df(payload))
    assert len(daily) == 20
    assert (daily["temp_max_c"] >= daily["temp_min_c"]).all()
    assert (daily["wind_max_ms"] >= daily["wind_mean_ms"]).all()
    assert daily["soil_moisture_frac"].between(0, 1).all()
    assert daily["sukhovey_day"].isin([0, 1]).all()


def test_forecast_soil_moisture_layers_merge():
    payload = _hourly_payload(dt.date(2025, 7, 1), dt.date(2025, 7, 3),
                              OpenMeteoClient.HOURLY_FORECAST)
    df = hourly_json_to_df(payload)
    assert "soil_moisture_0_to_7cm" in df.columns
    assert df["soil_moisture_0_to_7cm"].notna().all()


def test_client_drops_unsupported_variable():
    session = FakeSession(reject=("soil_moisture_0_to_7cm",))
    client = OpenMeteoClient(session=session, retries=1)
    payload = client.get_hourly_history(51.17, 71.45,
                                        dt.date(2025, 7, 1), dt.date(2025, 7, 3))
    assert "soil_moisture_0_to_7cm" not in payload["hourly"]
    assert "temperature_2m" in payload["hourly"]
    # столбец всё равно появится, просто пустым — конвейер не падает
    df = hourly_json_to_df(payload)
    assert df["soil_moisture_0_to_7cm"].isna().all()


def test_network_error_is_wrapped():
    class Broken:
        def get(self, *a, **k):
            raise OSError("сеть недоступна")
    client = OpenMeteoClient(session=Broken(), retries=1)
    try:
        client.get_hourly_history(51.17, 71.45, dt.date(2025, 7, 1), dt.date(2025, 7, 2))
    except OpenMeteoError:
        return
    raise AssertionError("ожидалась OpenMeteoError")


def test_feature_matrix_is_complete():
    client = OpenMeteoClient(session=FakeSession(), retries=1)
    params = AnalysisParams(lat=51.17, lon=71.45, calibration_days=400,
                            climatology_years=3, end_date=END_DATE)
    from main import collect_daily
    from decadal_risk import aggregate_to_decades, attach_climatology, build_climatology, daily_json_to_df

    daily = collect_daily(client, params, END_DATE)
    norms = build_climatology(daily_json_to_df(
        client.get_climatology(params.lat, params.lon, 3, end=END_DATE)))
    decades = attach_climatology(aggregate_to_decades(daily), norms)
    _, physics = load_model(MODEL_PATH, backend="numpy")
    frame = build_feature_frame(decades, physics, FeatureConfig())
    X = feature_matrix(frame)

    assert X.shape[1] == INPUT_DIM
    assert np.isfinite(X).all(), "в признаках остались NaN"
    # метеопризнаки 5-12 нормированы в [0, 1]
    assert X[:, 5:].min() >= 0.0 and X[:, 5:].max() <= 1.0
    # константы RUSLE взяты из чекпойнта
    assert np.allclose(X[:, 1], physics["K_factor"])
    assert np.allclose(X[:, 2], physics["LS_factor"])
    assert np.allclose(X[:, 4], physics["P_factor"])
    assert frame["C_factor"].between(0.03, 1.0).all()
    assert frame["rusle_A_norm"].between(0, 1).all()


def test_calibration_modes():
    raw = np.array([-5.0, -1.0, 0.0, 2.0, 7.0, 11.0])
    ranks = calibrate_outputs(raw, raw, mode="climatology")
    assert ranks.min() >= 0 and ranks.max() <= 1
    assert list(np.argsort(ranks)) == list(np.argsort(raw))       # порядок задаёт сеть
    sig = calibrate_outputs(raw, mode="sigmoid")
    assert 0 < sig.min() and sig.max() < 1
    assert np.allclose(calibrate_outputs(raw, mode="raw"), raw)
    assert np.allclose(calibrate_outputs(np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),
                                         np.full(6, 3.0), mode="climatology"), 0.5)


def test_end_to_end_with_pinn():
    client = OpenMeteoClient(session=FakeSession(), retries=1)
    params = AnalysisParams(lat=51.17, lon=71.45, farm="Тестовое хозяйство",
                            history_days=90, calibration_days=730,
                            climatology_years=5, forecast_days=7,
                            model_path=MODEL_PATH, backend="numpy",
                            end_date=END_DATE)
    report = run_analysis(client, params, verbose=False)

    assert len(report) >= 8, "ожидалось не меньше 8 декад в отчёте"
    assert report["composite"].between(0, 100).all()
    assert report["degradation_index"].notna().all(), "сеть не дала индекс"
    assert report["degradation_index"].between(0, 1).all()
    assert report["degradation_index"].nunique() > 3, "индекс деградации не меняется"
    assert report["pinn_raw_output"].notna().all()
    assert report["category"].isin(
        ["низкий", "умеренный", "повышенный", "высокий", "критический"]).all()
    assert (report["drought"].between(0, 100).all()
            and report["sukhovey"].between(0, 100).all()
            and report["early_snow"].between(0, 100).all())
    assert "прогноз" in " ".join(report["source"].tolist()), "прогнозные декады не попали"
    assert report["decade_start"].is_monotonic_increasing

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "risk_report.csv")
        report.to_csv(path, index=False, encoding="utf-8-sig")
        assert os.path.getsize(path) > 500
    return f"{len(report)} декад, композит {report['composite'].min():.1f}..{report['composite'].max():.1f}"


def test_pinn_changes_composite():
    """Индекс деградации должен реально влиять на композит, а не быть декорацией."""
    base = AnalysisParams(lat=51.17, lon=71.45, history_days=60,
                          calibration_days=500, climatology_years=3,
                          backend="numpy", end_date=END_DATE)
    without = run_analysis(OpenMeteoClient(session=FakeSession(), retries=1),
                           base, verbose=False)
    with_pinn = run_analysis(OpenMeteoClient(session=FakeSession(), retries=1),
                             AnalysisParams(**{**base.__dict__, "model_path": MODEL_PATH}),
                             verbose=False)
    assert with_pinn["degradation_index"].notna().all()
    assert without["degradation_index"].isna().all()
    assert (with_pinn["composite"] >= without["composite"] - 1e-9).all()
    assert (with_pinn["composite"] > without["composite"] + 1e-6).any()


def test_runs_without_model():
    client = OpenMeteoClient(session=FakeSession(), retries=1)
    report = run_analysis(client, AnalysisParams(lat=51.17, lon=71.45,
                                                 history_days=40, calibration_days=200,
                                                 climatology_years=2, end_date=END_DATE),
                          verbose=False)
    assert len(report) > 0
    assert report["composite"].between(0, 100).all()


def test_no_notifications_left():
    """Функциональность уведомлений должна быть удалена полностью."""
    assert not os.path.exists(os.path.join(SRC, "notifications.py"))
    assert not os.path.exists(os.path.join(ROOT, "config"))
    banned = ("smtplib", "mimetext", "notificationconfig", "maybe_notify",
              "send_email", "send_telegram", "api.telegram.org", "--notify")
    for folder in (SRC, os.path.join(ROOT, "tests")):
        for name in os.listdir(folder):
            if not name.endswith(".py"):
                continue
            text = open(os.path.join(folder, name), encoding="utf-8").read().lower()
            if name == "test_offline.py":
                continue
            for word in banned:
                assert word not in text, f"{name}: остался след '{word}'"


def test_only_openmeteo_sources():
    """В проекте не должно остаться ERA5/CDS/Kazhydromet."""
    banned = ("cdsapi", "era5landclient", "era5landrequest", "kazhydrometbulletin",
              "cds.climate.copernicus", "kazhydromet.kz", "reanalysis-era5")
    for name in os.listdir(SRC):
        if not name.endswith(".py"):
            continue
        text = open(os.path.join(SRC, name), encoding="utf-8").read().lower()
        for word in banned:
            assert word not in text, f"{name}: остался след '{word}'"
    requirements = open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8").read().lower()
    assert "cdsapi" not in requirements and "pyyaml" not in requirements


def test_parse_coordinates():
    for text in ("51.17 71.45", "51.17, 71.45", "51,17 71,45", "  51.17;71.45 "):
        place = parse_coordinates(text)
        assert place is not None and abs(place.lat - 51.17) < 1e-9
        assert abs(place.lon - 71.45) < 1e-9
    assert parse_coordinates("Акколь") is None
    assert parse_coordinates("100.0 200.0") is None        # вне диапазона
    assert parse_coordinates("") is None


def test_resolve_place_by_name():
    client = OpenMeteoClient(session=FakeSession(), retries=1)
    place = resolve_place(client, "Астана")
    assert abs(place.lat - 51.16) < 1e-6 and abs(place.lon - 71.47) < 1e-6
    assert "Астана" in place.title and "Казахстан" in place.title

    # из нескольких совпадений без вопроса берётся первое
    first = resolve_place(client, "Акколь")
    assert abs(first.lat - 51.99) < 1e-6
    # ...но выбор можно задать
    second = resolve_place(client, "Акколь", choose=lambda hits: 1)
    assert abs(second.lat - 45.00) < 1e-6

    # координаты не уходят в геокодер
    assert resolve_place(client, "51.17 71.45").lat == 51.17

    try:
        resolve_place(client, "Зззз-несуществующее")
    except ValueError:
        pass
    else:
        raise AssertionError("ожидалась ValueError для ненайденного места")


def test_cli_defaults_require_no_typing():
    args = parse_args([])
    assert args.forecast_days == 10, "прогноз должен быть включён по умолчанию"
    assert args.lat is None and args.lon is None, "координаты не обязательны"
    assert args.model is None                       # ищется сам
    assert args.history_days == 60 and args.out == "risk_report.csv"
    assert parse_args(["Акколь"]).place == ["Акколь"]
    assert parse_args(["Петропавловск", "Казахстан"]).place == ["Петропавловск", "Казахстан"]


def test_model_is_found_automatically():
    found = find_model(None, start_dir=SRC)
    assert found is not None and os.path.basename(found) == "pinn_model.pt"
    assert os.path.isfile(found)
    assert find_model(os.path.join(ROOT, "нет-такого.pt")) is None
    assert find_model(MODEL_PATH) == MODEL_PATH


def test_settings_remember_last_place():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "settings.json")
        assert load_settings(path) == {}                      # файла ещё нет
        assert save_settings({"lat": 51.99, "lon": 70.94, "title": "Акколь"}, path)
        restored = place_from_settings(load_settings(path))
        assert restored is not None and abs(restored.lat - 51.99) < 1e-9
        assert restored.title == "Акколь"
        assert place_from_settings({}) is None
        assert place_from_settings({"lat": "не число", "lon": 1}) is None


def test_resolve_location_priority():
    client = OpenMeteoClient(session=FakeSession(), retries=1)
    saved = {"lat": 51.99, "lon": 70.94, "title": "Акколь"}

    # 1. явные координаты важнее всего
    place = resolve_location(client, parse_args(["--lat", "55.0", "--lon", "60.0"]), saved)
    assert (place.lat, place.lon) == (55.0, 60.0)

    # 2. название из командной строки
    place = resolve_location(client, parse_args(["--yes", "Астана"]), saved)
    assert abs(place.lat - 51.16) < 1e-6

    # 3. без аргументов берётся прошлая точка — вводить ничего не нужно
    place = resolve_location(client, parse_args(["--yes"]), saved)
    assert place.title == "Акколь" and abs(place.lat - 51.99) < 1e-9

    # 4. если и прошлой точки нет — понятное сообщение, а не трассировка
    try:
        resolve_location(client, parse_args(["--yes"]), {})
    except ValueError as exc:
        assert "run.py" in str(exc)
    else:
        raise AssertionError("ожидалась ValueError без места и без настроек")


def test_end_to_end_from_place_name_only():
    """Сценарий «пользователь ввёл только название»: координаты и модель — сами."""
    client = OpenMeteoClient(session=FakeSession(), retries=1)
    place = resolve_location(client, parse_args(["--yes", "Астана"]), {})
    model_path = find_model(None, start_dir=SRC)
    assert model_path is not None
    report = run_analysis(client, AnalysisParams(
        lat=place.lat, lon=place.lon, farm=place.title, history_days=60,
        calibration_days=500, climatology_years=3, forecast_days=10,
        model_path=model_path, backend="numpy", end_date=END_DATE), verbose=False)
    assert len(report) >= 6
    assert report["degradation_index"].notna().all()
    return f"{place.title}: {len(report)} декад"


# ===========================================================================
# Веб-оболочка
# ===========================================================================
def _web_app():
    return AgroRiskApp(
        client_factory=lambda: OpenMeteoClient(session=FakeSession(), retries=1),
        model_path=MODEL_PATH, end_date=END_DATE, remember=False)


class _Server:
    """Поднимает сервер на свободном порту и гасит его по выходе из блока."""

    def __enter__(self):
        self.app = _web_app()
        self.httpd = create_server(self.app, "127.0.0.1", 0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def post(self, path, payload):
        request = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def wait(self, job_id, limit=90):
        for _ in range(limit):
            snapshot = self.get(f"/api/job?id={job_id}")
            if snapshot["state"] != "running":
                return snapshot
            time.sleep(0.4)
        raise AssertionError("расчёт не завершился за отведённое время")


def test_web_page_is_self_contained():
    """Страница должна открываться без интернета: никаких внешних CDN."""
    html = open(os.path.join(ROOT, "web", "index.html"), encoding="utf-8").read()
    assert "<script" in html and "<style>" in html
    for marker in ('src="http', "src='http", 'href="http', "href='http"):
        assert marker not in html, f"внешний ресурс на странице: {marker}"
    for element in ("ribbon", "detail", "sheet", "csv", "q", "go", "tuning"):
        assert f'id="{element}"' in html, f"на странице нет элемента {element}"


def test_web_json_is_safe():
    """NaN и даты не должны ломать выдачу браузеру."""
    frame = pd.DataFrame([{"a": np.nan, "b": np.float64(1.5), "c": np.int64(3),
                           "d": dt.date(2025, 9, 1), "e": "текст", "f": np.bool_(True)}])
    row = report_to_payload(frame)[0]
    assert row["a"] is None and row["b"] == 1.5 and row["c"] == 3
    assert row["d"] == "2025-09-01" and row["e"] == "текст" and row["f"] is True
    json.dumps(row)                                   # не должно бросить


def test_web_status_and_places():
    with _Server() as server:
        status = server.get("/api/status")
        assert status["model"] == "pinn_model.pt"
        assert status["version"]

        places = server.get("/api/places?q=" + quote("Акколь"))["places"]
        assert len(places) == 2
        assert places[0]["name"] == "Акколь"
        assert abs(places[0]["lat"] - 51.99) < 1e-6
        assert server.get("/api/places?q=")["places"] == []


def test_web_full_run():
    with _Server() as server:
        job = server.post("/api/run", {"query": "Астана", "history_days": 90,
                                       "forecast_days": 10, "climatology_years": 4,
                                       "calibration_days": 600})["job"]
        snapshot = server.wait(job)
        assert snapshot["state"] == "done", snapshot.get("error")

        result = snapshot["result"]
        assert result["model"] == "pinn_model.pt"
        assert "Астана" in result["place"]["title"]
        assert len(result["decades"]) >= 8
        assert len(snapshot["log"]) >= 5, "ход расчёта не показывается"

        for row in result["decades"]:
            assert 0 <= row["composite"] <= 100
            assert row["category"] in ("низкий", "умеренный", "повышенный",
                                       "высокий", "критический")
            assert row["degradation_index"] is None or 0 <= row["degradation_index"] <= 1
        assert any(row["source"] == "прогноз" for row in result["decades"])
        assert "NaN" not in json.dumps(result)

        with urllib.request.urlopen(server.base + f"/api/report.csv?id={job}") as resp:
            body = resp.read().decode("utf-8-sig")
            assert resp.headers.get("Content-Disposition", "").startswith("attachment")
        assert "composite" in body.splitlines()[0]
        assert len(body.splitlines()) == len(result["decades"]) + 1
        return f"{len(result['decades'])} декад через HTTP"


def test_web_accepts_coordinates_without_geocoder():
    with _Server() as server:
        job = server.post("/api/run", {"query": "Моё поле", "lat": 51.17, "lon": 71.45,
                                       "history_days": 40, "forecast_days": 0,
                                       "climatology_years": 2,
                                       "calibration_days": 200})["job"]
        snapshot = server.wait(job)
        assert snapshot["state"] == "done", snapshot.get("error")
        assert snapshot["result"]["place"]["title"] == "Моё поле"
        assert abs(snapshot["result"]["place"]["lat"] - 51.17) < 1e-9


def test_web_errors_are_readable():
    with _Server() as server:
        empty = server.wait(server.post("/api/run", {"query": ""})["job"])
        assert empty["state"] == "error"
        assert "населённый пункт" in empty["error"].lower()

        missing = server.wait(server.post("/api/run", {"query": "Зззз-нет-такого"})["job"])
        assert missing["state"] == "error"
        assert "не найден" in missing["error"].lower()

        try:
            server.get("/api/job?id=" + quote("нет-такого-расчёта"))
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("ожидался 404 для неизвестного расчёта")


def test_web_serves_page_and_404():
    with _Server() as server:
        with urllib.request.urlopen(server.base + "/") as resp:
            assert resp.status == 200
            assert "text/html" in resp.headers.get("Content-Type", "")
            assert "Агрориск" in resp.read().decode("utf-8")
        try:
            server.get(quote("/api/нет-такого"))
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("ожидался 404")


# ===========================================================================
def main() -> int:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failed = 0
    for test in tests:
        try:
            note = test()
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {test.__name__}: {exc}")
        except Exception as exc:                                  # noqa: BLE001
            failed += 1
            print(f"  ERROR {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            suffix = f"  ({note})" if isinstance(note, str) else ""
            print(f"  ok    {test.__name__}{suffix}")
    print(f"\n{len(tests) - failed}/{len(tests)} проверок пройдено")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
