"""
main.py
=======
Точка входа. Рассылка оповещений из проекта удалена полностью:
программа считает индекс риска, печатает его и сохраняет отчёт в CSV.

Вводить вручную почти ничего не нужно:

    python run.py                      # спросит место и посчитает
    python run.py Акколь               # место прямо в командной строке
    python run.py "51.17 71.45"        # или координаты
    python run.py                      # в следующий раз — просто Enter

Координаты ищутся геокодером Open-Meteo по названию, `pinn_model.pt`
находится сам рядом с программой, прогноз включён по умолчанию,
а последняя точка запоминается в `~/.agrorisk.json`.

Порядок работы:
  1. Один почасовой запрос в архив Open-Meteo за калибровочное окно
     (по умолчанию 2 года): температура, влажность, осадки, снег, ветер,
     порывы, влажность почвы.
  2. Один суточный запрос за N лет — климатическая норма по декадам.
  3. Почасовой прогноз, чтобы досчитать текущую декаду вперёд.
  4. Агрегация в декады и сборка 13 признаков нейросети.
  5. Инференс pinn_model.pt → индекс деградации почвы по каждой декаде.
  6. Под-индексы засухи / суховея / раннего снега + композит, отчёт в CSV.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

from data_sources import OpenMeteoClient, OpenMeteoError, archive_end_date
from decadal_risk import (
    aggregate_to_decades, attach_climatology, build_climatology,
    daily_json_to_df, evaluate_decade, hourly_json_to_df, hourly_to_daily,
)
from features import FeatureConfig, build_feature_frame, feature_matrix
from location import (
    Place, find_model, format_geocoding_hit, load_settings, place_from_settings,
    resolve_place, save_settings,
)
from pinn_model import load_model, predict_degradation_index

VERSION = "2.0"


@dataclass
class AnalysisParams:
    lat: float
    lon: float
    farm: str = "Хозяйство"
    history_days: int = 60
    calibration_days: int = 730
    climatology_years: int = 15
    forecast_days: int = 10
    model_path: Optional[str] = None
    backend: str = "auto"
    pinn_calibration: str = "climatology"
    first_snow_doy: int = 300
    sukhovey_min_hours: int = 3
    cover: Optional[float] = None
    c_factor: Optional[float] = None
    end_date: Optional[dt.date] = None


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Декадный индекс риска засухи / суховея / раннего снега "
                    "(Open-Meteo + PINN)",
        prog="run.py",
        epilog="Примеры:  python run.py Акколь  |  python run.py \"51.17 71.45\"  "
               "|  python run.py  (спросит место и запомнит его)")
    p.add_argument("place", nargs="*", default=None,
                   help="населённый пункт или координаты; "
                        "если не указать — программа спросит")
    p.add_argument("--place", dest="place_opt", type=str, default=None,
                   help="то же самое в виде ключа")
    p.add_argument("--lat", type=float, default=None, help="широта поля")
    p.add_argument("--lon", type=float, default=None, help="долгота поля")
    p.add_argument("--farm", type=str, default=None,
                   help="название хозяйства (по умолчанию — название места)")
    p.add_argument("--history-days", type=int, default=60,
                   help="глубина отчёта в сутках (по умолчанию 60)")
    p.add_argument("--calibration-days", type=int, default=730,
                   help="окно для калибровки выхода сети, суток (по умолчанию 730)")
    p.add_argument("--climatology-years", type=int, default=15,
                   help="сколько лет брать для климатической нормы")
    p.add_argument("--forecast-days", type=int, default=10,
                   help="досчитать N суток прогноза, 0-16 (по умолчанию 10)")
    p.add_argument("--model", type=str, default=None,
                   help="путь к pinn_model.pt (по умолчанию ищется рядом)")
    p.add_argument("--no-model", action="store_true",
                   help="считать без нейросети, только метеоиндексы")
    p.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto",
                   help="чем считать сеть (по умолчанию torch, если установлен)")
    p.add_argument("--pinn-calibration", choices=["climatology", "sigmoid", "raw"],
                   default="climatology",
                   help="как переводить выход сети в индекс 0..1")
    p.add_argument("--first-snow-doy", type=int, default=300,
                   help="климатический день года первого снега (по умолчанию ≈27 окт)")
    p.add_argument("--sukhovey-min-hours", type=int, default=3,
                   help="сколько часов условий суховея делают сутки суховейными")
    p.add_argument("--cover", type=float, default=None,
                   help="фиксированная доля растительного покрова 0..1 (например, по NDVI)")
    p.add_argument("--c-factor", type=float, default=None,
                   help="фиксированный RUSLE C-фактор 0..1 (приоритетнее --cover)")
    p.add_argument("--out", type=str, default="risk_report.csv")
    p.add_argument("--yes", action="store_true",
                   help="ничего не спрашивать: брать первое совпадение и значения по умолчанию")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Определение точки расчёта без ручного ввода координат
# ---------------------------------------------------------------------------
def _ask(prompt: str) -> str:
    """Запрос у пользователя; в неинтерактивном режиме возвращает пустую строку."""
    if not sys.stdin or not sys.stdin.isatty():
        return ""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def _choose_hit(hits: list) -> int:
    """Показывает совпадения геокодера и даёт выбрать нужное (по умолчанию первое)."""
    print("\nНайдено несколько мест:")
    for number, hit in enumerate(hits, 1):
        print(f"  {number}. {format_geocoding_hit(hit)} "
              f"({hit['latitude']:.3f}, {hit['longitude']:.3f})")
    answer = _ask("Номер нужного (Enter — 1): ")
    try:
        return int(answer) - 1
    except (TypeError, ValueError):
        return 0


def resolve_location(client: OpenMeteoClient, args: argparse.Namespace,
                     settings: dict) -> Place:
    """
    Определяет точку расчёта по принципу «чем меньше ввода, тем лучше»:
    координаты из ключей → место из аргумента → прошлая точка → вопрос.
    """
    if args.lat is not None and args.lon is not None:
        return Place(lat=args.lat, lon=args.lon, title=args.farm or "")

    query = args.place_opt or (" ".join(args.place).strip() if args.place else "")
    previous = place_from_settings(settings)

    if not query:
        if args.yes and previous is not None:
            return previous
        hint = f" (Enter — {previous.describe()})" if previous else \
               " (например: Акколь  или  51.17 71.45)"
        query = _ask(f"Населённый пункт или координаты{hint}: ")

    if not query:
        if previous is not None:
            return previous
        raise ValueError(
            "Не указано место расчёта. Запустите так: python run.py Акколь "
            "— или передайте координаты ключами --lat и --lon.")

    return resolve_place(client, query, choose=None if args.yes else _choose_hit)


# ---------------------------------------------------------------------------
# Конвейер
# ---------------------------------------------------------------------------
def collect_daily(client: OpenMeteoClient, params: AnalysisParams,
                  end: dt.date) -> pd.DataFrame:
    """Почасовой архив (+ прогноз) → суточная таблица."""
    start = end - dt.timedelta(days=max(params.calibration_days, params.history_days))
    archive = client.get_hourly_history(params.lat, params.lon, start, end)
    daily = hourly_to_daily(hourly_json_to_df(archive), params.sukhovey_min_hours)
    daily["is_forecast"] = 0

    if params.forecast_days > 0:
        forecast = client.get_hourly_forecast(params.lat, params.lon, params.forecast_days)
        fc_daily = hourly_to_daily(hourly_json_to_df(forecast), params.sukhovey_min_hours)
        fc_daily["is_forecast"] = 1
        fc_daily = fc_daily[~fc_daily["date"].isin(set(daily["date"]))]
        if not fc_daily.empty:
            daily = pd.concat([daily, fc_daily], ignore_index=True)

    return daily.sort_values("date").reset_index(drop=True)


def run_analysis(client: OpenMeteoClient, params: AnalysisParams,
                 verbose: bool = True, on_progress=None) -> pd.DataFrame:
    """
    Прогоняет весь конвейер и возвращает декадную таблицу с индексами риска.

    `client` передаётся снаружи, поэтому конвейер целиком проверяется
    оффлайн-заглушкой (см. tests/test_offline.py).
    `on_progress(message)` — необязательный обработчик хода расчёта;
    через него веб-интерфейс показывает шаги, пока идут запросы.
    """
    def log(message: str):
        if verbose:
            print(message, flush=True)
        if on_progress is not None:
            on_progress(message.strip())

    end = params.end_date or archive_end_date()

    log(f"[1/5] Open-Meteo: почасовой архив по точке ({params.lat}, {params.lon})...")
    daily = collect_daily(client, params, end)
    if daily.empty:
        raise RuntimeError("Open-Meteo не вернул ни одних полных суток.")
    log(f"      получено суток: {len(daily)} "
        f"({daily['date'].min()} .. {daily['date'].max()})")

    log(f"[2/5] Климатическая норма по декадам за {params.climatology_years} лет...")
    norms = build_climatology(daily_json_to_df(
        client.get_climatology(params.lat, params.lon, params.climatology_years, end=end)))
    log(f"      норм рассчитано: {len(norms)} из 36 декад")

    log("[3/5] Агрегация в декады...")
    decades = attach_climatology(aggregate_to_decades(daily), norms)
    log(f"      декад в окне: {len(decades)}")

    log("[4/5] Нейросеть: индекс деградации почвы...")
    decades["degradation_index"] = np.nan
    decades["pinn_raw_output"] = np.nan
    config = FeatureConfig(cover_override=params.cover, c_factor_override=params.c_factor)

    if params.model_path:
        model, physics = load_model(params.model_path, backend=params.backend)
        decades = build_feature_frame(decades, physics, config)
        X = feature_matrix(decades)
        reference = decades["is_complete"].to_numpy()
        reference_X = X[reference] if reference.sum() >= 5 else None
        index, raw = predict_degradation_index(model, X, reference_X,
                                               mode=params.pinn_calibration)
        decades["degradation_index"] = index
        decades["pinn_raw_output"] = raw
        log(f"      бэкенд: {physics['backend']}; "
            f"K={physics['K_factor']:.2f} LS={physics['LS_factor']:.3f} "
            f"P={physics['P_factor']:.2f} max_rusle={physics['max_rusle']:.3f}")
        log(f"      калибровка: {params.pinn_calibration}; "
            f"индекс {np.nanmin(index):.3f}..{np.nanmax(index):.3f}, "
            f"эталонных декад: {0 if reference_X is None else len(reference_X)}")
    else:
        log("      модель не подключена: индекс деградации не считается.")

    log("[5/5] Индексы риска по декадам...")
    horizon = end - dt.timedelta(days=params.history_days)
    report = decades[decades["decade_start"] >= horizon].copy()
    if report.empty:
        report = decades.tail(3).copy()

    results = [
        evaluate_decade(row,
                        degradation_index=row.get("degradation_index"),
                        pinn_raw_output=row.get("pinn_raw_output"),
                        climatological_first_snow_doy=params.first_snow_doy)
        for _, row in report.iterrows()
    ]

    out = pd.DataFrame([asdict(r) for r in results])
    out["notes"] = out["notes"].apply(lambda notes: "; ".join(notes))
    return out


def print_report(farm: str, report: pd.DataFrame) -> None:
    """Печатает отчёт таблицей в консоль."""
    print(f"\nДекадный индекс риска — {farm}")
    header = (f"{'декада':>14} {'ист.':>12} {'засуха':>7} {'суховей':>8} "
              f"{'снег':>6} {'деград.':>8} {'композит':>9}  категория")
    print(header)
    print("-" * len(header))
    for row in report.itertuples():
        degradation = "—" if pd.isna(row.degradation_index) else f"{row.degradation_index:.2f}"
        print(f"{row.decade_label:>14} {row.source:>12} {row.drought:7.1f} "
              f"{row.sukhovey:8.1f} {row.early_snow:6.1f} {degradation:>8} "
              f"{row.composite:9.1f}  {row.category}")
    notes = report[report["notes"].astype(bool)]
    if not notes.empty:
        print("\nКомментарии:")
        for row in notes.itertuples():
            print(f"  {row.decade_label}: {row.notes}")


def main(argv=None) -> int:
    args = parse_args(argv)
    print(f"Агрориск {VERSION} — Open-Meteo + PINN")
    settings = load_settings()
    client = OpenMeteoClient()

    try:
        place = resolve_location(client, args, settings)
    except OpenMeteoError as exc:
        print(f"\nНе удалось найти место через Open-Meteo: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    model_path = None if args.no_model else find_model(args.model)
    if args.model and model_path is None:
        print(f"\nФайл модели не найден: {args.model}", file=sys.stderr)
        return 1
    if model_path is None and not args.no_model:
        print("Файл pinn_model.pt рядом не найден — считаю без нейросети "
              "(укажите путь ключом --model).")

    farm = args.farm or place.title or "Хозяйство"
    print(f"\nТочка расчёта: {place.describe()}")
    if model_path:
        print(f"Модель: {model_path}")

    params = AnalysisParams(
        lat=place.lat, lon=place.lon, farm=farm,
        history_days=args.history_days, calibration_days=args.calibration_days,
        climatology_years=args.climatology_years, forecast_days=args.forecast_days,
        model_path=model_path, backend=args.backend,
        pinn_calibration=args.pinn_calibration, first_snow_doy=args.first_snow_doy,
        sukhovey_min_hours=args.sukhovey_min_hours,
        cover=args.cover, c_factor=args.c_factor,
    )
    try:
        report = run_analysis(client, params)
    except OpenMeteoError as exc:
        print(f"\nОшибка обращения к Open-Meteo: {exc}", file=sys.stderr)
        return 2
    except (RuntimeError, ValueError, KeyError) as exc:
        print(f"\nОшибка расчёта: {exc}", file=sys.stderr)
        return 1

    print_report(farm, report)
    report.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nГотово. Отчёт сохранён в {args.out}")

    save_settings({"lat": place.lat, "lon": place.lon,
                   "title": place.title, "farm": farm})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
