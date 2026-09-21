"""
decadal_risk.py
================
Превращение почасовых данных Open-Meteo в декадный индекс риска.

Декада — стандартная единица агрометеорологии: 1-10, 11-20 и 21-конец месяца,
36 декад в году.

Считаются три под-индекса (каждый 0..100):
  * засуха (drought)      — дефицит осадков + дефицит влаги в почве + жара;
  * суховей (sukhovey)    — доля суховейных суток в декаде + ветер + сухость;
  * ранний снег (early_snow) — снег и заморозки раньше климатической нормы.

Композитный индекс — их взвешенная сумма, усиленная индексом деградации
почвы с нейросети (деградированная почва хуже держит влагу и сильнее
подвержена ветровой эрозии).

Что изменилось по сравнению с прошлой версией:
  * данные приходят почасовыми, поэтому суховейные сутки считаются по
    реальному синоптическому критерию, а не по средним за декаду;
  * влажность почвы берётся из Open-Meteo (раньше требовался ERA5-Land);
  * индекс деградации почвы реально приходит из сети, а не остаётся пустым.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# критерий суховейных суток (агрометеорологический стандарт)
SUKHOVEY_TEMP_C = 25.0
SUKHOVEY_RH_PCT = 30.0
SUKHOVEY_WIND_MS = 5.0
SUKHOVEY_MIN_HOURS = 3

RISK_WEIGHTS = {"drought": 0.45, "sukhovey": 0.35, "early_snow": 0.20}
DEGRADATION_BOOST = 0.15      # деградация почвы поднимает композит максимум на 15 %


# ---------------------------------------------------------------------------
# Декады
# ---------------------------------------------------------------------------
def decade_of(date: dt.date) -> int:
    """Номер декады внутри месяца: 1, 2 или 3."""
    if date.day <= 10:
        return 1
    return 2 if date.day <= 20 else 3


def decade_label(date: dt.date) -> str:
    return f"{date.year}-{date.month:02d}-Д{decade_of(date)}"


def decade_start_of(date: dt.date) -> dt.date:
    return dt.date(date.year, date.month, {1: 1, 2: 11, 3: 21}[decade_of(date)])


def decade_length(start: dt.date) -> int:
    """Сколько суток в декаде (третья декада — 8..11 дней)."""
    if start.day in (1, 11):
        return 10
    if start.month == 12:
        next_month = dt.date(start.year + 1, 1, 1)
    else:
        next_month = dt.date(start.year, start.month + 1, 1)
    return (next_month - start).days


# ---------------------------------------------------------------------------
# Разбор ответов Open-Meteo
# ---------------------------------------------------------------------------
def hourly_json_to_df(payload: dict) -> pd.DataFrame:
    """
    Почасовой ответ Open-Meteo → DataFrame.
    Слои влажности почвы из прогноза (0-1 / 1-3 / 3-9 см) сводятся к одному
    столбцу `soil_moisture_0_to_7cm` взвешиванием по толщине слоя.
    """
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise ValueError("В ответе Open-Meteo нет почасового блока 'hourly'.")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])

    if "soil_moisture_0_to_7cm" not in df.columns:
        layers = [("soil_moisture_0_to_1cm", 1.0),
                  ("soil_moisture_1_to_3cm", 2.0),
                  ("soil_moisture_3_to_9cm", 6.0)]
        present = [(c, w) for c, w in layers if c in df.columns]
        if present:
            total = sum(w for _, w in present)
            acc = sum(pd.to_numeric(df[c], errors="coerce") * w for c, w in present)
            df["soil_moisture_0_to_7cm"] = acc / total
        else:
            df["soil_moisture_0_to_7cm"] = np.nan

    for column in ("temperature_2m", "relative_humidity_2m", "precipitation",
                   "rain", "snowfall", "wind_speed_10m", "wind_gusts_10m",
                   "soil_moisture_0_to_7cm"):
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def hourly_to_daily(df: pd.DataFrame,
                    sukhovey_min_hours: int = SUKHOVEY_MIN_HOURS) -> pd.DataFrame:
    """
    Почасовые данные → суточные агрегаты. Здесь же помечаются суховейные
    сутки: не менее `sukhovey_min_hours` часов с T ≥ 25 °C, влажностью ≤ 30 %
    и ветром ≥ 5 м/с одновременно.
    """
    work = df.copy()
    work["date"] = work["time"].dt.date
    work["is_sukhovey_hour"] = (
        (work["temperature_2m"] >= SUKHOVEY_TEMP_C)
        & (work["relative_humidity_2m"] <= SUKHOVEY_RH_PCT)
        & (work["wind_speed_10m"] >= SUKHOVEY_WIND_MS)
    ).astype(int)

    daily = work.groupby("date").agg(
        temp_max_c=("temperature_2m", "max"),
        temp_min_c=("temperature_2m", "min"),
        temp_mean_c=("temperature_2m", "mean"),
        rel_humidity_pct=("relative_humidity_2m", "mean"),
        rh_min_pct=("relative_humidity_2m", "min"),
        precip_mm=("precipitation", "sum"),
        snowfall_cm=("snowfall", "sum"),
        wind_mean_ms=("wind_speed_10m", "mean"),
        wind_max_ms=("wind_gusts_10m", "max"),
        soil_moisture_frac=("soil_moisture_0_to_7cm", "mean"),
        sukhovey_hours=("is_sukhovey_hour", "sum"),
        hours=("time", "count"),
    ).reset_index()

    # сутки без полного ряда часов (край окна) не считаем полноценными
    daily = daily[daily["hours"] >= 20].copy()
    daily["sukhovey_day"] = (daily["sukhovey_hours"] >= sukhovey_min_hours).astype(int)
    if daily["wind_max_ms"].isna().all():
        daily["wind_max_ms"] = daily["wind_mean_ms"]
    else:
        daily["wind_max_ms"] = daily["wind_max_ms"].fillna(daily["wind_mean_ms"])
    return daily.sort_values("date").reset_index(drop=True)


def daily_json_to_df(payload: dict) -> pd.DataFrame:
    """Суточный ответ Open-Meteo (для климатологии) → DataFrame."""
    daily = payload.get("daily")
    if not daily or "time" not in daily:
        raise ValueError("В ответе Open-Meteo нет суточного блока 'daily'.")
    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"]).dt.date
    return df.drop(columns=["time"])


# ---------------------------------------------------------------------------
# Агрегация в декады и климатическая норма
# ---------------------------------------------------------------------------
def aggregate_to_decades(daily: pd.DataFrame) -> pd.DataFrame:
    """Суточные наблюдения → декадная таблица с агрометеорологическими агрегатами."""
    df = daily.copy()
    df["decade_label"] = df["date"].apply(decade_label)
    df["decade_start"] = df["date"].apply(decade_start_of)
    if "is_forecast" not in df.columns:
        df["is_forecast"] = 0

    agg = df.groupby("decade_label").agg(
        decade_start=("decade_start", "first"),
        precip_mm_decade=("precip_mm", "sum"),
        temp_mean_c=("temp_mean_c", "mean"),
        temp_max_c=("temp_max_c", "max"),
        temp_min_c=("temp_min_c", "min"),
        wind_mean_ms=("wind_mean_ms", "mean"),
        wind_max_ms=("wind_max_ms", "max"),
        rel_humidity_pct=("rel_humidity_pct", "mean"),
        soil_moisture_frac=("soil_moisture_frac", "mean"),
        snowfall_sum_cm=("snowfall_cm", "sum"),
        sukhovey_days=("sukhovey_day", "sum"),
        forecast_days=("is_forecast", "sum"),
        n_days=("date", "count"),
    ).reset_index()

    precip_lists = df.groupby("decade_label")["precip_mm"].apply(list)
    agg["daily_precip_mm"] = agg["decade_label"].map(precip_lists)

    agg["month"] = agg["decade_start"].apply(lambda d: d.month)
    agg["decade_num"] = agg["decade_start"].apply(lambda d: {1: 1, 11: 2, 21: 3}[d.day])
    agg["expected_days"] = agg["decade_start"].apply(decade_length)
    agg["is_complete"] = agg["n_days"] >= agg["expected_days"]
    agg["source"] = np.where(agg["forecast_days"] == 0, "факт",
                             np.where(agg["forecast_days"] >= agg["n_days"],
                                      "прогноз", "факт+прогноз"))
    return agg.sort_values("decade_start").reset_index(drop=True)


def build_climatology(daily_climate: pd.DataFrame) -> pd.DataFrame:
    """
    Климатическая норма по декадам месяца: средняя сумма осадков, её
    стандартное отклонение (нужно для SPI-подобной аномалии) и средняя
    температура. Неполные декады в норму не берутся.
    """
    df = daily_climate.copy()
    df["precipitation_sum"] = pd.to_numeric(df.get("precipitation_sum"), errors="coerce")
    df["temperature_2m_mean"] = pd.to_numeric(df.get("temperature_2m_mean"), errors="coerce")
    df["decade"] = df["date"].apply(decade_of)
    df["month"] = df["date"].apply(lambda d: d.month)
    df["year"] = df["date"].apply(lambda d: d.year)
    df["expected_days"] = df["date"].apply(lambda d: decade_length(decade_start_of(d)))

    per_year = df.groupby(["year", "month", "decade"]).agg(
        precip_mm=("precipitation_sum", "sum"),
        temp_mean_c=("temperature_2m_mean", "mean"),
        days=("date", "count"),
        expected_days=("expected_days", "first"),
    ).reset_index()
    per_year = per_year[per_year["days"] >= per_year["expected_days"]]

    norms = per_year.groupby(["month", "decade"]).agg(
        precip_norm_mm=("precip_mm", "mean"),
        precip_std_mm=("precip_mm", "std"),
        temp_norm_c=("temp_mean_c", "mean"),
        years_in_norm=("precip_mm", "count"),
    ).reset_index()
    return norms


def attach_climatology(decades: pd.DataFrame, norms: pd.DataFrame) -> pd.DataFrame:
    """Подмешивает нормы к декадной таблице по (месяц, номер декады)."""
    merged = decades.merge(norms, left_on=["month", "decade_num"],
                           right_on=["month", "decade"], how="left")
    return merged.drop(columns=["decade"], errors="ignore")


# ---------------------------------------------------------------------------
# Под-индексы риска
# ---------------------------------------------------------------------------
def drought_subindex(precip_mm: float, precip_norm_mm: Optional[float],
                     precip_std_mm: Optional[float],
                     soil_moisture_frac: Optional[float],
                     temp_anom_c: float) -> float:
    """
    Дефицит осадков (SPI-подобная аномалия, до 60 баллов)
    + дефицит влаги в почве (до 25) + аномальная жара (до 15).
    """
    if precip_norm_mm is None or not np.isfinite(precip_norm_mm):
        precip_score = 0.0
    else:
        std = precip_std_mm if (precip_std_mm and np.isfinite(precip_std_mm)
                                and precip_std_mm > 0) else max(precip_norm_mm * 0.3, 1.0)
        spi_like = (precip_mm - precip_norm_mm) / std
        precip_score = float(np.clip(-spi_like, 0, 3) / 3 * 60)

    if soil_moisture_frac is not None and np.isfinite(soil_moisture_frac):
        # здоровый диапазон объёмной влажности ≈ 0.20-0.35, ниже 0.15 тревожно
        soil_score = float(np.clip((0.20 - soil_moisture_frac) / 0.20, 0, 1) * 25)
    else:
        soil_score = 0.0

    heat_score = float(np.clip(temp_anom_c if np.isfinite(temp_anom_c) else 0, 0, 5) / 5 * 15)
    return float(np.clip(precip_score + soil_score + heat_score, 0, 100))


def sukhovey_subindex(sukhovey_days: float, n_days: int,
                      wind_mean_ms: float, rel_humidity_pct: float) -> float:
    """
    Доля суховейных суток в декаде (до 60 баллов) плюс надбавки за общую
    ветреность декады (до 20) и сухость воздуха (до 20).

    Надбавка за ветер считается по СРЕДНЕЙ скорости ветра за декаду, а не по
    максимальному порыву: критерий суховея (≥ 5 м/с) сформулирован именно для
    средней скорости, а максимальный порыв за десять суток превышает 15 м/с
    почти всегда и давал бы всем декадам одинаковую надбавку.
    """
    share = float(sukhovey_days) / max(int(n_days), 1)
    base = float(np.clip(share, 0, 1) * 60)
    wind = wind_mean_ms if np.isfinite(wind_mean_ms) else 0.0
    rh = rel_humidity_pct if np.isfinite(rel_humidity_pct) else 100.0
    wind_score = float(np.clip((wind - SUKHOVEY_WIND_MS) / 5, 0, 1) * 20)
    dry_score = float(np.clip((SUKHOVEY_RH_PCT - rh) / SUKHOVEY_RH_PCT, 0, 1) * 20)
    return float(np.clip(base + wind_score + dry_score, 0, 100))


def early_snow_subindex(decade_start: dt.date, snowfall_sum_cm: float,
                        temp_min_c: float,
                        climatological_first_snow_doy: int = 300) -> float:
    """
    Снег и заморозки раньше климатической даты первого снега — прямой риск
    для незавершённой уборки. `climatological_first_snow_doy` — день года
    (по умолчанию ≈27 октября), задайте по норме своего района.
    """
    doy = decade_start.timetuple().tm_yday
    # весенние декады к раннему снегу не относятся
    if doy < 182:
        early_bonus = 0.0
    else:
        days_early = climatological_first_snow_doy - doy
        snow = snowfall_sum_cm if np.isfinite(snowfall_sum_cm) else 0.0
        early_bonus = 0.0 if (days_early <= 0 or snow <= 0) \
            else float(np.clip(days_early / 30, 0, 1) * 60)

    snow = snowfall_sum_cm if np.isfinite(snowfall_sum_cm) else 0.0
    snowfall_score = float(np.clip(snow / 5, 0, 1) * 25)
    tmin = temp_min_c if np.isfinite(temp_min_c) else 0.0
    frost_score = float(np.clip((-tmin) / 10, 0, 1) * 15) if tmin < 0 else 0.0
    return float(np.clip(early_bonus + snowfall_score + frost_score, 0, 100))


def risk_category(score: float) -> str:
    if score < 20:
        return "низкий"
    if score < 40:
        return "умеренный"
    if score < 60:
        return "повышенный"
    if score < 80:
        return "высокий"
    return "критический"


def compute_composite_index(drought: float, sukhovey: float, early_snow: float,
                            degradation_index: Optional[float] = None) -> float:
    base = (RISK_WEIGHTS["drought"] * drought
            + RISK_WEIGHTS["sukhovey"] * sukhovey
            + RISK_WEIGHTS["early_snow"] * early_snow)
    if degradation_index is not None and np.isfinite(degradation_index):
        base *= 1 + DEGRADATION_BOOST * float(np.clip(degradation_index, 0, 1))
    return float(np.clip(base, 0, 100))


@dataclass
class DecadeRiskResult:
    decade_label: str
    decade_start: dt.date
    source: str
    n_days: int
    drought: float
    sukhovey: float
    early_snow: float
    degradation_index: Optional[float]
    pinn_raw_output: Optional[float]
    rusle_A_norm: Optional[float]
    composite: float
    category: str
    precip_mm: float
    precip_norm_mm: Optional[float]
    temp_mean_c: float
    soil_moisture_frac: Optional[float]
    sukhovey_days: int
    notes: list = field(default_factory=list)


def _opt(value) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def evaluate_decade(row: pd.Series,
                    degradation_index: Optional[float] = None,
                    pinn_raw_output: Optional[float] = None,
                    climatological_first_snow_doy: int = 300) -> DecadeRiskResult:
    """Считает три под-индекса и композит для одной декады."""
    precip_norm = _opt(row.get("precip_norm_mm"))
    precip_std = _opt(row.get("precip_std_mm"))
    temp_anom = _opt(row.get("temp_anom_c"))
    soil = _opt(row.get("soil_moisture_frac"))
    degradation = _opt(degradation_index)

    drought = drought_subindex(float(row["precip_mm_decade"]), precip_norm,
                               precip_std, soil, temp_anom if temp_anom is not None else 0.0)
    sukhovey = sukhovey_subindex(row.get("sukhovey_days", 0), int(row["n_days"]),
                                 float(row["wind_mean_ms"]), float(row["rel_humidity_pct"]))
    early_snow = early_snow_subindex(row["decade_start"], float(row["snowfall_sum_cm"]),
                                     float(row["temp_min_c"]), climatological_first_snow_doy)
    composite = compute_composite_index(drought, sukhovey, early_snow, degradation)

    notes = []
    if drought >= 60:
        notes.append("выраженный дефицит влаги — риск засухи")
    if sukhovey >= 60:
        notes.append("суховейные сутки — риск потери влаги и запала зерна")
    if early_snow >= 60:
        notes.append("риск раннего снега/заморозков до завершения уборки")
    if degradation is not None and degradation >= 0.75:
        notes.append("почва в верхнем диапазоне эрозионной уязвимости (PINN)")
    if not bool(row.get("is_complete", True)):
        notes.append(f"декада неполная: {int(row['n_days'])} из {int(row['expected_days'])} суток")

    return DecadeRiskResult(
        decade_label=row["decade_label"],
        decade_start=row["decade_start"],
        source=row.get("source", "факт"),
        n_days=int(row["n_days"]),
        drought=round(drought, 1),
        sukhovey=round(sukhovey, 1),
        early_snow=round(early_snow, 1),
        degradation_index=None if degradation is None else round(degradation, 3),
        pinn_raw_output=None if _opt(pinn_raw_output) is None else round(float(pinn_raw_output), 3),
        rusle_A_norm=None if _opt(row.get("rusle_A_norm")) is None else round(float(row["rusle_A_norm"]), 3),
        composite=round(composite, 1),
        category=risk_category(composite),
        precip_mm=round(float(row["precip_mm_decade"]), 1),
        precip_norm_mm=None if precip_norm is None else round(precip_norm, 1),
        temp_mean_c=round(float(row["temp_mean_c"]), 1),
        soil_moisture_frac=None if soil is None else round(soil, 3),
        sukhovey_days=int(row.get("sukhovey_days", 0)),
        notes=notes,
    )
