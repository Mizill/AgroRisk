"""
features.py
===========
Сборка 13 входных признаков нейросети из декадных данных Open-Meteo.

Это тот слой, которого раньше не было: без него сеть невозможно было
запустить, поэтому в отчёт уходил пустой `degradation_index`.

Три группы признаков:

1. Константы RUSLE из самого чекпойнта — K (эродируемость), LS (склон),
   P (агротехника). Их не надо ниоткуда тянуть, они лежат в pinn_model.pt.
2. R и C — считаются по метеоданным (R) и по фенологии культуры (C).
3. Метеопризнаки 5-12 — прямо из декадной агрегации Open-Meteo,
   нормируются в [0, 1] по таблице `FEATURE_RANGES`.

Масштабирование признаков
-------------------------
Scaler обучающей выборки в чекпойнте не сохранён, поэтому масштаб подобран
так, чтобы согласовываться с физикой самого чекпойнта:

  * `max_rusle = 4.2866` при K=0.3, LS=1.19, P=0.8 означает, что при C≤1
    величина R на обучении доходила примерно до 15-60 — то есть R подавался
    в физических единицах, а не нормированным в [0,1]. Поэтому RUSLE-факторы
    (признаки 0-4) остаются в натуральных единицах.
  * Метеопризнаки (5-12) нормируются min-max по явным физическим диапазонам:
    из проверенных вариантов именно такой даёт наибольшее согласие выхода
    сети с её собственным физическим остатком A/max_rusle.

Все диапазоны собраны в одном словаре ниже — их можно поменять под свой
регион, не трогая остальной код.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from pinn_model import INPUT_FEATURES

# --- физические диапазоны для min-max нормировки признаков 5-12 ------------
FEATURE_RANGES = {
    "precip_mm_decade":   (0.0, 80.0),    # мм за декаду
    "precip_norm_ratio":  (0.0, 2.0),     # доля от климатической нормы
    "temp_mean_c":        (-30.0, 40.0),  # °C
    "temp_anom_c":        (-10.0, 10.0),  # °C
    "wind_mean_ms":       (0.0, 15.0),    # м/с
    "wind_max_ms":        (0.0, 35.0),    # м/с
    "rel_humidity_pct":   (0.0, 100.0),   # %
    "soil_moisture_frac": (0.0, 0.5),     # м³/м³
}

# --- параметры эрозивности осадков (модель типа Richardson) ---------------
R_ALPHA = 0.29          # коэффициент суточной эрозивности
R_BETA = 1.8            # показатель степени
R_RAIN_THRESHOLD = 10.0  # мм: ниже этого порога дождь считается неэрозионным
C_REFERENCE = 0.5        # опорный C для расчёта потолка R из max_rusle


def minmax(value, lo: float, hi: float) -> np.ndarray:
    """Линейная нормировка в [0, 1] с обрезкой по краям."""
    value = np.asarray(value, dtype=np.float64)
    if hi <= lo:
        return np.zeros_like(value)
    return np.clip((value - lo) / (hi - lo), 0.0, 1.0)


def r_factor_cap(physics: dict, c_reference: float = C_REFERENCE) -> float:
    """
    Потолок R, при котором A = R*K*LS*C*P не вылезает за max_rusle
    чекпойнта при типичном покрове. Для констант чекпойнта даёт ≈30.
    """
    denom = physics["K_factor"] * physics["LS_factor"] * physics["P_factor"] * c_reference
    if denom <= 0:
        return float("inf")
    return float(physics["max_rusle"] / denom)


def rainfall_erosivity(daily_precip_mm, threshold: float = R_RAIN_THRESHOLD,
                       alpha: float = R_ALPHA, beta: float = R_BETA) -> float:
    """
    Эрозивность осадков за декаду: R = Σ α·P^β по эрозионным суткам
    (суточная модель EI30, Richardson et al.). Несколько сильных дождей
    дают вклад намного больше, чем та же сумма мороси.
    """
    p = np.asarray(list(daily_precip_mm), dtype=np.float64)
    p = p[np.isfinite(p) & (p >= threshold)]
    if p.size == 0:
        return 0.0
    return float(alpha * np.sum(p ** beta))


# ---------------------------------------------------------------------------
# Фактор растительного покрова C
# ---------------------------------------------------------------------------
# Сезонная кривая зелёного покрова для ярового зерновой ротации севера
# Казахстана: (день года, доля покрытия 0..1). Между узлами — линейная
# интерполяция. Если у вас есть NDVI — подставьте его через --cover.
COVER_CURVE_DOY = [1, 105, 135, 160, 190, 225, 260, 290, 366]
COVER_CURVE_VAL = [0.02, 0.05, 0.10, 0.45, 0.85, 0.80, 0.30, 0.05, 0.02]


def seasonal_cover(decade_start: dt.date) -> float:
    """Доля зелёного покрова по фенологическому календарю (0 = голая почва)."""
    doy = decade_start.timetuple().tm_yday
    return float(np.interp(doy, COVER_CURVE_DOY, COVER_CURVE_VAL))


def cover_fraction(decade_start: dt.date, temp_mean_c: float,
                   precip_norm_ratio: float) -> float:
    """
    Покров = сезонная кривая, поправленная на тепло и влагообеспеченность:
    в холодной декаде растительность не развивается, в засушливой — редеет.
    """
    cover = seasonal_cover(decade_start)
    if np.isfinite(temp_mean_c):
        if temp_mean_c < 0:
            cover *= 0.15
        elif temp_mean_c < 5:
            cover *= 0.50
    if np.isfinite(precip_norm_ratio):
        cover *= float(np.clip(0.5 + 0.5 * precip_norm_ratio, 0.4, 1.2))
    return float(np.clip(cover, 0.0, 1.0))


def c_factor_from_cover(cover: float) -> float:
    """
    Перевод доли покрова в RUSLE C-фактор: голая пашня ≈ 1.0,
    сомкнутый травостой ≈ 0.1.
    """
    return float(np.clip(1.0 - 0.9 * float(cover), 0.03, 1.0))


@dataclass
class FeatureConfig:
    """Настройки сборки признаков (все пробрасываются из CLI)."""
    cover_override: Optional[float] = None   # фиксированная доля покрова 0..1
    c_factor_override: Optional[float] = None  # фиксированный RUSLE C
    rain_threshold: float = R_RAIN_THRESHOLD
    r_alpha: float = R_ALPHA
    r_beta: float = R_BETA
    r_max: Optional[float] = None            # потолок R; None — считать из физики


def build_feature_frame(decades: pd.DataFrame, physics: dict,
                        config: Optional[FeatureConfig] = None) -> pd.DataFrame:
    """
    Достраивает декадную таблицу колонками R_factor, C_factor, cover_frac,
    rusle_A, rusle_A_norm и всеми 13 признаками нейросети (суффикс `f_`).

    Ожидает колонки из `decadal_risk.aggregate_to_decades` плюс
    `precip_norm_mm` / `temp_norm_c` из климатологии.
    """
    config = config or FeatureConfig()
    df = decades.copy()
    r_cap = config.r_max if config.r_max is not None else r_factor_cap(physics)

    precip_norm = pd.to_numeric(df.get("precip_norm_mm"), errors="coerce")
    ratio = df["precip_mm_decade"] / precip_norm.where(precip_norm > 0)
    df["precip_norm_ratio"] = ratio.fillna(1.0).clip(0, 5)

    temp_norm = pd.to_numeric(df.get("temp_norm_c"), errors="coerce")
    df["temp_anom_c"] = (df["temp_mean_c"] - temp_norm.fillna(df["temp_mean_c"])).fillna(0.0)

    # --- R: эрозивность осадков ------------------------------------------
    df["R_factor"] = [
        min(rainfall_erosivity(days, config.rain_threshold, config.r_alpha, config.r_beta), r_cap)
        for days in df["daily_precip_mm"]
    ]

    # --- C: растительный покров ------------------------------------------
    if config.c_factor_override is not None:
        df["cover_frac"] = np.clip((1.0 - config.c_factor_override) / 0.9, 0.0, 1.0)
        df["C_factor"] = float(np.clip(config.c_factor_override, 0.03, 1.0))
    elif config.cover_override is not None:
        cover = float(np.clip(config.cover_override, 0.0, 1.0))
        df["cover_frac"] = cover
        df["C_factor"] = c_factor_from_cover(cover)
    else:
        df["cover_frac"] = [
            cover_fraction(row.decade_start, row.temp_mean_c, row.precip_norm_ratio)
            for row in df.itertuples()
        ]
        df["C_factor"] = [c_factor_from_cover(c) for c in df["cover_frac"]]

    # --- константы чекпойнта ---------------------------------------------
    df["K_factor"] = physics["K_factor"]
    df["LS_factor"] = physics["LS_factor"]
    df["P_factor"] = physics["P_factor"]

    # --- физическая оценка RUSLE (для сверки с сетью) ---------------------
    df["rusle_A"] = (df["R_factor"] * physics["K_factor"] * physics["LS_factor"]
                     * df["C_factor"] * physics["P_factor"])
    df["rusle_A_norm"] = (df["rusle_A"] / physics["max_rusle"]).clip(0, 1)

    # --- итоговые 13 признаков -------------------------------------------
    df["f_R_factor"] = df["R_factor"]
    df["f_K_factor"] = df["K_factor"]
    df["f_LS_factor"] = df["LS_factor"]
    df["f_C_factor"] = df["C_factor"]
    df["f_P_factor"] = df["P_factor"]
    for name, (lo, hi) in FEATURE_RANGES.items():
        source = df[name].astype(float)
        df[f"f_{name}"] = minmax(source.fillna(source.median()).fillna(lo), lo, hi)
    return df


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Достаёт из таблицы матрицу [N, 13] в порядке INPUT_FEATURES."""
    missing = [n for n in INPUT_FEATURES if f"f_{n}" not in df.columns]
    if missing:
        raise KeyError(f"Не собраны признаки: {', '.join(missing)}")
    cols = [f"f_{name}" for name in INPUT_FEATURES]
    return df[cols].to_numpy(dtype=np.float64)
