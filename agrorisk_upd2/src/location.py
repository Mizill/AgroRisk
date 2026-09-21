"""
location.py
===========
Всё, что избавляет от ручного ввода параметров:

  * поиск координат по названию населённого пункта (геокодер Open-Meteo);
  * автопоиск файла `pinn_model.pt` рядом с программой;
  * запоминание последней точки в `~/.agrorisk.json`, чтобы в следующий раз
    хватило одного Enter.

Ничего из этого не требует ключей и не обращается к сторонним сервисам —
геокодер тот же Open-Meteo.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".agrorisk.json")
MODEL_FILENAME = "pinn_model.pt"

# «51.17 71.45», «51.17, 71.45», «51,17 71,45» — всё это координаты
_COORD_RE = re.compile(
    r"^\s*(-?\d{1,3}(?:[.,]\d+)?)\s*[,; ]\s*(-?\d{1,3}(?:[.,]\d+)?)\s*$")


@dataclass
class Place:
    """Точка расчёта: координаты плюс человекочитаемое название."""
    lat: float
    lon: float
    title: str = ""

    def describe(self) -> str:
        coords = f"{self.lat:.4f}, {self.lon:.4f}"
        return f"{self.title} ({coords})" if self.title else coords


# ---------------------------------------------------------------------------
# Разбор координат, введённых текстом
# ---------------------------------------------------------------------------
def parse_coordinates(text: str) -> Optional[Place]:
    """Пытается прочитать строку как пару координат. Иначе возвращает None."""
    match = _COORD_RE.match(text or "")
    if not match:
        return None
    try:
        lat = float(match.group(1).replace(",", "."))
        lon = float(match.group(2).replace(",", "."))
    except ValueError:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return Place(lat=lat, lon=lon)


def format_geocoding_hit(hit: dict) -> str:
    """Человекочитаемое название найденной точки: «Акколь, Акмолинская область, Казахстан»."""
    parts = [hit.get("name")]
    for key in ("admin1", "country"):
        value = hit.get(key)
        if value and value not in parts:
            parts.append(value)
    return ", ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Поиск чекпойнта
# ---------------------------------------------------------------------------
def find_model(explicit: Optional[str] = None,
               start_dir: Optional[str] = None) -> Optional[str]:
    """
    Ищет `pinn_model.pt`: сначала явно указанный путь, затем текущая папка,
    папка программы и на уровень выше каждой из них.
    """
    if explicit:
        return explicit if os.path.isfile(explicit) else None

    here = start_dir or os.path.dirname(os.path.abspath(__file__))
    candidates = []
    for base in (os.getcwd(), here, os.path.dirname(here)):
        candidates.append(os.path.join(base, MODEL_FILENAME))
        candidates.append(os.path.join(os.path.dirname(base), MODEL_FILENAME))
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    return None


# ---------------------------------------------------------------------------
# Сохранённые настройки
# ---------------------------------------------------------------------------
def load_settings(path: str = SETTINGS_PATH) -> dict:
    """Читает прошлые настройки. Любая ошибка означает «настроек нет»."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(settings: dict, path: str = SETTINGS_PATH) -> bool:
    """Сохраняет настройки. Возвращает False, если записать не удалось."""
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, ensure_ascii=False, indent=2)
        return True
    except OSError:
        return False


def place_from_settings(settings: dict) -> Optional[Place]:
    try:
        return Place(lat=float(settings["lat"]), lon=float(settings["lon"]),
                     title=str(settings.get("title", "")))
    except (KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Определение точки расчёта
# ---------------------------------------------------------------------------
def resolve_place(client, query: str, choose=None) -> Place:
    """
    Превращает текст запроса в точку: либо это координаты, либо название
    населённого пункта, которое ищется геокодером Open-Meteo.

    `choose(hits) -> int` вызывается, когда совпадений несколько; без него
    берётся первое (самое населённое) совпадение.
    """
    direct = parse_coordinates(query)
    if direct is not None:
        return direct

    hits = client.search_place(query)
    if not hits:
        raise ValueError(
            f"Населённый пункт «{query}» не найден. Уточните название или "
            f"введите координаты, например: 51.17 71.45")

    index = 0
    if len(hits) > 1 and choose is not None:
        index = max(0, min(int(choose(hits)), len(hits) - 1))
    hit = hits[index]
    return Place(lat=float(hit["latitude"]), lon=float(hit["longitude"]),
                 title=format_geocoding_hit(hit))
