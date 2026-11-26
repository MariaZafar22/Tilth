"""Farming tools the agent can call (function calling).

Every function has type hints and a docstring because Gemini reads them
to decide when and how to call each tool.
"""
from __future__ import annotations

from datetime import date

import requests

import rag

# Indicative nutrient need in kg per acre: (N, P2O5, K2O).
# Always confirm with a soil test and the local agriculture extension office.
CROP_NUTRIENTS = {
    "wheat": (49, 36, 24),
    "rice": (45, 28, 24),
    "cotton": (61, 24, 24),
    "sugarcane": (68, 45, 45),
    "maize": (100, 50, 40),
}

# Sowing windows for Punjab, Pakistan: (label, (start_month, start_day), (end_month, end_day))
SOWING_WINDOWS = {
    "wheat": [("Main season", (11, 1), (11, 30))],
    "rice": [("Nursery sowing", (5, 20), (6, 15)), ("Transplanting", (6, 20), (7, 31))],
    "cotton": [("Main season", (4, 1), (5, 31))],
    "sugarcane": [("Autumn crop", (9, 15), (10, 31)), ("Spring crop", (2, 15), (3, 31))],
    "maize": [("Spring crop", (2, 1), (3, 15)), ("Autumn crop", (7, 15), (8, 15))],
}


def _clean_crop(crop: str) -> str | None:
    crop = crop.strip().lower()
    return crop if crop in CROP_NUTRIENTS else None


def search_crop_guides(query: str, top_k: int = 3) -> list[dict]:
    """Search the crop knowledge base (wheat, rice, cotton, sugarcane, maize,
    soil, irrigation, pests) and return the most relevant passages.

    Args:
        query: What to look up, e.g. "wheat rust symptoms and control".
        top_k: How many passages to return (1-5).
    """
    top_k = max(1, min(int(top_k), 5))
    return rag.search(query, top_k=top_k)


def fertilizer_plan(crop: str, acres: float) -> dict:
    """Calculate how much Urea, DAP and MOP (potash) a crop needs for a field.

    Args:
        crop: One of wheat, rice, cotton, sugarcane, maize.
        acres: Field size in acres (must be above 0).
    """
    key = _clean_crop(crop)
    if key is None:
        return {"error": f"Unsupported crop '{crop}'. Choose from: {', '.join(CROP_NUTRIENTS)}."}
    if acres <= 0:
        return {"error": "Acres must be greater than 0."}

    n, p, k = CROP_NUTRIENTS[key]
    dap_kg = p * acres / 0.46          # DAP = 18% N, 46% P2O5
    n_from_dap = dap_kg * 0.18
    urea_kg = max(0.0, n * acres - n_from_dap) / 0.46   # Urea = 46% N
    mop_kg = k * acres / 0.60          # MOP = 60% K2O

    def pack(kg: float) -> dict:
        return {"kg": round(kg, 1), "bags_50kg": round(kg / 50, 2)}

    return {
        "crop": key,
        "acres": acres,
        "nutrient_target_kg_per_acre": {"N": n, "P2O5": p, "K2O": k},
        "dap": pack(dap_kg),
        "urea": pack(urea_kg),
        "mop_potash": pack(mop_kg),
        "advice": "Apply DAP and potash at sowing. Split urea into 2-3 doses. "
                  "These are indicative rates; confirm with a soil test.",
    }


def sowing_window(crop: str) -> dict:
    """Tell the user the recommended sowing window for a crop in Punjab and
    whether today falls inside it.

    Args:
        crop: One of wheat, rice, cotton, sugarcane, maize.
    """
    key = _clean_crop(crop)
    if key is None:
        return {"error": f"Unsupported crop '{crop}'. Choose from: {', '.join(CROP_NUTRIENTS)}."}

    today = date.today()
    windows = []
    open_now = []
    for label, (m1, d1), (m2, d2) in SOWING_WINDOWS[key]:
        start, end = date(today.year, m1, d1), date(today.year, m2, d2)
        windows.append(f"{label}: {start.strftime('%d %b')} to {end.strftime('%d %b')}")
        if start <= today <= end:
            open_now.append(label)
    return {
        "crop": key,
        "region": "Punjab, Pakistan",
        "windows": windows,
        "today": today.isoformat(),
        "open_now": open_now or "No window is open today",
    }


def get_weather_forecast(city: str, days: int = 5) -> dict:
    """Get a short weather forecast for a city and a farming advisory
    (spray, irrigation and heat or frost warnings).

    Args:
        city: City or town name, e.g. "Faisalabad".
        days: Number of forecast days (1-7).
    """
    days = max(1, min(int(days), 7))
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=10,
        ).json()
        if not geo.get("results"):
            return {"error": f"Could not find a place called '{city}'."}
        place = geo["results"][0]
        data = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"], "longitude": place["longitude"],
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto", "forecast_days": days,
            }, timeout=10,
        ).json()["daily"]
    except (requests.RequestException, KeyError, ValueError) as exc:
        return {"error": f"Weather service unavailable right now ({type(exc).__name__})."}

    forecast = [
        {"date": d, "max_c": hi, "min_c": lo, "rain_mm": rain}
        for d, hi, lo, rain in zip(
            data["time"], data["temperature_2m_max"],
            data["temperature_2m_min"], data["precipitation_sum"])
    ]
    return {"city": place["name"], "forecast": forecast, "advisory": _advisory(forecast)}


def _advisory(forecast: list[dict]) -> list[str]:
    notes = []
    rainy = [f["date"] for f in forecast if (f["rain_mm"] or 0) >= 2]
    if rainy:
        notes.append(f"Rain expected on {', '.join(rainy)}: avoid spraying and skip irrigation or fertilizer top-dressing before it.")
    else:
        notes.append("No significant rain expected: good window for spraying and field work.")
    if any((f["max_c"] or 0) >= 40 for f in forecast):
        notes.append("Heat of 40 C or more: irrigate in the evening and watch for heat stress.")
    if any((f["min_c"] or 99) <= 2 for f in forecast):
        notes.append("Near-frost nights: protect nurseries and vegetables, consider a light irrigation.")
    return notes


TOOLS = [search_crop_guides, fertilizer_plan, sowing_window, get_weather_forecast]
