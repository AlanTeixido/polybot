"""NWS Weather API for temperature forecast markets.

Uses api.weather.gov (free, no API key needed) to get actual forecasts
and compare against market prices. The #1 most profitable strategy on Simmer.
"""

import logging
import time
from typing import Any

import requests

logger = logging.getLogger("polybot.weather")

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "(Polybot, polybot@trading)",
    "Accept": "application/geo+json",
})
REQUEST_TIMEOUT = 10

# Cache grid lookups (lat/lon → grid endpoint) to respect NWS recommendations
_grid_cache: dict[str, dict] = {}

# Major cities with their coordinates
CITY_COORDS: dict[str, tuple[float, float]] = {
    "new york": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "houston": (29.7604, -95.3698),
    "phoenix": (33.4484, -112.0740),
    "philadelphia": (39.9526, -75.1652),
    "san antonio": (29.4241, -98.4936),
    "san diego": (32.7157, -117.1611),
    "dallas": (32.7767, -96.7970),
    "miami": (25.7617, -80.1918),
    "atlanta": (33.7490, -84.3880),
    "boston": (42.3601, -71.0589),
    "seattle": (47.6062, -122.3321),
    "denver": (39.7392, -104.9903),
    "washington": (38.9072, -77.0369),
    "las vegas": (36.1699, -115.1398),
    "portland": (45.5152, -122.6784),
    "detroit": (42.3314, -83.0458),
    "minneapolis": (44.9778, -93.2650),
    "tampa": (27.9506, -82.4572),
    "san francisco": (37.7749, -122.4194),
    "nashville": (36.1627, -86.7816),
    "austin": (30.2672, -97.7431),
    "london": (51.5074, -0.1278),
    "paris": (48.8566, 2.3522),
    "tokyo": (35.6762, 139.6503),
    "shanghai": (31.2304, 121.4737),
    "istanbul": (41.0082, 28.9784),
    "hong kong": (22.3193, 114.1694),
    "singapore": (1.3521, 103.8198),
    "sydney": (-33.8688, 151.2093),
    "berlin": (52.5200, 13.4050),
    "madrid": (40.4168, -3.7038),
    "rome": (41.9028, 12.4964),
    "mumbai": (19.0760, 72.8777),
    "dubai": (25.2048, 55.2708),
    "toronto": (43.6532, -79.3832),
    "mexico city": (19.4326, -99.1332),
    "seoul": (37.5665, 126.9780),
    "bangkok": (13.7563, 100.5018),
}


def _find_city_coords(city_name: str) -> tuple[float, float] | None:
    """Find coordinates for a city name (fuzzy match)."""
    city_lower = city_name.lower().strip()
    # Exact match
    if city_lower in CITY_COORDS:
        return CITY_COORDS[city_lower]
    # Partial match
    for name, coords in CITY_COORDS.items():
        if name in city_lower or city_lower in name:
            return coords
    return None


def _get_nws_grid(lat: float, lon: float) -> dict | None:
    """Get NWS grid point for lat/lon (US only). Cached."""
    cache_key = f"{lat:.2f},{lon:.2f}"
    if cache_key in _grid_cache:
        return _grid_cache[cache_key]

    try:
        resp = SESSION.get(
            f"https://api.weather.gov/points/{lat},{lon}",
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        props = data.get("properties", {})
        grid = {
            "office": props.get("gridId", ""),
            "gridX": props.get("gridX"),
            "gridY": props.get("gridY"),
            "forecast_url": props.get("forecast", ""),
            "hourly_url": props.get("forecastHourly", ""),
        }
        _grid_cache[cache_key] = grid
        return grid
    except Exception as e:
        logger.warning(f"NWS grid lookup failed: {e}")
        return None


def get_weather_forecast(city: str, target_date: str = "") -> dict[str, Any]:
    """Get weather forecast for a city. Returns high/low temps and conditions.

    Args:
        city: City name (e.g. "Atlanta", "New York")
        target_date: Optional date string (e.g. "2026-04-05"). If empty, returns next 7 days.
    """
    coords = _find_city_coords(city)
    if not coords:
        return {"error": f"City '{city}' not found in database. Available: {', '.join(sorted(CITY_COORDS.keys())[:10])}..."}

    lat, lon = coords
    is_us = -130 < lon < -60 and 24 < lat < 50  # Rough US bounds

    # NWS only works for US locations
    if is_us:
        return _get_nws_forecast(lat, lon, city, target_date)
    else:
        # For non-US, use Open-Meteo (free, no key)
        return _get_open_meteo_forecast(lat, lon, city, target_date)


def _get_nws_forecast(lat: float, lon: float, city: str, target_date: str) -> dict[str, Any]:
    """Get forecast from NWS (US only, free, highly accurate)."""
    grid = _get_nws_grid(lat, lon)
    if not grid or not grid.get("forecast_url"):
        return {"error": "NWS grid lookup failed", "city": city}

    try:
        resp = SESSION.get(grid["forecast_url"], timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        periods = data.get("properties", {}).get("periods", [])

        if not periods:
            return {"error": "No forecast periods", "city": city}

        forecasts = []
        for p in periods[:14]:  # 7 days (day + night)
            forecasts.append({
                "name": p.get("name", ""),
                "date": p.get("startTime", "")[:10],
                "temperature_f": p.get("temperature"),
                "temperature_c": round((p.get("temperature", 32) - 32) * 5 / 9, 1),
                "wind_speed": p.get("windSpeed", ""),
                "short_forecast": p.get("shortForecast", ""),
                "is_daytime": p.get("isDaytime", True),
            })

        # Filter to target date if specified
        if target_date:
            forecasts = [f for f in forecasts if f["date"] == target_date]

        return {
            "city": city,
            "source": "NWS (api.weather.gov)",
            "forecasts": forecasts,
            "high_temps_c": [f["temperature_c"] for f in forecasts if f["is_daytime"]],
            "low_temps_c": [f["temperature_c"] for f in forecasts if not f["is_daytime"]],
        }

    except Exception as e:
        logger.warning(f"NWS forecast failed for {city}: {e}")
        return {"error": str(e), "city": city}


def _get_open_meteo_forecast(lat: float, lon: float, city: str, target_date: str) -> dict[str, Any]:
    """Get forecast from Open-Meteo (global, free, no key needed)."""
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": 7,
        }
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})

        dates = daily.get("time", [])
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])
        precip_prob = daily.get("precipitation_probability_max", [])

        forecasts = []
        for i, date in enumerate(dates):
            f = {
                "date": date,
                "high_c": highs[i] if i < len(highs) else None,
                "low_c": lows[i] if i < len(lows) else None,
                "high_f": round(highs[i] * 9 / 5 + 32, 1) if i < len(highs) and highs[i] is not None else None,
                "precip_probability": precip_prob[i] if i < len(precip_prob) else None,
            }
            forecasts.append(f)

        # Filter to target date if specified
        if target_date:
            forecasts = [f for f in forecasts if f["date"] == target_date]

        return {
            "city": city,
            "source": "Open-Meteo (open-meteo.com)",
            "forecasts": forecasts,
            "high_temps_c": [f["high_c"] for f in forecasts if f["high_c"] is not None],
        }

    except Exception as e:
        logger.warning(f"Open-Meteo forecast failed for {city}: {e}")
        return {"error": str(e), "city": city}
