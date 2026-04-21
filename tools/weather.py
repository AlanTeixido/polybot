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
    "jeddah": (21.4858, 39.1925),
    "riyadh": (24.7136, 46.6753),
    "cairo": (30.0444, 31.2357),
    "lagos": (6.5244, 3.3792),
    "nairobi": (-1.2921, 36.8219),
    "cape town": (-33.9249, 18.4241),
    "johannesburg": (-26.2041, 28.0473),
    "lucknow": (26.8467, 80.9462),
    "delhi": (28.7041, 77.1025),
    "chennai": (13.0827, 80.2707),
    "kolkata": (22.5726, 88.3639),
    "kuala lumpur": (3.1390, 101.6869),
    "jakarta": (-6.2088, 106.8456),
    "manila": (14.5995, 120.9842),
    "ho chi minh": (10.8231, 106.6297),
    "chongqing": (29.4316, 106.9123),
    "shenzhen": (22.5431, 114.0579),
    "guangzhou": (23.1291, 113.2644),
    "beijing": (39.9042, 116.4074),
    "busan": (35.1796, 129.0756),
    "osaka": (34.6937, 135.5023),
    "buenos aires": (-34.6037, -58.3816),
    "sao paulo": (-23.5505, -46.6333),
    "rio de janeiro": (-22.9068, -43.1729),
    "lima": (-12.0464, -77.0428),
    "bogota": (4.7110, -74.0721),
    "santiago": (-33.4489, -70.6693),
    "melbourne": (-37.8136, 144.9631),
    "auckland": (-36.8485, 174.7633),
    "milan": (45.4642, 9.1900),
    "amsterdam": (52.3676, 4.9041),
    "vienna": (48.2082, 16.3738),
    "warsaw": (52.2297, 21.0122),
    "moscow": (55.7558, 37.6173),
    # Added 2026-04-21 after observing parse_failed from these cities in log:
    "wellington": (-41.2865, 174.7762),
    "panama city": (8.9824, -79.5199),
    "tel aviv": (32.0853, 34.7818),
    "chengdu": (30.5728, 104.0668),
    # Preventive additions (common weather-market cities worldwide):
    "stockholm": (59.3293, 18.0686),
    "copenhagen": (55.6761, 12.5683),
    "oslo": (59.9139, 10.7522),
    "helsinki": (60.1699, 24.9384),
    "dublin": (53.3498, -6.2603),
    "lisbon": (38.7223, -9.1393),
    "barcelona": (41.3851, 2.1734),
    "athens": (37.9838, 23.7275),
    "prague": (50.0755, 14.4378),
    "budapest": (47.4979, 19.0402),
    "zurich": (47.3769, 8.5417),
    "brussels": (50.8503, 4.3517),
    "taipei": (25.0330, 121.5654),
    "hanoi": (21.0285, 105.8542),
    "tianjin": (39.3434, 117.3616),
    "hangzhou": (30.2741, 120.1551),
    "wuhan": (30.5928, 114.3055),
    "caracas": (10.4806, -66.9036),
    "havana": (23.1136, -82.3666),
    "accra": (5.6037, -0.1870),
    "addis ababa": (9.0054, 38.7636),
    "ankara": (39.9334, 32.8597),
    "tehran": (35.6892, 51.3890),
    "karachi": (24.8607, 67.0011),
    "dhaka": (23.8103, 90.4125),
    "bangalore": (12.9716, 77.5946),
    "hyderabad": (17.3850, 78.4867),
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


def _estimate_probability(forecast_temp: float, threshold: float, comparison: str = "above", days_ahead: int = 1) -> float:
    """Estimate probability that actual temp will be above/below/equal to threshold.

    Uses a normal distribution around the forecast with std_dev that grows
    with forecast horizon — matching real forecast error patterns:
    - 1 day: ~1.5°C (very accurate)
    - 3 days: ~2.2°C
    - 7 days: ~3.8°C
    """
    import math

    # Dynamic forecast error: grows linearly with days ahead.
    # Widened on 2026-04-20 after observing 41% drift on N=9 trades:
    # model predicted ~0.94 avg, actual WR was 0.44. Old std_dev (1.5°C base,
    # 1.8°C at 2d) underestimated forecast uncertainty. Real MAE for 2-day
    # forecasts is ~2-3°C → std_dev ~2.5-3°C. Bumping base + slope.
    std_dev = max(2.5, 1.5 + days_ahead * 0.5)

    if std_dev == 0:
        return 1.0 if forecast_temp >= threshold else 0.0

    # Z-score: how many std devs the threshold is from forecast
    z = (threshold - forecast_temp) / std_dev

    # CDF using error function (no scipy needed)
    prob_below_threshold = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    prob_above_threshold = 1 - prob_below_threshold

    if comparison == "above":
        return round(prob_above_threshold, 3)
    elif comparison == "below":
        return round(prob_below_threshold, 3)
    elif comparison == "equal":
        # P(temp == X) ≈ P(X-0.5 < temp < X+0.5)
        p_low = 0.5 * (1 + math.erf((threshold - 0.5 - forecast_temp) / (std_dev * math.sqrt(2))))
        p_high = 0.5 * (1 + math.erf((threshold + 0.5 - forecast_temp) / (std_dev * math.sqrt(2))))
        return round(p_high - p_low, 3)
    else:
        return round(prob_above_threshold, 3)


def get_weather_forecast(
    city: str,
    target_date: str = "",
    threshold_c: float | None = None,
    threshold_c_high: float | None = None,
    comparison: str = "above",
    metric: str = "high",
) -> dict[str, Any]:
    """Get weather forecast and optionally calculate probability vs a threshold.

    Args:
        city: City name (e.g. "Atlanta", "Shanghai")
        target_date: Date to forecast (YYYY-MM-DD). Optional.
        threshold_c: Temperature threshold in °C. For ranges, this is the LOW bound.
        threshold_c_high: For range markets ("between X-Y"), the HIGH bound in °C.
        comparison: "above", "below", "equal", or "range". Use "range" with threshold_c + threshold_c_high.
        metric: "high" for max temp, "low" for min temp.

    Returns dict with:
        - forecasts: list of {date, high_c, low_c, high_f, low_f, source}
        - probability: P(temp [comparison] threshold) if threshold_c provided
        - edge_info: {forecast_temp_c, threshold_c, metric, probability, comparison}
    """
    coords = _find_city_coords(city)
    if not coords:
        available = ', '.join(sorted(CITY_COORDS.keys())[:15])
        return {"error": f"City '{city}' not found. Available: {available}..."}

    lat, lon = coords
    is_us = -130 < lon < -60 and 24 < lat < 50

    if is_us:
        result = _get_nws_forecast(lat, lon, city, target_date)
    else:
        result = _get_open_meteo_forecast(lat, lon, city, target_date)

    if "error" in result:
        return result

    # Calculate probability if threshold provided
    if threshold_c is not None and result.get("forecasts"):
        # Pick the right metric: high (max) or low (min)
        if metric == "low":
            temps = [f.get("low_c") for f in result["forecasts"] if f.get("low_c") is not None]
        else:
            temps = [f.get("high_c") or f.get("temperature_c") for f in result["forecasts"]
                     if (f.get("high_c") or f.get("temperature_c")) is not None
                     and f.get("is_daytime", True)]
        if temps:
            forecast_temp = temps[0]

            # Calculate days_ahead for dynamic uncertainty
            days_ahead = 1
            if target_date and result.get("forecasts"):
                try:
                    from datetime import datetime as _dt, timezone as _tz
                    target_dt = _dt.fromisoformat(target_date).replace(tzinfo=_tz.utc)
                    days_ahead = max(1, (target_dt - _dt.now(_tz.utc)).days)
                except Exception:
                    days_ahead = 3  # fallback

            # Range calculation: P(low <= temp <= high)
            if comparison == "range" and threshold_c_high is not None:
                import math as _math
                std_dev = max(2.5, 1.5 + days_ahead * 0.5)  # widened — see estimate_probability
                z_low = (threshold_c - forecast_temp) / std_dev
                z_high = (threshold_c_high - forecast_temp) / std_dev
                p_low = 0.5 * (1 + _math.erf(z_low / _math.sqrt(2)))
                p_high = 0.5 * (1 + _math.erf(z_high / _math.sqrt(2)))
                prob = round(p_high - p_low, 3)
            else:
                prob = _estimate_probability(forecast_temp, threshold_c, comparison, days_ahead=days_ahead)

            result["probability"] = prob
            result["days_ahead"] = days_ahead
            result["edge_info"] = {
                "forecast_temp_c": forecast_temp,
                "threshold_c": threshold_c,
                "threshold_c_high": threshold_c_high,
                "metric": metric,
                "comparison": comparison,
                "probability": prob,
                "days_ahead": days_ahead,
                "std_dev_used": round(max(1.5, 1.0 + days_ahead * 0.4), 2),
                "confidence": "high" if abs(forecast_temp - threshold_c) > 5 else "medium" if abs(forecast_temp - threshold_c) > 2 else "low",
            }

    return result


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

        # Group by date: daytime = high, nighttime = low
        by_date: dict[str, dict] = {}
        for p in periods[:14]:
            date = p.get("startTime", "")[:10]
            if not date:
                continue
            if date not in by_date:
                by_date[date] = {"date": date, "high_c": None, "low_c": None, "high_f": None, "low_f": None, "forecast": "", "is_daytime": True}
            temp_f = p.get("temperature", 32)
            temp_c = round((temp_f - 32) * 5 / 9, 1)
            if p.get("isDaytime", True):
                by_date[date]["high_f"] = temp_f
                by_date[date]["high_c"] = temp_c
                by_date[date]["forecast"] = p.get("shortForecast", "")
            else:
                by_date[date]["low_f"] = temp_f
                by_date[date]["low_c"] = temp_c

        forecasts = list(by_date.values())
        if target_date:
            forecasts = [f for f in forecasts if f["date"] == target_date]

        return {
            "city": city,
            "source": "NWS",
            "forecasts": forecasts,
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
            high_c = highs[i] if i < len(highs) else None
            low_c = lows[i] if i < len(lows) else None
            forecasts.append({
                "date": date,
                "high_c": high_c,
                "low_c": low_c,
                "high_f": round(high_c * 9 / 5 + 32, 1) if high_c is not None else None,
                "low_f": round(low_c * 9 / 5 + 32, 1) if low_c is not None else None,
                "forecast": "",
                "is_daytime": True,
            })

        if target_date:
            forecasts = [f for f in forecasts if f["date"] == target_date]

        return {
            "city": city,
            "source": "Open-Meteo",
            "forecasts": forecasts,
        }

    except Exception as e:
        logger.warning(f"Open-Meteo forecast failed for {city}: {e}")
        return {"error": str(e), "city": city}
