"""Weather-only trading bot for Simmer/Polymarket.

Pure Python, no LLM. Lorenzo-style strategy:
1. Fetch weather markets from Simmer
2. Parse city, date, threshold from title
3. Call NWS/Open-Meteo for real forecast
4. Calculate P(event) with normal distribution
5. Trade if |P_real - price| > min_edge

Runs 24/7. Deterministic. Auditable.
"""

import json
import logging
import math
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

import requests

# ---------------------------------------------------------------------------
# Config & logging
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
LOG_PATH = os.path.join(HERE, "weather-bot.log")
STATE_PATH = os.path.join(HERE, "state.json")

# Project root on sys.path so we can reuse shared helpers from tools/
# (same pattern as polybot-crypto/bot.py).
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.polymarket import get_positions  # noqa: E402

logger = logging.getLogger("weather-bot")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
_fh = RotatingFileHandler(LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
_fh.setFormatter(_fmt)
logger.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

SIMMER_API = "https://api.simmer.markets/api/sdk"

# ---------------------------------------------------------------------------
# City database (expand as needed)
# ---------------------------------------------------------------------------
CITY_COORDS: dict[str, tuple[float, float]] = {
    "new york": (40.7128, -74.0060), "new york city": (40.7128, -74.0060), "nyc": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437), "chicago": (41.8781, -87.6298),
    "houston": (29.7604, -95.3698), "phoenix": (33.4484, -112.0740),
    "philadelphia": (39.9526, -75.1652), "san antonio": (29.4241, -98.4936),
    "san diego": (32.7157, -117.1611), "dallas": (32.7767, -96.7970),
    "miami": (25.7617, -80.1918), "atlanta": (33.7490, -84.3880),
    "boston": (42.3601, -71.0589), "seattle": (47.6062, -122.3321),
    "denver": (39.7392, -104.9903), "washington": (38.9072, -77.0369),
    "las vegas": (36.1699, -115.1398), "portland": (45.5152, -122.6784),
    "detroit": (42.3314, -83.0458), "minneapolis": (44.9778, -93.2650),
    "tampa": (27.9506, -82.4572), "san francisco": (37.7749, -122.4194),
    "nashville": (36.1627, -86.7816), "austin": (30.2672, -97.7431),
    "london": (51.5074, -0.1278), "paris": (48.8566, 2.3522),
    "tokyo": (35.6762, 139.6503), "shanghai": (31.2304, 121.4737),
    "istanbul": (41.0082, 28.9784), "hong kong": (22.3193, 114.1694),
    "singapore": (1.3521, 103.8198), "sydney": (-33.8688, 151.2093),
    "berlin": (52.5200, 13.4050), "madrid": (40.4168, -3.7038),
    "rome": (41.9028, 12.4964), "mumbai": (19.0760, 72.8777),
    "dubai": (25.2048, 55.2708), "toronto": (43.6532, -79.3832),
    "tel aviv": (32.0853, 34.7818), "jerusalem": (31.7683, 35.2137),
    "beirut": (33.8938, 35.5018), "amman": (31.9454, 35.9284),
    "doha": (25.2854, 51.5310), "abu dhabi": (24.4539, 54.3773),
    "tehran": (35.6892, 51.3890), "baghdad": (33.3152, 44.3661),
    "mexico city": (19.4326, -99.1332), "seoul": (37.5665, 126.9780),
    "bangkok": (13.7563, 100.5018), "jeddah": (21.4858, 39.1925),
    "riyadh": (24.7136, 46.6753), "cairo": (30.0444, 31.2357),
    "lagos": (6.5244, 3.3792), "nairobi": (-1.2921, 36.8219),
    "cape town": (-33.9249, 18.4241), "johannesburg": (-26.2041, 28.0473),
    "lucknow": (26.8467, 80.9462), "delhi": (28.7041, 77.1025),
    "new delhi": (28.6139, 77.2090), "chennai": (13.0827, 80.2707),
    "karachi": (24.8607, 67.0011), "lahore": (31.5204, 74.3587),
    "islamabad": (33.6844, 73.0479), "dhaka": (23.8103, 90.4125),
    "kathmandu": (27.7172, 85.3240), "colombo": (6.9271, 79.8612),
    "hanoi": (21.0278, 105.8342), "taipei": (25.0330, 121.5654),
    "kolkata": (22.5726, 88.3639), "kuala lumpur": (3.1390, 101.6869),
    "jakarta": (-6.2088, 106.8456), "manila": (14.5995, 120.9842),
    "ho chi minh": (10.8231, 106.6297), "chongqing": (29.4316, 106.9123),
    "shenzhen": (22.5431, 114.0579), "guangzhou": (23.1291, 113.2644),
    "beijing": (39.9042, 116.4074), "wuhan": (30.5928, 114.3055),
    "chengdu": (30.5728, 104.0668), "hangzhou": (30.2741, 120.1551),
    "tianjin": (39.3434, 117.3616), "xi'an": (34.3416, 108.9398),
    "xian": (34.3416, 108.9398), "nanjing": (32.0603, 118.7969),
    "suzhou": (31.2990, 120.5853), "qingdao": (36.0671, 120.3826),
    "dalian": (38.9140, 121.6147), "busan": (35.1796, 129.0756),
    "osaka": (34.6937, 135.5023), "buenos aires": (-34.6037, -58.3816),
    "sao paulo": (-23.5505, -46.6333), "rio de janeiro": (-22.9068, -43.1729),
    "lima": (-12.0464, -77.0428), "bogota": (4.7110, -74.0721),
    "santiago": (-33.4489, -70.6693), "melbourne": (-37.8136, 144.9631),
    "auckland": (-36.8485, 174.7633), "milan": (45.4642, 9.1900),
    "amsterdam": (52.3676, 4.9041), "vienna": (48.2082, 16.3738),
    "warsaw": (52.2297, 21.0122), "moscow": (55.7558, 37.6173),
    # Added 2026-04-21 — parity with tools/weather.py after commit 33d89c3
    # missed the bot.py local dict. Confirmed in skip log: helsinki (42),
    # panama city (24). Rest mirror the preventive set from that commit.
    "wellington": (-41.2865, 174.7762), "panama city": (8.9824, -79.5199),
    "stockholm": (59.3293, 18.0686), "copenhagen": (55.6761, 12.5683),
    "oslo": (59.9139, 10.7522), "helsinki": (60.1699, 24.9384),
    "dublin": (53.3498, -6.2603), "lisbon": (38.7223, -9.1393),
    "barcelona": (41.3851, 2.1734), "athens": (37.9838, 23.7275),
    "prague": (50.0755, 14.4378), "budapest": (47.4979, 19.0402),
    "zurich": (47.3769, 8.5417), "brussels": (50.8503, 4.3517),
    "caracas": (10.4806, -66.9036), "havana": (23.1136, -82.3666),
    "accra": (5.6037, -0.1870), "addis ababa": (9.0054, 38.7636),
    "ankara": (39.9334, 32.8597), "bangalore": (12.9716, 77.5946),
    "hyderabad": (17.3850, 78.4867),
}

# ---------------------------------------------------------------------------
# HTTP sessions
# ---------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "(WeatherBot, trading@polybot)"})
REQUEST_TIMEOUT = 15

# Forecast caches (avoid duplicate API calls per cycle)
_forecast_cache: dict[str, tuple[float, dict]] = {}  # key: "lat,lon" -> (timestamp, forecast)
FORECAST_TTL = 1800  # 30 minutes


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def save_state(state: dict) -> None:
    try:
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning(f"save_state failed: {e}")


def load_state() -> dict:
    try:
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"traded_markets": [], "total_trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}


# ---------------------------------------------------------------------------
# Title parser — extracts city, threshold, comparison, metric
# ---------------------------------------------------------------------------
def parse_weather_market(title: str) -> dict | None:
    """Extract city/threshold/comparison/metric from a weather market title.

    Handles:
    - "Will the highest temperature in Atlanta be 23°C on April 14?"
    - "Will the highest temperature in Chicago be 39°F or below on April 14?"
    - "Will the highest temperature in NYC be between 80-81°F on April 15?"
    - "Will the highest temperature in Miami be 69°F or below on April 15?"
    """
    t = title.lower()

    # Must be a weather market
    if not any(kw in t for kw in ["temperature", "°c", "°f"]):
        return None

    # Extract metric (high vs low)
    if "minimum temperature" in t or "lowest temp" in t:
        metric = "low"
    else:
        metric = "high"

    # Extract city
    city = None
    for known_city in CITY_COORDS.keys():
        # Match "in {city}" to be precise
        if f" in {known_city}" in t or f" {known_city} " in t:
            city = known_city
            break
    if not city:
        # Try fuzzy match — any city name appearing in title
        for known_city in CITY_COORDS.keys():
            if known_city in t:
                city = known_city
                break
    if not city:
        return None

    # Extract threshold + unit + comparison
    # Handle "between X-Y°F" first (range markets)
    range_match = re.search(r"between\s+(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s*°?\s*([cf])", t)
    if range_match:
        low_val = float(range_match.group(1))
        high_val = float(range_match.group(2))
        unit = range_match.group(3)
        if unit == "f":
            low_c = (low_val - 32) * 5 / 9
            high_c = (high_val - 32) * 5 / 9
        else:
            low_c = low_val
            high_c = high_val
        return {
            "city": city,
            "metric": metric,
            "comparison": "range",
            "threshold_low_c": low_c,
            "threshold_high_c": high_c,
            "original_title": title,
        }

    # Handle "X°F or below", "X°F or higher", "X°C"
    match = re.search(r"(\d+(?:\.\d+)?)\s*°?\s*([cf])\s*(or\s+(below|lower|less|higher|above|more))?", t)
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2)
    qualifier = match.group(4) or ""

    # Convert to Celsius
    if unit == "f":
        threshold_c = (value - 32) * 5 / 9
    else:
        threshold_c = value

    # Determine comparison
    if qualifier in ("below", "lower", "less"):
        comparison = "below_or_equal"
    elif qualifier in ("higher", "above", "more"):
        comparison = "above_or_equal"
    else:
        comparison = "equal"  # "be X°C" without qualifier = exact match

    return {
        "city": city,
        "metric": metric,
        "comparison": comparison,
        "threshold_c": threshold_c,
        "original_title": title,
    }


def parse_target_date(title: str) -> str | None:
    """Extract target date from title like 'on April 14?' or 'April 13, 2026'."""
    # Try "Month Day" patterns
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    }
    t = title.lower()
    for name, num in months.items():
        m = re.search(rf"{name}\s+(\d{{1,2}})", t)
        if m:
            day = int(m.group(1))
            year = datetime.now(timezone.utc).year
            # If month already passed this year, assume next year
            now = datetime.now(timezone.utc)
            target = datetime(year, num, day, tzinfo=timezone.utc)
            if target < now - _timedelta_days(30):
                target = datetime(year + 1, num, day, tzinfo=timezone.utc)
            return target.strftime("%Y-%m-%d")
    return None


def _timedelta_days(days: int):
    from datetime import timedelta
    return timedelta(days=days)


# ---------------------------------------------------------------------------
# Forecast fetching
# ---------------------------------------------------------------------------
def get_forecast(city: str) -> dict | None:
    """Fetch forecast for a city. Returns dict with daily forecasts."""
    coords = CITY_COORDS.get(city.lower())
    if not coords:
        return None
    lat, lon = coords
    cache_key = f"{lat:.2f},{lon:.2f}"

    # Check cache
    if cache_key in _forecast_cache:
        ts, data = _forecast_cache[cache_key]
        if time.time() - ts < FORECAST_TTL:
            return data

    # Fetch from Open-Meteo (works globally, no auth)
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": "auto",
            "forecast_days": 10,
        }
        resp = SESSION.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})

        forecasts = {}
        dates = daily.get("time", [])
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])
        for i, date in enumerate(dates):
            forecasts[date] = {
                "high_c": highs[i] if i < len(highs) else None,
                "low_c": lows[i] if i < len(lows) else None,
            }

        result = {"city": city, "forecasts": forecasts}
        _forecast_cache[cache_key] = (time.time(), result)
        return result
    except Exception as e:
        logger.warning(f"Forecast fetch failed for {city}: {e}")
        return None


# ---------------------------------------------------------------------------
# Probability calculation
# ---------------------------------------------------------------------------
def estimate_probability(forecast_temp: float, threshold: float, comparison: str, days_ahead: int = 1) -> float:
    """Estimate P(temp [comparison] threshold) using normal distribution.

    Widened 2026-04-20 after observing 41% drift on N=9 trades (model
    predicted ~0.94 avg, actual WR was 0.44). Old formula (1.5°C base,
    1.8°C at 2d) underestimated forecast uncertainty. Real MAE for 2-day
    forecasts is ~2-3°C. Post-widening tails:
    - 1 day: std 2.5°C
    - 3 days: std 3.0°C
    - 7 days: std 5.0°C
    """
    std = max(2.5, 1.5 + days_ahead * 0.5)  # linearly growing uncertainty

    if std == 0:
        return 1.0 if forecast_temp >= threshold else 0.0

    z = (threshold - forecast_temp) / std
    prob_below = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    prob_above = 1 - prob_below

    if comparison == "above_or_equal":
        return round(prob_above, 3)
    elif comparison == "below_or_equal":
        return round(prob_below, 3)
    elif comparison == "equal":
        # P(temp in [X-0.5, X+0.5])
        z_low = (threshold - 0.5 - forecast_temp) / std
        z_high = (threshold + 0.5 - forecast_temp) / std
        p_low = 0.5 * (1 + math.erf(z_low / math.sqrt(2)))
        p_high = 0.5 * (1 + math.erf(z_high / math.sqrt(2)))
        return round(p_high - p_low, 3)
    else:
        return round(prob_above, 3)


def estimate_range_probability(
    forecast_temp: float, threshold_low: float, threshold_high: float, days_ahead: int = 1
) -> float:
    """Estimate P(threshold_low <= temp <= threshold_high).

    Uses the same widened std_dev as estimate_probability (see note there).
    """
    std = max(2.5, 1.5 + days_ahead * 0.5)
    z_low = (threshold_low - forecast_temp) / std
    z_high = (threshold_high - forecast_temp) / std
    p_low = 0.5 * (1 + math.erf(z_low / math.sqrt(2)))
    p_high = 0.5 * (1 + math.erf(z_high / math.sqrt(2)))
    return round(p_high - p_low, 3)


# ---------------------------------------------------------------------------
# Simmer API
# ---------------------------------------------------------------------------
def fetch_markets(api_key: str, limit: int = 100) -> list[dict]:
    """Fetch weather markets from Simmer.

    The default /markets endpoint returns mostly crypto noise. We use the q=temperature
    query to filter server-side for weather markets only.
    """
    try:
        resp = SESSION.get(
            f"{SIMMER_API}/markets",
            params={"limit": limit, "q": "temperature"},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("markets", [])
    except Exception as e:
        logger.error(f"fetch_markets failed: {e}")
        return []


def fetch_balance(api_key: str, venue: str = "sim", wallet_address: str = "") -> float:
    """Returns balance for the configured venue.

    For polymarket: reads USDC.e directly from Polygon blockchain (real balance).
    For sim: returns $SIM balance from Simmer API.
    """
    if venue == "polymarket" and wallet_address:
        # Read real USDC.e balance from Polygon blockchain
        try:
            from web3 import Web3
            # USDC.e bridged (Polymarket uses this), NOT native USDC (0x3c499c...)
            USDC = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
            abi = [{"constant": True, "inputs": [{"name": "_owner", "type": "address"}],
                    "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}],
                    "type": "function"}]
            for rpc in ["https://polygon-bor-rpc.publicnode.com",
                        "https://rpc.ankr.com/polygon",
                        "https://polygon.llamarpc.com"]:
                try:
                    w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
                    contract = w3.eth.contract(address=Web3.to_checksum_address(USDC), abi=abi)
                    raw = contract.functions.balanceOf(Web3.to_checksum_address(wallet_address)).call()
                    return raw / 1e6  # USDC has 6 decimals
                except Exception as _e:
                    logger.debug(f"RPC {rpc} failed: {_e}")
                    continue
            logger.error("All Polygon RPCs failed")
            return -1.0
        except Exception as e:
            logger.error(f"fetch_balance (polymarket) failed: {e}")
            return -1.0

    # Sim venue
    try:
        resp = SESSION.get(
            f"{SIMMER_API}/agents/me",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return float(resp.json().get("balance", 0))
    except Exception as e:
        logger.error(f"fetch_balance failed: {e}")
        return -1.0


def place_order(
    api_key: str,
    market_id: str,
    side: str,
    amount: float,
    reason: str,
    venue: str = "sim",
    dry_run: bool = False,
    order_type: str = "GTC",
    action: str = "buy",
    shares: float | None = None,
    source: str = "sdk:weather-bot",
    price: float | None = None,
) -> dict:
    """Place a trade on Simmer.

    action="buy" (default): open a position, size = amount USDC.
    action="sell": close/reduce a position, size = shares (preferred) or amount USDC.

    price: Polymarket requires tick-aligned limit prices (0.01 increment). If
    caller supplies a price, we round to 2 decimals before sending. Without
    this Simmer's server-side default ended up with fractional values like
    0.139 which the CLOB rejected with "breaks minimum tick size rule: 0.01".

    order_type defaults to "GTC" (Good-Till-Cancelled). Previously the Simmer
    default was FAK (Fill-and-Kill), which refuses any partial match and was
    killing ~75% of weather-bot orders in thin markets — the order would be
    rejected with "not filled at price X, no liquidity" even when an edge
    existed. GTC sits in the book until someone fills it, so valid edges get
    captured as long as counterparty appears.
    """
    if dry_run:
        logger.info(f"[DRY RUN] {venue} {action} {side} {shares or amount} on {market_id}: {reason}")
        return {"executed": False, "dry_run": True}
    payload = {
        "market_id": market_id,
        "side": side.lower(),
        "action": action,
        "venue": venue,
        "reasoning": reason[:200],
        "source": source,
        "order_type": order_type,
    }
    if action == "sell" and shares is not None:
        payload["shares"] = shares
    else:
        payload["amount"] = amount
    if price is not None:
        payload["price"] = round(float(price), 2)
    try:
        resp = SESSION.post(
            f"{SIMMER_API}/trade",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code >= 400:
            logger.error(f"place_order HTTP {resp.status_code}: {resp.text[:300]}")
            return {"executed": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        body = resp.json()
        # Simmer returns HTTP 200 with {success: false, error: ...} for logical failures
        if not body.get("success", False):
            err = body.get("error", "unknown") or "no success field"
            logger.error(f"place_order rejected: {err}")
            return {"executed": False, "error": err, "response": body}
        return {"executed": True, "response": body}
    except Exception as e:
        logger.error(f"place_order failed: {e}")
        return {"executed": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Trading logic
# ---------------------------------------------------------------------------
def evaluate_market(
    market: dict,
    min_edge: float,
    verbose: bool = False,
    venue_hint: str = "sim",
    skip_counts: dict | None = None,
) -> dict | None:
    """Evaluate a single weather market. Returns trade decision or None.

    skip_counts: optional dict mutated in-place to track why markets are skipped.
    Keys used: parse_failed, equal_paused, no_date, no_forecast, no_temp,
    edge_too_small, entry_too_high, entry_too_low, market_extreme, no_yes_on_exact,
    no_market_price, midpoint_no_liquidity.
    """
    def _tick(k: str) -> None:
        if skip_counts is not None and k in skip_counts:
            skip_counts[k] += 1

    title = market.get("question", "")
    market_id = market.get("id", "")

    # Simmer's current_probability is the midpoint of best-bid/best-ask. For
    # thinly-traded markets the spread can be 99¢/1¢ with midpoint 0.5 — an order
    # at "midpoint" fills at the ask (e.g., buy NO at 99¢ when midpoint is 0.50).
    # Trading at a missing or exact-0.5 price is unsafe; skip those markets.
    _prob = market.get("current_probability")
    if _prob is None:
        _prob = market.get("current_price")
    if _prob is None:
        if verbose:
            logger.info(f"  SKIP (no market price): {title[:70]}")
        _tick("no_market_price")
        return None
    current_price = float(_prob)
    # Widened 2026-04-22 from 1e-9 to ±0.05 after diagnostic confirmed bug:
    # 10/14 trades in 48h had bot-calc entry 0.485-0.50 but executed at real
    # ask of 96-99¢ (Buenos Aires NO @ 97¢, Singapore NO @ 99¢, Dallas YES
    # @ 99¢, etc). Simmer's current_probability is bid-ask midpoint; thin
    # markets with bid 0.99/ask 0.01 give midpoint exactly 0.50 and similar
    # variations (0.485, 0.49). Any midpoint within 5¢ of 0.5 indicates
    # unreliable price — refuse to trade.
    if abs(current_price - 0.5) < 0.05:
        if verbose:
            logger.info(f"  SKIP (midpoint {current_price:.3f} ~0.5, thin liquidity): {title[:70]}")
        _tick("midpoint_no_liquidity")
        return None

    parsed = parse_weather_market(title)
    if not parsed:
        if verbose:
            logger.info(f"  SKIP (parse failed): {title[:70]}")
        _tick("parse_failed")
        return None

    # PAUSED 2026-04-20: equal markets had Brier 0.40 / WR 57% on N=7 — actively
    # losing money. Disable until the model is recalibrated and we have evidence
    # the new wider std_dev fixes the systematic overconfidence.
    if parsed["comparison"] == "equal":
        if verbose:
            logger.info(f"  SKIP (equal markets paused for recalibration): {title[:70]}")
        _tick("equal_paused")
        return None

    target_date = parse_target_date(title)
    if not target_date:
        if verbose:
            logger.info(f"  SKIP (no date): {title[:70]}")
        _tick("no_date")
        return None

    forecast_data = get_forecast(parsed["city"])
    if not forecast_data:
        if verbose:
            logger.info(f"  SKIP (no forecast): {parsed['city']} | {title[:60]}")
        _tick("no_forecast")
        return None

    day_forecast = forecast_data["forecasts"].get(target_date)
    if not day_forecast:
        if verbose:
            logger.info(f"  SKIP (no forecast for {target_date}): {parsed['city']} | {title[:50]}")
        _tick("no_forecast")
        return None

    # Pick the right temp based on metric
    temp_c = day_forecast.get("high_c") if parsed["metric"] == "high" else day_forecast.get("low_c")
    if temp_c is None:
        if verbose:
            logger.info(f"  SKIP (no temp): {parsed['city']} {target_date}")
        _tick("no_temp")
        return None

    # Calculate days ahead for uncertainty
    try:
        target_dt = datetime.fromisoformat(target_date).replace(tzinfo=timezone.utc)
        days_ahead = max(1, (target_dt - datetime.now(timezone.utc)).days)
    except Exception:
        days_ahead = 3

    # Calculate probability
    if parsed["comparison"] == "range":
        p_yes = estimate_range_probability(
            temp_c, parsed["threshold_low_c"], parsed["threshold_high_c"], days_ahead
        )
    else:
        p_yes = estimate_probability(
            temp_c, parsed["threshold_c"], parsed["comparison"], days_ahead
        )

    edge = p_yes - current_price
    abs_edge = abs(edge)

    if verbose:
        logger.info(
            f"  {parsed['city']} {target_date} {parsed['metric']}: "
            f"forecast={temp_c:.1f}C threshold={parsed.get('threshold_c', '?')} "
            f"P_real={p_yes:.2f} market={current_price:.2f} edge={edge:+.2f}"
        )

    # Higher min_edge for exact-temp markets — they're harder to predict
    # and the distribution is narrow, so we need more edge to be confident
    effective_min_edge = min_edge
    if parsed["comparison"] == "equal":
        effective_min_edge = max(min_edge, 0.22)  # At least 22% edge for exact temps
    elif parsed["comparison"] == "range":
        effective_min_edge = max(min_edge, 0.25)  # At least 25% edge for range markets (narrow 1°C window)

    if abs_edge < effective_min_edge:
        _tick("edge_too_small")
        return None

    side = "yes" if edge > 0 else "no"

    # RULE: Never buy YES on exact-temp markets — probability is inherently low
    # (P(exactly X°C) ~ 10-20%) so cheap YES prices are correctly priced, not undervalued.
    # Only trade the NO side of exact-temp markets.
    if parsed["comparison"] == "equal" and side == "yes":
        if verbose:
            logger.info(f"  SKIP (no YES on exact-temp): {title[:60]} edge={edge:+.2f}")
        _tick("no_yes_on_exact")
        return None

    # Entry price is the side we're buying
    entry_price = current_price if side == "yes" else (1 - current_price)

    # Entry price gate: reject trades with terrible risk/reward
    # Polymarket 0.55: win pays 82% per trade, break-even WR = 55% (vs current 72%)
    # SIM 0.75: more relaxed since SIM is virtual
    max_entry = 0.75 if venue_hint != "polymarket" else 0.55
    if entry_price > max_entry:
        if verbose:
            logger.info(f"  SKIP (entry {entry_price:.2f} > {max_entry}): {title[:60]}")
        _tick("entry_too_high")
        return None

    # MIN entry price: reject trades where the side we're buying is too cheap (<5¢)
    # This usually means the market is already resolved/about to resolve, or the
    # other side is at >0.95 which means YES/NO is locked in.
    if entry_price < 0.05:
        if verbose:
            logger.info(f"  SKIP (entry {entry_price:.2f} < 0.05, market likely resolving): {title[:60]}")
        _tick("entry_too_low")
        return None

    # MARKET PRICE EXTREMES: skip markets at 0.99+ or 0.01- on the OPPOSITE side
    # If market_price (YES) is 1.00, market thinks YES is certain. Even if our
    # forecast disagrees, this is usually a sign the market is already settling.
    if current_price >= 0.99 or current_price <= 0.01:
        if verbose:
            logger.info(f"  SKIP (market extreme {current_price:.2f}, likely resolving): {title[:60]}")
        _tick("market_extreme")
        return None

    return {
        "market_id": market_id,
        "title": title,
        "city": parsed["city"],
        "metric": parsed["metric"],
        "forecast_temp_c": temp_c,
        "threshold_c": parsed.get("threshold_c") or f"{parsed.get('threshold_low_c')}-{parsed.get('threshold_high_c')}",
        "comparison": parsed["comparison"],
        "p_yes_real": p_yes,
        "market_price": current_price,
        "edge": round(edge, 3),
        "side": side,
        "entry_price": entry_price,
        "days_ahead": days_ahead,
    }


def compute_bet_size(
    edge: float,
    balance: float,
    config: dict,
    entry_price: float = 0.5,
    venue: str = "sim",
    p_real: float = 0.5,
) -> float:
    """Size bet using fractional Kelly criterion for optimal long-term growth.

    Kelly formula: f* = (p * b - q) / b
    where p = probability of winning, q = 1-p, b = payout odds (net profit / stake)

    We use HALF Kelly (f*/2) to reduce variance — full Kelly is too aggressive
    and assumes perfect probability estimates, which we don't have.

    Polymarket constraints:
    - Minimum $1 per order
    - Minimum 5 shares per order (cost = 5 * entry_price)
    - 10% fee (fee_rate_bps=1000)
    """
    max_bet = float(config.get("max_bet", config.get("max_bet_usdc", 50)))
    # Raised from 0.02 to 0.05 on 2026-04-21: with small real-money balance (~$180),
    # 2% cap meant base=$3.60 which combined with Kelly normalization often produced
    # bets that failed Polymarket's 5-share minimum after slippage. At 5%, small
    # balances have enough Kelly headroom while large balances still get capped by
    # max_bet_usdc (the explicit user-set dollar cap).
    max_pct = float(config.get("max_pct_balance", 0.05))

    abs_edge = abs(edge)

    # Kelly criterion sizing
    # p = our estimated probability of winning this bet
    # b = payout odds = (1 - entry_price) / entry_price for YES,
    #     simplified: what we gain per dollar risked
    if entry_price <= 0 or entry_price >= 1:
        return 0

    p_win = p_real if edge > 0 else (1 - p_real)  # our prob that our side wins
    payout_odds = (1 - entry_price) / entry_price  # net profit per $1 risked

    if payout_odds <= 0:
        return 0

    # Kelly fraction: f* = (p * b - q) / b
    q_lose = 1 - p_win
    kelly_fraction = (p_win * payout_odds - q_lose) / payout_odds

    # If Kelly says don't bet (negative edge), skip
    if kelly_fraction <= 0:
        return 0

    # Use HALF Kelly to reduce variance (standard practice)
    half_kelly = kelly_fraction / 2

    # Apply bankroll constraints
    base = min(max_bet, balance * max_pct)
    bet = base * min(1.0, half_kelly / 0.05)  # normalize: 5% half-kelly = full base bet

    if venue == "polymarket":
        # Simmer charges 10% fee BEFORE calculating shares, and price slips on impact.
        # Target 6 shares (1 buffer above 5 minimum) with fee compensation.
        # Formula: bet * 0.9 / entry_price >= 6  →  bet >= 6 * entry_price / 0.9
        min_bet_for_shares = max(1.0, (6.0 * entry_price) / 0.9)
        if bet < min_bet_for_shares:
            if min_bet_for_shares > max_bet:
                return 0
            bet = min_bet_for_shares

    return round(bet, 2)


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
def send_telegram(msg: str, config: dict) -> None:
    token = config.get("telegram_bot_token", "")
    chat_id = config.get("telegram_chat_id", "")
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown", "disable_web_page_preview": True},
            timeout=10,
        )
    except Exception as e:
        logger.warning(f"Telegram failed: {e}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
class WeatherBot:
    def __init__(self, config: dict):
        self.config = config
        self.running = True
        self.state = load_state()
        self.api_key = config["simmer_api_key"]
        self.wallet_address = config.get("wallet_address", "")
        self.venue = config.get("venue", "sim")
        self.min_edge = float(config.get("min_edge", 0.20))
        self.max_bet = float(config.get("max_bet", config.get("max_bet_usdc", 50)))
        self.dry_run = config.get("dry_run", False)
        self.cycle_interval = int(config.get("cycle_interval_seconds", 180))
        self.currency = "USDC" if self.venue == "polymarket" else "$SIM"
        # One-shot Telegram alert when an open position's unrealized P&L crosses
        # this threshold. Config override: loss_alert_pct (float, negative).
        # Alerted positions tracked in state to avoid re-alerts on the same market.
        self.loss_alert_pct = float(config.get("loss_alert_pct", -0.10))
        self.alerted_positions: set[str] = set(self.state.get("alerted_positions", []))
        # Bot-side stop-loss. Simmer's risk-monitor is inconsistent for SDK
        # trades (observed positions at -25%/-29% that never auto-closed), so
        # the bot runs its own safety net. When pnl / cost_basis <= stop_loss_pct
        # and the market_id has not been sold yet, the bot submits a GTC sell
        # for the full holding and remembers the market_id so it never retries.
        # Disable with "stop_loss_enabled": false.
        self.stop_loss_enabled = bool(config.get("stop_loss_enabled", True))
        self.stop_loss_pct = float(config.get("stop_loss_pct", -0.10))
        self.stop_loss_fired: set[str] = set(self.state.get("stop_loss_fired", []))
        # Skip counters — reset each summary window (~6h). Aggregate visibility
        # into why markets are filtered without needing verbose logging.
        self._skip_counts: dict[str, int] = {
            "equal_paused": 0,
            "parse_failed": 0,
            "no_date": 0,
            "no_forecast": 0,
            "no_temp": 0,
            "edge_too_small": 0,
            "entry_too_high": 0,
            "entry_too_low": 0,
            "market_extreme": 0,
            "no_yes_on_exact": 0,
            "no_market_price": 0,
            "midpoint_no_liquidity": 0,
            "correlation_limit": 0,
            "bet_too_small": 0,
            "shares_below_min": 0,
            "already_traded": 0,
            "already_failed": 0,
        }
        # Snapshot of total_trades at window start — used to compute trades
        # executed within the current window for the evaluated-count math.
        self._trades_at_window_start = self.state.get("total_trades", 0)

    def cycle(self) -> None:
        balance = fetch_balance(self.api_key, venue=self.venue, wallet_address=self.wallet_address)
        # Need at least max_bet to trade — no point evaluating markets if can't afford
        min_balance = self.max_bet * 1.1
        if balance < min_balance:
            # Log only every 30 cycles to avoid spam
            if getattr(self, "_low_balance_count", 0) % 30 == 0:
                logger.warning(f"Balance too low: {balance:.2f} {self.currency} (need {min_balance:.2f}). Skipping cycles.")
            self._low_balance_count = getattr(self, "_low_balance_count", 0) + 1
            return
        self._low_balance_count = 0

        markets = fetch_markets(self.api_key, limit=100)
        if not markets:
            logger.info("No markets fetched")
            return

        # Increment cycle counter
        self._cycle_num = getattr(self, "_cycle_num", 0) + 1

        # Clean stale traded_markets: remove IDs not in current market list
        # This prevents the state file from growing forever
        if self._cycle_num % 50 == 0:  # every ~50 cycles
            active_ids = {m.get("id") for m in markets if m.get("id")}
            old_count = len(self.state.get("traded_markets", []))
            self.state["traded_markets"] = [
                mid for mid in self.state.get("traded_markets", [])
                if mid in active_ids
            ]
            cleaned = old_count - len(self.state["traded_markets"])
            if cleaned > 0:
                logger.info(f"Cleaned {cleaned} stale entries from traded_markets")
                save_state(self.state)

        # Filter to weather markets
        weather_markets = [m for m in markets if any(kw in (m.get("question", "") or "").lower() for kw in ["temperature", "°c", "°f"])]

        # Verbose debug every 10 cycles
        verbose = (self._cycle_num % 10 == 1)
        logger.info(f"Cycle {self._cycle_num}: {len(markets)} markets, {len(weather_markets)} weather {'[VERBOSE]' if verbose else ''}")

        # Count existing positions per city+date for correlation limits
        _city_date_counts: dict[str, int] = {}
        for mid in self.state.get("traded_markets", []):
            # We track by market title but only have IDs in state;
            # count from the current weather_markets that are already traded
            pass
        for m in weather_markets:
            if m.get("id") in self.state.get("traded_markets", []):
                _title = (m.get("question", "") or "").lower()
                _parsed = parse_weather_market(_title)
                _tdate = parse_target_date(_title)
                if _parsed and _tdate:
                    _ck = f"{_parsed['city']}|{_tdate}"
                    _city_date_counts[_ck] = _city_date_counts.get(_ck, 0) + 1

        # Evaluate each
        opportunities = []
        max_correlated = 3
        failed_set = set(self.state.get("failed_markets", []))
        for m in weather_markets:
            if m.get("id") in self.state.get("traded_markets", []):
                self._skip_counts["already_traded"] += 1
                continue  # already traded this market
            if m.get("id") in failed_set:
                self._skip_counts["already_failed"] += 1
                continue  # already failed, don't retry
            decision = evaluate_market(
                m, self.min_edge, verbose=verbose, venue_hint=self.venue,
                skip_counts=self._skip_counts,
            )
            if decision:
                # Check correlation limit
                _ck = f"{decision['city']}|{parse_target_date(decision['title']) or ''}"
                if _city_date_counts.get(_ck, 0) >= max_correlated:
                    if verbose:
                        logger.info(f"  SKIP (correlation limit {max_correlated}): {decision['city']} {_ck}")
                    self._skip_counts["correlation_limit"] += 1
                    continue
                opportunities.append(decision)

        # Sort by edge descending
        opportunities.sort(key=lambda x: abs(x["edge"]), reverse=True)

        if not opportunities:
            logger.info("No opportunities with sufficient edge")
            return

        logger.info(f"Found {len(opportunities)} opportunities")

        # Execute top N per cycle (default 1 — be selective, pick only the best)
        max_per_cycle = int(self.config.get("max_trades_per_cycle", 1))
        min_bet = 1.0  # Simmer/Polymarket minimum $1 per order
        for opp in opportunities[:max_per_cycle]:
            bet = compute_bet_size(
                opp["edge"], balance, self.config,
                entry_price=opp.get("entry_price", 0.5),
                venue=self.venue,
                p_real=opp.get("p_yes_real", 0.5),
            )
            if bet < min_bet:
                logger.info(f"SKIP (bet too small): {opp['title'][:50]} bet={bet}")
                self._skip_counts["bet_too_small"] += 1
                continue

            # Polymarket requires ≥5 shares per order. shares = bet / entry_price.
            # If Kelly suggests a bet that produces fewer shares, skip rather than
            # let place_order fail downstream (which used to spam Telegram).
            _entry = opp.get("entry_price", 0.5)
            if _entry > 0 and bet / _entry < 5.0:
                logger.info(
                    f"SKIP (would be {bet / _entry:.2f} shares < 5): {opp['title'][:50]} bet={bet}"
                )
                self._skip_counts["shares_below_min"] += 1
                continue

            reason = (
                f"{opp['city']} {opp['metric']} forecast {opp['forecast_temp_c']:.1f}C "
                f"vs threshold {opp['threshold_c']} {opp['comparison']} "
                f"(days_ahead={opp['days_ahead']}). "
                f"P_real={opp['p_yes_real']}, market={opp['market_price']:.2f}, edge={opp['edge']:+.2f}"
            )
            # Aggressive-fill price: post a limit above the best ask so the GTC
            # order sweeps book depth instead of parking at midpoint. Capped at
            # max_entry so slippage cannot push us into the 99¢-trap zone.
            # Observed problem: Seattle 2026-04-21 placed $2.84 at 10.5¢ midpoint,
            # FAK filled only 1.71 shares ($0.19) because book had no depth at
            # exactly 10.5¢. Posting at 10.5¢ + slippage = 15.5¢ would have
            # captured the whole intended stake.
            _slippage = float(self.config.get("buy_slippage", 0.05))
            _max_entry = 0.55 if self.venue == "polymarket" else 0.75
            buy_price = max(0.01, min(_entry + _slippage, _max_entry, 0.99))
            buy_price = round(buy_price, 2)
            logger.info(
                f"TRADE [{self.venue}]: {opp['side'].upper()} {bet}{self.currency} "
                f"on '{opp['title'][:60]}' | {reason} | limit {buy_price:.2f}"
            )

            result = place_order(
                self.api_key, opp["market_id"], opp["side"], bet, reason,
                venue=self.venue, dry_run=self.dry_run,
                price=buy_price,
            )

            if result.get("executed"):
                self.state["traded_markets"].append(opp["market_id"])
                self.state["total_trades"] = self.state.get("total_trades", 0) + 1
                # Update correlation counter
                _ck = f"{opp['city']}|{parse_target_date(opp['title']) or ''}"
                _city_date_counts[_ck] = _city_date_counts.get(_ck, 0) + 1
                save_state(self.state)

                # Log trade with predicted probability for calibration analysis
                try:
                    _p_side = opp["p_yes_real"] if opp["side"] == "yes" else (1 - opp["p_yes_real"])
                    _cal_entry = {
                        "timestamp": time.time(),
                        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "market_id": opp["market_id"],
                        "title": opp["title"],
                        "city": opp["city"],
                        "side": opp["side"],
                        "p_predicted": round(_p_side, 4),
                        "market_price_entry": round(opp["entry_price"], 4),
                        "p_yes_real": round(opp["p_yes_real"], 4),
                        "edge": round(opp["edge"], 4),
                        "amount": bet,
                        "venue": self.venue,
                        "comparison": opp["comparison"],
                        "days_ahead": opp["days_ahead"],
                        "source": "weather-bot",
                    }
                    _cal_path = os.path.join(HERE, "..", "memory", "calibration_log.jsonl")
                    os.makedirs(os.path.dirname(_cal_path), exist_ok=True)
                    with open(_cal_path, "a") as _cf:
                        _cf.write(json.dumps(_cal_entry) + "\n")
                except Exception as _ce:
                    logger.warning(f"Calibration log failed: {_ce}")
                send_telegram(
                    f"*Weather Trade [{self.venue}]*\n{opp['side'].upper()} {bet}{self.currency}\n{opp['title'][:80]}\n"
                    f"Edge: {opp['edge']:+.2f} | P_real: {opp['p_yes_real']} | Price: {opp['market_price']:.2f}",
                    self.config,
                )
                balance -= bet
            else:
                err = result.get("error", "unknown")
                logger.error(f"Trade failed: {err}")

                err_lower = err.lower()
                is_network = any(kw in err_lower for kw in [
                    "timed out", "timeout", "connection", "httpsconnectionpool",
                    "read timed out", "max retries", "connection reset",
                    "503", "502", "504", "gateway",
                ])
                is_balance = "insufficient balance" in err_lower or "not enough" in err_lower

                # Only add to "do not retry" if it's a permanent error (not network/balance)
                if not is_network and not is_balance:
                    self.state.setdefault("failed_markets", []).append(opp["market_id"])
                    save_state(self.state)

                # Only notify on REAL errors. These are non-actionable noise:
                # - too small / below minimum: Kelly bet under 5-share floor (not a problem)
                # - no asks / order book / insufficient shares: thin liquidity, normal
                # - too high / correlation / already / spread / no stacking: pre-existing filters
                is_silent = is_network or is_balance or any(kw in err_lower for kw in [
                    "too high", "correlation", "already", "spread too high", "no stacking",
                    "too small", "below minimum", "no asks", "order book", "rounding",
                    "insufficient shares",
                ])
                if self.venue == "polymarket" and not is_silent:
                    send_telegram(
                        f"⚠️ *Trade FAILED*\n{opp['title'][:80]}\n{err[:150]}",
                        self.config,
                    )

        # Check for loss-threshold crossings on open positions (one-shot alerts).
        self._check_loss_alerts()

        # Bot-side stop-loss: sell anything that crossed the pain threshold.
        self._check_stop_loss()

    def _check_loss_alerts(self) -> None:
        """Send a one-time Telegram alert when any open position crosses
        ``loss_alert_pct`` unrealized. Alerted position titles persist in state
        so the same market never triggers twice. Real-money venue only.

        Disabled by default: data-api.polymarket.com/positions returns only
        resolved positions for Simmer-mediated wallets, and Simmer's own
        /positions endpoint returns [] for this account. Need to identify the
        correct Simmer endpoint (likely /agents/me/positions or similar).
        Opt in with config "loss_alerts_enabled": true once fixed.
        """
        if not self.config.get("loss_alerts_enabled", False):
            return
        if self.venue != "polymarket" or not self.wallet_address:
            return
        try:
            positions = get_positions(
                self.wallet_address, venue=self.venue, simmer_api_key=self.api_key
            )
        except Exception as e:
            logger.warning(f"loss_alert fetch_positions failed: {e}")
            return

        new_alerts: list[tuple[dict, float]] = []
        for pos in positions:
            avg = float(pos.get("avg_price", 0) or 0)
            curr = float(pos.get("current_price", 0) or 0)
            title = str(pos.get("title", "")).strip()
            size = float(pos.get("size", 0) or 0)
            # Polymarket's data API returns market_id="" for most positions, so we
            # key by title. Skip: resolved positions (curr at 0 or 1), dust (stake
            # under $0.50), and missing metadata.
            if avg <= 0 or curr <= 0 or curr >= 1 or not title:
                continue
            if avg * size < 0.50:
                continue
            key = title[:100]
            if key in self.alerted_positions:
                continue
            pnl_pct = (curr - avg) / avg
            if pnl_pct <= self.loss_alert_pct:
                new_alerts.append((pos, pnl_pct))
                self.alerted_positions.add(key)

        for pos, pct in new_alerts:
            title = str(pos.get("title", ""))[:80]
            side = str(pos.get("side", "")).upper()
            msg = (
                f"⚠️ *Position below {int(self.loss_alert_pct * 100)}%*\n"
                f"{title}\n"
                f"Side {side} | Entry {float(pos.get('avg_price', 0)):.2f} "
                f"→ Current {float(pos.get('current_price', 0)):.2f}\n"
                f"Unrealized: ${float(pos.get('unrealized_pnl', 0)):+.2f} ({pct * 100:+.1f}%)"
            )
            send_telegram(msg, self.config)
            logger.info(f"Loss alert: {title[:50]} at {pct * 100:+.1f}%")

        if new_alerts:
            self.state["alerted_positions"] = list(self.alerted_positions)
            save_state(self.state)

    def _check_stop_loss(self) -> None:
        """Bot-side stop-loss. Fetches open positions from Simmer's positions
        endpoint (which unlike data-api.polymarket.com returns live SDK-placed
        positions) and sells any with unrealized pnl / cost_basis below
        stop_loss_pct. One-shot per market: once we attempt to close a market
        the ID goes into stop_loss_fired and is never re-tried."""
        if not self.stop_loss_enabled:
            return
        if self.venue != "polymarket":
            return
        try:
            resp = SESSION.get(
                f"{SIMMER_API}/positions",
                params={"venue": "polymarket", "status": "active"},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            positions = resp.json().get("positions", [])
        except Exception as e:
            logger.warning(f"stop_loss fetch_positions failed: {e}")
            return

        for pos in positions:
            mid = str(pos.get("market_id", ""))
            if not mid or mid in self.stop_loss_fired:
                continue
            cost = float(pos.get("cost_basis") or 0)
            pnl = float(pos.get("pnl") or 0)
            if cost <= 0:
                continue
            pct = pnl / cost
            if pct > self.stop_loss_pct:
                continue

            shares_yes = float(pos.get("shares_yes") or 0)
            shares_no = float(pos.get("shares_no") or 0)
            if shares_yes > shares_no:
                side, shares = "yes", shares_yes
            elif shares_no > 0:
                side, shares = "no", shares_no
            else:
                continue
            if shares <= 0:
                continue

            title = str(pos.get("question") or "")[:80]
            # Current per-share market value (side-correct, always present).
            # Falls back to avg_cost if size is zero to avoid division-by-zero.
            current_value = float(pos.get("current_value") or 0)
            raw_price = (current_value / shares) if shares > 0 else float(pos.get("avg_cost") or 0.5)
            # Aggressive-sell slippage: post limit below best bid so the GTC
            # order sweeps through bids instead of parking at the mark and
            # waiting. Matters more for sells than buys: the whole point is
            # to exit fast when a position crosses the pain threshold.
            _sell_slippage = float(self.config.get("sell_slippage", 0.05))
            sell_price = max(0.01, round(raw_price - _sell_slippage, 2))
            reason = f"stop_loss: pnl {pct * 100:+.1f}% <= {self.stop_loss_pct * 100:.0f}%"
            logger.info(
                f"STOP-LOSS [{side.upper()} {shares:.2f} shares @ {sell_price:.3f}]: "
                f"{title[:50]} | {reason}"
            )
            result = place_order(
                self.api_key,
                mid,
                side,
                amount=0.0,
                reason=reason,
                venue=self.venue,
                dry_run=self.dry_run,
                action="sell",
                shares=round(shares, 4),
                source="sdk:weather-bot:stop-loss",
                price=sell_price,
            )

            if result.get("executed"):
                # Definitive: order accepted. Do not retry.
                self.stop_loss_fired.add(mid)
                send_telegram(
                    f"🛑 *Stop-loss fired*\n{title}\n"
                    f"Side {side.upper()} | {shares:.2f} shares @ {float(pos.get('avg_cost', 0)):.3f}\n"
                    f"Unrealized: ${pnl:+.2f} ({pct * 100:+.1f}%)\n"
                    f"Sell order placed (GTC) @ {round(sell_price, 2)}",
                    self.config,
                )
                continue

            err = str(result.get("error", "unknown"))
            err_lower = err.lower()
            # Transient / fixable errors: DO NOT mark fired, let next cycle retry
            # with a fresh price. Tick size violations happen when Simmer's
            # default price is fractional; timeouts and 5xx are pure transient.
            is_transient = any(kw in err_lower for kw in [
                "timed out", "timeout", "httpsconnectionpool", "tick size",
                "503", "502", "504", "gateway", "connection reset",
            ])
            if is_transient:
                logger.warning(f"stop_loss transient error (will retry): {err[:200]}")
                # No Telegram spam on retries — would flood if the endpoint is
                # down for 10 min.
                continue

            # Legitimate rejection (insufficient shares, order invalid for
            # non-tick reasons, etc.) → mark fired so we stop retrying.
            self.stop_loss_fired.add(mid)
            logger.warning(f"stop_loss sell rejected (giving up on this market): {err[:200]}")
            send_telegram(
                f"⚠️ *Stop-loss tried, sell rejected*\n{title}\n"
                f"{pct * 100:+.1f}% | {err[:150]}",
                self.config,
            )

        self.state["stop_loss_fired"] = list(self.stop_loss_fired)
        save_state(self.state)

    def run(self) -> None:
        logger.info("=" * 60)
        logger.info(f"WEATHER BOT STARTING [venue={self.venue.upper()}]")
        logger.info(f"Min edge: {self.min_edge}, Max bet: {self.max_bet}{self.currency}, Dry run: {self.dry_run}")
        logger.info(f"Cycle interval: {self.cycle_interval}s")
        logger.info("=" * 60)

        bal = fetch_balance(self.api_key, venue=self.venue, wallet_address=self.wallet_address)
        logger.info(f"Starting balance: {bal} {self.currency}")
        send_telegram(
            f"*Weather Bot started* [{self.venue}]\nBalance: {bal} {self.currency}\n"
            f"Min edge: {self.min_edge} | Max bet: {self.max_bet}{self.currency}",
            self.config,
        )

        last_summary_hour = ""
        while self.running:
            try:
                self.cycle()
            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)
                time.sleep(30)
                continue

            # Periodic summary every 6 hours (00, 06, 12, 18 UTC)
            current_hour = time.strftime("%Y-%m-%d-%H", time.gmtime())
            hour_int = int(time.strftime("%H", time.gmtime()))
            if current_hour != last_summary_hour and hour_int % 6 == 0:
                try:
                    self._send_summary()
                    last_summary_hour = current_hour
                except Exception as e:
                    logger.warning(f"Summary failed: {e}")

            for _ in range(self.cycle_interval):
                if not self.running:
                    break
                time.sleep(1)

        logger.info("Weather bot stopped")
        send_telegram("*Weather Bot stopped*", self.config)

    def _send_summary(self) -> None:
        """Send periodic Telegram summary with balance + stats + skip counters."""
        bal = fetch_balance(self.api_key, venue=self.venue, wallet_address=self.wallet_address)
        wins = self.state.get("wins", 0)
        losses = self.state.get("losses", 0)
        pnl = self.state.get("pnl", 0.0)
        total = self.state.get("total_trades", 0)
        wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

        msg_parts = [
            "📊 *Weather Bot Summary*",
            f"Balance: {bal:.2f} {self.currency}",
            f"Total trades: {total}",
            f"Wins/Losses: {wins}W / {losses}L (WR: {wr:.0f}%)",
            f"PnL: {pnl:+.2f} {self.currency}",
        ]

        # Skip counters for the current window
        nonzero = [(k, v) for k, v in self._skip_counts.items() if v > 0]
        if nonzero:
            nonzero.sort(key=lambda kv: kv[1], reverse=True)
            trades_executed_window = max(0, total - self._trades_at_window_start)
            total_evaluated = sum(v for _, v in nonzero) + trades_executed_window
            msg_parts.append("")
            msg_parts.append("Skips last 6h:")
            for k, v in nonzero:
                msg_parts.append(f"  {k}: {v}")
            msg_parts.append(f"  (total evaluated: ~{total_evaluated})")

        send_telegram("\n".join(msg_parts), self.config)

        # Reset window counters so the next summary shows only the next 6h
        for k in self._skip_counts:
            self._skip_counts[k] = 0
        self._trades_at_window_start = total


def main() -> None:
    config = load_config()
    bot = WeatherBot(config)

    def shutdown(sig, frame):
        logger.info("Shutdown signal received")
        bot.running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    bot.run()


if __name__ == "__main__":
    main()
