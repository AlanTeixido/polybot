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

    Forecast error grows with horizon:
    - 1 day: std ~1.5°C
    - 3 days: std ~2.5°C
    - 7 days: std ~3.5°C
    """
    std = max(1.5, 1.0 + days_ahead * 0.4)  # linearly growing uncertainty

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
    """Estimate P(threshold_low <= temp <= threshold_high)."""
    std = max(1.5, 1.0 + days_ahead * 0.4)
    z_low = (threshold_low - forecast_temp) / std
    z_high = (threshold_high - forecast_temp) / std
    p_low = 0.5 * (1 + math.erf(z_low / math.sqrt(2)))
    p_high = 0.5 * (1 + math.erf(z_high / math.sqrt(2)))
    return round(p_high - p_low, 3)


# ---------------------------------------------------------------------------
# Simmer API
# ---------------------------------------------------------------------------
def fetch_markets(api_key: str, limit: int = 100) -> list[dict]:
    try:
        resp = SESSION.get(
            f"{SIMMER_API}/markets",
            params={"limit": limit},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("markets", [])
    except Exception as e:
        logger.error(f"fetch_markets failed: {e}")
        return []


def fetch_balance(api_key: str, venue: str = "sim") -> float:
    """Returns balance for the configured venue (sim $SIM or polymarket USDC)."""
    try:
        resp = SESSION.get(
            f"{SIMMER_API}/agents/me",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if venue == "polymarket":
            return float(
                data.get("polymarket_balance")
                or data.get("polymarket_usdc")
                or data.get("balance", 0)
            )
        return float(data.get("balance", 0))
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
) -> dict:
    if dry_run:
        logger.info(f"[DRY RUN] {venue} {side} {amount} on {market_id}: {reason}")
        return {"executed": False, "dry_run": True}
    try:
        resp = SESSION.post(
            f"{SIMMER_API}/trade",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "market_id": market_id,
                "side": side.lower(),
                "amount": amount,
                "venue": venue,
                "reasoning": reason[:200],
                "source": "sdk:weather-bot",
            },
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code >= 400:
            logger.error(f"place_order HTTP {resp.status_code}: {resp.text[:300]}")
            return {"executed": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        return {"executed": True, "response": resp.json()}
    except Exception as e:
        logger.error(f"place_order failed: {e}")
        return {"executed": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Trading logic
# ---------------------------------------------------------------------------
def evaluate_market(market: dict, min_edge: float, verbose: bool = False) -> dict | None:
    """Evaluate a single weather market. Returns trade decision or None."""
    title = market.get("question", "")
    market_id = market.get("id", "")
    current_price = float(market.get("current_probability", market.get("current_price", 0.5)))

    parsed = parse_weather_market(title)
    if not parsed:
        if verbose:
            logger.info(f"  SKIP (parse failed): {title[:70]}")
        return None

    target_date = parse_target_date(title)
    if not target_date:
        if verbose:
            logger.info(f"  SKIP (no date): {title[:70]}")
        return None

    forecast_data = get_forecast(parsed["city"])
    if not forecast_data:
        if verbose:
            logger.info(f"  SKIP (no forecast): {parsed['city']} | {title[:60]}")
        return None

    day_forecast = forecast_data["forecasts"].get(target_date)
    if not day_forecast:
        if verbose:
            logger.info(f"  SKIP (no forecast for {target_date}): {parsed['city']} | {title[:50]}")
        return None

    # Pick the right temp based on metric
    temp_c = day_forecast.get("high_c") if parsed["metric"] == "high" else day_forecast.get("low_c")
    if temp_c is None:
        if verbose:
            logger.info(f"  SKIP (no temp): {parsed['city']} {target_date}")
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

    if abs_edge < min_edge:
        return None

    side = "yes" if edge > 0 else "no"
    # Entry price is the side we're buying
    entry_price = current_price if side == "yes" else (1 - current_price)

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
) -> float:
    """Size bet based on edge magnitude.

    Polymarket constraints:
    - Minimum $1 per order
    - Minimum 5 shares per order (cost = 5 * entry_price)
    - 10% fee (fee_rate_bps=1000)
    """
    max_bet = float(config.get("max_bet", config.get("max_bet_usdc", 50)))
    max_pct = float(config.get("max_pct_balance", 0.02))
    min_edge = float(config.get("min_edge", 0.15))

    abs_edge = abs(edge)
    # Linear scale: at min_edge → 0.3x, at 0.40+ → 1.0x
    scale = min(1.0, max(0.3, (abs_edge - min_edge) / (0.40 - min_edge) * 0.7 + 0.3))
    base = min(max_bet, balance * max_pct)
    bet = base * scale

    if venue == "polymarket":
        # Polymarket minimums: $1 order AND 5 shares minimum
        # 5 shares at entry_price = 5 * entry_price USDC needed
        min_for_5_shares = max(1.0, 5.0 * entry_price)
        if bet < min_for_5_shares:
            # Skip if we can't meet minimum without exceeding max_bet
            if min_for_5_shares > max_bet:
                return 0
            bet = min_for_5_shares

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
        self.venue = config.get("venue", "sim")
        self.min_edge = float(config.get("min_edge", 0.15))
        self.max_bet = float(config.get("max_bet", config.get("max_bet_usdc", 50)))
        self.dry_run = config.get("dry_run", False)
        self.cycle_interval = int(config.get("cycle_interval_seconds", 180))
        self.currency = "USDC" if self.venue == "polymarket" else "$SIM"

    def cycle(self) -> None:
        balance = fetch_balance(self.api_key, venue=self.venue)
        min_balance = 1 if self.venue == "polymarket" else 10
        if balance < min_balance:
            logger.warning(f"Balance too low: {balance} {self.currency}. Skipping cycle.")
            return

        markets = fetch_markets(self.api_key, limit=100)
        if not markets:
            logger.info("No markets fetched")
            return

        # Filter to weather markets
        weather_markets = [m for m in markets if any(kw in (m.get("question", "") or "").lower() for kw in ["temperature", "°c", "°f"])]

        # Verbose debug every 10 cycles
        self._cycle_num = getattr(self, "_cycle_num", 0) + 1
        verbose = (self._cycle_num % 10 == 1)
        logger.info(f"Cycle {self._cycle_num}: {len(markets)} markets, {len(weather_markets)} weather {'[VERBOSE]' if verbose else ''}")

        # Evaluate each
        opportunities = []
        for m in weather_markets:
            if m.get("id") in self.state.get("traded_markets", []):
                continue  # already traded this market
            decision = evaluate_market(m, self.min_edge, verbose=verbose)
            if decision:
                opportunities.append(decision)

        # Sort by edge descending
        opportunities.sort(key=lambda x: abs(x["edge"]), reverse=True)

        if not opportunities:
            logger.info("No opportunities with sufficient edge")
            return

        logger.info(f"Found {len(opportunities)} opportunities")

        # Execute top N per cycle
        max_per_cycle = int(self.config.get("max_trades_per_cycle", 3))
        min_bet = 1.0  # Simmer/Polymarket minimum $1 per order
        for opp in opportunities[:max_per_cycle]:
            bet = compute_bet_size(
                opp["edge"], balance, self.config,
                entry_price=opp.get("entry_price", 0.5),
                venue=self.venue,
            )
            if bet < min_bet:
                logger.info(f"SKIP (bet too small): {opp['title'][:50]} bet={bet}")
                continue

            reason = (
                f"{opp['city']} {opp['metric']} forecast {opp['forecast_temp_c']:.1f}C "
                f"vs threshold {opp['threshold_c']} {opp['comparison']} "
                f"(days_ahead={opp['days_ahead']}). "
                f"P_real={opp['p_yes_real']}, market={opp['market_price']:.2f}, edge={opp['edge']:+.2f}"
            )
            logger.info(f"TRADE [{self.venue}]: {opp['side'].upper()} {bet}{self.currency} on '{opp['title'][:60]}' | {reason}")

            result = place_order(
                self.api_key, opp["market_id"], opp["side"], bet, reason,
                venue=self.venue, dry_run=self.dry_run,
            )

            if result.get("executed"):
                self.state["traded_markets"].append(opp["market_id"])
                self.state["total_trades"] = self.state.get("total_trades", 0) + 1
                save_state(self.state)
                send_telegram(
                    f"*Weather Trade [{self.venue}]*\n{opp['side'].upper()} {bet}{self.currency}\n{opp['title'][:80]}\n"
                    f"Edge: {opp['edge']:+.2f} | P_real: {opp['p_yes_real']} | Price: {opp['market_price']:.2f}",
                    self.config,
                )
                balance -= bet
            else:
                err = result.get("error", "unknown")
                logger.error(f"Trade failed: {err}")
                # Hard stop if first real trade fails — fix before losing money
                if self.venue == "polymarket":
                    send_telegram(
                        f"*Weather Trade FAILED [{self.venue}]*\n{opp['title'][:80]}\nError: {err[:200]}",
                        self.config,
                    )

    def run(self) -> None:
        logger.info("=" * 60)
        logger.info(f"WEATHER BOT STARTING [venue={self.venue.upper()}]")
        logger.info(f"Min edge: {self.min_edge}, Max bet: {self.max_bet}{self.currency}, Dry run: {self.dry_run}")
        logger.info(f"Cycle interval: {self.cycle_interval}s")
        logger.info("=" * 60)

        bal = fetch_balance(self.api_key, venue=self.venue)
        logger.info(f"Starting balance: {bal} {self.currency}")
        send_telegram(
            f"*Weather Bot started* [{self.venue}]\nBalance: {bal} {self.currency}\n"
            f"Min edge: {self.min_edge} | Max bet: {self.max_bet}{self.currency}",
            self.config,
        )

        while self.running:
            try:
                self.cycle()
            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)
                time.sleep(30)
                continue

            for _ in range(self.cycle_interval):
                if not self.running:
                    break
                time.sleep(1)

        logger.info("Weather bot stopped")
        send_telegram("*Weather Bot stopped*", self.config)


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
