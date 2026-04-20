"""Crypto Up/Down arbitrage bot for Simmer SIM.

Strategy:
1. Find Polymarket "Bitcoin/Ethereum/XRP Up or Down - [time]" markets (short windows)
2. Fetch spot price from Binance (free public API)
3. Compute momentum-based fair probability
4. If |P_fair - P_market| > min_edge AND entry price is cheap → trade
5. Kelly sizing, edge-reversal exit

SIM-only by design. Migrate to Polymarket once proven.
"""

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tools.polymarket import get_markets, get_market_detail, place_order, get_positions  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("crypto-bot")

CONFIG_PATH = os.path.join(HERE, "config.json")
STATE_PATH = os.path.join(HERE, "state.json")
CAL_LOG = os.path.join(ROOT, "memory", "crypto_calibration_log.jsonl")

BINANCE_API = "https://api.binance.com/api/v3"

# Title patterns — capture asset and end time
# Examples:
#   "Bitcoin Up or Down - April 21, 10:00AM-10:05AM ET"
#   "Ethereum Up or Down - April 21, 10:00-10:15 ET"
TITLE_RE = re.compile(
    r"(Bitcoin|Ethereum|XRP|Solana|Dogecoin)\s+Up\s+or\s+Down\s*-\s*"
    r"(\w+)\s+(\d+),?\s+"
    r"(\d{1,2}):(\d{2})(?:AM|PM|am|pm)?\s*-\s*"
    r"(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\s*ET",
    re.IGNORECASE,
)

ASSET_TO_BINANCE = {
    "Bitcoin": "BTCUSDT",
    "Ethereum": "ETHUSDT",
    "XRP": "XRPUSDT",
    "Solana": "SOLUSDT",
    "Dogecoin": "DOGEUSDT",
}


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"No config at {CONFIG_PATH}. Copy config.example.json and fill in.")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return {"open_positions": {}, "traded_market_ids": []}
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {"open_positions": {}, "traded_market_ids": []}


def save_state(state: dict) -> None:
    try:
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Save state failed: {e}")


def log_calibration(record: dict) -> None:
    os.makedirs(os.path.dirname(CAL_LOG), exist_ok=True)
    try:
        with open(CAL_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"Cal log failed: {e}")


def parse_end_time(title: str) -> tuple[str, datetime] | None:
    """Extract asset + end datetime (UTC) from market title."""
    m = TITLE_RE.search(title)
    if not m:
        return None
    asset = m.group(1).title()
    month_name, day, _h1, _m1, h2, m2, ampm = m.group(2), m.group(3), m.group(4), m.group(5), m.group(6), m.group(7), m.group(8)

    try:
        month = datetime.strptime(month_name[:3], "%b").month
    except Exception:
        return None

    hour = int(h2)
    minute = int(m2)
    if ampm and ampm.lower() == "pm" and hour < 12:
        hour += 12
    if ampm and ampm.lower() == "am" and hour == 12:
        hour = 0

    # ET = UTC-4 (EDT) in April. Assume EDT.
    year = datetime.now(timezone.utc).year
    try:
        et_dt = datetime(year, month, int(day), hour, minute)
    except Exception:
        return None
    utc_dt = et_dt.replace(tzinfo=timezone.utc)  # treat as naive; add 4h for EDT → UTC
    from datetime import timedelta
    utc_dt = utc_dt + timedelta(hours=4)
    return asset, utc_dt


def fetch_spot_price(symbol: str) -> float | None:
    try:
        r = requests.get(f"{BINANCE_API}/ticker/price", params={"symbol": symbol}, timeout=5)
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception as e:
        logger.debug(f"Spot fetch {symbol} failed: {e}")
        return None


def fetch_klines(symbol: str, interval: str = "1m", limit: int = 10) -> list[dict] | None:
    """Recent candles from Binance. Returns list of {open, high, low, close, volume, close_time_ms}."""
    try:
        r = requests.get(
            f"{BINANCE_API}/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=5,
        )
        r.raise_for_status()
        raw = r.json()
        return [
            {
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
                "close_time_ms": int(c[6]),
            }
            for c in raw
        ]
    except Exception as e:
        logger.debug(f"Klines {symbol} failed: {e}")
        return None


def compute_fair_probability(klines: list[dict], spot: float) -> float:
    """Estimate P(price goes UP in next window) from recent momentum.

    Heuristic baseline — refine once we see calibration:
    - momentum_2m: (spot - close 2 min ago) / close
    - momentum_5m: (spot - close 5 min ago) / close
    - realized_vol (std of 1m returns)
    Map momentum → probability using sigmoid-ish curve.
    """
    if len(klines) < 5:
        return 0.5

    close_2m = klines[-3]["close"] if len(klines) >= 3 else klines[0]["close"]
    close_5m = klines[-6]["close"] if len(klines) >= 6 else klines[0]["close"]

    mom_2m = (spot - close_2m) / close_2m if close_2m else 0
    mom_5m = (spot - close_5m) / close_5m if close_5m else 0

    # Blended momentum (recent weighs more)
    momentum = 0.7 * mom_2m + 0.3 * mom_5m

    # Realized vol (std-ish) — use range of last 5 candles as proxy
    recent = klines[-5:]
    ranges = [(c["high"] - c["low"]) / c["close"] for c in recent if c["close"]]
    vol = sum(ranges) / len(ranges) if ranges else 0.003

    # Normalize: momentum in units of vol (z-score-ish)
    z = momentum / vol if vol > 0 else 0
    # Clamp and convert to probability via logistic
    # At z=0 → P=0.5. At z=+1 → P≈0.62. At z=+3 → P≈0.80 (but we clamp).
    z = max(-3, min(3, z))
    import math
    p_up = 1 / (1 + math.exp(-z * 0.55))
    return p_up


def evaluate_market(market: dict, config: dict, verbose: bool = False) -> dict | None:
    """Decide whether to trade this market."""
    title = market.get("title", market.get("question", ""))
    market_id = market.get("market_id", market.get("id", ""))

    parsed = parse_end_time(title)
    if not parsed:
        if verbose:
            logger.info(f"  SKIP (title parse fail): {title[:80]}")
        return None
    asset, end_utc = parsed

    # Window: only trade if market ends in [60s, 15min] from now
    now = datetime.now(timezone.utc)
    seconds_to_end = (end_utc - now).total_seconds()
    if seconds_to_end < 60 or seconds_to_end > 900:
        if verbose:
            logger.info(f"  SKIP (window {seconds_to_end:.0f}s out of [60,900]): {title[:80]}")
        return None

    symbol = ASSET_TO_BINANCE.get(asset)
    if not symbol:
        if verbose:
            logger.info(f"  SKIP (no binance mapping for {asset}): {title[:80]}")
        return None

    spot = fetch_spot_price(symbol)
    if spot is None:
        if verbose:
            logger.info(f"  SKIP (spot fetch fail {symbol}): {title[:80]}")
        return None
    klines = fetch_klines(symbol, "1m", 10)
    if not klines or len(klines) < 5:
        if verbose:
            logger.info(f"  SKIP (klines fetch fail {symbol}): {title[:80]}")
        return None

    p_fair = compute_fair_probability(klines, spot)

    # Get current market prices
    detail = get_market_detail(market_id, venue="sim", simmer_api_key=config["simmer_api_key"])
    if "error" in detail:
        if verbose:
            logger.info(f"  SKIP (detail error): {title[:80]}")
        return None
    yes_price = detail.get("yes_price", 0.5)
    no_price = detail.get("no_price", 0.5)

    # YES = price goes up
    edge_yes = p_fair - yes_price
    edge_no = (1 - p_fair) - no_price

    min_edge = config.get("min_edge", 0.15)
    max_entry = config.get("max_entry", 0.50)

    if verbose:
        logger.info(
            f"  {asset} {seconds_to_end:.0f}s: spot={spot:.2f} P_fair={p_fair:.2f} "
            f"yes={yes_price:.2f} no={no_price:.2f} "
            f"edge_yes={edge_yes:+.2f} edge_no={edge_no:+.2f}"
        )

    # Pick best side
    if edge_yes > min_edge and yes_price <= max_entry:
        return {
            "market_id": market_id,
            "title": title,
            "asset": asset,
            "spot": spot,
            "p_fair": round(p_fair, 3),
            "p_market_yes": yes_price,
            "side": "YES",
            "entry_price": yes_price,
            "edge": round(edge_yes, 3),
            "seconds_to_end": int(seconds_to_end),
        }
    if edge_no > min_edge and no_price <= max_entry:
        return {
            "market_id": market_id,
            "title": title,
            "asset": asset,
            "spot": spot,
            "p_fair": round(p_fair, 3),
            "p_market_yes": yes_price,
            "side": "NO",
            "entry_price": no_price,
            "edge": round(edge_no, 3),
            "seconds_to_end": int(seconds_to_end),
        }
    return None


def kelly_size(edge: float, entry: float, balance: float, max_bet: float) -> float:
    """Quarter-Kelly. Conservative because our model is unverified."""
    if entry <= 0 or entry >= 1:
        return 0
    b = (1 - entry) / entry  # profit odds
    p = edge + entry  # our P_win (edge = P - market_price)
    q = 1 - p
    kelly = (b * p - q) / b if b > 0 else 0
    kelly = max(0, min(kelly, 0.25))  # cap at 25%
    quarter_kelly = kelly * 0.25
    bet = balance * quarter_kelly
    return max(5.0, min(bet, max_bet))  # Simmer min order ≈ 5


def check_edge_reversal(
    state: dict, config: dict
) -> list[tuple[str, str, str]]:
    """For each open position, recompute edge. If flipped → sell. Returns list of (market_id, side, reason)."""
    to_exit: list[tuple[str, str, str]] = []
    for market_id, pos in list(state.get("open_positions", {}).items()):
        try:
            detail = get_market_detail(market_id, venue="sim", simmer_api_key=config["simmer_api_key"])
            if "error" in detail:
                continue
            title = detail.get("title", "")
            parsed = parse_end_time(title)
            if not parsed:
                continue
            asset, end_utc = parsed
            now = datetime.now(timezone.utc)
            secs = (end_utc - now).total_seconds()
            if secs < 30:
                # Market closing imminently, don't bother exiting
                continue
            symbol = ASSET_TO_BINANCE.get(asset)
            spot = fetch_spot_price(symbol) if symbol else None
            klines = fetch_klines(symbol, "1m", 10) if symbol else None
            if not spot or not klines:
                continue
            p_fair = compute_fair_probability(klines, spot)
            our_side = pos["side"].upper()
            our_entry = pos["entry_price"]
            our_p = p_fair if our_side == "YES" else (1 - p_fair)
            current_market = detail.get("yes_price", 0.5) if our_side == "YES" else detail.get("no_price", 0.5)
            current_edge = our_p - current_market
            if current_edge < -0.10:
                reason = f"Edge reversed: {current_edge:+.2f} (fair={our_p:.2f} vs market={current_market:.2f})"
                to_exit.append((market_id, our_side, reason))
        except Exception as e:
            logger.debug(f"Exit check failed for {market_id[:8]}: {e}")
    return to_exit


def get_sim_balance(api_key: str) -> float:
    try:
        r = requests.get(
            "https://api.simmer.markets/api/sdk/agents/me",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        return float(data.get("sim_balance", data.get("balance", 0)) or 0)
    except Exception as e:
        logger.warning(f"Balance fetch failed: {e}")
        return 0.0


def cycle(config: dict, state: dict, cycle_num: int) -> None:
    balance = get_sim_balance(config["simmer_api_key"])
    max_bet = float(config.get("max_bet", 200))
    min_balance = max_bet * 1.1
    if balance < min_balance:
        logger.warning(f"Balance too low: {balance:.0f} $SIM (need {min_balance:.0f}). Skipping.")
        return

    # Fetch all markets, filter crypto Up/Down
    markets = get_markets(
        keyword="up or down",
        limit=100,
        simmer_api_key=config["simmer_api_key"],
        venue="sim",
    )
    if isinstance(markets, dict) and "error" in markets:
        logger.error(f"Markets fetch: {markets['error']}")
        return

    candidates = []
    for m in (markets if isinstance(markets, list) else []):
        title = m.get("title", m.get("question", ""))
        if "up or down" not in title.lower():
            continue
        # Skip if already traded this cycle batch
        market_id = m.get("market_id", m.get("id", ""))
        if market_id in state.get("open_positions", {}):
            continue
        candidates.append(m)

    logger.info(f"Cycle {cycle_num}: {len(candidates)} candidate crypto markets, balance={balance:.0f} $SIM")

    # Evaluate each
    verbose = (cycle_num % 5 == 1)  # verbose every 5 cycles
    opps = []
    for m in candidates[:30]:  # cap to avoid rate limits
        dec = evaluate_market(m, config, verbose=verbose)
        if dec:
            opps.append(dec)

    opps.sort(key=lambda o: abs(o["edge"]), reverse=True)

    # Execute top N
    max_trades = int(config.get("max_trades_per_cycle", 3))
    for opp in opps[:max_trades]:
        bet = kelly_size(opp["edge"], opp["entry_price"], balance, max_bet)
        if bet < 5:
            continue
        logger.info(
            f"  TRADE {opp['side']} on '{opp['title'][:70]}' | "
            f"asset={opp['asset']} spot={opp['spot']:.2f} | "
            f"P_fair={opp['p_fair']} vs market={opp['entry_price']} | edge={opp['edge']:+.2f} | "
            f"bet={bet:.1f} $SIM | {opp['seconds_to_end']}s to end"
        )
        result = place_order(
            market_id=opp["market_id"],
            market_title=opp["title"],
            side=opp["side"],
            amount_usdc=bet,
            reason=f"Momentum arb: edge={opp['edge']:+.2f}, P_fair={opp['p_fair']}",
            simmer_api_key=config["simmer_api_key"],
            venue="sim",
            entry_price=opp["entry_price"],
        )
        if result.get("executed"):
            state.setdefault("open_positions", {})[opp["market_id"]] = {
                "title": opp["title"],
                "side": opp["side"],
                "entry_price": opp["entry_price"],
                "p_fair_at_entry": opp["p_fair"],
                "bet": bet,
                "opened_at": time.time(),
            }
            log_calibration({
                "timestamp": time.time(),
                "venue": "sim",
                "source": "crypto-bot",
                "market_id": opp["market_id"],
                "title": opp["title"],
                "asset": opp["asset"],
                "side": opp["side"],
                "entry_price": opp["entry_price"],
                "amount": bet,
                "p_predicted": opp["p_fair"] if opp["side"] == "YES" else (1 - opp["p_fair"]),
                "edge": opp["edge"],
                "spot_at_entry": opp["spot"],
            })
            balance -= bet
        else:
            err = result.get("error", "unknown")
            logger.warning(f"  Trade failed: {err}")

    # Check exits for existing positions
    to_exit = check_edge_reversal(state, config)
    for market_id, side, reason in to_exit:
        pos = state["open_positions"].get(market_id)
        if not pos:
            continue
        logger.info(f"  EXIT {side} on '{pos['title'][:70]}' | {reason}")
        # Simmer doesn't have easy sell endpoint, we close by buying opposite
        # Simplest: log the intent and remove from state. Manual close from dashboard for now.
        # TODO: implement actual sell via SDK once we validate the signal
        state["open_positions"].pop(market_id, None)

    save_state(state)


def main():
    config = load_config()
    state = load_state()
    interval = int(config.get("cycle_interval_seconds", 30))
    logger.info("=" * 60)
    logger.info("CRYPTO BOT STARTING [venue=SIM]")
    logger.info(f"Min edge: {config.get('min_edge', 0.15)} | Max entry: {config.get('max_entry', 0.50)} | "
                f"Max bet: {config.get('max_bet', 200)} $SIM | Cycle: {interval}s")
    logger.info("=" * 60)

    cycle_num = 0
    while True:
        cycle_num += 1
        try:
            cycle(config, state, cycle_num)
        except Exception as e:
            logger.error(f"Cycle {cycle_num} error: {e}")
        time.sleep(interval)


if __name__ == "__main__":
    main()
