"""Sports copy-trade bot for Simmer SIM (validation phase).

Watches a curated list of profitable Polymarket sports whales (refreshed
daily by scripts/whale_scout.py). Polls each whale's recent activity. When
a whale opens a NEW BUY trade on a market we can mirror in Simmer, we copy
it with a fixed bet.

SIM-only by design. Migrate to Polymarket real after 7+ days of positive
SIM P&L validation.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONFIG_PATH = os.path.join(HERE, "config.json")
LOG_PATH = os.path.join(HERE, "sports-bot.log")
STATE_PATH = os.path.join(HERE, "state.json")
WHALES_PATH = os.path.join(ROOT, "memory", "whales_sports.json")
CAL_LOG = os.path.join(ROOT, "memory", "sports_calibration_log.jsonl")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Project-wide HTTP endpoints
POLYMARKET_DATA = "https://data-api.polymarket.com"
SIMMER_API = "https://api.simmer.markets/api/sdk"

# Logging
logger = logging.getLogger("sports-bot")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
_fh = RotatingFileHandler(LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
_fh.setFormatter(_fmt)
logger.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "(SportsBot, trading@polybot)"})


# ---------------------------------------------------------------------------
# Config & state
# ---------------------------------------------------------------------------
def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"No config at {CONFIG_PATH}. Copy config.example.json and fill in.")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_state() -> dict:
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"copied_trade_hashes": [], "last_seen_ts": {}, "total_copies": 0,
                "blocked_whales": {}, "consecutive_losses": {}}


def save_state(state: dict) -> None:
    try:
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning(f"save_state failed: {e}")


def log_calibration(record: dict) -> None:
    os.makedirs(os.path.dirname(CAL_LOG), exist_ok=True)
    try:
        with open(CAL_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning(f"cal log failed: {e}")


# ---------------------------------------------------------------------------
# Whales: load curated list, refresh from disk each cycle
# ---------------------------------------------------------------------------
def load_whales() -> list[dict]:
    if not os.path.exists(WHALES_PATH):
        logger.warning(f"No whales file at {WHALES_PATH}. Run whale_scout.py first.")
        return []
    try:
        with open(WHALES_PATH) as f:
            payload = json.load(f)
        return payload.get("whales", [])
    except Exception as e:
        logger.error(f"load_whales failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Polymarket: fetch whale recent trades
# ---------------------------------------------------------------------------
def fetch_whale_activity(wallet: str, since_ts: int = 0, limit: int = 20) -> list[dict]:
    try:
        params = {
            "user": wallet,
            "type": "TRADE",
            "sortBy": "TIMESTAMP",
            "sortDirection": "DESC",
            "limit": limit,
        }
        if since_ts > 0:
            params["start"] = since_ts
        r = SESSION.get(f"{POLYMARKET_DATA}/activity", params=params, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.debug(f"fetch_whale_activity({wallet[:8]}) err: {e}")
        return []


# ---------------------------------------------------------------------------
# Simmer: find market by Polymarket condition_id, place SIM trade
# ---------------------------------------------------------------------------
def simmer_find_by_condition(condition_id: str, api_key: str) -> str | None:
    """Search Simmer's catalog for a market matching this Polymarket condition_id.

    Simmer imports many Polymarket markets; ours might be already there. If
    not found, we skip (don't try to import — keeps things simple for SIM
    validation).
    """
    try:
        r = SESSION.get(
            f"{SIMMER_API}/markets",
            params={"condition_id": condition_id, "limit": 5},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        body = r.json()
        markets = body.get("markets", body) if isinstance(body, dict) else body
        if not markets:
            return None
        m = markets[0] if isinstance(markets, list) else markets
        return m.get("id") or m.get("market_id")
    except Exception as e:
        logger.debug(f"simmer_find_by_condition err: {e}")
        return None


def simmer_place_trade(
    market_id: str, side: str, amount: float, reason: str, api_key: str
) -> dict:
    try:
        r = SESSION.post(
            f"{SIMMER_API}/trade",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "market_id": market_id,
                "side": side.lower(),
                "amount": amount,
                "venue": "sim",
                "reasoning": reason,
                "source": "sdk:polybot-sports",
            },
            timeout=30,
        )
        r.raise_for_status()
        return {"executed": True, "response": r.json()}
    except Exception as e:
        return {"executed": False, "error": str(e)[:200]}


def simmer_balance(api_key: str) -> float:
    try:
        r = SESSION.get(
            f"{SIMMER_API}/agents/me",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        r.raise_for_status()
        d = r.json()
        return float(d.get("balance") or d.get("sim_balance") or 0)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Cycle: detect new whale trades, mirror in SIM
# ---------------------------------------------------------------------------
def cycle(config: dict, state: dict) -> None:
    whales = load_whales()
    if not whales:
        logger.info("No whales loaded. Skip cycle.")
        return

    api_key = config["simmer_api_key"]
    bet_size = float(config.get("bet_size", 10.0))
    max_copies_per_cycle = int(config.get("max_copies_per_cycle", 3))
    min_balance = bet_size * 1.5

    bal = simmer_balance(api_key)
    if bal < min_balance:
        logger.warning(f"Balance {bal:.0f} SIM < {min_balance:.0f}. Skipping cycle.")
        return

    last_seen: dict = state.get("last_seen_ts", {})
    copied_hashes: set[str] = set(state.get("copied_trade_hashes", []))
    blocked: dict = state.get("blocked_whales", {})

    copies_this_cycle = 0

    for whale in whales[:15]:  # cap to top 15 to control rate
        wallet = whale["wallet"]

        # Skip if whale is in temporary block (after consecutive losses)
        if wallet in blocked:
            block_until = blocked.get(wallet, 0)
            if time.time() < block_until:
                continue
            del blocked[wallet]

        # Fetch recent activity since last-seen timestamp for this whale
        since_ts = int(last_seen.get(wallet, 0))
        trades = fetch_whale_activity(wallet, since_ts=since_ts)
        if not trades:
            continue

        # Process oldest-first so timestamp order is correct
        for t in reversed(trades):
            tx_hash = str(t.get("transactionHash") or t.get("id") or "")
            if not tx_hash or tx_hash in copied_hashes:
                continue

            ts = int(t.get("timestamp") or 0)
            if ts < since_ts:
                continue

            # Update last seen for this whale even if we skip
            last_seen[wallet] = max(last_seen.get(wallet, 0), ts)

            side = (t.get("side") or "").upper()
            condition_id = t.get("conditionId") or ""
            outcome = t.get("outcome") or ""
            usdc_size = float(t.get("usdcSize") or 0)
            price = float(t.get("price") or 0)

            # Filter: only copy BUY trades on actual outcomes
            if side != "BUY":
                continue
            if not condition_id or not outcome:
                continue
            # Filter: skip dust trades by whale (real signal is sized bets)
            if usdc_size < 50:
                continue
            # Filter: skip extreme prices
            if price < 0.05 or price > 0.95:
                continue

            # Find this market in Simmer's catalog
            sim_market_id = simmer_find_by_condition(condition_id, api_key)
            if not sim_market_id:
                logger.info(
                    f"  No Simmer match for cond={condition_id[:10]}... "
                    f"(whale {whale['name']} bet ${usdc_size:.0f})"
                )
                copied_hashes.add(tx_hash)  # don't retry
                continue

            # Map outcome → side. Polymarket outcomes are typically "Yes"/"No"
            # for binary markets. For multi-outcome (futures, sports rosters)
            # the outcome name is the "side" we mirror as YES on that token.
            outcome_lower = outcome.lower()
            mirror_side = "yes" if outcome_lower in ("yes", "y", "true", "1") else "no"
            # If the market is "Will X win?", a BUY on outcome=Y means YES on token Y.
            # Without deeper market intro, the safest mirror is YES on the same outcome.
            if outcome_lower not in ("yes", "no", "y", "n", "true", "false", "1", "0"):
                # Multi-outcome — too complex for v1, skip
                logger.info(
                    f"  Skip multi-outcome market: {whale['name']} bought "
                    f"'{outcome}' (cond {condition_id[:10]})"
                )
                copied_hashes.add(tx_hash)
                continue

            # Place mirror trade in SIM
            reason = (
                f"Copy from {whale['name']} ({wallet[:8]}): "
                f"BUY {outcome} @ {price:.2f} for ${usdc_size:.0f}. "
                f"Whale ROI {whale.get('roi_pct_week', 0):.1f}%/week"
            )
            logger.info(
                f"COPY: {whale['name']} {outcome} @ {price:.2f} "
                f"size ${usdc_size:.0f} → SIM bet ${bet_size:.0f}"
            )

            result = simmer_place_trade(sim_market_id, mirror_side, bet_size, reason, api_key)
            copied_hashes.add(tx_hash)
            copies_this_cycle += 1

            log_calibration({
                "timestamp": time.time(),
                "venue": "sim",
                "source": "sports-bot",
                "whale_wallet": wallet,
                "whale_name": whale.get("name"),
                "whale_roi_pct": whale.get("roi_pct_week"),
                "condition_id": condition_id,
                "sim_market_id": sim_market_id,
                "side": mirror_side,
                "outcome": outcome,
                "whale_price": price,
                "whale_size": usdc_size,
                "bet_amount": bet_size,
                "tx_hash": tx_hash,
                "copy_executed": result.get("executed", False),
                "error": result.get("error", ""),
            })

            if not result.get("executed"):
                logger.warning(f"  Copy failed: {result.get('error', 'unknown')[:120]}")

            if copies_this_cycle >= max_copies_per_cycle:
                break

        if copies_this_cycle >= max_copies_per_cycle:
            break

    # Persist state
    state["last_seen_ts"] = last_seen
    state["copied_trade_hashes"] = list(copied_hashes)[-2000:]  # keep recent only
    state["blocked_whales"] = blocked
    state["total_copies"] = state.get("total_copies", 0) + copies_this_cycle
    save_state(state)

    if copies_this_cycle == 0:
        logger.info(f"Cycle: 0 new copies (checked {len(whales)} whales, balance ${bal:.0f})")
    else:
        logger.info(f"Cycle: {copies_this_cycle} copies executed, balance ${bal:.0f}")


def main() -> None:
    config = load_config()
    state = load_state()
    interval = int(config.get("cycle_interval_seconds", 60))

    logger.info("=" * 60)
    logger.info(f"SPORTS BOT STARTING [venue=sim]")
    logger.info(f"Bet size: ${config.get('bet_size', 10):.0f} | "
                f"Max copies/cycle: {config.get('max_copies_per_cycle', 3)} | "
                f"Interval: {interval}s")
    bal = simmer_balance(config["simmer_api_key"])
    logger.info(f"Starting balance: {bal:.2f} SIM")
    logger.info("=" * 60)

    while True:
        try:
            cycle(config, state)
        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)
            time.sleep(30)
        time.sleep(interval)


if __name__ == "__main__":
    main()
