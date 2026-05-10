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
                "blocked_whales": {}, "consecutive_losses": {},
                "cond_copies": {}}


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


PERFORMANCE_PATH = os.path.join(ROOT, "memory", "whale_performance.json")


def load_whale_performance() -> dict:
    """Per-whale status from scripts/whale_performance.py.

    Returns dict[wallet_lower] -> { status: blocked|elite|normal|trial, ... }
    Empty dict if file doesn't exist yet (first hour after install).
    """
    if not os.path.exists(PERFORMANCE_PATH):
        return {}
    try:
        with open(PERFORMANCE_PATH) as f:
            payload = json.load(f)
        return {k.lower(): v for k, v in payload.get("by_wallet", {}).items()}
    except Exception as e:
        logger.warning(f"load_whale_performance failed: {e}")
        return {}


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
def simmer_find_by_condition(condition_id: str, api_key: str) -> tuple[str | None, str]:
    """Search Simmer's catalog for a market matching this Polymarket condition_id.

    Returns (market_id, title). title used downstream for noise filtering.
    """
    try:
        r = SESSION.get(
            f"{SIMMER_API}/markets",
            params={"condition_id": condition_id, "limit": 5},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if r.status_code != 200:
            return None, ""
        body = r.json()
        markets = body.get("markets", body) if isinstance(body, dict) else body
        if not markets:
            return None, ""
        m = markets[0] if isinstance(markets, list) else markets
        mid = m.get("id") or m.get("market_id")
        title = m.get("title") or m.get("question") or m.get("name") or m.get("slug") or ""
        return mid, title
    except Exception as e:
        logger.debug(f"simmer_find_by_condition err: {e}")
        return None, ""


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

    # Per-whale performance gates from whale_performance.py (hourly cron)
    perf = load_whale_performance()

    api_key = config["simmer_api_key"]
    bet_size = float(config.get("bet_size", 10.0))
    # Bumped 2026-05-10: 3 → 5 to give trial whales more chances to reach
    # MIN_N_FOR_CLASSIFICATION (=10 resolved) faster. With 7 copies/h prior,
    # discovering a new good whale took weeks. Higher cap accelerates the
    # trial → normal/elite/blocked transition.
    max_copies_per_cycle = int(config.get("max_copies_per_cycle", 5))
    min_balance = bet_size * 1.5
    # Safety cap: don't open new positions if open exposure > X% of total value.
    # Default 75%: leaves 25% of total SIM value as cash buffer.
    max_exposure_pct = float(config.get("max_exposure_pct", 0.75))

    bal = simmer_balance(api_key)
    if bal < min_balance:
        logger.warning(f"Balance {bal:.0f} SIM < {min_balance:.0f}. Skipping cycle.")
        return

    # Exposure check via portfolio endpoint
    try:
        r = SESSION.get(
            f"{SIMMER_API}/portfolio",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if r.status_code == 200:
            d = r.json()
            sim = d.get("sim", {})
            total_value = float(sim.get("balance") or 0)
            exposure = float(sim.get("total_exposure") or 0)
            if total_value > 0:
                exp_ratio = exposure / total_value
                if exp_ratio > max_exposure_pct:
                    logger.warning(
                        f"Exposure {exp_ratio*100:.0f}% > {max_exposure_pct*100:.0f}% "
                        f"(${exposure:.0f}/${total_value:.0f}). Skipping new trades."
                    )
                    return
    except Exception as e:
        logger.debug(f"exposure check failed: {e}")

    last_seen: dict = state.get("last_seen_ts", {})
    # Use dict (insertion-ordered) instead of set so the [-2000:] trim at save
    # time keeps the most recent hashes — set order is hash-based and would
    # silently drop newly added items.
    copied_hashes: dict[str, None] = dict.fromkeys(state.get("copied_trade_hashes", []))
    blocked: dict = state.get("blocked_whales", {})

    # Per-condition cooldown: prevent piling into the same market when the
    # same whale's order gets split into many fills, or when multiple whales
    # all crowd the same market. Tracks {condition_id: [ts, ts, ...]} (ts in
    # last 24h). New copies on a condition with >= max_copies_per_condition_24h
    # entries in the window are skipped. Default 2 = at most one extra copy
    # after the first, which already gives the same exposure as a doubled bet
    # without runaway scaling.
    cond_copies: dict[str, list[int]] = state.get("cond_copies", {})
    max_copies_per_cond = int(config.get("max_copies_per_condition_24h", 2))
    cond_window_sec = 24 * 3600
    now_ts = int(time.time())
    # Trim stale entries
    for cid in list(cond_copies.keys()):
        cond_copies[cid] = [t for t in cond_copies[cid] if now_ts - t < cond_window_sec]
        if not cond_copies[cid]:
            del cond_copies[cid]

    copies_this_cycle = 0

    for whale in whales[:15]:  # cap to top 15 to control rate
        wallet = whale["wallet"]

        # Skip if whale is in temporary block (after consecutive losses)
        if wallet in blocked:
            block_until = blocked.get(wallet, 0)
            if time.time() < block_until:
                continue
            del blocked[wallet]

        # Per-whale performance gate (auto-refreshed hourly by whale_performance.py)
        whale_perf = perf.get(wallet.lower(), {})
        whale_status = whale_perf.get("status", "trial")
        if whale_status == "blocked":
            # Skip permanently until next hourly recompute lifts the block
            continue

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

            # Pre-Simmer noise filter from Polymarket title (cheap, no extra API call).
            # Real check happens after Simmer lookup with the canonical title.
            poly_title = ""
            try:
                poly_title = (t.get("title") or t.get("slug") or "").lower()
            except Exception:
                pass
            if "up or down" in poly_title and 0.20 <= price <= 0.80:
                # Short-window crypto noise. The original 0.45-0.55 cap was too
                # narrow — audit on 2026-05-04 showed 162 such copies today at
                # prices 0.40-0.90 ($4,280) all bleeding small. Whales' "edge"
                # on 5-15 min crypto direction is just minute-level noise.
                # Genuine asymmetric entries (<0.20 or >0.80) might be real
                # signal so we keep that escape hatch.
                copied_hashes[tx_hash] = None
                continue

            # Tier-based bet sizing — better signal = bigger bet.
            # Default $10. "alltime_legend" = $15. "both" tier = $20.
            tier = whale.get("tier", "weekly_hot")
            if tier == "both_alltime+weekly":
                this_bet = bet_size * 2.0   # strongest signal: proven AND active
            elif tier == "alltime_legend":
                this_bet = bet_size * 1.5
            else:
                this_bet = bet_size

            # Elite multiplier on top of tier — whales that have actually been
            # making us money (per whale_performance.py daily recompute).
            # WR > 65% AND PnL > $50 SIM on N>=10 of OUR copies = "elite".
            if whale_status == "elite":
                this_bet = this_bet * 2.0

            # Filter: skip whales with weak weekly ROI (if data available)
            roi_week = whale.get("roi_pct_week")
            if roi_week is not None and roi_week < 3.0:
                # weekly hot but barely profitable — likely market maker fluke
                copied_hashes[tx_hash] = None
                continue

            # Per-condition cooldown: skip if we already copied this market enough.
            # Catches fill-splitting (one whale order = many fills) AND multiple
            # whales piling into the same market.
            already_on_cond = len(cond_copies.get(condition_id, []))
            if already_on_cond >= max_copies_per_cond:
                logger.debug(
                    f"  Skip cond={condition_id[:10]}: already {already_on_cond} "
                    f"copies in last 24h (max {max_copies_per_cond})"
                )
                copied_hashes[tx_hash] = None  # don't retry this fill
                continue

            # Find this market in Simmer's catalog
            sim_market_id, sim_title = simmer_find_by_condition(condition_id, api_key)
            if not sim_market_id:
                logger.info(
                    f"  No Simmer match for cond={condition_id[:10]}... "
                    f"(whale {whale['name']} bet ${usdc_size:.0f})"
                )
                copied_hashes[tx_hash] = None  # don't retry
                continue

            # Authoritative noise filter using Simmer's canonical title.
            # Catches the cases where Polymarket activity title is missing.
            sim_title_l = (sim_title or "").lower()
            if "up or down" in sim_title_l and 0.20 <= price <= 0.80:
                copied_hashes[tx_hash] = None
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
                copied_hashes[tx_hash] = None
                continue

            # Place mirror trade in SIM with tier-based sizing
            reason = (
                f"Copy from {whale['name']} ({wallet[:8]}, tier={tier}): "
                f"BUY {outcome} @ {price:.2f} for ${usdc_size:.0f}. "
                f"Whale week ROI {whale.get('roi_pct_week', 0):.1f}%, "
                f"alltime PnL ${whale.get('pnl_alltime', 0):.0f}"
            )
            elite_tag = " ⭐ELITE" if whale_status == "elite" else ""
            logger.info(
                f"COPY [{tier}{elite_tag}]: {whale['name']} {outcome} @ {price:.2f} "
                f"size ${usdc_size:.0f} → SIM bet ${this_bet:.0f}"
            )

            result = simmer_place_trade(sim_market_id, mirror_side, this_bet, reason, api_key)
            copied_hashes[tx_hash] = None
            cond_copies.setdefault(condition_id, []).append(now_ts)
            copies_this_cycle += 1

            log_calibration({
                "timestamp": time.time(),
                "venue": "sim",
                "source": "sports-bot",
                "whale_wallet": wallet,
                "whale_name": whale.get("name"),
                "whale_tier": tier,
                "whale_status": whale_status,  # trial|normal|elite|blocked
                "whale_roi_pct": whale.get("roi_pct_week"),
                "whale_pnl_alltime": whale.get("pnl_alltime"),
                "condition_id": condition_id,
                "sim_market_id": sim_market_id,
                "side": mirror_side,
                "outcome": outcome,
                "whale_price": price,
                "whale_size": usdc_size,
                "bet_amount": this_bet,
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
    state["cond_copies"] = cond_copies
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
