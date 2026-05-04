"""Per-whale performance tracker for the sports bot.

Reads sports_calibration_log.jsonl, fetches resolutions from Simmer,
computes per-whale WR + P&L over the last N copies. Outputs
memory/whale_performance.json with each whale tagged blocked/elite/normal.

Sports bot reads this file each cycle and adjusts behavior:
  blocked → skip the whale entirely (24h)
  elite   → 2x bet multiplier on top of tier sizing
  normal  → no change

Run hourly via cron:
  15 * * * * cd /root/polybot && /usr/bin/python3 scripts/whale_performance.py >> /root/polybot/whale_performance.log 2>&1
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPORTS_LOG = os.path.join(ROOT, "memory", "sports_calibration_log.jsonl")
WEATHER_CONFIG = os.path.join(ROOT, "polybot-weather", "config.json")
OUTPUT_FILE = os.path.join(ROOT, "memory", "whale_performance.json")
SIMMER_API = "https://api.simmer.markets/api/sdk"

# Significance + classification
MIN_N_FOR_CLASSIFICATION = 10   # need at least 10 copies before judging
WINDOW_LAST_N = 50              # only consider most recent 50 copies per whale
# Tightened 2026-05-04: previous (0.30 / -50) was too lenient — RN1 type whales
# (WR 42% / PnL ~$0) kept doing 100+ copies/day with no edge. New limits flag
# whales that aren't clearly profitable as "blocked" so we stop wasting bets.
BLOCK_WR_THRESHOLD = 0.45       # WR below this AND negative P&L → block
ELITE_WR_THRESHOLD = 0.65       # WR above this AND positive P&L → elite
ELITE_PNL_MIN = 50.0            # at least $50 SIM net to qualify
BLOCK_PNL_MAX = -25.0           # at least -$25 net to qualify for blocking

# Pre-trial virtual backtest config — applied to whales with no real copies yet.
# Saves us from sending money to whales that would have lost in backtest.
POLYMARKET_DATA = "https://data-api.polymarket.com"
BACKTEST_LIMIT = 200            # fetch last N Polymarket trades (more depth)
BACKTEST_MIN_AGE_DAYS = 5       # only consider trades this old (let them resolve)
BACKTEST_MIN_N_RESOLVED = 10    # need 10 resolved virtual trades to judge
BACKTEST_BLOCK_WR = 0.45        # virtual WR below this → block before any real bet
BACKTEST_BLOCK_PNL = -25.0      # AND virtual PnL below this


def simmer_find_by_condition(condition_id: str, api_key: str) -> str | None:
    """Find Simmer market_id for a Polymarket conditionId, or None."""
    try:
        r = requests.get(
            f"{SIMMER_API}/markets",
            params={"condition_id": condition_id, "limit": 1},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=8,
        )
        if r.status_code != 200:
            return None
        body = r.json()
        markets = body.get("markets", body) if isinstance(body, dict) else body
        if not markets:
            return None
        m = markets[0] if isinstance(markets, list) else markets
        return m.get("id") or m.get("market_id")
    except Exception:
        return None


def fetch_polymarket_activity(wallet: str, limit: int = 30) -> list[dict]:
    """Fetch a whale's recent Polymarket trades."""
    try:
        r = requests.get(
            f"{POLYMARKET_DATA}/activity",
            params={"user": wallet, "type": "TRADE",
                    "sortBy": "TIMESTAMP", "sortDirection": "DESC", "limit": limit},
            timeout=10,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def virtual_backtest(
    wallet: str,
    api_key: str,
    cond_to_sim: dict[str, str | None],
    outcome_cache: dict[str, bool | None],
) -> tuple[float, float, int]:
    """Simulate copying this whale's last N trades using the same filters as
    the real bot. Returns (wr, pnl_sim, n_resolved).

    Uses $10 virtual bet sizing, ignores fill-splitting (one copy per
    conditionId per direction). This mirrors the bot's per-condition cap.

    cond_to_sim and outcome_cache are passed in so lookups are cached across
    all whales in the same run (Polymarket conditions and Simmer markets are
    shared between whales).
    """
    trades = fetch_polymarket_activity(wallet, limit=BACKTEST_LIMIT)
    if not trades:
        return 0.0, 0.0, 0

    # Only consider trades old enough to have resolved.
    import time as _t
    cutoff_ts = _t.time() - BACKTEST_MIN_AGE_DAYS * 86400
    trades = [t for t in trades if (t.get("timestamp") or 0) <= cutoff_ts]
    if not trades:
        return 0.0, 0.0, 0

    seen_conds: set[str] = set()  # one virtual copy per (cond, side)
    wins, losses = 0, 0
    pnl = 0.0
    n_resolved = 0
    for t in trades:
        side = (t.get("side") or "").upper()
        if side != "BUY":
            continue
        cond = t.get("conditionId") or ""
        outcome = (t.get("outcome") or "").lower()
        if not cond or outcome not in ("yes", "no", "y", "n", "true", "false", "1", "0"):
            continue
        usdc_size = float(t.get("usdcSize") or 0)
        if usdc_size < 50:
            continue
        price = float(t.get("price") or 0)
        if price < 0.05 or price > 0.95:
            continue
        # Apply crypto Up/Down noise filter (same as bot)
        title = (t.get("title") or t.get("slug") or "").lower()
        if "up or down" in title and 0.20 <= price <= 0.80:
            continue
        # One virtual copy per (cond, side)
        key = f"{cond}:{outcome}"
        if key in seen_conds:
            continue
        seen_conds.add(key)
        # Resolve via Simmer (cached across whales)
        if cond not in cond_to_sim:
            cond_to_sim[cond] = simmer_find_by_condition(cond, api_key)
        sim_id = cond_to_sim[cond]
        if not sim_id:
            continue
        if sim_id not in outcome_cache:
            outcome_cache[sim_id] = fetch_market_outcome(sim_id, api_key)
        outcome_yes = outcome_cache[sim_id]
        if outcome_yes is None:
            continue
        n_resolved += 1
        mirror_yes = outcome in ("yes", "y", "true", "1")
        won = (outcome_yes and mirror_yes) or (not outcome_yes and not mirror_yes)
        bet = 10.0
        if won:
            wins += 1
            pnl += (1 - price) * bet / price
        else:
            losses += 1
            pnl -= bet

    wr = wins / n_resolved if n_resolved > 0 else 0.0
    return wr, pnl, n_resolved


def fetch_market_outcome(market_id: str, api_key: str) -> bool | None:
    """Return True if YES won, False if NO won, None if not resolved or error."""
    try:
        r = requests.get(
            f"{SIMMER_API}/markets/{market_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=8,
        )
        if r.status_code != 200:
            return None
        body = r.json()
        data = body.get("market", body) if isinstance(body, dict) else {}
        outcome = data.get("outcome")
        if outcome is None:
            return None
        if isinstance(outcome, bool):
            return outcome
        if isinstance(outcome, str):
            lo = outcome.lower()
            if lo in ("yes", "y", "true", "1"):
                return True
            if lo in ("no", "n", "false", "0"):
                return False
        return None
    except Exception:
        return None


def main() -> None:
    if not os.path.exists(SPORTS_LOG):
        print(f"No log at {SPORTS_LOG}")
        return

    config = json.load(open(WEATHER_CONFIG))
    api_key = config["simmer_api_key"]

    # Group all copies by whale wallet
    by_whale: dict[str, list[dict]] = defaultdict(list)
    with open(SPORTS_LOG) as f:
        for line in f:
            try:
                t = json.loads(line)
                wallet = t.get("whale_wallet")
                if wallet:
                    by_whale[wallet].append(t)
            except Exception:
                continue

    # Also include whales from scout files even if no copies yet — these get
    # the virtual backtest treatment so we never blindly send money to a new
    # whale without first checking how their last 30 trades would have done.
    name_by_wallet: dict[str, str] = {}
    tier_by_wallet: dict[str, str] = {}
    memory_dir = os.path.join(ROOT, "memory")
    if os.path.isdir(memory_dir):
        for fn in os.listdir(memory_dir):
            if not fn.startswith("whales_") or not fn.endswith(".json"):
                continue
            try:
                d = json.load(open(os.path.join(memory_dir, fn)))
                for w in d.get("whales", []):
                    wal = (w.get("wallet") or "").lower()
                    if not wal:
                        continue
                    name_by_wallet.setdefault(wal, w.get("name") or "?")
                    tier_by_wallet.setdefault(wal, w.get("tier") or "?")
                    if wal not in by_whale:
                        by_whale[wal] = []  # placeholder so loop visits it
            except Exception:
                continue

    print(f"=== Whale performance @ {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Tracked whales: {len(by_whale)} (incl. scout-only candidates)")

    # Caches shared across all whales for the run (sim_market_id and outcome
    # lookups are deterministic by condition / market, so they're worth
    # reusing — saves O(thousands) of HTTP calls when many whales touched the
    # same markets).
    resolution_cache: dict[str, bool | None] = {}
    cond_to_sim: dict[str, str | None] = {}
    perf: dict[str, dict] = {}

    for wallet, trades in by_whale.items():
        # Sort by ts and take most recent N
        trades.sort(key=lambda t: t.get("timestamp", 0))
        recent = trades[-WINDOW_LAST_N:]

        wins = 0
        losses = 0
        pnl = 0.0
        resolved_n = 0
        whale_name = (recent[-1].get("whale_name") if recent else None) or name_by_wallet.get(wallet, "?")
        whale_tier = (recent[-1].get("whale_tier") if recent else None) or tier_by_wallet.get(wallet, "?")

        for t in recent:
            mid = t.get("sim_market_id") or t.get("market_id")
            if not mid:
                continue
            if mid not in resolution_cache:
                resolution_cache[mid] = fetch_market_outcome(mid, api_key)
            outcome_yes = resolution_cache[mid]
            if outcome_yes is None:
                continue
            side = (t.get("side") or "").upper()
            won = (outcome_yes and side == "YES") or (not outcome_yes and side == "NO")
            entry = float(t.get("whale_price") or 0)
            amount = float(t.get("bet_amount") or 0)
            if entry <= 0 or entry >= 1 or amount <= 0:
                continue
            resolved_n += 1
            if won:
                wins += 1
                pnl += (1 - entry) * amount / entry
            else:
                losses += 1
                pnl -= amount

        wr = wins / resolved_n if resolved_n > 0 else 0.0

        # Virtual backtest only for genuinely unknown whales — those with no
        # real resolved copies. Whales with 1-9 resolved real copies stay in
        # 'trial' on real data alone (we already know they trade, just not
        # enough to judge). This keeps the run bounded — backtest is the
        # expensive part (Polymarket activity + Simmer resolution per trade).
        v_wr, v_pnl, v_n = (0.0, 0.0, 0)
        if resolved_n == 0:
            v_wr, v_pnl, v_n = virtual_backtest(
                wallet, api_key, cond_to_sim, resolution_cache,
            )

        # Classify
        if resolved_n >= MIN_N_FOR_CLASSIFICATION:
            if wr < BLOCK_WR_THRESHOLD and pnl < BLOCK_PNL_MAX:
                status = "blocked"
            elif wr > ELITE_WR_THRESHOLD and pnl > ELITE_PNL_MIN:
                status = "elite"
            else:
                status = "normal"
        elif v_n >= BACKTEST_MIN_N_RESOLVED:
            # Use backtest data as a proxy classification — never let a whale
            # with bad historical performance touch real money.
            if v_wr < BACKTEST_BLOCK_WR and v_pnl < BACKTEST_BLOCK_PNL:
                status = "blocked"
            elif v_wr > ELITE_WR_THRESHOLD and v_pnl > ELITE_PNL_MIN:
                status = "elite"
            else:
                status = "trial"
        else:
            status = "trial"

        perf[wallet] = {
            "wallet": wallet,
            "name": whale_name,
            "tier": whale_tier,
            "copies_recent_n": len(recent),
            "resolved_n": resolved_n,
            "wins": wins,
            "losses": losses,
            "wr": round(wr, 3),
            "pnl_sim": round(pnl, 2),
            "backtest_wr": round(v_wr, 3),
            "backtest_pnl_sim": round(v_pnl, 2),
            "backtest_n": v_n,
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Persist
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "params": {
            "min_n": MIN_N_FOR_CLASSIFICATION,
            "window_last_n": WINDOW_LAST_N,
            "block_wr_threshold": BLOCK_WR_THRESHOLD,
            "block_pnl_max": BLOCK_PNL_MAX,
            "elite_wr_threshold": ELITE_WR_THRESHOLD,
            "elite_pnl_min": ELITE_PNL_MIN,
            "backtest_block_wr": BACKTEST_BLOCK_WR,
            "backtest_block_pnl": BACKTEST_BLOCK_PNL,
            "backtest_min_n_resolved": BACKTEST_MIN_N_RESOLVED,
        },
        "by_wallet": perf,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    # Print summary
    statuses = defaultdict(list)
    for w, p in perf.items():
        statuses[p["status"]].append(p)

    print(f"\nClassification:")
    for st, items in statuses.items():
        print(f"  {st:<8}: {len(items)} whales")

    if statuses.get("elite"):
        print(f"\n=== ELITE WHALES (will get 2x bet) ===")
        for p in sorted(statuses["elite"], key=lambda x: -x["pnl_sim"])[:10]:
            print(f"  {p['wallet'][:8]}... {p['name']:<24} "
                  f"WR={p['wr']*100:.0f}%  N={p['resolved_n']}  PnL=${p['pnl_sim']:+.0f}  tier={p['tier']}")

    if statuses.get("blocked"):
        print(f"\n=== BLOCKED WHALES (skipped from now on) ===")
        for p in sorted(statuses["blocked"], key=lambda x: x["pnl_sim"])[:10]:
            print(f"  {p['wallet'][:8]}... {p['name']:<24} "
                  f"WR={p['wr']*100:.0f}%  N={p['resolved_n']}  PnL=${p['pnl_sim']:+.0f}  tier={p['tier']}")

    print(f"\nWrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
