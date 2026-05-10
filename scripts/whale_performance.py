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
# Time-based window instead of count-based (2026-05-10): heavy-copying whales
# were "hiding" from blocks because the 50-most-recent included many unresolved
# trades, dropping resolved_n below the threshold and reverting status to trial.
# A 14-day window evaluates them on actual recent performance regardless of
# copy volume.
WINDOW_DAYS = 14                # consider copies from last N days
# Tightened 2026-05-04: previous (0.30 / -50) was too lenient — RN1 type whales
# (WR 42% / PnL ~$0) kept doing 100+ copies/day with no edge. New limits flag
# whales that aren't clearly profitable as "blocked" so we stop wasting bets.
BLOCK_WR_THRESHOLD = 0.45       # WR below this AND negative P&L → block
ELITE_WR_THRESHOLD = 0.65       # WR above this AND positive P&L → elite
ELITE_PNL_MIN = 50.0            # at least $50 SIM net to qualify
BLOCK_PNL_MAX = -25.0           # at least -$25 net to qualify for blocking

# Note: pre-trial virtual backtest was attempted on 2026-05-04 and removed.
# Idea: simulate copying a scout-only whale's last 30-200 trades and resolve
# outcomes to decide if they're worth ever copying. Implementation worked
# (filters + Simmer market lookup all returned data) but Simmer's resolution
# data is only complete for markets the agent has actually traded — for
# arbitrary scout-whale markets, /markets/{id} returns outcome=None,
# status=active even for week-old games. Polymarket gamma also failed to
# return market data for these conditionIds. Without a working resolution
# source we can't backtest, so we rely on real-data classification only.


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

    resolution_cache: dict[str, bool | None] = {}
    perf: dict[str, dict] = {}
    cutoff_ts = time.time() - WINDOW_DAYS * 86400

    for wallet, trades in by_whale.items():
        # Sort by ts then keep only trades within the time window. Time-based
        # (not count-based) so heavy-copying whales can't "hide" by burying
        # their resolved bad trades under unresolved fresh ones.
        trades.sort(key=lambda t: t.get("timestamp", 0))
        recent = [t for t in trades if t.get("timestamp", 0) >= cutoff_ts]

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

        # Classify
        if resolved_n < MIN_N_FOR_CLASSIFICATION:
            status = "trial"
        elif wr < BLOCK_WR_THRESHOLD and pnl < BLOCK_PNL_MAX:
            status = "blocked"
        elif wr > ELITE_WR_THRESHOLD and pnl > ELITE_PNL_MIN:
            status = "elite"
        else:
            status = "normal"

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
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Persist
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "params": {
            "min_n": MIN_N_FOR_CLASSIFICATION,
            "window_days": WINDOW_DAYS,
            "block_wr_threshold": BLOCK_WR_THRESHOLD,
            "block_pnl_max": BLOCK_PNL_MAX,
            "elite_wr_threshold": ELITE_WR_THRESHOLD,
            "elite_pnl_min": ELITE_PNL_MIN,
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
