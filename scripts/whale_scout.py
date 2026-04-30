"""Daily whale identification using Polymarket leaderboard API.

Identifies profitable traders per category, filters out market makers and
suspected bots, persists top N candidates to memory/whales_<category>.json.

Run via cron daily at 00:30 UTC:
    30 0 * * * cd /root/polybot && /usr/bin/python3 scripts/whale_scout.py >> /root/polybot/whale_scout.log 2>&1
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MEMORY_DIR = os.path.join(ROOT, "memory")
LB_URL = "https://data-api.polymarket.com/v1/leaderboard"

# Polymarket categories we care about for copy-trading
TARGET_CATEGORIES = ["SPORTS", "POLITICS", "CRYPTO", "WEATHER"]

# Whale eligibility filters (per category, weekly)
MIN_PNL = 1500.0           # at least $1.5k profit/week (signal vs noise)
MIN_VOLUME = 20_000.0      # at least $20k traded (real participation)
MAX_VOLUME = 5_000_000.0   # below $5M (above is market makers — they break even)
MIN_ROI_PCT = 0.5          # pnl/volume >= 0.5% (eliminates pure MMs)


def fetch_leaderboard(category: str, time_period: str = "WEEK", limit: int = 50) -> list[dict]:
    try:
        r = requests.get(
            LB_URL,
            params={
                "category": category,
                "timePeriod": time_period,
                "orderBy": "PNL",
                "limit": limit,
            },
            timeout=15,
        )
        if r.status_code != 200:
            print(f"[{category}] HTTP {r.status_code}: {r.text[:200]}")
            return []
        data = r.json()
        return data if isinstance(data, list) else data.get("data", [])
    except Exception as e:
        print(f"[{category}] Error: {e}")
        return []


def filter_whales(rows: list[dict], category: str) -> list[dict]:
    out = []
    for row in rows:
        try:
            wallet = row.get("proxyWallet") or row.get("wallet") or ""
            name = row.get("userName") or row.get("name") or ""
            vol = float(row.get("vol") or row.get("volume") or 0)
            pnl = float(row.get("pnl") or 0)
            verified = bool(row.get("verifiedBadge"))

            if not wallet or len(wallet) != 42:
                continue
            if pnl < MIN_PNL:
                continue
            if vol < MIN_VOLUME or vol > MAX_VOLUME:
                continue
            roi_pct = (pnl / vol * 100) if vol > 0 else 0
            if roi_pct < MIN_ROI_PCT:
                continue

            out.append({
                "wallet": wallet.lower(),
                "name": name,
                "vol_week": round(vol, 2),
                "pnl_week": round(pnl, 2),
                "roi_pct_week": round(roi_pct, 2),
                "verified": verified,
                "category": category,
            })
        except Exception as e:
            print(f"  filter error on row: {e}")
            continue
    out.sort(key=lambda w: w["roi_pct_week"], reverse=True)
    return out


def persist(category: str, whales: list[dict]) -> None:
    os.makedirs(MEMORY_DIR, exist_ok=True)
    path = os.path.join(MEMORY_DIR, f"whales_{category.lower()}.json")
    payload = {
        "category": category,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(whales),
        "whales": whales,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{category}] Saved {len(whales)} whales → {path}")


def append_history(category: str, whales: list[dict]) -> None:
    """Append today's roster to a history JSONL so we can spot consistent winners."""
    os.makedirs(MEMORY_DIR, exist_ok=True)
    path = os.path.join(MEMORY_DIR, "whale_scout_history.jsonl")
    ts = int(time.time())
    with open(path, "a") as f:
        for w in whales:
            f.write(json.dumps({"ts": ts, **w}) + "\n")


def consistency_report() -> None:
    """Show wallets that have appeared on multiple daily snapshots — those are
    the truly stable winners worth copying."""
    path = os.path.join(MEMORY_DIR, "whale_scout_history.jsonl")
    if not os.path.exists(path):
        return
    counts: dict[tuple[str, str], int] = {}
    last_pnl: dict[tuple[str, str], float] = {}
    with open(path) as f:
        for line in f:
            try:
                e = json.loads(line)
                key = (e["category"], e["wallet"])
                counts[key] = counts.get(key, 0) + 1
                last_pnl[key] = e["pnl_week"]
            except Exception:
                continue
    print("\n=== CONSISTENT WHALES (appeared 3+ days in a row) ===")
    for (cat, wallet), n in sorted(counts.items(), key=lambda kv: -kv[1])[:30]:
        if n < 3:
            continue
        pnl = last_pnl.get((cat, wallet), 0)
        print(f"  [{cat:<10}] {wallet[:8]}... appeared {n}d | last pnl_week: ${pnl:.0f}")


def fetch_alltime_legends(category: str) -> list[dict]:
    """All-time top by PNL — these are the proven track-record traders.
    Different filters: profit-focused, not ROI-focused. A whale with $50M
    volume and $5M profit is a legend even if ROI is "only" 10%.
    """
    rows = fetch_leaderboard(category, time_period="ALL", limit=50)
    out = []
    for row in rows:
        try:
            wallet = (row.get("proxyWallet") or row.get("wallet") or "").lower()
            name = row.get("userName") or row.get("name") or ""
            vol = float(row.get("vol") or row.get("volume") or 0)
            pnl = float(row.get("pnl") or 0)
            verified = bool(row.get("verifiedBadge"))

            if not wallet or len(wallet) != 42:
                continue
            # All-time legend = absolute profit threshold
            if pnl < 50_000:        # at least $50k all-time profit
                continue
            if vol < 100_000:       # real participation
                continue

            out.append({
                "wallet": wallet,
                "name": name,
                "vol_alltime": round(vol, 2),
                "pnl_alltime": round(pnl, 2),
                "roi_pct_alltime": round((pnl / vol * 100) if vol > 0 else 0, 2),
                "verified": verified,
                "category": category,
                "tier": "alltime_legend",
            })
        except Exception:
            continue
    out.sort(key=lambda w: w["pnl_alltime"], reverse=True)
    return out


def merge_weekly_and_alltime(weekly: list[dict], alltime: list[dict]) -> list[dict]:
    """Combine the two lists, dedup by wallet. Mark each whale's tiers.
    A wallet appearing in BOTH is the strongest signal — proven long-term
    AND currently active.
    """
    by_wallet: dict[str, dict] = {}
    for w in weekly:
        by_wallet[w["wallet"]] = {**w, "tier": "weekly_hot"}
    for w in alltime:
        if w["wallet"] in by_wallet:
            # Merge — both tiers
            by_wallet[w["wallet"]]["tier"] = "both_alltime+weekly"
            by_wallet[w["wallet"]]["pnl_alltime"] = w["pnl_alltime"]
            by_wallet[w["wallet"]]["vol_alltime"] = w["vol_alltime"]
            by_wallet[w["wallet"]]["roi_pct_alltime"] = w["roi_pct_alltime"]
        else:
            by_wallet[w["wallet"]] = w
    # Order: both > alltime > weekly
    rank = {"both_alltime+weekly": 0, "alltime_legend": 1, "weekly_hot": 2}
    return sorted(by_wallet.values(), key=lambda w: (rank.get(w["tier"], 9),
                                                     -w.get("pnl_alltime", w.get("pnl_week", 0))))


def main() -> None:
    print(f"=== Whale scout @ {datetime.now(timezone.utc).isoformat()} ===")
    for cat in TARGET_CATEGORIES:
        # 1) Weekly active winners
        rows_week = fetch_leaderboard(cat, time_period="WEEK", limit=50)
        weekly = filter_whales(rows_week, cat)

        # 2) All-time legends by total profit
        alltime = fetch_alltime_legends(cat)

        # 3) Merge: combined whale roster with tier annotation
        combined = merge_weekly_and_alltime(weekly, alltime)

        persist(cat, combined)
        append_history(cat, weekly)  # history tracks weekly snapshots only

        # Show top 5 with tier
        print(f"\n[{cat}] {len(combined)} total whales "
              f"({len(weekly)} weekly + {len(alltime)} alltime, "
              f"{sum(1 for w in combined if w['tier'] == 'both_alltime+weekly')} in both):")
        for w in combined[:8]:
            tier = w["tier"][:18]
            name = (w.get("name") or "?")[:20]
            pnl_a = w.get("pnl_alltime", 0)
            pnl_w = w.get("pnl_week", 0)
            print(f"  [{tier:<18}] {w['wallet'][:8]}... {name:<22} "
                  f"alltime=${pnl_a:>+10.0f}  week=${pnl_w:>+8.0f}")

    consistency_report()


if __name__ == "__main__":
    main()
