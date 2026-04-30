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


def main() -> None:
    print(f"=== Whale scout @ {datetime.now(timezone.utc).isoformat()} ===")
    for cat in TARGET_CATEGORIES:
        rows = fetch_leaderboard(cat)
        whales = filter_whales(rows, cat)
        persist(cat, whales)
        append_history(cat, whales)
        # show top 5 inline
        for w in whales[:5]:
            print(f"  [{cat}] {w['wallet'][:8]}... {w['name']:<24} "
                  f"vol=${w['vol_week']:>10.0f}  pnl=${w['pnl_week']:>+8.0f}  "
                  f"roi={w['roi_pct_week']:>5.2f}%")
    consistency_report()


if __name__ == "__main__":
    main()
