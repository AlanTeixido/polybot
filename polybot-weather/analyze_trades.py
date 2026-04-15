"""Analyze weather bot's trades by fetching /trades from Simmer and matching outcomes.

Strategy:
1. Fetch ALL trades for the agent (paginated)
2. Filter to source='sdk:weather-bot'
3. For each market_id, fetch market status (resolved/active + outcome)
4. Calculate realized PnL for resolved trades only
"""

import json
import os
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
SIMMER_API = "https://api.simmer.markets/api/sdk"


def fetch_all_trades(api_key: str, limit: int = 200) -> list:
    """Fetch all trades paginated."""
    headers = {"Authorization": f"Bearer {api_key}"}
    all_trades = []
    offset = 0
    while True:
        resp = requests.get(
            f"{SIMMER_API}/trades",
            headers=headers,
            params={"limit": limit, "offset": offset},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        trades = data.get("trades", data if isinstance(data, list) else [])
        if not trades:
            break
        all_trades.extend(trades)
        if len(trades) < limit:
            break
        offset += limit
        if offset > 2000:  # safety limit
            break
    return all_trades


def fetch_market_status(api_key: str, market_id: str) -> dict | None:
    """Check if a market is resolved and get the outcome."""
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(
            f"{SIMMER_API}/markets/{market_id}",
            headers=headers,
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def main():
    with open(CONFIG_PATH) as f:
        api_key = json.load(f)["simmer_api_key"]

    print("Fetching all trades...")
    all_trades = fetch_all_trades(api_key)
    print(f"Total trades in account: {len(all_trades)}")

    # Filter to weather bot
    weather_trades = [t for t in all_trades if t.get("source") == "sdk:weather-bot"]
    print(f"Weather bot trades: {len(weather_trades)}\n")

    if not weather_trades:
        print("No weather bot trades found.")
        return

    # Group by market_id (since bot might stack positions)
    by_market: dict[str, list] = {}
    for t in weather_trades:
        mid = t.get("market_id", "")
        by_market.setdefault(mid, []).append(t)

    print(f"Unique markets: {len(by_market)}")
    print("Checking market statuses (sampling first 30)...\n")

    resolved_wins = []
    resolved_losses = []
    active_count = 0
    error_count = 0

    market_ids = list(by_market.keys())
    for i, mid in enumerate(market_ids[:50]):  # sample first 50 to avoid rate limits
        status = fetch_market_status(api_key, mid)
        if not status:
            error_count += 1
            continue

        is_resolved = status.get("status") == "resolved" or status.get("resolved")
        outcome = status.get("outcome")  # "yes" or "no"

        if not is_resolved:
            active_count += 1
            continue

        # Calculate PnL for all trades on this market
        for t in by_market[mid]:
            cost = float(t.get("cost", 0))
            shares = float(t.get("shares", 0))
            side = (t.get("side") or "").lower()

            # If outcome matches side, shares are worth $1 each. Otherwise $0.
            if outcome and outcome.lower() == side:
                payout = shares
                pnl = payout - cost
                resolved_wins.append({
                    "market": status.get("question", "?"),
                    "side": side,
                    "cost": cost,
                    "payout": payout,
                    "pnl": pnl,
                })
            else:
                pnl = -cost
                resolved_losses.append({
                    "market": status.get("question", "?"),
                    "side": side,
                    "cost": cost,
                    "pnl": pnl,
                })

        time.sleep(0.1)  # rate limit

    total_resolved = len(resolved_wins) + len(resolved_losses)
    print(f"Sampled {len(market_ids[:50])} markets:")
    print(f"  Active:   {active_count}")
    print(f"  Resolved: {total_resolved} trades")
    print(f"  Errors:   {error_count}")
    print()

    if total_resolved > 0:
        wr = len(resolved_wins) / total_resolved * 100
        total_pnl = sum(t["pnl"] for t in resolved_wins) + sum(t["pnl"] for t in resolved_losses)
        total_cost = sum(t["cost"] for t in resolved_wins) + sum(t["cost"] for t in resolved_losses)
        print(f"WINS:   {len(resolved_wins)}")
        print(f"LOSSES: {len(resolved_losses)}")
        print(f"WR:     {wr:.1f}%")
        print(f"Cost:   {total_cost:.2f} $SIM")
        print(f"PnL:    {total_pnl:+.2f} $SIM ({total_pnl/total_cost*100:+.1f}% on cost)")
        print()
        if resolved_wins:
            print("Top 5 wins:")
            for t in sorted(resolved_wins, key=lambda x: x["pnl"], reverse=True)[:5]:
                print(f"  +{t['pnl']:>6.2f}  {t['side'].upper():3s} {t['market'][:65]}")
        if resolved_losses:
            print("\nTop 5 losses:")
            for t in sorted(resolved_losses, key=lambda x: x["pnl"])[:5]:
                print(f"  {t['pnl']:>6.2f}  {t['side'].upper():3s} {t['market'][:65]}")
    else:
        print("No resolved markets yet in the sample. Weather bot trades are still active.")


if __name__ == "__main__":
    main()
