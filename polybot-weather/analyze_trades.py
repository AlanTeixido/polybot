"""Analyze weather bot's resolved trades by checking Simmer API for each market ID.

Reads state.json to get the market IDs traded, then queries Simmer to see which
have resolved and calculates real win rate + PnL for the weather bot specifically.
"""

import json
import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(HERE, "state.json")
CONFIG_PATH = os.path.join(HERE, "config.json")

SIMMER_API = "https://api.simmer.markets/api/sdk"


def main():
    with open(CONFIG_PATH) as f:
        api_key = json.load(f)["simmer_api_key"]
    with open(STATE_PATH) as f:
        state = json.load(f)

    traded_ids = state.get("traded_markets", [])
    print(f"Weather bot traded {len(traded_ids)} markets total")
    print("Fetching positions to get actual outcomes...\n")

    # Get all positions (active + resolved)
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(
        f"{SIMMER_API}/positions",
        headers=headers,
        params={"limit": 500, "include_resolved": "true"},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    positions = data if isinstance(data, list) else data.get("positions", data.get("data", []))

    print(f"Total positions in account: {len(positions)}")

    # Match positions to traded_ids
    traded_set = set(traded_ids)
    weather_positions = []
    for p in positions:
        market_id = p.get("market_id") or p.get("market") or p.get("id")
        if market_id in traded_set:
            weather_positions.append(p)

    print(f"Matched {len(weather_positions)} positions from weather bot\n")

    resolved = [p for p in weather_positions if p.get("resolved") or p.get("status") == "resolved"]
    active = [p for p in weather_positions if not (p.get("resolved") or p.get("status") == "resolved")]

    print(f"Resolved: {len(resolved)}")
    print(f"Active:   {len(active)}\n")

    if resolved:
        wins = []
        losses = []
        total_pnl = 0
        for p in resolved:
            pnl = float(p.get("realized_pnl") or p.get("pnl") or 0)
            total_pnl += pnl
            if pnl > 0:
                wins.append(p)
            else:
                losses.append(p)

        wr = len(wins) / len(resolved) * 100 if resolved else 0
        print(f"WINS:   {len(wins)}")
        print(f"LOSSES: {len(losses)}")
        print(f"WR:     {wr:.1f}%")
        print(f"PnL:    {total_pnl:+.2f} $SIM\n")

        print("Top 5 wins:")
        for p in sorted(wins, key=lambda x: float(x.get("realized_pnl") or 0), reverse=True)[:5]:
            print(f"  +{float(p.get('realized_pnl', 0)):.2f}  {p.get('title', p.get('question', '?'))[:70]}")
        print("\nTop 5 losses:")
        for p in sorted(losses, key=lambda x: float(x.get("realized_pnl") or 0))[:5]:
            print(f"  {float(p.get('realized_pnl', 0)):.2f}  {p.get('title', p.get('question', '?'))[:70]}")
    else:
        print("No resolved positions found yet.")
        if active:
            print("\nFirst 5 active positions:")
            for p in active[:5]:
                print(f"  {p.get('title', p.get('question', '?'))[:70]}")


if __name__ == "__main__":
    main()
