"""Seed initial knowledge base for Polybot. Run once: python3 scripts/seed_knowledge.py"""

import json
import os
import sys
import time

MEMORY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory")
KNOWLEDGE_FILE = os.path.join(MEMORY_DIR, "knowledge.json")

SEED_ENTRIES = [
    {
        "insight": "Near-resolution markets (92%+ probability, <7 days) are the most reliable strategy. Low risk, consistent returns.",
        "tags": ["near-resolution", "strategy", "high-confidence"],
    },
    {
        "insight": "Never trade crypto exact-price markets (BTC > $X). Impossible to predict short-term price targets.",
        "tags": ["avoid", "crypto", "exact-price"],
    },
    {
        "insight": "When 2+ whale wallets agree on the same side of a market, it's a strong signal. Follow smart money.",
        "tags": ["whale", "copytrading", "signal"],
    },
    {
        "insight": "Esports and gaming markets have low liquidity and unpredictable outcomes. Avoid.",
        "tags": ["avoid", "esports"],
    },
    {
        "insight": "Markets with <500 USDC volume have liquidity risk. Spread can eat all profits.",
        "tags": ["avoid", "low-volume"],
    },
    {
        "insight": "Political markets resolving within 7 days are good targets when there's polling data or news confirming the outcome.",
        "tags": ["political", "strategy", "near-resolution"],
    },
    {
        "insight": "Sports championship markets near resolution are reliable when one team has already qualified.",
        "tags": ["sports", "strategy", "near-resolution"],
    },
]


def main() -> None:
    os.makedirs(MEMORY_DIR, exist_ok=True)

    existing: list[dict] = []
    try:
        with open(KNOWLEDGE_FILE, "r") as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    existing_insights = {e.get("insight", "") for e in existing}
    added = 0

    for entry in SEED_ENTRIES:
        if entry["insight"] in existing_insights:
            print(f"SKIP (duplicate): {entry['insight'][:60]}...")
            continue

        existing.append({
            "insight": entry["insight"],
            "tags": entry["tags"],
            "source": "seed",
            "timestamp": str(time.time()),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        added += 1
        print(f"ADDED: {entry['insight'][:60]}...")

    with open(KNOWLEDGE_FILE, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\nDone. Added {added} entries, {len(existing)} total in knowledge base.")


if __name__ == "__main__":
    main()
