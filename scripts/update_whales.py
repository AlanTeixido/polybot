"""Auto-update whale wallets from Polymarket leaderboard. Run weekly: python3 scripts/update_whales.py"""

import json
import os
import shutil
import tempfile

import requests

CONFIG_PATH = os.environ.get("POLYBOT_CONFIG", "/root/polybot/config.json")
LEADERBOARD_URL = "https://data-api.polymarket.com/leaderboard"


def fetch_top_wallets(limit: int = 15) -> list[str]:
    """Fetch top wallets by monthly profit from Polymarket leaderboard."""
    resp = requests.get(
        LEADERBOARD_URL,
        params={"period": "monthly", "order_by": "profit_usdc", "limit": 20},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    if not isinstance(data, list):
        data = data.get("data", data.get("leaderboard", []))

    wallets = []
    for entry in data:
        addr = entry.get("address", entry.get("wallet", entry.get("user", "")))
        if addr and addr.startswith("0x") and len(addr) >= 42:
            wallets.append(addr.lower())
        if len(wallets) >= limit:
            break

    return wallets


def main() -> None:
    # Fetch new wallets
    print(f"Fetching top wallets from {LEADERBOARD_URL}...")
    try:
        new_wallets = fetch_top_wallets(15)
    except Exception as e:
        print(f"ERROR: Failed to fetch leaderboard: {e}")
        return

    if not new_wallets:
        print("ERROR: No wallets returned from leaderboard")
        return

    print(f"Found {len(new_wallets)} wallets")

    # Read current config
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Config not found at {CONFIG_PATH}")
        return

    old_wallets = set(w.lower() for w in config.get("whale_wallets", []))
    new_set = set(new_wallets)

    added = new_set - old_wallets
    removed = old_wallets - new_set

    # Update config
    config["whale_wallets"] = new_wallets

    # Atomic write
    dir_name = os.path.dirname(CONFIG_PATH)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        shutil.move(tmp_path, CONFIG_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    print(f"Updated whale_wallets: {len(new_wallets)} total")
    if added:
        print(f"  Added {len(added)}: {', '.join(list(added)[:3])}...")
    if removed:
        print(f"  Removed {len(removed)}: {', '.join(list(removed)[:3])}...")
    if not added and not removed:
        print("  No changes (same wallets)")


if __name__ == "__main__":
    main()
