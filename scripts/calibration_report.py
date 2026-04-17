"""Calibration + Brier score analysis for polybot trades.

Reads calibration_log.jsonl (trades with predicted probability) and cross-references
with Simmer API to determine actual outcomes. Outputs:
- Calibration curve (predicted vs actual WR by bucket)
- Brier score (vs climatology baseline of 0.25)
- Performance split by venue, source, comparison type

Usage:
    python scripts/calibration_report.py [--post-fix-date 2026-04-16]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CAL_LOG = os.path.join(ROOT, "memory", "calibration_log.jsonl")
CONFIG_PATH = os.path.join(ROOT, "config.json")

SIMMER_API = "https://api.simmer.markets/api/sdk"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_trades() -> list[dict]:
    """Read calibration log."""
    if not os.path.exists(CAL_LOG):
        print(f"No calibration log found at {CAL_LOG}")
        return []
    trades = []
    with open(CAL_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                trades.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return trades


def fetch_market_resolution(market_id: str, api_key: str) -> dict | None:
    """Get resolution status + outcome from Simmer."""
    try:
        resp = requests.get(
            f"{SIMMER_API}/markets/{market_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        status = (data.get("status") or "").lower()
        outcome = data.get("outcome") or data.get("resolution")
        if status in ("resolved", "closed", "settled") or outcome is not None:
            # Determine winning side
            if isinstance(outcome, str):
                outcome_lower = outcome.lower()
                if outcome_lower in ("yes", "y", "true", "1"):
                    return {"resolved": True, "winner": "YES"}
                elif outcome_lower in ("no", "n", "false", "0"):
                    return {"resolved": True, "winner": "NO"}
            elif isinstance(outcome, bool):
                return {"resolved": True, "winner": "YES" if outcome else "NO"}
        return {"resolved": False}
    except Exception as e:
        print(f"  Warning: failed to fetch {market_id[:8]}...: {e}")
        return None


def bucket_key(p: float) -> str:
    """Group probability into buckets: 0-10, 10-20, ..., 90-100."""
    b_lo = int(p * 10) * 10
    b_hi = b_lo + 10
    return f"{b_lo:02d}-{b_hi:02d}%"


def brier_score(predictions: list[tuple[float, int]]) -> float:
    """Brier score = mean((p - actual)^2). Lower = better. 0.25 = random baseline."""
    if not predictions:
        return float("nan")
    return sum((p - a) ** 2 for p, a in predictions) / len(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-fix-date", default="2026-04-16",
                        help="ISO date. Filter trades on/after this date.")
    parser.add_argument("--venue", default=None, help="Filter by venue: sim or polymarket")
    parser.add_argument("--source", default=None, help="Filter by source: agent or weather-bot")
    args = parser.parse_args()

    config = load_config()
    api_key = config.get("simmer_api_key", "")
    if not api_key:
        print("ERROR: simmer_api_key not set in config.json")
        sys.exit(1)

    trades = load_trades()
    if not trades:
        print("No trades logged yet. Run the bots and come back.")
        return

    # Filter
    cutoff_ts = datetime.fromisoformat(args.post_fix_date).timestamp()
    filtered = []
    for t in trades:
        if t.get("timestamp", 0) < cutoff_ts:
            continue
        if args.venue and t.get("venue") != args.venue:
            continue
        if args.source and t.get("source") != args.source:
            continue
        if t.get("p_predicted") is None:
            continue
        filtered.append(t)

    print(f"\n{'='*70}")
    print(f"CALIBRATION REPORT — {len(filtered)} trades (post-fix: {args.post_fix_date})")
    if args.venue:
        print(f"Venue filter: {args.venue}")
    if args.source:
        print(f"Source filter: {args.source}")
    print(f"{'='*70}\n")

    if not filtered:
        print("No trades match filters.")
        return

    # Fetch resolutions
    print("Fetching resolutions from Simmer API...")
    resolved_trades = []
    unresolved = 0
    for i, t in enumerate(filtered):
        if i > 0 and i % 20 == 0:
            print(f"  {i}/{len(filtered)}...")
        res = fetch_market_resolution(t["market_id"], api_key)
        if res and res.get("resolved"):
            actual_won = 1 if res["winner"] == t["side"].upper() else 0
            resolved_trades.append({**t, "actual_won": actual_won, "winner": res["winner"]})
        else:
            unresolved += 1

    print(f"\nResolved: {len(resolved_trades)} | Unresolved: {unresolved}\n")

    if not resolved_trades:
        print("No resolved trades yet. Come back in 1-2 days.")
        return

    # --- Brier score ---
    predictions = [(t["p_predicted"], t["actual_won"]) for t in resolved_trades]
    brier = brier_score(predictions)
    climatology = 0.25  # baseline: always predict 0.5
    avg_p = sum(p for p, _ in predictions) / len(predictions)
    avg_won = sum(a for _, a in predictions) / len(predictions)

    print(f"{'─'*70}")
    print(f"BRIER SCORE: {brier:.4f}  (lower = better, 0.25 = random)")
    print(f"  Beats climatology? {'YES ✓' if brier < climatology else 'NO ✗'}")
    print(f"  Avg predicted P:   {avg_p:.3f}")
    print(f"  Avg actual WR:     {avg_won:.3f}")
    print(f"  Overall bias:      {(avg_p - avg_won)*100:+.1f}pts  "
          f"({'overconfident' if avg_p > avg_won else 'underconfident'})")

    # --- Calibration curve ---
    print(f"\n{'─'*70}")
    print("CALIBRATION CURVE (predicted probability → actual win rate):\n")
    print(f"  {'Bucket':<12} {'N':>4}  {'Predicted':>10}  {'Actual':>8}  {'Error':>8}  Diagnosis")
    print(f"  {'─'*12} {'─'*4}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*20}")

    buckets: dict[str, list] = defaultdict(list)
    for t in resolved_trades:
        buckets[bucket_key(t["p_predicted"])].append((t["p_predicted"], t["actual_won"]))

    for bkt in sorted(buckets.keys()):
        preds = buckets[bkt]
        n = len(preds)
        avg_pred = sum(p for p, _ in preds) / n
        actual_wr = sum(a for _, a in preds) / n
        error = actual_wr - avg_pred
        if abs(error) < 0.05:
            diag = "well calibrated"
        elif error < -0.1:
            diag = "overconfident ✗"
        elif error > 0.1:
            diag = "underconfident"
        else:
            diag = "slight miscalib"
        print(f"  {bkt:<12} {n:>4}  {avg_pred*100:>8.1f}%   {actual_wr*100:>6.1f}%   "
              f"{error*100:>+6.1f}%  {diag}")

    # --- By comparison type ---
    print(f"\n{'─'*70}")
    print("BY COMPARISON TYPE:\n")
    comp_stats: dict[str, list] = defaultdict(list)
    for t in resolved_trades:
        comp = t.get("comparison", "unknown")
        comp_stats[comp].append((t["p_predicted"], t["actual_won"]))
    for comp, preds in sorted(comp_stats.items()):
        n = len(preds)
        wr = sum(a for _, a in preds) / n
        b = brier_score(preds)
        print(f"  {comp:<20} N={n:>3}  WR={wr*100:>5.1f}%  Brier={b:.3f}")

    # --- P&L realized (if possible) ---
    print(f"\n{'─'*70}")
    print("P&L REALIZED (post-fix):\n")
    total_pnl = 0
    wins_count = 0
    losses_count = 0
    for t in resolved_trades:
        entry = t.get("market_price_entry", 0.5)
        amt = t.get("amount", 0)
        if entry <= 0 or entry >= 1:
            continue
        shares = amt / entry
        if t["actual_won"]:
            pnl = shares * (1 - entry)  # each share worth $1, paid entry
            wins_count += 1
        else:
            pnl = -amt
            losses_count += 1
        total_pnl += pnl

    print(f"  Wins: {wins_count}   Losses: {losses_count}   "
          f"WR: {wins_count / (wins_count + losses_count) * 100:.1f}%")
    print(f"  Total P&L: {total_pnl:+.2f}")
    print(f"  Avg per trade: {total_pnl / len(resolved_trades):+.3f}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
