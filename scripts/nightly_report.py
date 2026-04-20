"""Nightly performance report for both bots.

Reads calibration logs, fetches resolutions from Simmer, computes WR/Brier/P&L
across rolling windows (24h/72h/7d). Sends Telegram report. Applies go/no-go
criteria for the macro bot decision.

Usage (cron):
    0 23 * * * cd /root/polybot && /usr/bin/python3 scripts/nightly_report.py >> /root/polybot/nightly.log 2>&1
"""

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEATHER_LOG = os.path.join(ROOT, "memory", "calibration_log.jsonl")
CRYPTO_LOG = os.path.join(ROOT, "memory", "crypto_calibration_log.jsonl")
WEATHER_CONFIG = os.path.join(ROOT, "polybot-weather", "config.json")
LOCK_FILE = "/tmp/polybot_nightly_report.lock"
SIMMER_API = "https://api.simmer.markets/api/sdk"
GAMMA_API = "https://gamma-api.polymarket.com"

# Trades younger than this many seconds are considered pending (not yet resolvable)
PENDING_THRESHOLD_SECS = 48 * 3600
NOW = time.time()


def acquire_lock() -> bool:
    if os.path.exists(LOCK_FILE):
        # Stale lock if older than 23h
        try:
            age = time.time() - os.path.getmtime(LOCK_FILE)
            if age < 23 * 3600:
                print(f"Another instance running (lock {age:.0f}s old). Exiting.")
                return False
        except Exception:
            pass
    try:
        with open(LOCK_FILE, "w") as f:
            f.write(str(os.getpid()))
        return True
    except Exception as e:
        print(f"Lock create failed: {e}")
        return False


def release_lock() -> None:
    try:
        os.remove(LOCK_FILE)
    except Exception:
        pass


def load_config() -> dict:
    with open(WEATHER_CONFIG) as f:
        return json.load(f)


def load_log(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def filter_window(trades: list[dict], window_secs: float) -> list[dict]:
    cutoff = NOW - window_secs
    return [t for t in trades if t.get("timestamp", 0) >= cutoff]


def fetch_resolution_simmer(market_id: str, api_key: str) -> dict | None:
    """Returns {'resolved': bool, 'winner': 'YES'|'NO'} or None on error."""
    try:
        r = requests.get(
            f"{SIMMER_API}/markets/{market_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        status = (data.get("status") or "").lower()
        outcome = data.get("outcome") or data.get("resolution")
        if status in ("resolved", "closed", "settled") or outcome is not None:
            if isinstance(outcome, str):
                lo = outcome.lower()
                if lo in ("yes", "y", "true", "1"):
                    return {"resolved": True, "winner": "YES"}
                if lo in ("no", "n", "false", "0"):
                    return {"resolved": True, "winner": "NO"}
            elif isinstance(outcome, bool):
                return {"resolved": True, "winner": "YES" if outcome else "NO"}
        return {"resolved": False}
    except Exception:
        return None


def fetch_resolution_polymarket(market_id: str) -> dict | None:
    """Polymarket Gamma API — outcomePrices = ['1','0'] (YES won) or ['0','1'] (NO won)."""
    try:
        r = requests.get(f"{GAMMA_API}/markets/{market_id}", timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        closed = bool(data.get("closed", False))
        uma_status = (data.get("umaResolutionStatus") or "").lower()
        if not (closed or uma_status in ("resolved", "settled")):
            return {"resolved": False}
        outcome_prices = data.get("outcomePrices", "")
        prices = []
        if isinstance(outcome_prices, str) and outcome_prices:
            try:
                prices = [float(p) for p in json.loads(outcome_prices)]
            except (json.JSONDecodeError, ValueError):
                pass
        elif isinstance(outcome_prices, list):
            prices = [float(p) for p in outcome_prices]
        if len(prices) >= 2:
            # outcomes is typically ["Yes", "No"], so prices[0] = YES, prices[1] = NO
            if prices[0] >= 0.99:
                return {"resolved": True, "winner": "YES"}
            if prices[1] >= 0.99:
                return {"resolved": True, "winner": "NO"}
        return {"resolved": False}
    except Exception:
        return None


def fetch_resolution(market_id: str, venue: str, api_key: str) -> dict | None:
    """Route to correct API based on venue."""
    if venue == "polymarket":
        return fetch_resolution_polymarket(market_id)
    return fetch_resolution_simmer(market_id, api_key)


def resolve_trades(
    trades: list[dict], api_key: str
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split into resolved, pending_young (<48h), and pending_unresolved (>48h but API
    couldn't confirm resolution — either still open on the venue or API failure).
    Returning three buckets makes it visible when a >48h trade is silently stuck.
    """
    resolved, pending_young, pending_unresolved = [], [], []
    cache: dict[tuple[str, str], dict | None] = {}
    for t in trades:
        age = NOW - t.get("timestamp", NOW)
        if age < PENDING_THRESHOLD_SECS:
            pending_young.append(t)
            continue
        mid = t.get("market_id", "")
        venue = t.get("venue", "sim")
        key = (venue, mid)
        if key not in cache:
            cache[key] = fetch_resolution(mid, venue, api_key)
        res = cache[key]
        if res and res.get("resolved"):
            won = 1 if res["winner"] == t.get("side", "").upper() else 0
            resolved.append({**t, "actual_won": won})
        else:
            pending_unresolved.append(t)
    return resolved, pending_young, pending_unresolved


def brier(resolved: list[dict]) -> float:
    if not resolved:
        return float("nan")
    return sum((t["p_predicted"] - t["actual_won"]) ** 2 for t in resolved) / len(resolved)


def wr(resolved: list[dict]) -> float:
    if not resolved:
        return float("nan")
    return sum(t["actual_won"] for t in resolved) / len(resolved)


def avg_predicted(resolved: list[dict]) -> float:
    if not resolved:
        return float("nan")
    return sum(t["p_predicted"] for t in resolved) / len(resolved)


def pnl(resolved: list[dict]) -> float:
    """Realized P&L. Each share pays $1 on win, $0 on loss; cost = entry_price * shares.

    Fail-loud: skip trades without an entry price field rather than defaulting to 0.5
    which would silently corrupt P&L. Prints WARN so missing schema is noticed.
    """
    total = 0.0
    for t in resolved:
        entry = t.get("market_price_entry") or t.get("entry_price") or t.get("market_price")
        if entry is None:
            print(f"WARN: trade {t.get('market_id', '?')[:12]} has no entry-price field — skipped from P&L")
            continue
        amt = t.get("amount", 0)
        if entry <= 0 or entry >= 1:
            continue
        shares = amt / entry
        if t["actual_won"]:
            total += shares * (1 - entry)
        else:
            total -= amt
    return total


def fmt_metric(label: str, n: int, w: float, b: float, p: float, currency: str) -> str:
    if n == 0:
        return f"  {label}: no trades"
    flag = " ⚠️ N bajo" if n < 5 else ""
    wr_str = f"{w*100:.0f}%" if not _isnan(w) else "—"
    b_str = f"{b:.2f}" if not _isnan(b) else "—"
    sign = "+" if p >= 0 else ""
    return f"  {label}: {n}t WR {wr_str} Brier {b_str} P&L {sign}{p:.1f}{currency}{flag}"


def _isnan(x) -> bool:
    return x != x


def section_for_bot(
    name: str,
    log_path: str,
    group_field: str,
    currency: str,
    api_key: str,
    window_secs: float,
    source_filter: str | None = None,
) -> tuple[str, dict]:
    """Build report section + return summary dict for go/no-go logic.

    source_filter: only count trades whose 'source' field matches (e.g. 'weather-bot').
    Without this, the shared calibration_log.jsonl mixes legacy agent.py trades
    with current weather-bot trades.
    """
    all_trades = filter_window(load_log(log_path), window_secs)
    if source_filter:
        trades = [t for t in all_trades if t.get("source") == source_filter]
    else:
        trades = all_trades
    resolved, pending_young, pending_unresolved = resolve_trades(trades, api_key)

    n_total = len(trades)
    n_res = len(resolved)
    n_young = len(pending_young)
    n_stuck = len(pending_unresolved)

    overall_wr = wr(resolved)
    overall_brier = brier(resolved)
    overall_pnl = pnl(resolved)
    overall_avg_pred = avg_predicted(resolved)
    drift = abs(overall_avg_pred - overall_wr) if not _isnan(overall_wr) else float("nan")

    lines = [f"\n{name} — {int(window_secs / 3600)}h"]
    stuck_marker = f", ⚠️ {n_stuck} stuck >48h" if n_stuck > 0 else ""
    lines.append(
        f"  Total: {n_total}t  ({n_res} resueltos, {n_young} pendientes <48h{stuck_marker})"
    )
    if n_res > 0:
        sign = "+" if overall_pnl >= 0 else ""
        drift_str = f"{drift*100:.0f}%" if not _isnan(drift) else "—"
        lines.append(
            f"  Resueltos: WR {overall_wr*100:.0f}% Brier {overall_brier:.2f} "
            f"P&L {sign}{overall_pnl:.1f}{currency} | drift {drift_str}"
        )

    # Per-group breakdown
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in resolved:
        key = t.get(group_field, "unknown")
        groups[key].append(t)
    for key in sorted(groups.keys()):
        sub = groups[key]
        lines.append(fmt_metric(f"  {key}", len(sub), wr(sub), brier(sub), pnl(sub), currency))

    return "\n".join(lines), {
        "n_resolved": n_res,
        "n_pending": n_pend,
        "wr": overall_wr,
        "brier": overall_brier,
        "pnl": overall_pnl,
        "drift": drift,
    }


def go_no_go(weather_72: dict, crypto_72: dict) -> str:
    """Apply criteria. Returns formatted decision block."""
    checks = []
    weather_ok_pnl = (not _isnan(weather_72["pnl"])) and weather_72["pnl"] >= 0
    weather_ok_n = weather_72["n_resolved"] >= 8
    crypto_ok_wr = (not _isnan(crypto_72["wr"])) and crypto_72["wr"] >= 0.40
    crypto_ok_n = crypto_72["n_resolved"] >= 10
    drift_ok_w = _isnan(weather_72["drift"]) or weather_72["drift"] < 0.15
    drift_ok_c = _isnan(crypto_72["drift"]) or crypto_72["drift"] < 0.15

    checks.append(("Weather P&L ≥ 0", weather_ok_pnl))
    checks.append(("Weather N ≥ 8", weather_ok_n))
    checks.append(("Crypto WR ≥ 40%", crypto_ok_wr))
    checks.append(("Crypto N ≥ 10", crypto_ok_n))
    checks.append(("Weather drift < 15%", drift_ok_w))
    checks.append(("Crypto drift < 15%", drift_ok_c))

    all_ok = all(ok for _, ok in checks)
    out = ["\n🎯 DECISIÓN MACRO BOT:"]
    out.append("  ✅ GO" if all_ok else "  ⏸️  NO-GO (sigue estabilizando)")
    for label, ok in checks:
        out.append(f"     {'✓' if ok else '✗'} {label}")
    return "\n".join(out)


def send_telegram(msg: str, config: dict) -> None:
    token = config.get("telegram_bot_token", "")
    chat_id = config.get("telegram_chat_id", "")
    if not token or not chat_id:
        print("Telegram not configured. Printing instead:")
        print(msg)
        return
    # Telegram limit ~4096 chars. No parse_mode — `_` in field names like
    # 'above_or_equal' would break Markdown parsing and the entire message
    # would be silently rejected with HTTP 400.
    chunks = [msg[i:i + 4000] for i in range(0, len(msg), 4000)]
    for chunk in chunks:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": chunk},
                timeout=10,
            )
            if r.status_code != 200:
                print(f"Telegram returned {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"Telegram send failed: {e}")


# TODO: cross-arb scanner
# - enumerar pares de markets con títulos similares (embeddings o heurística temp+ciudad+día)
# - chequear monotonía: P(temp ≥ 80) ≤ P(temp ≥ 75)
# - alertar si violación > 3% (potencial arbitraje)
# Prioridad: después de validar macro bot
def cross_arb_scanner_stub() -> None:
    pass


def main() -> None:
    if not acquire_lock():
        sys.exit(0)

    try:
        config = load_config()
        api_key = config.get("simmer_api_key", "")
        if not api_key:
            print("ERROR: simmer_api_key not set")
            sys.exit(1)

        is_sunday = datetime.now(timezone.utc).weekday() == 6
        windows = [("24h", 24 * 3600), ("72h", 72 * 3600)]
        if is_sunday:
            windows.append(("7d", 7 * 24 * 3600))

        report_lines = [
            f"📊 Polybot Status — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        ]

        # Per-window sections.
        # Filter by source to exclude legacy agent.py trades from shared log.
        decision_data = {}
        for label, secs in windows:
            report_lines.append(f"\n━━━━━ Ventana {label} ━━━━━")
            w_section, w_summary = section_for_bot(
                "WEATHER (Polymarket)", WEATHER_LOG, "comparison", "$", api_key, secs,
                source_filter="weather-bot",
            )
            c_section, c_summary = section_for_bot(
                "CRYPTO (SIM)", CRYPTO_LOG, "asset", "S", api_key, secs,
                source_filter="crypto-bot",
            )
            report_lines.append(w_section)
            report_lines.append(c_section)
            if label == "72h":
                decision_data["weather"] = w_summary
                decision_data["crypto"] = c_summary

        # Go/no-go (always uses 72h window)
        if decision_data:
            report_lines.append(go_no_go(decision_data["weather"], decision_data["crypto"]))

        msg = "\n".join(report_lines)
        send_telegram(msg, config)
        print(msg)
    finally:
        release_lock()


if __name__ == "__main__":
    main()
