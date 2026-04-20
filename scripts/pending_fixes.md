# Pending fixes (2026-04-20 PM)

Pre-approval review document. **Do NOT apply** any fix here without explicit user OK.

## DIAG-1 RESULT (2026-04-20): NO BUG. FIX-1 NOT NEEDED.

Diagnostic ran. Found 2 trades in last 48h with entry > 0.55:
- Chengdu 30°C NO @ 0.5545 (age 43h, entered ~2026-04-19 17:00 UTC)
- Miami 74-75°F NO @ 0.64 (age 42h, entered ~2026-04-19 18:00 UTC)

Both entered when max_entry was still 0.75/0.65 (before today's tightening
to 0.55 at 11:23 UTC). They are NOT gate bypasses — they are trades
executed under valid prior rules.

The 7 dashboard positions at 98-99¢ are older legacy from before the
get_market_detail bug fix. Calibration log shows them with
market_price_entry=0.5 (what the bot believed) while the SDK actually
executed at the real 99¢ price. **DIAG-2 below addresses this**.

---

## DIAG-1: Verify whether max_entry=0.55 gate is actually being bypassed

### Run on VPS first

```bash
cd ~/polybot && python3 -c "
import json, time
NOW = time.time()
violations = []
with open('memory/calibration_log.jsonl') as f:
    for line in f:
        t = json.loads(line)
        if t.get('source') != 'weather-bot': continue
        if NOW - t.get('timestamp', 0) > 48*3600: continue
        entry = t.get('market_price_entry', 0)
        if entry > 0.55:
            violations.append(t)
print(f'Trades weather-bot últimas 48h con entry>0.55: {len(violations)}')
for v in violations:
    age_h = (NOW - v['timestamp']) / 3600
    print(f\"  entry={v['market_price_entry']} side={v['side']} age={age_h:.0f}h p_pred={v.get('p_predicted')} title={v.get('title','?')[:50]}\")
"
```

### Two possible outcomes

**Outcome A (most likely): violations = 0**
- The 7 dashboard positions at 98-99¢ are LEGACY trades from before the
  get_market_detail fix on 2026-04-19. Calibration log shows them with
  `market_price_entry=0.5` (the broken default), even though the actual
  on-chain entry was 99¢.
- **No code fix needed.** The bug is already fixed for new trades.
- **But** P&L computation in nightly_report.py is corrupt for legacy trades
  because it uses the logged 0.5 entry → see DIAG-2 below.

**Outcome B (concerning): violations > 0**
- The gate is still being bypassed somehow. Apply FIX-1 below.

---

## FIX-1: Block fallback to current_price=0.5 default

**Apply ONLY if Outcome B above.**

### File: `polybot-weather/bot.py`, line 473

```python
# CURRENT (line 473):
current_price = float(market.get("current_probability", market.get("current_price", 0.5)))

# PROPOSED:
_raw_price = market.get("current_probability", market.get("current_price"))
if _raw_price is None:
    if verbose:
        logger.info(f"  SKIP (no price field, refusing to default to 0.5): {market.get('question', '')[:60]}")
    return None
current_price = float(_raw_price)
```

### Why this fixes it

Defaulting to 0.5 when the API returns no price field made the bot compute
`entry_price = 1 - 0.5 = 0.5` for NO trades, passing the max_entry gate
trivially. By refusing to trade when price is missing (fail-loud), we
guarantee the gate sees the real price.

### Side effects

- Markets without `current_probability` field will now be skipped
  (today they are evaluated with bogus price 0.5).
- Expected reduction: ~0-5% of evaluated markets, mostly newly-listed
  ones without trade history.
- No risk of false rejection on real markets — Simmer always returns
  `current_probability` for active markets.

### Restart commands after applying fix

```bash
pkill -9 -f "polybot-weather/bot.py"
screen -S weather-bot -X quit
screen -wipe
cd ~/polybot && git pull
screen -dmS weather-bot bash -c "cd ~/polybot/polybot-weather && python3 bot.py 2>&1 | tee -a ~/polybot/weather.log"
screen -ls
```

---

## DIAG-2: P&L corruption from legacy entry=0.5 trades

### Issue

`scripts/nightly_report.py:pnl()` computes `shares = amt / entry`. When
`entry = 0.5` (logged value) but real entry was 0.99, computed shares
are 2x what was actually bought, and the win payoff `shares * (1-entry) = shares * 0.5`
is wildly inflated relative to the real `shares * 0.01`.

### Confirmation command

```bash
cd ~/polybot && python3 -c "
import json
with open('memory/calibration_log.jsonl') as f:
    bot_trades = [json.loads(l) for l in f if 'weather-bot' in l]
n_with_05 = sum(1 for t in bot_trades if t.get('market_price_entry') == 0.5)
print(f'Weather-bot trades total: {len(bot_trades)}')
print(f'  With market_price_entry=0.5 (legacy bug): {n_with_05}')
print(f'  With real entry: {len(bot_trades) - n_with_05}')
"
```

### Proposed fix (NOT urgent)

Add to `pnl()` in `scripts/nightly_report.py`:

```python
# Skip trades with suspicious entry=0.5 exactly — almost always legacy
# get_market_detail bug, not a real coin-flip price.
if entry == 0.5:
    continue
```

### Why not urgent

- Pre/post-fix split in nightly report (FIX_TIMESTAMP_WEATHER) already
  isolates these trades to the pre-fix bucket.
- Pre-fix bucket is for diagnostic visibility, not go/no-go decision.
- Post-fix trades will have real entry prices and clean P&L.
- Safe to leave as-is until Wed-Thu review when we re-evaluate the script.

---

## Decision matrix for tomorrow morning

1. Run DIAG-1 command. Read output.
2. If Outcome A (0 violations) → no fix needed. Done.
3. If Outcome B (violations > 0) → apply FIX-1, restart bot, monitor.
4. Run DIAG-2 command separately. If many legacy entry=0.5 trades are
   inflating Pre-fix P&L, optionally apply DIAG-2 fix to nightly report.
   Otherwise leave for Wed.

**No new work until Wed-Thu** beyond these two diagnostics.
