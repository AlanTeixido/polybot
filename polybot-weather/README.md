# Polybot Weather

Pure-Python weather-only trading bot for Simmer. No LLM. Lorenzo-style.

## Strategy

1. Fetch weather markets from Simmer every 3 min
2. Parse city/threshold/comparison from title
3. Get real forecast from Open-Meteo (free, global, no key)
4. Calculate P(event) with normal distribution (uncertainty grows with horizon)
5. Trade if `|P_real - market_price| > min_edge`
6. Size bet proportional to edge magnitude

## Setup on server

```bash
cd /root/polybot
git pull
cd polybot-weather
cp /root/polybot/config.json config.json   # reuse main bot config (has simmer_api_key)
# Edit config.json if you want different params (min_edge, max_bet)
screen -dmS weather-bot python3 bot.py
```

## Check status

```bash
screen -r weather-bot   # attach (Ctrl+A, D to detach)
tail -30 /root/polybot/polybot-weather/weather-bot.log
cat /root/polybot/polybot-weather/state.json
```

## Config params

- `min_edge`: Minimum edge required to trade (0.15 = 15 points)
- `max_bet`: Max bet size in $SIM
- `max_pct_balance`: Max % of balance per trade (0.02 = 2%)
- `cycle_interval_seconds`: Time between cycles (180s = 3 min)
- `max_trades_per_cycle`: How many trades to execute per cycle

## Why it might work

- Lorenzo-agent on Simmer leaderboard: 91.5% WR, $9K profit with this exact strategy
- NOAA/Open-Meteo forecasts are more accurate than market consensus 1-3 days out
- Markets consistently mispice temperature thresholds in non-US cities
- No LLM = zero API cost, deterministic, auditable
