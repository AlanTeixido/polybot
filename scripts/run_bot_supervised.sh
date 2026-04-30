#!/usr/bin/env bash
# Auto-restart supervisor for any polybot.
#
# Usage:
#   ./scripts/run_bot_supervised.sh weather
#   ./scripts/run_bot_supervised.sh sports
#
# Restarts the bot process if it crashes. Backs off after consecutive crashes
# to avoid tight loops (5s, 30s, 5min, 5min, ...). Logs supervisor events to
# ~/polybot/supervisor.log.

set -u

BOT="${1:-}"
case "$BOT" in
  weather) BOT_DIR="$HOME/polybot/polybot-weather" ;;
  sports)  BOT_DIR="$HOME/polybot/polybot-sports"  ;;
  *)
    echo "Usage: $0 {weather|sports}" >&2
    exit 2
    ;;
esac

SUP_LOG="$HOME/polybot/supervisor.log"
crash_count=0

log_sup() {
  echo "[$(date -u +'%Y-%m-%d %H:%M:%S UTC')] [supervisor:$BOT] $*" | tee -a "$SUP_LOG"
}

log_sup "Starting supervisor for $BOT bot (cwd=$BOT_DIR)"

while true; do
  cd "$BOT_DIR" || { log_sup "ERROR: cannot cd to $BOT_DIR"; exit 1; }
  log_sup "Launching python3 bot.py (crash #$crash_count so far)"
  python3 bot.py
  exit_code=$?
  crash_count=$((crash_count + 1))
  log_sup "bot.py exited with code $exit_code (crash count $crash_count)"

  # Backoff: 5s, 30s, then 5 min for repeated crashes
  if [ $crash_count -le 1 ]; then
    sleep_for=5
  elif [ $crash_count -le 3 ]; then
    sleep_for=30
  else
    sleep_for=300
  fi
  log_sup "Backing off ${sleep_for}s before restart"
  sleep "$sleep_for"
done
