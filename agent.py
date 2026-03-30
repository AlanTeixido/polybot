"""Polybot - Autonomous trading agent for Polymarket.

Multi-turn LLM agent loop: reason -> call tools -> reason -> execute trades.
Runs 24/7, learns from every trade.
"""

import json
import logging
import os
import signal
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Any

import anthropic

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "polybot.log")

logger = logging.getLogger("polybot")
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(console_handler)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Telegram notifications
# ---------------------------------------------------------------------------
def send_telegram(message: str, config: dict) -> None:
    token = config.get("telegram_bot_token", "")
    chat_id = config.get("telegram_chat_id", "")
    if not token or not chat_id:
        return
    try:
        import requests
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=10)
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


# ---------------------------------------------------------------------------
# Tool definitions for Claude
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "get_balance",
        "description": "Get current USDC.e balance on Polygon wallet. Returns balance_usdc and alerts if low.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_positions",
        "description": "Get all open positions with unrealized P&L. Flags positions in significant loss.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_markets",
        "description": "Scan available Polymarket markets. Auto-filters crypto exact-price, esports, >30 day markets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "min_volume": {"type": "number", "description": "Minimum volume in USDC (default 500)"},
                "min_probability": {"type": "number", "description": "Min YES probability % (0-100)"},
                "max_probability": {"type": "number", "description": "Max YES probability % (0-100)"},
                "limit": {"type": "integer", "description": "Max markets to return (default 50)"},
                "keyword": {"type": "string", "description": "Filter by keyword/tag"},
                "category": {"type": "string", "description": "Filter by category"},
            },
            "required": [],
        },
    },
    {
        "name": "get_market_detail",
        "description": "Get detailed info for a specific market: prices, volume, liquidity, token IDs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "market_id": {"type": "string", "description": "The market ID"},
            },
            "required": ["market_id"],
        },
    },
    {
        "name": "analyze_market",
        "description": "Score a market: opportunity (0-10), risk (0-10), recommendation (BUY YES/NO, CONSIDER, SKIP).",
        "input_schema": {
            "type": "object",
            "properties": {
                "market_id": {"type": "string", "description": "The market ID to analyze"},
                "market_title": {"type": "string", "description": "Market title for context"},
            },
            "required": ["market_id"],
        },
    },
    {
        "name": "get_whale_activity",
        "description": "Track recent trades from whale wallets. Flags when 2+ whales agree on same market/side.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hours_back": {"type": "integer", "description": "Hours to look back (default 24)"},
            },
            "required": [],
        },
    },
    {
        "name": "calculate_edge",
        "description": "Compare your probability estimate vs market price. Returns edge in percentage points.",
        "input_schema": {
            "type": "object",
            "properties": {
                "market_id": {"type": "string", "description": "Market ID to check price"},
                "my_estimate": {"type": "number", "description": "Your probability estimate (0-100)"},
            },
            "required": ["market_id", "my_estimate"],
        },
    },
    {
        "name": "get_news",
        "description": "Fetch relevant news articles for a market query. Uses GNews or Google News RSS.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for relevant news"},
                "max_results": {"type": "integer", "description": "Max articles (default 5)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_search",
        "description": "Search memory (knowledge + trades) using BM25 with temporal decay. Recent memories weigh more.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "Number of results (default 5)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "save_knowledge",
        "description": "Save a learning insight to persistent memory. Use after each trade cycle for continuous learning.",
        "input_schema": {
            "type": "object",
            "properties": {
                "insight": {"type": "string", "description": "The learning insight to save"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization",
                },
            },
            "required": ["insight"],
        },
    },
    {
        "name": "save_trade_result",
        "description": "Record a trade result (win/loss) for learning. Call when a trade resolves.",
        "input_schema": {
            "type": "object",
            "properties": {
                "market_id": {"type": "string"},
                "title": {"type": "string"},
                "side": {"type": "string"},
                "amount_usdc": {"type": "number"},
                "pnl": {"type": "number"},
                "reason": {"type": "string"},
                "category": {"type": "string"},
                "resolved": {"type": "boolean"},
            },
            "required": ["market_id", "title", "side", "amount_usdc", "pnl", "reason", "category", "resolved"],
        },
    },
    {
        "name": "get_stats",
        "description": "Get overall trading stats: win rate, PnL, streak, best/worst trade.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_performance_by_category",
        "description": "Get win rate and PnL broken down by market category. Identifies strongest edges.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "place_order",
        "description": "Execute a trade on Polymarket. Includes safety checks (balance, duplicates, limits).",
        "input_schema": {
            "type": "object",
            "properties": {
                "market_id": {"type": "string", "description": "Market to trade"},
                "market_title": {"type": "string", "description": "Market title for logging"},
                "side": {"type": "string", "description": "YES or NO"},
                "amount_usdc": {"type": "number", "description": "Amount in USDC to bet"},
                "reason": {"type": "string", "description": "Explicit reasoning for this trade"},
            },
            "required": ["market_id", "market_title", "side", "amount_usdc", "reason"],
        },
    },
    {
        "name": "get_trade_history",
        "description": "Get resolved trade history from the blockchain for learning.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "detect_opportunities",
        "description": "Run opportunity detection across a set of markets. Returns ranked list by opportunity score.",
        "input_schema": {
            "type": "object",
            "properties": {
                "markets": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of market objects to analyze",
                },
            },
            "required": ["markets"],
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Eres Polybot, un agente de trading autónomo corriendo 24/7 en un servidor VPS.
Tu único objetivo es ganar dinero de forma consistente en Polymarket.
Operas con USDC real. Cada trade importa.

ESTRATEGIAS QUE FUNCIONAN:
1. Near-resolution trades: mercados con 92%+ de probabilidad que resuelven en menos de 7 días.
   Bajo riesgo, retorno casi garantizado. Prioridad máxima.
2. Whale copytrading: sigue wallets con track record probado.
   Si 2 o más whales coinciden en una posición, es señal muy fuerte — actúa.
3. Mercados con información no descontada: cuando hay noticias recientes que el mercado
   aún no ha reflejado completamente en el precio. Busca el edge.

ESTRATEGIAS QUE PIERDEN (nunca ejecutar):
- Crypto exact-price markets (ej: BTC > $95,000 el viernes) — impredecibles
- Esports markets — volátiles, poca liquidez, sesgados
- Mercados con resolución > 30 días — capital bloqueado demasiado tiempo
- Apostar contra probabilidades > 85% — el upside no justifica el riesgo
- Mercados con volumen < 500 USDC — riesgo de liquidez

PROCESO DE DECISIÓN (seguir este orden exacto cada ciclo):
1. get_balance() — si balance < 10 USDC, no operar, enviar alerta
2. get_positions() — revisar posiciones abiertas, no superar 10 simultáneas
3. memory_search('winning trades recent') — recordar qué ha funcionado
4. get_markets(min_volume=500) — escanear todos los mercados disponibles
5. get_whale_activity(hours_back=24) — identificar movimientos inteligentes
6. Para los top 5 candidatos: get_news(query) + analyze_market(id)
7. Calcular edge para cada candidato: mi_estimacion - probabilidad_mercado
8. Ejecutar MÁXIMO 3 trades por ciclo con estas prioridades:
   - Near-resolution + whale confirmation = ejecutar siempre
   - Edge > 5 puntos porcentuales + volumen alto = ejecutar
   - Diversificar: no 3 trades de la misma categoría
9. save_knowledge() — guardar el aprendizaje más importante del ciclo

CÁLCULO DE TAMAÑO DE APUESTA:
- Base: min(max_bet_usdc, balance * 0.20)
- Multiplicador por confianza:
  * Near-resolution 92%+: 1.5x
  * Whale confirmation: 1.3x
  * Solo análisis propio: 1.0x
- Nunca superar max_bet_usdc del config
- Nunca más del 20% del balance total en un trade

GESTIÓN DE RIESGO ESTRICTA:
- Si el balance cae 30% en 7 días: reducir max_bet_usdc a la mitad automáticamente
- Si 5 pérdidas consecutivas: parar y enviar alerta por Telegram
- Si balance < 5 USDC: parar completamente, notificar

RAZONAMIENTO ANTES DE CADA TRADE:
Antes de place_order, razona explícitamente:
  Mercado: [título]
  Probabilidad mercado: X%
  Mi estimación: Y%
  Edge: +Z puntos
  Whale activity: [sí/no, cuántos coinciden]
  Noticias relevantes: [resumen breve]
  Riesgo: [bajo/medio/alto]
  Tamaño: [X USDC] ([Y]% del balance)
  Decisión: OPERAR porque [razón concreta]

APRENDIZAJE CONTINUO:
- Trade ganado: save_knowledge con categoría, probabilidad inicial, tipo de señal que lo identificó
- Trade perdido: save_knowledge con qué falló y qué evitar
- Cada 50 trades: analizar win rate por categoría via get_performance_by_category()
- Las lecciones recientes pesan más (decaimiento temporal BM25, half-life 30 días)

EFICIENCIA DE TOKENS:
- Sé conciso en tu razonamiento. No repitas información que ya tienes.
- Usa los tools en el orden correcto para minimizar llamadas innecesarias.
- Si un ciclo no tiene oportunidades claras, di "Sin oportunidades este ciclo" y termina rápido."""


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------
def execute_tool(name: str, args: dict, config: dict) -> Any:
    """Route tool calls to the appropriate function."""
    from tools.polymarket import (
        get_markets, get_market_detail, get_balance,
        get_positions, place_order, get_trade_history,
    )
    from tools.memory import (
        memory_search, save_knowledge, save_trade_result,
        get_stats, get_performance_by_category,
    )
    from tools.analysis import (
        analyze_market, get_whale_activity, calculate_edge,
        detect_opportunities,
    )
    from tools.news import get_relevant_news

    wallet = config["wallet_address"]
    private_key = config.get("private_key", "")

    if name == "get_balance":
        return get_balance(wallet)

    elif name == "get_positions":
        return get_positions(wallet)

    elif name == "get_markets":
        return get_markets(
            min_volume=args.get("min_volume", config.get("min_volume", 500)),
            min_probability=args.get("min_probability", 0),
            max_probability=args.get("max_probability", 100),
            limit=args.get("limit", 50),
            keyword=args.get("keyword", ""),
            category=args.get("category", ""),
        )

    elif name == "get_market_detail":
        return get_market_detail(args["market_id"])

    elif name == "analyze_market":
        return analyze_market(args["market_id"], args.get("market_title", ""))

    elif name == "get_whale_activity":
        return get_whale_activity(
            config.get("whale_wallets", []),
            args.get("hours_back", 24),
        )

    elif name == "calculate_edge":
        detail = get_market_detail(args["market_id"])
        market_prob = detail.get("yes_price", 0.5) * 100
        return calculate_edge(args["my_estimate"], market_prob)

    elif name == "get_news":
        return get_relevant_news(
            args["query"],
            args.get("max_results", 5),
            config.get("news_api_key", ""),
        )

    elif name == "memory_search":
        return memory_search(args["query"], args.get("top_k", 5))

    elif name == "save_knowledge":
        return save_knowledge(args["insight"], args.get("tags", []))

    elif name == "save_trade_result":
        return save_trade_result(**args)

    elif name == "get_stats":
        return get_stats()

    elif name == "get_performance_by_category":
        return get_performance_by_category()

    elif name == "place_order":
        result = place_order(
            market_id=args["market_id"],
            market_title=args.get("market_title", ""),
            side=args["side"],
            amount_usdc=args["amount_usdc"],
            reason=args.get("reason", ""),
            wallet_address=wallet,
            private_key=private_key,
            max_bet_usdc=config.get("max_bet_usdc", 5),
            api_key=config.get("polymarket_api_key", ""),
            api_secret=config.get("polymarket_api_secret", ""),
            api_passphrase=config.get("polymarket_api_passphrase", ""),
        )
        # Notify via Telegram on trade execution
        if result.get("executed"):
            msg = (
                f"*TRADE EXECUTED*\n"
                f"Market: {args.get('market_title', '')}\n"
                f"Side: {args['side']} | Amount: {args['amount_usdc']} USDC\n"
                f"Reason: {args.get('reason', '')}"
            )
            send_telegram(msg, config)
        return result

    elif name == "get_trade_history":
        return get_trade_history(wallet)

    elif name == "detect_opportunities":
        return detect_opportunities(args.get("markets", []))

    else:
        return {"error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------
BALANCE_HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "memory", "balance_history.json"
)


def _record_balance(balance_usdc: float) -> None:
    """Append current balance to history for drawdown tracking."""
    os.makedirs(os.path.dirname(BALANCE_HISTORY_FILE), exist_ok=True)
    history = []
    try:
        with open(BALANCE_HISTORY_FILE, "r") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    history.append({"balance": balance_usdc, "timestamp": time.time()})

    # Keep only last 30 days of snapshots
    cutoff = time.time() - (30 * 86400)
    history = [h for h in history if h["timestamp"] >= cutoff]

    with open(BALANCE_HISTORY_FILE, "w") as f:
        json.dump(history, f)


def _check_drawdown(config: dict) -> tuple[bool, str]:
    """Check if balance has dropped 30%+ in last 7 days. Returns (triggered, message)."""
    try:
        with open(BALANCE_HISTORY_FILE, "r") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return False, ""

    if not history:
        return False, ""

    seven_days_ago = time.time() - (7 * 86400)
    old_snapshots = [h for h in history if h["timestamp"] <= seven_days_ago + 3600]

    if not old_snapshots:
        return False, ""

    # Use the earliest snapshot within the 7-day window
    reference_balance = old_snapshots[-1]["balance"]
    current_balance = history[-1]["balance"]

    if reference_balance <= 0:
        return False, ""

    drawdown_pct = (reference_balance - current_balance) / reference_balance * 100

    if drawdown_pct >= 30:
        new_max_bet = max(config.get("max_bet_usdc", 5) / 2, 1.0)
        config["max_bet_usdc"] = new_max_bet
        msg = (
            f"DRAWDOWN ALERT: Balance dropped {drawdown_pct:.1f}% in 7 days "
            f"({reference_balance:.2f} -> {current_balance:.2f} USDC). "
            f"max_bet_usdc reduced to {new_max_bet:.1f}"
        )
        return True, msg

    return False, ""


def check_risk_limits(config: dict) -> dict[str, Any]:
    """Check if risk limits are breached. Returns risk status and any adjustments."""
    from tools.memory import get_stats
    from tools.polymarket import get_balance

    stats = get_stats()
    streak = stats.get("current_streak", 0)
    alerts: list[str] = []
    should_stop = False

    # Record balance snapshot for drawdown tracking
    balance_info = get_balance(config["wallet_address"])
    balance_usdc = balance_info.get("balance_usdc", -1)
    if balance_usdc >= 0:
        _record_balance(balance_usdc)

    # 5 consecutive losses -> stop
    if streak <= -5:
        alerts.append(f"5+ consecutive losses (streak: {streak}). STOPPING.")
        should_stop = True

    # 30% drawdown in 7 days -> halve max_bet_usdc
    drawdown_triggered, drawdown_msg = _check_drawdown(config)
    if drawdown_triggered:
        alerts.append(drawdown_msg)
        logger.warning(drawdown_msg)

    return {
        "streak": streak,
        "alerts": alerts,
        "should_stop": should_stop,
        "stats": stats,
        "balance_usdc": balance_usdc,
    }


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
class PolybotAgent:
    """Multi-turn LLM agent that reasons and trades autonomously."""

    def __init__(self, config: dict):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config["anthropic_api_key"])
        self.model = "claude-sonnet-4-20250514"
        self.max_turns = 15  # Max tool-call turns per cycle
        self.running = True
        self.cycle_count = 0
        self.has_near_resolution = False

    def run_cycle(self) -> None:
        """Run one full trading cycle."""
        self.cycle_count += 1
        self.has_near_resolution = False
        logger.info(f"=== CYCLE {self.cycle_count} START ===")

        # Pre-cycle risk check
        risk = check_risk_limits(self.config)
        if risk["should_stop"]:
            for alert in risk["alerts"]:
                logger.warning(alert)
                send_telegram(f"*RISK ALERT*: {alert}", self.config)
            logger.info("Cycle skipped due to risk limits.")
            return

        # Build initial message with context
        cycle_msg = (
            f"Nuevo ciclo de trading (#{self.cycle_count}). "
            f"Config: max_bet={self.config.get('max_bet_usdc', 5)} USDC, "
            f"max_positions={self.config.get('max_positions', 10)}. "
            f"Sigue tu proceso de decisión paso a paso."
        )

        if risk["stats"].get("resolved_trades", 0) > 0:
            s = risk["stats"]
            cycle_msg += (
                f"\nEstadísticas actuales: {s['wins']}W/{s['losses']}L "
                f"(WR: {s['win_rate']}%), PnL: {s['total_pnl']} USDC, "
                f"Racha: {s['current_streak']}"
            )

        messages: list[dict] = [{"role": "user", "content": cycle_msg}]

        # Multi-turn loop
        for turn in range(self.max_turns):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                )
            except anthropic.APIError as e:
                logger.error(f"Anthropic API error: {e}")
                time.sleep(30)
                return

            # Process response
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Extract text blocks for logging
            for block in assistant_content:
                if hasattr(block, "text") and block.text:
                    logger.info(f"Agent: {block.text[:200]}")

            # If no tool use, cycle is complete
            if response.stop_reason == "end_turn":
                logger.info("Agent finished reasoning - cycle complete.")
                break

            # Process tool calls
            tool_results = []
            for block in assistant_content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input
                logger.info(f"Tool call: {tool_name}({json.dumps(tool_input, default=str)[:200]})")

                try:
                    result = execute_tool(tool_name, tool_input, self.config)

                    # Detect near-resolution opportunities
                    if tool_name == "analyze_market" and isinstance(result, dict):
                        if result.get("days_to_resolution", 999) <= 7 and result.get("favored_probability", 0) >= 92:
                            self.has_near_resolution = True

                except Exception as e:
                    logger.error(f"Tool {tool_name} failed: {e}")
                    result = {"error": str(e)}

                # Truncate large results to save tokens
                result_str = json.dumps(result, default=str)
                if len(result_str) > 3000:
                    result_str = result_str[:3000] + "...[truncated]"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        logger.info(f"=== CYCLE {self.cycle_count} END (turns: {turn + 1}) ===")

    def get_cycle_interval(self) -> int:
        """Return cycle interval, faster if near-resolution opportunities exist."""
        if self.has_near_resolution:
            return self.config.get("near_resolution_interval_seconds", 30)
        return self.config.get("cycle_interval_seconds", 120)

    def startup_report(self) -> None:
        """Print startup info."""
        from tools.polymarket import get_balance, get_positions
        from tools.memory import get_stats

        logger.info("=" * 60)
        logger.info("POLYBOT - Autonomous Trading Agent")
        logger.info("=" * 60)

        balance = get_balance(self.config["wallet_address"])
        logger.info(f"Wallet: {self.config['wallet_address']}")
        logger.info(f"Balance: {balance.get('balance_usdc', '?')} USDC")
        if balance.get("alert"):
            logger.warning(balance["alert"])

        positions = get_positions(self.config["wallet_address"])
        pos_count = len([p for p in positions if not isinstance(p, dict) or "error" not in p])
        logger.info(f"Open positions: {pos_count}")

        stats = get_stats()
        if stats.get("resolved_trades", 0) > 0:
            logger.info(
                f"Stats: {stats['wins']}W/{stats['losses']}L "
                f"(WR: {stats['win_rate']}%), PnL: {stats['total_pnl']} USDC"
            )
        else:
            logger.info("Stats: No resolved trades yet")

        logger.info(f"Max bet: {self.config.get('max_bet_usdc', 5)} USDC")
        logger.info(f"Cycle interval: {self.config.get('cycle_interval_seconds', 120)}s")
        logger.info("=" * 60)

        send_telegram(
            f"*Polybot started*\nBalance: {balance.get('balance_usdc', '?')} USDC\n"
            f"Open positions: {pos_count}",
            self.config,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    config = load_config()
    agent = PolybotAgent(config)

    # Graceful shutdown
    def shutdown(sig, frame):
        logger.info("Shutdown signal received. Stopping after current cycle...")
        agent.running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    agent.startup_report()

    while agent.running:
        try:
            agent.run_cycle()
        except Exception as e:
            logger.error(f"Cycle failed: {e}", exc_info=True)
            send_telegram(f"*ERROR*: Cycle failed: {e}", config)
            time.sleep(60)
            continue

        interval = agent.get_cycle_interval()
        logger.info(f"Next cycle in {interval}s...")

        # Sleep in small increments for responsive shutdown
        for _ in range(interval):
            if not agent.running:
                break
            time.sleep(1)

    logger.info("Polybot stopped gracefully.")
    send_telegram("*Polybot stopped.*", config)


if __name__ == "__main__":
    main()
