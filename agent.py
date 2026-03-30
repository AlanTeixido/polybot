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
        requests.post(url, json={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }, timeout=10)
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


def send_daily_summary(config: dict) -> None:
    """Send daily performance summary via Telegram."""
    from tools.memory import get_stats, get_performance_by_category
    from tools.polymarket import get_balance, get_positions

    venue = config.get("venue", "sim")
    currency = "$SIM" if venue == "sim" else "USDC"

    stats = get_stats()
    balance = get_balance(
        config["wallet_address"],
        venue=venue,
        simmer_api_key=config.get("simmer_api_key", ""),
    )
    positions = get_positions(
        config["wallet_address"],
        venue=venue,
        simmer_api_key=config.get("simmer_api_key", ""),
    )
    pos_count = len([p for p in positions if isinstance(p, dict) and "error" not in p])
    categories = get_performance_by_category()

    bal = balance.get("balance_usdc", "?")
    wr = stats.get("win_rate", 0)
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    pnl = stats.get("total_pnl", 0)
    streak = stats.get("current_streak", 0)

    # Best/worst categories
    cat_lines = ""
    if categories:
        sorted_cats = sorted(categories.items(), key=lambda x: x[1]["pnl"], reverse=True)
        for cat, data in sorted_cats[:3]:
            emoji = "+" if data["pnl"] >= 0 else ""
            cat_lines += f"  {cat}: {data['wins']}W/{data['losses']}L ({emoji}{data['pnl']} {currency})\n"

    msg = (
        f"*DAILY SUMMARY*\n"
        f"Balance: {bal} {currency}\n"
        f"Open positions: {pos_count}\n"
        f"Record: {wins}W/{losses}L (WR: {wr}%)\n"
        f"Total PnL: {'+' if pnl >= 0 else ''}{pnl} {currency}\n"
        f"Streak: {streak}\n"
    )
    if cat_lines:
        msg += f"\n*By category:*\n{cat_lines}"

    send_telegram(msg, config)


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
    {
        "name": "get_simmer_briefing",
        "description": "Get Simmer portfolio briefing: positions, risk alerts, new markets, performance summary.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_simmer_context",
        "description": "Check Simmer context for a market: real edge calculation, slippage, and TRADE/HOLD recommendation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "market_id": {"type": "string", "description": "Market ID to check"},
                "my_probability": {"type": "number", "description": "Your probability estimate (0-100)"},
            },
            "required": ["market_id", "my_probability"],
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

WHALE TRACKING:
- Whale wallets tracked are top Polymarket traders by profit.
- Follow their trades when 2+ agree on same market/side.

APRENDIZAJE CONTINUO:
- Trade ganado: save_knowledge con categoría, probabilidad inicial, tipo de señal que lo identificó
- Trade perdido: save_knowledge con qué falló y qué evitar
- Cada 50 trades: analizar win rate por categoría via get_performance_by_category()
- Las lecciones recientes pesan más (decaimiento temporal BM25, half-life 30 días)

VENUE SIM (cuando operas en Simmer):
- Volume y liquidity se muestran como "sim" — son virtuales, NO los uses para filtrar.
- En SIM, todos los mercados son operables. No descartes por volumen o liquidez.
- Opera más agresivamente en SIM: es dinero virtual, el objetivo es generar datos.
- Umbral de probabilidad mínimo: 80% (no 92%) para near-resolution en SIM.

EFICIENCIA DE TOKENS (CRÍTICO — cada token cuesta dinero real):
- MÁXIMO 3 tool calls por ciclo. No llames get_balance + get_positions + memory_search si ya tienes esos datos en el mensaje inicial.
- El prescan ya te da: balance, mercados, whale signals. USA esos datos, no los re-fetches.
- Si no hay oportunidades claras, responde "Sin oportunidades" y termina. No analices cada mercado.
- Razona en 2-3 frases, no párrafos. Cada palabra cuesta."""


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
        detect_opportunities, get_simmer_context,
    )
    from tools.news import get_relevant_news
    import requests as _requests

    wallet = config["wallet_address"]
    private_key = config.get("private_key", "")

    venue = config.get("venue", "sim")
    simmer_key = config.get("simmer_api_key", "")

    if name == "get_balance":
        return get_balance(wallet, custom_rpc=config.get("polygon_rpc", ""),
                           venue=venue, simmer_api_key=simmer_key)

    elif name == "get_positions":
        return get_positions(wallet, venue=venue, simmer_api_key=simmer_key)

    elif name == "get_markets":
        return get_markets(
            min_volume=args.get("min_volume", config.get("min_volume", 500)),
            min_probability=args.get("min_probability", 0),
            max_probability=args.get("max_probability", 100),
            limit=args.get("limit", 50),
            keyword=args.get("keyword", ""),
            category=args.get("category", ""),
            venue=venue,
            simmer_api_key=simmer_key,
        )

    elif name == "get_market_detail":
        return get_market_detail(args["market_id"], venue=venue, simmer_api_key=simmer_key)

    elif name == "analyze_market":
        return analyze_market(args["market_id"], args.get("market_title", ""),
                              venue=venue, simmer_api_key=simmer_key)

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
            dry_run=config.get("dry_run", True),
            venue=venue,
            simmer_api_key=simmer_key,
        )
        # Notify via Telegram only on real execution (not dry run)
        if result.get("executed"):
            currency = "$SIM" if config.get("venue", "sim") == "sim" else "USDC"
            msg = f"{args['side']} {args['amount_usdc']}{currency} | {args.get('market_title', '')}"
            send_telegram(msg, config)
        return result

    elif name == "get_trade_history":
        return get_trade_history(wallet)

    elif name == "detect_opportunities":
        return detect_opportunities(args.get("markets", []))

    elif name == "get_simmer_briefing":
        simmer_key = config.get("simmer_api_key", "")
        if not simmer_key:
            return {"skipped": "No Simmer API key configured"}
        try:
            resp = _requests.get(
                "https://api.simmer.markets/api/sdk/briefing",
                headers={"Authorization": f"Bearer {simmer_key}"},
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Simmer briefing failed: {e}")
            return {"error": str(e)}

    elif name == "get_simmer_context":
        return get_simmer_context(
            args["market_id"],
            args["my_probability"],
            config.get("simmer_api_key", ""),
        ) or {"skipped": "No Simmer API key configured"}

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


def check_risk_limits(config: dict, cached_balance: float = -1) -> dict[str, Any]:
    """Check if risk limits are breached. Returns risk status and any adjustments."""
    from tools.memory import get_stats

    stats = get_stats()
    streak = stats.get("current_streak", 0)
    alerts: list[str] = []
    should_stop = False

    # Use cached balance if available, otherwise fetch
    if cached_balance >= 0:
        balance_usdc = cached_balance
    else:
        from tools.polymarket import get_balance
        balance_info = get_balance(
            config["wallet_address"],
            custom_rpc=config.get("polygon_rpc", ""),
            venue=config.get("venue", "sim"),
            simmer_api_key=config.get("simmer_api_key", ""),
        )
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
        self.max_turns = 8  # Max tool-call turns per cycle (keep low to save tokens)
        self.running = True
        self.cycle_count = 0
        self.has_near_resolution = False
        self.previous_positions: list[dict] = []
        self.previous_market_ids: set[str] = set()
        # Balance cache (avoid double RPC calls per cycle)
        self._cached_balance: float = -1
        self._balance_cache_ts: float = 0
        self._balance_cache_ttl: float = 60

    def get_cached_balance(self) -> float:
        """Get balance with 60s cache to avoid redundant RPC calls."""
        now = time.time()
        if now - self._balance_cache_ts < self._balance_cache_ttl and self._cached_balance >= 0:
            return self._cached_balance
        from tools.polymarket import get_balance
        info = get_balance(
            self.config["wallet_address"],
            custom_rpc=self.config.get("polygon_rpc", ""),
            venue=self.config.get("venue", "sim"),
            simmer_api_key=self.config.get("simmer_api_key", ""),
        )
        self._cached_balance = info.get("balance_usdc", -1)
        self._balance_cache_ts = now
        return self._cached_balance

    def prescan(self) -> dict[str, Any]:
        """Fast Python-only scan. Returns whether LLM should be invoked and why."""
        from tools.polymarket import get_markets
        from tools.analysis import get_whale_activity

        reasons: list[str] = []
        near_res_markets: list[dict] = []
        whale_signals: list[dict] = []
        new_markets: list[str] = []

        # 1. Check balance — if too low, don't bother scanning
        balance = self.get_cached_balance()
        if balance >= 0 and balance < 10:
            logger.info(f"Prescan: balance too low ({balance} USDC), skipping LLM")
            return {"invoke_llm": False, "reasons": ["balance_too_low"], "balance": balance}

        # 2. Scan markets (uses 60s cache, essentially free)
        markets = get_markets(
            min_volume=self.config.get("min_volume", 500),
            venue=self.config.get("venue", "sim"),
            simmer_api_key=self.config.get("simmer_api_key", ""),
        )
        if markets and not (len(markets) == 1 and "error" in markets[0]):
            current_ids = {m["id"] for m in markets if "id" in m}

            # Near-resolution opportunities (lower threshold for sim venue)
            venue = self.config.get("venue", "sim")
            nr_threshold = 80 if venue == "sim" else 92
            for m in markets:
                prob = m.get("yes_probability", 50)
                days = m.get("days_to_resolution", 999)
                max_prob = max(prob, 100 - prob)
                if max_prob >= nr_threshold and days <= 7:
                    near_res_markets.append(m)

            if near_res_markets:
                reasons.append(f"near_resolution:{len(near_res_markets)}")

            # New markets since last cycle
            if self.previous_market_ids:
                new_ids = current_ids - self.previous_market_ids
                new_markets = [m["title"] for m in markets if m.get("id") in new_ids]
                if new_markets:
                    reasons.append(f"new_markets:{len(new_markets)}")

            self.previous_market_ids = current_ids

        # 3. Whale activity
        whale_data = get_whale_activity(
            self.config.get("whale_wallets", []), hours_back=24
        )
        whale_signals = whale_data.get("strong_signals", [])
        if whale_signals:
            reasons.append(f"whale_signals:{len(whale_signals)}")

        # SIM venue: always invoke LLM if there are any markets (operate more)
        if self.config.get("venue", "sim") == "sim" and markets and not (len(markets) == 1 and "error" in markets[0]):
            if not reasons:
                reasons.append("sim_venue_active")

        invoke = len(reasons) > 0
        if not invoke:
            logger.info("Prescan: no opportunities detected, skipping LLM")
        else:
            logger.info(f"Prescan: opportunities found [{', '.join(reasons)}] -> invoking LLM")

        return {
            "invoke_llm": invoke,
            "reasons": reasons,
            "balance": balance,
            "near_resolution_markets": near_res_markets,
            "whale_signals": whale_signals,
            "new_markets": new_markets,
            "total_markets_scanned": len(markets) if markets else 0,
        }

    def _detect_resolved_trades(self) -> None:
        """Compare previous vs current positions to detect resolved trades."""
        if not self.previous_positions:
            return

        from tools.polymarket import get_positions
        from tools.memory import save_trade_result, save_knowledge

        current = get_positions(self.config["wallet_address"])
        current_ids = {
            p["market_id"] for p in current
            if isinstance(p, dict) and "error" not in p
        }

        for prev_pos in self.previous_positions:
            if prev_pos.get("market_id") not in current_ids:
                # Position disappeared = resolved
                pnl = prev_pos.get("unrealized_pnl", 0)
                won = pnl > 0
                title = prev_pos.get("title", "unknown")
                side = prev_pos.get("side", "unknown")

                save_trade_result(
                    market_id=prev_pos.get("market_id", ""),
                    title=title,
                    side=side,
                    amount_usdc=prev_pos.get("size", 0) * prev_pos.get("avg_price", 0),
                    pnl=round(pnl, 4),
                    reason="auto-detected resolution",
                    category="unknown",
                    resolved=True,
                )

                outcome = "WIN" if won else "LOSS"
                save_knowledge(
                    insight=f"{outcome}: '{title}' ({side}) resolved with PnL {pnl:.4f} USDC. "
                            f"Avg price: {prev_pos.get('avg_price', 0)}, "
                            f"Final price: {prev_pos.get('current_price', 0)}",
                    tags=[outcome.lower(), "resolved", "auto-detected"],
                )

                currency = "$SIM" if self.config.get("venue", "sim") == "sim" else "USDC"
                pnl_sign = "+" if pnl >= 0 else ""
                msg = f"*{outcome}* {pnl_sign}{pnl:.2f} {currency}\n{title}\nSide: {side}"
                logger.info(msg.replace("*", ""))
                send_telegram(msg, self.config)

    def run_cycle(self) -> None:
        """Run one full trading cycle."""
        self.cycle_count += 1
        self.has_near_resolution = False
        logger.info(f"=== CYCLE {self.cycle_count} START ===")

        # Detect resolved trades by comparing with previous cycle positions
        self._detect_resolved_trades()

        # Pre-cycle risk check (uses cached balance)
        risk = check_risk_limits(self.config, cached_balance=self.get_cached_balance())
        if risk["should_stop"]:
            for alert in risk["alerts"]:
                logger.warning(alert)
                send_telegram(f"*RISK ALERT*: {alert}", self.config)
            logger.info("Cycle skipped due to risk limits.")
            return

        # --- PRESCAN: fast Python-only check before invoking LLM ---
        scan = self.prescan()
        if not scan["invoke_llm"]:
            # Still snapshot positions for resolved-trade detection
            from tools.polymarket import get_positions as _get_pos
            current_pos = _get_pos(self.config["wallet_address"])
            self.previous_positions = [
                p for p in current_pos if isinstance(p, dict) and "error" not in p
            ]
            logger.info(f"=== CYCLE {self.cycle_count} END (prescan: no opportunities) ===")
            return

        # Flag near-resolution for faster next cycle
        if scan.get("near_resolution_markets"):
            self.has_near_resolution = True

        # Build initial message with prescan context (saves LLM tool calls)
        cycle_msg = (
            f"Ciclo #{self.cycle_count}. "
            f"Config: max_bet={self.config.get('max_bet_usdc', 5)} USDC, "
            f"max_positions={self.config.get('max_positions', 10)}.\n"
            f"Balance: {scan['balance']} USDC.\n"
            f"Prescan detectó: {', '.join(scan['reasons'])}.\n"
        )

        # Include prescan data so LLM doesn't need to re-fetch
        if scan.get("near_resolution_markets"):
            nr = scan["near_resolution_markets"]
            cycle_msg += f"\nNear-resolution ({len(nr)}):\n"
            for m in nr[:5]:
                cycle_msg += (
                    f"  - {m['title']} | prob: {m['yes_probability']}% | "
                    f"days: {m['days_to_resolution']} | vol: {m['volume']}\n"
                )

        if scan.get("whale_signals"):
            ws = scan["whale_signals"]
            cycle_msg += f"\nWhale signals ({len(ws)}):\n"
            for s in ws[:3]:
                cycle_msg += (
                    f"  - {s.get('title', s.get('market_id', '?'))} | "
                    f"side: {s.get('side')} | whales: {s.get('whale_count')}\n"
                )

        if scan.get("new_markets"):
            cycle_msg += f"\nNew markets: {', '.join(scan['new_markets'][:5])}\n"

        cycle_msg += "\nAnaliza las oportunidades y ejecuta los mejores trades."

        if risk["stats"].get("resolved_trades", 0) > 0:
            s = risk["stats"]
            cycle_msg += (
                f"\nStats: {s['wins']}W/{s['losses']}L "
                f"(WR: {s['win_rate']}%), PnL: {s['total_pnl']} USDC, "
                f"Racha: {s['current_streak']}"
            )

        messages: list[dict] = [{"role": "user", "content": cycle_msg}]

        # Multi-turn loop
        for turn in range(self.max_turns):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
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

        # Snapshot current positions for next cycle's resolved-trade detection
        from tools.polymarket import get_positions as _get_pos
        current_pos = _get_pos(self.config["wallet_address"])
        self.previous_positions = [
            p for p in current_pos if isinstance(p, dict) and "error" not in p
        ]

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

        venue = self.config.get("venue", "sim")
        simmer_key = self.config.get("simmer_api_key", "")
        currency = "$SIM" if venue == "sim" else "USDC"

        logger.info("=" * 60)
        logger.info("POLYBOT - Autonomous Trading Agent")
        logger.info("=" * 60)

        if venue == "sim":
            logger.info("VENUE: SIM (virtual $SIM)")
        else:
            logger.info("VENUE: POLYMARKET (real USDC)")

        balance = get_balance(
            self.config["wallet_address"],
            custom_rpc=self.config.get("polygon_rpc", ""),
            venue=venue, simmer_api_key=simmer_key,
        )
        logger.info(f"Wallet: {self.config['wallet_address']}")
        logger.info(f"Balance: {balance.get('balance_usdc', '?')} {currency}")
        if balance.get("alert"):
            logger.warning(balance["alert"])

        positions = get_positions(
            self.config["wallet_address"],
            venue=venue, simmer_api_key=simmer_key,
        )
        pos_count = len([p for p in positions if isinstance(p, dict) and "error" not in p])
        logger.info(f"Open positions: {pos_count}")

        stats = get_stats()
        if stats.get("resolved_trades", 0) > 0:
            logger.info(
                f"Stats: {stats['wins']}W/{stats['losses']}L "
                f"(WR: {stats['win_rate']}%), PnL: {stats['total_pnl']} {currency}"
            )
        else:
            logger.info("Stats: No resolved trades yet")

        logger.info(f"Max bet: {self.config.get('max_bet_usdc', 5)} {currency}")
        logger.info(f"Cycle interval: {self.config.get('cycle_interval_seconds', 30)}s")
        if self.config.get("dry_run", True):
            logger.info("MODE: DRY RUN (no real trades)")
        else:
            logger.info("MODE: LIVE")
        logger.info("=" * 60)

        mode = "DRY RUN" if self.config.get("dry_run", True) else "LIVE"
        send_telegram(
            f"*Polybot started*\n"
            f"Venue: {venue} | Mode: {mode}\n"
            f"Balance: {balance.get('balance_usdc', '?')} {currency}\n"
            f"Positions: {pos_count}\n"
            f"Max bet: {self.config.get('max_bet_usdc', 5)} {currency}\n"
            f"Cycle: {self.config.get('cycle_interval_seconds', 30)}s",
            self.config,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    config = load_config()
    agent = PolybotAgent(config)
    last_daily_summary: str = ""

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

        # Daily summary at 00:00 UTC
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if today != last_daily_summary and time.strftime("%H", time.gmtime()) == "00":
            try:
                send_daily_summary(config)
                last_daily_summary = today
                logger.info("Daily summary sent")
            except Exception as e:
                logger.warning(f"Daily summary failed: {e}")

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
