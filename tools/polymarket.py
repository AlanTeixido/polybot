"""Polymarket API integration: markets, orders, positions, balance."""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import requests
from web3 import Web3

logger = logging.getLogger("polybot.polymarket")

# API endpoints
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

# Polygon USDC.e contract
USDC_ADDRESS = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"
USDC_ABI = json.dumps([{
    "constant": True,
    "inputs": [{"name": "_owner", "type": "address"}],
    "name": "balanceOf",
    "outputs": [{"name": "balance", "type": "uint256"}],
    "type": "function",
}])
POLYGON_RPC = "https://polygon-rpc.com"

# ClobClient singleton - initialized once, reused for all orders
_clob_client = None


def get_clob_client(
    private_key: str,
    wallet_address: str,
    api_key: str = "",
    api_secret: str = "",
    api_passphrase: str = "",
):
    """Return a singleton ClobClient, initializing only once."""
    global _clob_client
    if _clob_client is None:
        try:
            from py_clob_client.client import ClobClient
            _clob_client = ClobClient(
                CLOB_API,
                key=api_key or None,
                chain_id=137,
                private_key=private_key,
                signature_type=2,
                funder=wallet_address,
            )
            if api_key and api_secret and api_passphrase:
                _clob_client.set_api_creds(_clob_client.create_or_derive_api_creds())
            logger.info("ClobClient initialized (singleton)")
        except Exception as e:
            logger.error(f"ClobClient init failed: {e}")
            raise
    return _clob_client


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Polybot/1.0"})
REQUEST_TIMEOUT = 15

# Blacklist keywords for auto-filtering
BLACKLIST_KEYWORDS = [
    "exact price",
    "btc >", "btc <", "eth >", "eth <", "bitcoin >", "bitcoin <",
    "ethereum >", "ethereum <", "sol >", "sol <",
    "will reach", "will hit",
]
BLACKLIST_CATEGORIES = ["esports", "e-sports"]

# Market cache (60 seconds)
_markets_cache: dict[str, Any] = {"data": [], "timestamp": 0}
CACHE_TTL = 60


def _retry(func, retries: int = 3, backoff: float = 2.0):
    """Retry with exponential backoff."""
    last_err = None
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                wait = backoff ** attempt
                logger.warning(f"Retry {attempt+1}/{retries} after {wait}s: {e}")
                time.sleep(wait)
    raise last_err


def get_markets(
    min_volume: float = 500,
    min_probability: float = 0,
    max_probability: float = 100,
    limit: int = 50,
    keyword: str = "",
    category: str = "",
) -> list[dict[str, Any]]:
    """Fetch and filter available markets from Polymarket."""
    global _markets_cache

    # Return cached data if fresh
    if time.time() - _markets_cache["timestamp"] < CACHE_TTL and _markets_cache["data"]:
        logger.debug("Returning cached markets")
        return _markets_cache["data"]

    try:
        params: dict[str, Any] = {
            "limit": min(limit, 100),
            "active": True,
            "closed": False,
        }
        if keyword:
            params["tag"] = keyword

        def fetch():
            resp = SESSION.get(f"{GAMMA_API}/markets", params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()

        raw_markets = _retry(fetch)

        if not isinstance(raw_markets, list):
            raw_markets = raw_markets.get("data", []) if isinstance(raw_markets, dict) else []

        now = datetime.now(timezone.utc)
        filtered = []

        for m in raw_markets:
            title = (m.get("question", "") or m.get("title", "")).lower()
            cat = (m.get("category", "") or "").lower()

            # Apply blacklist filters
            if any(kw in title for kw in BLACKLIST_KEYWORDS):
                continue
            if any(bc in cat for bc in BLACKLIST_CATEGORIES):
                continue

            # Category filter
            if category and category.lower() not in cat:
                continue

            # Volume filter
            volume = float(m.get("volume", 0) or 0)
            if volume < min_volume:
                continue

            # Parse probability
            outcome_prices = m.get("outcomePrices", "")
            yes_prob = 50.0
            if isinstance(outcome_prices, str) and outcome_prices:
                try:
                    prices = json.loads(outcome_prices)
                    yes_prob = float(prices[0]) * 100
                except (json.JSONDecodeError, IndexError, TypeError):
                    pass
            elif isinstance(outcome_prices, list) and outcome_prices:
                yes_prob = float(outcome_prices[0]) * 100

            if not (min_probability <= yes_prob <= max_probability):
                continue

            # Days to resolution
            end_date_str = m.get("endDate") or m.get("end_date_iso", "")
            days_to_res = 999
            if end_date_str:
                try:
                    end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                    days_to_res = max((end_date - now).total_seconds() / 86400, 0)
                except (ValueError, TypeError):
                    pass

            # Skip markets > 30 days out
            if days_to_res > 30:
                continue

            filtered.append({
                "id": m.get("id", ""),
                "title": m.get("question", m.get("title", "")),
                "yes_probability": round(yes_prob, 1),
                "volume": volume,
                "liquidity": float(m.get("liquidity", 0) or 0),
                "end_date": end_date_str,
                "days_to_resolution": round(days_to_res, 1),
                "category": m.get("category", "unknown"),
                "condition_id": m.get("conditionId", ""),
            })

        # Quick score for pre-filtering: prioritize near-resolution + high volume + high probability
        for m in filtered:
            prob = m["yes_probability"]
            prob_score = prob / 100 if prob > 50 else (100 - prob) / 100
            vol_score = min(m["volume"] / 50000, 1.0)
            time_score = max(0, (7 - m["days_to_resolution"]) / 7)
            m["quick_score"] = round(prob_score + vol_score + time_score, 3)

        filtered.sort(key=lambda x: x["quick_score"], reverse=True)
        result = filtered[:15]

        # Update cache
        _markets_cache = {"data": result, "timestamp": time.time()}
        return result

    except Exception as e:
        logger.error(f"get_markets failed: {e}")
        return [{"error": str(e)}]


def get_market_detail(market_id: str) -> dict[str, Any]:
    """Get detailed market info including orderbook data."""
    try:
        def fetch():
            resp = SESSION.get(f"{GAMMA_API}/markets/{market_id}", timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()

        market = _retry(fetch)

        outcome_prices = market.get("outcomePrices", "")
        prices = []
        if isinstance(outcome_prices, str) and outcome_prices:
            try:
                prices = [float(p) for p in json.loads(outcome_prices)]
            except (json.JSONDecodeError, ValueError):
                pass
        elif isinstance(outcome_prices, list):
            prices = [float(p) for p in outcome_prices]

        yes_price = prices[0] if prices else 0.5
        no_price = prices[1] if len(prices) > 1 else 1 - yes_price

        return {
            "market_id": market_id,
            "title": market.get("question", ""),
            "description": (market.get("description", "") or "")[:300],
            "category": market.get("category", "unknown"),
            "yes_price": round(yes_price, 4),
            "no_price": round(no_price, 4),
            "volume": float(market.get("volume", 0) or 0),
            "liquidity": float(market.get("liquidity", 0) or 0),
            "end_date": market.get("endDate", ""),
            "outcomes": market.get("outcomes", []),
            "condition_id": market.get("conditionId", ""),
            "clob_token_ids": market.get("clobTokenIds", ""),
        }
    except Exception as e:
        logger.error(f"get_market_detail failed for {market_id}: {e}")
        return {"error": str(e), "market_id": market_id}


def get_balance(wallet_address: str) -> dict[str, Any]:
    """Get USDC.e balance on Polygon."""
    try:
        w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
        usdc = w3.eth.contract(
            address=Web3.to_checksum_address(USDC_ADDRESS),
            abi=json.loads(USDC_ABI),
        )
        raw_balance = usdc.functions.balanceOf(
            Web3.to_checksum_address(wallet_address)
        ).call()
        balance = raw_balance / 1e6  # USDC has 6 decimals

        alert = ""
        if balance < 5:
            alert = "CRITICAL: Balance below 5 USDC. Bot should stop trading."
        elif balance < 10:
            alert = "WARNING: Balance below 10 USDC. Reduce position sizes."

        return {
            "balance_usdc": round(balance, 2),
            "wallet": wallet_address,
            "alert": alert,
        }
    except Exception as e:
        logger.error(f"get_balance failed: {e}")
        return {"error": str(e), "balance_usdc": -1}


def get_positions(wallet_address: str) -> list[dict[str, Any]]:
    """Get open positions for the wallet."""
    try:
        def fetch():
            resp = SESSION.get(
                f"{GAMMA_API}/positions",
                params={"user": wallet_address},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()

        data = _retry(fetch)
        if not isinstance(data, list):
            data = data.get("data", []) if isinstance(data, dict) else []

        positions = []
        for p in data:
            size = float(p.get("size", 0) or 0)
            if size <= 0:
                continue

            current_price = float(p.get("currentPrice", p.get("price", 0)) or 0)
            avg_price = float(p.get("avgPrice", p.get("averagePrice", 0)) or 0)
            pnl = (current_price - avg_price) * size if avg_price > 0 else 0

            positions.append({
                "market_id": p.get("market", p.get("marketId", "")),
                "title": p.get("title", p.get("question", "")),
                "side": p.get("outcome", p.get("side", "")),
                "size": size,
                "avg_price": round(avg_price, 4),
                "current_price": round(current_price, 4),
                "unrealized_pnl": round(pnl, 4),
                "in_loss": pnl < -0.5,
            })

        return positions

    except Exception as e:
        logger.error(f"get_positions failed: {e}")
        return [{"error": str(e)}]


def place_order(
    market_id: str,
    market_title: str,
    side: str,
    amount_usdc: float,
    reason: str,
    wallet_address: str,
    private_key: str,
    max_bet_usdc: float,
    api_key: str = "",
    api_secret: str = "",
    api_passphrase: str = "",
    dry_run: bool = True,
) -> dict[str, Any]:
    """Place an order on Polymarket via CLOB API."""
    # --- Dry run mode ---
    if dry_run:
        logger.info(f"[DRY RUN] Would place: {side} {amount_usdc} USDC on '{market_title}'")
        return {
            "executed": False,
            "dry_run": True,
            "market_id": market_id,
            "market_title": market_title,
            "side": side,
            "amount_usdc": amount_usdc,
            "reason": reason,
        }

    # --- Safety checks ---
    if amount_usdc <= 0:
        return {"error": "Amount must be positive", "executed": False}

    if amount_usdc > max_bet_usdc:
        return {
            "error": f"Amount {amount_usdc} exceeds max_bet_usdc {max_bet_usdc}",
            "executed": False,
        }

    side = side.upper()
    if side not in ("YES", "NO"):
        return {"error": f"Invalid side: {side}. Must be YES or NO.", "executed": False}

    # Check balance first
    balance_info = get_balance(wallet_address)
    current_balance = balance_info.get("balance_usdc", 0)
    if current_balance < amount_usdc:
        return {
            "error": f"Insufficient balance: {current_balance} USDC < {amount_usdc} USDC",
            "executed": False,
        }

    # Check for duplicate position
    positions = get_positions(wallet_address)
    for pos in positions:
        if isinstance(pos, dict) and pos.get("market_id") == market_id:
            return {
                "error": f"Already have position in market {market_id}",
                "executed": False,
                "existing_position": pos,
            }

    # --- Execute order via CLOB ---
    try:
        from py_clob_client.clob_types import OrderArgs, OrderType

        client = get_clob_client(private_key, wallet_address, api_key, api_secret, api_passphrase)

        # Get market token IDs
        detail = get_market_detail(market_id)
        if "error" in detail:
            return {"error": f"Cannot fetch market detail: {detail['error']}", "executed": False}

        token_ids_raw = detail.get("clob_token_ids", "")
        token_ids = []
        if isinstance(token_ids_raw, str) and token_ids_raw:
            try:
                token_ids = json.loads(token_ids_raw)
            except json.JSONDecodeError:
                token_ids = [token_ids_raw]
        elif isinstance(token_ids_raw, list):
            token_ids = token_ids_raw

        if not token_ids:
            return {"error": "No CLOB token IDs found for market", "executed": False}

        # YES = index 0, NO = index 1
        token_id = token_ids[0] if side == "YES" else token_ids[1] if len(token_ids) > 1 else token_ids[0]

        # Get current price
        price = detail.get("yes_price", 0.5) if side == "YES" else detail.get("no_price", 0.5)
        size = amount_usdc / price if price > 0 else amount_usdc

        order_args = OrderArgs(
            price=price,
            size=size,
            side="BUY",
            token_id=token_id,
        )

        signed_order = client.create_order(order_args)
        resp = client.post_order(signed_order, OrderType.GTC)

        logger.info(
            f"ORDER PLACED: {side} on '{market_title}' | "
            f"Amount: {amount_usdc} USDC | Price: {price} | Reason: {reason}"
        )

        return {
            "executed": True,
            "market_id": market_id,
            "market_title": market_title,
            "side": side,
            "amount_usdc": amount_usdc,
            "price": price,
            "reason": reason,
            "order_response": str(resp) if resp else "submitted",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    except ImportError:
        logger.error("py-clob-client not installed properly")
        return {"error": "CLOB client not available", "executed": False}
    except Exception as e:
        logger.error(f"Order failed: {e}")
        return {"error": str(e), "executed": False}


def get_trade_history(wallet_address: str, limit: int = 50) -> list[dict[str, Any]]:
    """Get resolved trade history for learning."""
    try:
        def fetch():
            resp = SESSION.get(
                f"{GAMMA_API}/trades",
                params={"user": wallet_address, "limit": limit},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()

        data = _retry(fetch)
        if not isinstance(data, list):
            data = data.get("data", []) if isinstance(data, dict) else []

        trades = []
        for t in data:
            trades.append({
                "market_id": t.get("market", t.get("marketId", "")),
                "title": t.get("title", ""),
                "side": t.get("outcome", t.get("side", "")),
                "size": float(t.get("size", 0) or 0),
                "price": float(t.get("price", 0) or 0),
                "timestamp": t.get("timestamp", t.get("createdAt", "")),
            })

        return trades

    except Exception as e:
        logger.error(f"get_trade_history failed: {e}")
        return [{"error": str(e)}]
