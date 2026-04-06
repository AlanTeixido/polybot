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
SIMMER_API = "https://api.simmer.markets/api/sdk"

# Polygon USDC.e contract
USDC_ADDRESS = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"
USDC_ABI = json.dumps([{
    "constant": True,
    "inputs": [{"name": "_owner", "type": "address"}],
    "name": "balanceOf",
    "outputs": [{"name": "balance", "type": "uint256"}],
    "type": "function",
}])
POLYGON_RPCS = [
    "https://polygon-bor-rpc.publicnode.com",
    "https://rpc.ankr.com/polygon",
    "https://polygon.llamarpc.com",
    "https://rpc-mainnet.matic.quiknode.pro",
]

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
    venue: str = "sim",
    simmer_api_key: str = "",
) -> list[dict[str, Any]]:
    """Fetch and filter available markets."""
    global _markets_cache

    # Return cached data if fresh
    if time.time() - _markets_cache["timestamp"] < CACHE_TTL and _markets_cache["data"]:
        logger.debug("Returning cached markets")
        return _markets_cache["data"]

    # --- Simmer venue ---
    if venue == "sim" and simmer_api_key:
        try:
            # Try /opportunities first (ranked by edge+liquidity+urgency)
            resp = SESSION.get(
                f"{SIMMER_API}/markets/opportunities",
                params={"limit": 50},
                headers={"Authorization": f"Bearer {simmer_api_key}"},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                # Fallback to /markets
                resp = SESSION.get(
                    f"{SIMMER_API}/markets",
                    params={"limit": 50},
                    headers={"Authorization": f"Bearer {simmer_api_key}"},
                    timeout=REQUEST_TIMEOUT,
                )
            resp.raise_for_status()
            data = resp.json()
            markets = data if isinstance(data, list) else data.get("data", data.get("markets", []))

            now = datetime.now(timezone.utc)
            result = []
            for m in markets:
                # Parse probability (Simmer returns float 0-1)
                prob_raw = m.get("probability", m.get("yes_probability", 0.5))
                if isinstance(prob_raw, str):
                    prob_raw = float(prob_raw)
                yes_prob = round(prob_raw * 100 if prob_raw <= 1 else prob_raw, 1)

                # Parse days to resolution
                end_date_str = m.get("endDate", m.get("end_date", ""))
                days_to_res = 30
                if end_date_str:
                    try:
                        end_date = datetime.fromisoformat(str(end_date_str).replace("Z", "+00:00"))
                        days_to_res = max((end_date - now).total_seconds() / 86400, 0)
                    except (ValueError, TypeError):
                        pass

                # Classify market by researchability
                title_lower = (m.get("question", m.get("title", "")) or "").lower()
                cat_lower = (m.get("category", "") or "").lower()

                # --- TIER 1: High edge (verifiable with data/news) ---
                tier1_kw = [
                    "president", "election", "vote", "poll", "nominee", "primary",
                    "congress", "senate", "governor", "mayor", "parliament",
                    "gdp", "fed", "interest rate", "inflation", "unemployment",
                    "oscar", "grammy", "emmy", "award", "nobel",
                    "launch", "release", "ipo", "merger", "acquisition",
                    "war", "ceasefire", "peace", "treaty", "sanctions",
                    "indictment", "trial", "verdict", "ruling",
                ]
                tier1_cat = ["politics", "economics", "science", "entertainment"]

                # --- TIER 2: Medium edge (sports with clear standings) ---
                tier2_kw = [
                    "champion", "qualify", "playoffs", "final", "world cup",
                    "super bowl", "world series", "stanley cup",
                    "premier league", "la liga", "serie a", "bundesliga",
                    "nba", "nfl", "mlb", "nhl",
                    "relegation", "promotion", "seed",
                ]

                # --- TIER 3: Medium-low edge (match outcomes with context) ---
                tier3_kw = [
                    "spread", "o/u", "over/under", "points o/u",
                ]

                # --- TIER 1 BONUS: Weather markets (91.5% WR with forecast data) ---
                weather_kw = [
                    "temperature", "highest temp", "lowest temp", "°c", "°f",
                    "weather", "rain", "snow", "wind",
                ]
                is_weather = any(kw in title_lower for kw in weather_kw)

                # --- NOISE: No edge (crypto FDV, esports maps, exact crypto price) ---
                noise_kw = [
                    "exact price", "close above", "close below",
                    "will reach", "will hit",
                    "fdv above", "fdv below", "market cap above",
                    "map 2 winner", "map 3 winner", "map 1 winner",
                    "set 1 winner", "set 2 winner", "set 3 winner",
                ]

                # Determine tier and score
                if any(kw in title_lower for kw in noise_kw):
                    tier = "noise"
                    quick_score = -1.0  # Will be filtered out
                elif is_weather:
                    tier = "tier1-weather"
                    quick_score = 3.0 + abs(yes_prob - 50) / 50  # Highest priority
                elif any(kw in title_lower for kw in tier1_kw) or any(c in cat_lower for c in tier1_cat):
                    tier = "tier1"
                    quick_score = 2.0 + abs(yes_prob - 50) / 50
                elif any(kw in title_lower for kw in tier2_kw):
                    tier = "tier2"
                    quick_score = 1.0 + abs(yes_prob - 50) / 50
                elif any(kw in title_lower for kw in tier3_kw):
                    tier = "tier3"
                    quick_score = 0.3 + abs(yes_prob - 50) / 50
                else:
                    # Unknown markets: only include if probability is very skewed (>75% or <25%)
                    tier = "other"
                    skew = abs(yes_prob - 50)
                    quick_score = (skew / 50) - 0.5 if skew > 25 else -1.0

                # Near-resolution bonus (applies to all tiers)
                if days_to_res <= 3:
                    quick_score += 0.5
                elif days_to_res <= 7:
                    quick_score += 0.3

                result.append({
                    "id": m.get("id", m.get("market_id", "")),
                    "title": m.get("question", m.get("title", "")),
                    "yes_probability": yes_prob,
                    "volume": "sim",
                    "liquidity": "sim",
                    "end_date": end_date_str,
                    "days_to_resolution": round(days_to_res, 1),
                    "category": m.get("category", "unknown"),
                    "condition_id": m.get("conditionId", m.get("condition_id", "")),
                    "quick_score": round(quick_score, 3),
                    "tier": tier,
                    "venue": "sim",
                })

            # Filter out noise. Keep >95% only if resolving within 3 days (near-resolution strategy)
            result = [
                m for m in result
                if m["quick_score"] > 0
                and not (max(m["yes_probability"], 100 - m["yes_probability"]) > 95 and m["days_to_resolution"] > 3)
            ]
            result.sort(key=lambda x: x["quick_score"], reverse=True)
            result = result[:15]
            _markets_cache = {"data": result, "timestamp": time.time()}
            return result
        except Exception as e:
            logger.error(f"Simmer get_markets failed: {e}")
            return [{"error": str(e)}]

    # --- Polymarket venue ---
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

            # Skip near-resolved markets unless resolving within 3 days
            if max(yes_prob, 100 - yes_prob) > 95 and days_to_res > 3:
                continue

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


def get_market_detail(
    market_id: str,
    venue: str = "sim",
    simmer_api_key: str = "",
) -> dict[str, Any]:
    """Get detailed market info."""
    # --- Simmer venue ---
    if venue == "sim" and simmer_api_key:
        try:
            resp = SESSION.get(
                f"{SIMMER_API}/markets/{market_id}",
                headers={"Authorization": f"Bearer {simmer_api_key}"},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            m = resp.json()

            prob = float(m.get("probability", m.get("yes_probability", 0.5)) or 0.5)
            if prob > 1:
                prob = prob / 100
            yes_price = prob
            no_price = 1 - prob

            return {
                "market_id": market_id,
                "title": m.get("question", m.get("title", "")),
                "description": (m.get("description", "") or "")[:300],
                "category": m.get("category", "unknown"),
                "yes_price": round(yes_price, 4),
                "no_price": round(no_price, 4),
                "volume": float(m.get("volume", 0) or 0),
                "liquidity": float(m.get("liquidity", 0) or 0),
                "end_date": m.get("endDate", m.get("end_date", "")),
                "outcomes": m.get("outcomes", ["Yes", "No"]),
                "condition_id": m.get("conditionId", m.get("condition_id", "")),
                "clob_token_ids": "",
            }
        except Exception as e:
            logger.error(f"Simmer get_market_detail failed for {market_id}: {e}")
            return {"error": str(e), "market_id": market_id}

    # --- Polymarket venue ---
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


def get_balance(
    wallet_address: str,
    custom_rpc: str = "",
    venue: str = "sim",
    simmer_api_key: str = "",
) -> dict[str, Any]:
    """Get balance. Simmer venue returns $SIM, Polymarket venue returns USDC."""
    # --- Simmer venue ---
    if venue == "sim" and simmer_api_key:
        try:
            resp = SESSION.get(
                f"{SIMMER_API}/agents/me",
                headers={"Authorization": f"Bearer {simmer_api_key}"},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            balance = float(data.get("balance", data.get("sim_balance", 0)) or 0)

            alert = ""
            if balance < 5:
                alert = "CRITICAL: Balance below 5 $SIM. Bot should stop trading."
            elif balance < 10:
                alert = "WARNING: Balance below 10 $SIM. Reduce position sizes."

            return {
                "balance_usdc": round(balance, 2),
                "wallet": wallet_address,
                "venue": "sim",
                "alert": alert,
            }
        except Exception as e:
            logger.warning(f"Simmer get_balance failed: {e}")
            return {"balance_usdc": 0, "error": str(e), "wallet": wallet_address}

    # --- Polymarket venue (multi-RPC fallback) ---
    rpcs = []
    if custom_rpc:
        rpcs.append(custom_rpc)
    rpcs.extend(POLYGON_RPCS)

    checksum_wallet = Web3.to_checksum_address(wallet_address)
    checksum_usdc = Web3.to_checksum_address(USDC_ADDRESS)
    abi = json.loads(USDC_ABI)

    for rpc_url in rpcs:
        try:
            w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 10}))
            usdc = w3.eth.contract(address=checksum_usdc, abi=abi)
            raw_balance = usdc.functions.balanceOf(checksum_wallet).call()
            balance = raw_balance / 1e6  # USDC has 6 decimals

            alert = ""
            if balance < 5:
                alert = "CRITICAL: Balance below 5 USDC. Bot should stop trading."
            elif balance < 10:
                alert = "WARNING: Balance below 10 USDC. Reduce position sizes."

            return {
                "balance_usdc": round(balance, 2),
                "wallet": wallet_address,
                "venue": "polymarket",
                "alert": alert,
            }
        except Exception as e:
            logger.warning(f"RPC {rpc_url} failed: {e}")

    logger.error(f"All RPCs failed for get_balance({wallet_address})")
    return {"balance_usdc": 0, "error": "All RPCs failed", "wallet": wallet_address}


def get_positions(
    wallet_address: str,
    venue: str = "sim",
    simmer_api_key: str = "",
) -> list[dict[str, Any]]:
    """Get open positions."""
    # --- Simmer venue ---
    if venue == "sim" and simmer_api_key:
        try:
            resp = SESSION.get(
                f"{SIMMER_API}/positions",
                headers={"Authorization": f"Bearer {simmer_api_key}"},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                data = data.get("data", data.get("positions", []))

            positions = []
            for p in data:
                size = float(p.get("size", p.get("amount", 0)) or 0)
                if size <= 0:
                    continue
                current_price = float(p.get("currentPrice", p.get("current_price", p.get("price", 0))) or 0)
                avg_price = float(p.get("avgPrice", p.get("avg_price", p.get("averagePrice", 0))) or 0)
                pnl = (current_price - avg_price) * size if avg_price > 0 else 0
                positions.append({
                    "market_id": p.get("market_id", p.get("market", "")),
                    "title": p.get("title", p.get("question", "")),
                    "side": p.get("side", p.get("outcome", "")),
                    "size": size,
                    "avg_price": round(avg_price, 4),
                    "current_price": round(current_price, 4),
                    "unrealized_pnl": round(pnl, 4),
                    "in_loss": pnl < -0.5,
                })
            return positions
        except Exception as e:
            logger.warning(f"Simmer get_positions failed: {e}")
            return []

    # --- Polymarket venue ---
    try:
        def fetch():
            resp = SESSION.get(
                "https://data-api.polymarket.com/positions",
                params={"user": wallet_address, "limit": 50, "offset": 0},
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
        logger.warning(f"get_positions failed: {e}")
        return []


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
    venue: str = "sim",
    simmer_api_key: str = "",
) -> dict[str, Any]:
    """Place an order via Simmer (sim) or Polymarket CLOB."""
    # --- Dry run mode (extra safety layer, works on any venue) ---
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

    # --- Breakeven gate + max entry price (Polymarket only, SIM is unrestricted) ---
    if venue != "sim":
        detail_for_check = get_market_detail(market_id, venue=venue, simmer_api_key=simmer_api_key)
        if "error" not in detail_for_check:
            entry_price = detail_for_check.get("yes_price", 0.5) if side == "YES" else detail_for_check.get("no_price", 0.5)
            max_entry = 0.75
            if entry_price > max_entry:
                return {
                    "error": f"Entry price {entry_price:.2f} too high. Max: {max_entry}. "
                             f"Need {entry_price:.0%} WR to break even — seek cheaper side.",
                    "executed": False,
                }

    # Block duplicate and conflicting positions
    positions = get_positions(wallet_address, venue=venue, simmer_api_key=simmer_api_key)
    for pos in positions:
        if isinstance(pos, dict) and pos.get("market_id") == market_id:
            existing_side = pos.get("side", "").upper()
            if existing_side and existing_side != side:
                return {
                    "error": f"Conflicting: already have {existing_side}. Cannot bet {side}.",
                    "executed": False,
                }
            if existing_side == side:
                return {
                    "error": f"Already have {existing_side} position in this market. No stacking.",
                    "executed": False,
                }

    # --- Spread check (Polymarket only — SIM has no real spread) ---
    if venue != "sim":
        detail = get_market_detail(market_id, venue=venue, simmer_api_key=simmer_api_key)
        if "error" not in detail:
            yes_p = detail.get("yes_price", 0.5)
            no_p = detail.get("no_price", 0.5)
            spread = abs(1 - yes_p - no_p)
            max_spread = 0.05
            if spread > max_spread:
                return {
                    "error": f"Spread too high: {spread:.1%}. Max allowed: {max_spread:.1%}",
                    "executed": False,
                }

    # --- Simmer venue ---
    if venue == "sim" and simmer_api_key:
        try:
            resp = SESSION.post(
                f"{SIMMER_API}/trade",
                headers={"Authorization": f"Bearer {simmer_api_key}"},
                json={
                    "market_id": market_id,
                    "side": side.lower(),
                    "amount": amount_usdc,
                    "venue": "sim",
                    "reasoning": reason,
                    "source": "sdk:polybot",
                },
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            result = resp.json()

            logger.info(
                f"SIM ORDER: {side} on '{market_title}' | "
                f"Amount: {amount_usdc} $SIM | Reason: {reason}"
            )

            return {
                "executed": True,
                "venue": "sim",
                "market_id": market_id,
                "market_title": market_title,
                "side": side,
                "amount_usdc": amount_usdc,
                "reason": reason,
                "order_response": result,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            logger.error(f"Simmer order failed: {e}")
            return {"error": str(e), "executed": False, "venue": "sim"}

    # --- Polymarket venue ---
    # Check balance first
    balance_info = get_balance(wallet_address, venue="polymarket")
    current_balance = balance_info.get("balance_usdc", 0)
    if current_balance < amount_usdc:
        return {
            "error": f"Insufficient balance: {current_balance} USDC < {amount_usdc} USDC",
            "executed": False,
        }

    # Check for duplicate position
    positions = get_positions(wallet_address, venue="polymarket")
    for pos in positions:
        if isinstance(pos, dict) and pos.get("market_id") == market_id:
            return {
                "error": f"Already have position in market {market_id}",
                "executed": False,
                "existing_position": pos,
            }

    try:
        from py_clob_client.clob_types import OrderArgs, OrderType

        client = get_clob_client(private_key, wallet_address, api_key, api_secret, api_passphrase)

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

        token_id = token_ids[0] if side == "YES" else token_ids[1] if len(token_ids) > 1 else token_ids[0]
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
            "venue": "polymarket",
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
