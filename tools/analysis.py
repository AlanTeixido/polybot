"""Market analysis, whale tracking, and opportunity detection."""

import logging
import time
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger("polybot.analysis")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Polybot/1.0"})
REQUEST_TIMEOUT = 15

# Polymarket Gamma API
GAMMA_API = "https://gamma-api.polymarket.com"
SIMMER_API = "https://api.simmer.markets/api/sdk"


def analyze_market(
    market_id: str,
    market_title: str = "",
    venue: str = "sim",
    simmer_api_key: str = "",
) -> dict[str, Any]:
    """Score a market for opportunity and risk."""
    # --- Simmer venue: use context endpoint ---
    if venue == "sim" and simmer_api_key:
        try:
            resp = SESSION.get(
                f"{SIMMER_API}/context/{market_id}",
                headers={"Authorization": f"Bearer {simmer_api_key}"},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            ctx = resp.json()

            # Map Simmer context to our analysis format
            yes_price = float(ctx.get("probability", ctx.get("yes_price", 0.5)) or 0.5)
            if yes_price > 1:
                yes_price = yes_price / 100
            no_price = 1 - yes_price
            max_price = max(yes_price, no_price)

            return {
                "market_id": market_id,
                "title": ctx.get("question", ctx.get("title", market_title)),
                "category": ctx.get("category", "unknown"),
                "yes_price": round(yes_price, 4),
                "no_price": round(no_price, 4),
                "volume": float(ctx.get("volume", 0) or 0),
                "liquidity": float(ctx.get("liquidity", 0) or 0),
                "days_to_resolution": float(ctx.get("days_to_resolution", 30)),
                "opportunity_score": float(ctx.get("opportunity_score", 5)),
                "risk_score": float(ctx.get("risk_score", 5)),
                "recommendation": ctx.get("recommendation", "CONSIDER"),
                "favored_side": "YES" if yes_price > no_price else "NO",
                "favored_probability": round(max_price * 100, 1),
                "edge": ctx.get("edge"),
                "slippage": ctx.get("slippage"),
                "warnings": ctx.get("warnings", []),
            }
        except Exception as e:
            logger.warning(f"Simmer context failed for {market_id}: {e}")
            # Fallback: return CONSIDER with basic info
            return {
                "market_id": market_id,
                "title": market_title,
                "category": "unknown",
                "yes_price": 0.5,
                "no_price": 0.5,
                "volume": 0,
                "liquidity": 0,
                "days_to_resolution": 30,
                "opportunity_score": 5.0,
                "risk_score": 5.0,
                "recommendation": "CONSIDER",
                "favored_side": "YES",
                "favored_probability": 50.0,
            }

    # --- Polymarket venue: use Gamma API ---
    try:
        resp = SESSION.get(
            f"{GAMMA_API}/markets/{market_id}", timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        market = resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch market {market_id}: {e}")
        return {"error": str(e), "market_id": market_id}

    now = datetime.now(timezone.utc)
    end_date_str = market.get("endDate") or market.get("end_date_iso", "")
    days_to_resolution = 999
    if end_date_str:
        try:
            end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            days_to_resolution = max((end_date - now).total_seconds() / 86400, 0)
        except (ValueError, TypeError):
            pass

    volume = float(market.get("volume", 0) or 0)
    liquidity = float(market.get("liquidity", 0) or 0)

    # Parse outcomes/prices
    outcome_prices = market.get("outcomePrices", "")
    prices = []
    if isinstance(outcome_prices, str) and outcome_prices:
        try:
            import json
            prices = [float(p) for p in json.loads(outcome_prices)]
        except (ValueError, json.JSONDecodeError):
            prices = []
    elif isinstance(outcome_prices, list):
        prices = [float(p) for p in outcome_prices]

    yes_price = prices[0] if prices else 0.5
    no_price = prices[1] if len(prices) > 1 else 1 - yes_price

    # --- Opportunity Score (0-10) ---
    opp_score = 0.0
    if days_to_resolution <= 3:
        opp_score += 4.0
    elif days_to_resolution <= 7:
        opp_score += 3.0
    elif days_to_resolution <= 14:
        opp_score += 1.5

    max_price = max(yes_price, no_price)
    if max_price >= 0.92:
        opp_score += 3.0
    elif max_price >= 0.85:
        opp_score += 1.5

    if volume >= 50000:
        opp_score += 2.0
    elif volume >= 10000:
        opp_score += 1.5
    elif volume >= 1000:
        opp_score += 1.0

    if liquidity >= 10000:
        opp_score += 1.0
    elif liquidity >= 1000:
        opp_score += 0.5

    opp_score = min(opp_score, 10.0)

    # --- Risk Score (0-10, higher = riskier) ---
    risk_score = 0.0
    if days_to_resolution > 30:
        risk_score += 3.0
    elif days_to_resolution > 14:
        risk_score += 1.5
    if volume < 500:
        risk_score += 3.0
    elif volume < 1000:
        risk_score += 1.5
    if liquidity < 500:
        risk_score += 2.0
    if 0.35 < max_price < 0.65:
        risk_score += 2.0
    risk_score = min(risk_score, 10.0)

    rec = "SKIP"
    favored_side = "YES" if yes_price > no_price else "NO"
    if opp_score >= 7 and risk_score <= 4:
        rec = f"BUY {favored_side}"
    elif opp_score >= 5 and risk_score <= 5:
        rec = "CONSIDER"

    return {
        "market_id": market_id,
        "title": market.get("question", market_title),
        "category": market.get("category", "unknown"),
        "yes_price": round(yes_price, 4),
        "no_price": round(no_price, 4),
        "volume": volume,
        "liquidity": liquidity,
        "days_to_resolution": round(days_to_resolution, 1),
        "opportunity_score": round(opp_score, 1),
        "risk_score": round(risk_score, 1),
        "recommendation": rec,
        "favored_side": favored_side,
        "favored_probability": round(max_price * 100, 1),
    }


def get_whale_activity(
    whale_wallets: list[str], hours_back: int = 24
) -> dict[str, Any]:
    """Track recent activity from whale wallets on Polymarket."""
    if not whale_wallets:
        return {"wallets_checked": 0, "activities": [], "strong_signals": []}

    activities: list[dict] = []
    market_consensus: dict[str, dict[str, Any]] = {}

    for wallet in whale_wallets:
        wallet_short = wallet[:10] if len(wallet) > 10 else wallet
        try:
            # Use Polymarket's activity endpoint
            resp = SESSION.get(
                f"{GAMMA_API}/activity",
                params={"user": wallet, "limit": 20},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                logger.debug(f"No activity data for {wallet_short}")
                continue

            data = resp.json()
            if not isinstance(data, list):
                data = data.get("data", []) if isinstance(data, dict) else []

            cutoff = time.time() - (hours_back * 3600)

            for entry in data:
                ts_str = entry.get("timestamp") or entry.get("createdAt", "")
                try:
                    if isinstance(ts_str, (int, float)):
                        ts = float(ts_str)
                    else:
                        ts = datetime.fromisoformat(
                            ts_str.replace("Z", "+00:00")
                        ).timestamp()
                except (ValueError, TypeError):
                    continue

                if ts < cutoff:
                    continue

                market_id = entry.get("market", entry.get("marketId", ""))
                side = entry.get("side", entry.get("outcome", ""))
                size = float(entry.get("size", entry.get("amount", 0)) or 0)

                if not market_id:
                    continue

                act = {
                    "wallet": wallet_short,
                    "market_id": market_id,
                    "side": str(side).upper(),
                    "size_usdc": size,
                    "timestamp": ts,
                    "title": entry.get("title", entry.get("question", "")),
                }
                activities.append(act)

                # Track consensus
                key = f"{market_id}_{act['side']}"
                if key not in market_consensus:
                    market_consensus[key] = {
                        "market_id": market_id,
                        "side": act["side"],
                        "title": act["title"],
                        "wallets": [],
                        "total_size": 0,
                    }
                market_consensus[key]["wallets"].append(wallet_short)
                market_consensus[key]["total_size"] += size

        except Exception as e:
            logger.warning(f"Whale tracking failed for {wallet_short}: {e}")

    # Strong signals: 2+ whales on the same side
    strong_signals = [
        {
            **v,
            "whale_count": len(v["wallets"]),
            "total_size": round(v["total_size"], 2),
        }
        for v in market_consensus.values()
        if len(v["wallets"]) >= 2
    ]
    strong_signals.sort(key=lambda x: x["whale_count"], reverse=True)

    activities.sort(key=lambda x: x.get("size_usdc", 0), reverse=True)

    return {
        "wallets_checked": len(whale_wallets),
        "total_activities": len(activities),
        "activities": activities[:20],
        "strong_signals": strong_signals,
    }


def calculate_edge(my_estimate: float, market_probability: float) -> dict[str, Any]:
    """Calculate edge between own estimate and market price."""
    edge = my_estimate - market_probability
    abs_edge = abs(edge)

    should_trade = abs_edge >= 5.0
    side = "YES" if edge > 0 else "NO"

    confidence = "low"
    if abs_edge >= 15:
        confidence = "high"
    elif abs_edge >= 8:
        confidence = "medium"

    return {
        "my_estimate": my_estimate,
        "market_probability": market_probability,
        "edge": round(edge, 1),
        "abs_edge": round(abs_edge, 1),
        "recommended_side": side,
        "should_trade": should_trade,
        "confidence": confidence,
    }


def get_simmer_context(
    market_id: str, my_probability: float, simmer_api_key: str = ""
) -> dict[str, Any] | None:
    """Get Simmer context for a market: real edge, slippage, recommendation."""
    if not simmer_api_key:
        return None
    try:
        resp = SESSION.get(
            f"https://api.simmer.markets/api/sdk/context/{market_id}",
            params={"my_probability": my_probability},
            headers={"Authorization": f"Bearer {simmer_api_key}"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Simmer context failed for {market_id}: {e}")
        return None


def detect_opportunities(markets: list[dict]) -> list[dict]:
    """Score and rank a list of markets for opportunities."""
    scored: list[dict] = []

    for m in markets:
        market_id = m.get("id", m.get("market_id", ""))
        if not market_id:
            continue

        analysis = analyze_market(market_id, m.get("title", ""))
        if "error" in analysis:
            continue

        scored.append(analysis)

    scored.sort(key=lambda x: x.get("opportunity_score", 0), reverse=True)
    return scored[:10]
