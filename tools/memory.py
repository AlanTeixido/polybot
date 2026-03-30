"""BM25-based memory with temporal decay for trade learning."""

import json
import math
import os
import time
import tempfile
import shutil
import logging
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger("polybot.memory")

MEMORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory")
KNOWLEDGE_FILE = os.path.join(MEMORY_DIR, "knowledge.json")
TRADES_FILE = os.path.join(MEMORY_DIR, "trades.json")
MAX_KNOWLEDGE = 200
MAX_TRADES = 500
HALF_LIFE_DAYS = 30


def _ensure_dir() -> None:
    os.makedirs(MEMORY_DIR, exist_ok=True)


def _load_json(path: str) -> list[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_json(path: str, data: list[dict]) -> None:
    """Atomic write: write to temp file then replace."""
    _ensure_dir()
    fd, tmp = tempfile.mkstemp(dir=MEMORY_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
        shutil.move(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _temporal_decay(timestamp: str) -> float:
    """Exponential decay: score * e^(-ln(2)/half_life * age_days)."""
    try:
        age_seconds = time.time() - float(timestamp)
        age_days = max(age_seconds / 86400, 0)
        return math.exp(-math.log(2) / HALF_LIFE_DAYS * age_days)
    except (ValueError, TypeError):
        return 0.5


def memory_search(query: str, top_k: int = 5) -> list[dict]:
    """Search knowledge + trades using BM25 with temporal decay."""
    knowledge = _load_json(KNOWLEDGE_FILE)
    trades = _load_json(TRADES_FILE)

    docs: list[dict] = []
    for k in knowledge:
        text = f"{k.get('insight', '')} {' '.join(k.get('tags', []))}"
        docs.append({"text": text, "source": "knowledge", "data": k})
    for t in trades:
        text = f"{t.get('title', '')} {t.get('reason', '')} {t.get('category', '')}"
        docs.append({"text": text, "source": "trade", "data": t})

    if not docs:
        return []

    tokenized = [d["text"].lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    query_tokens = query.lower().split()
    raw_scores = bm25.get_scores(query_tokens)

    scored = []
    for i, doc in enumerate(docs):
        ts = doc["data"].get("timestamp", str(time.time()))
        decay = _temporal_decay(ts)
        final_score = float(raw_scores[i]) * decay
        scored.append({"score": round(final_score, 4), **doc["data"]})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def save_knowledge(insight: str, tags: list[str] | None = None, source: str = "agent") -> dict:
    """Save a learning insight with tags."""
    knowledge = _load_json(KNOWLEDGE_FILE)

    entry = {
        "insight": insight,
        "tags": tags or [],
        "source": source,
        "timestamp": str(time.time()),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    knowledge.append(entry)

    # Trim oldest entries if over limit
    if len(knowledge) > MAX_KNOWLEDGE:
        knowledge.sort(key=lambda x: float(x.get("timestamp", 0)))
        knowledge = knowledge[-MAX_KNOWLEDGE:]

    _save_json(KNOWLEDGE_FILE, knowledge)
    logger.info(f"Knowledge saved: {insight[:80]}...")
    return {"status": "saved", "total_entries": len(knowledge)}


def save_trade_result(
    market_id: str,
    title: str,
    side: str,
    amount_usdc: float,
    pnl: float,
    reason: str,
    category: str,
    resolved: bool,
) -> dict:
    """Save a trade result for learning."""
    trades = _load_json(TRADES_FILE)

    entry = {
        "market_id": market_id,
        "title": title,
        "side": side,
        "amount_usdc": amount_usdc,
        "pnl": pnl,
        "reason": reason,
        "category": category,
        "resolved": resolved,
        "won": pnl > 0 if resolved else None,
        "timestamp": str(time.time()),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    trades.append(entry)

    if len(trades) > MAX_TRADES:
        trades.sort(key=lambda x: float(x.get("timestamp", 0)))
        trades = trades[-MAX_TRADES:]

    _save_json(TRADES_FILE, trades)
    logger.info(f"Trade saved: {title[:50]} | PnL: {pnl}")
    return {"status": "saved", "total_trades": len(trades)}


def get_stats() -> dict:
    """Get overall trading statistics."""
    trades = _load_json(TRADES_FILE)
    resolved = [t for t in trades if t.get("resolved")]

    if not resolved:
        return {
            "total_trades": len(trades),
            "resolved_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "current_streak": 0,
            "best_trade": None,
            "worst_trade": None,
        }

    wins = [t for t in resolved if t.get("won")]
    pnls = [t.get("pnl", 0) for t in resolved]

    # Current streak
    streak = 0
    for t in reversed(resolved):
        if t.get("won"):
            streak += 1
        else:
            if streak == 0:
                streak = -1
                for t2 in reversed(resolved):
                    if not t2.get("won"):
                        streak -= 1
                    else:
                        break
                streak += 1  # Correct off-by-one
            break

    return {
        "total_trades": len(trades),
        "resolved_trades": len(resolved),
        "wins": len(wins),
        "losses": len(resolved) - len(wins),
        "win_rate": round(len(wins) / len(resolved) * 100, 1) if resolved else 0,
        "total_pnl": round(sum(pnls), 2),
        "current_streak": streak,
        "best_trade": max(pnls) if pnls else 0,
        "worst_trade": min(pnls) if pnls else 0,
    }


def get_performance_by_category() -> dict:
    """Get win rate and PnL broken down by market category."""
    trades = _load_json(TRADES_FILE)
    resolved = [t for t in trades if t.get("resolved")]

    categories: dict[str, dict[str, Any]] = {}
    for t in resolved:
        cat = t.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"wins": 0, "losses": 0, "pnl": 0}
        if t.get("won"):
            categories[cat]["wins"] += 1
        else:
            categories[cat]["losses"] += 1
        categories[cat]["pnl"] += t.get("pnl", 0)

    result = {}
    for cat, data in categories.items():
        total = data["wins"] + data["losses"]
        result[cat] = {
            "trades": total,
            "wins": data["wins"],
            "losses": data["losses"],
            "win_rate": round(data["wins"] / total * 100, 1) if total > 0 else 0,
            "pnl": round(data["pnl"], 2),
        }

    return result
