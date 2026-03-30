"""News fetching for market intelligence."""

import logging
import time
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import quote_plus

import requests

logger = logging.getLogger("polybot.news")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Polybot/1.0"})
REQUEST_TIMEOUT = 10


def get_relevant_news(
    query: str,
    max_results: int = 5,
    news_api_key: str = "",
) -> list[dict[str, Any]]:
    """Fetch relevant news from multiple sources with fallback chain."""
    articles: list[dict[str, Any]] = []

    # Try GNews API first (free, 100 req/day)
    if news_api_key:
        articles = _try_gnews(query, max_results, news_api_key)
        if articles:
            return articles

    # Fallback: Google News RSS (no key needed)
    articles = _try_google_news_rss(query, max_results)
    if articles:
        return articles

    return [{"error": "No news sources available", "query": query}]


def _try_gnews(query: str, max_results: int, api_key: str) -> list[dict[str, Any]]:
    """Try GNews API."""
    try:
        url = "https://gnews.io/api/v4/search"
        params = {
            "q": query,
            "max": min(max_results, 10),
            "lang": "en",
            "token": api_key,
        }
        resp = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        return [
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "source": a.get("source", {}).get("name", "unknown"),
                "url": a.get("url", ""),
                "published": a.get("publishedAt", ""),
            }
            for a in data.get("articles", [])[:max_results]
        ]
    except Exception as e:
        logger.warning(f"GNews failed: {e}")
        return []


def _try_google_news_rss(query: str, max_results: int) -> list[dict[str, Any]]:
    """Fallback: Google News RSS feed (no API key needed)."""
    try:
        encoded_q = quote_plus(query)
        url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-US&gl=US&ceid=US:en"
        resp = SESSION.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        root = ET.fromstring(resp.content)
        items = root.findall(".//item")

        articles = []
        for item in items[:max_results]:
            title = item.findtext("title", "")
            # Google News titles often end with " - Source Name"
            source = ""
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0]
                source = parts[1] if len(parts) > 1 else ""

            articles.append({
                "title": title,
                "description": item.findtext("description", ""),
                "source": source,
                "url": item.findtext("link", ""),
                "published": item.findtext("pubDate", ""),
            })

        return articles
    except Exception as e:
        logger.warning(f"Google News RSS failed: {e}")
        return []
