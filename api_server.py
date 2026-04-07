"""Mini API to expose local bot data (trades, stats, balance history).

Run alongside the bot:
    pip install flask flask-cors
    screen -dmS polybot-api python api_server.py
"""

import json
import os
import sys

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

# Add parent dir so we can import tools.memory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.memory import get_stats, get_performance_by_category

app = Flask(__name__)
CORS(app)

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory")
API_KEY = os.environ.get("POLYBOT_API_KEY", "polybot-vps-secret-2024")


def check_auth() -> Response | None:
    key = request.headers.get("X-API-Key", "")
    if key != API_KEY:
        return jsonify({"error": "unauthorized"}), 401
    return None


@app.before_request
def auth_middleware():
    return check_auth()


@app.route("/api/trades")
def trades():
    path = os.path.join(MEMORY_DIR, "trades.json")
    try:
        with open(path) as f:
            data = json.load(f)
        # Return most recent first
        if isinstance(data, list):
            data.sort(key=lambda t: float(t.get("timestamp", 0)), reverse=True)
        return jsonify(data)
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify([])


@app.route("/api/stats")
def stats():
    return jsonify({
        "stats": get_stats(),
        "categories": get_performance_by_category(),
    })


@app.route("/api/balance-history")
def balance_history():
    path = os.path.join(MEMORY_DIR, "balance_history.json")
    try:
        with open(path) as f:
            data = json.load(f)
        return jsonify(data)
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify([])


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print(f"Polybot API starting on port 8080")
    print(f"API key: {API_KEY[:8]}...")
    app.run(host="0.0.0.0", port=8080)
