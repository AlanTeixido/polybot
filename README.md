# Polybot - Autonomous Trading Agent for Polymarket

Autonomous LLM-powered trading bot that runs 24/7, making profitable trades on Polymarket using Claude Sonnet for reasoning.

## Architecture

Multi-turn agent loop inspired by CashClaw:
1. Agent reasons about current market state
2. Calls tools (markets, whale tracking, news, memory)
3. Reasons again with tool results
4. Executes best trades with explicit justification
5. Learns from every outcome

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and edit config
cp config.example.json config.json
# Edit config.json with your real keys

# Run
python3 agent.py
```

## Production (VPS)

```bash
# With screen (recommended)
screen -S polybot python3 agent.py

# Or with nohup
nohup python3 agent.py > /dev/null 2>&1 &
```

## Configuration

Edit `config.json`:

| Key | Description |
|-----|-------------|
| `anthropic_api_key` | Claude API key |
| `private_key` | Polygon wallet private key |
| `max_bet_usdc` | Maximum bet per trade |
| `max_positions` | Maximum simultaneous positions |
| `cycle_interval_seconds` | Time between trading cycles |
| `telegram_bot_token` | Optional: Telegram notifications |
| `whale_wallets` | Wallet addresses to track |

## Strategies

- **Near-resolution**: Markets at 92%+ probability resolving within 7 days
- **Whale copytrading**: Follow proven wallets, strong signal when 2+ agree
- **Information edge**: News not yet reflected in market prices

## Risk Management

- Max 20% of balance per trade
- Auto-halves bet size if balance drops 30% in 7 days
- Stops after 5 consecutive losses
- Stops if balance falls below 5 USDC
- Diversification: max 3 trades per cycle, different categories
