# PA_MCP — Personal Analyst MCP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Full-stack A-share quantitative trading MCP server. Let AI agents (Claude, etc.) query Chinese stock market data, run technical analysis, execute trading strategies, and perform backtesting — all through natural language.

## Architecture

```
+==========================================================+
|  MCP Interface (FastMCP + HTTP/SSE + stdio)              |
|  20+ Tools: data | screener | analysis | review | agent   |
+==========================================================+
|  AI Agent Decision (asyncio, dual-mode fast/deep)        |
+==========================================================+
|  Hard RiskGuard (NON-BYPASSABLE position/risk controls)  |
+==========================================================+
|  Engine (indicators + 20 strategies + backtest)          |
+==========================================================+
|  Data (DuckDB + AKShare + pre-computation cache)         |
+==========================================================+
```

## Features

- **Market Data**: Real-time quotes, historical K-line (daily/weekly/monthly/minute), index data
- **Screening**: Limit-up scan, volume surge, breakout detection, event tracking
- **Analysis**: Technical indicators (150+), fundamentals, sentiment, capital flow
- **Review**: Daily limit-up review, dragon-tiger board analysis, market sentiment
- **Strategy Engine**: 20 built-in strategies across 8 categories (trend/swing/board/value/grid/reversal/event/momentum)
- **Backtest**: Vectorized backtesting with realistic A-share constraints (T+1, limit-up buyability, slippage)
- **Agent Analysis**: Multi-analyst research reports (fast mode <15s, deep mode <60s)
- **RiskGuard**: Non-bypassable position limits, loss circuit breakers, systemic risk early warning
- **Events**: Block trades, lockup expiry, insider trades, pledge ratios, institutional visits

## Quick Start

### Prerequisites

- Python 3.12+
- Docker & Docker Compose (for containerized deployment)
- TA-Lib C library (auto-installed in Docker)

### Local Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Copy default config
cp config/default.yaml config/local.yaml
export PA_MCP_CONFIG=config/local.yaml

# Run in stdio mode (for Claude Desktop)
python -m pa_mcp.server

# Run in HTTP mode
export PA_MCP_SERVER__TRANSPORT=http
python -m pa_mcp.server
```

### Docker Deployment

```bash
# Build and start
docker compose up -d

# Check health
curl http://localhost:8080/health

# View logs
docker compose logs -f pa-mcp-server
```

## Claude Desktop Configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pa-mcp": {
      "command": "python",
      "args": ["-m", "pa_mcp.server"],
      "cwd": "/path/to/pa-mcp",
      "env": {
        "PA_MCP_CONFIG": "config/default.yaml"
      }
    }
  }
}
```

## MCP Tools

### Market Data
| Tool | Description |
|------|-------------|
| `get_realtime_quote` | Real-time quote with 5-level depth |
| `get_kline` | Historical K-line (daily/weekly/monthly/minute) |
| `get_market_overview` | Market breadth, turnover, northbound flow |
| `search_stock` | Search stocks by name or code |
| `get_stock_info` | Stock basic info |

### Screener
| Tool | Description |
|------|-------------|
| `scan_limit_up` | Limit-up stocks with seal strength |
| `scan_volume_surge` | Abnormal volume detection |
| `get_major_events` | Block trades, lockup, insider trades, pledge |

### Review
| Tool | Description |
|------|-------------|
| `review_daily_limit_up` | Daily limit-up review report |
| `review_dragon_tiger` | Dragon-tiger board seat analysis |
| `get_market_sentiment` | Market mood + position suggestion |

### Strategy
| Tool | Description |
|------|-------------|
| `list_strategies` | Available strategies by category |

## Configuration

See [config/default.yaml](config/default.yaml) for all options.

Key environment variables:
| Variable | Default | Description |
|----------|---------|-------------|
| `PA_MCP_SERVER__TRANSPORT` | `stdio` | `stdio` or `http` |
| `PA_MCP_SERVER__HTTP_PORT` | `8080` | HTTP listen port |
| `PA_MCP_DATABASE__PATH` | `data/pa_mcp.duckdb` | Database path |
| `PA_MCP_CACHE__BACKEND` | `memory` | `memory` / `redis` / `none` |
| `PA_MCP_LLM__MODEL_FAST` | `claude-sonnet-5` | Fast analysis model |
| `PA_MCP_LLM__MODEL_DEEP` | `claude-opus-5` | Deep analysis model |

## Project Structure

```
pa-mcp/
├── src/pa_mcp/
│   ├── server.py           # MCP Server entry
│   ├── config.py           # Pydantic Settings
│   ├── data/               # Data layer (DuckDB, cache, sources, quality)
│   ├── engine/             # Engine layer (indicators, strategies, market state)
│   ├── backtest/           # Backtest engine (vectorbt + DuckDB SQL)
│   ├── risk/               # Hard RiskGuard (non-bypassable)
│   ├── agent/              # AI Agent decision layer
│   ├── info/               # News, sentiment, event processing
│   └── tools/              # MCP tool definitions
├── config/default.yaml     # Default configuration
├── docker-compose.yml      # Docker deployment
├── Dockerfile              # Multi-stage build
└── docs/superpowers/specs/ # Design specifications
```

## Data Sources

- **AKShare** (primary): Real-time quotes, daily K-line, financials, dragon-tiger, events
- **TickFlow** (fallback): Real-time quotes, daily K-line

## A-Share Reality Constraints

The backtest engine models real A-share trading constraints:
- T+1 settlement (buy today, sell tomorrow)
- Limit-up buyability probability (strong seals = unbuyable)
- Limit-down sellability (consecutive limit-downs = near-impossible exits)
- Liquidity constraints (<5% of daily volume per order)
- Differentiated slippage (large-cap 0.1%, small-cap 0.3%, ChiNext 0.5%)
- Suspension handling (skip suspended days)
- Commission + stamp duty + transfer fee per exchange

## Disclaimer

**This is a research tool, not investment advice.**

- All analysis is for educational and reference purposes only
- Past performance does not guarantee future results
- Trading involves risk of loss; never trade with money you cannot afford to lose
- No "buy" or "sell" recommendations are made — only strength scores and evidence presentation
- Data is sourced from free public APIs and may have delays or inaccuracies

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

PA_MCP draws inspiration and architectural ideas from:
- [TradingAgents-astock](https://github.com/simonlin1212/TradingAgents-astock) — Multi-analyst design
- [DeepPulse](https://github.com/wwyharry/DeepPulse) — Strategy engine + memory system
- [Vibe-Research](https://github.com/simonlin1212/Vibe-Research) — Bull-bear debate pattern
- [cn-financial-mcp](https://github.com/ccq1/cn-financial-mcp) — Tool coverage design
- [QuantsPlaybook](https://github.com/hugo2046/QuantsPlaybook) — Strategy library
