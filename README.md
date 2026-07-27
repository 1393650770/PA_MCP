# PA_MCP — Personal Analyst MCP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-76%2F76-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Mac-success.svg)]()

Full-stack A-share quantitative trading MCP server. Let AI agents (Claude, etc.) query Chinese stock market data, run technical analysis, execute trading strategies, and perform backtesting — all through natural language.

**Zero C compilation. Works on Windows/Linux/Mac with one command.**

---

## 🚀 Quick Start (Windows)

```powershell
# 1. One-click install
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

# 2. Configure LLM API key
notepad config\llm_config.json

# 3. Run
venv\Scripts\activate
python -m pa_mcp.server
```

### Linux / Mac

```bash
# Install
pip install -e ".[dev]"

# Configure
cp config/llm_config.example.json config/llm_config.json
# Edit config/llm_config.json with your API key

# Run
python -m pa_mcp.server
```

### Docker

```bash
docker compose up -d
# Health check
curl http://localhost:8080/health
```

---

## 🏗️ Architecture (5 Layers)

```
+==========================================================+
|  MCP Interface (FastMCP + HTTP/SSE + stdio)              |
|  20+ Tools | 2 Resources | Multi-provider LLM support     |
+==========================================================+
|  AI Agent Decision (asyncio, dual-mode fast/deep)        |
|  5 analyst prompts | Bull-bear self-debate | SQL memory  |
+==========================================================+
|  Hard RiskGuard (NON-BYPASSABLE)                          |
|  Position caps | Circuit breakers | Systemic alarms       |
+==========================================================+
|  Engine (150+ indicators + 14 strategies + backtest)     |
|  Pure Python indicators | DuckDB SQL vectorized backtest  |
+==========================================================+
|  Data (DuckDB + AKShare + lru_cache)                     |
|  Multi-source fallback | 9 validators | Cron scheduler   |
+==========================================================+
```

---

## ✨ Features

### Market Data
- Real-time quotes (with delay annotation), historical K-line (daily/weekly/monthly/1m/5m/30m/60m)
- Multi-source: AKShare primary + TickFlow fallback
- Every response includes `data_source`, `data_freshness`, `data_delay_seconds`

### Technical Analysis
- **150+ indicators, pure Python** — zero C compilation, works everywhere
- MA (SMA/EMA), MACD, RSI, KDJ, Bollinger, ATR, OBV, CCI, Williams %R, ADX, MFI, VWAP
- K-line pattern recognition, volume-price analysis

### Strategy Engine (14 Strategies)
| Category | Strategies |
|----------|-----------|
| 🔥 Board (打板) | First board breakout, Dragon second wave |
| 📈 Trend | Platform breakout, MA golden cross |
| 🌊 Swing | MACD divergence, Bollinger mean reversion |
| 💎 Value | ROE-PB value screen |
| 🕸️ Grid | Range grid (ATR-based) |
| 🚀 Momentum | Volume-price breakout |
| 🔄 Reversal | Oversold bounce (RSI <25) |
| 📰 Event | Insider buying follow, Lockup expiry avoidance, Block trade discount alert, Pledge risk |

### Backtest Engine
- **DuckDB SQL vectorized**: 5000-stock full-market scan in seconds
- **A-share reality constraints** (these are what make the difference):
  - T+1 enforcement
  - Limit-up buyability probability (seal time + seal strength model)
  - Limit-down sellability (consecutive limit-down decay)
  - Liquidity constraints (<5% daily volume per order)
  - Differentiated slippage (large-cap 0.1%, small-cap 0.3%, panic +0.2%)
  - Commission + stamp duty + transfer fee per exchange

### Hard RiskGuard
**Non-bypassable risk controls** — sits between agent output and user response:
- Single stock max 20%, sector max 40%, total max 80%
- Daily loss >3% → force liquidate
- 3 consecutive losses → pause 1 week
- Systemic risk triggers: mass limit-down, volume collapse, bear market, northbound exodus
- Seasonal defense: auto-reduce position in historically weak months

### AI Agent Analysis
- **Fast mode** (~15s, 8K tokens): Single LLM call, 5 dimensions in one response
- **Deep mode** (~60s, 50K tokens): 5 parallel analysts + self-debate + risk review
- **Long-term memory** (SQLite): Decision tracking, outcome monitoring, bias detection, strategy weight adjustment

### Multi-Provider LLM
- Supports: Anthropic Claude / OpenAI / DeepSeek / 智谱 GLM / 通义千问
- Unified API, retry with backoff, response caching
- Configure via `config/llm_config.json` (gitignored, not committed)

### Event Data
- Block trades (大宗交易), Lockup expiry (限售股解禁)
- Insider trading (大股东增减持), Pledge ratios (股权质押)
- Institutional visits (机构调研)
- Exposed as single `get_major_events(symbol)` tool

---

## 📡 MCP Tools (20+)

### Market Data
| Tool | Description |
|------|-------------|
| `get_realtime_quote` | Real-time quote + 5-level depth + delay annotation |
| `get_kline` | Historical K-line (daily/weekly/monthly/1m/5m/30m/60m) |
| `get_market_overview` | Market breadth, turnover, northbound flow |
| `search_stock` | Search stocks by name or code |
| `get_stock_info` | Company info, sector, market cap |

### Screener
| Tool | Description |
|------|-------------|
| `scan_limit_up` | Limit-up stocks + seal strength + sector distribution |
| `scan_volume_surge` | Abnormal volume vs 20-day average |
| `get_major_events` | Block trades, lockup, insider trades, pledge, visits |

### Analysis
| Tool | Description |
|------|-------------|
| `analyze_technical` | Multi-indicator technical report |
| `analyze_fundamental` | Financials + valuation + DuPont |
| `analyze_sentiment` | News sentiment score + trend |
| `analyze_capital_flow` | Main/retail/northbound fund flow |

### Review
| Tool | Description |
|------|-------------|
| `review_daily_limit_up` | Daily limit-up review (seal time, sector heatmap) |
| `review_dragon_tiger` | Dragon-tiger seat analysis + tracking |
| `get_market_sentiment` | Market mood + position suggestion + risk alerts |

### Strategy
| Tool | Description |
|------|-------------|
| `list_strategies` | Available strategies by category + market suitability |

### Alerts
| Tool | Description |
|------|-------------|
| `watch_price_alert` | Create price condition alert |
| `watch_volume_alert` | Create volume surge alert |
| `list_alerts` | List active/triggered alerts |

### Agent
| Tool | Description |
|------|-------------|
| `agent_analyze_stock` | AI multi-dimensional analysis (fast/deep) |
| `agent_market_state` | Current regime + position suggestion |

### Portfolio
| Tool | Description |
|------|-------------|
| `portfolio_summary` | Holdings + P&L |
| `portfolio_add` | Add holding |
| `portfolio_remove` | Remove holding |

---

## 🔧 Configuration

### LLM API (`config/llm_config.json`)

```json
{
  "active_provider": "anthropic",
  "providers": {
    "anthropic": {
      "api_key": "sk-ant-...",
      "base_url": "https://api.anthropic.com",
      "models": { "fast": "claude-sonnet-5-20251001", "deep": "claude-opus-5-20251001" }
    },
    "deepseek": {
      "api_key": "sk-...",
      "base_url": "https://api.deepseek.com/v1",
      "models": { "fast": "deepseek-chat", "deep": "deepseek-reasoner" }
    }
  },
  "token_budget": {
    "fast_analysis_max": 8000,
    "deep_analysis_max": 50000,
    "daily_scan_budget": 200000
  }
}
```

See [config/llm_config.example.json](config/llm_config.example.json) for all 5 providers.

### Server (`config/default.yaml`)

Key environment variables:
| Variable | Default | Description |
|----------|---------|-------------|
| `PA_MCP_SERVER__TRANSPORT` | `stdio` | `stdio` or `http` |
| `PA_MCP_SERVER__HTTP_PORT` | `8080` | HTTP port |
| `PA_MCP_DATABASE__PATH` | `data/pa_mcp.duckdb` | Database file |
| `PA_MCP_CACHE__BACKEND` | `memory` | `memory` / `redis` / `none` |
| `PA_MCP_RISK__MAX_SINGLE_STOCK` | `0.20` | Max single position |
| `PA_MCP_RISK__MAX_TOTAL_POSITION` | `0.80` | Max total exposure |

---

## 🖥️ Claude Desktop Integration

```json
{
  "mcpServers": {
    "pa-mcp": {
      "command": "F:\\PA_MCP\\venv\\Scripts\\python.exe",
      "args": ["-m", "pa_mcp.server"],
      "cwd": "F:\\PA_MCP"
    }
  }
}
```

---

## 📂 Project Structure

```
pa-mcp/
├── src/pa_mcp/
│   ├── server.py              # MCP Server entry (20+ tools)
│   ├── config.py              # Pydantic Settings (9 config classes)
│   ├── data/
│   │   ├── store.py           # DuckDB (13 tables + shadow swap)
│   │   ├── cache.py           # Multi-level cache (lru + optional Redis)
│   │   ├── quality.py         # 9 automated data validators
│   │   ├── scheduler.py       # 8-phase daily pipeline
│   │   └── sources/
│   │       └── akshare_adapter.py  # 15 endpoints, rate-limited, retry
│   ├── engine/
│   │   ├── indicators/
│   │   │   └── indicators.py  # 150+ pure Python indicators
│   │   ├── strategies/
│   │   │   ├── base.py        # ABC + registry + signal format
│   │   │   ├── trend.py       # Platform breakout, MA golden cross
│   │   │   ├── limit_up.py    # First board, Dragon second wave
│   │   │   ├── swing.py       # MACD divergence, Bollinger, Value, Grid, Momentum
│   │   │   └── event_driven.py # Insider, Lockup, Oversold, Block trade
│   │   ├── market_state.py    # 5-state detector (hysteresis-based)
│   │   └── signals.py         # Signal aggregation + conflict resolution
│   ├── backtest/
│   │   ├── engine.py          # DuckDB SQL vectorized backtester
│   │   ├── rules.py           # A-share rules, fees, buyability/sellability models
│   │   └── calendar.py        # Trading day calendar
│   ├── risk/
│   │   └── guard.py           # Non-bypassable risk checks + systemic alarms
│   ├── agent/
│   │   ├── orchestrator.py    # Fast/deep analysis orchestration
│   │   ├── llm_client.py      # Multi-provider LLM client + caching
│   │   └── memory.py          # Long-term memory (SQLite, bias detection)
│   └── info/
│       └── news.py            # News aggregation + sentiment analysis
├── config/
│   ├── default.yaml           # Server configuration
│   ├── llm_config.example.json # LLM provider template
│   └── llm_config.json        # Your API keys (gitignored)
├── scripts/
│   ├── setup.ps1              # Windows one-click setup
│   ├── init_db.py             # Database initialization
│   └── run_backtest.py        # CLI backtest runner
├── tests/                     # 76 tests, 5 files
├── docker-compose.yml         # 3 services (server, cron, optional redis)
├── Dockerfile                 # Single-stage, zero C build
├── docs/superpowers/specs/    # Design specifications (4 docs)
├── pyproject.toml             # 12 pure Python deps
├── DISCLAIMER.md              # Legal disclaimer
└── README.md                  # This file
```

---

## 📊 Project Stats

| Metric | Value |
|--------|-------|
| Source files | 33 |
| Tests | 76/76 |
| Strategies | 14 across 8 categories |
| Technical indicators | 150+ (pure Python) |
| MCP tools | 20+ |
| DuckDB tables | 13 |
| Data validators | 9 |
| Python deps | 12 (zero C compilation) |
| LLM providers | 5 (Anthropic/OpenAI/DeepSeek/智谱/通义) |
| Docs | 4 spec documents |

---

## ⚠️ Disclaimer

**This is a research tool, not investment advice.**

- All analysis is for educational and reference purposes only
- Past performance does not guarantee future results
- No "buy" or "sell" recommendations — only strength scores and evidence
- Data from free public APIs may have delays or inaccuracies
- Backtest results systematically overestimate real returns by 30-50%
- See [DISCLAIMER.md](DISCLAIMER.md) for full legal notice

---

## 🎯 Acknowledgments

PA_MCP draws from 45+ open-source projects, with core architectural ideas from:

- [TradingAgents-astock](https://github.com/simonlin1212/TradingAgents-astock) (2.6k★) — Multi-analyst + bull-bear debate
- [DeepPulse](https://github.com/wwyharry/DeepPulse) (34★) — Strategy engine + long-term memory + circuit breaker
- [Vibe-Research](https://github.com/simonlin1212/Vibe-Research) (1k★) — Fact-presentation, no buy/sell conclusions
- [cn-financial-mcp](https://github.com/ccq1/cn-financial-mcp) (30★) — 42-tool comprehensive MCP design
- [QuantsPlaybook](https://github.com/hugo2046/QuantsPlaybook) (5.7k★) — 100+ strategy library
- [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) (21k★) — Chinese financial sentiment
- [Microsoft Qlib](https://github.com/microsoft/qlib) (46k★) — AI quant platform design patterns

## 📜 License

MIT — see [LICENSE](LICENSE)
