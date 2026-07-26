# PA_MCP — Final Design Specification (v2.0)

**Date:** 2026-07-26
**Version:** 2.0
**Status:** Approved for Implementation
**Reviewers:** 3 refine agents + 3 review agents

---

## Change Log from v1.0

| Change | Source | Severity |
|--------|--------|----------|
| backtrader → vectorbt + DuckDB SQL backtest | Architecture review | CRITICAL |
| Added Hard RiskGuard layer (non-bypassable) | Practical review | CRITICAL |
| Signal output: buy/sell → strength score + factor list | Compliance review | CRITICAL |
| efinance → TickFlow; data sources reduced to 2 | Correctness review | CRITICAL |
| 7 layers → 5 layers (merged analysis+strategy→engine) | Architecture review | HIGH |
| Added 3-level degradation plan | Architecture review | HIGH |
| Added pre-computation strategy (indicator_cache table) | Performance review | HIGH |
| Added event data module (大宗/解禁/增减持/质押) | Practical review | HIGH |
| 40 strategies → 20 core + YAML declarative extension | Maintenance review | HIGH |
| Redis → optional (lru_cache default) | Performance review | MEDIUM |
| Dual-mode Agent: fast (single call) / deep (debate) | Token cost review | HIGH |
| Added compliance disclaimer system | Legal review | HIGH |

---

## 1. Project Overview

### 1.1 Goal

Build a full-stack A-share quantitative MCP Server enabling AI agents (Claude, etc.) to:
- Query A-share real-time/historical market data
- Execute technical, fundamental, sentiment, and event-driven analysis
- Run 20+ built-in strategies and output strength scores (not buy/sell orders)
- Backtest strategies with realistic A-share constraints
- Produce multi-dimensional research reports via Agent debate (dual-mode)
- Provide watchlist alerts, daily review, and portfolio analysis

### 1.2 Core Principles

- **Free & open-source first**: AKShare (primary) + TickFlow (fallback), zero paid API dependency
- **Full style coverage**: Ultra-short-term, swing/trend, medium-long-term value
- **Docker one-click deploy**: `docker compose up`
- **Absorb best ideas from 10+ open-source projects**
- **Research tool, not trading advisor**: Output strength scores + evidence, never "buy now!"
- **Hard risk control**: Non-bypassable RiskGuard layer — lives are saved by discipline, not genius

### 1.3 Reference Projects

| Project | Absorbed |
|---------|----------|
| TradingAgents-astock (2.6k stars) | Multi-analyst + bull-bear debate + A-share constraints |
| aiagents-stock (1.7k stars) | Dragon-tiger team + capital flow + sector rotation |
| DeepPulse (34 stars) | 40 strategies + ReAct engine + long-term memory + circuit breaker |
| Vibe-Research (1k stars) | Bull-bear debate + fact presentation (no buy/sell conclusion) |
| cn-financial-mcp (30 stars) | 42-tool comprehensive coverage design |
| china-stock-mcp (44 stars) | Multi-source fallback + dual transport |
| StockAgent (353 stars) | Distributed microservice architecture |
| EasyQuant (52 stars) | Dual-LLM cross-review factor mining |
| QuantsPlaybook (5.7k stars) | 100+ strategy library |
| FinGPT (21k stars) | Chinese financial sentiment analysis |

---

## 2. Architecture (5 Layers)

```
+==========================================================+
|  MCP Interface (FastMCP + HTTP/SSE + stdio)              |
|  25+ Tools | Resources | Prompts                          |
+==========================================================+
|  AI Agent Decision Layer (asyncio + optional LangGraph)   |
|  Fast mode (single call) / Deep mode (5 analysts+debate)  |
+==========================================================+
|  Hard RiskGuard (NON-BYPASSABLE)                          |
|  Position caps | Loss circuit breakers | Systemic alarms  |
+==========================================================+
|  Engine Layer (indicators + strategies + backtest)        |
|  20 core strategies | vectorbt + DuckDB SQL backtest      |
|  TA-Lib + pandas-ta | Signal aggregation                  |
+==========================================================+
|  Data Layer (DuckDB + optional Redis)                     |
|  AKShare primary + TickFlow fallback | Pre-computation    |
|  Daily auto-update | Data quality validation              |
+==========================================================+
```

### 2.1 Key Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **5 layers, not 7** | Merged analysis+strategy into Engine; merged info into Data. Fewer interfaces, less maintenance |
| **DuckDB, not PostgreSQL** | Embedded OLAP, zero-config, 5000-stock full scan <1s. Single-writer (cron only) avoids lock issues |
| **vectorbt, not backtrader** | backtrader last released 2020, dead project. vectorbt active, vectorized, A-share adaptable |
| **lru_cache default, Redis optional** | Single-process MCP server doesn't need network cache. Redis enabled only for multi-instance |
| **asyncio first, LangGraph optional** | Agent flow is simple DAG. asyncio.gather for 5 analysts, await chain for debate. LangGraph only if complexity grows |

### 2.2 Directory Structure

```
pa-mcp/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── README.md
├── DISCLAIMER.md                  # Legal disclaimer
├── docs/
│   └── superpowers/specs/
├── src/pa_mcp/
│   ├── __init__.py
│   ├── server.py                  # MCP Server entry point
│   ├── config.py                  # Pydantic Settings
│   ├── data/                      # ---- DATA LAYER ----
│   │   ├── __init__.py
│   │   ├── sources/
│   │   │   ├── __init__.py
│   │   │   ├── akshare_adapter.py # Primary: 15 endpoints
│   │   │   └── tickflow_adapter.py# Fallback: 8 endpoints
│   │   ├── aggregator.py          # Multi-source + fallback + consistency check
│   │   ├── cache.py               # lru_cache + optional Redis
│   │   ├── store.py               # DuckDB CRUD
│   │   ├── scheduler.py           # Cron update pipeline
│   │   ├── quality.py             # 9 validation checks
│   │   └── events.py              # Key events: block trade, lockup, insider trading, pledge
│   ├── engine/                    # ---- ENGINE LAYER ----
│   │   ├── __init__.py
│   │   ├── indicators/            # Pure calculation (TA-Lib + pandas-ta wrappers)
│   │   │   ├── __init__.py
│   │   │   ├── trend.py           # MA, MACD, ADX, etc.
│   │   │   ├── momentum.py        # RSI, KDJ, CCI, WR, etc.
│   │   │   ├── volatility.py      # BOLL, ATR, etc.
│   │   │   ├── volume.py          # OBV, MFI, VWAP, etc.
│   │   │   └── patterns.py        # 60+ candlestick patterns
│   │   ├── strategies/            # 20 core strategies
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # BaseStrategy ABC
│   │   │   ├── registry.py        # Strategy registry (auto-discovery)
│   │   │   ├── limit_up.py        # 3 board strategies
│   │   │   ├── trend.py           # 3 trend strategies
│   │   │   ├── swing.py           # 3 swing strategies
│   │   │   ├── value.py           # 2 value strategies
│   │   │   ├── grid.py            # 2 grid strategies
│   │   │   ├── reversal.py        # 2 reversal strategies
│   │   │   ├── event_driven.py    # 3 event-driven strategies
│   │   │   └── momentum.py        # 2 momentum strategies
│   │   ├── signals.py             # Signal generation + JSON schema
│   │   ├── aggregation.py         # Multi-strategy signal aggregation
│   │   └── market_state.py        # Market regime detection
│   ├── backtest/                  # ---- BACKTEST ENGINE ----
│   │   ├── __init__.py
│   │   ├── engine.py              # vectorbt wrapper
│   │   ├── duckdb_engine.py       # DuckDB SQL vectorized backtest (fast path)
│   │   ├── rules.py               # A-share trading rules (Protocol-based)
│   │   ├── calendar.py            # A-share trading calendar
│   │   ├── constraints.py         # Liquidity/slippage/limit-up reality checks
│   │   └── optimization.py        # Monte Carlo + grid + genetic
│   ├── risk/                      # ---- HARD RISKGUARD ----
│   │   ├── __init__.py
│   │   ├── guard.py               # Non-bypassable risk check pipeline
│   │   ├── position.py            # Position limits (single/sector/total)
│   │   ├── circuit_breaker.py     # Loss-based circuit breakers
│   │   ├── systemic.py            # Systemic risk early warning
│   │   └── rules.py               # Configurable risk rules
│   ├── agent/                     # ---- AGENT LAYER ----
│   │   ├── __init__.py
│   │   ├── orchestrator.py        # asyncio-based agent orchestration
│   │   ├── analysts.py            # 5 analyst prompt templates
│   │   ├── debate.py              # Bull-bear debate logic
│   │   ├── memory.py              # Long-term memory (SQLite)
│   │   └── token_optimizer.py     # Token budget + data compression
│   ├── info/                      # ---- INFO PROCESSING ----
│   │   ├── __init__.py
│   │   ├── news.py                # News aggregation (CaiLianShe, JuChao)
│   │   ├── sentiment.py           # Sentiment analysis (bardsai/finance-sentiment-zh-base)
│   │   └── events.py              # Event extraction + theme mapping
│   └── tools/                     # ---- MCP TOOLS ----
│       ├── __init__.py
│       ├── market_data.py         # 5 tools
│       ├── screener.py            # 5 tools
│       ├── analysis.py            # 4 tools
│       ├── review.py              # 4 tools
│       ├── strategy.py            # 4 tools
│       ├── alerts.py              # 3 tools
│       ├── agent_tools.py         # 4 tools
│       └── portfolio.py           # 3 tools
├── tests/
├── config/
│   ├── default.yaml
│   └── strategies/                # YAML declarative strategies
└── docker/
    ├── Dockerfile
    └── cron-scheduler/
```

---

## 3. MCP Tools (25+ Tools, Simplified)

### 3.1 Market Data (5 tools)

| Tool | Params | Returns |
|------|--------|---------|
| `get_realtime_quote` | symbol, source | Quote + 5-level depth + data_delay_seconds |
| `get_kline` | symbol, period, start, end, adjust | OHLCV array |
| `get_market_overview` | - | Indices, breadth, turnover, northbound flow |
| `search_stock` | keyword | Stock matches |
| `get_stock_info` | symbol | Company info, sector, market cap, list date |

### 3.2 Screener (5 tools)

| Tool | Params | Returns |
|------|--------|---------|
| `scan_limit_up` | date, board | Limit-up stocks + seal strength + consecutive stats |
| `scan_volume_surge` | ratio, market_cap_min | Abnormal volume stocks |
| `scan_breakout` | period, type | Breakout candidates |
| `scan_hot_sector` | top_n | Hot sectors + leading stocks |
| `get_major_events` | symbol | Block trades, lockup expiry, insider trades, pledge ratio, institutional visits |

### 3.3 Analysis (4 tools)

| Tool | Params | Returns |
|------|--------|---------|
| `analyze_technical` | symbol, indicators[] | Multi-indicator report |
| `analyze_fundamental` | symbol | Financials + valuation + DuPont |
| `analyze_sentiment` | symbol, days | News sentiment score + trend |
| `analyze_capital_flow` | symbol, days | Main/retail/northbound fund flow |

### 3.4 Review (4 tools)

| Tool | Params | Returns |
|------|--------|---------|
| `review_daily_limit_up` | date | Limit-up review: seal time, break rate, sector distribution |
| `review_dragon_tiger` | date | Dragon-tiger: seat analysis, famous trader tracking, seat credibility scores |
| `review_sector_rotation` | days | Sector rotation + persistence |
| `review_market_sentiment` | date | Market mood: breadth, break rate, streak height, volume |

### 3.5 Strategy (4 tools)

| Tool | Params | Returns |
|------|--------|---------|
| `list_strategies` | category, market_state | Strategies + suitability for current market |
| `run_strategy` | strategy_name, symbol | Strength score (0-100) + reasoning + risk flags |
| `backtest_strategy` | strategy_name, symbol, start, end, capital | Backtest report with realistic constraints |
| `optimize_strategy` | strategy_name, param_ranges, symbol | Optimal params + sensitivity heatmap |

### 3.6 Alerts (3 tools)

| Tool | Params | Returns |
|------|--------|---------|
| `watch_price_alert` | symbol, condition, price | Alert ID |
| `watch_volume_alert` | symbol, volume_ratio | Alert ID |
| `list_alerts` | status | Alert list |

### 3.7 Agent (4 tools)

| Tool | Params | Returns |
|------|--------|---------|
| `agent_analyze_stock` | symbol, depth(fast/deep) | Multi-analyst report + strength score |
| `agent_scan_market` | strategy, risk_level, max_results | Top candidates + reasoning |
| `agent_market_state` | - | Current regime + position suggestion + risk alerts |
| `agent_morning_brief` | - | Pre-market brief: overnight news, global markets, today's watchlist |

### 3.8 Portfolio (3 tools)

| Tool | Params | Returns |
|------|--------|---------|
| `portfolio_summary` | - | Holdings + P&L + risk metrics |
| `portfolio_risk` | - | VaR, max drawdown, correlation matrix, stress test |
| `portfolio_rebalance` | target_weights | Rebalance suggestions |

### Compliance Notice

**Every tool response includes:**
```json
{
  "disclaimer": "This is a research tool, not investment advice. All analysis is for reference only. Past performance does not guarantee future results. Trading involves risk of loss.",
  "data_source": "akshare",
  "data_freshness": "2026-07-26T17:35:00+08:00",
  "degraded_components": []
}
```

Output is always **strength scores** (0-100) and **evidence presentation**, never explicit "buy" or "sell" orders.

---

## 4. Hard RiskGuard — Non-Bypassable

The RiskGuard sits between Agent output and the final response. It cannot be debated, overridden, or "special-cased."

### 4.1 Position Hard Constraints

```python
MAX_SINGLE_STOCK = 0.20       # Max 20% in one stock
MAX_SECTOR_EXPOSURE = 0.40    # Max 40% in one sector
MAX_TOTAL_POSITION = 0.80     # Always keep 20% cash
MAX_DAILY_LOSS = 0.03         # Force liquidate if daily loss >3%
MAX_CONSECUTIVE_LOSSES = 3    # Pause 1 week after 3 consecutive losses
PAUSE_AFTER_BIG_LOSS = 0.05   # Pause 3 days if single-day loss >5%
```

### 4.2 Systemic Risk Early Warning (Auto De-risk)

| Trigger | Action |
|---------|--------|
| Limit-down >50 AND limit-up <30 | Reduce to 30% position |
| Market turnover <500B CNY | Reduce to 20% position |
| Index below 200-day MA | Reduce to 10% position |
| Northbound net outflow >10B CNY for 3 consecutive days | Reduce to 30% position |
| Margin balance declining >1%/day for 3 days | Reduce to 40% position |
| Month = May/June/September/November | Auto-lower position ceiling to 50% |

### 4.3 "Time to Rest" Detection

```
Historical pattern: A-shares are tradable ~3-4 months/year
  Jan: Low (pre-CNY liquidity crunch)
  Feb-Apr: HIGH (annual report season + Two Sessions catalyst)
  May-Jun: LOW ("Sell in May") — historically worst 2 months
  Jul-Aug: MED-HIGH (semi-annual report season)
  Sep: LOW (pre-National Day wait-and-see)
  Oct: MED-HIGH (post-holiday fund回流)
  Nov-Dec: LOW (institutional year-end settlement)
```

This seasonal pattern feeds into the market state detector.

---

## 5. Engine Layer

### 5.1 Strategy Classification (Two-Dimensional)

| X-Axis (Selection) | Y-Axis (Position Mgmt) | Strategies |
|---------------------|------------------------|------------|
| Board (打板) | Fixed fraction | First board, 2nd board relay |
| Trend (趋势) | Trailing stop | MA bullish alignment, platform breakout |
| Reversal (反转) | Grid scale-in | Oversold bounce, panic dip buy |
| Value (价值) | Value averaging | ROE-PB, dividend yield |
| Event (事件驱动) | Event-driven sizing | Lockup expiry avoidance, insider buying follow |
| Momentum (动量) | Kelly fraction | Mid-term momentum, volume-price breakout |
| Swing (波段) | ATR-based | Bollinger band, MACD divergence |

**Key principle:** Selection logic and position management are orthogonal and independently configurable.

### 5.2 Strategy Registration

```python
# Python class registration (for complex strategies)
class PlatformBreakoutStrategy(BaseStrategy):
    name = "platform_breakout"
    category = StrategyCategory.TREND
    params = {"ma_period": 20, "consolidation_min_days": 15, "breakout_volume_ratio": 1.5}

# YAML declarative registration (for simple rule-based strategies)
# config/strategies/ma_crossover.yaml
name: ma_crossover
category: trend
description: "Golden cross: MA5 crosses above MA20 with volume confirmation"
rules:
  - condition: "cross(ma5, ma20, direction='above')"
  - condition: "volume > sma(volume, 20) * 1.2"
  - signal: "strength = 60 + (volume_ratio - 1.2) * 20"
```

### 5.3 Signal Format (Strength-Based, Not Buy/Sell)

```json
{
  "symbol": "000001",
  "strategy": "platform_breakout",
  "strength_score": 72,
  "strength_breakdown": {
    "technical": 80,
    "volume": 75,
    "sector": 60,
    "sentiment": 65
  },
  "direction": "bullish",
  "evidence": [
    {"type": "technical", "claim": "21-day consolidation, volume at minimum", "confidence": 0.85},
    {"type": "volume", "claim": "Today's volume 1.8x 20-day average", "confidence": 0.90},
    {"type": "sector", "claim": "Banking sector net inflow for 5 days", "confidence": 0.70}
  ],
  "risk_flags": [
    {"type": "market_regime", "severity": "high", "detail": "Market in downturn phase, systemic risk elevated"},
    {"type": "resistance", "severity": "medium", "detail": "Dense resistance zone at 13.80"}
  ],
  "risk_reward_ratio": 2.1,
  "suggested_max_position_pct": 0.05,
  "disclaimer": "Research output, not investment advice.",
  "timestamp": "2026-07-26T14:30:00+08:00"
}
```

### 5.4 Signal Aggregation (Conflict Resolution)

When multiple strategies produce conflicting signals:

1. **Market state filter**: Only enable strategies suitable for current regime
2. **Weighted vote**: Strategy weight = historical win rate in similar market states × recency decay
3. **Bayesian ensemble**: P(bullish | strategies) = Σ P(bullish | strategy_i) × P(strategy_i | market_state)
4. **Output**: Aggregated strength score + per-strategy breakdown + conflict analysis

---

## 6. Backtest Engine

### 6.1 Dual Engine Design

| Path | Engine | Use Case | Speed |
|------|--------|----------|-------|
| Fast | DuckDB SQL | Full-market scan, simple rule strategies | <1s (5000 stocks) |
| Detailed | vectorbt | Single stock deep analysis, complex state-machine strategies | ~5s per stock |

### 6.2 A-Share Reality Constraints (CRITICAL)

These MUST be simulated to avoid backtest-optimism:

| Constraint | Implementation |
|------------|---------------|
| **Limit-up buyability** | If seal_amount/volume > 5, mark as "unbuyable". If seal_time < 09:35, mark as "unbuyable". Post-14:00 limit-up: 40% buy probability |
| **Limit-down sellability** | If consecutive limit-down, exit probability decays exponentially |
| **Liquidity constraint** | Single order < 5% of daily volume. Small-cap (<3B CNY) impact cost 0.3% |
| **Slippage model** | Large-cap: 0.1%, Small-cap: 0.3%, ChiNext/STAR: 0.5%, Panic selling: +0.2% |
| **T+1 enforcement** | Entry date tracked; same-day exit blocked |
| **Suspension handling** | Skip suspended days; re-list day: gap-adjusted entry |

### 6.3 Backtest API

```python
result = engine.backtest(
    symbol="000001",
    strategy_name="platform_breakout",
    start="2024-01-01",
    end="2025-12-31",
    initial_capital=100000,
    params={"ma_period": 20, "consolidation_min_days": 15}
)
# Returns: BacktestReport with all metrics + trade log + equity curve
```

---

## 7. Agent Decision Layer

### 7.1 Dual Mode Design

| Mode | Call Pattern | Token ~ | Latency | Use Case |
|------|-------------|---------|---------|----------|
| **Fast** | Single LLM call, structured output, 5 dimensions in one response | ~8K | 10-15s | Daily scanning, quick checks |
| **Deep** | 5 analysts parallel → debate → risk review | ~50K | 40-60s | High-conviction candidates, manual review |

### 7.2 Fast Mode Prompt Structure

```
You are a seasoned A-share analyst. Analyze {symbol} across these 5 dimensions:
1. TECHNICAL: Key patterns, indicator alignment, support/resistance
2. CAPITAL: Fund flow direction, dragon-tiger activity, northbound position
3. SENTIMENT: News sentiment, social media heat, sector mood
4. FUNDAMENTAL: Financial health, valuation percentile, growth trajectory
5. EVENT: Upcoming lockup expiry, insider trades, block trades, pledge risk

For each dimension: strength_score (0-100), top 3 evidence points, risk flags.
Final: aggregated strength_score, key risk, risk/reward ratio.
Do NOT output "buy" or "sell" — output scores and evidence only.
```

### 7.3 Deep Mode: asyncio Orchestration

```python
async def deep_analyze(symbol: str, market_state: MarketState):
    # Stage 1: Parallel data fetch
    kline, financials, news, capital_flow, events, lhb = await asyncio.gather(
        fetch_kline(symbol),
        fetch_financials(symbol),
        fetch_news(symbol),
        fetch_capital_flow(symbol),
        fetch_events(symbol),
        fetch_dragon_tiger(symbol),
    )

    # Stage 2: 5 analysts in parallel
    reports = await asyncio.gather(
        technical_analyst(kline, market_state),
        capital_analyst(capital_flow, lhb),
        sentiment_analyst(news, market_state),
        fundamental_analyst(financials),
        event_analyst(events),
    )

    # Stage 3: Debate (optional, deep mode only)
    debate_result = await bull_bear_debate(reports, kline)

    # Stage 4: Risk review
    risk_assessment = await risk_review(debate_result, market_state)

    # Stage 5: Hard RiskGuard check (NON-BYPASSABLE)
    guard_result = risk_guard.check(risk_assessment)
    if not guard_result.passed:
        risk_assessment.overridden_by = guard_result

    return risk_assessment
```

### 7.4 Bull-Bear Debate (Simplified)

Instead of 2 separate agents debating each other (3+ LLM calls), use **self-debate**:

```
Based on the analyst reports, do the following in ONE response:
1. List the TOP 3 bull arguments (specific, data-backed)
2. List the TOP 3 bear arguments (specific, data-backed)  
3. For each argument on BOTH sides, identify the weakest assumption
4. Synthesize: what do both sides agree on? What is genuinely disputed?
5. Final: strength_score_adjusted, key uncertainty, "what would change the thesis"
```

This reduces debate from 3-6 LLM calls to 1 call while preserving the adversarial thinking structure.

### 7.5 Long-Term Memory

SQLite-based, tracks:
- **Decision log**: symbol, date, signal, strength_score, market_state, analyst reports hash
- **Outcome tracking**: 5-day/10-day/20-day forward return, max adverse excursion
- **Bias detection**: Overconfidence (high confidence + negative outcome pattern), disposition effect (holding losers too long)
- **Strategy weights**: Bayesian blending based on rolling 60-trade performance

---

## 8. Data Layer

### 8.1 Data Source Strategy (Simplified)

| Source | Role | Coverage |
|--------|------|----------|
| **AKShare** | PRIMARY | Real-time quotes, daily kline, financials, dragon-tiger, capital flow, events |
| **TickFlow** | FALLBACK | Real-time quotes, daily kline (efinance successor by same author) |

**Drop efinance** — author moved to TickFlow. **Drop BaoStock** — only historical daily data, no real-time, redundant with AKShare.

### 8.2 Data Freshness Annotation

Every query result includes:

```json
{
  "data_source": "akshare",
  "data_freshness": "2026-07-26T14:55:00+08:00",
  "data_delay_seconds": 5,
  "reliability": "near_realtime"
}
```

Realtime data is "near realtime" (3-15s delay), not true realtime. This is documented, not hidden.

### 8.3 Pre-Computation Strategy

Daily at 17:30, cron triggers:

```
Phase 1: Fetch (17:30-17:35)
  → Daily kline (5000 stocks, incremental)
  → Financials (incremental, new filings only)
  → Capital flow + Dragon-tiger

Phase 2: Validate (17:35-17:36)
  → Row count check (expected vs actual)
  → Cross-source OHLC consistency (<1% deviation)
  → Negative price / volume sanity

Phase 3: Pre-compute (17:36-17:40)
  → 150 technical indicators → indicator_cache table (DuckDB)
  → Strategy signals → signal_cache table
  → Sentiment scores → sentiment_cache table

Phase 4: Swap (17:40)
  → Atomic table swap (write to _new, rename)
  → Invalidate lru_cache entries
```

After pre-computation, real-time queries hit cache directly — no Python calculation needed.

### 8.4 DuckDB Core Tables

```sql
-- Stock daily kline
CREATE TABLE kline_daily (
    symbol VARCHAR(10),
    date DATE,
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
    volume DOUBLE, amount DOUBLE,
    adjust_factor DOUBLE,
    PRIMARY KEY (symbol, date)
);

-- Pre-computed indicators (wide table for OLAP)
CREATE TABLE indicator_cache (
    symbol VARCHAR(10),
    date DATE,
    ma5 DOUBLE, ma10 DOUBLE, ma20 DOUBLE, ma60 DOUBLE, ma120 DOUBLE, ma250 DOUBLE,
    macd DOUBLE, macd_signal DOUBLE, macd_hist DOUBLE,
    rsi6 DOUBLE, rsi14 DOUBLE, rsi24 DOUBLE,
    kdj_k DOUBLE, kdj_d DOUBLE, kdj_j DOUBLE,
    boll_upper DOUBLE, boll_mid DOUBLE, boll_lower DOUBLE,
    atr14 DOUBLE, obv DOUBLE,
    -- ... 150 indicators total
    PRIMARY KEY (symbol, date)
);

-- Key events
CREATE TABLE major_events (
    symbol VARCHAR(10),
    event_date DATE,
    event_type VARCHAR(20),  -- block_trade, lockup_expiry, insider_trade, pledge_change, inst_visit
    detail JSON,
    PRIMARY KEY (symbol, event_date, event_type)
);

-- Dragon-tiger board
CREATE TABLE dragon_tiger (
    trade_date DATE,
    symbol VARCHAR(10),
    seat_name VARCHAR(100),
    buy_amount DOUBLE, sell_amount DOUBLE,
    seat_type VARCHAR(20),  -- institution, retail, quant
    PRIMARY KEY (trade_date, symbol, seat_name)
);
```

### 8.5 Cache Strategy

```
L1: Python lru_cache (in-process, ~ns)
  - Real-time quotes (3s TTL)
  - Stock basic info (24h TTL)
  - Recent kline queries (5min TTL)

L2: DuckDB (local file, ~0.1ms)
  - Historical kline, indicators, financials
  - Pre-computed strategy signals

L3: Redis (network, ~0.5ms) — OPTIONAL, enable via CACHE_BACKEND=redis
  - Only for multi-instance deployments
```

### 8.6 Key Event Data Module

AKShare endpoints for the missing event data:

| Event | AKShare Function | Importance |
|-------|-----------------|------------|
| Block trades | `ak.stock_dzjy_mrmx()` | Institutional flow stealth tracker |
| Lockup expiry | `ak.stock_restricted_release_queue_summary_sina()` | Pre-expiry sell-off risk |
| Insider trades | `ak.stock_share_hold_change()` | Strong buy/sell signal |
| Pledge ratio | `ak.stock_gpzy_pledge_ratio()` | Crash risk indicator |
| Institutional visits | `ak.stock_institute_research()` | Pre-move intelligence |

Exposed as `get_major_events(symbol)` — returning all event data in one call.

---

## 9. Degradation Plan

```
Level 0 (Full): LLM Agent + Strategy signals + Real-time data
  → All systems operational

Level 1 (LLM degraded): Rule-based signals + Real-time data
  → Trigger: LLM API unavailable or rate-limited
  → Agent tools return rule-based analysis instead of LLM-generated

Level 2 (Data degraded): Cached data + Rule-based signals
  → Trigger: Both AKShare and TickFlow unavailable
  → Return last successful cache + stale_data_warning
  → Stale data max age: 24h

Level 3 (Minimal): Cached data + Pre-computed signals only
  → Trigger: All data sources down for >24h
  → Query-only mode, no new analysis
```

Health endpoint: `GET /health` returns per-component status:
```json
{
  "duckdb": "ok",
  "akshare": "degraded",
  "tickflow": "ok",
  "llm_api": "ok", 
  "cache": "ok",
  "overall": "degraded"
}
```

---

## 10. Development Phases

### Phase 1: Data Foundation (Week 1)
- AKShare adapter (15 endpoints)
- TickFlow adapter (8 endpoints, fallback)
- DuckDB schema + store + quality validation
- Event data module (5 data types)
- lru_cache + optional Redis
- Cron scheduler (8-phase pipeline)
- 5 market data MCP Tools

### Phase 2: Engine (Week 2)
- TA-Lib + pandas-ta indicators (150+)
- 20 core strategies (class + YAML declarative)
- Market state detector
- Signal aggregation
- vectorbt + DuckDB SQL backtest
- A-share reality constraints
- 8 screener + analysis Tools

### Phase 3: Risk + Agent (Week 2-3)
- Hard RiskGuard (non-bypassable)
- Systemic risk early warning
- Fast mode Agent (single call)
- Deep mode Agent (asyncio orchestration)
- Bull-bear self-debate
- Long-term memory (SQLite)
- Token optimizer
- 8 strategy + agent Tools

### Phase 4: Info + Polish (Week 3)
- News aggregation (CaiLianShe, JuChao)
- Sentiment analysis (bardsai/finance-sentiment-zh-base)
- Event-driven theme extraction
- 7 review + alert + portfolio Tools

### Phase 5: Deploy + Test (Week 3-4)
- Multi-stage Dockerfile
- docker-compose (3 services)
- Full test suite (unit 75%, integration 20%, E2E 5%)
- CI/CD (lint, test, build, publish)
- Performance benchmarks
- README + DISCLAIMER.md

---

## 11. Risk & Mitigation

| Risk | Mitigation |
|------|-----------|
| Free data source instability | Dual-source fallback (AKShare→TickFlow) + 3-level degradation |
| AKShare interface changes | Adapter isolation + version pin + daily CI health check |
| LLM hallucination | Self-debate + evidence tagging + memory-based bias detection |
| Backtest overfitting | Walk-forward validation + parameter sensitivity heatmap + OOS testing |
| Limit-up strategy backtest-reality gap | Realistic liquidity constraints (buyability probability model) |
| Single maintainer burnout | 20 strategies (not 40) + YAML declarative extension + clear module boundaries |
| Compliance/legal | Strength scores not buy/sell + disclaimer on every response + research-only positioning |
| DuckDB write lock during updates | Shadow table strategy (write _new, atomic rename) |
| Token cost explosion | Dual mode (fast/deep) + token budget ceiling per analysis |

---

## 12. Success Criteria (v2.0)

- [ ] 25+ MCP Tools all callable
- [ ] 20 core strategies output strength scores
- [ ] Backtest engine simulates limit-up buyability, limit-down sellability, liquidity constraints
- [ ] Hard RiskGuard layer blocks rule violations (verified via intentional violation test)
- [ ] Agent fast mode <15s, deep mode <60s per stock
- [ ] Full market 20-strategy scan <10s (via pre-computation)
- [ ] `docker compose up` single command startup
- [ ] Dual data source fallback tested
- [ ] 3-level degradation tested (mock API failures)
- [ ] Daily auto-update success rate >95%
- [ ] All responses include disclaimer + data_freshness + data_source
