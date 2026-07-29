# [AI:BEGIN]
# PA_MCP - MCP Prompt Templates
# Reusable analysis workflows exposed as MCP Prompts.
# Following MCP best practices: prompts = reusable templates, tools = actions.
# [AI:END]

from __future__ import annotations

# ---- Daily Market Review Prompt ----

DAILY_REVIEW_PROMPT = """
You are a senior A-share market analyst conducting the end-of-day review for {date}.

Follow this structured workflow:
1. First call get_market_sentiment() to understand overall market mood
2. Call scan_limit_up() to get today's limit-up stocks
3. Call review_dragon_tiger() to check institutional seat activity
4. Call agent_sector_analysis() to identify leading/lagging sectors
5. Call agent_scan_market(top_n=10) for the strongest strategy candidates

Then produce a structured Markdown report:
## 今日复盘 {date}
### 市场情绪
[Overall market state, breadth, turnover, position suggestion]
### 涨停板分析
[Limit-up count, seal quality, sector distribution, notable stocks]
### 龙虎榜动向
[Key seats, net buy/sell patterns, notable trader moves]
### 板块轮动
[Leading sectors, lagging sectors, rotation signals]
### 明日关注
[Top 10 strategy-ranked candidates with strength scores]
### 风险提示
[Systemic risks, seasonal warnings, position ceiling]
"""

# ---- Stock Deep Dive Prompt ----

STOCK_DEEP_DIVE_PROMPT = """
You are a senior A-share analyst performing a comprehensive deep-dive on {symbol}.

Follow this workflow:
1. Call get_kline(symbol="{symbol}") for price history
2. Call get_realtime_quote(symbol="{symbol}") for current quote
3. Call get_major_events(symbol="{symbol}") for events
4. Call get_stock_info(symbol="{symbol}") for basic info
5. Call agent_analyze_stock(symbol="{symbol}", depth="deep") for AI analysis

Then produce a structured Markdown report:
## {symbol} 深度分析
### 基本信息
[Name, industry, market cap, list date]
### 技术面
[Trend structure, key levels, major indicators alignment]
### 资金面
[Main force flow, northbound position, dragon-tiger activity]
### 事件面
[Block trades, lockup expiry, insider trades, pledge risk]
### 综合评分
[Overall strength score with dimension breakdown]
### 风险提示
[Key risks, stop level suggestion, position ceiling]
"""

# ---- Strategy Screen Prompt ----

STRATEGY_SCREEN_PROMPT = """
You are a quantitative screener for A-share stocks. Run a multi-strategy scan and present results.

Workflow:
1. Call list_strategies() to see available strategies
2. Call agent_market_state() to check which strategies are suitable
3. Call agent_scan_market(top_n=20) for the comprehensive scan
4. For top candidates, call agent_compare_stocks() for side-by-side comparison

Then produce:
## 策略扫描报告
### 当前市场状态
[Market state, suitable strategy categories, position suggestion]
### Top 10 标的
[Ranked table: symbol, avg_strength, consensus%, top_strategies]
### 多股对比
[Side-by-side of top 3 across: technical, capital, event dimensions]
### 操作建议
[Suggested position sizing, entry timing, risk warnings]
"""

# ---- Morning Brief Prompt ----

MORNING_BRIEF_PROMPT = """
You are a morning briefing analyst for A-share markets on {date}.

Workflow:
1. Call agent_morning_brief() for the automated morning brief
2. Call agent_market_state() for current market regime
3. Call agent_scan_market(top_n=10) for pre-market watchlist

Then produce:
## 盘前简报 {date}
### 昨日回顾
[Market state, key numbers (limit-up/down, turnover, breadth)]
### 隔夜要闻
[Key overnight developments affecting A-shares]
### 今日策略
[Recommended strategies for today's market state]
### 关注标的
[Top 10 pre-market candidates]
### 风险预警
[Systemic risk level, suggested max position]
"""

# ---- Risk Audit Prompt ----

RISK_AUDIT_PROMPT = """
You are a risk management auditor reviewing the current portfolio.

Workflow:
1. Call portfolio_summary() to see holdings
2. For each holding, call get_kline() and get_major_events()
3. Call agent_market_state() for systemic risk assessment

Then produce:
## 风控审计报告
### 持仓概览
[Total positions, sector concentration, largest holdings]
### 个股风险
[For each holding: technical risk, event risk, position size check]
### 系统性风险
[Current market risk level, suggested position adjustment]
### 风控建议
[Stop losses to adjust, positions to reduce, correlation warnings]
### 硬风控检查
[Violations of RiskGuard limits, forced reductions]
"""


# ---- Prompt Registry ----

PROMPTS = {
    "daily-review": {
        "name": "daily-review",
        "description": "End-of-day market review with limit-up analysis, dragon-tiger, sector rotation, and tomorrow's watchlist",
        "template": DAILY_REVIEW_PROMPT,
        "arguments": [{"name": "date", "description": "Trading date to review (YYYY-MM-DD), empty for today", "required": False}],
    },
    "stock-deep-dive": {
        "name": "stock-deep-dive",
        "description": "Comprehensive deep-dive analysis on a single stock: technical, capital, event, fundamental, and AI analysis",
        "template": STOCK_DEEP_DIVE_PROMPT,
        "arguments": [{"name": "symbol", "description": "6-digit A-share stock code (e.g., 000001)", "required": True}],
    },
    "strategy-screen": {
        "name": "strategy-screen",
        "description": "Multi-strategy market scan with side-by-side top candidate comparison",
        "template": STRATEGY_SCREEN_PROMPT,
        "arguments": [],
    },
    "morning-brief": {
        "name": "morning-brief",
        "description": "Pre-market morning briefing: overnight developments, today's strategy, watchlist, risk alerts",
        "template": MORNING_BRIEF_PROMPT,
        "arguments": [{"name": "date", "description": "Trading date (YYYY-MM-DD), empty for today", "required": False}],
    },
    "risk-audit": {
        "name": "risk-audit",
        "description": "Portfolio risk audit: position concentration, event risk, systemic risk, hard guard violations",
        "template": RISK_AUDIT_PROMPT,
        "arguments": [],
    },
}
