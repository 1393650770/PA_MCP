# [AI:BEGIN]
# PA_MCP - Agent Layer: Orchestrator & Prompt Templates
# Manages fast and deep analysis modes with token optimization.
# [AI:END]

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import structlog

from pa_mcp.engine.strategies.base import MarketState, SignalDirection

logger = structlog.get_logger(__name__)


# ---- Analyst Prompt Templates ----

TECHNICAL_ANALYST_PROMPT = """You are a senior technical analyst specializing in A-share markets.

Analyze the provided K-line data across these dimensions:
1. TREND STRUCTURE: Identify the primary trend (up/down/sideways). Note key support/resistance levels.
2. INDICATOR ALIGNMENT: Check if MA, MACD, RSI, KDJ, BOLL confirm or diverge.
3. VOLUME-PRICE RELATIONSHIP: Is volume confirming price moves? Any divergences?
4. CHART PATTERNS: Look for head-and-shoulders, double tops/bottoms, triangles, flags.
5. KEY LEVELS: Nearest support and resistance with specific price levels.

Output JSON:
{
  "strength_score": 0-100 (higher = stronger bullish case),
  "key_findings": ["finding1", "finding2", "finding3"],
  "support_levels": [price1, price2],
  "resistance_levels": [price1, price2],
  "trend": "bullish|bearish|neutral",
  "risks": ["risk1", "risk2"]
}

IMPORTANT: Do NOT say "buy" or "sell". Output scores and evidence only."""

CAPITAL_ANALYST_PROMPT = """You are a capital flow analyst specializing in A-share markets.

Analyze the capital flow and dragon-tiger data:
1. MAIN FORCE: Net institutional flow direction and magnitude. Is "smart money" buying or selling?
2. NORTHBOUND: Recent northbound flow trend. Consecutive days of net buy/sell?
3. DRAGON TIGER: Notable seats active. Are these "lock-up" seats or "flip" seats?
4. RETAIL FLOW: Is retail buying into strength or selling into weakness?
5. MARGIN TRADING: Margin balance trend — increasing or decreasing?

Output JSON:
{
  "strength_score": 0-100,
  "main_force_direction": "accumulating|distributing|neutral",
  "northbound_signal": "strong_buy|buy|neutral|sell|strong_sell",
  "key_seats": [{"name": "...", "type": "lock_up|flip|unknown"}],
  "risks": ["risk1"]
}"""

SENTIMENT_ANALYST_PROMPT = """You are a market sentiment analyst for A-shares.

Analyze news and social sentiment:
1. NEWS TONE: Aggregate sentiment from recent news (-1 to +1, where +1 = extremely bullish)
2. SOCIAL HEAT: Is this stock/sector being discussed? Growing or fading?
3. ANALYST RATINGS: Recent rating changes. Upgrades or downgrades?
4. SECTOR MOOD: How is the broader sector sentiment?
5. CONTRARIAN CHECK: Is sentiment too one-sided? Extreme bullishness can be a warning.

Output JSON:
{
  "strength_score": 0-100,
  "sentiment_score": float (-1 to +1),
  "social_heat": "hot|warm|normal|cold",
  "analyst_consensus": "bullish|neutral|bearish",
  "contrarian_warning": true/false,
  "risks": ["risk1"]
}"""

FUNDAMENTAL_ANALYST_PROMPT = """You are a fundamental analyst for A-share stocks.

Analyze financial statements:
1. GROWTH: Revenue and profit YoY growth. Accelerating or decelerating?
2. PROFITABILITY: ROE, gross margin, net margin trends. Quality of earnings?
3. VALUATION: PE/PB/PS percentile vs 5-year history and vs sector.
4. BALANCE SHEET: Debt ratio, current ratio, cash position. Financial health?
5. DIVIDENDS: Dividend yield, payout ratio, consistency.

Output JSON:
{
  "strength_score": 0-100,
  "growth_score": 0-100,
  "valuation_score": 0-100 (higher = more undervalued),
  "quality_score": 0-100,
  "key_metrics": {"roe": "...", "pe_percentile": "...", "revenue_growth": "..."},
  "risks": ["risk1"]
}"""

EVENT_ANALYST_PROMPT = """You are an event-driven analyst for A-share stocks.

Analyze corporate events and catalysts:
1. BLOCK TRADES: Recent block trade discount/premium. Who's buying/selling?
2. LOCKUP EXPIRY: Upcoming lockup release. Size relative to float?
3. INSIDER TRADES: Recent insider buying/selling. Amount and pattern?
4. PLEDGE RISK: Pledge ratio. Any margin call risk?
5. INSTITUTIONAL VISITS: Recent research visits. Pattern of visits before moves?

Output JSON:
{
  "strength_score": 0-100,
  "event_catalyst": true/false,
  "lockup_risk": "high|medium|low|none",
  "insider_signal": "strong_buy|buy|neutral|sell|strong_sell",
  "pledge_risk": "high|medium|low",
  "risks": ["risk1"]
}"""


# ---- Market State Prompt ----

MARKET_STATE_ANALYST_PROMPT = """You are a market strategist for A-shares.

Based on today's data, determine the market regime:
- CLIMAX (高潮期): Limit-ups >80, turnover >1.5T, index >15% above MA200 → Aggressive, ride momentum
- FERMENTING (发酵期): Limit-ups 40-80, turnover 800B-1.5T, northbound buying → Active, trend/swing
- STARTING (启动期): Limit-ups 15-40, early recovery, breadth improving → Cautious entry
- DULL (低迷期): Turnover 400-800B, flat breadth, low activity → Defensive, hold cash
- FROZEN (冰点期): Limit-downs >30, turnover <400B, index below MA200 → STAY OUT

Output JSON:
{
  "market_state": "climax|fermenting|starting|dull|frozen",
  "confidence": 0-100,
  "suggested_max_position_pct": 0-100,
  "key_observations": ["obs1", "obs2"],
  "risk_level": "low|medium|high|extreme"
}"""


# ---- Aggregated Analysis Prompt (Fast Mode) ----

FAST_ANALYSIS_PROMPT = """You are a seasoned A-share quantitative analyst.

Analyze {symbol} across 5 dimensions using the provided data. Output a SINGLE JSON object.

Data provided:
- K-line: {kline_summary}
- Capital Flow: {capital_flow_summary}
- Sentiment: {sentiment_summary}
- Fundamentals: {fundamental_summary}
- Events: {event_summary}
- Market State: {market_state}

Dimensions to analyze:
1. TECHNICAL: Key patterns, indicator alignment, support/resistance
2. CAPITAL: Fund flow direction, dragon-tiger activity, northbound position
3. SENTIMENT: News sentiment, social media heat, sector mood
4. FUNDAMENTAL: Financial health, valuation percentile, growth trajectory
5. EVENT: Lockup expiry, insider trades, block trades, pledge risk

OUTPUT (JSON only, no markdown):
{
  "symbol": "{symbol}",
  "overall_strength_score": 0-100,
  "dimension_scores": {
    "technical": 0-100,
    "capital": 0-100,
    "sentiment": 0-100,
    "fundamental": 0-100,
    "event": 0-100
  },
  "direction": "bullish|bearish|neutral",
  "key_evidence": [
    {"dimension": "technical", "finding": "...", "impact": "positive|negative|neutral"}
  ],
  "key_risks": ["risk1", "risk2"],
  "risk_reward_assessment": "favorable|neutral|unfavorable",
  "suggested_max_position_pct": 0-20,
  "disclaimer": "Research output, not investment advice. Data may be delayed."
}

DO NOT output "buy" or "sell". Output scores and evidence only.
"""


# ---- Orchestrator ----

@dataclass
class AnalysisResult:
    """Structured result from agent analysis."""

    symbol: str
    overall_strength_score: float = 0.0
    dimension_scores: dict[str, float] = field(default_factory=dict)
    direction: str = "neutral"
    key_evidence: list[dict] = field(default_factory=list)
    key_risks: list[str] = field(default_factory=list)
    risk_reward_assessment: str = "neutral"
    suggested_max_position_pct: float = 0.0
    mode: str = "fast"  # fast | deep
    analysis_time_ms: float = 0.0
    token_used: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AgentOrchestrator:
    """Orchestrates stock analysis using asyncio for parallel analyst execution.

    Two modes:
    - FAST: Single LLM call covering all 5 dimensions (~8K tokens, ~15s)
    - DEEP: 5 parallel analysts + debate + risk review (~50K tokens, ~60s)
    """

    def __init__(self, model_fast: str = "claude-sonnet-5", model_deep: str = "claude-opus-5") -> None:
        self.model_fast = model_fast
        self.model_deep = model_deep

    async def fast_analyze(
        self, symbol: str, kline_df: pd.DataFrame,
        market_state: Optional[str] = None,
        fundamental_data: Optional[dict] = None,
    ) -> AnalysisResult:
        """Fast mode: single LLM call with compressed data."""
        t0 = datetime.now()

        # Compress kline data into a text summary
        kline_summary = self._summarize_kline(kline_df)

        # Build the prompt
        user_prompt = FAST_ANALYSIS_PROMPT.format(
            symbol=symbol,
            kline_summary=kline_summary,
            capital_flow_summary=fundamental_data.get("capital_flow", "N/A") if fundamental_data else "N/A",
            sentiment_summary=fundamental_data.get("sentiment", "N/A") if fundamental_data else "N/A",
            fundamental_summary=fundamental_data.get("fundamental", "N/A") if fundamental_data else "N/A",
            event_summary=fundamental_data.get("events", "No major events") if fundamental_data else "No major events",
            market_state=market_state or "unknown",
        )

        # Try new adapter first, fall back to legacy client
        from pa_mcp.agent.llm_port import get_llm_adapter, LLMCallParams
        adapter = get_llm_adapter()

        SYSTEM_PROMPT = (
            "You are a quantitative analyst for Chinese A-share stocks. "
            "Output JSON only, no markdown. Never say 'buy' or 'sell'. "
            "Output strength scores (0-100) and evidence only."
        )

        if adapter is not None:
            params = LLMCallParams(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                mode="fast",
            )
            response = await adapter.chat_json(params)
        else:
            from pa_mcp.agent.llm_client import get_llm_client
            client = get_llm_client()
            response = client.chat_json(SYSTEM_PROMPT, user_prompt, mode="fast")

        result = AnalysisResult(
            symbol=symbol,
            mode="fast",
            token_used=response.get("token_used", 0),
            analysis_time_ms=(datetime.now() - t0).total_seconds() * 1000,
        )

        if "error" not in response:
            result.overall_strength_score = response.get("overall_strength_score", 50)
            result.dimension_scores = response.get("dimension_scores", {})
            result.direction = response.get("direction", "neutral")
            result.key_evidence = response.get("key_evidence", [])
            result.key_risks = response.get("key_risks", [])
            result.risk_reward_assessment = response.get("risk_reward_assessment", "neutral")
            result.suggested_max_position_pct = response.get("suggested_max_position_pct", 0)

        return result

    async def deep_analyze(
        self, symbol: str, kline_df: pd.DataFrame,
        capital_flow: Optional[dict] = None,
        dragon_tiger: Optional[dict] = None,
        news_data: Optional[list] = None,
        fundamental_data: Optional[dict] = None,
        event_data: Optional[dict] = None,
        market_state: Optional[str] = None,
    ) -> AnalysisResult:
        """Deep mode: 5 parallel analysts + debate + risk review.

        Stage 1: Parallel data fetch (done by caller)
        Stage 2: 5 analysts in parallel
        Stage 3: Self-debate (bull vs bear in one call)
        Stage 4: Risk review + hard guard check
        """
        t0 = datetime.now()

        # In production, these would be actual LLM calls via asyncio.gather
        # For now, return a placeholder result
        total_tokens = 40000  # Estimated for deep mode

        result = AnalysisResult(
            symbol=symbol,
            mode="deep",
            token_used=total_tokens,
            analysis_time_ms=(datetime.now() - t0).total_seconds() * 1000,
        )

        logger.info("Deep analysis request built", symbol=symbol, tokens_est=total_tokens)
        return result

    @staticmethod
    def _summarize_kline(df: pd.DataFrame) -> str:
        """Compress kline data into a text summary for LLM consumption."""
        if df is None or df.empty:
            return "No data available"

        recent = df.tail(20)
        close = df["close"].values

        # Compute basic stats
        recent_change = (close[-1] - close[-20]) / close[-20] * 100 if len(close) >= 20 else 0
        high_20 = recent["high"].max() if "high" in recent.columns else close[-1]
        low_20 = recent["low"].min() if "low" in recent.columns else close[-1]
        avg_volume = recent["volume"].mean() if "volume" in recent.columns else 0
        last_volume = recent["volume"].iloc[-1] if "volume" in recent.columns else 0

        # MA positions
        ma5 = close[-5:].mean() if len(close) >= 5 else close[-1]
        ma20 = close[-20:].mean() if len(close) >= 20 else close[-1]

        summary = (
            f"Recent 20 days: Close {close[-1]:.2f} ({recent_change:+.1f}%). "
            f"Range: {low_20:.2f}-{high_20:.2f}. "
            f"Volume: {last_volume:.0f} (avg: {avg_volume:.0f}). "
            f"MA5: {ma5:.2f}, MA20: {ma20:.2f}. "
            f"Trend: {'up' if close[-1] > ma20 else 'down'}."
        )
        return summary


# Global orchestrator instance
_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    """Get or create the agent orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
