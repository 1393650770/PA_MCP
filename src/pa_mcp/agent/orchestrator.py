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


# ---- 市场状态 → 策略路由（两阶段：诊断 → 路由 → 决策） ----

MARKET_STATE_STRATEGY_ROUTING: dict[str, dict[str, list[str]]] = {
    "climax": {
        "label": "高潮期——情绪亢奋，跟随强势，注意止盈",
        "strategies": ["volume_price_momentum", "first_board_breakout",
                       "platform_breakout", "dragon_second_wave"],
        "risk_notes": "追高风险大，仓位上限收紧，避免追最后一棒",
    },
    "fermenting": {
        "label": "发酵期——赚钱效应扩散，趋势与波段并重",
        "strategies": ["ma_golden_cross", "volume_price_momentum",
                       "platform_breakout", "macd_divergence_swing"],
        "risk_notes": "参与度高，注意板块轮动节奏",
    },
    "starting": {
        "label": "启动期——底部回暖，左侧布局与反转为主",
        "strategies": ["oversold_bounce", "bollinger_mean_reversion",
                       "macd_divergence_swing"],
        "risk_notes": "确认信号再进场，避免过早重仓",
    },
    "dull": {
        "label": "低迷期——量能萎缩，防御优先",
        "strategies": ["roe_pb_value", "range_grid", "bollinger_mean_reversion"],
        "risk_notes": "降低仓位，以低波动标的为主",
    },
    "frozen": {
        "label": "冰点期——空头主导，空仓观望",
        "strategies": ["roe_pb_value", "oversold_bounce"],
        "risk_notes": "严格限制仓位，等待情绪修复信号",
    },
}

MARKET_STATE_ZH: dict[str, str] = {
    "climax": "高潮期", "fermenting": "发酵期", "starting": "启动期",
    "dull": "低迷期", "frozen": "冰点期",
}


def route_strategy_by_market_state(market_state: str) -> dict[str, Any]:
    """市场状态 → 策略路由建议（确定性规则，供诊断/UI/MCP 复用）。"""
    return MARKET_STATE_STRATEGY_ROUTING.get(
        market_state,
        MARKET_STATE_STRATEGY_ROUTING["dull"] | {"_unknown_state": market_state},
    )


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
    # 两阶段分析：市场诊断结果（Stage 1 输出，注入分析 prompt）
    market_diagnosis: Optional[dict] = None
    # 辩论阶段（TradingAgents 风格）：多头/空头论点 + 投资大师裁定
    debate: Optional[dict] = None
    master_verdict: Optional[dict] = None


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

        # RAG 增强：注入历史参考案例（同标的最近分析结论 + 事后验证）
        history_text = self._build_experience_context(symbol)

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
        if history_text:
            user_prompt += f"\n\n{history_text}"

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

        # 自动写入经验库（best-effort，不影响主流程）
        self._save_to_experience(symbol, result)
        return result

    # ---- 两阶段：市场诊断（Stage 1） ----

    async def market_diagnosis(self, market_context: Optional[dict] = None) -> Optional[dict]:
        """Stage 1 市场诊断：判断市场状态 + 给出策略路由建议。

        market_context: 可选的市场观测（涨停数/成交额/北向/指数均线等原始数据），
        由调用方提供（如 agent_market_state 的上下文）。无 LLM 时返回确定性诊断。
        """
        from pa_mcp.agent.llm_port import get_llm_adapter, LLMCallParams
        adapter = get_llm_adapter()
        if adapter is None:
            return self._diagnosis_deterministic(market_context)

        ctx = "未提供市场观测数据" if not market_context else str(market_context)[:1500]
        params = LLMCallParams(
            system_prompt=(
                "你是有经验的 A 股市场策略师。只输出合法 JSON。"
                "输出是研究参考，不是投资建议。"
            ),
            user_prompt=(
                f"{MARKET_STATE_ANALYST_PROMPT}\n\n【市场观测】\n{ctx}"
            ),
            mode="fast", max_tokens=600,
        )
        raw = await self._chat_json_with_retry(
            adapter, params, self._validate_diagnosis_json)
        if raw is None:
            return self._diagnosis_deterministic(market_context)

        state = str(raw.get("market_state", "dull")).lower()
        if state not in MARKET_STATE_STRATEGY_ROUTING:
            state = "dull"
        routing = route_strategy_by_market_state(state)
        diagnosis = {
            "market_state": state,
            "market_state_zh": MARKET_STATE_ZH[state],
            "confidence": raw.get("confidence", 60),
            "suggested_max_position_pct": raw.get("suggested_max_position_pct", 50),
            "risk_level": raw.get("risk_level", "medium"),
            "key_observations": raw.get("key_observations", []),
            "strategy_routing": routing,
        }
        return diagnosis

    @staticmethod
    def _diagnosis_deterministic(market_context: Optional[dict]) -> dict:
        """无 LLM 时的确定性诊断（基于成交额/涨跌统计，不编造）。"""
        state, notes = "dull", []
        try:
            if isinstance(market_context, dict):
                turnover = float(market_context.get("turnover_billion", 0) or 0)
                limit_up = int(market_context.get("limit_up_count", 0) or 0)
                limit_down = int(market_context.get("limit_down_count", 0) or 0)
                if limit_down > 30 or turnover < 400:
                    state = "frozen"
                    notes = [f"跌停 {limit_down} 家/成交额 {turnover:.0f} 亿，冰点特征"]
                elif turnover >= 1500 or limit_up > 80:
                    state = "climax"
                    notes = [f"成交额 {turnover:.0f} 亿/涨停 {limit_up} 家，情绪亢奋"]
                elif limit_up >= 40 or turnover >= 800:
                    state = "fermenting"
                    notes = [f"成交额 {turnover:.0f} 亿/涨停 {limit_up} 家，赚钱效应扩散"]
                elif limit_up >= 15:
                    state = "starting"
                    notes = [f"涨停 {limit_up} 家，局部回暖"]
                else:
                    notes = [f"成交额 {turnover:.0f} 亿/涨停 {limit_up} 家，市场低迷"]
        except Exception:
            notes = ["无可靠市场观测数据，保守判定低迷期"]
        return {
            "market_state": state,
            "market_state_zh": MARKET_STATE_ZH[state],
            "confidence": 55,
            "suggested_max_position_pct": 30 if state == "dull" else 50,
            "risk_level": "high" if state in ("frozen", "climax") else "medium",
            "key_observations": notes,
            "strategy_routing": route_strategy_by_market_state(state),
            "mode": "deterministic",
        }

    @staticmethod
    def _validate_diagnosis_json(raw: dict) -> list[str]:
        errors = []
        if str(raw.get("market_state", "")).lower() not in MARKET_STATE_STRATEGY_ROUTING:
            errors.append("market_state 必须是 climax/fermenting/starting/dull/frozen")
        try:
            conf = float(raw.get("confidence", -1))
            if not (0 <= conf <= 100):
                errors.append("confidence 应为 0-100")
        except Exception:
            errors.append("confidence 非数值")
        return errors

    # ---- JSON 校验 + 重试（借鉴 PA_Agent validation_retry 机制） ----

    @staticmethod
    async def _chat_json_with_retry(adapter, params: Any,
                                     validate_fn, max_retries: int = 1) -> Optional[dict]:
        """chat_json + 校验；校验失败把错误反馈给 LLM 重试。

        返回 None 表示最终失败（调用方自行降级）。
        """
        from pa_mcp.agent.llm_port import LLMCallParams
        raw = await adapter.chat_json(params)
        if not isinstance(raw, dict) or "error" in raw:
            return None
        errors = validate_fn(raw)
        if not errors:
            return raw
        retry = 0
        while retry < max_retries:
            retry += 1
            retry_params = LLMCallParams(
                system_prompt=params.system_prompt,
                user_prompt=(
                    f"{params.user_prompt}\n\n【校验失败，请修正后重新输出完整 JSON】\n"
                    + "\n".join(f"- {e}" for e in errors)
                    + "\n只输出修正后的 JSON。"
                ),
                mode=params.mode, max_tokens=params.max_tokens,
            )
            raw2 = await adapter.chat_json(retry_params)
            if not isinstance(raw2, dict) or "error" in raw2:
                continue
            errors2 = validate_fn(raw2)
            if not errors2:
                return raw2
        return None

    @staticmethod
    def _validate_analyst_json(raw: dict) -> list[str]:
        errors = []
        try:
            s = float(raw.get("strength_score", -1))
            if not (0 <= s <= 100):
                errors.append("strength_score 应为 0-100")
        except Exception:
            errors.append("strength_score 非数值")
        return errors

    @staticmethod
    def _validate_pm_json(raw: dict) -> list[str]:
        errors = []
        if str(raw.get("direction", "")).lower() not in ("bullish", "bearish", "neutral"):
            errors.append("direction 必须是 bullish/bearish/neutral")
        try:
            s = float(raw.get("overall_strength_score", -1))
            if not (0 <= s <= 100):
                errors.append("overall_strength_score 应为 0-100")
        except Exception:
            errors.append("overall_strength_score 非数值")
        return errors

    # ---- 两阶段组合入口：诊断 → 深度分析 ----

    async def analyze_with_diagnosis(
        self, symbol: str, kline_df: pd.DataFrame,
        market_context: Optional[dict] = None,
        debate: bool = False,
        **kwargs,
    ) -> AnalysisResult:
        """两阶段分析：Stage 1 市场诊断 → Stage 2 深度分析（诊断注入）。

        诊断结论（市场状态 + 策略路由）写入 AnalysisResult.market_diagnosis，
        并注入 5 位分析师的 prompt 作为市场环境上下文。
        debate=True 时额外执行 Bull/Bear 辩论 + 投资大师裁定。
        """
        diagnosis = await self.market_diagnosis(market_context)
        kwargs["diagnosis"] = diagnosis
        kwargs["debate"] = debate
        return await self.deep_analyze(symbol, kline_df, **kwargs)

    # 分析师角色配置（借鉴 ai-hedge-fund 的多 agent 模式）
    ANALYSTS = [
        ("technical", TECHNICAL_ANALYST_PROMPT,
         lambda k, c, d, f, e, m: f"K线数据（近60日）：\n{k}"),
        ("capital", CAPITAL_ANALYST_PROMPT,
         lambda k, c, d, f, e, m: f"资金面数据：\n主力资金流：{c or '无'}\n龙虎榜：{d or '无'}"),
        ("sentiment", SENTIMENT_ANALYST_PROMPT,
         lambda k, c, d, f, e, m: f"情绪数据：\n市场状态：{m or '未知'}\n新闻：{f.get('news', '无') if isinstance(f, dict) else '无'}"),
        ("fundamental", FUNDAMENTAL_ANALYST_PROMPT,
         lambda k, c, d, f, e, m: f"基本面数据：\n{f.get('fundamental', '无') if isinstance(f, dict) else '无'}"),
        ("event", EVENT_ANALYST_PROMPT,
         lambda k, c, d, f, e, m: f"事件数据：\n{f.get('events', '无') if isinstance(f, dict) else '无'}"),
    ]

    # ---- 辩论阶段（借鉴 TradingAgents bull/bear/debate 机制） ----

    BULL_ARGUMENT_PROMPT = """你是多头代表（Bull），为 {symbol} 的多头立场辩护。

    分析师结论：
    {analyst_results}

    你的任务（输出 JSON）：
    1. 从所有看涨证据中提炼 **3 个最强多头论点**（含数据支撑）
    2. 预判空头会攻击的 2 个点，并逐一反驳
    3. 给出你认为合理的**目标仓位上限**（0-20）

    输出 JSON（仅 JSON）：
    {{
      "bull_points": [{{"point": "...", "evidence": "..."}}],
      "bear_rebuttals": [{{"attack": "...", "rebuttal": "..."}}],
      "suggested_position_pct": 0-20
    }}"""

    BEAR_ARGUMENT_PROMPT = """你是空头代表（Bear），为 {symbol} 的空头立场辩护。

    分析师结论：
    {analyst_results}
    多头观点：
    {bull_arguments}

    你的任务（输出 JSON）：
    1. 从所有看跌证据中提炼 **3 个最强空头论点**（含数据支撑）
    2. 逐一反驳多头的 3 个论点
    3. 指出多头忽略的最大风险

    输出 JSON（仅 JSON）：
    {{
      "bear_points": [{{"point": "...", "evidence": "..."}}],
      "bull_rebuttals": [{{"attack": "...", "rebuttal": "..."}}],
      "biggest_missed_risk": "..."
    }}"""

    # 投资大师最终裁定（综合分析师 + 辩论，输出确定性结论）
    MASTER_VERDICT_PROMPT = """你是投资大师（综合格雷厄姆的安全边际、索罗斯的反身性与证伪、利弗莫尔的关键点与趋势），对 {symbol} 做最终裁定。

    分析师结论：
    {analyst_results}

    辩论观点：
    {debate_summary}

    裁定要求：
    1. 不追求完美预测，只追求「风险调整后赔率是否值得参与」
    2. 明确证伪条件（什么情况下你的判断是错的）
    3. 仓位建议必须考虑市场环境与集中度风险
    4. 输出 JSON（仅 JSON）：
    {{
      "final_direction": "bullish|bearish|neutral",
      "final_strength_score": 0-100,
      "suggested_max_position_pct": 0-20,
      "master_style": "价值|趋势|反身性|混合（简短说明采用了哪种思想）",
      "verdict_reason": "一句话结论（中文）",
      "key_evidence_used": ["使用的关键证据"],
      "falsification_conditions": ["什么情况下判断错误"],
      "final_risks": ["最终风险清单"]
    }}
    DO NOT output buy/sell orders. Output scores and evidence only."""

    # 组合经理 prompt（汇总 5 分析师）
    PORTFOLIO_MANAGER_PROMPT = """你是投资组合经理，汇总 5 位分析师对 {symbol} 的分析，做出综合决策。

分析师结论：
{analyst_results}

决策规则：
1. 综合各维度权重（技术40% 资金20% 情绪10% 基本面20% 事件10%）
2. 必须考虑风险：仓位建议不得超过 20%
3. 输出 JSON（仅 JSON）：
{{
  "overall_strength_score": 0-100,
  "dimension_scores": {{"technical": 0-100, "capital": 0-100, "sentiment": 0-100, "fundamental": 0-100, "event": 0-100}},
  "direction": "bullish|bearish|neutral",
  "key_evidence": [{{"dimension": "...", "finding": "...", "impact": "positive|negative|neutral"}}],
  "key_risks": ["..."],
  "risk_reward_assessment": "favorable|neutral|unfavorable",
  "suggested_max_position_pct": 0-20,
  "disclaimer": "Research output, not investment advice."
}}

DO NOT output buy/sell. Output scores and evidence only."""

    async def deep_analyze(
        self, symbol: str, kline_df: pd.DataFrame,
        capital_flow: Optional[dict] = None,
        dragon_tiger: Optional[dict] = None,
        news_data: Optional[list] = None,
        fundamental_data: Optional[dict] = None,
        event_data: Optional[dict] = None,
        market_state: Optional[str] = None,
        diagnosis: Optional[dict] = None,
        debate: bool = False,
    ) -> AnalysisResult:
        """Deep mode: 5 parallel LLM analysts + portfolio manager synthesis.

        Stage 1: Market diagnosis (optional, via analyze_with_diagnosis)
        Stage 2: 5 analysts in parallel (asyncio.gather) with JSON retry
        Stage 3: Portfolio manager synthesis (with JSON retry)
        Stage 4: Bull/Bear debate + master verdict (optional, TradingAgents 风格)
        Stage 5: RiskGuard clamp (position cap)
        """
        t0 = datetime.now()

        from pa_mcp.agent.llm_port import get_llm_adapter as _get_adapter

        adapter = _get_adapter()
        if adapter is None:
            # 无 LLM → 降级为确定性规则分析（不返回空占位）
            result = self._rule_based_deep(symbol, kline_df, market_state, fundamental_data)
            result.market_diagnosis = diagnosis
            return result

        from pa_mcp.agent.llm_port import LLMCallParams

        kline_summary = self._summarize_kline(kline_df)

        # 市场环境上下文（诊断注入：市场状态 + 策略路由）
        env_context = ""
        if diagnosis:
            env_context = (
                f"【市场环境】状态：{diagnosis.get('market_state_zh', diagnosis.get('market_state', ''))}"
                f"，风险等级：{diagnosis.get('risk_level', '')}"
                f"，建议总仓位上限：{diagnosis.get('suggested_max_position_pct', '')}%"
            )

        # Stage 2: 5 分析师并行（JSON 校验 + 一次重试）
        async def run_analyst(dim: str, prompt: str, data_builder) -> tuple[str, dict]:
            user_data = data_builder(kline_summary, capital_flow, dragon_tiger,
                                     fundamental_data, event_data, market_state)
            if env_context:
                user_data = f"{env_context}\n{user_data}"
            params = LLMCallParams(
                system_prompt=prompt, user_prompt=user_data,
                mode="fast", max_tokens=800,
            )
            resp = await self._chat_json_with_retry(
                adapter, params, self._validate_analyst_json)
            return dim, resp

        results = await asyncio.gather(*[
            run_analyst(dim, prompt, builder)
            for dim, prompt, builder in self.ANALYSTS
        ], return_exceptions=True)

        analyst_results = {}
        for r in results:
            if isinstance(r, Exception) or r is None:
                continue
            dim, data = r
            if isinstance(data, dict) and "error" not in data:
                analyst_results[dim] = data

        if not analyst_results:
            return self._rule_based_deep(symbol, kline_df, market_state, fundamental_data)

        # Stage 3: 组合经理汇总（RAG 增强：注入历史参考案例）
        analyst_text = "\n".join(
            f"[{dim}] 分数={data.get('strength_score', 50)} "
            f"风险={data.get('risks', [])}"
            for dim, data in analyst_results.items()
        )
        history_text = self._build_experience_context(symbol)
        pm_user = f"请对 {symbol} 做综合决策"
        if history_text:
            pm_user += f"\n\n{history_text}"
        pm_params = LLMCallParams(
            system_prompt=self.PORTFOLIO_MANAGER_PROMPT.format(
                symbol=symbol, analyst_results=analyst_text),
            user_prompt=pm_user,
            mode="deep", max_tokens=1200,
        )
        pm_resp = await self._chat_json_with_retry(
            adapter, pm_params, self._validate_pm_json)

        result = AnalysisResult(
            symbol=symbol,
            mode="deep",
            token_used=sum(
                r.get("token_used", 0) for r in analyst_results.values()
            ) if analyst_results else 0,
            analysis_time_ms=(datetime.now() - t0).total_seconds() * 1000,
        )
        result.market_diagnosis = diagnosis

        if isinstance(pm_resp, dict) and "error" not in pm_resp:
            result.overall_strength_score = float(pm_resp.get("overall_strength_score", 50))
            result.dimension_scores = pm_resp.get("dimension_scores", {})
            result.direction = pm_resp.get("direction", "neutral")
            result.key_evidence = pm_resp.get("key_evidence", [])
            result.key_risks = pm_resp.get("key_risks", [])
            result.risk_reward_assessment = pm_resp.get("risk_reward_assessment", "neutral")
            # Stage 4: RiskGuard 仓位上限（不可绕过）
            suggested = float(pm_resp.get("suggested_max_position_pct", 5))
            result.suggested_max_position_pct = min(max(suggested, 0), 20)
        else:
            # 组合经理失败 → 用分析师平均
            scores = [float(d.get("strength_score", 50)) for d in analyst_results.values()]
            result.overall_strength_score = round(sum(scores) / len(scores), 1)
            result.dimension_scores = {
                dim: float(d.get("strength_score", 50))
                for dim, d in analyst_results.items()
            }
            result.direction = "bullish" if result.overall_strength_score >= 60 else (
                "bearish" if result.overall_strength_score <= 40 else "neutral")
            result.suggested_max_position_pct = 5.0

        # Stage 4: Bull/Bear 辩论 + 投资大师裁定（可选，TradingAgents 风格）
        if debate and analyst_results:
            await self._run_debate(
                symbol, analyst_results, result, pm_resp,
                diagnosis=diagnosis, history_text=history_text)

        # 自动写入经验库（best-effort）
        self._save_to_experience(symbol, result)

        logger.info(
            "Deep analysis complete",
            symbol=symbol, analysts=len(analyst_results),
            score=result.overall_strength_score,
            elapsed_ms=round(result.analysis_time_ms),
        )
        return result

    # ---- 辩论 + 投资大师裁定（TradingAgents 风格） ----

    @staticmethod
    def _validate_bull_bear_json(raw: dict) -> list[str]:
        errors = []
        if not isinstance(raw.get("bull_points", raw.get("bear_points", [])), list) \
                or not raw.get("bull_points", raw.get("bear_points", [])):
            errors.append("缺少论点数组（bull_points 或 bear_points）")
        return errors

    @staticmethod
    def _validate_master_json(raw: dict) -> list[str]:
        errors = []
        if str(raw.get("final_direction", "")).lower() not in ("bullish", "bearish", "neutral"):
            errors.append("final_direction 必须是 bullish/bearish/neutral")
        try:
            s = float(raw.get("final_strength_score", -1))
            if not (0 <= s <= 100):
                errors.append("final_strength_score 应为 0-100")
        except Exception:
            errors.append("final_strength_score 非数值")
        return errors

    async def _run_debate(self, symbol: str, analyst_results: dict,
                          result: AnalysisResult, pm_resp: Optional[dict],
                          diagnosis: Optional[dict] = None,
                          history_text: str = "") -> None:
        """Bull 论证 → Bear 反驳 → 投资大师裁定；失败不影响主流程。"""
        try:
            from pa_mcp.agent.llm_port import get_llm_adapter, LLMCallParams
            adapter = get_llm_adapter()
            if adapter is None:
                return
            if pm_resp is not None and "error" not in pm_resp:
                pm_line = (f"组合经理结论：方向={pm_resp.get('direction')} "
                           f"强度={pm_resp.get('overall_strength_score')} "
                           f"仓位上限={pm_resp.get('suggested_max_position_pct')}%")
            else:
                pm_line = "组合经理结论：未生成（降级用分析师均值）"
            analyst_text = "\n".join(
                f"[{dim}] 分数={d.get('strength_score', 50)} 风险={d.get('risks', [])}"
                for dim, d in analyst_results.items()
            ) + f"\n{pm_line}"
            if history_text:
                analyst_text += f"\n\n{history_text}"

            # 1) Bull 论证
            bull_params = LLMCallParams(
                system_prompt=self.BULL_ARGUMENT_PROMPT.format(
                    symbol=symbol, analyst_results=analyst_text),
                user_prompt=f"请为 {symbol} 的多头立场辩护",
                mode="fast", max_tokens=900,
            )
            bull = await self._chat_json_with_retry(
                adapter, bull_params, self._validate_bull_bear_json)
            bull_txt = "多头论证不可用"
            if bull:
                bull_txt = "\n".join(
                    f"- {p.get('point', '')}（{p.get('evidence', '')}）"
                    for p in bull.get("bull_points", []))
                bull_txt += "\n空头反驳预案：" + "; ".join(
                    f"{r.get('attack', '')} → {r.get('rebuttal', '')}"
                    for r in bull.get("bear_rebuttals", []))

            # 2) Bear 反驳（能看到多头论点）
            bear_params = LLMCallParams(
                system_prompt=self.BEAR_ARGUMENT_PROMPT.format(
                    symbol=symbol, analyst_results=analyst_text,
                    bull_arguments=bull_txt),
                user_prompt=f"请反驳 {symbol} 的多头观点并论证空头立场",
                mode="fast", max_tokens=900,
            )
            bear = await self._chat_json_with_retry(
                adapter, bear_params, self._validate_bull_bear_json)
            bear_txt = "空头论证不可用"
            if bear:
                bear_txt = "\n".join(
                    f"- {p.get('point', '')}（{p.get('evidence', '')}）"
                    for p in bear.get("bear_points", []))
                bear_txt += "\n最大遗漏风险：" + str(bear.get("biggest_missed_risk", ""))
                bear_txt += "\n对多头反驳：" + "; ".join(
                    f"{r.get('attack', '')} → {r.get('rebuttal', '')}"
                    for r in bear.get("bull_rebuttals", []))

            result.debate = {"bull": bull, "bear": bear}

            # 3) 投资大师裁定
            master_params = LLMCallParams(
                system_prompt=self.MASTER_VERDICT_PROMPT.format(
                    symbol=symbol, analyst_results=analyst_text,
                    debate_summary=f"【多头】\n{bull_txt}\n\n【空头】\n{bear_txt}"),
                user_prompt=f"请对 {symbol} 做出最终裁定",
                mode="deep", max_tokens=900,
            )
            master = await self._chat_json_with_retry(
                adapter, master_params, self._validate_master_json)
            if not master:
                return
            result.master_verdict = master
            # 大师裁定生效（RiskGuard 20% 硬上限保留）
            d = str(master.get("final_direction", "")).lower()
            if d in ("bullish", "bearish", "neutral"):
                result.direction = d
            try:
                score = float(master.get("final_strength_score", result.overall_strength_score))
                result.overall_strength_score = round(min(100, max(0, score)), 1)
            except Exception:
                pass
            try:
                pos = float(master.get("suggested_max_position_pct",
                                       result.suggested_max_position_pct))
                result.suggested_max_position_pct = round(min(20, max(0, pos)), 1)
            except Exception:
                pass
            risks = master.get("final_risks")
            if isinstance(risks, list) and risks:
                result.key_risks = [str(r) for r in risks][:8]
            logger.info(
                "Debate + master verdict done", symbol=symbol,
                direction=result.direction, score=result.overall_strength_score,
                style=master.get("master_style", ""),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Debate failed, keeping PM result", symbol=symbol, error=str(e))

    def _rule_based_deep(
        self, symbol: str, kline_df: pd.DataFrame,
        market_state: Optional[str] = None,
        fundamental_data: Optional[dict] = None,
    ) -> AnalysisResult:
        """无 LLM 时的确定性规则分析（真实数据，不编造）。"""
        t0 = datetime.now()
        result = AnalysisResult(symbol=symbol, mode="deep")

        if kline_df is not None and not kline_df.empty:
            close = kline_df["close"].values
            ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
            ret20 = (close[-1] / close[-21] - 1) * 100 if len(close) > 21 else 0
            score = 50.0
            if close[-1] > ma20:
                score += 15
            score += max(-15, min(15, ret20 / 2))
            result.overall_strength_score = round(min(100, max(0, score)), 1)
            result.dimension_scores = {
                "technical": result.overall_strength_score,
                "capital": 50, "sentiment": 50, "fundamental": 50, "event": 50,
            }
            result.direction = ("bullish" if result.overall_strength_score >= 60
                                else "bearish" if result.overall_strength_score <= 40
                                else "neutral")
            result.key_evidence = [{
                "dimension": "technical",
                "finding": (f"MA20 {ma20:.2f}，20日涨跌 {ret20:+.1f}%"
                            f"（{'多头' if close[-1] > ma20 else '空头'}趋势）"),
                "impact": "positive" if close[-1] > ma20 else "negative",
            }]
            result.key_risks = ["无 LLM 配置，此为确定性规则分析（非 AI 解读）"]
            result.suggested_max_position_pct = 5.0
            result.analysis_time_ms = (datetime.now() - t0).total_seconds() * 1000
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

    # ---- 经验库（RAG 增强） ----

    @staticmethod
    def _build_experience_context(symbol: str) -> str:
        """检索同标的历史分析案例，格式化为 prompt 注入文本。"""
        try:
            from pa_mcp.agent.experience import get_experience_service
            entries = get_experience_service().search_experience(symbol=symbol, limit=5)
            if not entries:
                return ""
            # 优先展示已有事后验证的案例
            entries.sort(key=lambda e: e.outcome != "pending")
            return get_experience_service().format_experience(entries, limit=5)
        except Exception as e:  # noqa: BLE001
            logger.debug("experience context unavailable", error=str(e))
            return ""

    @staticmethod
    def _save_to_experience(symbol: str, result) -> None:
        """分析完成后自动落盘经验库（best-effort，异常不影响主流程）。"""
        try:
            from pa_mcp.agent.experience import get_experience_service
            get_experience_service().save_analysis(symbol, result)
        except Exception as e:  # noqa: BLE001
            logger.debug("save to experience failed", symbol=symbol, error=str(e))


# Global orchestrator instance
_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    """Get or create the agent orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
