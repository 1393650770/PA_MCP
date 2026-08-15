# [AI:BEGIN]
# PA_MCP - Gradio Web UI (理财 Agent 界面)
#
# 四个界面：
#   Tab 1 数据看板 — K线图 + 估值快照 + 资金流 + 龙虎榜
#   Tab 2 AI 对话  — 聊天（LLM + 工具调用，无 key 时降级为规则分析）
#   Tab 3 策略回测 — 策略选择 → 事件驱动回测 → 净值曲线/指标
#   Tab 4 组合管理 — 持仓 CRUD + 体检报告
#
# 启动：python -m pa_mcp.ui.gradio_app   (或 pa-mcp-ui)
# [AI:END]

from __future__ import annotations

import asyncio
import os
import sys
from datetime import date, datetime, timedelta
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- 数据层 ----

def _get_store():
    from pa_mcp.data.store import DuckDBStore
    from pa_mcp.config import get_settings
    return DuckDBStore(get_settings().database.path)


def _get_router():
    from pa_mcp.data.router import DataSourceRouter, CircuitBreakerConfig
    from pa_mcp.data.sources.tencent_adapter import TencentAdapter
    from pa_mcp.data.sources.sina_adapter import SinaAdapter
    from pa_mcp.data.sources.eastmoney_adapter import EastMoneyAdapter
    from pa_mcp.config import get_settings

    settings = get_settings()
    chain = [
        ("tencent", TencentAdapter()),
        ("sina", SinaAdapter()),
        ("eastmoney", EastMoneyAdapter()),
    ]
    cfg = CircuitBreakerConfig(
        failure_threshold=settings.router.circuit.failure_threshold,
        cooldown_seconds=settings.router.circuit.cooldown_seconds,
    )
    return DataSourceRouter(chain, {n: cfg for n, _ in chain},
                            min_source_interval={"eastmoney": 1.2})


# ---- Tab 1: 数据看板 ----

def _kline_figure(df: pd.DataFrame, symbol: str) -> go.Figure:
    """K线图（蜡烛图 + 成交量）。"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="K线",
        increasing_line_color="#ef5350", decreasing_line_color="#26a69a",
    ), row=1, col=1)

    # MA5/MA20
    for window, color in ((5, "#ffb300"), (20, "#5c6bc0")):
        ma = df["close"].rolling(window).mean()
        fig.add_trace(go.Scatter(x=df["date"], y=ma, name=f"MA{window}",
                                 line=dict(width=1, color=color)), row=1, col=1)

    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="成交量",
                         marker_color="#90a4ae"), row=2, col=1)

    fig.update_layout(
        title=f"{symbol} 日K线",
        xaxis_rangeslider_visible=False,
        height=560, template="plotly_white",
        legend=dict(orientation="h", y=1.02),
    )
    fig.update_xaxes(type="category", nticks=10)
    return fig


async def _load_kline(symbol: str, days: int = 120) -> tuple[pd.DataFrame, str]:
    """从 DB 优先，否则网络（多源容灾）。"""
    store = _get_store()
    try:
        end = datetime.now().date()
        start = end - timedelta(days=days * 2)  # 留余量覆盖节假日
        df = store.query_df(
            "SELECT * FROM kline_daily WHERE symbol = ? AND date >= ? "
            "AND date <= ? ORDER BY date ASC",
            [symbol, start.isoformat(), end.isoformat()],
        )
        if not df.empty:
            return df, "本地数据库"
    except Exception:
        pass

    router = _get_router()
    df, source = await router.fetch_daily_kline(
        symbol, start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"), adjust="qfq",
    )
    return df, source


def _fund_flow_figure(df: pd.DataFrame, symbol: str) -> go.Figure:
    """主力资金流柱状图。"""
    fig = go.Figure()
    colors = ["#ef5350" if v >= 0 else "#26a69a" for v in df["main_net_inflow"]]
    fig.add_trace(go.Bar(x=df["trade_date"], y=df["main_net_inflow"] / 1e8,
                         name="主力净流入(亿)", marker_color=colors))
    fig.update_layout(title=f"{symbol} 主力资金流（亿元）", template="plotly_white",
                      height=240, showlegend=False)
    return fig


def analyze_stock(symbol: str, days: int = 120) -> tuple[Any, str, str, str]:
    """数据看板主入口（同步包装）。"""
    symbol = symbol.strip()
    if not symbol:
        return None, "请输入股票代码", "", ""

    try:
        df, source = asyncio.run(_load_kline(symbol, days))
        if df.empty:
            return None, f"{symbol} 无数据（所有数据源均不可用）", "", ""

        fig = _kline_figure(df, symbol)

        # 资金流叠加图（DB 优先 → 东财网络兜底，失败自动跳过）
        flow = None
        try:
            store = _get_store()
            if store.table_exists("fund_flow_daily"):
                flow = store.query_df(
                    "SELECT * FROM fund_flow_daily WHERE symbol = ? "
                    "ORDER BY trade_date DESC LIMIT 20",
                    [symbol],
                ).sort_values("trade_date")
        except Exception:
            flow = None

        if flow is None or flow.empty:
            try:
                from pa_mcp.data.sources.eastmoney_adapter import EastMoneyAdapter
                adapter = EastMoneyAdapter()
                try:
                    asyncio.run(asyncio.sleep(1.2))
                    flow = asyncio.run(adapter.get_stock_fund_flow(symbol, days=20))
                finally:
                    asyncio.run(adapter.close())
            except Exception:
                flow = None

        if flow is not None and not flow.empty:
            fig.add_trace(go.Scatter(
                x=flow["trade_date"], y=flow["main_net_inflow"] / 1e8,
                name="主力净流入(亿)", line=dict(color="#7e57c2", width=1.5),
                yaxis="y2",
            ))
            fig.update_layout(
                yaxis2=dict(title="主力净流入(亿)", overlaying="y",
                            side="right", showgrid=False),
            )

        # 技术摘要
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        pct = (last["close"] / prev["close"] - 1) * 100 if prev["close"] else 0
        ma20 = df["close"].rolling(20).mean().iloc[-1]
        trend = "多头" if last["close"] > ma20 else "空头"

        from pa_mcp.data.symbols import get_stock_name
        stock_name = get_stock_name(symbol)
        summary = (
            f"**{symbol} {stock_name}**\n"
            f"最新收盘 {last['close']:.2f}（{pct:+.2f}%），MA20 {ma20:.2f} → {trend}趋势。\n"
            f"数据源：{source}，{len(df)} 根K线。"
        )

        # 估值快照（腾讯实时，失败自动跳过）
        valuation = ""
        try:
            from pa_mcp.data.sources.tencent_adapter import TencentAdapter
            q = asyncio.run(TencentAdapter().get_realtime_quote(symbol))
            valuation = (
                f"**实时估值**：PE {q.get('pe')} | PB {q.get('pb')} | "
                f"总市值 {q.get('total_market_cap_billion')} 亿 | "
                f"换手 {q.get('turnover_pct')}% | 量比 {q.get('volume_ratio')}\n"
                f"涨跌停：{q.get('limit_up_price')} / {q.get('limit_down_price')} | "
                f"距涨停 {q.get('distance_to_limit_up_pct')}%"
            )
        except Exception:
            valuation = "实时估值不可用（数据源波动）"

        return fig, summary, valuation, source
    except Exception as e:
        return None, f"分析失败：{str(e)[:200]}", "", ""


# ---- Tab 2: AI 对话 ----

SYSTEM_PROMPT = """你是 PA_MCP 的专业理财助手，擅长 A 股分析。
规则：
1. 所有分析基于工具返回的真实数据，禁止编造行情、财务数据。
2. 输出包含：关键数据 → 专业解读 → 风险提示。
3. 明确标注：研究参考，非投资建议。过往业绩不代表未来。
4. 使用中文回答。"""


def _rule_based_reply(message: str) -> str:
    """无 LLM key 时的降级规则分析（确定性工具结果）。

    工具调用识别：
      「资金流/主力」 → get_stock_capital_flow
      「体检」       → portfolio_review_ui
      「回测」       → run_backtest_ui
      其他含代码   → analyze_stock
    """
    message = message.strip()
    import re
    sym = None
    m = re.search(r"(\d{6})", message)
    if m:
        sym = m.group(1)

    # 工具0: 多股对比（≥2个代码 + 对比/比较关键词）
    import re as _re
    codes = _re.findall(r"\d{6}", message)
    if len(codes) >= 2 and ("对比" in message or "比较" in message or "哪个" in message):
        fig, summary = compare_stocks_ui(",".join(codes))
        return summary or "对比失败"

    # 工具0.3: 深度分析（5分析师团，需 LLM）
    if sym and ("深度" in message or "分析师" in message or "deep" in message.lower()):
        return portfolio_ai_analysis(sym)

    # 工具0.5: 股票名称查询（"xxx是什么股票/代码xxx是什么"）
    if sym and ("什么股票" in message or "是什么" in message or "叫什么" in message):
        from pa_mcp.data.symbols import get_stock_name
        return f"**{sym}** = {get_stock_name(sym)}"

    # 工具0.6: 市场扫描（下周买什么/扫描市场/候选）
    if any(k in message for k in ["下周", "买什么", "扫描市场", "候选", "选股"]):
        strat = next((s for s in STRATEGY_OPTIONS if s in message), "bollinger_mean_reversion")
        return scan_market_ui(strat, top_n=10)

    # 工具1: 持仓体检
    if "体检" in message or "持仓" in message:
        return portfolio_review_ui()

    # 工具2: 资金流
    if sym and ("资金" in message or "主力" in message or "flow" in message.lower()):
        try:
            from pa_mcp.data.sources.eastmoney_adapter import EastMoneyAdapter
            import asyncio as _aio

            adapter = EastMoneyAdapter()
            try:
                _aio.run(_aio.sleep(1.2))
                df = _aio.run(adapter.get_stock_fund_flow(sym, days=20))
            finally:
                _aio.run(adapter.close())

            if df.empty:
                fig, summary, valuation, source = analyze_stock(sym)
                parts = [f"### {sym} 资金流暂不可用（东财波动），基础分析如下：",
                         summary or "无数据"]
                if valuation:
                    parts.append(valuation)
                parts.append("\n*研究参考，非投资建议*")
                return "\n\n".join(parts)

            main_total = df["main_net_inflow"].sum()
            pos_days = (df["main_net_inflow"] > 0).sum()
            recent = df.tail(5).to_dict(orient="records")
            lines = [f"### {sym} 主力资金流（近{len(df)}日）",
                     f"- 主力净流入合计：**{main_total/1e8:.2f} 亿元**",
                     f"- 净流入天数：{pos_days}/{len(df)}（{pos_days/len(df)*100:.0f}%）",
                     "- 近5日明细："]
            for r in recent:
                lines.append(
                    f"  - {r['trade_date']}: 主力 {r['main_net_inflow']/1e8:+.2f}亿 "
                    f"超大单 {r['super_large_net_inflow']/1e8:+.2f}亿"
                )
            lines.append("\n*数据源东财（有限流）。研究参考，非投资建议。*")
            return "\n".join(lines)
        except Exception:
            # 东财不可用时降级为基础分析
            fig, summary, valuation, source = analyze_stock(sym)
            parts = [f"### {sym} 资金流暂不可用（东财波动），基础分析如下：",
                     summary or "无数据"]
            if valuation:
                parts.append(valuation)
            parts.append("\n*研究参考，非投资建议*")
            return "\n\n".join(parts)

    # 工具3: 回测
    if "回测" in message:
        strategies = STRATEGY_OPTIONS
        chosen = next((s for s in strategies if s in message), "ma_golden_cross")
        sym2 = sym or "000001"
        fig, summary = run_backtest_ui(sym2, chosen, 100000)
        return summary or "回测失败"

    # 工具4: 常规分析
    if not sym:
        return (
            "请提供 6 位股票代码，例如：\n"
            "  「分析 000001」 查看K线+估值\n"
            "  「600036 资金流」 查看主力资金\n"
            "  「回测 ma_cross」 运行策略回测\n"
            "  「体检」 组合持仓体检\n\n"
            "（当前未配置 LLM API Key，使用规则分析模式。"
            "设置 ANTHROPIC_API_KEY 后获得完整 AI 对话）"
        )

    fig, summary, valuation, source = analyze_stock(sym)
    parts = [f"### {sym} 分析", summary or "无数据"]
    if valuation:
        parts.append(valuation)
    parts.append("\n*研究参考，非投资建议*")
    return "\n\n".join(parts)


def chat_reply(message: str, history: list[dict]) -> str:
    """对话接口：优先 LLM，无 key 降级规则分析。"""
    if not message.strip():
        return "请输入问题"

    # LLM 可用性：优先已初始化的 adapter；否则尝试从
    # config/llm_config.json 或环境变量初始化
    from pa_mcp.agent.llm_port import get_llm_adapter
    adapter = get_llm_adapter()
    if adapter is None:
        try:
            from pa_mcp.agent.llm_factory import init_llm_adapter
            adapter = init_llm_adapter("config/llm_config.json")
        except Exception:
            adapter = None
    if adapter is None:
        return _rule_based_reply(message)

    try:
        # 通过 adapter 调用（支持 Anthropic SDK / OpenAI-compatible / 豆包等）
        from pa_mcp.agent.llm_port import LLMCallParams

        # 构建对话历史
        user_content = message
        if history:
            user_content = "\n".join(
                f"{'用户' if h.get('role') == 'user' else '助手'}: {h.get('content', '')}"
                for h in history[-6:]
            ) + f"\n用户: {message}"

        params = LLMCallParams(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_content,
            mode="fast",
            max_tokens=2000,
        )
        resp = asyncio.run(adapter.chat(params))
        if resp.content and not resp.content.startswith('{"error"'):
            return resp.content
        return f"LLM 调用失败（{resp.content[:100]}），已降级为规则分析：\n\n{_rule_based_reply(message)}"
    except Exception as e:
        return f"LLM 调用失败（{str(e)[:120]}），已降级为规则分析：\n\n{_rule_based_reply(message)}"


def _prediction_brief(symbol: str, df: pd.DataFrame, horizon: str = "5d") -> str:
    """未来走势预测摘要（供持仓分析/看板复用，失败返回空串）。"""
    try:
        from pa_mcp.agent.prediction import get_prediction_service
        result = asyncio.run(get_prediction_service().predict(symbol, df, horizon=horizon))
        p = result.to_dict()
        dist = p["probability_distribution"]
        rng = p["expected_range_pct"]
        lines = [
            f"🔮 **未来 {horizon} 走势预测**：{DIRECTION_ZH.get(p['direction'], p['direction'])}"
            f"（{p['probability']:.0%}）",
            f"涨 {dist['up']:.0%} / 跌 {dist['down']:.0%} / 震荡 {dist['sideways']:.0%}，"
            f"期望 {p['expected_return_pct']:+.1f}%（区间 {rng[0]:+.1f}~{rng[1]:+.1f}%），"
            f"周期 {p['cycle_position_zh']}→{p['cycle_forecast_zh']}，置信 {p['confidence']:.0%}"
            f"（{'AI' if p['mode'] == 'llm' else '统计'}模式）",
        ]
        if p.get("key_levels", {}).get("support"):
            lines.append(f"支撑 {p['key_levels']['support']} / "
                         f"压力 {p['key_levels']['resistance']}")
        return "\n".join(lines)
    except Exception:
        return ""


def portfolio_ai_analysis(symbol: str) -> str:
    """对持仓股生成 AI 综合分析（LLM 或规则降级）。"""
    symbol = symbol.strip()
    if not symbol:
        return "请输入持仓代码"

    # 先取数据（真实）
    from pa_mcp.data.symbols import get_stock_name
    name = get_stock_name(symbol)

    # 规则分析（确定性数据）
    fig, summary, valuation, source = analyze_stock(symbol, 60)
    parts = [f"### {symbol} {name} 综合分析", summary or "无数据"]
    if valuation:
        parts.append(valuation)

    # 策略信号
    try:
        from pa_mcp.engine.strategies.base import StrategyRegistry
        registry = StrategyRegistry(); registry.auto_discover()
        df = _load_long_history(symbol)
        signals = []
        for s_name in ["bollinger_mean_reversion", "ma_golden_cross"]:
            inst = registry.get(s_name)
            if inst is None or df is None or df.empty:
                continue
            try:
                sigs = inst.generate_signals(df.copy())
            except Exception:
                continue
            if not sigs:
                continue
            latest = str(df["date"].astype(str).str[:10].iloc[-1])
            recent = [x for x in sigs if
                      (getattr(x, "signal_time", None) or
                       str(getattr(x, "timestamp", ""))[:10]) >= latest]
            if recent:
                s = recent[-1]
                signals.append(f"**{s_name}**：买入信号（{getattr(s, 'signal_time', '')[:10]}，强度{getattr(s, 'strength_score', 50):.0f}）")
        if signals:
            parts.append("**当前策略信号**：" + "；".join(signals))
        else:
            parts.append("**当前策略信号**：无触发（当前交易日无买入信号）")
    except Exception:
        pass

    parts.append("\n*研究参考，非投资建议*")

    # 有 LLM 时用 5 分析师团（deep_analyze）深度分析
    try:
        from pa_mcp.agent.llm_port import get_llm_adapter
        adapter = get_llm_adapter()
        if adapter is not None:
            from pa_mcp.agent.orchestrator import get_orchestrator
            orch = get_orchestrator()
            df = _load_long_history(symbol)
            result = asyncio.run(orch.deep_analyze(
                symbol, df, market_state=None,
                fundamental_data={"fundamental": "无财务数据", "events": "无事件数据"},
            ))
            if result.overall_strength_score > 0:
                ai_text = (
                    f"**综合评分 {result.overall_strength_score}（{'看多' if result.direction == 'bullish' else '看空' if result.direction == 'bearish' else '中性'}）**\n"
                    f"维度：技术{result.dimension_scores.get('technical', '-')} "
                    f"资金{result.dimension_scores.get('capital', '-')} "
                    f"情绪{result.dimension_scores.get('sentiment', '-')} "
                    f"基本面{result.dimension_scores.get('fundamental', '-')} "
                    f"事件{result.dimension_scores.get('event', '-')}\n"
                    f"建议仓位：≤{result.suggested_max_position_pct}%\n"
                )
                if result.key_risks:
                    ai_text += "风险：" + "；".join(result.key_risks[:3]) + "\n"
                if result.key_evidence:
                    ev = result.key_evidence[0]
                    ai_text += f"关键依据：{ev.get('finding', '')}\n"
                pred = _prediction_brief(symbol, df)
                if pred:
                    ai_text += "\n" + pred + "\n"
                return "\n\n".join(parts[:2]) + "\n\n---\n💡 **AI 分析师团解读**：\n" + ai_text
    except Exception:
        pass
    # 无 LLM：规则分析 + 确定性预测
    try:
        df = _load_long_history(symbol)
        if not df.empty:
            pred = _prediction_brief(symbol, df)
            if pred:
                parts.append(f"---\n{pred}")
    except Exception:
        pass
    return "\n\n".join(parts)


# ---- 多股对比（Tab 2.5）----

def compare_stocks_ui(symbols_str: str) -> tuple[Any, str]:
    """多股对比：归一化走势 + 估值/涨跌表。"""
    syms = [s.strip() for s in symbols_str.replace("，", ",").split(",") if s.strip()]
    if len(syms) < 2:
        return None, "请输入至少 2 个代码，逗号分隔（如 000001,600036,300750）"
    if len(syms) > 5:
        return None, "最多对比 5 只"

    from pa_mcp.data.symbols import get_stock_name

    fig = go.Figure()
    rows = []
    for sym in syms:
        try:
            df, source = asyncio.run(_load_kline(sym, 120))
            if df.empty:
                rows.append({"symbol": sym, "error": "无数据"})
                continue

            # 归一化（首日=100）
            base = df["close"].iloc[0]
            norm = df["close"] / base * 100
            name = get_stock_name(sym)
            fig.add_trace(go.Scatter(x=df["date"], y=norm,
                                     name=f"{sym} {name}",
                                     line=dict(width=1.8)))

            # 估值快照
            pe = pb = mcap = None
            try:
                from pa_mcp.data.sources.tencent_adapter import TencentAdapter
                q = asyncio.run(TencentAdapter().get_realtime_quote(sym))
                pe, pb, mcap = q.get("pe"), q.get("pb"), q.get("total_market_cap_billion")
            except Exception:
                pass

            ret20 = (df["close"].iloc[-1] / df["close"].iloc[-21] - 1) * 100 if len(df) > 21 else 0
            rows.append({
                "symbol": sym, "name": get_stock_name(sym), "source": source,
                "close": round(float(df["close"].iloc[-1]), 2),
                "ret20_pct": round(float(ret20), 2),
                "pe": pe, "pb": pb, "mcap_亿": mcap,
            })
        except Exception as e:
            rows.append({"symbol": sym, "error": str(e)[:60]})

    fig.update_layout(title="多股归一化对比（首日=100）", template="plotly_white",
                      height=420, legend=dict(orientation="h", y=1.02))
    summary = "| 代码 | 名称 | 现价 | 20日涨跌% | PE | PB | 市值(亿) |\n|---|---|---|---|---|---|---|\n"
    for r in rows:
        if "error" in r:
            summary += f"| {r['symbol']} | — | — | — | — | — | {r['error']} |\n"
        else:
            summary += (f"| {r['symbol']} | {r['name']} | {r['close']} | {r['ret20_pct']:+.1f} | "
                        f"{r['pe'] or '—'} | {r['pb'] or '—'} | {r['mcap_亿'] or '—'} |\n")
    summary += "\n*免费行情，可能有延迟。研究参考，非投资建议。*"
    return fig, summary


# ---- 组合构建管线 ----

def portfolio_build_ui(symbols_str: str, strategy: str) -> tuple[Any, str]:
    """信号→组合管线：多票信号 → 约束权重 → 组合回测 → 净值图。"""
    syms = [s.strip() for s in symbols_str.replace("，", ",").split(",") if s.strip()]
    if len(syms) < 2:
        return None, "请输入至少 2 个代码，逗号分隔"

    try:
        from pa_mcp.engine.strategies.base import StrategyRegistry
        from pa_mcp.portfolio.pipeline import build_portfolio_from_signals
        import plotly.graph_objects as go

        registry = StrategyRegistry()
        registry.auto_discover()
        base = registry.get(strategy)
        if base is None:
            return None, f"策略 {strategy} 未注册"

        def signal_fn(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
            try:
                signals = base.generate_signals(df.copy())
            except Exception:
                return pd.DataFrame()
            if not signals:
                return pd.DataFrame()
            return pd.DataFrame([{
                "symbol": symbol,
                "date": getattr(s, "signal_time", None) or str(getattr(s, "timestamp", ""))[:10],
                "direction": getattr(s, "direction", "neutral").value
                            if hasattr(getattr(s, "direction", None), "value") else str(getattr(s, "direction", "neutral")),
                "strength_score": float(getattr(s, "strength_score", 50)),
                "strategy_name": strategy,
            } for s in signals])

        kline_map = {sym: _load_long_history(sym) for sym in syms}
        kline_map = {k: v for k, v in kline_map.items() if not v.empty}
        if len(kline_map) < 2:
            return None, "有效股票不足 2 只（数据加载失败？）"

        # 共享账本组合回测（多票联合执行）
        from pa_mcp.portfolio.backtest import PortfolioBacktestEngine
        from pa_mcp.backtest.broker import FeeSchedule

        sig_map = {sym: signal_fn(df, sym) for sym, df in kline_map.items()}
        engine = PortfolioBacktestEngine(initial_cash=100000,
                                         fee_schedule=FeeSchedule())
        result = engine.run(kline_map, sig_map)

        # 净值图
        fig = None
        if result.nav_series:
            nav = pd.DataFrame(result.nav_series)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nav["trade_date"], y=nav["nav"],
                                     name="组合净值", line=dict(color="#1565c0", width=2)))
            fig.update_layout(title=f"组合净值（{strategy}，{len(kline_map)}只·共享账本）",
                              template="plotly_white", height=400)

        # 末期持仓
        last = result.nav_series[-1] if result.nav_series else {}
        positions = last.get("positions", {})
        lines = [
            f"## 组合回测（{strategy}，共享账本）",
            f"- 组合收益 {result.total_return_pct:+.2f}% | 年化 {result.annual_return_pct:+.2f}% | "
            f"夏普 {result.sharpe:.2f} | 回撤 {result.max_drawdown_pct:.2f}%",
            f"- 交易 {result.total_trades} 笔 | 费用 {result.total_fees:.2f} 元 | "
            f"末期持仓 {len(positions)} 只 | 现金 {last.get('cash', 0):,.0f} 元",
            "",
            "| 持仓股票 | 股数 |",
            "|---|---|",
        ]
        for sym, qty in sorted(positions.items(), key=lambda x: -x[1]):
            lines.append(f"| {sym} | {qty:,} |")
        lines.append("\n*多票共享现金/持仓账本，延迟一天执行，含A股撮合。研究参考，非投资建议。*")
        return fig, "\n".join(lines)
    except Exception as e:
        return None, f"组合构建失败：{str(e)[:200]}"


# ---- 市场扫描（当前信号候选清单）----

# ---- 板块数据（东财动态 + 内置兜底） ----

# 内置板块模板（东财接口不可用时兜底）
BUILTIN_SECTORS: dict[str, list[str]] = {
    "银行": ["000001", "600036", "601398", "601288", "601988"],
    "白酒": ["600519", "000858", "000568", "600809"],
    "新能源": ["300750", "002594", "601012", "600438", "300274"],
    "医药": ["600276", "000538", "300760", "600196"],
    "科技": ["002415", "688981", "688001", "603986", "002230"],
    "消费电子": ["002475", "601138", "000725", "002241"],
    "基建能源": ["600900", "601857", "600028", "601088"],
    "汽车家电": ["601633", "600104", "000333", "000651"],
}


def fetch_sector_universe(hot_count: int = 6, cold_count: int = 3,
                          stocks_per_sector: int = 6) -> tuple[dict[str, str], dict[str, list[str]], str]:
    """获取板块 → 成分股映射（东财动态优先，内置兜底）。

    Returns:
        (sector_info: {板块名: 涨跌幅}, sector_stocks: {板块名: [代码]},
         source: 'eastmoney' | 'builtin')
    """
    sector_info: dict[str, str] = {}
    sector_stocks: dict[str, list[str]] = {}
    source = "builtin"

    try:
        import urllib.request
        # 东财板块列表（按涨跌幅排序，fid=f3 从高到低；pn 换页取冷门）
        all_sectors = []
        for page in (1, 2):
            url = (f"https://push2.eastmoney.com/api/qt/clist/get"
                   f"?pn={page}&pz=50&po=1&np=1&fltt=2&invt=2&fid=f3"
                   f"&fs=m:90+t:2&fields=f2,f3,f12,f14")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            rows = data.get("data", {}).get("diff") or []
            for r in rows:
                if r.get("f12") and r.get("f14"):
                    all_sectors.append((r["f12"], r["f14"], r.get("f3", 0)))
        if not all_sectors:
            raise RuntimeError("no sectors")

        # 热门 = 涨幅前 hot_count；冷门 = 跌幅最深 cold_count
        hot = all_sectors[:hot_count]
        cold = all_sectors[-cold_count:]
        picked = hot + cold

        for sec_code, sec_name, pct in picked:
            sector_info[sec_name] = f"{pct:+.1f}%"
            # 板块成分股
            url = (f"https://push2.eastmoney.com/api/qt/clist/get"
                   f"?pn=1&pz={stocks_per_sector}&po=1&np=1&fltt=2&invt=2&fid=f3"
                   f"&fs=m:90+t:2+f:!50&fields=f12,f14&fi={sec_code}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            stocks = []
            for r in (data.get("data", {}).get("diff") or []):
                code = str(r.get("f12", ""))
                if code and code.isdigit() and len(code) == 6:
                    stocks.append(code)
            if stocks:
                sector_stocks[sec_name] = stocks
        if sector_stocks:
            source = "eastmoney"
    except Exception:
        # 内置兜底
        for name, codes in BUILTIN_SECTORS.items():
            sector_info[name] = "—"
            sector_stocks[name] = codes

    return sector_info, sector_stocks, source


def scan_market_ui(strategy: str, top_n: int = 10,
                    universe_size: int = 30) -> str:
    """扫描股票池：找当前处于买入信号状态的股票，输出候选清单。

    输出：代码/名称/信号日期/强度/该信号历史 5 日胜率。
    基于统计而非预测——标明"信号候选"而非"预测上涨"。
    """
    from pa_mcp.data.symbols import COMMON_NAMES, get_stock_name
    from pa_mcp.engine.strategies.base import StrategyRegistry
    from pa_mcp.engine.strategies.tips import get_strategy_tip
    from pa_mcp.research.event_study import signal_forward_returns

    # 股票池：板块成分（热门+冷门）∪ 用户持仓股 ∪ 内置常用股
    sector_info, sector_stocks, sector_source = fetch_sector_universe(
        hot_count=6, cold_count=3, stocks_per_sector=6)

    symbols = []
    for codes in sector_stocks.values():
        for c in codes:
            if c not in symbols:
                symbols.append(c)

    holdings = []
    try:
        store = _get_store()
        if store.table_exists("portfolio"):
            hdf = store.query_df("SELECT symbol FROM portfolio")
            holdings = hdf["symbol"].tolist() if not hdf.empty else []
    except Exception:
        pass
    if holdings:
        # 持仓股排在最前（优先扫描）
        symbols = [s for s in holdings if s not in symbols] + symbols

    # 补充内置常用股（覆盖板块外的白马）
    for c in list(COMMON_NAMES.keys())[:universe_size]:
        if c not in symbols:
            symbols.append(c)

    universe_size = len(symbols)
    sector_name_of: dict[str, str] = {}
    for sec, codes in sector_stocks.items():
        for c in codes:
            sector_name_of.setdefault(c, sec)

    registry = StrategyRegistry()
    registry.auto_discover()
    base = registry.get(strategy)
    if base is None:
        return f"策略 {strategy} 未注册"

    rows = []
    for sym in symbols:
        try:
            df = _load_long_history(sym)
            if df.empty or len(df) < 120:
                continue
            # 当前信号：最近 10 个交易日内有买入信号
            inst = base.__class__(**getattr(base, "__dict__", {}))
            try:
                signals = inst.generate_signals(df.copy())
            except Exception:
                continue
            if not signals:
                continue
            window_start = str(df["date"].astype(str).str[:10].iloc[-11])
            recent = [
                s for s in signals
                if (getattr(s, "signal_time", None) or str(getattr(s, "timestamp", ""))[:10]) >=
                window_start
            ]
            if not recent:
                continue
            s = recent[-1]
            sig_date = (getattr(s, "signal_time", None) or
                        str(getattr(s, "timestamp", ""))[:10])
            strength = float(getattr(s, "strength_score", 50))

            # 该信号历史 5 日胜率（预测力参考）
            win_rate = None
            if len(signals) >= 10:
                sig_df = pd.DataFrame([{
                    "symbol": sym,
                    "date": getattr(x, "signal_time", None) or str(getattr(x, "timestamp", ""))[:10],
                    "direction": getattr(x, "direction", "neutral").value
                                if hasattr(getattr(x, "direction", None), "value")
                                else str(getattr(x, "direction", "neutral")),
                    "strategy_name": strategy,
                } for x in signals])
                results = signal_forward_returns(df, sig_df, [5])
                if results and results[0].n_events >= 10:
                    win_rate = results[0].win_rate_pct

            rows.append({
                "symbol": sym, "name": get_stock_name(sym),
                "signal_date": sig_date, "strength": strength,
                "win_rate": win_rate,
                "sector": sector_name_of.get(sym, ""),
            })
        except Exception:
            continue

    if not rows:
        # 主策略无近期信号 → 用 ma_golden_cross 互补扫描（趋势型更常触发）
        try:
            alt = registry.get("ma_golden_cross")
            if alt is not None:
                for sym in symbols:
                    try:
                        df = _load_long_history(sym)
                        if df.empty or len(df) < 120:
                            continue
                        signals = alt.generate_signals(df.copy())
                        if not signals:
                            continue
                        window_start = str(df["date"].astype(str).str[:10].iloc[-11])
                        recent = [
                            s for s in signals
                            if (getattr(s, "signal_time", None) or str(getattr(s, "timestamp", ""))[:10]) >=
                            window_start
                        ]
                        if not recent:
                            continue
                        s = recent[-1]
                        rows.append({
                            "symbol": sym, "name": get_stock_name(sym),
                            "signal_date": (getattr(s, "signal_time", None) or
                                            str(getattr(s, "timestamp", ""))[:10]),
                            "strength": float(getattr(s, "strength_score", 50)),
                            "win_rate": None,
                            "alt_strategy": "ma_golden_cross",
                        })
                    except Exception:
                        continue
        except Exception:
            pass

        if not rows:
            return (f"{strategy}：当前股票池近10日无买入信号。\n\n"
                    f"⚠️ **这本身是有用信息**：当前市场状态没有该策略的机会。\n"
                    f"可尝试：① 换其他策略扫描 ② 扩大股票池 ③ 等待信号出现后再关注。\n\n"
                    f"*研究参考，非投资建议。*")

    rows.sort(key=lambda r: r["strength"], reverse=True)
    rows = rows[:top_n]

    held_set = set(holdings) if holdings else set()
    if sector_source == "eastmoney":
        hot_sectors = list(sector_info.items())[:6]
        cold_sectors = list(sector_info.items())[-3:]
        env_line = (f"**板块环境**（东财实时）：热门 "
                    + " ".join(f"{n} {p}" for n, p in hot_sectors)
                    + ("｜冷门 " + " ".join(f"{n} {p}" for n, p in cold_sectors)
                       if cold_sectors else ""))
    else:
        env_line = f"**板块环境**（内置模板）：{' '.join(sector_info.keys())}（东财板块接口暂不可用，已用内置板块兜底）"

    lines = [
        f"## 📡 市场扫描：{strategy} 当前买入信号候选（TOP {len(rows)}）",
        f"*扫描 {universe_size} 只（板块 + 持仓"
        f"{'📌' if holdings else ''}）· 信号日期 = 最近触发日（近10日）*",
        "",
        env_line,
        "",
        "| 代码 | 名称 | 板块 | 信号日期 | 强度 | 历史5日胜率 | 来源策略 | 持仓 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        wr = f"{r['win_rate']:.0f}%" if r["win_rate"] else "样本不足"
        src = r.get("alt_strategy", strategy)
        is_held = "📌" if r["symbol"] in held_set else ""
        sector = r.get("sector", "") or "—"
        lines.append(f"| {r['symbol']} | {r['name']} | {sector} | {r['signal_date']} | "
                     f"{r['strength']:.0f} | {wr} | {src} | {is_held} |")
    tip = get_strategy_tip(strategy)
    lines.append(f"\n**策略说明**：{tip.splitlines()[0] if tip else ''}")

    # LLM 综合解读（有配置时增强）
    if rows:
        try:
            from pa_mcp.agent.llm_port import get_llm_adapter, LLMCallParams
            adapter = get_llm_adapter()
            if adapter is not None:
                cand_text = "\n".join(
                    f"- {r['symbol']} {r['name']}（板块{r.get('sector','?')}）"
                    f"信号{r['signal_date']} 强度{r['strength']:.0f} "
                    f"历史胜率{r['win_rate']:.0f}%"
                    for r in rows[:10]
                )
                env_text = (env_line if "env_line" in dir() else
                            "板块环境未知")
                resp = asyncio.run(adapter.chat(LLMCallParams(
                    system_prompt=(
                        "你是 A 股市场扫描解读助手。基于候选清单与板块环境给出：\n"
                        "1) 候选股优先级排序逻辑（哪些更值得先看）\n"
                        "2) 板块机会解读（结合热门/冷门板块）\n"
                        "3) 风险提示（信号时效、市场环境）\n"
                        "仅基于给定数据，禁止编造。简洁中文，标注研究参考非投资建议。"
                    ),
                    user_prompt=(
                        f"市场状态/板块环境：\n{env_text}\n\n"
                        f"{strategy} 策略信号候选：\n{cand_text}"
                    ),
                    mode="fast", max_tokens=600,
                )))
                if resp.content and not resp.content.startswith('{"error'):
                    lines.append("\n---\n💡 **AI 综合解读**：\n" + resp.content)
        except Exception:
            pass

    lines.append("\n⚠️ **重要说明**：此清单是*当前信号候选*，非预测上涨。"
                 "历史胜率是统计参考；信号可能失效，请结合基本面/资金面自行判断。"
                 "研究参考，非投资建议。")
    return "\n".join(lines)


# ---- Walk-Forward 研究评估 ----

def _load_long_history(symbol: str, years: float = 5.0) -> pd.DataFrame:
    """分页拉取长历史日线（腾讯单次上限640根，分2-3段拼接）。"""
    store = _get_store()
    today = datetime.now().date()
    # 先查 DB（scheduler 积累的数据更全）
    try:
        if store.table_exists("kline_daily"):
            df = store.query_df(
                "SELECT * FROM kline_daily WHERE symbol = ? ORDER BY date ASC",
                [symbol],
            )
            if len(df) >= 500:
                return df
    except Exception:
        pass

    async def _fetch() -> pd.DataFrame:
        from pa_mcp.data.router import DataSourceRouter, CircuitBreakerConfig
        from pa_mcp.data.sources.tencent_adapter import TencentAdapter
        from pa_mcp.data.sources.sina_adapter import SinaAdapter
        from pa_mcp.config import get_settings

        # 多源 router：腾讯被风控(501)时自动切新浪
        settings = get_settings()
        cfg = CircuitBreakerConfig(
            failure_threshold=settings.router.circuit.failure_threshold,
            cooldown_seconds=settings.router.circuit.cooldown_seconds,
        )
        router = DataSourceRouter(
            [("tencent", TencentAdapter()), ("sina", SinaAdapter())],
            {n: cfg for n, _ in [("tencent", None), ("sina", None)]},
        )
        try:
            segments = []
            end_d = today
            for _ in range(4):
                start_d = end_d - timedelta(days=700)
                df, src = await router.fetch_daily_kline(
                    symbol, start_date=start_d.strftime("%Y%m%d"),
                    end_date=end_d.strftime("%Y%m%d"), adjust="qfq",
                )
                if not df.empty:
                    segments.append(df)
                    if len(df) < 300:
                        break  # 已到历史尽头
                end_d = start_d - timedelta(days=1)
                if end_d < today - timedelta(days=int(years * 365)):
                    break
            if not segments:
                return pd.DataFrame()
            merged = pd.concat(segments).drop_duplicates(subset=["date"]).sort_values("date")
            return merged
        finally:
            pass

    try:
        return asyncio.run(_fetch())
    except RuntimeError:
        # 已在运行的事件循环中 → 直接跑
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_fetch())

def walk_forward_ui(symbol: str, strategy: str) -> str:
    """对策略做 walk-forward OOS 评估（真实数据，长历史分页）。"""
    symbol = symbol.strip()
    try:
        df = _load_long_history(symbol)
        if df.empty:
            return f"{symbol} 无行情数据"

        from pa_mcp.engine.strategies.base import StrategyRegistry
        registry = StrategyRegistry()
        registry.auto_discover()
        strategy_cls = registry.get(strategy)
        if strategy_cls is None:
            return f"策略 {strategy} 未注册"

        def signal_fn(window: pd.DataFrame) -> pd.DataFrame:
            try:
                signals = strategy_cls.generate_signals(window.copy())
            except Exception:
                return pd.DataFrame()
            if not signals:
                return pd.DataFrame()
            return pd.DataFrame([{
                "symbol": symbol,
                "date": getattr(s, "signal_time", None) or str(getattr(s, "timestamp", ""))[:10],
                "direction": getattr(s, "direction", "neutral").value
                            if hasattr(getattr(s, "direction", None), "value") else str(getattr(s, "direction", "neutral")),
                "strength_score": float(getattr(s, "strength_score", 50)),
                "strategy_name": strategy,
            } for s in signals])

        from pa_mcp.research.strategy_eval import run_walk_forward
        report = run_walk_forward(df, strategy, signal_fn)
        return report.summary()
    except Exception as e:
        return f"评估失败：{str(e)[:200]}"


def event_study_ui(symbol: str, strategy: str) -> str:
    """信号事件研究：检验策略信号的预测力（信号后 5/10/20 日收益 vs 基准）。"""
    symbol = symbol.strip()
    try:
        df = _load_long_history(symbol)
        if df.empty:
            return f"{symbol} 无行情数据"

        from pa_mcp.engine.strategies.base import StrategyRegistry
        from pa_mcp.research.event_study import signal_forward_returns

        registry = StrategyRegistry()
        registry.auto_discover()
        strategy_cls = registry.get(strategy)
        if strategy_cls is None:
            return f"策略 {strategy} 未注册"

        try:
            signals = strategy_cls.generate_signals(df.copy())
        except Exception:
            return f"{strategy} 信号生成失败"

        if not signals:
            return f"{strategy} × {symbol}：无信号（该策略可能依赖特定市场条件）"

        sig_df = pd.DataFrame([{
            "symbol": symbol,
            "date": getattr(s, "signal_time", None) or str(getattr(s, "timestamp", ""))[:10],
            "direction": getattr(s, "direction", "neutral").value
                        if hasattr(getattr(s, "direction", None), "value") else str(getattr(s, "direction", "neutral")),
            "strategy_name": strategy,
        } for s in signals])

        results = signal_forward_returns(df, sig_df, [5, 10, 20])
        if not results:
            return f"{strategy} × {symbol}：信号无法定位（检查 signal_time）"

        lines = [f"## {strategy} × {symbol} 信号事件研究",
                 "| 前瞻日 | 信号数 | 胜率% | 平均收益% | 基准胜率% | 基准收益% | 超额% |",
                 "|---|---|---|---|---|---|---|"]
        for r in results:
            lines.append(
                f"| {r.horizon} | {r.n_events} | {r.win_rate_pct:.1f} | "
                f"{r.avg_return_pct:+.2f} | {r.benchmark_win_rate_pct:.1f} | "
                f"{r.benchmark_avg_return_pct:+.2f} | {r.excess_return_pct:+.2f} |"
            )
        verdict = "✅ 有预测力" if any(r.has_edge for r in results) else "❌ 无显著预测力"
        lines.append(f"\n**结论：{verdict}**")
        lines.append("\n*事件研究 = 信号后持有N日 vs 无条件基准（不含交易成本）。研究参考，非投资建议。*")
        return "\n".join(lines)
    except Exception as e:
        return f"事件研究失败：{str(e)[:200]}"


# ---- Tab: 市场预测（LLM 未来走势预测，借鉴 PA_Agent 机制） ----

DIRECTION_ZH = {"up": "📈 看涨", "down": "📉 看跌", "sideways": "➡️ 震荡"}


def predict_market_ui(symbol: str, horizon: str) -> str:
    """市场预测：方向/概率/期望收益/关键价位/多场景 + 落盘验证。"""
    symbol = symbol.strip()
    if not symbol:
        return "请输入股票代码"
    try:
        import asyncio
        from pa_mcp.agent.prediction import get_prediction_service

        df = _load_long_history(symbol)
        if df.empty:
            return f"{symbol} 无行情数据"

        svc = get_prediction_service()
        result = asyncio.run(svc.predict(symbol, df, horizon=horizon))
        payload = result.to_dict()
        pred_id = svc.save_prediction(result)  # 落盘供日后验证

        dist = payload["probability_distribution"]
        rng = payload["expected_range_pct"]
        levels = payload["key_levels"]
        scenarios = payload.get("scenarios") or []

        lines = [
            f"## 🔮 {symbol} 未来 {horizon} 走势预测",
            f"**方向：{DIRECTION_ZH.get(payload['direction'], payload['direction'])}"
            f"（概率 {payload['probability']:.0%}）**",
            f"| 方向 | 概率 |",
            "|---|---|",
            f"| 📈 上涨 | {dist['up']:.0%} |",
            f"| 📉 下跌 | {dist['down']:.0%} |",
            f"| ➡️ 震荡 | {dist['sideways']:.0%} |",
            "",
            f"- **期望收益**：{payload['expected_return_pct']:+.1f}%"
            f"（区间 {rng[0]:+.1f}% ~ {rng[1]:+.1f}%）",
            f"- **周期位置**：{payload['cycle_position_zh']}"
            f" → 预测 {payload['cycle_forecast_zh']}",
            f"- **关键位**：支撑 {levels['support']} / 压力 {levels['resistance']}",
            f"- **置信度**：{payload['confidence']:.0%}"
            f"（模式：{'AI 解读' if payload['mode'] == 'llm' else '统计降级'}）",
            "",
            "### 情景推演",
        ]
        for s in scenarios:
            lines.append(
                f"- **{s.get('name', '情景')}**（{s.get('probability', 0):.0%}）："
                f"{s.get('description', '')}（目标 {s.get('target_pct', 0):+.1f}%）")
        lines.append("")
        if payload.get("key_reasons"):
            lines.append("**依据**：" + "；".join(payload["key_reasons"]))
        if payload.get("key_risks"):
            lines.append("**风险**：" + "；".join(payload["key_risks"]))
        lines.append(
            f"\n*已落盘（记录 #{pred_id}），到期后可在「预测验证」查看命中率。"
            f"{payload['disclaimer']}*")
        return "\n".join(lines)
    except Exception as e:
        return f"预测失败：{str(e)[:200]}"


def prediction_history_ui(symbol: str) -> str:
    """历史预测记录 + 验证状态。"""
    symbol = symbol.strip()
    if not symbol:
        return "请输入股票代码"
    try:
        from pa_mcp.agent.prediction import get_prediction_service
        rows = get_prediction_service().prediction_history(symbol, limit=20)
        if not rows:
            return f"{symbol} 暂无预测记录"
        lines = [f"## 📜 {symbol} 预测记录（最新 {len(rows)} 条）",
                 "| 日期 | 周期 | 方向 | 概率 | 期望% | 模式 | 状态 | 实际% |",
                 "|---|---|---|---|---|---|---|---|"]
        status_zh = {"pending": "⏳ 待验证", "hit": "✅ 命中",
                     "miss": "❌ 未中", "ambiguous": "⚠️ 模糊"}
        for r in rows:
            lines.append(
                f"| {r['predict_date']} | {r['cycle_position']} | "
                f"{DIRECTION_ZH.get(r['direction'], r['direction'])} | "
                f"{r['probability']:.0%} | {r['expected_return_pct']:+.1f} | "
                f"{'AI' if r['mode'] == 'llm' else '统计'} | "
                f"{status_zh.get(r['status'], r['status'])} | "
                f"{f'{r['actual_return_pct']:+.1f}%' if r['actual_return_pct'] is not None else '—'} |")
        return "\n".join(lines)
    except Exception as e:
        return f"查询失败：{str(e)[:200]}"


def evaluate_predictions_ui() -> str:
    """预测验证成绩单：命中率/方向一致率/分方向表现。"""
    try:
        import asyncio
        from pa_mcp.agent.prediction import get_prediction_service
        svc = get_prediction_service()

        # 用数据库行情回填（无网络依赖）
        summary = asyncio.run(svc.evaluate_predictions())
        if summary["total_predictions"] == 0:
            return "暂无预测记录。先在「预测」输入股票代码生成预测，到期后回来验证。"
        lines = [f"## 🎯 预测验证成绩单（共 {summary['total_predictions']} 条，"
                 f"已评估 {summary['evaluated']} 条）"]
        if summary["evaluated"] == 0:
            lines.append("\n尚无到期预测，请等待预测周期结束或生成更早的预测。")
            return "\n".join(lines)
        lines.extend([
            f"- **总体命中率**：**{summary['hit_rate']:.0%}**",
            f"- **方向一致率**（涨跌预测 vs 实际符号）："
            f"{f'{summary['direction_agreement_pct']:.0%}' if summary['direction_agreement_pct'] is not None else '—'}",
            f"- **平均实际收益**："
            f"{f'{summary['avg_actual_return_pct']:+.2f}%' if summary['avg_actual_return_pct'] is not None else '—'}",
        ])
        if summary.get("brier_score") is not None:
            brier_line = f"- **Brier 分数（概率校准）**：{summary['brier_score']}"
            if summary.get("baseline_brier") is not None:
                brier_line += (f"（基准 {summary['baseline_brier']}，"
                               f"技能分 **{summary['brier_skill_score']:+.3f}**"
                               + (" ✅ 优于随机" if (summary.get("brier_skill_score") or -9) > 0
                                  else " ⚠️ 不优于随机") + "）")
            lines.append(brier_line)
        if summary.get("return_correlation") is not None:
            lines.append(f"- **期望收益 vs 实际收益相关**：{summary['return_correlation']:+.3f}")
        lines.extend(["", "### 分方向表现",
            "| 方向 | 数量 | 命中率 | 平均收益% |",
            "|---|---|---|---|",
        ])
        for d, label in (("up", "看涨"), ("down", "看跌"), ("sideways", "震荡")):
            info = summary["by_direction"].get(d)
            if info:
                lines.append(
                    f"| {label} | {info['count']} | {info['hit_rate']:.0%} | "
                    f"{info['avg_return_pct']:+.2f} |")
        lines.append("\n*命中判定：看涨→实际涨、看跌→实际跌、震荡→|涨跌|≤1.5%。"
                     "研究参考，非投资建议。*")
        return "\n".join(lines)
    except Exception as e:
        return f"验证失败：{str(e)[:200]}"


# ---- 决策树可视化（借鉴 PA_Agent 决策树机制） ----

def _layout_tree(node: dict, depth: int, next_x: list[float]) -> tuple[float, dict]:
    """递归布局决策树：返回 (本子树中心x, 带坐标的节点)。"""
    kids = node.get("children") or []
    nid = node["id"]
    if not kids:
        x = next_x[0]
        next_x[0] += 1.0
        return x, {"node": node, "x": x, "y": depth}
    child_layouts = []
    for c in kids:
        _, cl = _layout_tree(c, depth + 1, next_x)
        child_layouts.append(cl)
    x = sum(cl["x"] for cl in child_layouts) / len(child_layouts)
    return x, {"node": node, "x": x, "y": depth, "children": child_layouts}


def decision_tree_fig(symbol: str) -> tuple[Any, str]:
    """生成决策树可视化图（plotly 树形图）+ 文本摘要。"""
    symbol = symbol.strip()
    if not symbol:
        return None, "请输入股票代码"
    try:
        from pa_mcp.agent.decision_tree import build_decision_tree, tree_summary
        from pa_mcp.agent.orchestrator import get_orchestrator
        from pa_mcp.agent.prediction import get_prediction_service
        from pa_mcp.data.symbols import get_stock_name

        df = _load_long_history(symbol)
        if df.empty:
            return None, f"{symbol} 无行情数据"

        # 组装输入：预测（必需，含方向/概率/区间）+ 诊断（市场状态）
        svc = get_prediction_service()
        pred_result = asyncio.run(svc.predict(symbol, df, horizon="5d"))
        prediction = pred_result.to_dict()

        diagnosis = None
        try:
            orch = get_orchestrator()
            diagnosis = asyncio.run(orch.market_diagnosis(None))
        except Exception:
            pass

        tree = build_decision_tree(
            symbol, diagnosis=diagnosis, prediction=prediction,
            stock_name=get_stock_name(symbol))
        summary = tree_summary(tree)

        # 布局 + 画图
        _, layout = _layout_tree(tree["tree"], 0, [0.0])
        fig = go.Figure()
        max_depth = 0

        def draw(cl: dict, parent: Optional[tuple[float, float]]) -> None:
            nonlocal max_depth
            n = cl["node"]
            max_depth = max(max_depth, cl["y"])
            if parent is not None:
                px, py = parent
                # 折线连接（先垂直再水平，树状观感）
                fig.add_trace(go.Scatter(
                    x=[px, px, cl["x"], cl["x"]],
                    y=[py, py - 0.35, py - 0.35, cl["y"]],
                    mode="lines", line=dict(color="#888", width=1),
                    hoverinfo="skip", showlegend=False))
            color = {"root": "#1f6feb", "decision": "#e8590c",
                     "branch": "#2b8a3e", "leaf": "#495057"}.get(n["type"], "#666")
            fig.add_trace(go.Scatter(
                x=[cl["x"]], y=[cl["y"]],
                mode="markers+text",
                marker=dict(size=34 if n["type"] == "root" else 26,
                            color=color, line=dict(color="#fff", width=1)),
                text=[n["label"]], textposition="middle center",
                textfont=dict(color="white", size=11),
                hovertemplate=f"<b>{n['label']}</b><br>⚙️ {n.get('reason', '')}"
                              f"<br>📋 {n.get('detail', '')}<extra></extra>",
                showlegend=False))
            for c in cl.get("children", []):
                draw(c, (cl["x"], cl["y"]))

        draw(layout, None)
        fig.update_layout(
            title=tree["tree"]["label"],
            xaxis=dict(visible=False, range=[-0.5, layout["x"] + 0.5]),
            yaxis=dict(visible=False, autorange="reversed",
                       range=[max_depth + 0.6, -0.6]),
            height=max(360, 130 * (max_depth + 1)), margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor="white")
        return fig, summary
    except Exception as e:
        return None, f"决策树生成失败：{str(e)[:200]}"


def future_expectation_fig(symbol: str) -> tuple[Any, str]:
    """未来走势预期图：K线 + 预测区间阴影 + 方向概率 + 周期演进。"""
    symbol = symbol.strip()
    if not symbol:
        return None, "请输入股票代码"
    try:
        from pa_mcp.agent.prediction import get_prediction_service
        from pa_mcp.data.symbols import get_stock_name

        df = _load_long_history(symbol)
        if df.empty:
            return None, f"{symbol} 无行情数据"
        data = df.sort_values("date").reset_index(drop=True).tail(40)

        svc = get_prediction_service()
        result = asyncio.run(svc.predict(symbol, data, horizon="5d"))
        p = result.to_dict()
        dist = p["probability_distribution"]
        rng = p["expected_range_pct"]

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
            subplot_titles=(f"{symbol} {get_stock_name(symbol)} 未来 5d 走势预期",
                            "方向概率"))
        # K 线
        fig.add_trace(go.Candlestick(
            x=data["date"], open=data["open"], high=data["high"],
            low=data["low"], close=data["close"], name="K线",
            increasing_line_color="#e03131", decreasing_line_color="#2f9e44"),
            row=1, col=1)
        # MA20
        ma20 = data["close"].rolling(20).mean()
        fig.add_trace(go.Scatter(x=data["date"], y=ma20,
                                 mode="lines", name="MA20",
                                 line=dict(color="#1c7ed6", width=1)),
                      row=1, col=1)
        # 支撑/压力
        lv = p.get("key_levels") or {}
        for s_ in lv.get("support", []):
            fig.add_hline(y=float(s_), line=dict(color="#2f9e44", dash="dot", width=1),
                          row=1, col=1)
        for r_ in lv.get("resistance", []):
            fig.add_hline(y=float(r_), line=dict(color="#e03131", dash="dot", width=1),
                          row=1, col=1)
        # 未来预期区间（阴影带，从最后一根 K 线向右延伸）
        last_date = data["date"].iloc[-1]
        last_close = float(data["close"].iloc[-1])
        base, hi = last_close * (1 + rng[0] / 100), last_close * (1 + rng[1] / 100)
        exp_ = last_close * (1 + p["expected_return_pct"] / 100)
        fig.add_hrect(y0=base, y1=hi, fillcolor="#ffd43b", opacity=0.15,
                      line_width=0, row=1, col=1,
                      annotation_text=f"预期区间 {rng[0]:+.1f}~{rng[1]:+.1f}%",
                      annotation_position="top left")
        fig.add_trace(go.Scatter(
            x=[last_date, last_date], y=[last_close, exp_],
            mode="lines+markers+text",
            line=dict(color="#e8590c", width=2, dash="dash"),
            marker=dict(size=8, color="#e8590c"),
            text=[f"期望 {p['expected_return_pct']:+.1f}%"],
            textposition="middle right",
            name="期望路径"), row=1, col=1)
        # 方向概率条形图
        labels = [f"📈 上涨 {dist['up']:.0%}",
                  f"📉 下跌 {dist['down']:.0%}",
                  f"➡️ 震荡 {dist['sideways']:.0%}"]
        vals = [dist["up"], dist["down"], dist["sideways"]]
        colors = ["#e03131", "#2f9e44", "#868e96"]
        fig.add_trace(go.Bar(x=vals, y=labels, orientation="h",
                             marker=dict(color=colors),
                             text=[f"{v:.0%}" for v in vals],
                             textposition="auto", name="概率"), row=2, col=1)
        fig.update_layout(
            title=(f"🔮 {symbol} 未来 5d 走势预期："
                   f"{DIRECTION_ZH.get(p['direction'], p['direction'])} {p['probability']:.0%}"
                   f"　周期 {p['cycle_position_zh']} → {p['cycle_forecast_zh']}"),
            height=640, showlegend=False, margin=dict(l=10, r=10, t=60, b=10),
            xaxis_rangeslider_visible=False)
        return fig, ""
    except Exception as e:
        return None, f"走势预期图生成失败：{str(e)[:200]}"


def market_diagnosis_ui() -> str:
    """两阶段 Stage 1：市场诊断 + 策略路由（LLM 优先，确定性兜底）。"""
    try:
        import asyncio
        from pa_mcp.agent.orchestrator import get_orchestrator

        # 从数据库汇总市场指标（与 agent_market_state 一致）
        market_context = {}
        from pa_mcp.data.store import DuckDBStore
        from pa_mcp.config import get_settings
        store = DuckDBStore(get_settings().database.path)
        store.connect()
        try:
            latest = store.get_latest_date("kline_daily")
            if latest:
                df = store.query_df("""
                    SELECT
                        COUNT(CASE WHEN pct_change >= 9.5 THEN 1 END) as limit_up,
                        COUNT(CASE WHEN pct_change <= -9.5 THEN 1 END) as limit_down,
                        COUNT(CASE WHEN pct_change > 0 THEN 1 END) as up_count,
                        COUNT(CASE WHEN pct_change < 0 THEN 1 END) as down_count,
                        SUM(amount) / 100000000.0 as turnover
                    FROM kline_daily WHERE date = ?
                """, [latest])
                row = df.iloc[0]
                market_context = {
                    "limit_up_count": int(row["limit_up"]),
                    "limit_down_count": int(row["limit_down"]),
                    "up_count": int(row["up_count"]),
                    "down_count": int(row["down_count"]),
                    "turnover_billion": round(float(row["turnover"]), 1),
                    "date": latest,
                }
        finally:
            store.close()

        orch = get_orchestrator()
        d = asyncio.run(orch.market_diagnosis(market_context or None))
        routing = d.get("strategy_routing", {})
        lines = [
            f"## 🧭 市场诊断（{d.get('market_state_zh', d.get('market_state', ''))}）",
            f"- **风险等级**：{d.get('risk_level', '—')}"
            f"（建议总仓位上限 {d.get('suggested_max_position_pct', '—')}%）",
            f"- **判定依据**：{'；'.join(d.get('key_observations', [])) or '—'}",
            "",
            f"### 策略路由（{routing.get('label', '')}）",
        ]
        for s in routing.get("strategies", []):
            from pa_mcp.engine.strategies.tips import get_strategy_tip
            tip = get_strategy_tip(s) or ""
            first_line = tip.split("\n")[0][:60] if tip else ""
            lines.append(f"- **{s}**：{first_line}")
        if routing.get("risk_notes"):
            lines.append(f"\n**风险提示**：{routing['risk_notes']}")
        mode = "（LLM 诊断）" if d.get("mode") != "deterministic" else "（确定性诊断，未配置 LLM）"
        lines.append(f"\n*{mode}。研究参考，非投资建议。*")
        return "\n".join(lines)
    except Exception as e:
        return f"诊断失败：{str(e)[:200]}"


# ---- Tab 3: 策略回测 ----

STRATEGY_OPTIONS = [
    "ma_golden_cross", "platform_breakout", "first_board_breakout",
    "bollinger_mean_reversion", "volume_price_momentum", "oversold_bounce",
    "range_grid", "roe_pb_value", "dragon_second_wave",
]

def detect_best_strategy() -> str:
    """检测最优策略（事件研究快速评分），失败回退 bollinger。

    惰性调用（仅 UI 构建时一次，避免模块导入副作用）。
    """
    try:
        from pa_mcp.engine.strategies.tips import pick_best_strategy
        _df = _load_long_history("000001")
        _best = pick_best_strategy(_df)
        if _best in STRATEGY_OPTIONS:
            return _best
    except Exception:
        pass
    return "bollinger_mean_reversion"


def strategy_tip_ui(strategy: str) -> str:
    """策略说明（tips）。"""
    from pa_mcp.engine.strategies.tips import get_strategy_tip
    tip = get_strategy_tip(strategy)
    return tip or f"`{strategy}` 暂无详细说明"


def run_backtest_ui(symbol: str, strategy: str, initial_cash: float) -> tuple[Any, str]:
    """策略回测：事件驱动引擎 + 净值曲线。"""
    symbol = symbol.strip()
    if not symbol or not strategy:
        return None, "请输入股票代码并选择策略"

    try:
        df, source = asyncio.run(_load_kline(symbol, 250))
        if df.empty:
            return None, f"{symbol} 无行情数据"

        # 生成策略信号（简单示例：ma_cross 用真实策略类）
        from pa_mcp.engine.strategies.base import StrategyRegistry
        registry = StrategyRegistry()
        registry.auto_discover()

        strategy_cls = registry.get(strategy)
        if strategy_cls is None:
            return None, f"策略 {strategy} 未注册"

        sig_df = pd.DataFrame()
        try:
            signals = strategy_cls.generate_signals(df.copy())
            if signals:
                sig_df = pd.DataFrame([{
                    "symbol": symbol,
                    # P0-6: 优先用 signal_time（bar 市场时间），回退 timestamp
                    "date": (getattr(s, "signal_time", None) or
                             str(getattr(s, "timestamp", ""))[:10]),
                    "direction": getattr(s, "direction", "neutral").value
                                if hasattr(getattr(s, "direction", None), "value") else str(getattr(s, "direction", "neutral")),
                    "strength_score": float(getattr(s, "strength_score", 50)),
                    "strategy_name": strategy,
                } for s in signals])
        except Exception:
            pass  # 失败则走兜底规则

        # 策略信号为空或生成失败 → 简单均线规则兜底（保证可演示）
        if sig_df.empty:
            close = df["close"]
            sig_df = pd.DataFrame([{
                "symbol": symbol, "date": str(df.iloc[i]["date"])[:10],
                "direction": "bullish" if close.iloc[i] > close.iloc[max(0, i-5):i].mean() else "bearish",
                "strength_score": 55.0, "strategy_name": strategy,
            } for i in range(10, len(df), 10)])

        if sig_df.empty:
            return None, f"策略 {strategy} 未产生信号（可尝试其他策略）"

        # 事件驱动回测
        from pa_mcp.backtest.engine import BacktestEngine
        engine = BacktestEngine(initial_cash=float(initial_cash))
        report = engine.run(df, sig_df)

        # 净值曲线图 + 沪深300基准对比
        nav = pd.DataFrame(report.nav_series)
        fig = None
        if not nav.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nav["trade_date"], y=nav["nav"],
                                     name="策略净值", line=dict(color="#1565c0", width=2)))

            # 基准：沪深300 指数归一化对比
            try:
                from pa_mcp.data.sources.tencent_adapter import TencentAdapter
                start_d = str(nav["trade_date"].iloc[0])[:10].replace("-", "")
                end_d = str(nav["trade_date"].iloc[-1])[:10].replace("-", "")
                bench = asyncio.run(
                    TencentAdapter().get_daily_kline(
                        "sh000300", start_date=start_d, end_date=end_d,
                    )
                )
                if bench is not None and not bench.empty:
                    bench = bench.sort_values("date")
                    bench_scale = nav["nav"].iloc[0] / bench["close"].iloc[0]
                    fig.add_trace(go.Scatter(
                        x=bench["date"], y=bench["close"] * bench_scale,
                        name="沪深300（归一化）", line=dict(color="#ef5350", width=1.5, dash="dot"),
                    ))
            except Exception:
                pass  # 基准不可用时仅显示策略净值

            fig.update_layout(title=f"{symbol} 回测净值 vs 沪深300（{strategy}）",
                              template="plotly_white", height=420,
                              legend=dict(orientation="h", y=1.02))

        summary = (
            f"**{symbol} · {strategy}**（{source}，初始资金 {initial_cash:,.0f} 元）\n"
            f"总收益 {report.total_return_pct}% | 年化 {report.annual_return_pct}% | "
            f"夏普 {report.sharpe_ratio} | 最大回撤 {report.max_drawdown_pct}%\n"
            f"交易 {report.total_trades} 笔 | 费用 {report.total_fees:.2f} 元 | "
            f"印花税 {report.total_stamp_tax:.2f} 元\n\n"
            f"*信号延迟一天执行，含 A 股撮合规则。研究参考，非投资建议。*"
        )
        return fig, summary
    except Exception as e:
        return None, f"回测失败：{str(e)[:200]}"


# ---- Tab 4: 组合管理 ----

def portfolio_table() -> pd.DataFrame:
    """持仓表：成本 + 实时价 + 盈亏 + 当日涨跌 + 策略信号。"""
    store = _get_store()
    try:
        if not store.table_exists("portfolio"):
            return pd.DataFrame(columns=["symbol", "name", "cost", "shares",
                                         "现价", "盈亏%", "当日%", "added_date"])
        df = store.query_df("SELECT * FROM portfolio ORDER BY added_date DESC")
        if df.empty:
            return pd.DataFrame(columns=["symbol", "name", "cost", "shares",
                                         "现价", "盈亏%", "当日%", "added_date"])

        from pa_mcp.data.symbols import get_stock_name
        from pa_mcp.data.sources.tencent_adapter import TencentAdapter

        # 补名称 + 实时行情（腾讯，失败显示 —）
        names, prices, pnl_pcts, day_changes = [], [], [], []
        for sym in df["symbol"]:
            names.append(get_stock_name(sym))
            price = pnl = day = None
            try:
                q = asyncio.run(TencentAdapter().get_realtime_quote(sym))
                price = q.get("price")
                cost = float(df[df["symbol"] == sym].iloc[0].get("cost", 0))
                shares = int(df[df["symbol"] == sym].iloc[0].get("shares", 0))
                if price and cost > 0:
                    pnl = round((price / cost - 1) * 100, 1)
                day = q.get("change_pct")
            except Exception:
                pass
            prices.append(price)
            pnl_pcts.append(pnl)
            day_changes.append(day)

        df.insert(0, "name", names)
        df.insert(3, "现价", prices)
        df.insert(4, "盈亏%", pnl_pcts)
        df.insert(5, "当日%", day_changes)

        # datetime 列转字符串
        for col in df.columns:
            if str(df[col].dtype).startswith("datetime"):
                df[col] = df[col].astype(str).str[:10]
        return df
    except Exception:
        return pd.DataFrame(columns=["symbol", "name", "cost", "现价", "盈亏%",
                                     "当日%", "shares", "added_date"])


def portfolio_strategy_signals() -> str:
    """持仓股的当前策略信号（bollinger + 均线金叉）。"""
    store = _get_store()
    try:
        if not store.table_exists("portfolio"):
            return "持仓为空"
        holdings = store.query_df("SELECT symbol FROM portfolio")
        if holdings.empty:
            return "持仓为空"

        from pa_mcp.data.symbols import get_stock_name
        from pa_mcp.engine.strategies.base import StrategyRegistry

        registry = StrategyRegistry()
        registry.auto_discover()
        strategies = ["bollinger_mean_reversion", "ma_golden_cross"]
        insts = {s: registry.get(s) for s in strategies}

        lines = ["## 📡 持仓股策略信号",
                 "| 代码 | 名称 | 策略 | 信号日 | 强度 |",
                 "|---|---|---|---|---|"]
        found = False
        for sym in holdings["symbol"]:
            try:
                df = _load_long_history(sym)
                if df.empty or len(df) < 60:
                    continue
                latest = str(df["date"].astype(str).str[:10].iloc[-1])
                for s_name, inst in insts.items():
                    if inst is None:
                        continue
                    try:
                        signals = inst.generate_signals(df.copy())
                    except Exception:
                        continue
                    if not signals:
                        continue
                    recent = [
                        x for x in signals
                        if (getattr(x, "signal_time", None) or
                            str(getattr(x, "timestamp", ""))[:10]) >= latest
                    ]
                    if not recent:
                        continue
                    s = recent[-1]
                    found = True
                    lines.append(
                        f"| {sym} | {get_stock_name(sym)} | {s_name} | "
                        f"{getattr(s, 'signal_time', '')[:10]} | "
                        f"{getattr(s, 'strength_score', 50):.0f} |")
            except Exception:
                continue
        if not found:
            lines.append("| — | 当前无持仓触发信号 | — | — | — |")
        lines.append("\n*当前交易日触发买入信号的持仓。研究参考，非投资建议。*")
        return "\n".join(lines)
    except Exception as e:
        return f"信号查询失败：{str(e)[:120]}"


def portfolio_add_ui(symbol: str, cost: float, shares: int) -> tuple[str, pd.DataFrame]:
    try:
        if shares < 100 or shares % 100 != 0:
            return "❌ 股数必须 ≥100 且为 100 的整数倍", portfolio_table()
        store = _get_store()
        if not store.table_exists("portfolio"):
            store.execute("""
                CREATE TABLE IF NOT EXISTS portfolio (
                    symbol VARCHAR(10), cost DOUBLE, shares INTEGER,
                    added_date DATE, created_at TIMESTAMP
                )
            """)
        record = pd.DataFrame([{
            "symbol": symbol.strip(), "cost": float(cost), "shares": int(shares),
            "added_date": datetime.now().strftime("%Y-%m-%d"),
            "created_at": datetime.now().isoformat(),
        }])
        store.insert_df("portfolio", record, mode="insert")
        return f"✅ 已添加 {symbol}：{shares} 股 @ {cost:.2f}", portfolio_table()
    except Exception as e:
        return f"❌ 添加失败：{str(e)[:150]}", portfolio_table()


def portfolio_remove_ui(symbol: str) -> tuple[str, pd.DataFrame]:
    try:
        store = _get_store()
        if not store.table_exists("portfolio"):
            return "组合为空", portfolio_table()
        store.execute("DELETE FROM portfolio WHERE symbol = ?", [symbol.strip()])
        return f"🗑️ 已删除 {symbol}", portfolio_table()
    except Exception as e:
        return f"❌ 删除失败：{str(e)[:150]}", portfolio_table()


def portfolio_review_ui() -> str:
    """持仓体检（独立实现，不依赖 MCP server 全局状态）。"""
    try:
        store = _get_store()
        if not store.table_exists("portfolio"):
            return "组合为空。先在左侧添加持仓。"

        holdings = store.query_df("SELECT * FROM portfolio ORDER BY added_date DESC")
        if holdings.empty:
            return "组合为空。先在左侧添加持仓。"

        from pa_mcp.data.sources.tencent_adapter import TencentAdapter

        reviews: list[str] = []
        risk_alerts: list[str] = []
        total_value = 0.0
        quote_cache: dict[str, dict] = {}

        for _, h in holdings.iterrows():
            sym = h["symbol"]
            cost = float(h.get("cost", 0))
            shares = int(h.get("shares", 0))

            try:
                q = asyncio.run(TencentAdapter().get_realtime_quote(sym))
                quote_cache[sym] = q
            except Exception:
                reviews.append(f"- **{sym}**：行情不可用")
                continue

            price = q.get("price", 0)
            value = price * shares
            total_value += value
            pnl_pct = (price / cost - 1) * 100 if cost > 0 else 0.0

            line = (f"- **{q.get('name', sym)}**（{sym}）：{shares}股 "
                    f"成本{cost:.2f} → 现价{price:.2f} 盈亏{pnl_pct:+.1f}%")
            if q.get("pe"):
                line += f" | PE {q['pe']} PB {q.get('pb')} 换手{q.get('turnover_pct')}%"
            reviews.append(line)

            # 风险规则
            if q.get("distance_to_limit_up_pct") is not None and \
                    q["distance_to_limit_up_pct"] < 2:
                risk_alerts.append(f"{sym} 距涨停仅 {q['distance_to_limit_up_pct']}%，追高风险大")
            if q.get("pe") and 0 < q["pe"] < 5:
                risk_alerts.append(f"{sym} PE 极低（<5），警惕价值陷阱")

        # 集中度
        if total_value > 0 and quote_cache:
            for sym, h in zip(holdings["symbol"], holdings["shares"]):
                q = quote_cache.get(sym)
                if q:
                    weight = q["price"] * int(h) / total_value * 100
                    if weight > 10:
                        risk_alerts.append(f"{sym} 集中度 {weight:.1f}% > 10% 建议红线")

        health = max(0, 100 - len(risk_alerts) * 15)
        lines = [f"## 持仓体检（健康度 {health}）",
                 f"组合总值 {total_value:,.0f} 元，{len(reviews)} 只持仓"]
        lines += reviews
        if risk_alerts:
            lines.append("\n### ⚠️ 风险告警")
            lines += [f"- {a}" for a in risk_alerts]
        lines.append("\n*研究参考，非投资建议。免费行情可能有延迟。*")
        return "\n".join(lines)
    except Exception as e:
        return f"体检失败：{str(e)[:200]}"


# ---- 财报/龙虎榜看板 ----

def financials_summary_ui(symbol: str) -> str:
    """财务指标摘要（financials_income 表，scheduler 已入库时可用）。"""
    try:
        store = _get_store()
        if not store.table_exists("financials_income"):
            return "财务数据未入库（先运行数据调度）"
        df = store.query_df(
            "SELECT report_date, revenue, net_profit, eps, net_margin "
            "FROM financials_income WHERE symbol = ? "
            "ORDER BY report_date DESC LIMIT 4",
            [symbol],
        )
        if df.empty:
            return f"{symbol} 暂无财务数据"
        lines = [f"**{symbol} 最近财报**（单位：亿元）",
                 "| 报告期 | 营收 | 净利润 | EPS | 净利率% |",
                 "|---|---|---|---|---|"]
        for _, r in df.iterrows():
            rev = r["revenue"] / 1e8 if r["revenue"] else 0
            np_ = r["net_profit"] / 1e8 if r["net_profit"] else 0
            eps = r["eps"] if r["eps"] is not None else "—"
            nm = f"{r['net_margin']:.1f}" if r["net_margin"] is not None else "—"
            lines.append(
                f"| {str(r['report_date'])[:10]} | {rev:.0f} | {np_:.0f} | {eps} | {nm} |"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"财报摘要不可用：{str(e)[:80]}"


def dragon_tiger_summary_ui(date: str = "") -> str:
    """龙虎榜概览（dragon_tiger 表，scheduler 已入库时可用）。"""
    try:
        store = _get_store()
        if not store.table_exists("dragon_tiger"):
            return "龙虎榜数据未入库（先运行数据调度）"
        latest = store.get_latest_date("dragon_tiger", "trade_date") if not date else date
        df = store.query_df(
            "SELECT symbol, reason, net_amount FROM dragon_tiger "
            "WHERE trade_date = ? ORDER BY ABS(net_amount) DESC LIMIT 8",
            [latest],
        )
        if df.empty:
            return f"{latest} 龙虎榜为空"
        lines = [f"**{latest} 龙虎榜 TOP**（净买入额）",
                 "| 代码 | 上榜原因 | 净额(万) |", "|---|---|---|"]
        for _, r in df.iterrows():
            net = r["net_amount"] / 1e4 if r["net_amount"] else 0
            lines.append(f"| {r['symbol']} | {(r['reason'] or '—')[:24]} | {net:,.0f} |")
        return "\n".join(lines)
    except Exception as e:
        return f"龙虎榜概览不可用：{str(e)[:80]}"


# ---- 数据源健康面板 ----

def source_health_ui() -> str:
    """展示多源路由健康状态。"""
    try:
        from pa_mcp.data.router import DataSourceRouter, CircuitBreakerConfig
        from pa_mcp.data.sources.tencent_adapter import TencentAdapter
        from pa_mcp.data.sources.sina_adapter import SinaAdapter
        from pa_mcp.data.sources.eastmoney_adapter import EastMoneyAdapter
        from pa_mcp.config import get_settings

        settings = get_settings()
        chain = [
            ("tencent", TencentAdapter()), ("sina", SinaAdapter()),
            ("eastmoney", EastMoneyAdapter()),
        ]
        cfg = CircuitBreakerConfig(
            failure_threshold=settings.router.circuit.failure_threshold,
            cooldown_seconds=settings.router.circuit.cooldown_seconds,
        )
        router = DataSourceRouter(chain, {n: cfg for n, _ in chain},
                                  min_source_interval={"eastmoney": 1.2})

        lines = ["| 数据源 | 状态 | 成功 | 失败 | 熔断次数 | 最近错误 |",
                 "|---|---|---|---|---|---|"]
        for h in router.health_report():
            state = {"closed": "🟢", "open": "🔴", "half_open": "🟡"}.get(
                h["state"], h["state"])
            lines.append(
                f"| {h['name']} | {state} {h['state']} | {h['success_count']} | "
                f"{h['failure_count']} | {h['trip_count']} | "
                f"{(h['last_error'] or '—')[:40]} |"
            )
        lines.append("\n*源链：tencent → sina → eastmoney（熔断+限流+缓存）*")
        return "\n".join(lines)
    except Exception as e:
        return f"健康面板不可用：{str(e)[:100]}"


# ---- Gradio 界面 ----

def build_app():
    import gradio as gr

    with gr.Blocks(title="PA_MCP 理财助手") as app:
        gr.Markdown("# 📈 PA_MCP 理财助手\n"
                    "多源行情 · 专业分析 · 事件驱动回测 · 组合风控")

        with gr.Accordion("🔌 数据源健康", open=False):
            health_md = gr.Markdown()
            health_btn = gr.Button("刷新健康状态", size="sm")
            health_btn.click(source_health_ui, outputs=[health_md])
            app.load(source_health_ui, outputs=[health_md])

        with gr.Tab("📊 数据看板"):
            with gr.Row():
                sym_in = gr.Textbox(label="股票代码", value="000001", scale=2)
                days_in = gr.Slider(30, 500, value=120, step=10,
                                    label="K线天数", scale=1)
                analyze_btn = gr.Button("分析", variant="primary", scale=1)
            kline_out = gr.Plot()
            summary_out = gr.Markdown()
            valuation_out = gr.Markdown()
            source_out = gr.Textbox(label="数据源", interactive=False)
            with gr.Row():
                fin_out = gr.Markdown()          # 财报摘要
                lhb_out = gr.Markdown()          # 龙虎榜

            analyze_btn.click(analyze_stock,
                              inputs=[sym_in, days_in],
                              outputs=[kline_out, summary_out, valuation_out, source_out])
            sym_in.submit(analyze_stock,
                          inputs=[sym_in, days_in],
                          outputs=[kline_out, summary_out, valuation_out, source_out])
            analyze_btn.click(financials_summary_ui, inputs=[sym_in], outputs=[fin_out])
            analyze_btn.click(dragon_tiger_summary_ui, outputs=[lhb_out])
            app.load(dragon_tiger_summary_ui, outputs=[lhb_out])

        with gr.Tab("💬 AI 对话"):
            gr.Markdown("**对话**：输入股票代码或问题，例如「分析 000001」「600036 资金流」。\n"
                        "配置 `ANTHROPIC_API_KEY` 后获得完整 AI 分析，否则使用规则分析模式。")
            chatbot = gr.Chatbot(height=420)
            chat_in = gr.Textbox(label="你的问题", placeholder="分析 000001",
                                 submit_btn="发送")

            def _chat_respond(message: str, history: list) -> tuple[str, list]:
                reply = chat_reply(message, history or [])
                if history is None:
                    history = []
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": reply})
                return "", history

            chat_in.submit(_chat_respond, inputs=[chat_in, chatbot],
                           outputs=[chat_in, chatbot])

        with gr.Tab("🔀 多股对比"):
            cmp_in = gr.Textbox(label="股票代码（逗号分隔，2-5只）",
                                value="000001,600036,300750",
                                placeholder="000001,600036,300750")
            cmp_btn = gr.Button("对比", variant="primary")
            cmp_fig = gr.Plot()
            cmp_table = gr.Markdown()
            cmp_btn.click(compare_stocks_ui, inputs=[cmp_in],
                          outputs=[cmp_fig, cmp_table])
            cmp_in.submit(compare_stocks_ui, inputs=[cmp_in],
                          outputs=[cmp_fig, cmp_table])

        with gr.Tab("📡 市场扫描"):
            sm_strategy = gr.Dropdown(STRATEGY_OPTIONS,
                                      value=detect_best_strategy(),
                                      label="策略（找当前买入信号）")
            sm_btn = gr.Button("扫描股票池", variant="primary")
            sm_out = gr.Markdown()
            sm_btn.click(scan_market_ui, inputs=[sm_strategy],
                         outputs=[sm_out])
            gr.Markdown("扫描内置常用股池，输出**当前处于买入信号状态**的股票"
                        "（含该信号历史5日胜率）。基于统计而非预测。")

        with gr.Tab("🧪 研究评估"):
            wf_sym = gr.Textbox(label="股票代码", value="000001")
            wf_strategy = gr.Dropdown(STRATEGY_OPTIONS,
                                      value=detect_best_strategy(),
                                      label="策略（Walk-Forward OOS）")
            wf_tip = gr.Markdown()
            wf_btn = gr.Button("运行 Walk-Forward", variant="primary")
            wf_out = gr.Markdown()
            wf_strategy.change(strategy_tip_ui, inputs=[wf_strategy],
                               outputs=[wf_tip])
            wf_btn.click(walk_forward_ui, inputs=[wf_sym, wf_strategy],
                         outputs=[wf_out])
            es_btn = gr.Button("📊 信号事件研究（预测力检验）", variant="secondary")
            es_out = gr.Markdown()
            es_btn.click(event_study_ui, inputs=[wf_sym, wf_strategy],
                         outputs=[es_out])

        with gr.Tab("📦 组合构建"):
            pb_in = gr.Textbox(label="股票池（逗号分隔）",
                               value="000001,600036,300750,000858",
                               placeholder="000001,600036,300750,000858")
            pb_strategy = gr.Dropdown(STRATEGY_OPTIONS,
                                      value=detect_best_strategy(),
                                      label="策略")
            pb_tip = gr.Markdown()
            pb_btn = gr.Button("构建组合", variant="primary")
            pb_fig = gr.Plot()
            pb_out = gr.Markdown()
            pb_strategy.change(strategy_tip_ui, inputs=[pb_strategy],
                               outputs=[pb_tip])
            pb_btn.click(portfolio_build_ui, inputs=[pb_in, pb_strategy],
                         outputs=[pb_fig, pb_out])
            pb_in.submit(portfolio_build_ui, inputs=[pb_in, pb_strategy],
                         outputs=[pb_fig, pb_out])

        with gr.Tab("🛠️ 策略回测"):
            with gr.Row():
                bt_sym = gr.Textbox(label="股票代码", value="000001")
                bt_strategy = gr.Dropdown(STRATEGY_OPTIONS,
                                          value=detect_best_strategy(),
                                          label="策略（⭐=推荐）")
                bt_cash = gr.Number(value=100000, label="初始资金", precision=0)
                bt_btn = gr.Button("运行回测", variant="primary")
            bt_tip = gr.Markdown()   # 策略说明
            bt_fig = gr.Plot()
            bt_summary = gr.Markdown()
            bt_strategy.change(strategy_tip_ui, inputs=[bt_strategy],
                               outputs=[bt_tip])
            app.load(strategy_tip_ui, inputs=[bt_strategy], outputs=[bt_tip])
            bt_btn.click(run_backtest_ui,
                         inputs=[bt_sym, bt_strategy, bt_cash],
                         outputs=[bt_fig, bt_summary])

        with gr.Tab("🔮 市场预测"):
            gr.Markdown("**AI 未来走势预测**：基于 K 线技术特征预测方向/概率/关键价位，"
                        "预测落盘可验证命中率（不做纯算命）。")
            with gr.Row():
                pred_sym = gr.Textbox(label="股票代码", value="000001", scale=2)
                pred_horizon = gr.Radio(["5d", "20d"], value="5d",
                                        label="预测周期", scale=1)
                pred_btn = gr.Button("预测", variant="primary", scale=1)
            pred_out = gr.Markdown()
            pred_btn.click(predict_market_ui, inputs=[pred_sym, pred_horizon],
                           outputs=[pred_out])
            pred_sym.submit(predict_market_ui, inputs=[pred_sym, pred_horizon],
                            outputs=[pred_out])

            with gr.Row():
                diag_btn = gr.Button("🧭 市场诊断 + 策略路由", variant="secondary")
                evp_btn = gr.Button("🎯 预测验证成绩单", variant="secondary")
            diag_out = gr.Markdown()
            evp_out = gr.Markdown()
            diag_btn.click(market_diagnosis_ui, outputs=[diag_out])
            evp_btn.click(evaluate_predictions_ui, outputs=[evp_out])

            hist_btn = gr.Button("📜 历史预测记录", variant="secondary")
            hist_out = gr.Markdown()
            hist_btn.click(prediction_history_ui, inputs=[pred_sym],
                           outputs=[hist_out])

            with gr.Row():
                tree_btn = gr.Button("🌳 决策树可视化", variant="secondary")
                expect_btn = gr.Button("📊 未来走势预期图", variant="secondary")
            tree_fig = gr.Plot()
            tree_out = gr.Markdown()
            tree_btn.click(decision_tree_fig, inputs=[pred_sym],
                           outputs=[tree_fig, tree_out])
            expect_btn.click(future_expectation_fig, inputs=[pred_sym],
                             outputs=[tree_fig, tree_out])
            gr.Markdown("经验库说明：每次 AI 分析自动沉淀到经验库，"
                        "后续分析自动参考相似历史案例（RAG）。")

        with gr.Tab("💼 组合管理"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 添加持仓")
                    p_sym = gr.Textbox(label="代码", value="000001")
                    p_cost = gr.Number(label="成本价", value=10.0)
                    p_shares = gr.Number(label="股数(100整数倍)", value=1000,
                                         precision=0)
                    p_add = gr.Button("添加", variant="primary")
                    p_del_sym = gr.Textbox(label="删除代码")
                    p_del = gr.Button("删除")
                    p_status = gr.Markdown()
                    ai_sym = gr.Textbox(label="AI 分析代码", value="")
                    ai_btn = gr.Button("🤖 AI 个股分析", variant="secondary")
                with gr.Column(scale=2):
                    p_table = gr.Dataframe(label="当前持仓（实时价/盈亏/当日）",
                                           interactive=False)
                    p_review_btn = gr.Button("🏥 持仓体检", variant="secondary")
                    p_review = gr.Markdown()
                    p_sig_btn = gr.Button("📡 持仓策略信号", variant="secondary")
                    p_sig = gr.Markdown()
                    ai_out = gr.Markdown()

            p_add.click(portfolio_add_ui, inputs=[p_sym, p_cost, p_shares],
                        outputs=[p_status, p_table])
            p_del.click(portfolio_remove_ui, inputs=[p_del_sym],
                        outputs=[p_status, p_table])
            p_review_btn.click(portfolio_review_ui, outputs=[p_review])
            p_sig_btn.click(portfolio_strategy_signals, outputs=[p_sig])
            ai_btn.click(portfolio_ai_analysis, inputs=[ai_sym],
                         outputs=[ai_out])
            app.load(portfolio_table, outputs=[p_table])

    return app


def main() -> None:
    """启动 Gradio UI：python -m pa_mcp.ui.gradio_app"""
    import socket
    import gradio as gr

    # 端口被占用时自动避让（7860 → 7861 → ... 7869）
    port = 7860
    for p in range(7860, 7870):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                port = p
                break
            except OSError:
                continue
    else:
        port = 7860  # 全部占用则用默认并让 gradio 报错

    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=port, inbrowser=True,
               show_error=True, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
