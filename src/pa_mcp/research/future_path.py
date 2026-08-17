# [AI:BEGIN]
# PA_MCP - Research: 未来 K 线路径预测（LLM 全维分析 → 预测 K 线图）
#
# 把 agent 能拿到的全部上下文喂给 LLM：
#   历史 K 线特征（均线/RSI/MACD/ADX/ATR/量比/布林/周期位置）
#   + 策略信号（布林/金叉/海龟最新触发）
#   + 市场状态（5 态诊断 + 仓位上限）
#   + 板块/情绪上下文（所属板块 RS、游资情绪阶段）
#   + 近期走势描述（近 20 日高低/趋势）
# → LLM 输出未来 N 日三条情景路径（bull/base/bear OHLC 序列）+ 逻辑 +
#   关键价位 + 风险。画成「历史 K 线 + 预测 K 线（虚线）」图。
#
# 确定性降级：无 LLM 时用预测方向 + 历史波动率（ATR）生成统计模拟路径，
#   诚实标注「统计模拟」。
# 预测路径是研究模拟，不是真实行情——图上明确标注。
# [AI:END]

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_HORIZON = 20  # 预测未来交易日数


def _store():
    from pa_mcp.config import get_settings
    from pa_mcp.data.store import DuckDBStore
    store = DuckDBStore(get_settings().database.path)
    store.connect()
    return store


# ---- 全维度上下文构建 ----

def _recent_summary(df: pd.DataFrame) -> str:
    """近 20 日走势文字摘要。"""
    d = df.tail(20)
    last = float(d["close"].iloc[-1])
    hi = float(d["high"].max())
    lo = float(d["low"].min())
    chg20 = (last / float(d["close"].iloc[0]) - 1) * 100
    up_days = int((d["close"].diff().dropna() > 0).sum())
    return (f"近20日：收 {last:.2f}（区间 {lo:.2f}-{hi:.2f}，"
            f"20日涨跌 {chg20:+.1f}%，上涨 {up_days}/20 天）")


def _strategy_signals(df: pd.DataFrame) -> str:
    """核心策略最新信号摘要（best-effort）。"""
    parts = []
    try:
        from pa_mcp.engine.strategies.base import StrategyRegistry
        reg = StrategyRegistry()
        reg.auto_discover()
        for name in ("bollinger_mean_reversion", "ma_golden_cross", "turtle"):
            inst = reg.get(name)
            if inst is None:
                continue
            try:
                sigs = inst.generate_signals(df.copy())
            except Exception:  # noqa: BLE001
                continue
            if not sigs:
                continue
            s = sigs[-1]
            parts.append(f"{name}:{getattr(s, 'direction', 'neutral').value if hasattr(getattr(s, 'direction', None), 'value') else getattr(s, 'direction', 'neutral')}"
                         f"({getattr(s, 'signal_time', '')[:10]})")
    except Exception:  # noqa: BLE001
        pass
    return "；".join(parts) if parts else "无触发信号"


def _market_context() -> str:
    """市场状态 + 情绪（best-effort）。"""
    ctx = []
    try:
        from pa_mcp.engine.market_state import MarketStateDetector, MarketIndicators
        from pa_mcp.research.sentiment_cycle import SentimentCycleAnalyzer
        store = _store()
        try:
            latest = store.get_latest_date("kline_daily")
            if latest:
                df = store.query_df("""
                    SELECT COUNT(CASE WHEN pct_change >= 9.5 THEN 1 END) lu,
                           COUNT(CASE WHEN pct_change <= -9.5 THEN 1 END) ld,
                           COUNT(CASE WHEN pct_change > 0 THEN 1 END) up,
                           COUNT(CASE WHEN pct_change < 0 THEN 1 END) dn,
                           SUM(amount) / 1e8 as turnover
                    FROM kline_daily WHERE date = ?""", [latest])
                row = df.iloc[0]
                ind = MarketIndicators(
                    limit_up_count=int(row["lu"]), limit_down_count=int(row["ld"]),
                    up_count=int(row["up"]), down_count=int(row["dn"]),
                    turnover_billion=float(row["turnover"]))
                state = MarketStateDetector().detect(ind)
                ctx.append(f"市场状态:{state.label if hasattr(state, 'label') else state.value}")
            sc = SentimentCycleAnalyzer(store_path=None)
            s = sc.sentiment_summary() or {}
            if s:
                ctx.append(f"情绪:{s.get('stage_zh', '')}({s.get('sentiment_score', '')})")
        finally:
            store.close()
    except Exception:  # noqa: BLE001
        pass
    return "；".join(ctx) if ctx else "市场上下文不可用"


def build_context(symbol: str, kline_df: pd.DataFrame) -> dict[str, str]:
    """聚合全维度上下文（K线特征/策略/市场/情绪/板块）。"""
    from pa_mcp.agent.prediction import extract_features, format_features

    feats = extract_features(kline_df)
    ctx: dict[str, str] = {
        "features": format_features(feats) if "error" not in feats else "无数据",
        "recent": _recent_summary(kline_df),
        "strategies": _strategy_signals(kline_df),
        "market": _market_context(),
        "sector": "",
    }
    # 板块上下文（best-effort：sector_daily 有数据时取轮动速度与强势板块）
    try:
        from pa_mcp.research.sector_rotation import SectorRotationAnalyzer
        a = SectorRotationAnalyzer().analyze()
        if "error" not in a:
            top = "、".join(s["name"] for s in a.get("ranked_sectors", [])[:3])
            ctx["sector"] = (f"板块轮动速度:{a.get('rotation_speed')}，"
                             f"强势板块:{top}")
    except Exception:  # noqa: BLE001
        pass
    return ctx


# ---- LLM 路径生成 ----

_PATH_PROMPT = """你是资深 A 股量化研究员。今天是 {today}。基于以下全部上下文，预测 {symbol} 未来 {horizon} 个交易日的走势路径。

【历史特征】{features}
【近期走势】{recent}
【策略信号】{strategies}
【市场环境】{market}
【板块环境】{sector}

输出三条情景路径（每情景 {horizon} 根日 K：date/open/high/low/close，date 用未来自然日 YYYY-MM-DD，跳过周末更真实）：
- bull：乐观情景（催化兑现）
- base：最可能情景
- bear：悲观情景

要求：
0. date 必须是 {today} 之后的未来日期（YYYY-MM-DD，跳过周末）
1. 第一根 open 与最后收盘价衔接（差距 <1%）
2. 波动率与 ATR 匹配：单日振幅 0.5%-3%，不编造极端值
3. 路径与历史趋势/因子/市场环境一致（如市场冰点则 bear 权重高）
4. JSON 保持紧凑（无多余空格/缩进），logic ≤40 字，先输出 base 情景
5. 只输出合法 JSON：
{{"scenarios": [{{"name": "bull", "weight": 0.3, "logic": "…", "path": [{{"date": "YYYY-MM-DD", "open": 0, "high": 0, "low": 0, "close": 0}}]}}],
  "base_direction": "up/down/sideways",
  "expected_return_pct": 0.0,
  "key_levels": {{"support": [], "resistance": []}},
  "logic": "整体逻辑一句话",
  "risks": ["风险1"]}}
预测是研究模拟，不是投资建议。"""


def _parse_path_json(raw: dict) -> Optional[dict]:
    """校验 LLM 输出结构，返回清洗后的路径数据。"""
    if not isinstance(raw, dict) or "scenarios" not in raw:
        return None
    scenarios = []
    for sc in raw.get("scenarios", [])[:3]:
        path = []
        for p in sc.get("path", [])[:30]:
            try:
                path.append({
                    "date": str(p["date"])[:10],
                    "open": float(p["open"]), "high": float(p["high"]),
                    "low": float(p["low"]), "close": float(p["close"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
        if len(path) >= 5:  # 至少 5 根才算有效情景
            scenarios.append({"name": sc.get("name", "base"),
                              "weight": float(sc.get("weight", 0.33)),
                              "logic": sc.get("logic", ""), "path": path})
    if not scenarios:
        return None
    return {
        "scenarios": scenarios,
        "base_direction": raw.get("base_direction", "sideways"),
        "expected_return_pct": float(raw.get("expected_return_pct", 0.0)),
        "key_levels": raw.get("key_levels", {}),
        "logic": raw.get("logic", ""),
        "risks": raw.get("risks", []),
    }


async def _compute_drift(df: pd.DataFrame) -> float:
    """确定性预测方向漂移（async 上下文直接 await，避免嵌套 asyncio.run）。"""
    try:
        from pa_mcp.agent.prediction import get_prediction_service
        svc = get_prediction_service()
        r = await svc.predict(str(df["symbol"].iloc[0]), kline_df=df,
                              horizon="5d", use_llm=False)
        return (r.expected_return_pct / 100) / 5
    except Exception:  # noqa: BLE001
        return 0.0


def _deterministic_paths(df: pd.DataFrame, horizon: int,
                         drift: float = 0.0) -> dict:
    """无 LLM 时统计模拟：预测方向 + ATR 波动率生成三情景路径。

    基于当前预测方向（确定性统计）与历史波动率，诚实标注统计模拟。
    """
    from pa_mcp.agent.prediction import extract_features
    import random

    last_close = float(df["close"].iloc[-1])
    last_date = pd.to_datetime(df["date"].iloc[-1])
    feats = extract_features(df)
    atr_pct = float(feats.get("atr_pct", 1.5) or 1.5) / 100
    scales = {"bull": 1.6, "base": 1.0, "bear": 0.4}
    weights = {"bull": 0.3, "base": 0.45, "bear": 0.25}
    random.seed(int(last_close * 100) % 10000)
    scenarios = []
    for name, scale in scales.items():
        price = last_close
        path = []
        d = last_date
        for i in range(horizon):
            d = d + timedelta(days=1)
            while d.weekday() >= 5:  # 跳过周末
                d = d + timedelta(days=1)
            ret = drift * scale + random.gauss(0, atr_pct * 0.6 * scale)
            o = price
            c = price * (1 + ret)
            hi = max(o, c) * (1 + abs(random.gauss(0, atr_pct * 0.4)))
            lo = min(o, c) * (1 - abs(random.gauss(0, atr_pct * 0.4)))
            path.append({"date": d.strftime("%Y-%m-%d"),
                         "open": round(o, 2), "high": round(hi, 2),
                         "low": round(lo, 2), "close": round(c, 2)})
            price = c
        scenarios.append({"name": name, "weight": weights[name],
                          "logic": "统计模拟（历史波动率+预测方向漂移）",
                          "path": path})
    return {
        "scenarios": scenarios,
        "base_direction": "up" if drift > 0.001 else ("down" if drift < -0.001 else "sideways"),
        "expected_return_pct": round(drift * horizon * 100, 1),
        "key_levels": {"support": [float(feats.get("support_20d", last_close * 0.95))],
                       "resistance": [float(feats.get("resistance_20d", last_close * 1.05))]},
        "logic": "统计模拟：方向来自确定性预测，波动率来自历史 ATR",
        "risks": ["统计模拟非 LLM 分析；配置 LLM 后启用全维分析"],
        "mode": "deterministic",
    }


async def predict_future_path(symbol: str, kline_df: pd.DataFrame,
                              horizon: int = DEFAULT_HORIZON,
                              use_llm: bool = True) -> dict[str, Any]:
    """未来 K 线路径预测（LLM 全维分析 → 三情景 OHLC 路径）。

    Args:
        symbol: 股票代码
        kline_df: 历史日线（≥60 根效果佳）
        horizon: 预测交易日数（默认 20，≤30）
        use_llm: 是否尝试 LLM（无配置自动降级统计模拟）

    Returns:
        history: 历史 K 线（近 60 根，含 date/open/high/low/close）
        scenarios: 三情景路径 [{name, weight, logic, path}]
        base_direction / expected_return_pct / key_levels / logic / risks
        mode: llm | deterministic
        report: markdown 摘要
    """
    if kline_df is None or kline_df.empty:
        return {"error": "无行情数据"}
    horizon = max(5, min(horizon, 30))
    df = kline_df.sort_values("date").reset_index(drop=True)
    last_close = float(df["close"].iloc[-1])

    result: Optional[dict] = None
    mode = "deterministic"
    ctx = build_context(symbol, df)
    if use_llm:
        try:
            from pa_mcp.agent.llm_factory import ensure_llm_adapter
            from pa_mcp.agent.llm_port import LLMCallParams
            adapter = ensure_llm_adapter()
            if adapter is not None:
                params = LLMCallParams(
                    system_prompt="你是有经验的 A 股量化研究员。只输出合法 JSON，"
                                  "不输出任何其他文本。预测是研究模拟，非投资建议。",
                    user_prompt=_PATH_PROMPT.format(
                        symbol=symbol, horizon=horizon,
                        today=datetime.now().strftime("%Y-%m-%d"), **ctx),
                    mode="deep", max_tokens=8000,
                    timeout_seconds=180,  # 长输出（20 根×3 情景）需放宽超时
                )
                raw = await adapter.chat_json(params)
                parsed = _parse_path_json(raw)
                if parsed:
                    parsed["mode"] = "llm"
                    result = parsed
                    mode = "llm"  # 同步更新返回值标签（历史 bug：LLM 成功
                    # 但 mode 恒为 deterministic）
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM 路径预测失败，降级统计模拟: %s", str(e)[:80])

    if result is None:
        drift = await _compute_drift(df)
        result = _deterministic_paths(df, horizon, drift)

    # 拼接历史（近 60 根）与预测
    history = df.tail(60)[["date", "open", "high", "low", "close"]]
    history["date"] = history["date"].astype(str).str[:10]

    # 关键价位补默认（缺失时）
    lv = result.get("key_levels") or {}
    lv.setdefault("support", [round(last_close * 0.95, 2)])
    lv.setdefault("resistance", [round(last_close * 1.05, 2)])

    report = format_report(symbol, result, last_close, mode)
    return {
        "symbol": symbol,
        "history": history.to_dict("records"),
        "scenarios": result["scenarios"],
        "base_direction": result.get("base_direction", "sideways"),
        "expected_return_pct": result.get("expected_return_pct", 0.0),
        "key_levels": lv,
        "logic": result.get("logic", ""),
        "risks": result.get("risks", []),
        "mode": mode,
        "report": report,
    }


def format_report(symbol: str, result: dict, last_close: float,
                  mode: str) -> str:
    """路径预测结果 → markdown 摘要。"""
    mode_txt = "🤖 LLM 全维分析" if mode == "llm" else "📐 统计模拟（未配置 LLM）"
    lines = [f"## 📈 {symbol} 未来 K 线路径预测（{mode_txt}）",
             f"**基准方向**：{result.get('base_direction', 'sideways')}；"
             f"**期望收益**：{result.get('expected_return_pct', 0):+.1f}%",
             f"**逻辑**：{result.get('logic', '')}", ""]
    for sc in result.get("scenarios", []):
        path = sc.get("path", [])
        if not path:
            continue
        end = path[-1]["close"]
        chg = (end / last_close - 1) * 100
        lines.append(f"- **{sc.get('name', '')}**（权重 {sc.get('weight', 0):.0%}，"
                     f"终点 {end:.2f}，{chg:+.1f}%）：{sc.get('logic', '')[:60]}")
    lv = result.get("key_levels") or {}
    lines.append(f"\n**关键价位**：支撑 {lv.get('support', [])} / "
                 f"压力 {lv.get('resistance', [])}")
    if result.get("risks"):
        lines.append("**风险**：" + "；".join(result["risks"][:3]))
    lines.append("\n*预测路径为研究模拟，非真实行情、非投资建议。*")
    return "\n".join(lines)


def build_chart(data: dict[str, Any]):
    """未来 K 线预测图：历史 K 线（实） + 三情景预测 K 线（虚线）。

    依赖 plotly（函数内 import，UI 专用）；返回 plotly Figure。
    预测段用半透明虚线 + 标注「预测模拟」。
    """
    import plotly.graph_objects as go

    hist = data.get("history", [])
    scenarios = data.get("scenarios", [])
    if not hist:
        return None

    fig = go.Figure()
    hx = [h["date"] for h in hist]
    fig.add_trace(go.Candlestick(
        x=hx, open=[h["open"] for h in hist],
        high=[h["high"] for h in hist],
        low=[h["low"] for h in hist],
        close=[h["close"] for h in hist],
        name="历史K线", increasing_line_color="#e03131",
        decreasing_line_color="#2f9e44"))

    # MA20（历史段）
    closes = [h["close"] for h in hist]
    if len(closes) >= 20:
        ma = pd.Series(closes).rolling(20).mean()
        fig.add_trace(go.Scatter(x=hx, y=ma, mode="lines", name="MA20",
                                 line=dict(color="#1c7ed6", width=1)))

    colors = {"bull": "#e03131", "base": "#1c7ed6", "bear": "#2f9e44"}
    last_date = hx[-1]
    for sc in scenarios:
        path = sc.get("path", [])
        if not path:
            continue
        name = sc.get("name", "base")
        px = [last_date] + [p["date"] for p in path]
        # 衔接线（历史最后收盘 → 预测第一根 open）
        po = [hist[-1]["close"]] + [p["open"] for p in path]
        ph = [hist[-1]["close"]] + [p["high"] for p in path]
        pl = [hist[-1]["close"]] + [p["low"] for p in path]
        pc = [hist[-1]["close"]] + [p["close"] for p in path]
        color = colors.get(name, "#1c7ed6")
        fig.add_trace(go.Candlestick(
            x=px, open=po, high=ph, low=pl, close=pc,
            name=f"预测-{name}({sc.get('weight', 0):.0%})",
            increasing_line_color=color, decreasing_line_color=color,
            increasing_fillcolor="rgba(0,0,0,0)",
            decreasing_fillcolor="rgba(0,0,0,0)",
            line=dict(width=1), opacity=0.55))

    # 关键价位
    lv = data.get("key_levels") or {}
    for s_ in lv.get("support", []):
        fig.add_hline(y=float(s_), line=dict(color="#2f9e44", dash="dot", width=1))
    for r_ in lv.get("resistance", []):
        fig.add_hline(y=float(r_), line=dict(color="#e03131", dash="dot", width=1))

    fig.add_vline(x=last_date, line=dict(color="#666", dash="dash", width=1))
    fig.add_annotation(x=last_date, y=1.02, yref="paper", text="← 预测模拟 →",
                       showarrow=False, font=dict(size=11, color="#666"))
    fig.update_layout(
        title=f"{data.get('symbol', '')} 未来 K 线路径（{data.get('base_direction', '')}"
              f" {data.get('expected_return_pct', 0):+.1f}%，{data.get('mode', '')}）",
        height=520, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=1.10))
    return fig
