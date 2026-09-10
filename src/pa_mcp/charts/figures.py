"""Plotly Figure 构造器（纯函数，不碰 DB / IO，便于测试与复用）。

配色采用中国惯例：涨红 #ef5350 / 跌绿 #26a69a；plotly_white 模板；
K 线含 MA + 成交量副图；资金流叠加主力/超大/大/中/小单；多股对比首日=100
归一化；板块与情绪各自独立成图。
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 中国惯例配色（与 pa_mcp.ui.gradio_app 保持一致）
RED = "#ef5350"   # 涨
GREEN = "#26a69a" # 跌
GRAY = "#90a4ae"  # 量 / 中性

# 标准化趋势线条色
MA_COLORS = {
    5: "#ffb300",
    10: "#fb8c00",
    20: "#5c6bc0",
    30: "#7e57c2",
    60: "#66bb6a",
}

# 资金流多档配色（沿用 matplotlib/gradio 默认红绿，再加蓝/橙区分超大/大单）
FUND_FLOW_COLORS = {
    "main_net_inflow": RED,
    "super_large_net_inflow": "#c2185b",
    "large_net_inflow": "#ff7043",
    "mid_net_inflow": "#42a5f5",
    "small_net_inflow": "#26c6da",
}


def _label(symbol: str, name: Optional[str]) -> str:
    return f"{symbol} {name}" if name else symbol


def kline_figure(
    df: pd.DataFrame,
    symbol: str,
    name: Optional[str] = None,
    ma_list: Iterable[int] = (5, 10, 20, 60),
    with_prediction: Optional[Mapping[str, list[dict]]] = None,
) -> go.Figure:
    """K线图：蜡烛 + MA 叠加 + 成交量副图；可叠加预测路径。

    Args:
        df: 含 date/open/high/low/close/volume 的行情 DataFrame
        symbol, name: 标题
        ma_list: 要叠加的均线窗口
        with_prediction: 预测路径（可选），格式 {"bull": [{"date","open","high","low","close"},...],
            "base": [...], "bear": [...]}；用点状虚线叠加在主图
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="K线",
        increasing_line_color=RED, decreasing_line_color=GREEN,
    ), row=1, col=1)

    for window in ma_list:
        color = MA_COLORS.get(window, "#888")
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["close"].rolling(window).mean(),
            name=f"MA{window}",
            line=dict(width=1, color=color),
        ), row=1, col=1)

    if with_prediction:
        for key, color, dash in (("bull", RED, "solid"),
                                  ("base", "#1c7ed6", "dash"),
                                  ("bear", GREEN, "dot")):
            series = with_prediction.get(key) or []
            if not series:
                continue
            pdf = pd.DataFrame(series)
            pdf["date"] = pd.to_datetime(pdf["date"]).dt.strftime("%Y-%m-%d")
            fig.add_trace(go.Scatter(
                x=pdf["date"], y=pdf["close"], mode="lines+markers",
                name=f"预测-{key}", line=dict(color=color, dash=dash, width=1.6),
                marker=dict(size=4), opacity=0.85,
            ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df["date"], y=df["volume"], name="成交量",
        marker_color=GRAY,
    ), row=2, col=1)

    fig.update_layout(
        title=f"{_label(symbol, name)} 日K线",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=560, legend=dict(orientation="h", y=1.02),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(type="category", nticks=10)
    return fig


def fund_flow_figure(df: pd.DataFrame, symbol: str, name: Optional[str] = None) -> go.Figure:
    """资金流图：主力 / 超大 / 大 / 中 / 小单 净流入柱状（亿元）。

    df 必含 trade_date + 上述 5 列；按交易日期排序。
    """
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%m-%d")

    fig = go.Figure()
    for col in ("main_net_inflow", "super_large_net_inflow", "large_net_inflow",
                "mid_net_inflow", "small_net_inflow"):
        if col not in df.columns:
            continue
        vals = df[col].fillna(0) / 1e8  # 元 → 亿元
        fig.add_trace(go.Bar(
            x=df["date"], y=vals, name=col.replace("_net_inflow", ""),
            marker_color=FUND_FLOW_COLORS[col],
        ))

    fig.update_layout(
        title=f"{_label(symbol, name)} 资金流（亿元）",
        template="plotly_white",
        barmode="group", height=320,
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def compare_figure(
    series: Mapping[str, pd.DataFrame],
    name_map: Optional[Mapping[str, str]] = None,
    days: int = 120,
    title: Optional[str] = None,
) -> go.Figure:
    """多股归一化对比（首日=100）。

    Args:
        series: {symbol: DataFrame(date,close,...)}，各 df 长度/日期不必一致，
                会按各自首日对齐
        name_map: {symbol: 名称}，可选
        days: 标题里标注的天数
    """
    name_map = name_map or {}
    fig = go.Figure()
    palette = ["#ef5350", "#1c7ed6", "#2b8a3e", "#f08c00", "#9c27b0"]
    for i, (sym, df) in enumerate(series.items()):
        if df is None or df.empty or "close" not in df.columns:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        base = float(df["close"].iloc[0])
        if base <= 0:
            continue
        norm = df["close"] / base * 100.0
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"),
            y=norm,
            name=f"{sym} {name_map.get(sym, '')}".strip(),
            line=dict(width=1.8, color=palette[i % len(palette)]),
            mode="lines",
        ))

    fig.update_layout(
        title=title or f"多股归一化对比（首日=100，近 {days} 交易日）",
        template="plotly_white",
        height=420, legend=dict(orientation="h", y=1.02),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def sector_figure(df: pd.DataFrame, top_n: int = 20,
                  as_of: str | None = None) -> go.Figure:
    """板块强度图：当日涨幅/资金流横向柱状。

    df 必含 sector_code/name + 至少一个排序指标（pct_change / main_net_inflow）。

    去重：sector_daily 里合成兜底板块（sector_code 以 SYN_ 开头）会与真实
    行业板块同名，直接出图会出现重复类目、柱子叠在一起并错位 —— 因此按
    name 去重，优先保留真实代码（非 SYN_）那一行。
    """
    if df.empty:
        return _empty("板块数据为空")
    metric = "pct_change" if "pct_change" in df.columns else "main_net_inflow"
    df = df.dropna(subset=[metric]).copy()

    if "name" in df.columns:
        if "sector_code" in df.columns:
            df["_syn"] = df["sector_code"].astype(str).str.startswith("SYN_")
            df = df.sort_values(["_syn", "sector_code"]).drop_duplicates(
                subset=["name"], keep="first").drop(columns=["_syn"])
        else:
            df = df.drop_duplicates(subset=["name"], keep="first")

    df = df.sort_values(metric).tail(top_n)
    colors = [RED if v >= 0 else GREEN for v in df[metric]]
    fig = go.Figure(go.Bar(
        x=df[metric], y=df.get("name", df.get("sector_code", df.index)).astype(str),
        orientation="h", marker_color=colors,
        text=[f"{v:+.2f}" for v in df[metric]], textposition="outside",
    ))
    head = f"板块强度（{as_of}，" if as_of else "板块强度（"
    fig.update_layout(
        title=f"{head}Top {min(top_n, len(df))}，按 {metric}）",
        template="plotly_white", height=420,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def sentiment_figure(df: pd.DataFrame) -> go.Figure:
    """情绪周期图：涨停/跌停家数 + 连板高度 + 情绪分位折线。

    df 必含 date + limit_up_count + max_board_height + sentiment_score（可选）。
    """
    if df.empty:
        return _empty("情绪数据为空")
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%m-%d")

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=df["date"], y=df["limit_up_count"].fillna(0),
        name="涨停家数", marker_color=RED, opacity=0.6,
    ), secondary_y=False)
    if "limit_down_count" in df.columns:
        fig.add_trace(go.Bar(
            x=df["date"], y=df["limit_down_count"].fillna(0),
            name="跌停家数", marker_color=GREEN, opacity=0.6,
        ), secondary_y=False)
    if "max_board_height" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["max_board_height"].fillna(0),
            name="连板高度", mode="lines+markers",
            line=dict(color="#1c7ed6", width=1.8),
        ), secondary_y=True)
    if "sentiment_score" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["sentiment_score"].fillna(0),
            name="情绪评分", mode="lines+markers",
            line=dict(color="#f08c00", width=1.4, dash="dot"),
        ), secondary_y=True)

    fig.update_layout(
        title="游资情绪周期", template="plotly_white", barmode="group",
        height=360, legend=dict(orientation="h", y=1.1),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_yaxes(title_text="家数", secondary_y=False)
    fig.update_yaxes(title_text="高度/评分", secondary_y=True)
    return fig


def _empty(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title, template="plotly_white", height=200,
        annotations=[dict(text="暂无数据", x=0.5, y=0.5,
                         xref="paper", yref="paper", showarrow=False,
                         font=dict(size=16, color="#888"))],
    )
    return fig