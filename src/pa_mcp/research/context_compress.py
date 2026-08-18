# [AI:BEGIN]
# PA_MCP - Research: 上下文压缩（token 成本控制 + 超限防护）
#
# 两类压缩（借鉴开源 Agent 的上下文管理实践）：
#   1. compress_kline：K 线两级压缩——近期 N 根逐根明细 + 更早部分
#      周聚合统计（长历史塞 prompt 时 token 可控），并保留整体趋势
#      摘要（区间/均量/涨跌幅）。比 extract_features 更通用（可控长度）。
#   2. compress_text：长文本 LLM 精炼压缩（保留关键数字/结论/风险，
#      输出 ≤ max_chars）；无 LLM 时结构保头尾截断。
#
# 应用：长报告/多轮证据/长历史 K 线注入 LLM 前调用，防超限降成本。
# [AI:END]

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def compress_kline(df: pd.DataFrame, detail_bars: int = 30,
                   summary_weeks: int = 26) -> str:
    """K 线两级压缩为文本（近期明细 + 早期周聚合 + 整体统计）。

    Args:
        df: 日线（date/close/high/low/volume 至少）
        detail_bars: 近期逐根明细根数（默认 30）
        summary_weeks: 早期部分聚合周数上限（默认 26）

    Returns:
        压缩文本：整体统计 + 近期明细 + 早期周聚合。
    """
    if df is None or df.empty:
        return "无K线数据"
    d = df.sort_values("date").reset_index(drop=True)
    n = len(d)
    last = float(d["close"].iloc[-1])

    # 整体统计
    hi = float(d["high"].max())
    lo = float(d["low"].min())
    chg = (last / float(d["close"].iloc[0]) - 1) * 100
    vol_avg = float(d["volume"].mean()) if "volume" in d else 0.0
    lines = [f"K线{n}根：收 {last:.2f}（区间 {lo:.2f}-{hi:.2f}，"
             f"区间涨跌 {chg:+.1f}%，日均量 {vol_avg:,.0f}）"]

    # 近期明细
    recent = d.tail(detail_bars)
    if len(recent) > 0:
        rows = []
        for _, r in recent.iterrows():
            dt = str(r["date"])[:10]
            c = float(r["close"])
            pct = ""
            if "pct_change" in r and pd.notna(r["pct_change"]):
                pct = f"({float(r['pct_change']):+.1f}%)"
            rows.append(f"{dt}:{c:.2f}{pct}")
        lines.append(f"近{len(recent)}日：{' '.join(rows)}")

    # 早期周聚合
    early = d.iloc[:-detail_bars] if n > detail_bars else pd.DataFrame()
    if not early.empty:
        early = early.copy()
        early["week"] = pd.to_datetime(early["date"]).dt.to_period("W")
        agg = early.groupby("week").agg(
            o=("open", "first"), c=("close", "last"),
            h=("high", "max"), l=("low", "min"),
            v=("volume", "mean")).tail(summary_weeks)
        weeks = []
        for _, w in agg.iterrows():
            weeks.append(f"{w['o']:.1f}~{w['c']:.1f}({w['h']:.1f}/{w['l']:.1f})")
        lines.append(f"早期周聚合({len(agg)}周)：{' '.join(weeks[:summary_weeks])}")
    return "\n".join(lines)


_COMPRESS_PROMPT = """压缩以下内容为 ≤{max_chars} 字符的精炼摘要，要求：
1. 保留全部关键数字、结论、风险提示（这些最值钱）
2. 删除重复/修饰/过程性描述
3. 输出纯文本（无 markdown 表格），原样保留关键字段名

【内容】
{text}

【压缩后】"""


def _truncate_preserve(text: str, max_chars: int) -> str:
    """无 LLM：保头尾截断（头 60% + 尾 40%），中间省略标记。"""
    if len(text) <= max_chars:
        return text
    marker = "\n…[已压缩，省略中间]…\n"
    head = int(max_chars * 0.6)
    tail = max_chars - head - len(marker)
    return text[:head] + marker + text[-tail:]


async def compress_text(text: str, max_chars: int = 2000,
                        use_llm: bool = True) -> dict[str, Any]:
    """长文本压缩（LLM 精炼，无 LLM 保头尾截断）。

    Args:
        text: 待压缩内容
        max_chars: 目标长度（默认 2000）
        use_llm: 是否尝试 LLM（无配置自动降级截断）

    Returns:
        {compressed, original_len, compressed_len, mode(llm/truncate),
         ratio}
    """
    text = (text or "").strip()
    if not text:
        return {"compressed": "", "original_len": 0, "compressed_len": 0,
                "mode": "truncate", "ratio": 0.0}
    max_chars = max(200, min(max_chars, 8000))
    if len(text) <= max_chars:
        return {"compressed": text, "original_len": len(text),
                "compressed_len": len(text), "mode": "none", "ratio": 1.0}

    mode = "truncate"
    compressed = _truncate_preserve(text, max_chars)
    if use_llm:
        try:
            from pa_mcp.agent.llm_factory import ensure_llm_adapter
            from pa_mcp.agent.llm_port import LLMCallParams
            adapter = ensure_llm_adapter()
            if adapter is not None:
                params = LLMCallParams(
                    system_prompt="你是信息压缩助手，只输出压缩后的文本，"
                                  "不输出任何其他内容。",
                    user_prompt=_COMPRESS_PROMPT.format(
                        max_chars=max_chars, text=text[:20000]),
                    mode="fast", max_tokens=3000,
                )
                resp = await adapter.chat(params)
                out = (resp.content or "").strip()
                if out:
                    compressed = out[:max_chars]
                    mode = "llm"
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM 压缩失败，降级截断: %s", str(e)[:80])

    return {
        "compressed": compressed,
        "original_len": len(text),
        "compressed_len": len(compressed),
        "mode": mode,
        "ratio": round(len(compressed) / len(text), 3),
    }
