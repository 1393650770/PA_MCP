# [AI:BEGIN]
# PA_MCP - Research: 选股多 Agent 辩论（候选逐一裁定）
#
# 对选股候选（CANSLIM/因子/价值动量等任意来源）逐一运行
# deep_analyze(debate=True)：Bull 论证 → Bear 反驳 → 3 位投资大师
# （格雷厄姆价值/索罗斯反身性/利弗莫尔趋势）并行判断 → 置信加权合议，
# 输出每票裁定 + 加权排名——「用辩论验证选股，而不是只看分数」。
#
# 复用：orchestrator.deep_analyze（含 _run_debate 全流程与 RiskGuard
# 仓位上限）；无 LLM 时 orchestrator 自动降级规则分析，本模块诚实标注
# 「未配置 LLM，无多 Agent 辩论」。
# [AI:END]

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

MAX_PICKS = 5          # 最多辩论的候选数（每票 ≈90s + 5 次 LLM 调用）
KLINE_LIMIT = 250


def _store():
    from pa_mcp.config import get_settings
    from pa_mcp.data.store import DuckDBStore
    store = DuckDBStore(get_settings().database.path)
    store.connect()
    return store


def _load_kline(store, symbol: str) -> Optional[Any]:
    try:
        df = store.query_df(
            "SELECT * FROM kline_daily WHERE symbol = ? "
            "ORDER BY date DESC LIMIT ?", [symbol, KLINE_LIMIT])
        if df is None or df.empty:
            return None
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:  # noqa: BLE001
        logger.debug("kline load failed", symbol=symbol, error=str(e))
        return None


async def debate_picks(symbols: list[str], top_n: int = 3,
                       store=None) -> dict[str, Any]:
    """对选股候选逐一多 Agent 辩论裁定 + 置信加权排名。

    Args:
        symbols: 候选代码列表（来自 CANSLIM/因子/价值动量等选股工具）
        top_n: 最多辩论几只（默认 3，控制成本；≤ MAX_PICKS）
        store: 可注入数据仓库（测试隔离）

    Returns:
        picks: 每票 {symbol, name, direction, score, verdict(或 None),
                     max_position, mode}
        debate_mode: "llm"（真辩论）| "deterministic"（无 LLM 降级）
        report: markdown
    """
    syms = [s.strip() for s in symbols if s.strip()][:min(top_n, MAX_PICKS)]
    if not syms:
        return {"error": "候选清单为空（先跑选股工具拿候选）"}

    own = store is None
    store = store or _store()
    try:
        from pa_mcp.agent.orchestrator import get_orchestrator
        from pa_mcp.data.symbols import get_stock_name

        orch = get_orchestrator()
        picks: list[dict[str, Any]] = []
        for sym in syms:
            df = _load_kline(store, sym)
            if df is None:
                picks.append({"symbol": sym, "error": "行情数据不足"})
                continue
            try:
                result = await orch.deep_analyze(sym, df, debate=True)
                verdict = result.master_verdict
                picks.append({
                    "symbol": sym,
                    "name": get_stock_name(sym),
                    "direction": result.direction,
                    "score": result.overall_strength_score,
                    "mode": result.mode,
                    "max_position_pct": result.suggested_max_position_pct,
                    "verdict": verdict,
                    "evidence": (result.key_evidence or [])[:3],
                    "risks": (result.key_risks or [])[:3],
                })
            except Exception as e:  # noqa: BLE001 单票失败不拖垮
                logger.debug("debate pick failed", symbol=sym, error=str(e))
                picks.append({"symbol": sym, "error": f"分析失败：{str(e)[:60]}"})

        llm_mode = any(p.get("verdict") for p in picks)
        mode = "llm" if llm_mode else "deterministic"

        # 排名：有裁定的按 master verdict 分数降序，无裁定按分析分数降序
        def _key(p: dict) -> float:
            if p.get("verdict"):
                try:
                    return float(p["verdict"]["final_strength_score"])
                except Exception:
                    return 0.0
            return float(p.get("score") or 0)

        ranked = sorted(
            [p for p in picks if "error" not in p],
            key=_key, reverse=True)
        errors = [p for p in picks if "error" in p]

        report = format_report(ranked, errors, mode)
        return {"picks": ranked, "errors": errors,
                "debate_mode": mode, "report": report}
    finally:
        if own:
            store.close()


def format_report(ranked: list[dict], errors: list[dict],
                  mode: str) -> str:
    """辩论结果 → markdown。"""
    lines = ["## ⚔️ 选股多 Agent 辩论裁定",
             ""]
    if mode == "llm":
        lines.append("*模式：LLM 多 Agent 辩论（Bull 论证 → Bear 反驳 → "
                     "格雷厄姆/索罗斯/利弗莫尔 3 大师合议）*")
    else:
        lines.append("> ⚠️ 未配置 LLM：无多 Agent 辩论，以下为确定性规则分析"
                     "（配置 config/llm_config.json 后启用辩论）")
    lines.append("")
    if ranked:
        lines.append("| 排名 | 代码 | 名称 | 裁定方向 | 大师团分数 | 建议仓位 | 关键证据 |")
        lines.append("|---|---|---|---|---|---|---|")
        for i, p in enumerate(ranked, 1):
            v = p.get("verdict")
            if v:
                direction = v.get("final_direction", "neutral")
                score = f"{v.get('final_strength_score', 0):.0f}"
                pos = f"{v.get('suggested_max_position_pct', 0):.0f}%"
                reason = v.get("verdict_reason", "")
                evidence = "；".join(p.get("evidence", []) or [])[:60]
                line = (f"| {i} | {p['symbol']} | {p.get('name', '')} | "
                        f"{direction} | {score} | {pos} | {evidence} |")
                if reason:
                    line += f"\n  - *{reason}*"
                lines.append(line)
            else:
                lines.append(
                    f"| {i} | {p['symbol']} | {p.get('name', '')} | "
                    f"{p.get('direction', '—')} | {p.get('score', '—')} | "
                    f"{p.get('max_position_pct', 0):.0f}% | "
                    f"{'；'.join(p.get('evidence', []) or [])[:60]} |")
        lines.append("")
        lines.append("**裁定逻辑**：大师团置信加权投票（分数越极端权重越高），"
                     "建议仓位 ≤20% RiskGuard 硬上限。")
    else:
        lines.append("（无候选通过分析）")
    if errors:
        lines.append("\n**跳过**：" + "、".join(
            f"{e['symbol']}（{e.get('error', '')[:40]}）" for e in errors))
    lines.append("\n---")
    lines.append("*辩论基于规则与数据，研究参考，非投资建议。*")
    return "\n".join(lines)
