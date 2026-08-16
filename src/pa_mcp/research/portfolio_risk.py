# [AI:BEGIN]
# PA_MCP - Research: 持仓风险面板
#
# 持仓管理高价值整合视图（非必要不造轮子，全复用）：
#   1. 持仓实时盈亏（最新收盘 vs 成本）
#   2. 批量预测（复用 PredictionService，方向/概率/期望）
#   3. 集中度：单票占比 + 行业分布 + HHI（赫芬达尔指数）
#   4. 风险评分（0-100）：集中度 + 预测分歧度 + 单票波动
#
# 输出：每票明细 + 组合级风险指标 + 风险提示（可追溯）。
# [AI:END]

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioRiskDashboard:
    """持仓风险面板（盈亏 × 预测 × 集中度 × 评分）。"""

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._store_path = store_path

    def _store(self):
        from pa_mcp.config import get_settings
        from pa_mcp.data.store import DuckDBStore
        path = self._store_path or get_settings().database.path
        store = DuckDBStore(path)
        store.connect()
        return store

    async def analyze(self, use_llm: bool = False,
                      use_market_bias: bool = True) -> dict[str, Any]:
        """持仓风险分析。

        Args:
            use_llm: 预测是否用 LLM（默认 False 控制成本）
            use_market_bias: 是否注入指数结构方向（偏空上调风险，
                偏多下调；库中无指数数据时跳过）
        """
        store = self._store()
        try:
            if not store.table_exists("portfolio"):
                return {"error": "无持仓（先 portfolio_add 添加）"}
            df = store.query_df("SELECT * FROM portfolio", [])
            if df.empty:
                return {"error": "无持仓（先 portfolio_add 添加）"}

            holdings = []
            total_value = 0.0
            total_cost = 0.0
            sector_weights: dict[str, float] = {}
            for _, r in df.iterrows():
                sym = str(r["symbol"])
                cost = float(r["cost"])
                shares = float(r["shares"])
                # 最新收盘
                price = None
                try:
                    q = store.query_df(
                        "SELECT close FROM kline_daily WHERE symbol = ? "
                        "ORDER BY date DESC LIMIT 1", [sym])
                    if not q.empty:
                        price = float(q.iloc[0]["close"])
                except Exception:
                    pass
                if price is None:
                    continue
                value = price * shares
                cost_val = cost * shares
                total_value += value
                total_cost += cost_val
                sector = ""
                try:
                    sb = store.query_df(
                        "SELECT sector FROM stock_basic WHERE symbol = ?",
                        [sym])
                    if not sb.empty:
                        sector = str(sb.iloc[0]["sector"] or "")
                except Exception:
                    pass
                sector_weights[sector or "未知"] = sector_weights.get(
                    sector or "未知", 0.0) + value

                # 预测（analyze 已异步 → 直接 await，不嵌套 asyncio.run）
                pred = None
                try:
                    from pa_mcp.agent.prediction import PredictionService
                    kdf = store.query_df(
                        "SELECT * FROM kline_daily WHERE symbol = ? "
                        "ORDER BY date DESC LIMIT 160", [sym])
                    if not kdf.empty:
                        result = await PredictionService(
                            self._store_path).predict(
                            sym, kdf, horizon="5d", use_llm=use_llm)
                        p = result.to_dict()
                        pred = {
                            "direction": p["direction"],
                            "probability": p["probability"],
                            "expected_return_pct": p["expected_return_pct"],
                            "range": p["expected_range_pct"],
                        }
                except Exception as e:  # noqa: BLE001
                    logger.debug("prediction failed in risk dashboard",
                                 symbol=sym, error=str(e))

                # 多周期共振（best-effort，独立拉行情）
                resonance = None
                try:
                    from pa_mcp.research.resonance import ResonanceAnalyzer
                    rdf = store.query_df(
                        "SELECT * FROM kline_daily WHERE symbol = ? "
                        "ORDER BY date DESC LIMIT 160", [sym])
                    if not rdf.empty:
                        res = await ResonanceAnalyzer(
                            self._store_path).analyze(sym, kline_df=rdf)
                        if "error" not in res:
                            resonance = {
                                "signal": res["signal"],
                                "strength": res["strength"],
                                "resonance": res["resonance"],
                            }
                except Exception:
                    pass

                # 综合信号（best-effort，独立拉行情）
                consensus = None
                try:
                    from pa_mcp.research.consensus import ConsensusAnalyzer
                    cdf = store.query_df(
                        "SELECT * FROM kline_daily WHERE symbol = ? "
                        "ORDER BY date DESC LIMIT 200", [sym])
                    if not cdf.empty:
                        con = await ConsensusAnalyzer(
                            self._store_path).analyze(sym, kline_df=cdf)
                        if "error" not in con:
                            consensus = {
                                "signal": con["signal"],
                                "strength": con["strength"],
                                "level": con["level"],
                                "agreement": con["agreement"],
                            }
                except Exception:
                    pass

                holdings.append({
                    "symbol": sym,
                    "cost": cost, "shares": shares,
                    "price": price,
                    "value": round(value, 2),
                    "pnl_pct": round((price / cost - 1) * 100, 2)
                    if cost > 0 else 0.0,
                    "weight_pct": 0.0,  # 最后统一算
                    "sector": sector,
                    "prediction": pred,
                    "resonance": resonance,
                    "consensus": consensus,
                })

            if not holdings or total_value <= 0:
                return {"error": "持仓均无最新行情"}

            for h in holdings:
                h["weight_pct"] = round(h["value"] / total_value * 100, 2)

            # 集中度：HHI（权重平方和，1=完全集中）
            hhi = sum((w / 100) ** 2 for w in
                      [h["weight_pct"] for h in holdings])
            top_weight = max(h["weight_pct"] for h in holdings)
            # 行业集中
            sector_weights = {s: round(v / total_value * 100, 2)
                              for s, v in sector_weights.items()}
            top_sector = max(sector_weights.items(),
                             key=lambda x: x[1]) if sector_weights else ("", 0.0)

            # 预测分歧（看涨 vs 看跌占比）
            n_pred = sum(1 for h in holdings if h["prediction"])
            n_up = sum(1 for h in holdings
                       if h["prediction"] and h["prediction"]["direction"] == "up")
            n_down = sum(1 for h in holdings
                         if h["prediction"] and h["prediction"]["direction"] == "down")

            # 风险评分（0=低风险 100=高风险）
            score = 0.0
            score += min(30, hhi * 300)                    # 集中度（HHI）
            score += min(25, max(0, top_weight - 20) * 2)  # 单票 >20%
            score += min(20, max(0, top_sector[1] - 40))   # 行业 >40%
            if n_pred > 0:
                score += min(15, n_down / n_pred * 30)     # 看跌占比
            score += min(10, max(0, 100 - len(holdings) * 10))  # 数量少
            score = round(min(100, max(0, score)), 1)

            # 市场结构方向注入（指数缠论：偏空逆风上调，偏多顺风下调）
            market_bias = None
            bias_adjust = 0.0
            if use_market_bias:
                try:
                    from pa_mcp.research.market_structure import (
                        MarketStructureAnalyzer)
                    ms = await MarketStructureAnalyzer(
                        self._store_path).analyze(use_network=False)
                    if ms["index"]["rows"] > 0:
                        market_bias = ms["joint"]["bias"]
                        if market_bias == "偏空":
                            bias_adjust = 10
                        elif market_bias == "偏多":
                            bias_adjust = -5
                        score = round(min(100, max(0, score + bias_adjust)), 1)
                except Exception:
                    pass

            return {
                "total_value": round(total_value, 2),
                "total_cost": round(total_cost, 2),
                "total_pnl_pct": round(
                    (total_value / total_cost - 1) * 100, 2)
                if total_cost > 0 else 0.0,
                "n_holdings": len(holdings),
                "holdings": holdings,
                "concentration": {
                    "hhi": round(hhi, 4),
                    "top_weight_pct": top_weight,
                    "sector_weights": sector_weights,
                    "top_sector": list(top_sector),
                },
                "prediction_summary": {
                    "n_predicted": n_pred,
                    "n_up": n_up, "n_down": n_down,
                },
                "risk_score": score,
                "risk_level": ("低" if score < 30 else "中" if score < 60
                               else "高"),
                "risk_notes": self._risk_notes(score, holdings,
                                               top_weight, top_sector,
                                               market_bias, bias_adjust),
                "market_bias": market_bias,
                "market_bias_adjust": bias_adjust,
            }
        finally:
            store.close()

    @staticmethod
    def _risk_notes(score: float, holdings: list, top_weight: float,
                    top_sector: tuple, market_bias: Optional[str] = None,
                    bias_adjust: float = 0.0) -> list[str]:
        notes = []
        if market_bias == "偏空":
            notes.append(f"指数偏空（缠论结构）→ 风险上调 {bias_adjust:+.0f} 分")
        elif market_bias == "偏多":
            notes.append(f"指数偏多（缠论结构）→ 风险下调 {bias_adjust:+.0f} 分")
        if top_weight > 30:
            notes.append(f"单票 {top_weight:.0f}% 集中度过高（>30%）")
        if top_sector[1] > 50:
            notes.append(f"行业 {top_sector[0]} 占比 {top_sector[1]:.0f}% 过重")
        down = [h for h in holdings
                if h["prediction"] and h["prediction"]["direction"] == "down"]
        if down:
            notes.append("预测看跌持仓：" + "、".join(h["symbol"] for h in down))
        res_down = [h for h in holdings
                    if h.get("resonance")
                    and h["resonance"]["signal"] == "down"
                    and h["resonance"]["strength"] >= 0.7]
        if res_down:
            notes.append("**强共振看跌持仓**（三周期一致，优先减仓）："
                         + "、".join(h["symbol"] for h in res_down))
        con_down = [h for h in holdings
                    if h.get("consensus")
                    and h["consensus"]["signal"] == "down"
                    and h["consensus"]["strength"] >= 0.6]
        if con_down:
            notes.append("**⚠️ 综合信号看跌持仓**（5 源投票一致，最优先减仓）："
                         + "、".join(h["symbol"] for h in con_down))
        if len(holdings) < 3:
            notes.append("持仓数量过少（<3 只），非系统性风险高")
        if score >= 60:
            notes.append("综合风险偏高，建议分散")
        return notes or ["持仓结构较均衡"]


_dashboard: Optional[PortfolioRiskDashboard] = None


def get_risk_dashboard() -> PortfolioRiskDashboard:
    """单例获取风险面板。"""
    global _dashboard
    if _dashboard is None:
        _dashboard = PortfolioRiskDashboard()
    return _dashboard


def format_risk_dashboard(result: dict[str, Any]) -> str:
    """风险面板 → markdown。"""
    if "error" in result:
        return f"持仓风险面板不可用：{result['error']}"
    lines = [
        f"## 🛡️ 持仓风险面板",
        f"**总市值 {result['total_value']:,.0f} 元 | 成本 {result['total_cost']:,.0f} 元 | "
        f"盈亏 {result['total_pnl_pct']:+.1f}%**",
        f"**风险评分：{result['risk_score']}（{result['risk_level']}）**"
        + (f"（指数{'偏空' if result.get('market_bias') == '偏空' else '偏多'}"
           f"调整 {result.get('market_bias_adjust', 0):+.0f}）"
           if result.get("market_bias") else ""),
        "",
        "### 持仓明细",
        "| 代码 | 成本 | 现价 | 盈亏% | 占比% | 板块 | 预测(5d) | 共振 | 综合信号 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    dir_zh = {"up": "📈", "down": "📉", "sideways": "➡️"}
    for h in result["holdings"]:
        p = h["prediction"]
        pred_txt = (f"{dir_zh.get(p['direction'], '')} {p['probability']:.0%}"
                    f"({p['expected_return_pct']:+.1f}%)"
                    if p else "—")
        res = h.get("resonance")
        res_txt = ""
        if res:
            res_txt = (f"{dir_zh.get(res['signal'], '')} {res['strength']:.0%}"
                       if res["strength"] >= 0.7 else f"分歧 {res['strength']:.0%}")
        con = h.get("consensus")
        con_txt = ""
        if con:
            con_txt = (f"{dir_zh.get(con['signal'], '')} {con['strength']:.0%}"
                       if con["strength"] >= 0.6 else f"分歧 {con['strength']:.0%}")
        lines.append(
            f"| {h['symbol']} | {h['cost']:.2f} | {h['price']:.2f} | "
            f"{h['pnl_pct']:+.1f}% | {h['weight_pct']:.1f}% | "
            f"{h['sector'] or '—'} | {pred_txt} | {res_txt or '—'} | "
            f"{con_txt or '—'} |")
    c = result["concentration"]
    lines.extend([
        "",
        "### 集中度",
        f"- HHI：{c['hhi']:.3f}（1=完全集中）| 单票最大 {c['top_weight_pct']:.1f}%",
        f"- 行业分布：" + "、".join(
            f"{s} {w:.0f}%" for s, w in
            sorted(c["sector_weights"].items(),
                   key=lambda x: -x[1])[:5]),
    ])
    ps = result["prediction_summary"]
    if ps["n_predicted"]:
        lines.append(f"- 预测：{ps['n_up']} 看涨 / {ps['n_down']} 看跌"
                     f"（{ps['n_predicted']} 只已预测）")
    lines.append("\n### 风险提示")
    for n in result["risk_notes"]:
        lines.append(f"- {n}")
    lines.append("\n*研究参考，非投资建议。*")
    return "\n".join(lines)
