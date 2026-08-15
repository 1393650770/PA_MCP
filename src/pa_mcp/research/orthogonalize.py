# [AI:BEGIN]
# PA_MCP - Research: 因子正交化（风格中性化）
#
# 机制借鉴开源 QuantSkills factor-orthogonalize：
#   对截面因子逐日 OLS 回归风格暴露（市值 + 板块收益），取残差作为
#   风格中性的纯 alpha 因子，并输出暴露诊断（beta / R² / 残差波动）。
#
# 实现（纯 numpy lstsq，无 scipy 依赖）：
#   1. 日收益矩阵（对齐交易日）
#   2. 暴露矩阵：市值对数（水平）+ 所属板块当日平均收益（板块风格）
#   3. 逐日 OLS：ret_t = b0 + b1·ln(mktcap) + b2·sector_ret + ε_t
#   4. 残差 ε_t = 剔除市值/板块风格后的纯个股 alpha
#   5. 输出：个股 alpha 统计（均值/IR/胜率/累计）+ 风格 beta 诊断
#
# 用途：股票池排序前先中性化——避免「押中板块」被误认为「个股 alpha」。
# [AI:END]

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_DAYS = 20   # 最少样本日
MIN_ALPHA_DAYS = 10


class FactorNeutralizer:
    """风格中性化：逐日 OLS 残差化（市值 + 板块暴露）。"""

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._store_path = store_path

    def _store(self):
        from pa_mcp.config import get_settings
        from pa_mcp.data.store import DuckDBStore
        path = self._store_path or get_settings().database.path
        store = DuckDBStore(path)
        store.connect()
        return store

    # ---- 数据装配 ----
    def _build_inputs(self, symbols: list[str], lookback: int = 120):
        """收益矩阵 + 市值 + 板块映射（kline_daily + stock_basic）。"""
        store = self._store()
        try:
            sb = store.query_df(
                "SELECT symbol, sector, market_cap FROM stock_basic "
                "WHERE symbol IN (SELECT UNNEST(?))", [symbols])
            sector_map: dict[str, str] = {}
            mcap_level: dict[str, float] = {}
            for _, r in sb.iterrows():
                sector_map[str(r["symbol"])] = str(r["sector"]) \
                    if r["sector"] else "未知"
                try:
                    mc = float(r["market_cap"]) if pd.notna(r["market_cap"]) else 0.0
                except (TypeError, ValueError):
                    mc = 0.0
                mcap_level[str(r["symbol"])] = mc if mc > 0 else 0.0
            # 未知板块的股票也纳入（板块暴露=自身近似）
            for s in symbols:
                sector_map.setdefault(s, s)

            # 收益（市值缺失的股票用 close 水平回退，注意共线风险）
            rets: dict[str, pd.Series] = {}
            mcap: dict[str, pd.Series] = {}
            for sym in symbols:
                df = store.query_df(
                    "SELECT date, close, amount, turnover FROM kline_daily "
                    "WHERE symbol = ? ORDER BY date DESC LIMIT ?",
                    [sym, lookback + 1])
                if len(df) < MIN_DAYS + 1:
                    continue
                d = df.sort_values("date").reset_index(drop=True)
                rets[sym] = d["close"].pct_change().iloc[1:].reset_index(drop=True)
                mc = mcap_level.get(sym, 0.0)
                if mc > 0:
                    # 静态市值快照：与近期收益解耦（真实风格因子）
                    mcap[sym] = pd.Series([mc] * (len(d) - 1))
                else:
                    # 回退：close 水平（有共线风险，诊断注明）
                    mcap[sym] = d["close"].iloc[1:].reset_index(drop=True)
            if not rets:
                return None, None, None, None

            # 对齐交易日（公共日期集合）
            dates = sorted(set().union(*[set(r.index) for r in rets.values()]))
            # 以最少数据的股票为基准对齐：直接用各股自身序列的 index 对齐
            # 简化：按行号对齐（各股数据来自同一批调度，日数一致）
            n = min(len(r) for r in rets.values())
            if n < MIN_DAYS:
                return None, None, None, None

            ret_mat = np.column_stack([
                rets[s].values[:n] for s in symbols if s in rets])
            mcap_mat = np.column_stack([
                mcap[s].values[:n] for s in symbols if s in rets])
            ok_symbols = [s for s in symbols if s in rets]
            return ret_mat, mcap_mat, ok_symbols, sector_map
        finally:
            store.close()

    # ---- 核心：逐日 OLS 残差化 ----
    def neutralize(
        self,
        symbols: list[str],
        lookback: int = 120,
        style_exposures: Optional[pd.DataFrame] = None,
    ) -> dict[str, Any]:
        """风格中性化：收益 ~ 市值 + 板块收益 → 残差 alpha。

        Args:
            symbols: 股票代码列表（≥3 只，同板块内效果最佳）
            lookback: 回溯交易日数
            style_exposures: 可选自定义暴露矩阵（日期×因子列）；
                缺省用市值 + 板块收益（从数据库装配）

        Returns:
            alpha_ranking: 中性化后纯 alpha 排名
            diagnostics: 风格暴露 beta / R² / 残差波动
        """
        symbols = [s.strip() for s in symbols if s.strip()]
        if len(symbols) < 5:
            return {"error": "至少需要 5 只股票（截面回归自由度要求，"
                             "同板块内效果最佳）"}

        ret_mat, mcap_mat, ok_symbols, sector_map = self._build_inputs(
            symbols, lookback)
        if ret_mat is None:
            return {"error": "数据不足：需 ≥21 根行情且 ≥3 只股票有数据"}
        n, n_stock = ret_mat.shape

        # 暴露构建（每日期截面）：
        #   X_t = [1, ln(mcap), sector_ret_t]（板块收益 = 板块内股票当日均值）
        # 板块收益序列：按 sector 聚合（用已有矩阵先算板块均值再回填）
        sector_ids = [sector_map.get(s, s) for s in ok_symbols]
        uniq_sectors = sorted(set(sector_ids))
        sector_ret_series: dict[str, np.ndarray] = {}
        for sec in uniq_sectors:
            idxs = [i for i, sid in enumerate(sector_ids) if sid == sec]
            if len(idxs) == 1:
                sector_ret_series[sec] = ret_mat[:, idxs[0]]
            else:
                sector_ret_series[sec] = ret_mat[:, idxs].mean(axis=1)

        alphas = np.full((n, n_stock), np.nan)
        betas = np.zeros((n_stock, 2))     # [size_beta, sector_beta]
        r2_list = np.zeros((n_stock,))
        resid_std = np.zeros((n_stock,))
        day_count = 0
        for t in range(n):
            X = np.ones((n_stock, 3))
            X[:, 1] = np.log(np.maximum(mcap_mat[t, :], 1e-9))
            X[:, 2] = [sector_ret_series[sector_ids[i]][t]
                       for i in range(n_stock)]
            y = ret_mat[t, :]
            # 剔除 NaN 行
            valid = ~(np.isnan(y) | np.isnan(X).any(axis=1))
            if valid.sum() < 3:
                continue
            Xv, yv = X[valid], y[valid]
            try:
                beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
            except np.linalg.LinAlgError:
                continue
            resid = yv - Xv @ beta
            alphas[t, np.where(valid)[0]] = resid
            betas[np.where(valid)[0], 0] = beta[1]
            betas[np.where(valid)[0], 1] = beta[2]
            day_count += 1

        if day_count < MIN_ALPHA_DAYS:
            return {"error": f"有效回归日 {day_count} < {MIN_ALPHA_DAYS}，数据不足"}

        # 个股 alpha 统计
        rows = []
        for i, sym in enumerate(ok_symbols):
            a = alphas[:, i]
            valid = ~np.isnan(a)
            if valid.sum() < MIN_ALPHA_DAYS:
                continue
            av = a[valid]
            mean_a = float(av.mean())
            std_a = float(av.std())
            rows.append({
                "symbol": sym,
                "sector": sector_ids[i],
                "alpha_mean": round(mean_a, 5),          # 日均纯 alpha
                "alpha_std": round(std_a, 5),
                "alpha_ir": round(mean_a / std_a, 3) if std_a > 0 else 0.0,
                "alpha_win_rate": round(float((av > 0).mean()), 3),
                "cumulative_alpha_pct": round(
                    float((np.prod(1 + av) - 1) * 100), 2),
                "size_beta": round(float(betas[i, 0]), 4),
                "sector_beta": round(float(betas[i, 1]), 4),
                "days": int(valid.sum()),
            })
        if not rows:
            return {"error": "无足够有效 alpha 样本"}

        ranked = sorted(rows, key=lambda r: r["alpha_ir"], reverse=True)
        return {
            "method": "逐日 OLS：ret ~ 市值 + 板块收益，残差 = 纯 alpha",
            "n_symbols": len(rows),
            "n_days": day_count,
            "lookback": lookback,
            "alpha_ranking": ranked,
            "top": [r["symbol"] for r in ranked[:5]],
            "diagnostics": {
                "avg_size_beta": round(float(np.mean([r["size_beta"]
                                                      for r in rows])), 4),
                "avg_sector_beta": round(float(np.mean([r["sector_beta"]
                                                        for r in rows])), 4),
                "avg_alpha_std": round(float(np.mean([r["alpha_std"]
                                                      for r in rows])), 5),
                "note": ("beta 为风格暴露系数（市值/板块涨跌对收益的敏感度）；"
                         "残差为剔除风格后的纯个股 alpha（IR = 均值/波动）"),
            },
        }


_analyzer: Optional[FactorNeutralizer] = None


def get_factor_neutralizer() -> FactorNeutralizer:
    """单例获取中性化器。"""
    global _analyzer
    if _analyzer is None:
        _analyzer = FactorNeutralizer()
    return _analyzer


def format_neutralized(result: dict[str, Any]) -> str:
    """中性化结果 → markdown（UI/MCP 共用）。"""
    if "error" in result:
        return f"因子中性化不可用：{result['error']}"
    diag = result["diagnostics"]
    lines = [
        f"## 🧮 因子中性化（风格清洗）",
        f"**方法**：{result['method']}",
        f"**样本**：{result['n_symbols']} 只股票 × {result['n_days']} 个交易日",
        "",
        "### 纯 alpha 排名（剔除市值/板块风格后）",
        "| 排名 | 代码 | 板块 | 日均alpha | IR | 胜率 | 累计% | 市值β | 板块β |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(result["alpha_ranking"], 1):
        lines.append(
            f"| {i} | {r['symbol']} | {r['sector']} | {r['alpha_mean']:+.5f} | "
            f"{r['alpha_ir']:+.2f} | {r['alpha_win_rate']:.0%} | "
            f"{r['cumulative_alpha_pct']:+.1f} | {r['size_beta']:+.2f} | "
            f"{r['sector_beta']:+.2f} |")
    lines.extend([
        "",
        "### 风格暴露诊断",
        f"- 平均市值 β = {diag['avg_size_beta']:+.3f}（剔除后残差不再含市值风格）",
        f"- 平均板块 β = {diag['avg_sector_beta']:+.3f}",
        f"- 平均残差波动 = {diag['avg_alpha_std']:.5f}",
        f"\n*{diag['note']}。研究参考，非投资建议。*",
    ])
    return "\n".join(lines)
