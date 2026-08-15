# [AI:BEGIN]
# PA_MCP - Research: 因子工厂（因子注册 / 批量验证 / 打包）
#
# 机制借鉴开源 QuantSkills factor-skill-factory + Qlib 因子工作流：
#   1. FactorRegistry：装饰器注册因子（name/category/description/version）
#   2. 内置因子：复用已有指标库（MA/RSI/MACD/ATR/BOLL/量比/动量/缠论）
#      ——非必要不造轮子
#   3. evaluate_factor：单因子检验——IC（秩相关）+ 分层（Q1-Q5 组平均
#      未来收益）+ 单调性 + 覆盖率（量化界标准因子检验）
#   4. scan_factors：股票池批量验证 → 因子排行（有效因子清单）
#   5. pack_factor：因子定义打包为 JSON（可版本化/复用/跨会话）
# [AI:END]

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HORIZON_DEFAULT = 5      # 默认前瞻窗口（交易日）
IC_MIN_SAMPLES = 30      # IC 最小样本
QUANTILES = 5            # 分层数


# ---- 因子定义与注册表 ----

@dataclass
class FactorDefinition:
    name: str
    category: str            # momentum/mean_reversion/volatility/volume/trend/value
    description: str
    version: str = "v1"
    lookback: int = 20       # 参考信息窗口（说明用）
    fn: Optional[Callable] = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "category": self.category,
                "description": self.description, "version": self.version,
                "lookback": self.lookback}


class FactorRegistry:
    """因子注册表（装饰器模式，类比 StrategyRegistry）。"""

    def __init__(self) -> None:
        self._factors: dict[str, FactorDefinition] = {}

    def register(self, fn: Callable, name: str, category: str,
                 description: str, version: str = "v1",
                 lookback: int = 20) -> FactorDefinition:
        """注册因子。"""
        fd = FactorDefinition(name=name, category=category,
                              description=description, version=version,
                              lookback=lookback, fn=fn)
        self._factors[name] = fd
        return fd

    def factor(self, name: str, category: str, description: str,
               version: str = "v1", lookback: int = 20):
        """装饰器用法：@registry.factor(...)"""
        def deco(fn: Callable) -> Callable:
            self.register(fn, name, category, description, version, lookback)
            return fn
        return deco

    def get(self, name: str) -> Optional[FactorDefinition]:
        return self._factors.get(name)

    def list_all(self) -> list[FactorDefinition]:
        return list(self._factors.values())

    def list_by_category(self, category: str) -> list[FactorDefinition]:
        return [f for f in self._factors.values()
                if f.category == category]


# ---- 内置因子（复用已有指标库） ----

_registry = FactorRegistry()


def _ensure(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("date").reset_index(drop=True)


@_registry.factor("ma_gap20", "mean_reversion", "收盘相对 MA20 偏离（%）：负 = 超跌",
                  lookback=20)
def _f_ma_gap20(df: pd.DataFrame) -> pd.Series:
    d = _ensure(df)
    ma = d["close"].rolling(20).mean()
    return (d["close"] / ma - 1) * 100


@_registry.factor("rsi14", "mean_reversion", "RSI(14) 水平：高 = 超买", lookback=14)
def _f_rsi14(df: pd.DataFrame) -> pd.Series:
    from pa_mcp.engine.indicators.indicators import calc_rsi
    r = calc_rsi(_ensure(df))
    return r["rsi14"] if "rsi14" in r else pd.Series(dtype=float)


@_registry.factor("macd_hist", "momentum", "MACD 柱（归一化到价格 %）", lookback=26)
def _f_macd_hist(df: pd.DataFrame) -> pd.Series:
    from pa_mcp.engine.indicators.indicators import calc_macd
    m = calc_macd(_ensure(df))
    if "macd_hist" not in m:
        return pd.Series(dtype=float)
    return m["macd_hist"] / _ensure(df)["close"] * 100


@_registry.factor("atr_pct", "volatility", "ATR(14) 占价格 %（波动率）", lookback=14)
def _f_atr_pct(df: pd.DataFrame) -> pd.Series:
    from pa_mcp.engine.indicators.indicators import calc_atr
    a = calc_atr(_ensure(df))
    if "atr14" not in a:
        return pd.Series(dtype=float)
    return a["atr14"] / _ensure(df)["close"] * 100


@_registry.factor("boll_position", "mean_reversion", "布林 %B：>100 突破上轨",
                  lookback=20)
def _f_boll_pos(df: pd.DataFrame) -> pd.Series:
    from pa_mcp.engine.indicators.indicators import calc_bollinger
    b = calc_bollinger(_ensure(df))
    if not all(c in b for c in ("boll_up", "boll_low")):
        return pd.Series(dtype=float)
    d = _ensure(df)
    rng = (b["boll_up"] - b["boll_low"]).replace(0, np.nan)
    return (d["close"] - b["boll_low"]) / rng * 100


@_registry.factor("vol_ratio", "volume", "量比：当日量 / 20 日均量", lookback=20)
def _f_vol_ratio(df: pd.DataFrame) -> pd.Series:
    d = _ensure(df)
    avg = d["volume"].rolling(20).mean().replace(0, np.nan)
    return d["volume"] / avg


@_registry.factor("ret20", "momentum", "20 日收益率（%）", lookback=20)
def _f_ret20(df: pd.DataFrame) -> pd.Series:
    d = _ensure(df)
    return d["close"].pct_change(20) * 100


@_registry.factor("ret60", "momentum", "60 日收益率（%）", lookback=60)
def _f_ret60(df: pd.DataFrame) -> pd.Series:
    d = _ensure(df)
    return d["close"].pct_change(60) * 100


@_registry.factor("high_52w_dist", "trend", "距 52 周新高（%，0 = 新高）",
                  lookback=250)
def _f_high52_dist(df: pd.DataFrame) -> pd.Series:
    d = _ensure(df)
    hi = d["high"].rolling(250).max()
    return (d["close"] / hi - 1) * 100


def get_factor_registry() -> FactorRegistry:
    """获取因子注册表（单例）。"""
    return _registry


# ---- 因子检验 ----

def evaluate_factor(factor: FactorDefinition, kline_df: pd.DataFrame,
                    horizon: int = HORIZON_DEFAULT) -> dict[str, Any]:
    """单因子检验（量化标准）：IC + 分层 + 单调性 + 覆盖率。

    - IC：因子值与未来 horizon 日收益的 Spearman 秩相关（全部样本）
    - 分层：因子值分位 Q1-Q5，各组未来收益均值 → 单调性 + 高低组差
    - 覆盖率：有效样本比例（NA 剔除后）
    """
    if factor.fn is None:
        return {"error": f"因子 {factor.name} 无实现函数"}
    if kline_df is None or kline_df.empty:
        return {"error": "无行情数据"}
    d = _ensure(kline_df)
    try:
        values = factor.fn(d)
    except Exception as e:  # noqa: BLE001
        return {"error": f"因子计算失败：{e}"}
    if values is None or len(values) != len(d):
        return {"error": "因子输出长度与行情不一致"}

    values = pd.Series(values.to_numpy(), index=d.index)
    fwd = d["close"].shift(-horizon) / d["close"] - 1  # 未来 horizon 收益
    valid = values.notna() & fwd.notna()
    n_valid = int(valid.sum())
    if n_valid < IC_MIN_SAMPLES:
        return {"error": f"有效样本 {n_valid} < {IC_MIN_SAMPLES}，无法检验"}

    v = values[valid].astype(float)
    r = fwd[valid].astype(float)
    # Spearman IC（rank + pearson，无 scipy）
    ic = float(v.rank().corr(r.rank())) if v.rank().corr(r.rank()) is not None \
        else 0.0
    # 分层
    q = pd.qcut(v, QUANTILES, labels=False, duplicates="drop")
    group_ret = r.groupby(q).mean()
    if len(group_ret) >= 2:
        spread = float(group_ret.iloc[-1] - group_ret.iloc[0])
        monotonic = all(
            group_ret.iloc[i] < group_ret.iloc[i + 1] or
            group_ret.iloc[i] > group_ret.iloc[i + 1]
            for i in range(len(group_ret) - 1))
    else:
        spread, monotonic = 0.0, False

    return {
        "factor": factor.name,
        "category": factor.category,
        "version": factor.version,
        "description": factor.description,
        "horizon": horizon,
        "n_samples": n_valid,
        "coverage_pct": round(n_valid / len(d) * 100, 1),
        "ic": round(ic, 4),
        "quantile_returns": [round(float(x), 4) for x in group_ret],
        "spread_pct": round(spread * 100, 3),      # Q5 - Q1 未来收益差
        "monotonic": bool(monotonic),
        "useful": abs(ic) >= 0.03,                 # |IC| ≥ 0.03 视为有信息
    }


def scan_factors(kline_df: pd.DataFrame,
                 horizon: int = HORIZON_DEFAULT) -> list[dict[str, Any]]:
    """批量验证全部注册因子 → 按 |IC| 排序的因子排行。"""
    results = []
    for fd in get_factor_registry().list_all():
        r = evaluate_factor(fd, kline_df, horizon=horizon)
        if "error" in r:
            r["factor"] = fd.name
            r["error"] = r["error"]
            results.append(r)
            continue
        results.append(r)
    results.sort(key=lambda x: abs(x.get("ic", 0)), reverse=True)
    return results


def pack_factor(name: str) -> Optional[dict[str, Any]]:
    """因子定义打包为 JSON（可版本化/复用）。"""
    fd = get_factor_registry().get(name)
    if fd is None:
        return None
    return {"factor": fd.to_dict(),
            "interface": "fn(kline_df: DataFrame) -> Series（与行情等长）",
            "packed_at_note": "定义打包（不含函数体）；跨会话复用需重新注册"}


def format_factor_report(result: dict[str, Any]) -> str:
    """单因子检验 → markdown。"""
    if "error" in result:
        return f"因子 {result.get('factor', '')}：{result['error']}"
    mono = "✅ 单调" if result["monotonic"] else "❌ 非单调"
    useful = "✅ 有信息量" if result["useful"] else "⚠️ 弱信息"
    q_str = " → ".join(f"{x:+.2f}%" for x in result["quantile_returns"])
    return (
        f"## 🧬 因子检验：{result['factor']}（{result['category']} · "
        f"{result['version']}）\n"
        f"**IC**：{result['ic']:+.4f}（{useful}）｜ 前瞻 {result['horizon']} 日\n"
        f"**分层收益**（Q1→Q5）：{q_str}\n"
        f"**高低组差**：{result['spread_pct']:+.2f}% ｜ {mono}\n"
        f"**样本**：{result['n_samples']}（覆盖率 {result['coverage_pct']}%）\n"
        f"*IC = 因子值与未来收益的秩相关（|IC|≥0.03 视为有信息）。"
        f"研究参考，非投资建议。*"
    )


def format_scan(results: list[dict[str, Any]]) -> str:
    """因子批量扫描 → markdown。"""
    if not results:
        return "无因子可检验"
    lines = ["## 🧬 因子批量扫描（按 |IC| 排序）",
             "| 排名 | 因子 | 类别 | IC | 高低组差% | 单调 | 样本 |",
             "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(results, 1):
        if "error" in r:
            lines.append(f"| {i} | {r.get('factor', '?')} | — | ❌ {r['error'][:40]} |")
            continue
        lines.append(
            f"| {i} | {r['factor']} | {r['category']} | {r['ic']:+.4f} | "
            f"{r['spread_pct']:+.2f} | {'✅' if r['monotonic'] else '❌'} | "
            f"{r['n_samples']} |")
    lines.append("\n*IC 为因子值与未来收益的秩相关；高低组差 = Q5-Q1 未来收益。"
                 "研究参考，非投资建议。*")
    return "\n".join(lines)
