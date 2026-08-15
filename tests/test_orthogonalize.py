# [AI:BEGIN]
# PA_MCP - 因子正交化（风格中性化）测试
# [AI:END]

from __future__ import annotations

import numpy as np
import pandas as pd

from pa_mcp.research.orthogonalize import (
    FactorNeutralizer,
    format_neutralized,
)


def _seed(tmp_path, specs, seed=7):
    """灌行情 + 板块映射。

    specs: {symbol: (sector, daily_ret, alpha_daily)}
      daily_ret: 基础日收益（板块联动部分）
      alpha_daily: 个股独立 alpha（叠加）
    每只股票叠加独立白噪声（防 alpha 直接推高市值的共线假象——
    真实数据中市值由成交/发行决定，与短期 alpha 弱相关）。
    """
    from pa_mcp.data.store import DuckDBStore
    store = DuckDBStore(str(tmp_path / "orth_test.duckdb"))
    store.connect()
    rng = np.random.default_rng(seed)
    n = 80
    dates = pd.date_range("2026-05-01", periods=n, freq="B")
    sb_rows, k_rows = [], []
    for i, (sym, (sector, base, alpha)) in enumerate(specs.items()):
        # 静态市值：与 alpha 解耦（独立随机，同数量级）
        sb_rows.append({"symbol": sym, "name": sym, "sector": sector,
                        "industry": sector, "is_st": False,
                        "market_cap": 5e9 + 3e9 * (i % 3)})
        close = 10.0
        noise = rng.normal(0, 0.002, n)  # 独立于 alpha 的个股噪声（信噪比充足）
        for i in range(n):
            ret = base + alpha + noise[i]
            close *= 1 + ret
            k_rows.append({
                "symbol": sym, "date": dates[i], "open": close * 0.99,
                "close": close, "high": close * 1.01, "low": close * 0.99,
                "volume": 1e6, "amount": 1e7, "pct_change": ret * 100,
                "turnover": 1.0, "change": ret * 100, "amplitude": 2.0,
                "adjust_factor": 1.0,
            })
    store.insert_df("stock_basic", pd.DataFrame(sb_rows))
    store.insert_df("kline_daily", pd.DataFrame(k_rows))
    store.close()
    return store.db_path


def test_neutralize_isolates_alpha(tmp_path):
    """同板块联动 + 一只独立 alpha：中性化后该股 alpha 排名第一。"""
    # 银行板块 4 只共享 +0.1%/日；000001 另有 +0.15%/日独立 alpha
    db = _seed(tmp_path, {
        "000001": ("银行", 0.001, 0.0015),
        "000002": ("银行", 0.001, 0.0),
        "000003": ("银行", 0.001, -0.0004),
        "000004": ("银行", 0.001, 0.0002),
        "600001": ("白酒", 0.0005, 0.0),
        "600002": ("白酒", 0.0005, 0.0005),
        "600003": ("白酒", 0.0005, -0.0002),
    })
    r = FactorNeutralizer(store_path=db).neutralize(
        ["000001", "000002", "000003", "000004",
         "600001", "600002", "600003"])
    assert "error" not in r
    assert r["top"][0] == "000001"  # 纯 alpha 最高者居首
    assert r["n_symbols"] == 7
    assert r["n_days"] >= 70
    # 诊断字段齐全
    d = r["diagnostics"]
    assert "avg_size_beta" in d and "avg_sector_beta" in d
    text = format_neutralized(r)
    assert "纯 alpha" in text and "风格暴露" in text


def test_neutralize_few_symbols():
    r = FactorNeutralizer().neutralize(["000001", "000002", "000003", "000004"])
    assert "error" in r


def test_neutralize_no_data(tmp_path):
    r = FactorNeutralizer(store_path=str(tmp_path / "none.db")).neutralize(
        ["000001", "000002", "000003"])
    assert "error" in r


def test_neutralize_consistent_ranking(tmp_path):
    """同板块内 alpha 排序与 alpha 大小一致。"""
    db = _seed(tmp_path, {
        "000001": ("银行", 0.001, 0.0010),
        "000002": ("银行", 0.001, 0.0005),
        "000003": ("银行", 0.001, 0.0000),
        "000004": ("银行", 0.001, -0.0005),
        "000005": ("银行", 0.001, -0.0010),
    })
    r = FactorNeutralizer(store_path=db).neutralize(
        ["000001", "000002", "000003", "000004", "000005"])
    ranking = {x["symbol"]: i for i, x in enumerate(r["alpha_ranking"])}
    assert ranking["000001"] < ranking["000002"] < ranking["000003"] \
        < ranking["000004"] < ranking["000005"]
