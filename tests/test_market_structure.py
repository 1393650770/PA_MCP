# [AI:BEGIN]
# PA_MCP - 市场结构联合分析测试
# [AI:END]

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from pa_mcp.research.market_structure import MarketStructureAnalyzer


def _zigzag_index(n=220, seed=5):
    """锯齿行情（产生中枢/背驰）。"""
    rng = np.random.default_rng(seed)
    close = 3000.0
    rows = []
    for i in range(n):
        phase = (i // 50) % 4
        if phase in (0, 2):
            close *= 1 + 0.003 + rng.normal(0, 0.004)
        else:
            close *= 1 - 0.002 + rng.normal(0, 0.004)
        rows.append({"date": pd.Timestamp("2025-09-01") + pd.Timedelta(days=i),
                     "open": close * 0.995, "high": close * 1.01,
                     "low": close * 0.99, "close": close, "volume": 1e8})
    return pd.DataFrame(rows)


def test_joint_verdict_structure():
    """联合判断：结构/环境/偏向来判定。"""
    from pa_mcp.research.market_structure import MarketStructureAnalyzer as M
    # 中枢上方 + 无背驰
    class FakeChan:
        position = "中枢上方"
        beichi_signal = "none"
        bi_list = []
        zhongshu_list = []
        beichi_detail = "无背驰"

    j = M._joint(FakeChan(), {"sentiment_stage": "fermenting",
                              "regime_label": "扩散期"})
    assert "多头结构" in j["structure"]
    assert j["bias"] == "偏多"

    # 高潮 + 上涨背驰 → 风险叠加
    class FakeChan2:
        position = "中枢上方"
        beichi_signal = "bearish"
        bi_list = []
        zhongshu_list = []
        beichi_detail = "上涨背驰"

    j2 = M._joint(FakeChan2(), {"sentiment_stage": "climax",
                                "regime_label": "亢奋顶"})
    assert "衰竭" in j2["structure"]
    assert "风险信号叠加" in j2["environment"]

    # 冰点 → 中性偏防守
    class FakeChan3:
        position = "中枢下方"
        beichi_signal = "bullish"
        bi_list = []
        zhongshu_list = []
        beichi_detail = "下跌背驰"

    j3 = M._joint(FakeChan3(), {"sentiment_stage": "frozen",
                                "regime_label": "冰点"})
    assert "下跌背驰" in j3["structure"]
    assert "冰点" in j3["environment"]


def test_full_analysis_with_index_db(tmp_path):
    """灌 index_daily → 完整分析（不依赖网络）。"""
    from pa_mcp.data.store import DuckDBStore
    db = str(tmp_path / "ms_test.duckdb")
    store = DuckDBStore(db)
    store.connect()
    df = _zigzag_index()
    store.insert_df("index_daily", pd.DataFrame({
        **{k: v for k, v in df.items()},
        "symbol": ["sh000001"] * len(df),
    }))
    store.close()

    r = asyncio.run(MarketStructureAnalyzer(store_path=db).analyze())
    assert r["index"]["rows"] >= 100
    assert r["index"]["last_close"] is not None
    assert r["chan"] is not None
    assert r["joint"]["bias"] in ("偏多", "偏空", "中性")
    assert "市场结构联合分析" in r["report"]


def test_no_index_data(tmp_path):
    r = asyncio.run(MarketStructureAnalyzer(
        store_path=str(tmp_path / "none.db")).analyze(use_network=False))
    assert r["index"]["rows"] == 0
    assert r["chan"] is None
    assert "数据不足" in r["joint"]["verdict"]
