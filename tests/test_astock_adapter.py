# a-stock-data 适配器（astock）测试 — fixture 驱动，零网络

from __future__ import annotations

import pandas as pd
import pytest

from pa_mcp.data.sources.astock_adapter import AstockAdapter


def test_prefix_routing():
    """前缀路由：沪指数白名单/ETF/北交所 920 段优先判断。"""
    a = AstockAdapter()
    assert a._prefix("000001") == "sz000001"
    assert a._prefix("600036") == "sh600036"
    assert a._prefix("000300") == "sh000300"   # 沪指数白名单
    assert a._prefix("510050") == "sh510050"   # 沪 ETF
    assert a._prefix("920982") == "bj920982"   # 北交所 920 段先于 9x
    assert a._prefix("830799") == "bj830799"
    assert a._prefix("sh600036") == "sh600036"  # 显式前缀透传
    assert a._prefix("300750") == "sz300750"


def test_capabilities():
    assert AstockAdapter.supports("daily_bars")
    assert AstockAdapter.supports("realtime_quote")
    assert not AstockAdapter.supports("minute_bars")
    assert not AstockAdapter.supports("spot_all")


def _fake_quote_line(symbol: str, price: str, amount: str = "12345") -> str:
    """构造腾讯行情响应行（GBK 编码模拟）。"""
    vals = ["1", "测试股", symbol[-6:], price, price, price] + ["0"] * 25
    # 补齐到 53 字段：31=涨跌额 32=涨跌幅 33=高 34=低 37=成交额(万) 38=换手
    # 39=PE 43=振幅 44=流通市值 45=总市值 46=PB 47/48=涨跌停 49=量比 52=静态PE
    while len(vals) < 53:
        vals.append("0")
    vals[31] = "0.1"
    vals[32] = "1.5"
    vals[33] = price
    vals[34] = price
    vals[37] = amount
    vals[38] = "0.5"
    vals[39] = "15.0"
    vals[45] = "1000"
    vals[46] = "1.2"
    return f'v_{AstockAdapter._prefix(symbol)}="{"~".join(vals)}"'


async def test_stale_quote_detection(monkeypatch):
    """僵尸报价检测：成交额 0 且价==昨收 → is_stale=True。"""
    a = AstockAdapter()

    def _fake_open(url, timeout=None, **kw):
        class R:
            def read(self):
                return _fake_quote_line("600036", "10.0", "0").encode("gbk")
        return R()

    monkeypatch.setattr("urllib.request.urlopen", _fake_open)
    q = await a.get_realtime_quote("600036")
    assert q is not None
    assert q["is_stale"] is True  # 停牌/老码定格报价被标记


async def test_normal_quote(monkeypatch):
    """正常报价：is_stale=False，市值/PE 字段正确。"""
    a = AstockAdapter()

    def _fake_open(url, timeout=None, **kw):
        class R:
            def read(self):
                return _fake_quote_line("601728", "6.57", "12345").encode("gbk")
        return R()

    monkeypatch.setattr("urllib.request.urlopen", _fake_open)
    q = await a.get_realtime_quote("601728")
    assert q is not None
    assert q["price"] == 6.57
    assert q["is_stale"] is False
    assert q["pe"] == 15.0
    assert q["mcap_yi"] == 1000.0
    assert q["source"] == "astock"


async def test_baidu_kline_parse(monkeypatch):
    """百度 K 线解析：keys+marketData 分号串 → 标准化 DataFrame。"""
    a = AstockAdapter()
    keys = ["time", "open", "close", "high", "low", "volume", "amount",
            "ma5avgprice"]
    ts = 1784500000  # 2026 年时间戳
    payload = {
        "Result": {"newMarketData": {
            "keys": keys,
            "marketData": (f"{ts},10.0,10.2,10.3,9.9,1000,10000,10.1;"
                           f"{ts + 86400},10.2,10.4,10.5,10.1,1200,12000,10.3;"),
        }}
    }
    import json

    def _fake_open(url, timeout=None, **kw):
        class R:
            def read(self):
                return json.dumps(payload).encode("utf-8")
        return R()

    monkeypatch.setattr("urllib.request.urlopen", _fake_open)
    df = await a.get_daily_kline("601728", start_date="20260101")
    assert len(df) == 2
    assert set(df["source"]) == {"astock"}
    assert "ma5avgprice" in df.columns
    assert df.iloc[-1]["close"] == 10.4


async def test_baidu_kline_failure_empty(monkeypatch):
    """百度 K 线失败/空返回 → 空 DataFrame（router 视为失败切源）。"""

    def _fake_open(url, timeout=None, **kw):
        class R:
            def read(self):
                return b"{}"
        return R()

    monkeypatch.setattr("urllib.request.urlopen", _fake_open)
    df = await AstockAdapter().get_daily_kline("601728")
    assert df.empty
