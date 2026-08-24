# ETF 支持测试：代码识别 / 列表 / 批量行情折溢价 / 数据源前缀映射

from __future__ import annotations

import asyncio
import json

import pytest

from pa_mcp.research import etf as em


# ---- 代码识别 ----

@pytest.mark.parametrize("code,expected", [
    ("510300", True), ("588000", True), ("159915", True), ("512880", True),
    ("561590", True), ("600036", False), ("000001", False), ("300750", False),
    ("sh510300", True), ("sz159915", True), ("920001", False),
])
def test_is_etf(code, expected):
    assert em.is_etf(code) is expected


def test_etf_exchange():
    assert em.etf_exchange("510300") == "sh"
    assert em.etf_exchange("588000") == "sh"
    assert em.etf_exchange("159915") == "sz"
    assert em.etf_exchange("sh510300") == "sh"


# ---- 列表（mock 东财响应） ----

FAKE_LIST_RESP = {"data": {"diff": [
    {"f12": "510300", "f14": "沪深300ETF华泰柏瑞", "f2": 4.627, "f3": -1.13,
     "f6": 1.2e9, "f8": 3.5},
    {"f12": "159915", "f14": "创业板ETF易方达", "f2": 3.457, "f3": -2.89,
     "f6": 8e8, "f8": 4.2},
    {"f12": "600036", "f14": "招商银行", "f2": 36.5, "f3": 1.0,
     "f6": 5e8, "f8": 1.0},  # 非 ETF 应被过滤
]}}

FAKE_QUOTE_RESP = (
    'v_sh510300="1~沪深300ETF华泰柏瑞~510300~4.627~4.680~4.660~'
    '123456~111~222~333~444~555~666~777~888~999~1000~1100~1200~1300~'
    '1400~1500~1600~1700~1800~1900~2000~2100~2200~2300~2400~2500~'
    '2600~-1.13~2700~2800~2900~3000~3100~3200~3300~3400~3500~3600~'
    '3700~3800~3900~4000~4100~4200~4300~4.630~";'
)


@pytest.fixture(autouse=True)
def _mock_network(monkeypatch):
    """mock 东财列表与腾讯行情。"""
    async def _fake_fetch_json(url):
        return FAKE_LIST_RESP

    async def _fake_quotes_raw(symbols):
        return FAKE_QUOTE_RESP

    monkeypatch.setattr(em, "_fetch_json", _fake_fetch_json)
    monkeypatch.setattr(em, "_fetch_quotes_raw", _fake_quotes_raw)
    em.clear_etf_list_cache()


def test_list_filters_non_etf():
    """列表只保留 ETF（过滤混入股票）。"""
    lst = asyncio.run(em.fetch_etf_list(limit=10))
    syms = [x["symbol"] for x in lst]
    assert "510300" in syms and "159915" in syms
    assert "600036" not in syms
    hit = next(x for x in lst if x["symbol"] == "510300")
    assert hit["name"] == "沪深300ETF华泰柏瑞"
    assert hit["price"] == 4.627
    assert hit["amount_billion"] == 12.0  # 1.2e9 元 = 12 亿


def test_quote_premium_calc():
    """批量行情：IOPV 与折溢价计算（4.627 vs 4.630）。"""
    qs = asyncio.run(em.fetch_etf_quotes(["510300"]))
    q = qs.get("510300")
    assert q is not None
    assert q["name"] == "沪深300ETF华泰柏瑞"
    assert q["price"] == 4.627
    assert q["iopv"] == 4.630
    assert q["premium_pct"] == pytest.approx(-0.06, abs=0.02)
    assert q["is_stale"] is False


def _set_field(raw: str, idx: int, value: str) -> str:
    """按字段位替换腾讯行情文本（避免索引错位）。"""
    head, _, body = raw.partition('"')
    vals = body.split("~")
    if idx < len(vals):
        vals[idx] = value
    return head + '"' + "~".join(vals)


def test_quote_stale_detection():
    """僵尸报价检测：价=昨收且成交 0。"""
    # 价格=昨收（f3=f4）且成交额（f37）为 0 → 僵尸报价
    raw = _set_field(_set_field(FAKE_QUOTE_RESP, 3, "4.680"), 37, "0")
    import pa_mcp.research.etf as em2
    async def _fake_raw(symbols):
        return raw
    orig = em2._fetch_quotes_raw
    em2._fetch_quotes_raw = _fake_raw
    try:
        qs = asyncio.run(em2.fetch_etf_quotes(["510300"]))
    finally:
        em2._fetch_quotes_raw = orig
    assert qs["510300"]["is_stale"] is True


def test_format_etf_line():
    q = {"symbol": "510300", "name": "沪深300ETF华泰柏瑞", "price": 4.627,
         "change_pct": -1.13, "premium_pct": -0.06, "is_stale": False}
    line = em.format_etf_line(q)
    assert "510300" in line and "折溢-0.06%" in line


def test_get_etf_name_cache():
    """名称缓存：列表拉取后名称可查。"""
    asyncio.run(em.fetch_etf_list(limit=10))
    assert em.get_etf_name("510300") == "沪深300ETF华泰柏瑞"


# ---- 数据源前缀映射（ETF 修复） ----

def test_tencent_prefix_etf():
    from pa_mcp.data.sources.tencent_adapter import TencentAdapter
    t = TencentAdapter()
    assert t._to_tencent_code("510300") == "sh510300"
    assert t._to_tencent_code("588000") == "sh588000"
    assert t._to_tencent_code("159915") == "sz159915"
    assert t._to_tencent_code("000001") == "sz000001"   # 平安银行（股票优先）
    assert t._to_tencent_code("sh000001") == "sh000001"  # 显式前缀=上证指数
    assert t._to_tencent_code("000300") == "sh000300"    # 沪深300 指数白名单
    assert t._to_tencent_code("920001") == "bj920001"    # 北交所先于 9x 判断


def test_sina_prefix_etf():
    from pa_mcp.data.sources.sina_adapter import SinaAdapter
    s = SinaAdapter()
    assert s._to_sina_code("510300") == "sh510300"
    assert s._to_sina_code("159915") == "sz159915"
