# 同花顺（ths）adapter 测试 — fixture 字符串驱动，零网络
#
# 通过注入 fake AsyncClient 隔离网络；monkeypatch _throttle 避免限流等待。

from __future__ import annotations

import pytest

from pa_mcp.data.sources.ths_adapter import ThsAdapter

THS_2025 = (
    'quotebridge_v6_line_hs_000001_01_2025('
    '{"data":"20250102,10.0,10.5,10.8,9.9,123456,1234567.0,1.23,,,0;'
    '20250103,10.8,11.0,11.2,10.6,200000,2200000.0,2.10,,,0;","error":""})'
)
THS_EMPTY = 'quotebridge_v6_line_hs_000001_01_1991({"data":"","error":""})'


class FakeResp:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        pass


class FakeClient:
    """按 URL 片段返回预置响应或抛错；记录调用 URL。"""

    def __init__(self, mapping=None, errors=None):
        self._mapping = mapping or {}
        self._errors = errors or {}
        self.calls: list[str] = []

    async def get(self, url: str) -> FakeResp:
        self.calls.append(url)
        for frag, err in self._errors.items():
            if frag in url:
                raise err
        content = self._mapping.get(url, b"")
        if not content:
            content = THS_EMPTY.encode()
        return FakeResp(content)


@pytest.fixture
def adapter(monkeypatch):
    a = ThsAdapter()

    async def _no_throttle():
        return None
    monkeypatch.setattr(a, "_throttle", _no_throttle)
    return a


# ---- 代码映射 ----

def test_code_mapping():
    assert ThsAdapter._to_ths_code("600036") == "hs_600036"
    assert ThsAdapter._to_ths_code("000001") == "hs_000001"
    assert ThsAdapter._to_ths_code("300750") == "hs_300750"
    assert ThsAdapter._to_ths_code("830799") == "bj_830799"
    assert ThsAdapter._to_ths_code("920001") == "bj_920001"
    assert ThsAdapter._to_ths_code("sh600036") == "hs_600036"
    assert ThsAdapter._to_ths_code("bj830799") == "bj_830799"


# ---- JSONP 剥壳 ----

def test_strip_jsonp():
    d = ThsAdapter._strip_jsonp(THS_2025)
    assert d["data"].startswith("20250102,10.0")


def test_strip_jsonp_invalid():
    with pytest.raises(ValueError):
        ThsAdapter._strip_jsonp("not jsonp at all")


# ---- 单年数据解析 ----

def test_parse_year_data():
    rows = ThsAdapter._parse_year_data(
        "20250102,10.0,10.5,10.8,9.9,123456,1234567.0,1.23,,,0;"
        "20250103,10.8,11.0,11.2,10.6,200000,2200000.0,2.10,,,0;")
    assert len(rows) == 2
    d, o, h, lo, c, v, amt = rows[0]
    assert d == "20250102" and o == 10.0 and h == 10.5 and lo == 10.8
    assert c == 9.9
    assert v == 123456      # volume 保持股，不换算
    assert amt == 1234567.0
    # 第 8-10 字段（turnover/空/flag）被忽略
    assert rows[1][0] == "20250103"


def test_parse_year_data_empty():
    assert ThsAdapter._parse_year_data("") == []
    assert ThsAdapter._parse_year_data(None) == []


def test_parse_year_data_bad_row_skipped():
    rows = ThsAdapter._parse_year_data(
        "20250102,10.0,10.5,10.8,9.9,123456,1234567.0,1.23,,,0;"
        "badrow;"
        "20250104,1,2,3,4,5,6,,,0;")
    assert len(rows) == 2  # badrow 被跳过


# ---- 年份区间 ----

async def test_year_range_explicit(adapter, monkeypatch):
    calls = []

    async def fake_fetch(client, ths_code, year):
        calls.append(year)
        return []

    monkeypatch.setattr(adapter, "_fetch_year", fake_fetch)
    await adapter.get_daily_kline("000001", start_date="20230101",
                                  end_date="20251231")
    assert calls == [2023, 2024, 2025]


async def test_year_range_full_history(adapter, monkeypatch):
    calls = []

    async def fake_fetch(client, ths_code, year):
        calls.append(year)
        return []

    monkeypatch.setattr(adapter, "_fetch_year", fake_fetch)
    await adapter.get_daily_kline("000001")  # 空参数 → 全历史
    assert calls[0] == 1991
    assert calls[-1] >= 2025


# ---- 拼接 / 过滤 / 标准化 ----

async def test_by_year_concat_and_filter(adapter, monkeypatch):
    """跨年 rows 拼接、日期过滤、升序、列标准化。"""
    year_data = {
        2023: [("20231229", 10.0, 10.1, 10.2, 9.9, 100, 1000.0)],
        2024: [("20240102", 10.5, 10.6, 10.7, 10.4, 200, 2000.0),
               ("20241231", 11.0, 11.1, 11.2, 10.9, 300, 3000.0)],
        2025: [("20250102", 11.5, 11.6, 11.7, 11.4, 400, 4000.0)],
    }

    async def fake_fetch(client, ths_code, year):
        return year_data.get(year, [])

    monkeypatch.setattr(adapter, "_fetch_year", fake_fetch)
    df = await adapter.get_daily_kline("000001", start_date="20240101",
                                       end_date="20250131")
    assert len(df) == 3  # 20231229 被日期过滤掉
    assert list(df["date"].astype(str)) == [
        "2024-01-02", "2024-12-31", "2025-01-02"]  # 升序
    assert set(df["symbol"]) == {"000001"}
    assert set(df["source"]) == {"ths"}
    assert set(df["price_adjust_mode"]) == {"bfq"}
    assert list(df.columns) == [
        "symbol", "date", "open", "close", "high", "low",
        "volume", "amount", "source", "price_adjust_mode"]


async def test_empty_year_not_fatal(adapter, monkeypatch):
    """中间某年无数据不致命；全部为空返回空 DataFrame。"""
    year_data = {2024: [("20240102", 10.0, 10.1, 10.2, 9.9, 100, 1000.0)]}

    async def fake_fetch(client, ths_code, year):
        return year_data.get(year, [])

    monkeypatch.setattr(adapter, "_fetch_year", fake_fetch)
    df = await adapter.get_daily_kline("000001", start_date="20230101",
                                       end_date="20250131")
    assert len(df) == 1  # 2023/2025 空年不抛错

    async def fake_empty(client, ths_code, year):
        return []

    monkeypatch.setattr(adapter, "_fetch_year", fake_empty)
    df2 = await adapter.get_daily_kline("000001")
    assert df2.empty  # 全空 → 空 DataFrame（router 视为失败自动切源）


async def test_network_failure_raises(adapter, monkeypatch):
    """网络异常透传 raise（router 记录并熔断）。"""
    import httpx

    async def fake_fetch(client, ths_code, year):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(adapter, "_fetch_year", fake_fetch)
    with pytest.raises(httpx.ConnectError):
        await adapter.get_daily_kline("000001")


# ---- 能力边界 ----

async def test_bfq_only_adjust(adapter, monkeypatch):
    """adjust=qfq 仍返回数据且 price_adjust_mode 恒 bfq（hfq 实测无效）。"""

    async def fake_fetch(client, ths_code, year):
        return [("20250102", 10.0, 10.1, 10.2, 9.9, 100, 1000.0)]

    monkeypatch.setattr(adapter, "_fetch_year", fake_fetch)
    df = await adapter.get_daily_kline("000001", adjust="qfq",
                                       start_date="20250101",
                                       end_date="20250131")
    assert len(df) == 1
    assert set(df["price_adjust_mode"]) == {"bfq"}


async def test_period_unsupported(adapter):
    with pytest.raises(NotImplementedError):
        await adapter.get_daily_kline("000001", period="weekly")


def test_capabilities():
    assert ThsAdapter.supports("daily_bars")
    assert not ThsAdapter.supports("minute_bars")
    assert not ThsAdapter.supports("realtime_quote")
    assert not ThsAdapter.supports("spot_all")
    assert ThsAdapter.volume_unit == "shares"


# ---- router 集成 ----

def test_router_integration(monkeypatch):
    """DataSourceRouter 链内 ths 正常工作，source 覆写为 'ths'。"""
    import asyncio

    from pa_mcp.data.router import DataSourceRouter, CircuitBreakerConfig

    a = ThsAdapter()

    async def _no_throttle():
        return None
    monkeypatch.setattr(a, "_throttle", _no_throttle)

    async def fake_fetch(client, ths_code, year):
        return [("20250102", 10.0, 10.1, 10.2, 9.9, 100, 1000.0)]

    monkeypatch.setattr(a, "_fetch_year", fake_fetch)
    a._client = FakeClient()  # 不需要真实网络

    router = DataSourceRouter([("ths", a)],
                              {"ths": CircuitBreakerConfig()})

    async def _main():
        df, src = await router.fetch_daily_kline(
            symbol="000001", period="daily",
            start_date="20250101", end_date="20250131", adjust="bfq")
        return df, src

    df, src = asyncio.run(_main())
    assert src == "ths"
    assert len(df) == 1
    assert set(df["source"]) == {"ths"}
