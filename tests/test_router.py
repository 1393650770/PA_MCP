# PA_MCP - 多数据源容灾测试
# 覆盖：CircuitBreaker 状态机、DataSourceRouter 顺序回退/熔断/健康报告、
# 腾讯/东财 adapter 解析、scheduler 集成。

from __future__ import annotations

import sys
from datetime import date, datetime

import pandas as pd
import pytest

sys.path.insert(0, "src")

from pa_mcp.data.router import (
    CircuitBreaker, CircuitBreakerConfig, DataSourceRouter,
    AllSourcesFailedError, BreakerState,
)
from pa_mcp.data.sources.tencent_adapter import TencentAdapter
from pa_mcp.data.sources.eastmoney_adapter import EastMoneyAdapter
from pa_mcp.data.sources.base import MarketDataSourcePort


# ---- Fake adapters ----

class FakeSource:
    """Configurable fake adapter for router tests."""

    def __init__(self, name: str, fail: bool = False,
                 capabilities: dict[str, str] | None = None) -> None:
        self.name = name
        self.fail = fail
        self.calls = 0
        self.CAPABILITIES = capabilities or {"daily_bars": "available"}

    def supports(self, capability: str) -> bool:
        return self.CAPABILITIES.get(capability, "unavailable") != "unavailable"

    async def get_daily_kline(self, symbol: str, period: str = "daily",
                              start_date: str = "", end_date: str = "",
                              adjust: str = "qfq") -> pd.DataFrame:
        self.calls += 1
        if self.fail:
            raise RuntimeError(f"{self.name} is down")
        return pd.DataFrame([{
            "symbol": symbol, "date": date(2026, 7, 30),
            "open": 10.0, "high": 11.0, "low": 9.5, "close": 10.8,
            "volume": 1e6, "amount": 1e7,
            "amplitude": 5.0, "pct_change": 2.5, "change": 0.3,
            "turnover": 1.2, "adjust_factor": 1.0,
        }])

    async def get_realtime_spot_all(self) -> pd.DataFrame:
        if self.fail:
            raise RuntimeError(f"{self.name} is down")
        return pd.DataFrame([{"代码": "000001", "名称": "测试", "总市值": 1e10}])


# ---- CircuitBreaker tests ----

class TestCircuitBreaker:
    def test_closed_allows_requests(self):
        cb = CircuitBreaker("test")
        assert cb.allow_request() is True
        assert cb.health.state == BreakerState.CLOSED

    def test_failures_trip_to_open(self):
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3, cooldown_seconds=60))
        cb.record_failure("err1")
        cb.record_failure("err2")
        assert cb.health.state == BreakerState.CLOSED  # 未到阈值
        cb.record_failure("err3")
        assert cb.health.state == BreakerState.OPEN
        assert cb.health.trip_count == 1
        assert cb.allow_request() is False  # OPEN 阻断

    def test_cooldown_transitions_to_half_open(self, monkeypatch):
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2, cooldown_seconds=10))
        cb.record_failure("a")
        cb.record_failure("b")
        assert cb.health.state == BreakerState.OPEN
        assert cb.allow_request() is False

        # 模拟冷却时间已过
        cb.health.last_trip_at = 0  # 过去很久
        assert cb.allow_request() is True  # HALF_OPEN 放行探测
        assert cb.health.state == BreakerState.HALF_OPEN

    def test_success_closes_half_open(self):
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2, cooldown_seconds=10))
        cb.record_failure("a")
        cb.record_failure("b")
        cb.health.last_trip_at = 0
        assert cb.allow_request()  # -> HALF_OPEN
        cb.record_success()
        assert cb.health.state == BreakerState.CLOSED  # 探测成功闭合
        assert cb.health.consecutive_failures == 0

    def test_health_stats(self):
        cb = CircuitBreaker("test")
        cb.record_success()
        cb.record_failure("boom")
        assert cb.health.success_count == 1
        assert cb.health.failure_count == 1
        assert "boom" in cb.health.last_error


# ---- DataSourceRouter tests ----

class TestDataSourceRouter:
    @pytest.mark.asyncio
    async def test_primary_success(self):
        primary = FakeSource("primary")
        router = DataSourceRouter([("primary", primary)])
        df, source = await router.fetch_daily_kline("000001")
        assert source == "primary"
        assert not df.empty
        assert primary.calls == 1

    @pytest.mark.asyncio
    async def test_failover_to_backup(self):
        primary = FakeSource("primary", fail=True)
        backup = FakeSource("backup")
        router = DataSourceRouter([("primary", primary), ("backup", backup)])
        df, source = await router.fetch_daily_kline("000001")
        assert source == "backup"
        assert primary.calls == 1
        assert backup.calls == 1

    @pytest.mark.asyncio
    async def test_all_failed_raises_with_summary(self):
        primary = FakeSource("primary", fail=True)
        backup = FakeSource("backup", fail=True)
        router = DataSourceRouter([("primary", primary), ("backup", backup)])
        with pytest.raises(AllSourcesFailedError) as exc:
            await router.fetch_daily_kline("000001")
        assert "primary" in exc.value.source_errors
        assert "backup" in exc.value.source_errors
        assert "primary is down" in exc.value.source_errors["primary"]

    @pytest.mark.asyncio
    async def test_circuit_breaker_skips_open_source(self):
        primary = FakeSource("primary", fail=True)
        backup = FakeSource("backup")
        # 禁用缓存，确保每次请求都打到源（熔断行为验证）
        router = DataSourceRouter(
            [("primary", primary), ("backup", backup)],
            {"primary": CircuitBreakerConfig(failure_threshold=2, cooldown_seconds=60)},
            cache_ttl_seconds=0,
        )

        # 第1、2次：primary 失败2次 -> OPEN
        await router.fetch_daily_kline("000001")
        await router.fetch_daily_kline("000001")
        assert router.health_report()[0]["state"] == "open"

        # 第3次：primary 被跳过（熔断），backup 服务
        df, source = await router.fetch_daily_kline("000001")
        assert source == "backup"
        assert primary.calls == 2  # 未再调用

    @pytest.mark.asyncio
    async def test_get_source_for_capability(self):
        primary = FakeSource("primary")
        backup = FakeSource("backup", capabilities={"daily_bars": "available"})
        router = DataSourceRouter([("primary", primary), ("backup", backup)])
        name, adapter = router.get_source_for("daily_bars")
        assert name == "primary"

    @pytest.mark.asyncio
    async def test_get_source_for_unsupported_returns_none(self):
        src = FakeSource("primary", capabilities={"daily_bars": "available"})
        router = DataSourceRouter([("primary", src)])
        assert router.get_source_for("corporate_actions") is None

    @pytest.mark.asyncio
    async def test_health_report_counts(self):
        primary = FakeSource("primary", fail=True)
        backup = FakeSource("backup")
        router = DataSourceRouter([("primary", primary), ("backup", backup)])
        await router.fetch_daily_kline("000001")
        report = router.health_report()
        assert report[0]["failure_count"] == 1
        assert report[1]["success_count"] == 1
        assert report[1]["state"] == "closed"

    @pytest.mark.asyncio
    async def test_reset_clears_breakers(self):
        primary = FakeSource("primary", fail=True)
        router = DataSourceRouter(
            [("primary", primary)],
            {"primary": CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=60)},
        )
        with pytest.raises(AllSourcesFailedError):
            await router.fetch_daily_kline("000001")
        assert router.health_report()[0]["state"] == "open"
        router.reset()
        assert router.health_report()[0]["state"] == "closed"

    @pytest.mark.asyncio
    async def test_fetch_realtime_spot_failover(self):
        primary = FakeSource("primary", fail=True)
        backup = FakeSource("backup")
        router = DataSourceRouter([("primary", primary), ("backup", backup)])
        df, source = await router.fetch_realtime_spot_all()
        assert source == "backup"
        assert "000001" in df["代码"].values


# ---- Tencent / EastMoney adapter parsing tests ----

class TestTencentAdapter:
    def test_code_mapping(self):
        assert TencentAdapter._to_tencent_code("000001") == "sz000001"
        assert TencentAdapter._to_tencent_code("600036") == "sh600036"
        assert TencentAdapter._to_tencent_code("688001") == "sh688001"
        assert TencentAdapter._to_tencent_code("300750") == "sz300750"
        assert TencentAdapter._to_tencent_code("830799") == "bj830799"  # 北交所

    def test_parse_kline_qfq(self):
        data = {
            "code": 0,
            "data": {
                "sz000001": {
                    "qfqkline": [
                        ["2026-07-30", "10.00", "10.80", "10.90", "9.90", "123456"],
                        ["2026-07-31", "10.90", "11.20", "11.30", "10.80", "200000"],
                    ],
                }
            },
        }
        rows = TencentAdapter._parse_kline(data, "sz000001", "day")
        assert len(rows) == 2
        assert rows[0]["close"] == 10.80
        # 手 -> 股 (×100)
        assert rows[0]["volume"] == 12345600

    def test_parse_kline_raw_day(self):
        data = {
            "code": 0,
            "data": {
                "sh600036": {
                    "day": [
                        ["2026-07-30", "35.00", "36.00", "36.20", "34.80", "50000"],
                    ],
                }
            },
        }
        rows = TencentAdapter._parse_kline(data, "sh600036", "day")
        assert len(rows) == 1
        assert rows[0]["open"] == 35.00
        assert rows[0]["volume"] == 5000000

    def test_parse_kline_empty(self):
        assert TencentAdapter._parse_kline({"data": {}}, "sz000001", "day") == []
        assert TencentAdapter._parse_kline({"data": {"sz000001": {}}}, "sz000001", "day") == []

    def test_parse_kline_qfqday_current_structure(self):
        """当前线上接口结构: data -> {code: {qfqday: [...]}}"""
        data = {
            "code": 0,
            "data": {
                "sz000001": {
                    "qfqday": [
                        ["2026-07-30", "10.050", "10.160", "10.180", "9.990", "906890.000"],
                    ],
                    "qt": {}, "mx_price": "10.160", "prec": 2, "version": "2.8.19",
                }
            },
        }
        rows = TencentAdapter._parse_kline(data, "sz000001", "daily", fq="qfq")
        assert len(rows) == 1
        assert rows[0]["date"] == "2026-07-30"
        assert rows[0]["close"] == 10.16
        assert rows[0]["volume"] == 90689000  # 手 ×100 → 股

    def test_parse_realtime_quote_fixture(self):
        """腾讯实时行情 qt.gtimg.cn 响应解析（GBK 文本 → 字段）。

        真实字段布局（split by ~）：
          0=51 1=名称 2=代码 3=现价 4=昨收 5=今开 6=成交量(手) 7=外盘 8=内盘
          9=买一价 10=买一量 ... 19=卖一价 20=卖一量 ... 29=最近逐笔
          30=时间 31=涨跌 32=涨跌% 33=最高 34=最低
          35=价/量/额 36=成交量 37=成交额(万) 38=换手率 39=PE 43=振幅
          44=流通市值(亿) 45=总市值(亿) 46=PB 47=涨停 48=跌停 49=量比 50=均价
        """
        fields: list[str] = []
        # 0-8: 基础
        fields += ["51", "平安银行", "000001", "11.11", "11.25", "11.22",
                   "832344", "334699", "497645"]
        # 9-18: 买一~买五 (价,量)
        fields += ["11.11", "100", "11.10", "200", "11.09", "300",
                   "11.08", "400", "11.07", "500"]
        # 19-28: 卖一~卖五 (价,量)
        fields += ["11.12", "600", "11.13", "700", "11.14", "800",
                   "11.15", "900", "11.16", "1000"]
        # 29-34: 最近逐笔, 时间, 涨跌, 涨跌%, 最高, 最低
        fields += ["", "20260815161900", "-0.14", "-1.24", "11.23", "11.11"]
        # 35-39: 价/量/额, 成交量, 成交额(万), 换手率, PE
        fields += ["11.11/832344/929098438", "832344", "92910", "0.43", "4.96"]
        # 40-45: 空, 最高2, 空, 振幅, 流通市值, 总市值
        fields += ["", "11.11", "1.07", "2155.97", "2156.00", "0.46"]
        # 46-50: PB, 涨停, 跌停, 量比, 均价  — wait, 需对齐索引
        # 修正: 40=? 41=最高 42=? 43=振幅 44=流通市值 45=总市值 46=PB 47=涨停 48=跌停 49=量比 50=均价
        # 上面 40-45 放了 6 个，重新构造：
        fields = fields[:40] + ["", "11.25", "", "1.07", "2155.97", "2156.00",
                                "0.46", "12.38", "10.13", "1.09", "16274"]

        text = 'v_sz000001="' + "~".join(fields) + '";'

        import asyncio
        from unittest.mock import AsyncMock

        adapter = TencentAdapter()
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.content = text.encode("gbk")
        mock_response.raise_for_status = lambda: None  # sync, avoid coroutine warning
        mock_client.get.return_value = mock_response
        adapter._client = mock_client

        quote = asyncio.run(adapter.get_realtime_quote("000001"))
        assert quote["name"] == "平安银行"
        assert quote["price"] == 11.11
        assert quote["pe"] == 4.96
        assert quote["pb"] == 0.46
        assert quote["total_market_cap_billion"] == 2156.00
        assert quote["limit_up_price"] == 12.38
        assert quote["limit_down_price"] == 10.13
        assert quote["volume_ratio"] == 1.09


class TestEastMoneyAdapter:
    def test_secid_mapping(self):
        assert EastMoneyAdapter._to_secid("600036") == "1.600036"   # 沪
        assert EastMoneyAdapter._to_secid("000001") == "0.000001"   # 深
        assert EastMoneyAdapter._to_secid("300750") == "0.300750"   # 创业板
        assert EastMoneyAdapter._to_secid("830799") == "0.830799"   # 北交所

    def test_parse_klines(self):
        data = {
            "data": {
                "klines": [
                    "2026-07-30,10.00,10.80,10.90,9.90,1234567,12345678.00,8.00,8.00,0.80,1.20",
                    "2026-07-31,10.90,11.20,11.30,10.80,2000000,22400000.00,3.70,3.70,0.40,2.10",
                ]
            }
        }
        rows = EastMoneyAdapter._parse_klines(data)
        assert len(rows) == 2
        assert rows[0]["close"] == 10.80
        assert rows[0]["volume"] == 123456700    # 东财 volume 是手 → ×100 转股
        assert rows[0]["amount"] == 12345678.0
        assert rows[1]["pct_change"] == 3.70

    def test_parse_klines_empty(self):
        assert EastMoneyAdapter._parse_klines({}) == []
        assert EastMoneyAdapter._parse_klines({"data": {"klines": None}}) == []


# ---- Scheduler integration with fake router ----

class TestSchedulerRouterIntegration:
    @pytest.mark.asyncio
    async def test_scheduler_uses_router(self):
        import tempfile, os
        from pa_mcp.data.store import DuckDBStore
        from pa_mcp.data.scheduler import DataUpdateScheduler, PhaseStatus

        tmp = tempfile.mkdtemp()
        store = DuckDBStore(os.path.join(tmp, "test.duckdb"))
        store.connect()

        router = DataSourceRouter([("backup", FakeSource("backup"))])
        scheduler = DataUpdateScheduler(store, data_router=router)

        # 只跑必选 phases：calendar + stock_basic + daily_kline
        report = await scheduler.run()

        # 全部 8 个 phase 均已实现 — required phases 必须 SUCCESS
        required = {"1_calendar", "2_stock_basic", "3_daily_kline",
                    "4_minute_kline", "5_financials", "6_capital_flow",
                    "7_dragon_tiger", "8_indicators"}
        for p in report.phases:
            if p.phase_name in required:
                assert p.success, f"{p.phase_name} should succeed: {p.status}"
        # 无未实现阶段
        assert report.blocking_failures == [], report.blocking_failures

        # stock_basic 有 1 行（fake spot）
        count = store.row_count("stock_basic")
        assert count >= 1

        # kline_daily 应有 1 行（fake kline 已按 schema 匹配写入）
        kline_count = store.row_count("kline_daily")
        assert kline_count == 1, f"Expected 1 kline row, got {kline_count}"

        store.close()
