# PA_MCP - 数据层集成测试
# 测试 DuckDBStore 显式列映射、事务、类型校验以及
# DataUpdateScheduler 的 PhaseResult 状态和入口。

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from pa_mcp.data.store import DuckDBStore
from pa_mcp.data.scheduler import PhaseStatus, PhaseResult, PipelineReport


class TestDuckDBStore:
    """测试 DuckDBStore 显式列映射和 Schema 校验。"""

    @pytest.fixture
    def store(self):
        """创建临时数据库。"""
        tmp = tempfile.mkdtemp()
        db_path = str(Path(tmp) / "test.duckdb")
        s = DuckDBStore(db_path)
        s.connect()
        yield s
        s.close()

    def test_table_initialization(self, store):
        """所有 DDL 表创建成功。"""
        for table in ["kline_daily", "stock_basic", "trade_calendar",
                       "dragon_tiger", "signal_cache", "indicator_cache"]:
            assert store.table_exists(table), f"Table {table} should exist"

    def test_get_table_columns(self, store):
        """正确返回表列名。"""
        cols = store._get_table_columns("kline_daily")
        assert "symbol" in cols
        assert "date" in cols
        assert "close" in cols
        assert "volume" in cols
        assert cols[0] == "symbol"  # 第一个应该是symbol
        assert cols[1] == "date"

    def test_insert_with_exact_columns(self, store):
        """列完全匹配时写入成功。"""
        df = pd.DataFrame([{
            "symbol": "000001", "date": "2026-07-30",
            "open": 10.0, "high": 11.0, "low": 9.5, "close": 10.8,
            "volume": 1e6, "amount": 1e7, "amplitude": 5.0,
            "pct_change": 2.5, "change": 0.3, "turnover": 1.2,
            "adjust_factor": 1.0,
        }])
        store.insert_df("kline_daily", df)
        result = store.query_df("SELECT * FROM kline_daily WHERE symbol = '000001'")
        assert len(result) == 1
        assert float(result.iloc[0]["close"]) == 10.8

    def test_insert_with_shuffled_columns(self, store):
        """列顺序打乱时仍正确写入。"""
        df = pd.DataFrame([{
            "close": 10.8, "date": "2026-07-30", "symbol": "000002",
            "open": 10.0, "high": 11.0, "low": 9.5,
            "volume": 1e6, "amount": 1e7, "amplitude": 5.0,
            "pct_change": 2.5, "turnover": 1.2, "change": 0.3,
            "adjust_factor": 1.0,
        }])
        store.insert_df("kline_daily", df)
        result = store.query_df("SELECT * FROM kline_daily WHERE symbol = '000002'")
        assert len(result) == 1
        assert float(result.iloc[0]["close"]) == 10.8
        # 验证symbol在正确位置
        assert result.iloc[0]["symbol"] == "000002"

    def test_insert_with_missing_optional_cols(self, store):
        """缺失可选列时用默认值填充。"""
        df = pd.DataFrame([{
            "symbol": "000003", "date": "2026-07-30",
            "open": 10.0, "high": 11.0, "low": 9.5, "close": 10.8,
            "volume": 1e6, "amount": 1e7,
        }])
        # 缺失 amplitude, pct_change, change, turnover, adjust_factor
        store.insert_df("kline_daily", df)
        result = store.query_df("SELECT * FROM kline_daily WHERE symbol = '000003'")
        assert len(result) == 1
        # 可选字段应该被填充为 None/NaN
        assert pd.isna(result.iloc[0]["amplitude"]) or result.iloc[0]["amplitude"] is None

    def test_insert_with_extra_columns_raises(self, store):
        """额外列应抛出异常。"""
        df = pd.DataFrame([{
            "symbol": "000004", "date": "2026-07-30",
            "open": 10.0, "high": 11.0, "low": 9.5, "close": 10.8,
            "volume": 1e6, "amount": 1e7,
            "amplitude": 5.0, "pct_change": 2.5, "change": 0.3,
            "turnover": 1.2, "adjust_factor": 1.0,
            "unknown_column": "should fail",  # 表中不存在的列
        }])
        with pytest.raises(ValueError, match="unknown_column"):
            store.insert_df("kline_daily", df)

    def test_insert_into_nonexistent_table_raises(self, store):
        """写入不存在的表抛出异常。"""
        df = pd.DataFrame([{"x": 1}])
        with pytest.raises(ValueError, match="does not exist"):
            store.insert_df("nonexistent_table", df)

    def test_insert_empty_df_skips(self, store):
        """空DataFrame不报错但也不写入。"""
        df = pd.DataFrame()
        before = store.row_count("kline_daily")
        store.insert_df("kline_daily", df)
        after = store.row_count("kline_daily")
        assert after == before

    def test_replace_mode_deletes_existing(self, store):
        """Replace模式先删后写。"""
        # 先写入
        df1 = pd.DataFrame([{
            "symbol": "000005", "date": "2026-07-29",
            "open": 10.0, "high": 11.0, "low": 9.5, "close": 10.8,
            "volume": 1e6, "amount": 1e7,
            "amplitude": 5.0, "pct_change": 2.5, "change": 0.3,
            "turnover": 1.2, "adjust_factor": 1.0,
        }])
        store.insert_df("kline_daily", df1, mode="append")
        assert store.row_count("kline_daily") == 1

        # Replace
        df2 = pd.DataFrame([{
            "symbol": "000005", "date": "2026-07-30",
            "open": 11.0, "high": 12.0, "low": 10.5, "close": 11.5,
            "volume": 2e6, "amount": 2e7,
            "amplitude": 5.0, "pct_change": 3.0, "change": 0.5,
            "turnover": 2.0, "adjust_factor": 1.0,
        }])
        store.insert_df("kline_daily", df2, mode="replace")
        result = store.query_df("SELECT * FROM kline_daily WHERE symbol = '000005'")
        assert len(result) == 1
        assert float(result.iloc[0]["close"]) == 11.5

    def test_row_count(self, store):
        """row_count 正确计数。"""
        assert store.row_count("kline_daily") == 0
        df = pd.DataFrame([{
            "symbol": f"00000{i}", "date": "2026-07-30",
            "open": 10.0, "high": 11.0, "low": 9.5, "close": 10.8,
            "volume": 1e6, "amount": 1e7,
            "amplitude": 5.0, "pct_change": 2.5, "change": 0.3,
            "turnover": 1.2, "adjust_factor": 1.0,
        } for i in range(1, 6)])
        store.insert_df("kline_daily", df)
        assert store.row_count("kline_daily") == 5


class TestSchedulerPhaseResult:
    """测试调度器的 PhaseResult 和 PipelineReport 状态机。"""

    def test_success_phase(self):
        p = PhaseResult(phase_name="1_calendar", status=PhaseStatus.SUCCESS, rows_updated=365)
        assert p.success is True
        assert p.is_blocking_failure is False

    def test_skipped_phase(self):
        p = PhaseResult(phase_name="7_dragon_tiger", status=PhaseStatus.SKIPPED_NOT_REQUIRED)
        assert p.success is True  # skipped is not a failure
        assert p.is_blocking_failure is False

    def test_not_implemented_phase(self):
        p = PhaseResult(phase_name="4_minute_kline", status=PhaseStatus.NOT_IMPLEMENTED)
        assert p.success is False
        assert p.is_blocking_failure is True  # not_implemented IS blocking

    def test_failed_phase(self):
        p = PhaseResult(phase_name="1_calendar", status=PhaseStatus.FAILED, error="boom")
        assert p.success is False
        assert p.is_blocking_failure is True

    def test_pipeline_report_all_success(self):
        report = PipelineReport()
        report.phases = [
            PhaseResult("1_calendar", PhaseStatus.SUCCESS, 365),
            PhaseResult("2_stock_basic", PhaseStatus.SUCCESS, 5000),
            PhaseResult("3_daily_kline", PhaseStatus.SUCCESS, 100),
            PhaseResult("4_minute_kline", PhaseStatus.NOT_IMPLEMENTED),
        ]
        assert report.all_success is False
        assert "4_minute_kline" in report.blocking_failures

    def test_pipeline_report_mixed(self):
        report = PipelineReport()
        report.phases = [
            PhaseResult("1_calendar", PhaseStatus.SUCCESS, 365),
            PhaseResult("2_stock_basic", PhaseStatus.SUCCESS, 5000),
            PhaseResult("3_daily_kline", PhaseStatus.SUCCESS, 100),
            PhaseResult("7_dragon_tiger", PhaseStatus.SKIPPED_NOT_REQUIRED),
        ]
        assert report.all_success is True
        assert report.blocking_failures == []
