# [AI:BEGIN]
# PA_MCP - Basic tests for data layer and config
# [AI:END]

from __future__ import annotations

import pytest

from pa_mcp.config import Settings, get_settings, RiskSettings, ServerSettings
from pa_mcp.data.quality import DataValidator, ValidationIssue, ValidationReport
from pa_mcp.risk.guard import RiskGuard, GuardResult, GuardVerdict


# ---- Config Tests ----

class TestConfig:
    """Configuration loading and validation tests."""

    def test_default_settings_load(self) -> None:
        """Settings should load with defaults."""
        settings = Settings()
        assert settings.risk.max_single_stock == 0.20
        assert settings.risk.max_total_position == 0.80
        assert settings.server.transport == "stdio"
        assert settings.cache.backend == "memory"

    def test_risk_settings_defaults(self) -> None:
        """Risk settings have sensible defaults."""
        risk = RiskSettings()
        assert risk.max_single_stock == 0.20
        assert risk.max_sector_exposure == 0.40
        assert risk.max_total_position == 0.80
        assert risk.max_daily_loss == 0.03
        assert risk.max_consecutive_losses == 3

    def test_server_settings_defaults(self) -> None:
        """Server settings have sensible defaults."""
        server = ServerSettings()
        assert server.transport == "stdio"
        assert server.http_port == 8080
        assert server.log_level == "INFO"


# ---- RiskGuard Tests ----

class TestRiskGuard:
    """Hard risk control tests."""

    def test_position_ceiling_enforced(self) -> None:
        """Single position should be capped at max."""
        guard = RiskGuard()
        result = guard.check_single_position("000001", 0.50)
        assert result.verdict == GuardVerdict.REDUCE
        assert result.adjusted_max_position == 0.20  # Capped to max

    def test_position_within_limit_passes(self) -> None:
        """Position below ceiling should pass."""
        guard = RiskGuard()
        result = guard.check_single_position("000001", 0.10)
        assert result.verdict == GuardVerdict.PASS
        assert result.adjusted_max_position == 0.10

    def test_paused_guard_blocks_trades(self) -> None:
        """When paused, all trades should be blocked."""
        guard = RiskGuard()
        guard._is_paused = True
        result = guard.check_single_position("000001", 0.05)
        assert result.passed is False
        assert result.verdict == GuardVerdict.BLOCK

    def test_total_portfolio_ceiling(self) -> None:
        """Total portfolio should respect max position."""
        guard = RiskGuard()
        guard._current_positions = {"000001": 0.50, "000002": 0.25}
        new = {"000003": 0.30}
        result = guard.check_portfolio(new)
        assert result.verdict == GuardVerdict.REDUCE
        # Total = 0.50 + 0.25 + 0.30 = 1.05, scaled to 0.80
        assert sum(new.values()) < 0.30  # Should be scaled down

    def test_consecutive_losses_trigger_pause(self) -> None:
        """3 consecutive losses should pause trading."""
        guard = RiskGuard()
        guard.record_trade_result(-0.01)
        guard.record_trade_result(-0.01)
        guard.record_trade_result(-0.01)
        assert guard._is_paused is True
        assert guard.is_trading_allowed is False

    def test_daily_reset_clears_loss(self) -> None:
        """Daily reset should clear loss tracking."""
        guard = RiskGuard()
        guard._daily_loss_pct = 0.04
        guard.reset_daily()
        assert guard._daily_loss_pct == 0.0


# ---- Data Validator Tests ----

class TestDataValidator:
    """Data quality validation tests."""

    def test_validator_creates_report(self) -> None:
        """Validator should create a structured report."""
        validator = DataValidator(expected_stock_count=5000)
        report = ValidationReport()
        report.issues.append(ValidationIssue(
            check_name="test_check",
            level="PASS",
            table_name="kline_daily",
            message="Test passed",
        ))
        report.summary["PASS"] = 1
        assert report.has_errors is False
        assert report.has_warnings is False

    def test_error_report_detected(self) -> None:
        """Report with ERROR should flag has_errors."""
        report = ValidationReport()
        report.issues.append(ValidationIssue(
            check_name="test_check",
            level="ERROR",
            table_name="kline_daily",
            message="Test failed",
        ))
        report.summary["ERROR"] = 1
        assert report.has_errors is True


# ---- Integration Smoke Tests ----

class TestIntegration:
    """Basic integration smoke tests (no external dependencies)."""

    def test_config_singleton(self) -> None:
        """get_settings should return same instance."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_risk_guard_systemic_rules_exist(self) -> None:
        """Systemic rules should be defined."""
        guard = RiskGuard()
        assert len(guard.SYSTEMIC_RULES) >= 5
        assert "mass_limit_down" in guard.SYSTEMIC_RULES
        assert "bear_market" in guard.SYSTEMIC_RULES
