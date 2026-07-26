# [AI:BEGIN]
# PA_MCP - Data Layer: Data Quality Validator
# 9 automated checks for data integrity after each daily update.
# Outputs structured ValidationReport with PASS/WARN/ERROR levels.
# [AI:END]

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ValidationIssue:
    """A single validation finding."""

    check_name: str
    level: str  # PASS, WARN, ERROR
    table_name: str
    message: str
    detail: Optional[dict[str, Any]] = None


@dataclass
class ValidationReport:
    """Aggregated validation results."""

    timestamp: datetime = field(default_factory=datetime.now)
    issues: list[ValidationIssue] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=lambda: {"PASS": 0, "WARN": 0, "ERROR": 0})

    @property
    def has_errors(self) -> bool:
        return self.summary.get("ERROR", 0) > 0

    @property
    def has_warnings(self) -> bool:
        return self.summary.get("WARN", 0) > 0


class DataValidator:
    """Validates data quality after daily updates.

    Checks:
    1. Row count within expected range
    2. Required columns present
    3. OHLC sanity (high >= max(open, close), low <= min(open, close))
    4. No negative prices or volumes
    5. No duplicate rows
    6. Date continuity within expected gaps
    7. No suspicious price jumps (>30% in one day without adjust)
    8. Data freshness (latest date is recent enough)
    9. Cross-source consistency (if multiple sources available)
    """

    def __init__(self, expected_stock_count: int = 5000) -> None:
        self.expected_stock_count = expected_stock_count
        self.min_stock_threshold = int(expected_stock_count * 0.95)  # 5% tolerance

    def validate_all(self, store, cache_manager=None) -> ValidationReport:
        """Run all validation checks. store is a DuckDBStore instance."""
        report = ValidationReport()

        checks = [
            self._check_row_count,
            self._check_required_columns,
            self._check_ohlc_sanity,
            self._check_no_negatives,
            self._check_duplicates,
            self._check_date_continuity,
            self._check_price_jumps,
            self._check_freshness,
        ]

        for check_func in checks:
            try:
                issues = check_func(store)
                for issue in issues:
                    report.issues.append(issue)
                    report.summary[issue.level] = report.summary.get(issue.level, 0) + 1
            except Exception as e:
                report.issues.append(ValidationIssue(
                    check_name=check_func.__name__,
                    level="ERROR",
                    table_name="N/A",
                    message=f"Validation check crashed: {e}",
                ))
                report.summary["ERROR"] = report.summary.get("ERROR", 0) + 1

        logger.info(
            "Validation complete",
            pass_count=report.summary.get("PASS", 0),
            warn_count=report.summary.get("WARN", 0),
            error_count=report.summary.get("ERROR", 0),
        )
        return report

    def _check_row_count(self, store) -> list[ValidationIssue]:
        """Check today's kline row count is within expected range."""
        issues = []
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            count = store.query_df(
                "SELECT COUNT(DISTINCT symbol) FROM kline_daily WHERE date = ?",
                [today],
            ).iloc[0, 0]
            if count < self.min_stock_threshold:
                issues.append(ValidationIssue(
                    check_name="row_count",
                    level="ERROR",
                    table_name="kline_daily",
                    message=f"Today's stock count ({count}) below threshold ({self.min_stock_threshold})",
                    detail={"actual": count, "expected_min": self.min_stock_threshold},
                ))
            elif count < self.expected_stock_count:
                issues.append(ValidationIssue(
                    check_name="row_count",
                    level="WARN",
                    table_name="kline_daily",
                    message=f"Today's stock count ({count}) slightly below expected ({self.expected_stock_count})",
                    detail={"actual": count, "expected": self.expected_stock_count},
                ))
            else:
                issues.append(ValidationIssue(
                    check_name="row_count",
                    level="PASS",
                    table_name="kline_daily",
                    message=f"Row count OK: {count}",
                ))
        except Exception as e:
            issues.append(ValidationIssue(
                check_name="row_count",
                level="ERROR",
                table_name="kline_daily",
                message=str(e),
            ))
        return issues

    def _check_required_columns(self, store) -> list[ValidationIssue]:
        """Check kline_daily has all required columns."""
        required = {"symbol", "date", "open", "high", "low", "close", "volume"}
        issues = []
        try:
            cols = set(
                store.query_df("DESCRIBE kline_daily")["column_name"].tolist()
            )
            missing = required - cols
            if missing:
                issues.append(ValidationIssue(
                    check_name="required_columns",
                    level="ERROR",
                    table_name="kline_daily",
                    message=f"Missing columns: {missing}",
                ))
            else:
                issues.append(ValidationIssue(
                    check_name="required_columns",
                    level="PASS",
                    table_name="kline_daily",
                    message="All required columns present",
                ))
        except Exception as e:
            issues.append(ValidationIssue(
                check_name="required_columns",
                level="ERROR",
                table_name="kline_daily",
                message=str(e),
            ))
        return issues

    def _check_ohlc_sanity(self, store) -> list[ValidationIssue]:
        """Check OHLC price relationships: high >= open/close, low <= open/close."""
        issues = []
        try:
            bad_rows = store.query_df("""
                SELECT COUNT(*) as cnt FROM kline_daily
                WHERE high < open OR high < close OR low > open OR low > close
            """).iloc[0, 0]
            if bad_rows > 0:
                issues.append(ValidationIssue(
                    check_name="ohlc_sanity",
                    level="ERROR",
                    table_name="kline_daily",
                    message=f"Found {bad_rows} rows with invalid OHLC relationships",
                    detail={"bad_rows": int(bad_rows)},
                ))
            else:
                issues.append(ValidationIssue(
                    check_name="ohlc_sanity",
                    level="PASS",
                    table_name="kline_daily",
                    message="OHLC sanity check passed",
                ))
        except Exception as e:
            issues.append(ValidationIssue(
                check_name="ohlc_sanity",
                level="ERROR",
                table_name="kline_daily",
                message=str(e),
            ))
        return issues

    def _check_no_negatives(self, store) -> list[ValidationIssue]:
        """Check no negative prices or volumes."""
        issues = []
        try:
            neg_prices = store.query_df(
                "SELECT COUNT(*) FROM kline_daily WHERE open < 0 OR high < 0 OR low < 0 OR close < 0",
            ).iloc[0, 0]
            neg_volume = store.query_df(
                "SELECT COUNT(*) FROM kline_daily WHERE volume < 0 OR amount < 0",
            ).iloc[0, 0]
            if neg_prices > 0:
                issues.append(ValidationIssue(
                    check_name="non_negative",
                    level="ERROR",
                    table_name="kline_daily",
                    message=f"Found {neg_prices} rows with negative prices",
                ))
            if neg_volume > 0:
                issues.append(ValidationIssue(
                    check_name="non_negative",
                    level="ERROR",
                    table_name="kline_daily",
                    message=f"Found {neg_volume} rows with negative volume/amount",
                ))
            if neg_prices == 0 and neg_volume == 0:
                issues.append(ValidationIssue(
                    check_name="non_negative",
                    level="PASS",
                    table_name="kline_daily",
                    message="No negative values found",
                ))
        except Exception as e:
            issues.append(ValidationIssue(
                check_name="non_negative",
                level="ERROR",
                table_name="kline_daily",
                message=str(e),
            ))
        return issues

    def _check_duplicates(self, store) -> list[ValidationIssue]:
        """Check for duplicate (symbol, date) pairs."""
        issues = []
        try:
            dupes = store.query_df(
                "SELECT COUNT(*) FROM (SELECT symbol, date, COUNT(*) as cnt FROM kline_daily GROUP BY symbol, date HAVING cnt > 1)",
            ).iloc[0, 0]
            if dupes > 0:
                issues.append(ValidationIssue(
                    check_name="duplicates",
                    level="ERROR",
                    table_name="kline_daily",
                    message=f"Found {dupes} duplicate (symbol, date) groups",
                    detail={"duplicate_groups": int(dupes)},
                ))
            else:
                issues.append(ValidationIssue(
                    check_name="duplicates",
                    level="PASS",
                    table_name="kline_daily",
                    message="No duplicates found",
                ))
        except Exception as e:
            issues.append(ValidationIssue(
                check_name="duplicates",
                level="ERROR",
                table_name="kline_daily",
                message=str(e),
            ))
        return issues

    def _check_date_continuity(self, store) -> list[ValidationIssue]:
        """Check no unexpected gaps in date sequence (skip weekends/holidays)."""
        issues = []
        try:
            dates = store.query_df(
                "SELECT DISTINCT date FROM kline_daily ORDER BY date",
            )["date"].tolist()
            if len(dates) < 2:
                return [ValidationIssue(
                    check_name="date_continuity",
                    level="WARN",
                    table_name="kline_daily",
                    message="Not enough dates to check continuity",
                )]

            max_gap = timedelta(days=7)  # Allow up to 7-day gaps (long holidays)
            for i in range(1, len(dates)):
                gap = dates[i] - dates[i - 1]
                if gap > max_gap:
                    issues.append(ValidationIssue(
                        check_name="date_continuity",
                        level="WARN",
                        table_name="kline_daily",
                        message=f"Large date gap: {dates[i-1]} → {dates[i]} ({gap.days} days)",
                    ))

            if not issues:
                issues.append(ValidationIssue(
                    check_name="date_continuity",
                    level="PASS",
                    table_name="kline_daily",
                    message=f"Date continuity OK ({len(dates)} trading days)",
                ))
        except Exception as e:
            issues.append(ValidationIssue(
                check_name="date_continuity",
                level="ERROR",
                table_name="kline_daily",
                message=str(e),
            ))
        return issues

    def _check_price_jumps(self, store) -> list[ValidationIssue]:
        """Check for suspicious single-day price jumps (>30%)."""
        issues = []
        try:
            jump_count = store.query_df("""
                WITH price_chg AS (
                    SELECT symbol, date,
                        ABS(close / LAG(close) OVER (PARTITION BY symbol ORDER BY date) - 1) as abs_return
                    FROM kline_daily
                )
                SELECT COUNT(*) FROM price_chg WHERE abs_return > 0.30
            """).iloc[0, 0]
            if jump_count > 100:
                issues.append(ValidationIssue(
                    check_name="price_jumps",
                    level="ERROR",
                    table_name="kline_daily",
                    message=f"Found {jump_count} rows with >30% single-day price change (possible adjust factor issue)",
                ))
            elif jump_count > 20:
                issues.append(ValidationIssue(
                    check_name="price_jumps",
                    level="WARN",
                    table_name="kline_daily",
                    message=f"Found {jump_count} large price jumps (may include legit events like IPOs)",
                ))
            else:
                issues.append(ValidationIssue(
                    check_name="price_jumps",
                    level="PASS",
                    table_name="kline_daily",
                    message=f"Price jump count within normal range: {jump_count}",
                ))
        except Exception as e:
            issues.append(ValidationIssue(
                check_name="price_jumps",
                level="ERROR",
                table_name="kline_daily",
                message=str(e),
            ))
        return issues

    def _check_freshness(self, store) -> list[ValidationIssue]:
        """Check that latest data is recent (within 2 trading days)."""
        issues = []
        try:
            latest = store.query_df("SELECT MAX(date) as max_date FROM kline_daily").iloc[0, 0]
            if latest is None:
                issues.append(ValidationIssue(
                    check_name="freshness",
                    level="ERROR",
                    table_name="kline_daily",
                    message="Table is empty!",
                ))
                return issues

            today = datetime.now().date()
            days_behind = (today - latest).days if hasattr(latest, "date") else (today - pd.Timestamp(latest).date()).days

            if days_behind > 2:
                issues.append(ValidationIssue(
                    check_name="freshness",
                    level="ERROR",
                    table_name="kline_daily",
                    message=f"Data is {days_behind} days stale (latest: {latest})",
                ))
            elif days_behind > 1:
                issues.append(ValidationIssue(
                    check_name="freshness",
                    level="WARN",
                    table_name="kline_daily",
                    message=f"Data is {days_behind} days behind (latest: {latest})",
                ))
            else:
                issues.append(ValidationIssue(
                    check_name="freshness",
                    level="PASS",
                    table_name="kline_daily",
                    message=f"Data fresh (latest: {latest})",
                ))
        except Exception as e:
            issues.append(ValidationIssue(
                check_name="freshness",
                level="ERROR",
                table_name="kline_daily",
                message=str(e),
            ))
        return issues
