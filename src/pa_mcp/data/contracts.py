# [AI:BEGIN]
# PA_MCP - PIT Data Contracts
#
# Point-in-time data model for trustworthy A-share research.
# Every record that could change across history (price, status, financials,
# events, index membership, corporate actions) carries time-bounded validity
# and availability fields. This is the foundation for eliminating survivorship
# bias and look-ahead in strategy evaluation.
#
# Design principle: the "fact" is the raw unadjusted value.
# Adjustments (corporate actions, restatements, sector reclassifications)
# are separate events. Research queries join them as-of a decision_time.
# [AI:END]

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional


# ------- Universal temporal columns -------
# Every mutable fact table should include these (or a subset as appropriate):
#
#   event_time    — when the economic fact happened
#   available_at  — earliest time a strategy could have known about it
#   ingested_at   — when our system actually received it
#   valid_from    — start of the time window where this record is "current"
#   valid_to      — end of that window (None = still current)
#   revision_seq  — version number if the same fact is later revised
#   source        — which provider supplied this record
#   source_id     — provider's own identifier for dedup/trace
#   ingestion_run_id — batch identifier for reproducibility
#   schema_version   — contract version for the record shape
#


# ------- Universe & Securities -------

class SecurityStatus(str, Enum):
    NORMAL = "normal"
    ST = "st"                # Special Treatment (5% limit)
    STAR_ST = "star_st"      # *ST (delisting warning)
    SUSPENDED = "suspended"
    DELISTED = "delisted"
    IPO = "ipo"              # First listing day(s) — no price limit


class Board(str, Enum):
    SH_MAIN = "sh_main"             # 上海主板 (10%)
    SZ_MAIN = "sz_main"             # 深圳主板 (10%)
    CHINEXT = "chinext"             # 创业板 (20%)
    STAR = "star_market"            # 科创板 (20%)
    BSE = "beijing_exchange"        # 北交所 (30%)


@dataclass
class SecurityRecord:
    """One versioned security identity record (SCD2-style)."""
    symbol: str
    name: str
    board: Board
    status: SecurityStatus
    list_date: date
    delist_date: Optional[date] = None
    valid_from: date = field(default_factory=date.today)
    valid_to: Optional[date] = None
    source: str = ""
    source_id: str = ""
    schema_version: int = 1


@dataclass
class UniverseRecord:
    """A stock universe membership at a point in time."""
    universe_name: str         # e.g. "csi300", "csi500", "all_a"
    symbol: str
    effective_date: date       # when membership became effective
    removal_date: Optional[date] = None
    source: str = ""
    schema_version: int = 1


# ------- Market Data -------

@dataclass
class DailyBar:
    """One unadjusted daily bar."""
    symbol: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float              # shares
    amount: float               # yuan
    source: str = ""
    schema_version: int = 1


@dataclass
class MinuteBar:
    """One unadjusted intraday bar."""
    symbol: str
    bar_time: datetime           # UTC, but represents Asia/Shanghai local
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    period_seconds: int = 60    # 60 = 1min, 300 = 5min, etc.
    available_at: Optional[datetime] = None  # bar_end + feed_latency
    source: str = ""
    schema_version: int = 1


# ------- Corporate Actions -------

class CorpActionType(str, Enum):
    DIVIDEND = "dividend"
    SPLIT = "split"
    RIGHTS_ISSUE = "rights_issue"
    BONUS_SHARES = "bonus_shares"
    SPINOFF = "spinoff"
    MERGER = "merger"
    TICKER_CHANGE = "ticker_change"


@dataclass
class CorporateAction:
    """One corporate action event."""
    symbol: str
    action_type: CorpActionType
    ex_date: date              # ex-rights/ex-dividend date
    event_time: datetime       # when the action was effective
    available_at: datetime     # earliest time market knew about it
    # Dividend
    cash_dividend_per_share: float = 0.0
    # Bonus shares / split
    bonus_ratio: float = 0.0   # e.g. 0.5 = 10送5
    # Rights issue
    rights_ratio: float = 0.0
    rights_price: float = 0.0
    # Adjustment factor (cumulative, for computing adjusted prices)
    adjust_factor_daily: float = 1.0
    source: str = ""
    source_id: str = ""
    schema_version: int = 1


# ------- Financial Data -------

@dataclass
class FinancialRecord:
    """One financial statement data point with revision support."""
    symbol: str
    report_period: date        # e.g. 2026-06-30 for Q2 report
    pub_date: date             # when the company announced it
    available_at: datetime     # earliest time strategies can use it
    revision_seq: int = 0      # 0 = original; 1+ = revision
    # Income
    revenue: Optional[float] = None
    operating_profit: Optional[float] = None
    net_profit_parent: Optional[float] = None
    eps: Optional[float] = None
    roe: Optional[float] = None
    # Balance
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    equity_parent: Optional[float] = None
    # Cashflow
    cf_operations: Optional[float] = None
    free_cash_flow: Optional[float] = None
    source: str = ""
    schema_version: int = 1


# ------- Events & Announcements -------

class EventType(str, Enum):
    EARNINGS_PREANNOUNCE = "earnings_preannounce"
    LOCKUP_EXPIRY = "lockup_expiry"
    INSIDER_TRADE = "insider_trade"
    BLOCK_TRADE = "block_trade"
    PLEDGE_CHANGE = "pledge_change"
    INSTITUTIONAL_VISIT = "institutional_visit"
    SHARE_REPURCHASE = "share_repurchase"
    MAJOR_CONTRACT = "major_contract"
    REGULATORY_ACTION = "regulatory_action"


@dataclass
class EventRecord:
    """One corporate event/announcement with PIT availability."""
    symbol: str
    event_type: EventType
    event_time: datetime       # when the event occurred
    announcement_time: datetime  # when publicly announced
    available_at: datetime     # earliest tradable time
    detail: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    source_id: str = ""
    schema_version: int = 1


# ------- Dataset Snapshot -------

@dataclass
class DatasetSnapshot:
    """Immutable reference to a specific data state."""
    snapshot_id: str           # unique hash or UUID
    created_at: datetime
    description: str = ""
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None
    table_hashes: dict[str, str] = field(default_factory=dict)
    ingestion_run_id: str = ""
    source_summary: dict[str, str] = field(default_factory=dict)


# ------- Research Run Manifest (stub for Phase D) -------

@dataclass
class RunManifest:
    """Minimal run manifest for research reproducibility."""
    run_id: str
    git_commit: str = ""
    git_dirty: bool = False
    dataset_snapshot_id: str = ""
    strategy_class: str = ""
    strategy_params: dict[str, Any] = field(default_factory=dict)
    train_start: Optional[date] = None
    train_end: Optional[date] = None
    validation_start: Optional[date] = None
    validation_end: Optional[date] = None
    test_start: Optional[date] = None
    test_end: Optional[date] = None
    seed: int = 42
    python_version: str = ""
    created_at: datetime = field(default_factory=datetime.now)
