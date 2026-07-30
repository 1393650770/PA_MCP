# [AI:BEGIN]
# PA_MCP - Research: Time Series Splits
#
# Walk-forward / expanding window with purge/embargo support.
#
# Purge: overlapping hold periods between adjacent folds are removed.
# Embargo: a gap after the training window prevents lookahead.
# Sealed holdout: the final test window is never seen during tuning.
# [AI:END]

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta
from typing import Iterator, Optional


@dataclass
class Fold:
    """One walk-forward fold."""
    fold_number: int
    train_start: Date
    train_end: Date
    validation_start: Date
    validation_end: Date
    test_start: Optional[Date] = None  # Only populated for final test fold
    test_end: Optional[Date] = None


class TimeSeriesSplitter:
    """Walk-forward time series split with purge and embargo."""

    def __init__(
        self,
        start_date: Date,
        end_date: Date,
        train_years: float = 3.0,
        validation_months: int = 6,
        test_months: int = 6,
        step_months: int = 6,
        embargo_days: int = 5,         # Gap after train window
        purge_days: int = 30,          # Min gap from train end to avoid overlap
        min_train_days: int = 252,     # Minimum training bars per fold
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.train_years = train_years
        self.validation_months = validation_months
        self.test_months = test_months
        self.step_months = step_months
        self.embargo_days = embargo_days
        self.purge_days = purge_days
        self.min_train_days = min_train_days

    def generate_folds(self) -> list[Fold]:
        """Generate walk-forward folds from start to end."""
        folds: list[Fold] = []
        train_delta = timedelta(days=int(self.train_years * 365.25))
        val_delta = timedelta(days=self.validation_months * 30)
        step_delta = timedelta(days=self.step_months * 30)
        embargo_delta = timedelta(days=self.embargo_days)
        purge_delta = timedelta(days=self.purge_days)

        current = self.start_date
        fold_num = 0

        while current + train_delta + val_delta <= self.end_date:
            train_end = current + train_delta - purge_delta
            if train_end - current < timedelta(days=self.min_train_days):
                current += step_delta
                continue

            val_start = train_end + embargo_delta
            val_end = val_start + val_delta

            if val_end > self.end_date:
                break

            folds.append(Fold(
                fold_number=fold_num,
                train_start=current,
                train_end=train_end,
                validation_start=val_start,
                validation_end=val_end,
            ))

            fold_num += 1
            current += step_delta

        # Last fold → test window
        if folds and current < self.end_date:
            last = folds[-1]
            last.test_start = last.validation_end + embargo_delta
            last.test_end = min(last.test_start + timedelta(days=self.test_months * 30), self.end_date)

        return folds

    def get_sealed_holdout(self, folds: list[Fold]) -> Optional[Fold]:
        """Return the final sealed holdout fold (test-only)."""
        for fold in reversed(folds):
            if fold.test_start is not None:
                return fold
        return None
