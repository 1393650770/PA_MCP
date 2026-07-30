# [AI:BEGIN]
# PA_MCP - Research: Recorder & Run Manifest
#
# Qlib-style experiment tracking. Every research run logs an immutable
# manifest and all output artifacts. This enables:
#   1. Reproducibility (same snapshot + seed → same result)
#   2. Walk-forward governance (test window is never used for tuning)
#   3. Strategy promotion gates (OOS alpha, parameter stability, cost sensitivity)
# [AI:END]

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4


@dataclass
class RunManifest:
    """Immutable record of one research run."""
    run_id: str = field(default_factory=lambda: uuid4().hex[:16])
    strategy_class: str = ""
    strategy_params: dict[str, Any] = field(default_factory=dict)
    strategy_version: str = ""
    dataset_snapshot_id: str = ""
    feature_set_hash: str = ""
    train_start: Optional[Date] = None
    train_end: Optional[Date] = None
    validation_start: Optional[Date] = None
    validation_end: Optional[Date] = None
    test_start: Optional[Date] = None
    test_end: Optional[Date] = None
    seed: int = 42
    git_commit: str = ""
    git_dirty: bool = False
    python_version: str = ""
    initial_cash: float = 100_000.0
    fee_model: str = "default_v1"
    slippage_model: str = "conservative_v1"
    benchmark: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "created"  # created → running → completed/failed
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Output of one research run."""
    manifest: RunManifest
    nav_series: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    attribution: dict[str, Any] = field(default_factory=dict)
    data_hashes: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "run_id": self.manifest.run_id,
            "strategy": self.manifest.strategy_class,
            "params": self.manifest.strategy_params,
            "metrics": self.metrics,
            "n_trades": len(self.trades),
            "train_period": f"{self.manifest.train_start} → {self.manifest.train_end}",
            "test_period": f"{self.manifest.test_start} → {self.manifest.test_end}",
        }, indent=2, default=str)


class RunRecorder:
    """Records research runs to disk for reproducibility."""

    def __init__(self, base_dir: str = "data/research_runs") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._runs: list[RunResult] = []

    def start_run(self, manifest: RunManifest) -> RunManifest:
        """Register a new run."""
        manifest.status = "running"

        # Auto-detect git info
        if not manifest.git_commit:
            try:
                import subprocess
                r = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True, text=True, timeout=5,
                )
                if r.returncode == 0:
                    manifest.git_commit = r.stdout.strip()[:8]
                    r2 = subprocess.run(
                        ["git", "status", "--porcelain"],
                        capture_output=True, text=True, timeout=5,
                    )
                    manifest.git_dirty = bool(r2.stdout.strip())
            except Exception:
                pass

        if not manifest.python_version:
            import sys
            manifest.python_version = sys.version.split()[0]

        return manifest

    def complete_run(self, result: RunResult) -> None:
        """Record a completed run."""
        result.manifest.status = "completed"
        self._runs.append(result)

        # Save to disk
        run_dir = self.base_dir / result.manifest.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "manifest.json", "w") as f:
            f.write(result.to_json())

        with open(run_dir / "nav.json", "w") as f:
            json.dump(result.nav_series, f)

        with open(run_dir / "trades.json", "w") as f:
            json.dump(result.trades, f, default=str)

    def list_runs(self, strategy_filter: str = "") -> list[RunResult]:
        """List completed runs, optionally filtered by strategy."""
        if strategy_filter:
            return [r for r in self._runs
                    if strategy_filter in r.manifest.strategy_class]
        return list(self._runs)

    @staticmethod
    def hash_dataframe(df) -> str:
        """Content hash of a DataFrame for tracking."""
        return hashlib.md5(
            df.to_csv(index=False).encode()
        ).hexdigest()[:16]
