# [AI:BEGIN]
# PA_MCP - Research: Style-Matched Benchmarks
#
# Benchmark selection rules:
#   Large-cap pool  → CSI 300 total return (沪深300)
#   Mid-cap pool    → CSI 500 total return (中证500)
#   Small-cap pool  → CSI 1000 total return (中证1000)
#   Mixed pool      → Pre-registered weighted blend
#   Event-driven    → Match style benchmark at event time
#
# The benchmark is chosen BEFORE the run, never after seeing results.
# [AI:END]

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BenchmarkId(str, Enum):
    CSI300 = "csi300"       # 沪深300 (large cap)
    CSI500 = "csi500"       # 中证500 (mid cap)
    CSI1000 = "csi1000"     # 中证1000 (small cap)
    CSI_ALL = "csi_all"     # 中证全指 (total market proxy)
    CUSTOM = "custom"       # User-defined blend


@dataclass
class BenchmarkSpec:
    """Pre-registered benchmark specification."""
    benchmark_id: BenchmarkId
    symbol: str              # Index code for data lookup
    description: str
    weights: dict[BenchmarkId, float] = None  # For custom blends

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = {self.benchmark_id: 1.0}


class BenchmarkRegistry:
    """Maps strategy universe to pre-registered benchmark."""

    # Default style-based mapping
    DEFAULT_MAPPING = {
        "large_cap": BenchmarkSpec(
            BenchmarkId.CSI300, "sh000300",
            "CSI 300 Total Return — large-cap A-share stocks",
        ),
        "mid_cap": BenchmarkSpec(
            BenchmarkId.CSI500, "sh000905",
            "CSI 500 Total Return — mid-cap A-share stocks",
        ),
        "small_cap": BenchmarkSpec(
            BenchmarkId.CSI1000, "sh000852",
            "CSI 1000 Total Return — small-cap A-share stocks",
        ),
        "all_market": BenchmarkSpec(
            BenchmarkId.CSI_ALL, "sh000985",
            "CSI All-Share Total Return",
        ),
    }

    def __init__(self) -> None:
        self._overrides: dict[str, BenchmarkSpec] = {}

    def register(self, name: str, spec: BenchmarkSpec) -> None:
        """Register or override a benchmark mapping."""
        self._overrides[name] = spec

    def get(self, style: str) -> BenchmarkSpec:
        """Get the benchmark for a given style."""
        if style in self._overrides:
            return self._overrides[style]
        if style in self.DEFAULT_MAPPING:
            return self.DEFAULT_MAPPING[style]
        # Fall back to all-market
        return self.DEFAULT_MAPPING["all_market"]

    def get_for_strategy(self, strategy_name: str, declared_style: str = "") -> BenchmarkSpec:
        """Get benchmark for a specific strategy."""
        if declared_style:
            return self.get(declared_style)
        # Default: all-market
        return self.get("all_market")
