# [AI:BEGIN]
# PA_MCP - Engine Layer: Strategy Base Class and Registry
# Core abstractions for strategy definition, registration, and signal generation.
# [AI:END]

from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


# ---- Enums ----

class StrategyCategory(str, Enum):
    """Top-level strategy categories (selection dimension)."""
    BOARD = "board"           # Limit-up board strategies
    TREND = "trend"           # Trend following
    SWING = "swing"           # Swing trading
    REVERSAL = "reversal"     # Mean reversion / dip buying
    VALUE = "value"           # Value investing
    EVENT = "event"           # Event-driven
    MOMENTUM = "momentum"     # Momentum-based
    GRID = "grid"             # Grid trading


class SignalDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class MarketState(str, Enum):
    CLIMAX = "climax"         # Fever pitch, high risk
    FERMENTING = "fermenting" # Heating up
    STARTING = "starting"     # Early recovery
    DULL = "dull"             # Low activity
    FROZEN = "frozen"         # Ice age, stay out


class TimeFrame(str, Enum):
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


# ---- Data Classes ----

@dataclass
class ParamRange:
    """Parameter optimization range definition."""
    name: str
    min_val: float
    max_val: float
    step: float = 1.0
    param_type: str = "float"  # float | int | choice

    def grid_values(self) -> list[float]:
        """Generate grid search values."""
        current = self.min_val
        values = []
        while current <= self.max_val:
            values.append(current)
            current += self.step
        return values


@dataclass
class Signal:
    """Strategy output signal."""

    symbol: str
    strategy_name: str
    strength_score: float  # 0-100, higher = stronger bullish signal
    direction: SignalDirection = SignalDirection.NEUTRAL
    evidence: list[dict[str, Any]] = field(default_factory=list)
    risk_flags: list[dict[str, Any]] = field(default_factory=list)
    risk_reward_ratio: Optional[float] = None
    suggested_max_position_pct: float = 0.05  # 5% default
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy_name,
            "strength_score": self.strength_score,
            "direction": self.direction.value,
            "evidence": self.evidence,
            "risk_flags": self.risk_flags,
            "risk_reward_ratio": self.risk_reward_ratio,
            "suggested_max_position_pct": self.suggested_max_position_pct,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


# ---- Base Strategy ----

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses must implement:
    - generate_signals(data, market_state) -> list[Signal]
    - get_params_space() -> list[ParamRange]

    Strategy parameters are defined as class attributes on the subclass
    and can be overridden at instantiation time.

    Example:
        class MACrossStrategy(BaseStrategy):
            name = "ma_crossover"
            category = StrategyCategory.TREND
            description = "MA5 crosses above MA20 with volume confirmation"
            timeframe = TimeFrame.DAILY
            ma_fast: int = 5
            ma_slow: int = 20
            volume_ratio: float = 1.2

            def generate_signals(self, data, market_state=None):
                ...
    """

    # Subclass must set these
    name: str = ""
    category: StrategyCategory = StrategyCategory.TREND
    description: str = ""
    timeframe: TimeFrame = TimeFrame.DAILY

    # Market state suitability (subclass can restrict)
    suitable_states: list[MarketState] = field(default_factory=lambda: list(MarketState))

    def __init__(self, **kwargs: Any) -> None:
        """Initialize strategy with optional param overrides."""
        # Apply overrides from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @abstractmethod
    def generate_signals(
        self, data: pd.DataFrame, market_state: Optional[MarketState] = None,
    ) -> list[Signal]:
        """Generate trading signals from OHLCV data.

        Args:
            data: DataFrame with columns [date, open, high, low, close, volume]
            market_state: Current market regime (for state-aware strategies)

        Returns:
            List of Signal objects (typically one per stock per date)
        """
        ...

    def get_params(self) -> dict[str, Any]:
        """Get current parameter values."""
        params: dict[str, Any] = {}
        for key in dir(self):
            if not key.startswith("_") and key not in (
                "name", "category", "description", "timeframe", "suitable_states",
            ):
                value = getattr(self, key)
                if not callable(value) and not isinstance(value, (classmethod, staticmethod)):
                    params[key] = value
        return params

    def get_params_space(self) -> list[ParamRange]:
        """Get parameter optimization search space. Override in subclasses."""
        return []

    def is_suitable_for(self, market_state: MarketState) -> bool:
        """Check if this strategy is suitable for given market state."""
        if not self.suitable_states:
            return True  # All states by default
        return market_state in self.suitable_states

    def get_info(self) -> dict[str, Any]:
        """Get strategy metadata."""
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "timeframe": self.timeframe.value,
            "suitable_market_states": [s.value for s in self.suitable_states],
            "params": self.get_params(),
        }


# ---- Strategy Registry ----

class StrategyRegistry:
    """Singleton registry for auto-discovering and managing strategies.

    Usage:
        registry = StrategyRegistry()
        registry.auto_discover("pa_mcp.engine.strategies")
        strategy = registry.get("platform_breakout")
        signals = strategy.generate_signals(data)
    """

    _instance: Optional["StrategyRegistry"] = None

    def __new__(cls) -> "StrategyRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry: dict[str, type[BaseStrategy]] = {}
            cls._instance._instances: dict[str, BaseStrategy] = {}
        return cls._instance

    def register(self, strategy_cls: type[BaseStrategy]) -> type[BaseStrategy]:
        """Register a strategy class."""
        if not strategy_cls.name:
            raise ValueError(f"Strategy {strategy_cls.__name__} must define a 'name' class attribute")
        self._registry[strategy_cls.name] = strategy_cls
        logger.debug("Strategy registered", name=strategy_cls.name, category=strategy_cls.category.value)
        return strategy_cls

    def get(self, name: str, **kwargs: Any) -> BaseStrategy:
        """Get or create a strategy instance by name."""
        if name not in self._registry:
            raise KeyError(f"Strategy '{name}' not found. Available: {list(self._registry.keys())}")

        # Cache instances with same params
        cache_key = f"{name}:{str(sorted(kwargs.items()))}"
        if cache_key not in self._instances:
            self._instances[cache_key] = self._registry[name](**kwargs)
        return self._instances[cache_key]

    def list_by_category(self, category: Optional[StrategyCategory] = None) -> list[dict[str, Any]]:
        """List strategies, optionally filtered by category."""
        result = []
        for name, cls in self._registry.items():
            if category and cls.category != category:
                continue
            result.append({
                "name": name,
                "category": cls.category.value,
                "description": cls.description,
                "timeframe": cls.timeframe.value,
            })
        return result

    def list_all(self) -> list[str]:
        """List all registered strategy names."""
        return list(self._registry.keys())

    def auto_discover(self, package_name: str = "pa_mcp.engine.strategies") -> int:
        """Auto-discover and register strategies from a package.

        Returns count of strategies discovered.
        """
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            logger.warning("Strategy package not found", package=package_name)
            return 0

        count = 0
        package_path = getattr(package, "__path__", [])
        for _, module_name, is_pkg in pkgutil.iter_modules(package_path):
            if is_pkg:
                continue  # Skip sub-packages
            try:
                module = importlib.import_module(f"{package_name}.{module_name}")
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseStrategy)
                        and attr is not BaseStrategy
                        and attr.name  # Has a name set
                    ):
                        self.register(attr)
                        count += 1
            except Exception as e:
                logger.warning("Failed to load strategy module", module=module_name, error=str(e))

        logger.info("Auto-discovery complete", strategies_found=count)
        return count

    @property
    def strategy_count(self) -> int:
        return len(self._registry)
