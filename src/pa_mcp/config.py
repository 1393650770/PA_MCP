# [AI:BEGIN]
# PA_MCP - Configuration Management (Pydantic Settings)
# [AI:END]

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---- Sub-settings: use BaseModel (not BaseSettings) to avoid "extra" validation issues ----

class DataSourceSettings(BaseModel):
    """Individual data source configuration."""
    rate_limit_rps: int = Field(default=5, description="Max requests per second")
    retry_count: int = Field(default=3, description="Number of retries on failure")
    timeout_seconds: int = Field(default=30, description="API call timeout")
    burst_max: int = Field(default=10, description="Burst capacity for token bucket")


class CacheSettings(BaseModel):
    """Cache layer configuration."""
    backend: Literal["memory", "redis", "none"] = "memory"
    redis_url: str = "redis://localhost:6379"
    default_ttl: int = 300
    realtime_ttl: int = 3
    kline_ttl: int = 21600


class DatabaseSettings(BaseModel):
    """DuckDB database configuration."""
    path: str = "data/pa_mcp.duckdb"
    read_only: bool = False
    shadow_table_swap: bool = True


class SchedulerSettings(BaseModel):
    """Cron scheduler configuration."""
    update_time: str = "17:30"
    retry_count: int = 3
    retry_backoff: Literal["linear", "exponential"] = "linear"
    precompute_indicators: bool = True
    precompute_signals: bool = True


class SentimentSettings(BaseModel):
    """Sentiment analysis configuration."""
    model: str = "bardsai/finance-sentiment-zh-base"
    batch_size: int = 32
    device: str = "cpu"


class LLMSettings(BaseModel):
    """LLM API configuration for Agent layer."""
    provider: Literal["anthropic", "openai"] = "anthropic"
    model_fast: str = "claude-sonnet-5"
    model_deep: str = "claude-opus-5"
    max_tokens_per_analysis: int = 50000
    temperature: float = 0.1


class RiskSettings(BaseModel):
    """Hard RiskGuard configuration — non-bypassable rules."""
    max_single_stock: float = 0.20
    max_sector_exposure: float = 0.40
    max_total_position: float = 0.80
    max_daily_loss: float = 0.03
    max_consecutive_losses: int = 3
    pause_days_after_big_loss: int = 3


class ServerSettings(BaseModel):
    """MCP server configuration."""
    transport: Literal["stdio", "http"] = "stdio"
    http_port: int = 8080
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


# ---- Root Settings: uses BaseSettings for env var support ----

class Settings(BaseSettings):
    """Root settings aggregating all sub-configs."""

    model_config = SettingsConfigDict(env_prefix="PA_MCP_", env_nested_delimiter="__", extra="allow")

    akshare: DataSourceSettings = Field(default_factory=DataSourceSettings)
    tickflow: DataSourceSettings = Field(default_factory=DataSourceSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    sentiment: SentimentSettings = Field(default_factory=SentimentSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    @classmethod
    def from_yaml(cls, yaml_path: Optional[str] = None) -> "Settings":
        """Load settings from YAML file, with env var overrides."""
        if yaml_path is None:
            yaml_path = os.environ.get("PA_MCP_CONFIG", "config/default.yaml")

        config_data: dict = {}
        yaml_file = Path(yaml_path)
        if yaml_file.exists():
            with open(yaml_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}

        # Flatten data_sources section
        if "data_sources" in config_data:
            ds = config_data.pop("data_sources")
            if "akshare" in ds and isinstance(ds["akshare"], dict):
                config_data["akshare"] = ds["akshare"]
            if "tickflow" in ds and isinstance(ds["tickflow"], dict):
                config_data["tickflow"] = ds["tickflow"]

        return cls(**config_data)


# Global settings instance (lazy init)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_yaml()
    return _settings
