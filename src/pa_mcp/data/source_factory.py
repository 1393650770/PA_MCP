# [AI:BEGIN]
# PA_MCP - Data Layer: 数据源工厂（单一事实源）
#
# server lifespan / scheduler CLI / UI 数据源健康面板 三处此前各自维护
# 一份 source_factory 与 router 构建逻辑（三份重复，新增源要打三处补丁）。
# 本模块统一收口：
#   build_source_factory(existing)  —— 配置名 → 实例工厂注册表
#   build_router(settings, ...)     —— 按配置顺序建链 + 熔断 + 限流
#
# 数据源注册表（配置名 → 工厂）：
#   akshare / sina   —— 传入既有单例（server lifespan 持有），缺省惰性新建
#   tencent/eastmoney/ths —— 类，调用即实例化
# [AI:END]

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def build_source_factory(
    existing: Optional[dict[str, Any]] = None,
) -> dict[str, Callable[[], Any]]:
    """配置名 → 实例工厂（调用返回 adapter 实例）。

    Args:
        existing: 既有单例 {"akshare": ..., "sina": ...}（server lifespan 传入），
            缺省惰性新建（scheduler CLI / UI 健康面板用）。
    """
    existing = existing or {}

    def _lazy(name: str):
        def _make():
            return existing.get(name) or _new_adapter(name)
        return _make

    return {
        "astock": _new_astock,
        "akshare": _lazy("akshare"),
        "sina": _lazy("sina"),
        "tencent": _new_tencent,
        "eastmoney": _new_eastmoney,
        "ths": _new_ths,
    }


def _new_adapter(name: str):
    """惰性新建 akshare/sina 单例（模块级缓存，避免重复实例化）。"""
    if name == "akshare":
        from pa_mcp.data.sources.akshare_adapter import AKShareAdapter
        _new_adapter.cache[name] = AKShareAdapter()
    elif name == "sina":
        from pa_mcp.data.sources.sina_adapter import SinaAdapter
        _new_adapter.cache[name] = SinaAdapter()
    return _new_adapter.cache.get(name)


_new_adapter.cache: dict[str, Any] = {}


def _new_astock():
    from pa_mcp.data.sources.astock_adapter import AstockAdapter
    return AstockAdapter()


def _new_tencent():
    from pa_mcp.data.sources.tencent_adapter import TencentAdapter
    return TencentAdapter()


def _new_eastmoney():
    from pa_mcp.data.sources.eastmoney_adapter import EastMoneyAdapter
    return EastMoneyAdapter()


def _new_ths():
    from pa_mcp.data.sources.ths_adapter import ThsAdapter
    return ThsAdapter()


def build_router(settings, min_source_interval: Optional[dict] = None,
                 existing: Optional[dict[str, Any]] = None):
    """按 settings.router.sources 构建多源容灾路由。

    与既有逻辑一致：未知源 warn、实例化失败跳过、链空返回 None（调用方降级）。

    Args:
        settings: Settings（含 router.sources / router.circuit）
        min_source_interval: {源名: 最小请求间隔秒}（如 eastmoney 1.2s 防封）
        existing: 既有单例（server lifespan 的 _akshare/_sina）
    """
    from pa_mcp.data.router import DataSourceRouter, CircuitBreakerConfig

    factory = build_source_factory(existing)
    chain: list[tuple[str, Any]] = []
    for name in settings.router.sources:
        maker = factory.get(name)
        if maker is None:
            logger.warning("Unknown data source in config", source=name)
            continue
        try:
            chain.append((name, maker()))
        except Exception as e:
            logger.warning("Failed to init data source", source=name,
                           error=str(e))

    if not chain:
        logger.warning("No data sources in chain — router disabled")
        return None

    breaker_cfg = CircuitBreakerConfig(
        failure_threshold=settings.router.circuit.failure_threshold,
        cooldown_seconds=settings.router.circuit.cooldown_seconds,
    )
    router = DataSourceRouter(
        chain,
        {name: breaker_cfg for name, _ in chain},
        min_source_interval=min_source_interval,
    )
    logger.info("Data source router ready",
                chain=[n for n, _ in chain],
                throttled=list((min_source_interval or {}).keys()))
    return router
