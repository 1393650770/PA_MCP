# PA_MCP - Execution: Broker Adapters（券商可插拔层）
#
# get_broker(settings) → BrokerPort 实例：
#   - mode=paper（默认）：PaperBroker 纸面交易（零风险，立即可用）
#   - mode=qmt：QmtBroker 实盘（xtquant 惰性导入，enable_live_trading 总闸）
# 所有 adapter 实现 BrokerPort 抽象（幂等/风控不可绕过/断线重连）。

from __future__ import annotations

import logging
from typing import Optional

from pa_mcp.execution.brokers.base import BrokerPort

logger = logging.getLogger(__name__)

_broker: Optional[BrokerPort] = None
_broker_mode: str = ""


def get_broker(settings=None, force: bool = False) -> BrokerPort:
    """按配置构建/复用 BrokerPort 实例（进程内单例）。

    Args:
        settings: Settings（含 broker.mode / broker.qmt）；缺省加载全局
        force: 强制重建（配置变更/测试用）

    Returns:
        BrokerPort：paper（默认）或 qmt；QMT 未配置时自动降级 paper 并告警。
    """
    global _broker, _broker_mode
    if settings is None:
        from pa_mcp.config import get_settings
        settings = get_settings()
    mode = getattr(settings.broker, "mode", "paper")

    if _broker is not None and _broker_mode == mode and not force:
        return _broker

    if mode == "qmt":
        from pa_mcp.execution.brokers.qmt import QmtBroker
        q = settings.broker.qmt
        try:
            _broker = QmtBroker(
                user_data_path=q.user_data_path, account_id=q.account_id,
                enable_live_trading=q.enable_live_trading)
            _broker_mode = "qmt"
            logger.info("broker 模式: qmt（实盘闸=%s）", q.enable_live_trading)
            return _broker
        except Exception as e:  # noqa: BLE001
            logger.warning("QMT 初始化失败，降级纸面: %s", str(e)[:80])

    from pa_mcp.execution.brokers.paper import PaperBroker
    _broker = PaperBroker()
    _broker_mode = "paper"
    logger.info("broker 模式: paper（纸面交易）")
    return _broker


def reset_broker() -> None:
    """重置 broker 单例（测试用）。"""
    global _broker, _broker_mode
    _broker, _broker_mode = None, ""
