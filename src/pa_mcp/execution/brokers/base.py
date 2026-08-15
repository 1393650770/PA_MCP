# [AI:BEGIN]
# PA_MCP - Execution: Broker Port (券商抽象接口)
#
# 券商接入的统一端口。QMT/miniQMT 券商确认后实现 QMTBrokerAdapter。
# 设计原则（与 plan 一致）：
#   1. LLM 进程不持有券商凭据、不引用 BrokerPort
#   2. 订单必须携带 RiskDecision ID（风控不可绕过）
#   3. client_order_id 幂等（重复请求不产生第二笔订单）
#   4. 支持断线重连、迟到成交、日终对账
#   5. 先 shadow/paper，再 canary，再实盘
# [AI:END]

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime
from enum import Enum
from typing import Optional


class BrokerOrderStatus(str, Enum):
    PENDING = "pending"            # 已提交，未确认
    SUBMITTED = "submitted"        # 券商已受理
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class BrokerOrder:
    """提交给券商的订单。"""
    client_order_id: str            # 幂等键（全局唯一）
    symbol: str
    side: str                       # buy | sell
    quantity: int
    limit_price: Optional[float] = None
    risk_decision_id: str = ""      # RiskGuard 审批 ID（必填）
    strategy_name: str = ""
    status: BrokerOrderStatus = BrokerOrderStatus.PENDING
    broker_order_id: str = ""       # 券商返回的订单号
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BrokerFill:
    """券商成交回报。"""
    broker_fill_id: str
    client_order_id: str
    broker_order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    commission: float = 0.0
    trade_time: str = ""


@dataclass
class BrokerPosition:
    symbol: str
    quantity: int
    available_quantity: int        # T+1 可卖
    avg_cost: float = 0.0


@dataclass
class BrokerAccount:
    account_id: str
    cash: float = 0.0
    frozen_cash: float = 0.0
    positions: list[BrokerPosition] = field(default_factory=list)


class BrokerPort(abc.ABC):
    """券商抽象端口 — 所有券商 adapter 实现此接口。"""

    @property
    @abc.abstractmethod
    def broker_name(self) -> str:
        """券商名称（如 'qmt', 'xtp', 'ptrade'）。"""

    @abc.abstractmethod
    async def connect(self) -> None:
        """建立连接（含认证）。"""

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """断开连接。"""

    @abc.abstractmethod
    async def submit_order(self, order: BrokerOrder) -> BrokerOrder:
        """提交订单（幂等：同 client_order_id 不重复下单）。"""

    @abc.abstractmethod
    async def cancel_order(self, client_order_id: str) -> bool:
        """撤单。"""

    @abc.abstractmethod
    async def query_order(self, client_order_id: str) -> BrokerOrder:
        """查询订单状态（超时重查用，避免重复下单）。"""

    @abc.abstractmethod
    async def get_account(self) -> BrokerAccount:
        """查询账户：现金/持仓/可卖。"""

    @abc.abstractmethod
    async def get_fills(self, since: Optional[datetime] = None) -> list[BrokerFill]:
        """获取成交回报（断线补单用）。"""


class QMTBrokerAdapter(BrokerPort):
    """QMT/miniQMT 券商适配器（骨架）。

    券商确认后填充实现：
      - 连接方式：miniQMT xtquant 库（券商标配 Python 接口）
      - 下单：xt_trader.order_stock / passorder
      - 查询：xt_trader.query_stock_orders / query_stock_positions
      - 凭据：账号/密码经密钥管理注入（不落盘）
    """

    broker_name = "qmt"

    def __init__(self, account_id: str = "", account_type: str = "STOCK") -> None:
        self.account_id = account_id
        self.account_type = account_type
        self._xt_trader = None
        self._connected = False

    async def connect(self) -> None:
        """连接 QMT（骨架 — 券商确认后实现 xtquant 初始化）。

        券商确认后实现：
          from xtquant import xttrader, xtconstant
          self._xt_trader = xttrader.XtQuantTrader(path, session_id)
          self._xt_trader.start()
          self._xt_trader.connect()
        """
        raise NotImplementedError(
            "QMT adapter 骨架：请在确认券商后实现连接逻辑。"
            "参考：xtquant 的 XtQuantTrader 初始化流程。"
        )

    async def disconnect(self) -> None:
        if self._xt_trader is not None:
            self._xt_trader.stop()
            self._xt_trader = None
        self._connected = False

    async def submit_order(self, order: BrokerOrder) -> BrokerOrder:
        if not order.risk_decision_id:
            raise ValueError("订单缺少 RiskDecision ID — 风控不可绕过")
        raise NotImplementedError("QMT 下单待券商确认后实现")

    async def cancel_order(self, client_order_id: str) -> bool:
        raise NotImplementedError("QMT 撤单待券商确认后实现")

    async def query_order(self, client_order_id: str) -> BrokerOrder:
        raise NotImplementedError("QMT 查询待券商确认后实现")

    async def get_account(self) -> BrokerAccount:
        raise NotImplementedError("QMT 账户查询待券商确认后实现")

    async def get_fills(self, since: Optional[datetime] = None) -> list[BrokerFill]:
        raise NotImplementedError("QMT 成交回报待券商确认后实现")
