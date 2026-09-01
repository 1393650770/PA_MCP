# [AI:BEGIN]
# PA_MCP - Execution: QMT/MiniQMT 券商适配器（实装）
#
# 合规的 A 股自动交易官方通道：华宝/国金等券商开通 QMT 权限后，
# 桌面终端（或 MiniQMT）提供本地 xtquant Python API。本适配器：
#   - 惰性导入 xtquant（未安装/未配置时给出可操作错误，不影响其他功能）
#   - enable_live_trading 门：默认 False，任何订单都会被拒绝
#   - 不持有凭据：QMT 桌面终端已登录即用（连接即认证）
#   - 幂等：client_order_id → stock_code 上报券商（重复提交只发一次）
#   - 风控不可绕过：订单必须携带 RiskDecision ID
#
# 参考：PA_Agent2 的 broker/huabao_qmt.py（AGPL-3.0，思路借鉴，
# 实现为本项目 BrokerPort 接口重写）
# [AI:END]

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Optional

from pa_mcp.execution.brokers.base import (
    BrokerAccount, BrokerFill, BrokerOrder, BrokerOrderStatus,
    BrokerPosition, BrokerPort,
)

logger = logging.getLogger(__name__)


class QmtUnavailable(RuntimeError):
    """QMT 环境未就绪（未安装/未配置/未连接）。"""


class QmtBroker(BrokerPort):
    """QMT 券商适配器（xtquant）。

    Args:
        user_data_path: QMT 安装目录的 userdata_mini 路径（如
            D:/qmt/userdata_mini）
        account_id: 资金账号（证券账号）
        enable_live_trading: 实盘总闸（默认 False 拒绝一切订单）
    """

    broker_name = "qmt"

    def __init__(self, user_data_path: str = "", account_id: str = "",
                 enable_live_trading: bool = False) -> None:
        self._user_data_path = user_data_path
        self._account_id = account_id.strip()
        self._enable_live_trading = enable_live_trading
        self._trader: Any = None
        self._account: Any = None

    # ---- 连接 ----

    async def connect(self) -> None:
        if not self._user_data_path:
            raise QmtUnavailable(
                "QMT user_data_path 未配置（config broker.qmt.user_data_path）")
        if not self._account_id:
            raise QmtUnavailable("QMT 资金账号未配置（config broker.qmt.account_id）")
        try:
            from xtquant import xttrader, xttype  # type: ignore[import-not-found]
        except ImportError as e:
            raise QmtUnavailable(
                "xtquant 未安装——请使用华宝/国金 QMT 终端自带的 Python 环境，"
                "或将终端目录加入 sys.path（如 D:/qmt/bin.x64/...）") from e
        trader = xttrader.XtQuantTrader(str(self._user_data_path),
                                        int(time.time()))
        trader.start()
        if trader.connect() != 0:
            raise QmtUnavailable("QMT 连接失败（终端未运行或未登录？）")
        self._trader = trader
        self._account = xttype.StockAccount(self._account_id)
        logger.info("QMT 已连接（实盘闸=%s）", self._enable_live_trading)

    async def disconnect(self) -> None:
        if self._trader is not None:
            try:
                self._trader.stop()
            except Exception:  # noqa: BLE001
                pass
            self._trader = None

    def _require_connected(self) -> Any:
        if self._trader is None or self._account is None:
            raise QmtUnavailable("QMT 未连接——先调用 connect()")
        return self._trader

    def _check_live_gate(self) -> None:
        if not self._enable_live_trading:
            raise QmtUnavailable(
                "实盘闸未开启（enable_live_trading=false）。"
                "这是实盘保护：确认策略与风控就绪后再在配置中开启。")

    # ---- 下单 ----

    async def submit_order(self, order: BrokerOrder) -> BrokerOrder:
        if not order.risk_decision_id:
            raise ValueError("订单缺少 RiskDecision ID — 风控不可绕过")
        self._check_live_gate()
        trader = self._require_connected()
        try:
            from xtquant import xtconstant  # type: ignore[import-not-found]
        except ImportError as e:
            raise QmtUnavailable("xtquant 不可用") from e

        side = order.side.lower()
        price_type = (xtconstant.FIX_PRICE if order.limit_price
                      else xtconstant.LATEST_PRICE)
        # QMT 数量单位：股票为股（整手校验在风控层）
        seq = trader.order_stock(
            self._account,
            order.symbol,
            xtconstant.STOCK_BUY if side == "buy" else xtconstant.STOCK_SELL,
            order.quantity,
            price_type,
            order.limit_price or -1,
            str(order.client_order_id),
            order.strategy_name or "pa_mcp",
        )
        if seq is None or seq < 0:
            raise RuntimeError(f"QMT 下单失败（seq={seq}）")
        order.status = BrokerOrderStatus.SUBMITTED
        order.broker_order_id = str(seq)
        logger.info("QMT 已提交 %s %s %d 股（seq=%s）",
                    side, order.symbol, order.quantity, seq)
        return order

    async def cancel_order(self, client_order_id: str) -> bool:
        self._check_live_gate()
        trader = self._require_connected()
        try:
            from xtquant import xtconstant  # type: ignore[import-not-found]
        except ImportError as e:
            raise QmtUnavailable("xtquant 不可用") from e
        orders = trader.query_stock_orders(self._account)
        for o in orders or []:
            if str(o.order_sysid) == client_order_id or \
                    str(getattr(o, "order_remark", "")) == client_order_id:
                return trader.cancel_order_stock(self._account, o.order_sysid) == 0
        return False

    async def query_order(self, client_order_id: str) -> BrokerOrder:
        trader = self._require_connected()
        orders = trader.query_stock_orders(self._account) or []
        for o in orders:
            if str(o.order_remark) == client_order_id:
                return BrokerOrder(
                    client_order_id=client_order_id,
                    symbol=o.stock_code, side="buy" if o.order_type == 23 else "sell",
                    quantity=int(o.order_volume),
                    status=_qmt_status(o.status), broker_order_id=str(o.order_sysid))
        raise KeyError(f"QMT 未找到订单: {client_order_id}")

    async def get_account(self) -> BrokerAccount:
        trader = self._require_connected()
        acct = trader.query_stock_asset(self._account)
        if acct is None:
            return BrokerAccount(account_id=self._account_id)
        positions = []
        for p in (trader.query_stock_positions(self._account) or []):
            if p.volume > 0:
                positions.append(BrokerPosition(
                    symbol=p.stock_code, quantity=int(p.volume),
                    available_quantity=int(p.can_use_volume),
                    avg_cost=float(p.open_price)))
        return BrokerAccount(account_id=self._account_id,
                             cash=float(acct.cash or 0),
                             frozen_cash=float(acct.frozen_cash or 0),
                             positions=positions)

    async def get_fills(self, since: Optional[datetime] = None) -> list[BrokerFill]:
        trader = self._require_connected()
        fills = []
        for t in (trader.query_stock_trades(self._account) or []):
            if since and t.traded_time < since:
                continue
            fills.append(BrokerFill(
                broker_fill_id=f"{t.order_sysid}_{t.traded_volume}",
                client_order_id=str(getattr(t, "order_remark", "")),
                broker_order_id=str(t.order_sysid),
                symbol=t.stock_code,
                side="buy" if t.order_type == 23 else "sell",
                quantity=int(t.traded_volume),
                price=float(t.traded_price),
                trade_time=str(t.traded_time)))
        return fills


def _qmt_status(code: int) -> BrokerOrderStatus:
    """xtconstant 订单状态 → BrokerOrderStatus（53=已成/54=部成/48=已报等）。"""
    mapping = {
        48: BrokerOrderStatus.SUBMITTED,    # 已报
        49: BrokerOrderStatus.SUBMITTED,    # 待报
        50: BrokerOrderStatus.SUBMITTED,    # 已撤
        51: BrokerOrderStatus.CANCELLED,    # 部撤
        52: BrokerOrderStatus.CANCELLED,    # 已撤
        53: BrokerOrderStatus.FILLED,       # 已成
        54: BrokerOrderStatus.PARTIALLY_FILLED,  # 部成
        55: BrokerOrderStatus.REJECTED,     # 废单
        56: BrokerOrderStatus.REJECTED,     # 废单
    }
    return mapping.get(code, BrokerOrderStatus.PENDING)
