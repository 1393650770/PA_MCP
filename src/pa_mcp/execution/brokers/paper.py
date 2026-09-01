# [AI:BEGIN]
# PA_MCP - Execution: PaperBroker（纸面交易账户）
#
# 零风险自动买卖：把"虚拟下单"完整模拟（持仓/盈亏/手续费/T+1），
# 用于验证策略信号 → 下单链路，无需券商。
#
# 设计（对齐 BrokerPort 抽象）：
#   - 幂等：client_order_id 重复提交返回已有订单（不重复成交）
#   - 成交价：限价单=限价；市价单=实时行情价（腾讯/多源）
#   - 费用：佣金万 2.5（最低 5 元）+ 卖出印花税 0.05%（2023-08 后标准）
#   - T+1：买入当日 available=0，次交易日 09:30 后可卖
#   - 持久化：data/paper_account.json（原子写，进程安全锁）
#   - 初始资金 1,000,000 可配置
# [AI:END]

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from pa_mcp.execution.brokers.base import (
    BrokerAccount, BrokerFill, BrokerOrder, BrokerOrderStatus,
    BrokerPort, BrokerPosition,
)

logger = logging.getLogger(__name__)

DEFAULT_CASH = 1_000_000.0
COMMISSION_RATE = 0.00025        # 佣金万 2.5
COMMISSION_MIN = 5.0             # 最低 5 元
STAMP_TAX_RATE = 0.0005          # 印花税 0.05%（仅卖出）


class PaperBroker(BrokerPort):
    """纸面交易账户（文件持久化，进程安全）。"""

    broker_name = "paper"

    def __init__(self, data_path: Optional[str] = None,
                 initial_cash: float = DEFAULT_CASH) -> None:
        if data_path is None:
            from pa_mcp.config import PROJECT_ROOT
            data_path = str(PROJECT_ROOT / "data" / "paper_account.json")
        self.data_path = data_path
        self.initial_cash = initial_cash
        self._lock = threading.RLock()
        self._orders: dict[str, dict] = {}
        self._fills: list[dict] = []
        self._load()

    # ---- 持久化 ----

    def _load(self) -> None:
        try:
            if os.path.exists(self.data_path):
                with open(self.data_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._orders = data.get("orders", {})
                self._fills = data.get("fills", [])
                self._cash = float(data.get("cash", self.initial_cash))
                self._positions: dict[str, dict] = data.get("positions", {})
                return
        except Exception as e:  # noqa: BLE001
            logger.warning("纸面账户读取失败，重置: %s", str(e)[:60])
        self._cash = self.initial_cash
        self._positions: dict[str, dict] = {}

    def _save(self) -> None:
        with self._lock:
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            data = {"cash": self._cash, "positions": self._positions,
                    "orders": self._orders, "fills": self._fills}
            tmp = self.data_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.data_path)  # 原子替换

    # ---- BrokerPort 实现 ----

    async def connect(self) -> None:
        self._load()

    async def disconnect(self) -> None:
        self._save()

    async def submit_order(self, order: BrokerOrder) -> BrokerOrder:
        if not order.risk_decision_id:
            raise ValueError("订单缺少 RiskDecision ID — 风控不可绕过")
        with self._lock:
            # 幂等：同 client_order_id 返回已有订单
            if order.client_order_id in self._orders:
                existing = self._orders[order.client_order_id]
                order.status = BrokerOrderStatus(existing["status"])
                order.broker_order_id = existing.get("broker_order_id", "")
                return order

            side = order.side.lower()
            if side not in ("buy", "sell"):
                raise ValueError(f"非法方向: {side}")
            if order.quantity <= 0 or order.quantity % 100 != 0:
                raise ValueError("数量必须为 100 的整数倍且 > 0")

            # 成交价：限价单=限价；市价单=实时行情
            price = order.limit_price
            if price is None:
                price = await self._fetch_price(order.symbol)
            if price is None or price <= 0:
                raise ValueError(f"{order.symbol} 无法获取行情，市价单失败")

            amount = price * order.quantity
            commission = max(amount * COMMISSION_RATE, COMMISSION_MIN)
            stamp = amount * STAMP_TAX_RATE if side == "sell" else 0.0
            total_cost = amount + commission + stamp

            if side == "buy":
                if total_cost > self._cash:
                    raise ValueError(
                        f"现金不足：需要 {total_cost:,.2f}，可用 {self._cash:,.2f}")
                self._cash -= total_cost
                pos = self._positions.setdefault(order.symbol, {
                    "quantity": 0, "available": 0, "avg_cost": 0.0})
                new_qty = pos["quantity"] + order.quantity
                pos["avg_cost"] = (
                    (pos["avg_cost"] * pos["quantity"] + amount + commission)
                    / new_qty)
                pos["quantity"] = new_qty
                # T+1：买入当日不可卖
                pos["available"] = pos["available"] + 0
                pos["buy_date"] = datetime.now().strftime("%Y-%m-%d")
            else:  # sell
                pos = self._positions.get(order.symbol)
                if not pos or pos["available"] < order.quantity:
                    raise ValueError(
                        f"{order.symbol} 可卖不足："
                        f"需要 {order.quantity}，可卖 {pos['available'] if pos else 0}"
                        "（T+1：当日买入次日 09:30 后可卖）")
                pos["quantity"] -= order.quantity
                pos["available"] -= order.quantity
                if pos["quantity"] <= 0:
                    self._positions.pop(order.symbol)
                self._cash += amount - commission - stamp

            broker_id = f"paper_{uuid.uuid4().hex[:12]}"
            self._orders[order.client_order_id] = {
                "status": BrokerOrderStatus.FILLED.value,
                "broker_order_id": broker_id,
                "symbol": order.symbol, "side": side,
                "quantity": order.quantity, "price": price,
            }
            self._fills.append({
                "broker_fill_id": f"fill_{uuid.uuid4().hex[:12]}",
                "client_order_id": order.client_order_id,
                "broker_order_id": broker_id,
                "symbol": order.symbol, "side": side,
                "quantity": order.quantity, "price": price,
                "commission": round(commission, 2),
                "trade_time": datetime.now().isoformat(),
            })
            self._save()
            order.status = BrokerOrderStatus.FILLED
            order.broker_order_id = broker_id
            logger.info("纸面成交 %s %s %d 股 @ %.2f",
                        side, order.symbol, order.quantity, price)
            return order

    async def _fetch_price(self, symbol: str) -> Optional[float]:
        """实时行情（腾讯优先；ETF/股票通用）。"""
        try:
            import asyncio
            from pa_mcp.research.etf import fetch_etf_quotes, is_etf
            if is_etf(symbol):
                q = (await fetch_etf_quotes([symbol])).get(symbol)
                if q and q.get("price"):
                    return float(q["price"])
            from pa_mcp.data.sources.tencent_adapter import TencentAdapter
            q = await TencentAdapter().get_realtime_quote(symbol)
            return float(q.get("price")) if q and q.get("price") else None
        except Exception:  # noqa: BLE001
            return None

    async def cancel_order(self, client_order_id: str) -> bool:
        with self._lock:
            if client_order_id in self._orders:
                self._orders[client_order_id]["status"] = BrokerOrderStatus.CANCELLED.value
                self._save()
                return True
        return False

    async def query_order(self, client_order_id: str) -> BrokerOrder:
        with self._lock:
            rec = self._orders.get(client_order_id)
        if not rec:
            raise KeyError(f"订单不存在: {client_order_id}")
        return BrokerOrder(
            client_order_id=client_order_id, symbol=rec["symbol"],
            side=rec["side"], quantity=rec["quantity"],
            limit_price=rec.get("price"), status=BrokerOrderStatus(rec["status"]),
            broker_order_id=rec.get("broker_order_id", ""))

    async def get_account(self) -> BrokerAccount:
        with self._lock:
            positions = [
                BrokerPosition(symbol=s, quantity=p["quantity"],
                               available_quantity=p.get("available", 0),
                               avg_cost=p.get("avg_cost", 0.0))
                for s, p in self._positions.items()]
            return BrokerAccount(account_id="paper", cash=round(self._cash, 2),
                                 positions=positions)

    async def get_fills(self, since: Optional[datetime] = None) -> list[BrokerFill]:
        with self._lock:
            fills = []
            for f in self._fills:
                if since and f.get("trade_time", "")[:19] < since.isoformat()[:19]:
                    continue
                fills.append(BrokerFill(**{k: f.get(k) for k in (
                    "broker_fill_id", "client_order_id", "broker_order_id",
                    "symbol", "side", "quantity", "price", "commission",
                    "trade_time")}))
            return fills

    # ---- 附加（UI/对账用） ----

    def account_summary(self) -> dict[str, Any]:
        """账户快照（同步读，供面板/工具在任意上下文调用）。"""
        with self._lock:
            cash = self._cash
            positions = [{
                "symbol": s, "quantity": p["quantity"],
                "available": p.get("available", 0), "avg_cost": p.get("avg_cost", 0),
            } for s, p in self._positions.items()]
        lines = [f"**纸面账户**（{self.data_path}）",
                 f"- 现金 {cash:,.2f} 元｜持仓 {len(positions)} 只",
                 "",
                 "| 代码 | 数量 | 可卖(T+1) | 成本 |",
                 "|---|---|---|---|"]
        for p in positions:
            lines.append(f"| {p['symbol']} | {p['quantity']} | {p['available']} "
                         f"| {p['avg_cost']:.2f} |")
        return {"cash": round(cash, 2), "positions": positions,
                "report": "\n".join(lines)}
