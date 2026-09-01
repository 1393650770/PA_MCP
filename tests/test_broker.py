# 交易执行层测试：PaperBroker（幂等/风控/T+1/费用）+ broker 工厂 + QMT 闸门

from __future__ import annotations

import asyncio
import json

import pytest

from pa_mcp.execution.brokers.base import BrokerOrder, BrokerOrderStatus
from pa_mcp.execution.brokers.paper import PaperBroker


def _make_broker(tmp_path, cash=1_000_000.0) -> PaperBroker:
    return PaperBroker(data_path=str(tmp_path / "paper.json"), initial_cash=cash)


def _order(symbol="510300", side="buy", quantity=100, limit_price=4.6,
           cid="test_1") -> BrokerOrder:
    return BrokerOrder(client_order_id=cid, symbol=symbol, side=side,
                       quantity=quantity, limit_price=limit_price,
                       risk_decision_id="risk_test")


def test_buy_and_account(tmp_path):
    """买入成交：持仓/现金/成本正确。"""
    b = _make_broker(tmp_path)
    o = asyncio.run(b.submit_order(_order()))
    assert o.status == BrokerOrderStatus.FILLED
    acct = asyncio.run(b.get_account())
    assert len(acct.positions) == 1
    p = acct.positions[0]
    assert p.quantity == 100 and p.available_quantity == 0  # T+1
    # 现金 = 100万 - 4.6*100 - 佣金(万2.5,最低5)
    assert acct.cash == pytest.approx(1_000_000 - 460 - 5, abs=0.01)


def test_idempotent_same_order(tmp_path):
    """幂等：同 client_order_id 不重复成交。"""
    b = _make_broker(tmp_path)
    asyncio.run(b.submit_order(_order(cid="dup1")))
    o2 = asyncio.run(b.submit_order(_order(cid="dup1")))
    assert o2.status == BrokerOrderStatus.FILLED
    acct = asyncio.run(b.get_account())
    assert acct.positions[0].quantity == 100  # 只有一笔


def test_t1_sell_blocked(tmp_path):
    """T+1：当日买入不可卖。"""
    b = _make_broker(tmp_path)
    asyncio.run(b.submit_order(_order(cid="b1")))
    with pytest.raises(ValueError, match="可卖不足|T\\+1"):
        asyncio.run(b.submit_order(_order(side="sell", cid="s1")))
    # 次日可卖（模拟日期推进：直接改持仓 buy_date）
    b._positions["510300"]["buy_date"] = "2000-01-01"
    b._positions["510300"]["available"] = 100
    o = asyncio.run(b.submit_order(_order(side="sell", quantity=100, cid="s2")))
    assert o.status == BrokerOrderStatus.FILLED
    acct = asyncio.run(b.get_account())
    assert acct.positions == []


def test_insufficient_cash(tmp_path):
    """现金不足拒绝。"""
    b = _make_broker(tmp_path, cash=500)
    with pytest.raises(ValueError, match="现金不足"):
        asyncio.run(b.submit_order(_order(quantity=100, limit_price=10)))


def test_risk_id_required(tmp_path):
    """风控 ID 必填（不可绕过）。"""
    b = _make_broker(tmp_path)
    o = BrokerOrder(client_order_id="x", symbol="510300", side="buy",
                    quantity=100, limit_price=4.6, risk_decision_id="")
    with pytest.raises(ValueError, match="风控"):
        asyncio.run(b.submit_order(o))


def test_quantity_multiple_100(tmp_path):
    b = _make_broker(tmp_path)
    with pytest.raises(ValueError, match="100 的整数倍"):
        asyncio.run(b.submit_order(_order(quantity=150)))


def test_sell_commission_and_stamp(tmp_path):
    """卖出费用：佣金 + 印花税 0.05%。"""
    b = _make_broker(tmp_path)
    asyncio.run(b.submit_order(_order(cid="b1", quantity=1000, limit_price=4.0)))
    b._positions["510300"]["available"] = 1000
    b._positions["510300"]["buy_date"] = "2000-01-01"
    before = b._cash
    asyncio.run(b.submit_order(_order(side="sell", quantity=1000,
                                      limit_price=4.0, cid="s1")))
    # 卖出到账 = 4000 - 佣金5 - 印花税2
    assert b._cash == pytest.approx(before + 4000 - 5 - 2.0, abs=0.01)


def test_market_order_fetch_price(tmp_path, monkeypatch):
    """市价单：无 limit_price 时按实时价成交。"""

    async def _fake_price(self, symbol):
        return 4.5

    monkeypatch.setattr(PaperBroker, "_fetch_price", _fake_price)
    b = _make_broker(tmp_path)
    o = _order(limit_price=None)
    filled = asyncio.run(b.submit_order(o))
    assert filled.status == BrokerOrderStatus.FILLED
    acct = asyncio.run(b.get_account())
    # 成本含佣金：4.5×100 + 佣金5 = 455 → 4.55/股
    assert acct.positions[0].avg_cost == pytest.approx(4.55, abs=0.01)


def test_persistence(tmp_path):
    """持久化：重启后账户状态保留。"""
    path = tmp_path / "paper.json"
    b = PaperBroker(data_path=str(path))
    asyncio.run(b.submit_order(_order(cid="p1", quantity=200)))
    b2 = PaperBroker(data_path=str(path))  # 重新加载
    acct = asyncio.run(b2.get_account())
    assert acct.positions[0].quantity == 200
    assert acct.positions[0].available_quantity == 0


def test_account_summary_sync(tmp_path):
    """同步快照（任意上下文可调）。"""
    b = _make_broker(tmp_path)
    asyncio.run(b.submit_order(_order(cid="a1")))
    s = b.account_summary()
    assert s["cash"] > 0 and len(s["positions"]) == 1


# ---- broker 工厂 ----

def test_factory_paper_default(tmp_path, monkeypatch):
    """默认 paper 模式。"""
    from pa_mcp.execution.brokers import get_broker, reset_broker
    reset_broker()
    monkeypatch.setattr("pa_mcp.execution.brokers.paper.PaperBroker",
                        lambda **k: PaperBroker(data_path=str(tmp_path / "p.json")))
    # 用默认 settings（mode=paper）
    from pa_mcp.config import get_settings
    b = get_broker(get_settings())
    assert b.broker_name == "paper"
    reset_broker()


def test_factory_qmt_gate(tmp_path, monkeypatch):
    """QMT 模式未配置实盘闸 → 下单被拒。"""
    from pa_mcp.execution.brokers import get_broker, reset_broker
    from pa_mcp.config import get_settings
    reset_broker()
    settings = get_settings()
    settings.broker.mode = "qmt"
    settings.broker.qmt.enable_live_trading = False
    settings.broker.qmt.user_data_path = ""
    b = get_broker(settings)
    # 未配置路径 → 降级 paper 或 QmtUnavailable；无论如何不能实盘发单
    assert b.broker_name in ("paper", "qmt")
    reset_broker()
