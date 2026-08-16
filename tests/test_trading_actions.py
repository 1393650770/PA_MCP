# [AI:BEGIN]
# PA_MCP - 今日操作面板测试
# [AI:END]

from __future__ import annotations

import asyncio

import pandas as pd

from pa_mcp.research.trading_actions import _holding_action, trading_actions


def test_holding_action_rules():
    """止盈止损规则（可追溯）。"""
    # 盈利 18% + 综合看跌 → 分批止盈
    h = {"symbol": "000001", "pnl_pct": 18.0, "price": 11.8, "cost": 10.0,
         "prediction": {"direction": "up"}, "resonance": {"signal": "up"},
         "consensus": {"signal": "down", "strength": 0.7}}
    a = _holding_action(h)
    assert a["action"] == "分批止盈"

    # 亏损 12% → 纪律止损
    h2 = {"symbol": "000001", "pnl_pct": -12.0, "price": 8.8, "cost": 10.0,
          "prediction": {"direction": "up"}, "resonance": {},
          "consensus": {}}
    assert _holding_action(h2)["action"] == "纪律止损"

    # 盈利 12% + 预测看跌 → 考虑止盈
    h3 = {"symbol": "000001", "pnl_pct": 12.0, "price": 11.2, "cost": 10.0,
          "prediction": {"direction": "down"}, "resonance": {},
          "consensus": {}}
    assert _holding_action(h3)["action"] == "考虑止盈"

    # 亏损 9% + 综合看跌 → 止损
    h4 = {"symbol": "000001", "pnl_pct": -9.0, "price": 9.1, "cost": 10.0,
          "prediction": {"direction": "up"}, "resonance": {},
          "consensus": {"signal": "down", "strength": 0.8}}
    assert _holding_action(h4)["action"] == "止损"

    # 正常 → 持有观察（带触发价）
    h5 = {"symbol": "000001", "pnl_pct": 3.0, "price": 10.3, "cost": 10.0,
          "prediction": {"direction": "up"}, "resonance": {},
          "consensus": {}}
    a5 = _holding_action(h5)
    assert a5["action"] == "持有观察"
    assert a5["stop_level"] and a5["take_level"]


def test_trading_actions_panel(tmp_path, monkeypatch):
    """面板结构：基调/持仓/买入候选 + 报告可读。"""
    # 种子库（含持仓）
    import sys, os
    sys.path.insert(0, "scripts")
    import verify_interfaces as v
    db = v._seed(str(tmp_path))

    import pa_mcp.config as cfg
    real = cfg.get_settings()
    d = real.model_dump()
    d["database"] = {"path": db, "read_only": False}
    monkeypatch.setattr(cfg, "_settings", type(real)(**d))

    # LLM 隔离
    import pa_mcp.agent.llm_port as lp
    import pa_mcp.agent.llm_factory as lf
    import pa_mcp.agent.llm_client as lc
    lp._adapter = None
    lf.init_llm_adapter = lambda *a, **k: None
    lc._client = None

    r = asyncio.run(trading_actions(
        ["000001", "000002", "000003", "000004", "000005", "000006"]))
    assert "tone" in r
    assert "total_position" in r["tone"]
    assert "holdings" in r
    assert isinstance(r["buy_candidates"], list)
    # 持仓（种子库 000001/000002）→ 有操作
    assert r["holdings"], "应有持仓操作"
    assert all(h["action"] for h in r["holdings"])
    assert "今日操作面板" in r["report"]
    assert "操作基调" in r["report"]


def test_trading_actions_no_portfolio(tmp_path, monkeypatch):
    """无持仓 → 面板正常（持仓段提示，不崩）。"""
    import pa_mcp.config as cfg
    real = cfg.get_settings()
    d = real.model_dump()
    d["database"] = {"path": str(tmp_path / "none.duckdb"),
                     "read_only": False}
    monkeypatch.setattr(cfg, "_settings", type(real)(**d))

    r = asyncio.run(trading_actions(["000001"]))
    assert r["holdings"] == []
    assert "无持仓记录" in r["report"]
