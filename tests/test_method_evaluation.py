# 开源方法评价（method_evaluation）测试
#
# 纯函数 + monkeypatch：数据层（_load_holdings/_load_klines）与各方法
# evaluator 均可注入，不依赖真实 DB/网络/LLM。

from __future__ import annotations

import pandas as pd
import pytest

from pa_mcp.research import method_evaluation as me


def _fake_klines() -> dict[str, pd.DataFrame]:
    """合成日线（>60 根，供缠论等要求历史长度的 evaluator）。"""
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    df = pd.DataFrame({
        "symbol": ["000001"] * 120,
        "date": dates,
        "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0,
        "volume": 1_000_000, "amount": 10_000_000.0,
        "pct_change": 0.0,
    })
    return {"000001": df}


class FakeStore:
    """数据层隔离：永不触碰真实 DuckDB（可能被其他进程占用）。"""

    def __init__(self, has_portfolio: bool = False):
        self._has_portfolio = has_portfolio
        self.closed = False

    def table_exists(self, name: str) -> bool:
        return name == "portfolio" and self._has_portfolio

    def query_df(self, sql: str, params=None):
        if "FROM portfolio" in sql and self._has_portfolio:
            return pd.DataFrame({"symbol": ["000001"]})
        return pd.DataFrame()

    def close(self) -> None:
        self.closed = True


# ---- 注册表 ----

def test_evaluator_registry_complete():
    """理财+量化方法编目完整，字段齐全。"""
    cats = {}
    for mid, spec in me.METHOD_EVALUATORS.items():
        assert spec["category"] in ("理财方法", "量化方法"), f"{mid} 类别非法"
        assert spec["name_zh"], f"{mid} 缺名称"
        assert spec["verification"], f"{mid} 缺可信度说明"
        cats[spec["category"]] = cats.get(spec["category"], 0) + 1
    assert cats.get("理财方法", 0) >= 4, "理财方法不足"
    assert cats.get("量化方法", 0) >= 1, "量化方法不足"
    # 下半部分表格依赖的列与注册表一致
    for mid in ("graham", "value_momentum", "canslim", "chan", "consensus"):
        assert mid in me.METHOD_EVALUATORS


# ---- 单元格与报告渲染（纯逻辑） ----

def test_fmt_cell():
    assert me._fmt_cell(None) == "—"
    assert me._fmt_cell({}) == "—"
    assert me._fmt_cell({"label": "up 80%", "detail": "强", "ok": True}) == \
        "🟢 up 80%（强）"
    assert me._fmt_cell({"label": "up", "ok": True}) == "🟢 up"
    assert me._fmt_cell({"label": "数据不足", "detail": "", "ok": False}) == \
        "⚪ 数据不足"


def test_format_report_full():
    cred = {
        "pool": ["000001"],
        "quant": {
            "n_strategies": 2,
            "strategies": [
                {"strategy": "bollinger_mean_reversion", "total_events": 30,
                 "n_stocks": 1, "excess_5d_pct": 0.5, "win_rate_5d": 60.0,
                 "edge_stock_fraction": 1.0, "useful": True},
                {"strategy": "turtle", "total_events": 20, "n_stocks": 1,
                 "excess_5d_pct": -0.3, "win_rate_5d": 40.0,
                 "edge_stock_fraction": 0.0, "useful": False},
            ],
            "useful_strategies": ["bollinger_mean_reversion"],
        },
        "methods": [
            {"method": "graham", "name_zh": "格雷厄姆筛选",
             "status": "池内 2/3 只有财务数据可评", "conclusion": "通过 1 只",
             "verification": "确定性规则"},
        ],
    }
    hold = {
        "holdings": ["000001"],
        "rows": [{
            "symbol": "000001",
            "graham": {"label": "5/7 通过", "detail": "安全边际 +12%", "ok": True},
            "value_momentum": {"label": "便宜且走强", "detail": "复合 1.20", "ok": True},
            "canslim": {"label": "未达标/数据不足", "detail": "", "ok": False},
            "chan": {"label": "偏多", "detail": "无背驰", "ok": True},
            "consensus": {"label": "up 80%", "detail": "强·一致度70%", "ok": True},
        }],
    }
    report = me.format_report(cred, hold)
    for head in ("① 量化方法可信度", "② 理财方法评估状态", "③ 持仓×方法评价",
                 "bollinger_mean_reversion", "格雷厄姆筛选", "000001",
                 "便宜且走强", "up 80%"):
        assert head in report, f"report 缺 {head}"
    assert "✅" in report and "❌" in report
    assert "可信（有效判定）" in report


def test_format_report_degraded():
    """数据不足时各区块优雅降级。"""
    report = me.format_report(
        {"pool": [], "quant": {"error": "无行情数据"},
         "methods": [{"name_zh": "格雷厄姆筛选", "status": "评估失败",
                      "conclusion": "无数据", "verification": "确定性规则"}]},
        {"error": "无持仓（先添加持仓），无法做持仓×方法评价"})
    assert "无行情数据" in report and "无持仓" in report
    assert "③ 持仓×方法评价" in report


# ---- 数据层隔离：无持仓 / 无数据降级 ----

async def test_evaluate_holdings_empty(monkeypatch):
    monkeypatch.setattr(me, "_load_holdings", lambda store=None: [])
    monkeypatch.setattr(me, "_store", lambda: FakeStore())
    result = await me.evaluate_holdings(symbols=None)
    assert "error" in result
    assert "无持仓" in result["error"]


def test_evaluate_credibility_no_data(monkeypatch):
    monkeypatch.setattr(me, "_load_holdings", lambda store=None: [])
    monkeypatch.setattr(me, "_store", lambda: FakeStore())
    monkeypatch.setattr(
        me, "_load_klines",
        lambda store=None, symbols=None, limit=250: {})
    result = me.evaluate_credibility()
    assert "error" in result


# ---- 集成：注入 evaluator + klines ----

async def test_evaluate_holdings_with_injected_evaluators(monkeypatch):
    monkeypatch.setattr(me, "_load_holdings", lambda store=None: ["000001"])
    monkeypatch.setattr(me, "_store", lambda: FakeStore())
    monkeypatch.setattr(
        me, "_load_klines", lambda store=None, symbols=None, limit=250: _fake_klines())

    async def _consensus(symbol, kline_df=None):
        return {"label": "up 80%", "detail": "强", "ok": True}

    monkeypatch.setattr(me, "_eval_consensus", _consensus)
    monkeypatch.setattr(me, "_eval_graham",
                        lambda sym, klines, quotes=None:
                        {"label": "6/7 通过", "detail": "安全边际 +15%", "ok": True})
    monkeypatch.setattr(me, "_eval_value_momentum",
                        lambda sym, klines:
                        {"label": "便宜且走强", "detail": "复合 1.1", "ok": True})
    monkeypatch.setattr(me, "_eval_canslim",
                        lambda sym, klines:
                        {"label": "达标", "detail": "评分 5/6", "ok": True})
    monkeypatch.setattr(me, "_eval_chan",
                        lambda sym, kline_df:
                        {"label": "偏多", "detail": "无背驰", "ok": True})

    result = await me.evaluate_holdings(symbols=None)
    assert "error" not in result
    row = result["rows"][0]
    assert row["symbol"] == "000001"
    assert row["graham"]["label"] == "6/7 通过"
    assert row["consensus"]["label"] == "up 80%"
    # 五列齐全
    for col in ("graham", "value_momentum", "canslim", "chan", "consensus"):
        assert col in row, f"缺列 {col}"


async def test_evaluator_failure_degrades(monkeypatch):
    """单个方法失败只降级该格，不拖垮整行。"""
    monkeypatch.setattr(me, "_load_holdings", lambda store=None: ["000001"])
    monkeypatch.setattr(me, "_store", lambda: FakeStore())
    monkeypatch.setattr(
        me, "_load_klines", lambda store=None, symbols=None, limit=250: _fake_klines())

    async def _consensus(symbol, kline_df=None):
        raise RuntimeError("LLM 不可用")

    monkeypatch.setattr(me, "_eval_consensus", _consensus)
    # 其余 evaluator 让全部方法都失败，验证降级行为一致
    for name in ("graham", "value_momentum", "canslim", "chan"):
        def _boom(*a, **k):
            raise RuntimeError("模拟失败")
        monkeypatch.setattr(me, f"_eval_{name}", _boom)

    result = await me.evaluate_holdings(symbols=None)
    assert "error" not in result
    row = result["rows"][0]
    assert row["consensus"]["label"] == "评估失败"
    assert row["graham"]["label"] == "评估失败"


def test_credibility_quant_reuses_compare(monkeypatch):
    """量化可信度复用 compare_all_strategies，构造行结构正确。"""
    monkeypatch.setattr(me, "_load_holdings", lambda store=None: ["000001"])
    monkeypatch.setattr(me, "_store", lambda: FakeStore())
    monkeypatch.setattr(
        me, "_load_klines", lambda store=None, symbols=None, limit=250: _fake_klines())

    def _fake_compare(klines):
        return {
            "n_strategies": 1,
            "strategies": [{
                "strategy": "bollinger_mean_reversion", "total_events": 25,
                "n_stocks": 1, "excess_5d_pct": 0.8,
                "edge_stock_fraction": 1.0, "useful": True,
                "horizons": {"5": {"win_rate": 65.0, "excess": 0.8,
                                   "has_edge": 1, "n_events": 25,
                                   "n_stocks": 1}},
            }],
            "useful_strategies": ["bollinger_mean_reversion"],
        }
    monkeypatch.setattr(
        "pa_mcp.research.strategy_compare.compare_all_strategies",
        _fake_compare)
    # 理财方法评估需真实分析器 → 也替换
    monkeypatch.setattr(me, "_evaluate_method_credibility",
                        lambda klines, quotes=None: [])
    result = me.evaluate_credibility()
    assert "error" not in result
    quant = result["quant"]
    assert quant["n_strategies"] == 1
    assert quant["strategies"][0]["win_rate_5d"] == 65.0
    assert quant["strategies"][0]["useful"] is True
    assert quant["useful_strategies"] == ["bollinger_mean_reversion"]


async def test_evaluate_methods_report_combined(monkeypatch):
    """组合入口：上下两部分 + markdown。"""
    monkeypatch.setattr(me, "_load_holdings", lambda store=None: ["000001"])
    monkeypatch.setattr(me, "_store", lambda: FakeStore())
    monkeypatch.setattr(
        me, "_load_klines", lambda store=None, symbols=None, limit=250: _fake_klines())
    monkeypatch.setattr(me, "_evaluate_credibility_quant",
                        lambda klines: {
                            "n_strategies": 0, "strategies": [],
                            "useful_strategies": []})
    monkeypatch.setattr(me, "_evaluate_method_credibility",
                        lambda klines, quotes=None: [])

    async def _no_quotes(symbols):
        return {}
    monkeypatch.setattr(me, "_fetch_quotes", _no_quotes)
    async def _consensus(symbol, kline_df=None):
        return {"label": "up 60%", "detail": "", "ok": True}
    monkeypatch.setattr(me, "_eval_consensus", _consensus)
    for name in ("graham", "value_momentum", "canslim", "chan"):
        monkeypatch.setattr(me, f"_eval_{name}",
                            lambda *a, **k: {"label": "—", "detail": "", "ok": False})

    result = await me.evaluate_methods_report()
    assert "credibility" in result and "holdings" in result
    report = result["report"]
    assert "① 量化方法可信度" in report
    assert "③ 持仓×方法评价" in report
    assert "000001" in report
