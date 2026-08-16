# [AI:BEGIN]
# PA_MCP - 全接口冒烟验证脚本
#
# 验证所有 MCP 工具（89 个）与 agent 核心接口是否可用：
#   1. 种子临时 DuckDB（6 股票 × 250 根 K 线 + 财务/板块/持仓/指数/
#      情绪/预测日志）——绝不接触真实 data/pa_mcp.duckdb
#   2. 覆盖 server 全局 _store（临时库），_router=None（避免网络依赖）
#   3. 枚举 FastMCP 工具注册表，逐个调用（最小合法参数）
#   4. 结果分类：✅ 成功 / ⚠️ 预期降级（无数据/网络/未配置） / ❌ 崩溃
#   5. 另验证 agent 服务公开函数（prediction/experience/graham 等）
#
# 运行：venv\\Scripts\\python.exe scripts/verify_interfaces.py
# [AI:END]

from __future__ import annotations

import asyncio
import inspect
import io
import os
import sys
import tempfile
from typing import Any, Optional, get_args, get_origin

# UTF-8 输出（Windows GBK 控制台打印 emoji 会崩；保留原对象避免 structlog 崩溃）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

# ---- 种子数据 ----

def _seed(tmp: str) -> str:
    from pa_mcp.data.store import DuckDBStore
    db = os.path.join(tmp, "verify.duckdb")
    store = DuckDBStore(db)
    store.connect()
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-06-01", periods=250, freq="B")
    symbols = [f"00000{i}" for i in range(1, 7)]
    # K 线
    krows = []
    for sym in symbols:
        close = 10.0
        for i in range(250):
            close *= 1 + 0.001 + rng.normal(0, 0.01)
            krows.append({
                "symbol": sym, "date": dates[i],
                "open": close * 0.995, "high": close * 1.01,
                "low": close * 0.99, "close": close, "volume": 1e6,
                "amount": 1e7, "pct_change": 0.2, "turnover": 1.0,
                "change": 0.2, "amplitude": 2.0, "adjust_factor": 1.0,
            })
    store.insert_df("kline_daily", pd.DataFrame(krows))
    # 股票基本信息
    store.insert_df("stock_basic", pd.DataFrame([
        {"symbol": s, "name": f"股票{s}", "sector": "银行" if i < 3 else "科技",
         "industry": "x", "market_cap": 500e8, "list_date": "2020-01-01",
         "exchange": "SH", "board": "main", "is_st": False,
         "is_suspended": False}
        for i, s in enumerate(symbols)]))
    # 财务
    fin = []
    for sym in symbols:
        for i in range(4):
            fin.append({"symbol": sym,
                        "report_date": pd.Timestamp(f"2025-{12-i*3:02d}-01"),
                        "pub_date": pd.Timestamp(f"2025-{12-i*3:02d}-15"),
                        "revenue": 1e9, "operating_profit": 2e8,
                        "net_profit": 1.5e8, "net_profit_parent": 1.5e8,
                        "eps": 0.8, "roe": 12.0, "gross_margin": 30.0,
                        "net_margin": 15.0, "revenue_yoy": 15.0,
                        "profit_yoy": 18.0})
    store.insert_df("financials_income", pd.DataFrame(fin))
    # 指数
    idx = []
    close = 3000.0
    for i in range(250):
        close *= 1 + 0.0005 + rng.normal(0, 0.005)
        idx.append({"symbol": "sh000001", "date": dates[i],
                    "open": close * 0.995, "high": close * 1.01,
                    "low": close * 0.99, "close": close, "volume": 1e8,
                    "amount": 1e9})
    store.insert_df("index_daily", pd.DataFrame(idx))
    # 板块
    srows = []
    for code, name, ret in (("BK0001", "银行", 0.003), ("BK0002", "科技", 0.002)):
        close = 100.0
        for i in range(120):
            close *= 1 + ret
            srows.append({"sector_code": code, "name": name,
                          "date": dates[i + 130], "open": close * 0.99,
                          "close": close, "high": close * 1.01,
                          "low": close * 0.99, "volume": 1e7,
                          "amount": 1e9, "pct_change": ret * 100,
                          "turnover": 2.0})
    store.insert_df("sector_daily", pd.DataFrame(srows))
    # 情绪
    store.insert_df("sentiment_daily", pd.DataFrame([{
        "date": str(dates[-1])[:10], "limit_up_count": 30,
        "limit_down_count": 3, "max_board_height": 3, "board2_count": 5,
        "board3_count": 2, "board4p_count": 1, "first_board_count": 20,
        "promotion_rate": 0.5, "sentiment_score": 60, "stage": ""}]))
    # 持仓（与 portfolio_add 工具一致的 DDL）
    store.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY,
            symbol VARCHAR(10), cost DOUBLE, shares INTEGER,
            added_date DATE, created_at TIMESTAMP
        )
    """)
    store.insert_df("portfolio", pd.DataFrame([
        {"id": 1, "symbol": "000001", "cost": 10.0, "shares": 10000,
         "added_date": "2026-07-01", "created_at": "2026-07-01"},
        {"id": 2, "symbol": "000002", "cost": 10.0, "shares": 5000,
         "added_date": "2026-07-01", "created_at": "2026-07-01"}]))
    # 预测日志（几条已评估）
    from pa_mcp.agent.prediction import PredictionResult, PredictionService
    svc = PredictionService(store_path=db)
    for i in range(6):
        svc.save_prediction(PredictionResult(
            symbol=symbols[i % 6], predict_date="2026-07-06", horizon="5d",
            direction="up", prob_up=0.7, prob_down=0.15, prob_sideways=0.15))
    store.close()
    return db


# ---- 参数生成（从 FastMCP Tool JSON Schema） ----

def _gen_kwargs(tool) -> dict[str, Any]:
    """按 Tool 的 JSON Schema 生成 kwargs（参数名/类型/必填正确）。"""
    params = tool.parameters or {}
    props = params.get("properties") or {}
    required = set(params.get("required") or [])
    kwargs = {}
    for name, spec in props.items():
        # symbols 类参数即使可选也填充（默认 "" 会导致空池）
        if name not in required and "symbol" not in name.lower():
            continue  # 其余可选参数用默认值
        t = spec.get("type", "string")
        if t == "string":
            enum = spec.get("enum")
            if enum:
                kwargs[name] = enum[0]
            else:
                kwargs[name] = (
                    "000001" if name == "symbol"
                    else "2026-08-14" if "date" in name.lower()
                    else "" if "pool" in name.lower()
                    else "000001,000002,000003,000004,000005,000006"
                    if "symbols" in name.lower()
                    else "bollinger_mean_reversion" if "strategy" in name.lower()
                    else "sh000001" if "index" in name.lower()
                    else "银行" if name == "sector"
                    else "up" if "direction" in name.lower()
                    else "x")
        elif t == "integer":
            kwargs[name] = (
                200 if "share" in name.lower() or "holding" in name.lower()
                else 10 if "top" in name.lower() or "n_" in name.lower()
                or "limit" in name.lower() or "days" in name.lower()
                or "period" in name.lower() or "window" in name.lower()
                else 5)
        elif t == "number":
            kwargs[name] = 100000.0 if "cash" in name.lower() \
                or "value" in name.lower() else 5.0
        elif t == "boolean":
            kwargs[name] = False
        else:
            kwargs[name] = None
    return kwargs


# ---- 主验证 ----

async def main() -> None:
    tmp = tempfile.mkdtemp(prefix="pa_mcp_verify_")
    db = _seed(tmp)

    # 彻底隔离：get_settings 全局指向临时库（所有模块共用）
    import pa_mcp.config as cfg
    real = cfg.get_settings()
    real_dump = real.model_dump()
    real_dump["database"] = {"path": db, "read_only": False}
    cfg._settings = type(real)(**real_dump)

    import pa_mcp.server as server
    from pa_mcp.data.store import DuckDBStore

    # 覆盖全局（临时库，绝不碰真实库）
    server._store = DuckDBStore(db)
    server._store.connect()
    server._router = None
    # 初始化 adapter（lifespan 等价物；网络失败由各工具降级）
    from pa_mcp.data.sources.akshare_adapter import AKShareAdapter
    from pa_mcp.data.sources.sina_adapter import SinaAdapter
    server._akshare = AKShareAdapter()
    server._sina = SinaAdapter()
    # LLM 隔离：强制走「无 LLM 降级」路径（不触发真实 doubao 调用）
    import pa_mcp.agent.llm_port as lp
    lp._adapter = None
    import pa_mcp.agent.llm_factory as lf
    lf.init_llm_adapter = lambda *a, **k: None  # 阻止 legacy client 重新初始化
    import pa_mcp.agent.llm_client as lc
    lc._client = None  # 清 legacy 单例缓存（其内部可能已持有真实 adapter）

    tools = server.mcp._tool_manager.list_tools()
    print(f"=== MCP 工具验证（共 {len(tools)} 个） ===")
    ok, warn, fail = [], [], []
    for tool in sorted(tools, key=lambda t: t.name):
        name = tool.name
        fn = tool.fn
        try:
            kwargs = _gen_kwargs(tool)
            result = await tool.fn(**kwargs)
            if isinstance(result, dict):
                if result.get("success") is False:
                    err = str(result.get("error", ""))[:60]
                    if any(k in err for k in ("网络", "无数据", "unavailable",
                                              "未配置", "not found", "未注册",
                                              "暂未", "不可用", "none",
                                              "disconnected", "timeout",
                                              "超时", "connection",
                                              "call failed", "retries")):
                        warn.append((name, f"⚠️ 降级：{err}"))
                    else:
                        fail.append((name, f"❌ 返回失败：{err}"))
                else:
                    ok.append((name, "✅"))
            else:
                ok.append((name, "✅（非 dict 返回）"))
        except Exception as e:  # noqa: BLE001
            fail.append((name, f"❌ 崩溃：{type(e).__name__}: {str(e)[:60]}"))

    for name, status in ok + warn + fail:
        print(f"  {status} {name}")
    print(f"\n=== 汇总：✅ {len(ok)} / ⚠️ {len(warn)} / ❌ {len(fail)}（共 {len(tools)}） ===")
    if fail:
        print("\n❌ 崩溃工具：")
        for name, st in fail:
            print(f"  {name}: {st}")

    # Agent 核心服务接口
    print("\n=== Agent 服务接口 ===")
    from pa_mcp.agent.prediction import PredictionService
    from pa_mcp.agent.experience import ExperienceService
    from pa_mcp.research.graham import GrahamScreener
    from pa_mcp.research.factors import get_factor_registry
    from pa_mcp.research.value_momentum import ValueMomentumScreen

    checks = []
    try:
        r = await PredictionService(store_path=db).predict(
            "000001", _kline(db), horizon="5d", use_llm=False)
        checks.append(("PredictionService.predict", "✅" if r.mode else "❌"))
    except Exception as e:
        checks.append(("PredictionService.predict", f"❌ {e}"))
    try:
        es = ExperienceService(store_path=db)
        sid = es.save_analysis("000001", {"direction": "bullish",
                                          "overall_strength_score": 70})
        n = len(es.search_experience(symbol="000001"))
        checks.append(("ExperienceService.save/search", f"✅ id={sid} n={n}"))
    except Exception as e:
        checks.append(("ExperienceService", f"❌ {e}"))
    try:
        g = GrahamScreener(store_path=db).screen(
            ["000001", "000002", "000003"],
            quotes={s: {"pe": 10, "pb": 1, "price": 10}
                    for s in ("000001", "000002", "000003")})
        checks.append(("GrahamScreener.screen", f"✅ {len(g)} 只"))
    except Exception as e:
        checks.append(("GrahamScreener.screen", f"❌ {e}"))
    try:
        reg = get_factor_registry()
        checks.append(("FactorRegistry", f"✅ {len(reg.list_all())} 因子"))
    except Exception as e:
        checks.append(("FactorRegistry", f"❌ {e}"))
    try:
        vm = ValueMomentumScreen(store_path=db).screen(
            ["600001", "600002", "600003", "600004", "600005", "600006"])
        checks.append(("ValueMomentumScreen", f"✅ {vm.get('n_scored', '?')} 只"))
    except Exception as e:
        checks.append(("ValueMomentumScreen", f"❌ {e}"))
    for name, st in checks:
        print(f"  {st} {name}")

    server._store.close()
    sys.exit(1 if fail else 0)


def _kline(db: str) -> pd.DataFrame:
    from pa_mcp.data.store import DuckDBStore
    store = DuckDBStore(db)
    store.connect()
    try:
        return store.query_df(
            "SELECT * FROM kline_daily WHERE symbol = '000001' ORDER BY date",
            [])
    finally:
        store.close()


if __name__ == "__main__":
    asyncio.run(main())
