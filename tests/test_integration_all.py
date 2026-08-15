# Full integration test: all A-F module chains
import sys; sys.path.insert(0, 'src')

import tempfile, os, pandas as pd, numpy as np
from datetime import date, datetime

# ======== 1. Data Layer ========
print("=== 1. 数据层 ===")
from pa_mcp.data.store import DuckDBStore
from pa_mcp.data.contracts import SecurityRecord, DailyBar, CorporateAction, DatasetSnapshot
from pa_mcp.data.repositories import PointInTimeRepository
from pa_mcp.data.sources.base import DataSourceCapability

tmp = tempfile.mkdtemp()
db = DuckDBStore(os.path.join(tmp, 'test.duckdb'))
db.connect()
df = pd.DataFrame([{
    'symbol': '000001', 'date': date(2026, 7, 30), 'open': 10.0, 'high': 11.0,
    'low': 9.5, 'close': 10.8, 'volume': 1e6, 'amount': 1e7,
    'amplitude': 5.0, 'pct_change': 2.5, 'change': 0.3, 'turnover': 1.2,
    'adjust_factor': 1.0,
}])
db.insert_df('kline_daily', df)
r = db.query_df("SELECT * FROM kline_daily WHERE symbol='000001'")
assert len(r) == 1 and float(r.iloc[0]['close']) == 10.8
repo = PointInTimeRepository(db)
univ = repo.get_universe(datetime(2026, 7, 30, 15, 0, 0))
print(f"  store insert/query OK, universe={len(univ)} stocks")
print("  contracts + base port OK")

# ======== 2. Scheduler ========
print("=== 2. 调度器 ===")
from pa_mcp.data.scheduler import PhaseStatus, PhaseResult, PipelineReport
p_ok = PhaseResult('cal', PhaseStatus.SUCCESS, 365)
p_stub = PhaseResult('min', PhaseStatus.NOT_IMPLEMENTED)
p_fail = PhaseResult('fin', PhaseStatus.FAILED, error='boom')
assert p_ok.success and not p_ok.is_blocking_failure
assert not p_stub.success and p_stub.is_blocking_failure
report = PipelineReport(phases=[p_ok, p_stub, p_fail])
assert not report.all_success and len(report.blocking_failures) == 2
print("  PhaseResult/PipelineReport OK")

# ======== 3. Backtest ========
print("=== 3. 回测引擎 ===")
from pa_mcp.backtest.events import EventType
from pa_mcp.backtest.orders import Order, OrderSide, OrderStatus
from pa_mcp.backtest.broker import compute_limit_price, DailyBroker, FeeSchedule
from pa_mcp.backtest.ledger import Ledger
from pa_mcp.backtest.engine import BacktestEngine

o = Order(symbol='000001', side=OrderSide.BUY, quantity=1000)
o.transition(OrderStatus.SUBMITTED)
o.transition(OrderStatus.ACCEPTED)
assert o.is_active
try:
    o.transition(OrderStatus.CREATED)
    assert False
except ValueError:
    pass
print("  Order state machine OK")

assert compute_limit_price(10.0, 'sh_main', False, 'up') == 11.0
assert compute_limit_price(10.0, 'chinext', False, 'up') == 12.0
assert compute_limit_price(10.0, 'sh_main', True, 'up') == 10.5
print("  Price limits OK")

l = Ledger(cash=100000)
l.add_lot('000001', 1000, 10.0, date(2026, 7, 29))
assert l.get_sellable_quantity('000001', date(2026, 7, 29)) == 0
assert l.get_sellable_quantity('000001', date(2026, 7, 30)) == 1000
assert l.compute_nav({'000001': 11.0}) == 111000
print("  Ledger T+1/NAV OK")

np.random.seed(42)
dates_pd = pd.date_range('2026-07-01', periods=30, freq='B')
close = 10.0 * np.cumprod(1 + np.random.randn(30) * 0.02)
kline = pd.DataFrame({
    'symbol': '000001', 'date': dates_pd, 'open': close * 0.99,
    'high': close * 1.02, 'low': close * 0.98, 'close': close,
    'volume': [1e7] * 30, 'amount': close * 1e7,
})
sig = pd.DataFrame([{
    'symbol': '000001', 'date': dates_pd[i], 'direction': 'bullish',
    'strength_score': 60, 'strategy_name': 'ma_cross',
} for i in range(0, 30, 5)])
report = BacktestEngine(initial_cash=100000).run(kline, sig)
assert report.total_trades >= 0 and len(report.nav_series) > 0
print(f"  BacktestEngine: {report.total_trades} trades, ret={report.total_return_pct}%, SR={report.sharpe_ratio}, DD={report.max_drawdown_pct}%")

# ======== 4. Research ========
print("=== 4. 研究层 ===")
from pa_mcp.research.splits import TimeSeriesSplitter
from pa_mcp.research.recorder import RunRecorder, RunManifest, RunResult
from pa_mcp.research.benchmarks import BenchmarkRegistry, BenchmarkId

splitter = TimeSeriesSplitter(date(2020, 1, 1), date(2026, 7, 30))
folds = splitter.generate_folds()
assert len(folds) >= 4
sealed = splitter.get_sealed_holdout(folds)
print(f"  {len(folds)} folds, sealed holdout: {sealed.test_start} -> {sealed.test_end}")

rec = RunRecorder(os.path.join(tmp, 'research'))
m = rec.start_run(RunManifest(strategy_class='MaCross', strategy_params={'fast': 5, 'slow': 20}))
r = RunResult(manifest=m, metrics={'sharpe': 1.2, 'max_dd': -10.0})
rec.complete_run(r)
assert len(rec.list_runs()) == 1
print("  Recorder OK")

spec = BenchmarkRegistry().get('large_cap')
assert spec.benchmark_id == BenchmarkId.CSI300
print("  Benchmarks OK")

# ======== 5. Portfolio + Risk ========
print("=== 5. 组合与风控 ===")
from pa_mcp.portfolio.construction import PortfolioConstructor, TargetWeight
from pa_mcp.portfolio.optimizer import Optimizer
from pa_mcp.risk.guard import RiskGuard, PortfolioSnapshot, CandidateOrder, RiskDecision, GuardVerdict

pc = PortfolioConstructor()
target = pc.build([
    TargetWeight('000001', 0.05, 70, 'bank', 12.0, 1e8),
    TargetWeight('000002', 0.05, 65, 'tech', 25.0, 5e7),
])
assert target.cash_weight >= 0.19
print(f"  Portfolio: {target.total_stocks} stocks, cash={target.cash_weight:.1%}")

shares = Optimizer().integer_shares(
    {'000001': 0.08, '000002': 0.05},
    {'000001': 12.0, '000002': 25.0}, 100000)
assert all(q % 100 == 0 for q in shares.values())
print("  Optimizer integer shares OK")

guard = RiskGuard()
snap = PortfolioSnapshot(positions={'000001': 0.75}, nav=100000, drawdown_pct=0.05)
result = guard.check_single_order(snap, CandidateOrder('000002', 'buy', 2400, 12.5, 0.30))
assert result.decision == RiskDecision.ADJUST and result.adjusted_quantity == 400
print(f"  RiskGuard multi-constraint: ADJUST to {result.adjusted_quantity}")

r2 = guard.check_single_order(
    PortfolioSnapshot(nav=80000, peak_nav=100000, drawdown_pct=0.20),
    CandidateOrder('000001', 'buy', 1000, 10, 0.125))
assert r2.decision == RiskDecision.REJECT
print("  RiskGuard hard stop REJECT OK")

g = guard.check_single_position('000001', 0.50)
assert g.verdict == GuardVerdict.REDUCE and g.adjusted_max_position == 0.20
print("  RiskGuard legacy API OK")

# ======== 6. LLM Port ========
print("=== 6. LLM 端口 ===")
from pa_mcp.agent.llm_port import LLMCallParams
assert LLMCallParams(system_prompt='test', user_prompt='hello', mode='fast').mode == 'fast'
print("  LLMPort params OK")

db.close()
print("\n========== 全部 6/6 模块链接口测试通过 ==========")
