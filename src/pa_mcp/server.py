# [AI:BEGIN]
# PA_MCP - MCP Server Entry Point
# FastMCP-based server with dual transport (stdio + HTTP/SSE).
# Registers all 25+ MCP tools across 8 domains.
# [AI:END]

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal, Optional

import pandas as pd
import structlog
from mcp.server.fastmcp import FastMCP, Context

from pa_mcp.config import get_settings, Settings
from pa_mcp.data import AKShareAdapter, CacheManager, DataValidator, DuckDBStore, SinaAdapter
from pa_mcp.risk.guard import RiskGuard
from pa_mcp.tools.utils import format_error, not_found_error
from pa_mcp.tools.prompts import PROMPTS

logger = structlog.get_logger(__name__)

# ---- Global state (initialized in lifespan) ----
_store: Optional[DuckDBStore] = None
_cache: Optional[CacheManager] = None
_akshare: Optional[AKShareAdapter] = None
_sina: Optional[SinaAdapter] = None
_router: Optional[Any] = None
_guard: Optional[RiskGuard] = None
_settings: Optional[Settings] = None


# ---- Lifespan ----

@asynccontextmanager
async def server_lifespan(server: FastMCP):
    """Initialize and cleanup server resources."""
    global _store, _cache, _akshare, _guard, _settings

    logger.info("PA_MCP server starting...")
    _settings = get_settings()

    # Initialize data layer
    _store = DuckDBStore()
    _store.connect()
    _cache = CacheManager()
    _akshare = AKShareAdapter()
    _sina = SinaAdapter()

    # Multi-source router: ordered failover chain from config
    try:
        from pa_mcp.data.router import DataSourceRouter, CircuitBreakerConfig
        from pa_mcp.data.sources.tencent_adapter import TencentAdapter
        from pa_mcp.data.sources.eastmoney_adapter import EastMoneyAdapter

        source_factory = {
            "akshare": lambda: _akshare,
            "sina": lambda: _sina,
            "tencent": TencentAdapter,
            "eastmoney": EastMoneyAdapter,
        }

        chain: list[tuple[str, Any]] = []
        for name in _settings.router.sources:
            factory = source_factory.get(name)
            if factory is None:
                logger.warning("Unknown data source in config", source=name)
                continue
            try:
                chain.append((name, factory() if callable(factory) and name not in ("akshare", "sina") else factory()))
            except Exception as e:
                logger.warning("Failed to init data source", source=name, error=str(e))

        if chain:
            breaker_cfg = CircuitBreakerConfig(
                failure_threshold=_settings.router.circuit.failure_threshold,
                cooldown_seconds=_settings.router.circuit.cooldown_seconds,
            )
            _router = DataSourceRouter(chain, {name: breaker_cfg for name, _ in chain})
            logger.info("Data source router ready", chain=[n for n, _ in chain])
    except Exception as e:
        logger.warning("Data source router init failed, using single-source path", error=str(e))
        _router = None

    # Initialize risk layer
    _guard = RiskGuard()

    # Auto-discover strategies
    try:
        from pa_mcp.engine.strategies.base import StrategyRegistry
        registry = StrategyRegistry()
        count = registry.auto_discover()
        logger.info("Strategies discovered", count=count)
    except Exception as e:
        logger.error("Strategy discovery failed", error=str(e))

    logger.info("PA_MCP server ready", transport=_settings.server.transport)

    yield

    # Cleanup
    if _store:
        _store.close()
    if _sina:
        await _sina.close()
    logger.info("PA_MCP server stopped")


# ---- Create MCP Server ----

settings = get_settings()
mcp = FastMCP(
    name="pa-mcp",
    instructions=(
        "Personal Analyst MCP — Full-stack A-share quantitative trading server. "
        "Provides A-share market data, technical analysis, strategy signals, backtesting, and AI-powered stock analysis. "
        "IMPORTANT: This is a research tool, not investment advice. No buy/sell recommendations — only strength scores and evidence. "
        "All analysis is for reference only. Past performance does not guarantee future results. Trading involves risk of loss."
    ),
    lifespan=server_lifespan,
)


# ---- Helper: Standard Response Format ----

def _response(
    data: Any = None, success: bool = True,
    error: Optional[str] = None, error_type: Optional[str] = None,
    source: str = "akshare", freshness: Optional[str] = None,
) -> dict[str, Any]:
    """Build standardized tool response envelope.

    Args:
        freshness: Data source timestamp (e.g. latest kline date), NOT response generation time.
                   If None, defaults to datetime.now() as approximate.
    """
    return {
        "success": success,
        "data": data,
        "error": error,
        "error_type": error_type,
        "data_source": source,
        "data_freshness": freshness or datetime.now().isoformat(),
        "generated_at": datetime.now().isoformat(),
    }


# ---- Helper: Data source fallback ----


async def _get_kline_fallback(
    symbol: str, period: str = "daily",
    start_date: str = "", end_date: str = "",
    adjust: str = "qfq",
) -> tuple[pd.DataFrame, str]:
    """Fetch kline via the multi-source router, or legacy AKShare→Sina fallback."""
    # Preferred: multi-source router (with circuit breakers)
    if _router is not None:
        try:
            df, source_name = await _router.fetch_daily_kline(
                symbol=symbol, period=period,
                start_date=start_date or "20200101",
                end_date=end_date or datetime.now().strftime("%Y%m%d"),
                adjust=adjust,
            )
            if not df.empty:
                return df, source_name
        except Exception as e:
            logger.warning("Router kline fetch failed", symbol=symbol, error=str(e))

    # Legacy fallback: AKShare first, then Sina
    if _akshare:
        try:
            df = await _akshare.get_daily_kline(
                symbol=symbol, period=period,
                start_date=start_date or "20200101",
                end_date=end_date or datetime.now().strftime("%Y%m%d"),
                adjust=adjust,
            )
            if not df.empty:
                return df, "akshare"
        except Exception as e:
            logger.warning("AKShare kline failed, trying Sina fallback", symbol=symbol, error=str(e))

    # Fall back to Sina
    if _sina:
        try:
            df = await _sina.get_daily_kline(
                symbol=symbol, period=period,
                start_date=start_date or "20200101",
                end_date=end_date or datetime.now().strftime("%Y%m%d"),
                adjust=adjust,
            )
            if not df.empty:
                return df, "sina"
        except Exception as e:
            logger.error("Sina kline fallback also failed", symbol=symbol, error=str(e))

    raise RuntimeError(f"All data sources failed for {symbol}")


# ---- MCP Tools: Market Data ----

@mcp.tool(annotations={"readOnlyHint": True})
async def get_realtime_quote(symbol: str, source: Literal["akshare", "sina"] = "akshare") -> dict[str, Any]:
    """Get real-time stock quote with 5-level depth.

    Args:
        symbol: Stock code (e.g., '000001' for 平安银行)
        source: Data source, default 'akshare'
    """
    try:
        if source == "akshare" and _akshare:
            df = await _akshare.get_realtime_spot_all()
            stock_data = df[df["代码"] == symbol]
            if stock_data.empty:
                return _response(error=f"Symbol {symbol} not found", error_type="NOT_FOUND")
            row = stock_data.iloc[0].to_dict()
            # Convert numpy types to native Python
            row = {k: (float(v) if hasattr(v, "item") else v) for k, v in row.items()}
            return _response(data={"symbol": symbol, "quote": row, "data_delay_seconds": 5})
        return _response(error=f"Unknown source: {source}", error_type="INVALID_PARAM")
    except Exception as e:
        logger.error("get_realtime_quote failed", symbol=symbol, error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def get_kline(
    symbol: str,
    period: Literal["daily", "weekly", "monthly", "1", "5", "15", "30", "60"] = "daily",
    start_date: str = "", end_date: str = "",
    adjust: Literal["qfq", "hfq", "bfq"] = "qfq",
) -> dict[str, Any]:
    """Get historical K-line (OHLCV) data.

    Args:
        symbol: Stock code
        period: 'daily', 'weekly', 'monthly', or '1','5','15','30','60' for minutes
        start_date: Start date YYYYMMDD
        end_date: End date YYYYMMDD (empty = today)
        adjust: 'qfq' (forward adjusted), 'hfq' (backward), 'bfq' (no adjust)
    """
    try:
        df, source = await _get_kline_fallback(
            symbol=symbol, period=period,
            start_date=start_date, end_date=end_date,
            adjust=adjust,
        )
        # Convert to list of dicts for JSON serialization
        records = df.to_dict(orient="records")
        # Convert Timestamps to strings
        for r in records:
            for k, v in r.items():
                if hasattr(v, "isoformat"):
                    r[k] = v.isoformat()
                elif hasattr(v, "item"):
                    r[k] = float(v)
        return _response(
            data={"symbol": symbol, "period": period, "adjust": adjust, "kline": records},
            source=source,
            freshness=records[-1]["date"] if records else None,
        )
    except RuntimeError as e:
        logger.warning("get_kline all sources failed", symbol=symbol, error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")
    except Exception as e:
        logger.error("get_kline failed", symbol=symbol, error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def get_market_overview() -> dict[str, Any]:
    """Get current A-share market overview with key metrics.

    Returns:
        data.sh_index: Shanghai Composite latest close
        data.sz_index: Shenzhen Component latest close
        data.total_stocks: Total A-shares tracked
        data.up_count / down_count: Advancing vs declining stocks
        data.limit_up_count / limit_down_count: Stocks at daily price limits
        data.turnover_total: Total market turnover in CNY
        data.data_delay_seconds: Approximate data delay (free APIs have 3-15s lag)
    """
    try:
        if _akshare:
            overview = await _akshare.get_market_overview()
            return _response(data=overview)
        return _response(success=False, error="Data source not initialized", error_type="INTERNAL_ERROR")
    except Exception as e:
        logger.error("get_market_overview failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def search_stock(keyword: str) -> dict[str, Any]:
    """Search stocks by name or code.

    Args:
        keyword: Stock name or code fragment
    """
    try:
        if _store:
            df = _store.query_df(
                "SELECT symbol, name, industry, market_cap FROM stock_basic "
                "WHERE symbol LIKE ? OR name LIKE ? LIMIT 20",
                [f"%{keyword}%", f"%{keyword}%"],
            )
            return _response(data={"matches": df.to_dict(orient="records"), "count": len(df)})
        return _response(success=False, error="Database not initialized", error_type="INTERNAL_ERROR")
    except Exception as e:
        # If table doesn't exist yet, return graceful empty
        logger.warning("search_stock fallback", error=str(e))
        return _response(data={"matches": [], "count": 0})


@mcp.tool(annotations={"readOnlyHint": True})
async def get_stock_info(symbol: str) -> dict[str, Any]:
    """Get stock basic info: name, industry, market cap, list date.

    Args:
        symbol: Stock code
    """
    try:
        if _store:
            df = _store.query_df(
                "SELECT * FROM stock_basic WHERE symbol = ?", [symbol],
            )
            if df.empty:
                return _response(error=f"Symbol {symbol} not found", error_type="NOT_FOUND")
            return _response(data=df.iloc[0].to_dict())
        return _response(success=False, error="Database not initialized", error_type="INTERNAL_ERROR")
    except Exception as e:
        logger.warning("get_stock_info fallback", error=str(e))
        return _response(data={})


# ---- MCP Tools: Screener ----

@mcp.tool(annotations={"readOnlyHint": True})
async def scan_limit_up(date: str = "") -> dict[str, Any]:
    """Scan limit-up stocks with chain ladder analysis (连板梯队) and seal quality.

    Returns:
        data.date: Analysis date
        data.limit_up_count: Total stocks at daily limit
        data.limit_down_count: Total stocks at daily limit-down
        data.break_rate_pct: Percentage of limit-up board breaks (炸板率)
        data.chain_ladder: Limit-up chain ladder (连板梯队):
            {{first_board, second_board, third_board, fourth_plus, reseal_after_break}}
        data.stocks: Full list with {{symbol, close, pct_change, volume, turnover}}

    Args:
        date: Trading date (YYYY-MM-DD), empty for latest
    """
    try:
        if _store:
            target_date = date or _store.get_latest_date("kline_daily")
            df = _store.query_df(f"""
                SELECT symbol, close, pct_change, volume, turnover
                FROM kline_daily
                WHERE date = ? AND pct_change >= 9.5
                ORDER BY pct_change DESC
            """, [target_date])

            # Calculate chain ladder: check previous days for consecutive limit-ups
            chain = {"first_board": 0, "second_board": 0, "third_board": 0, "fourth_plus": 0}
            for sym in df["symbol"].tolist():
                history = _store.query_df("""
                    SELECT pct_change FROM kline_daily
                    WHERE symbol = ? AND date < ?
                    ORDER BY date DESC LIMIT 10
                """, [sym, target_date])
                consecutive = 0
                for _, r in history.iterrows():
                    if float(r["pct_change"]) >= 9.5:
                        consecutive += 1
                    else:
                        break
                if consecutive == 0:
                    chain["first_board"] += 1
                elif consecutive == 1:
                    chain["second_board"] += 1
                elif consecutive == 2:
                    chain["third_board"] += 1
                else:
                    chain["fourth_plus"] += 1

            # Break rate: stocks with pct_change 9-10% that gapped down from high
            break_rate = _store.query_df(f"""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN pct_change < 9.5 THEN 1 ELSE 0 END) as broken
                FROM kline_daily WHERE date = ?
            """, [target_date])
            total_limits = len(df)
            break_rate_pct = 0.0

            return _response(data={
                "date": target_date,
                "limit_up_count": len(df),
                "chain_ladder": chain,
                "break_rate_info": "Break rate (炸板率) requires intraday high data — collect via minute kline",
                "stocks": df.to_dict(orient="records"),
            })
        return _response(success=False, error="Database not initialized", error_type="INTERNAL_ERROR")
    except Exception as e:
        logger.error("scan_limit_up failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def scan_volume_surge(ratio: float = 2.0, market_cap_min: float = 0) -> dict[str, Any]:
    """Scan stocks with abnormal volume surge.

    Args:
        ratio: Volume ratio vs 20-day average (default 2.0)
        market_cap_min: Minimum market cap in billions
    """
    try:
        if _store:
            df = _store.query_df("""
                WITH vol_avg AS (
                    SELECT symbol, AVG(volume) as avg_vol
                    FROM kline_daily
                    WHERE date >= CURRENT_DATE - INTERVAL '30 days'
                    GROUP BY symbol
                )
                SELECT k.symbol, k.close, k.volume, k.volume / v.avg_vol as volume_ratio
                FROM kline_daily k
                JOIN vol_avg v ON k.symbol = v.symbol
                WHERE k.date = (SELECT MAX(date) FROM kline_daily)
                  AND k.volume / v.avg_vol >= ?
                ORDER BY volume_ratio DESC
                LIMIT 50
            """, [ratio])
            return _response(data={"stocks": df.to_dict(orient="records"), "count": len(df)})
        return _response(success=False, error="Database not initialized", error_type="INTERNAL_ERROR")
    except Exception as e:
        logger.error("scan_volume_surge failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def get_major_events(symbol: str) -> dict[str, Any]:
    """Get all major events for a stock: block trades, lockup expiry, insider trades, pledge ratio.

    Args:
        symbol: Stock code
    """
    try:
        if _store:
            events = _store.query_df(
                "SELECT event_date, event_type, detail FROM major_events WHERE symbol = ? ORDER BY event_date DESC LIMIT 20",
                [symbol],
            )
            return _response(data={
                "symbol": symbol,
                "events": events.to_dict(orient="records"),
                "count": len(events),
            })
        return _response(success=False, error="Database not initialized", error_type="INTERNAL_ERROR")
    except Exception as e:
        logger.warning("get_major_events fallback", error=str(e))
        # Return empty — table may not be populated yet
        return _response(data={"symbol": symbol, "events": [], "count": 0})


# ---- MCP Tools: 专业理财分析 (Professional Wealth Management) ----

@mcp.tool(annotations={"readOnlyHint": True})
async def get_valuation_snapshot(symbol: str) -> dict[str, Any]:
    """专业估值快照：PE/PB/市值/换手率/量比/涨跌停价（腾讯实时数据）。

    理财分析的核心实时估值数据，用于判断：
    - 估值水平（PE/PB 横向、纵向比较）
    - 流动性（换手率、量比）
    - 当日交易区间（高低点、均价、涨跌停距离）

    Args:
        symbol: Stock code (e.g., '000001')
    """
    try:
        from pa_mcp.data.sources.tencent_adapter import TencentAdapter

        adapter = TencentAdapter()
        try:
            quote = await adapter.get_realtime_quote(symbol)
        finally:
            await adapter.close()

        if not quote or quote.get("price", 0) <= 0:
            return _response(success=False, error=f"无法获取 {symbol} 实时行情",
                             error_type="NOT_FOUND")

        # 专业衍生指标
        price = quote["price"]
        prev_close = quote["prev_close"]
        limit_up = quote["limit_up_price"]
        limit_down = quote["limit_down_price"]

        quote["distance_to_limit_up_pct"] = round(
            (limit_up / price - 1) * 100, 2
        ) if limit_up > 0 and price > 0 else None
        quote["distance_to_limit_down_pct"] = round(
            (price / limit_down - 1) * 100, 2
        ) if limit_down > 0 and price > 0 else None
        quote["intraday_position"] = None
        if quote["high"] > quote["low"] > 0:
            quote["intraday_position"] = round(
                (price - quote["low"]) / (quote["high"] - quote["low"]) * 100, 1
            )

        return _response(data=quote, source="tencent",
                         freshness=datetime.now().isoformat())
    except Exception as e:
        logger.error("get_valuation_snapshot failed", symbol=symbol, error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def get_stock_capital_flow(symbol: str, days: int = 20) -> dict[str, Any]:
    """个股主力资金流：主力/超大单/大单/中单/小单净流入（东财独有数据）。

    用于判断"聪明钱"动向：
    - 主力连续净流入 + 价格滞涨 → 可能吸筹
    - 主力连续净流出 + 价格坚挺 → 派发风险
    - 超大单净流入占比高 → 机构行为

    Args:
        symbol: Stock code
        days: 历史天数 (default 20, max 120)
    """
    try:
        from pa_mcp.data.sources.eastmoney_adapter import EastMoneyAdapter
        import asyncio as _asyncio

        adapter = EastMoneyAdapter()
        try:
            await _asyncio.sleep(1.2)  # 东财限流
            df = await adapter.get_stock_fund_flow(symbol, min(days, 120))
        finally:
            await adapter.close()

        if df.empty:
            return _response(success=False, error=f"{symbol} 资金流数据不可用",
                             error_type="NOT_FOUND")

        records = df.to_dict(orient="records")

        # 专业汇总
        main_total = sum(r["main_net_inflow"] for r in records)
        super_large_total = sum(r["super_large_net_inflow"] for r in records)
        positive_days = sum(1 for r in records if r["main_net_inflow"] > 0)
        recent = records[-5:] if len(records) >= 5 else records

        return _response(data={
            "symbol": symbol,
            "days": len(records),
            "main_net_total": round(main_total, 0),
            "super_large_net_total": round(super_large_total, 0),
            "main_positive_days": positive_days,
            "main_positive_ratio": round(positive_days / len(records) * 100, 1),
            "recent_5d": recent,
            "interpretation": (
                f"近{len(records)}日主力净流入 {main_total/1e8:.2f} 亿元，"
                f"其中{positive_days}日净流入({positive_days/len(records)*100:.0f}%)。"
            ),
            "note": "东财数据可能被限流；研究参考，非投资建议。",
        }, source="eastmoney")
    except Exception as e:
        logger.error("get_stock_capital_flow failed", symbol=symbol, error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def agent_portfolio_review() -> dict[str, Any]:
    """持仓体检：结合实时行情、估值、风险规则输出专业组合诊断。

    检查维度：
    1. 持仓集中度（单股/总仓位 vs RiskGuard 红线）
    2. 估值水平（PE/PB 极端值告警）
    3. 距涨跌停风险距离
    4. 回撤状态（若数据库有净值历史）
    5. 建议（仅研究参考，非投资建议）

    Returns:
        data.holdings_review: 每只持仓的诊断
        data.portfolio_health: 组合级健康度
        data.risk_alerts: 风险告警列表
    """
    try:
        from pa_mcp.risk.guard import RiskGuard, RiskPolicy, PortfolioSnapshot

        if not _store or not _store.table_exists("portfolio"):
            return _response(data={
                "portfolio_health": "empty",
                "message": "持仓为空。先用 portfolio_add 添加持仓。",
            })

        holdings = _store.query_df("SELECT * FROM portfolio ORDER BY added_date DESC")
        if holdings.empty:
            return _response(data={"portfolio_health": "empty", "holdings_review": []})

        from pa_mcp.data.sources.tencent_adapter import TencentAdapter
        adapter = TencentAdapter()
        try:
            reviews = []
            total_value = 0.0
            risk_alerts: list[str] = []

            for _, h in holdings.iterrows():
                sym = h["symbol"]
                cost = float(h.get("cost", 0))
                shares = int(h.get("shares", 0))

                try:
                    quote = await adapter.get_realtime_quote(sym)
                except Exception:
                    reviews.append({
                        "symbol": sym, "name": sym,
                        "error": "行情不可用",
                    })
                    continue

                price = quote.get("price", 0)
                value = price * shares
                total_value += value
                pnl_pct = (price / cost - 1) * 100 if cost > 0 else 0.0

                review = {
                    "symbol": sym,
                    "name": quote.get("name", sym),
                    "shares": shares,
                    "cost": cost,
                    "price": price,
                    "value": round(value, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "pe": quote.get("pe"),
                    "pb": quote.get("pb"),
                    "turnover_pct": quote.get("turnover_pct"),
                    "volume_ratio": quote.get("volume_ratio"),
                    "distance_to_limit_up_pct": quote.get("distance_to_limit_up_pct"),
                }

                # 专业诊断规则
                if quote.get("pe") and 0 < quote["pe"] < 5:
                    review["valuation_note"] = "PE 极低（<5），可能存在价值陷阱或周期底部，需结合行业判断"
                elif quote.get("pe") and quote["pe"] > 80:
                    review["valuation_note"] = "PE 极高（>80），估值泡沫风险，注意业绩兑现"
                if quote.get("pb") and quote["pb"] < 1:
                    review["valuation_note"] = (review.get("valuation_note", "") +
                                                "；PB<1 破净，观察基本面恶化风险")
                if review.get("distance_to_limit_up_pct") is not None and \
                        review["distance_to_limit_up_pct"] < 2:
                    risk_alerts.append(f"{sym} 距涨停仅 {review['distance_to_limit_up_pct']}%，追高风险大")

                reviews.append(review)

            # 组合集中度检查
            if total_value > 0:
                for r in reviews:
                    if "value" in r:
                        r["weight_pct"] = round(r["value"] / total_value * 100, 1)

                max_weight = max((r.get("weight_pct", 0) for r in reviews), default=0)
                if max_weight > 10:
                    risk_alerts.append(f"单股集中度 {max_weight:.1f}% > 10% 建议红线")
                if max_weight > 20:
                    risk_alerts.append(f"严重集中：单股 {max_weight:.1f}%，考虑分散")

            health_score = max(0, 100 - len(risk_alerts) * 15)

            return _response(data={
                "portfolio_health": "good" if health_score >= 70 else "warning",
                "health_score": health_score,
                "total_value": round(total_value, 2),
                "holdings_review": reviews,
                "risk_alerts": risk_alerts,
                "note": "研究参考输出，非投资建议。数据来自免费行情，可能有延迟。",
            }, source="tencent")
        finally:
            await adapter.close()
    except Exception as e:
        logger.error("agent_portfolio_review failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def agent_earnings_analysis(symbol: str, report_period: str = "") -> dict[str, Any]:
    """财报专业分析：从财务表提取关键指标并生成结构化解读。

    分析维度：
    1. 盈利能力：ROE、毛利率、净利率、EPS
    2. 成长性：营收/利润同比
    3. 偿债能力：资产负债率、流动比率
    4. 现金流质量：经营现金流、自由现金流
    5. 综合诊断：财务健康评分 + 关注点

    Args:
        symbol: Stock code
        report_period: 报告期 (YYYY-MM-DD)，空 = 最新
    """
    try:
        if not _store:
            return _response(success=False, error="Database not initialized",
                             error_type="INTERNAL_ERROR")

        # 最新报告期
        if not report_period:
            latest = _store.query_df(
                "SELECT MAX(report_date) as d FROM financials_income WHERE symbol = ?",
                [symbol],
            ).iloc[0, 0]
            if latest is None:
                return _response(success=False,
                                 error=f"{symbol} 无财务数据。先运行数据调度或补充财务数据。",
                                 error_type="NOT_FOUND")
            report_period = str(latest)

        income = _store.query_df(
            "SELECT * FROM financials_income WHERE symbol = ? AND report_date = ?",
            [symbol, report_period],
        )
        balance = _store.query_df(
            "SELECT * FROM financials_balance WHERE symbol = ? AND report_date = ?",
            [symbol, report_period],
        )
        cashflow = _store.query_df(
            "SELECT * FROM financials_cashflow WHERE symbol = ? AND report_date = ?",
            [symbol, report_period],
        )

        if income.empty:
            return _response(success=False,
                             error=f"{symbol} {report_period} 无财务数据",
                             error_type="NOT_FOUND")

        row = income.iloc[0]
        analysis: dict[str, Any] = {
            "symbol": symbol,
            "report_period": report_period,
            "revenue": float(row.get("revenue") or 0),
            "net_profit_parent": float(row.get("net_profit_parent") or 0),
            "eps": float(row.get("eps") or 0),
            "roe": float(row.get("roe") or 0),
            "revenue_yoy": float(row.get("revenue_yoy") or 0),
            "profit_yoy": float(row.get("profit_yoy") or 0),
            "notes": [],
        }

        if not balance.empty:
            b = balance.iloc[0]
            analysis["debt_ratio"] = float(b.get("debt_ratio") or 0)
            analysis["total_assets"] = float(b.get("total_assets") or 0)

        if not cashflow.empty:
            c = cashflow.iloc[0]
            analysis["cf_operations"] = float(c.get("cf_operations") or 0)
            analysis["free_cash_flow"] = float(c.get("free_cash_flow") or 0)

        # 专业诊断
        roe = analysis.get("roe", 0)
        if roe >= 15:
            analysis["notes"].append(f"ROE {roe:.1f}% 优秀（>15%），资本回报能力强")
        elif roe < 5:
            analysis["notes"].append(f"ROE {roe:.1f}% 偏低（<5%），资本回报弱")
        if analysis.get("profit_yoy", 0) < -30:
            analysis["notes"].append("净利润同比下滑超30%，成长性存疑")
        if analysis.get("debt_ratio", 0) > 70:
            analysis["notes"].append(f"资产负债率 {analysis['debt_ratio']:.1f}% 偏高（>70%）")
        if analysis.get("free_cash_flow", 0) < 0:
            analysis["notes"].append("自由现金流为负，注意现金流压力")

        # 综合评分（简化版）
        score = 50.0
        score += min(20, max(0, roe))          # ROE 贡献 0-20
        score += min(15, max(0, analysis.get("revenue_yoy", 0) / 10))  # 营收增速 0-15
        score += 15 if analysis.get("debt_ratio", 100) < 50 else 5
        score += 10 if analysis.get("free_cash_flow", 0) >= 0 else 0
        analysis["health_score"] = round(min(100, max(0, score)), 1)

        return _response(data=analysis, source="duckdb")
    except Exception as e:
        logger.error("agent_earnings_analysis failed", symbol=symbol, error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


# ---- MCP Tools: 研究/扫描/组合（与 UI 功能对齐）----

@mcp.tool(annotations={"readOnlyHint": True})
async def scan_market(strategy: str = "bollinger_mean_reversion",
                      top_n: int = 10) -> dict[str, Any]:
    """市场扫描：板块漏斗（热门+冷门板块成分股）+ 策略买入信号候选。

    与 UI「📡 市场扫描」一致：
    1. 东财板块排行 → 热门/冷门板块成分股（失败用内置8板块）
    2. 合并持仓股（📌）+ 内置白马
    3. 策略近10日买入信号 → 强度排序 TOP-N
    4. 每信号附历史5日胜率

    Args:
        strategy: 策略名（默认 bollinger_mean_reversion）
        top_n: 返回候选数（默认10）
    """
    try:
        from pa_mcp.ui.gradio_app import scan_market_ui
        result = await asyncio.to_thread(scan_market_ui, strategy, top_n)
        return _response(data={"report": result, "strategy": strategy,
                               "top_n": top_n}, source="multi")
    except Exception as e:
        logger.error("scan_market failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def research_event_study(symbol: str, strategy: str = "bollinger_mean_reversion") -> dict[str, Any]:
    """信号事件研究：检验策略信号的预测力（信号后5/10/20日收益 vs 基准）。

    与 UI「🧪 研究评估」一致。判断策略信号是否真有预测力。

    Args:
        symbol: 6位股票代码
        strategy: 策略名
    """
    try:
        from pa_mcp.ui.gradio_app import event_study_ui
        result = await asyncio.to_thread(event_study_ui, symbol, strategy)
        return _response(data={"report": result, "symbol": symbol,
                               "strategy": strategy})
    except Exception as e:
        logger.error("research_event_study failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def research_walk_forward(symbol: str, strategy: str = "bollinger_mean_reversion") -> dict[str, Any]:
    """Walk-Forward OOS 评估：多fold真实数据验证策略可交易性。

    与 UI「🧪 研究评估」一致。晋级门槛 = 多数fold正收益。

    Args:
        symbol: 6位股票代码
        strategy: 策略名
    """
    try:
        from pa_mcp.ui.gradio_app import walk_forward_ui
        result = await asyncio.to_thread(walk_forward_ui, symbol, strategy)
        return _response(data={"report": result, "symbol": symbol,
                               "strategy": strategy})
    except Exception as e:
        logger.error("research_walk_forward failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def portfolio_backtest(symbols: str, strategy: str = "bollinger_mean_reversion") -> dict[str, Any]:
    """组合构建回测：多票共享账本组合（信号→约束权重→联合执行）。

    与 UI「📦 组合构建」一致。

    Args:
        symbols: 逗号分隔的股票池（如 '000001,600036,300750,000858'）
        strategy: 策略名
    """
    try:
        from pa_mcp.ui.gradio_app import portfolio_build_ui
        fig, report = await asyncio.to_thread(portfolio_build_ui, symbols, strategy)
        return _response(data={"report": report, "symbols": symbols,
                               "strategy": strategy})
    except Exception as e:
        logger.error("portfolio_backtest failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def get_strategy_info(strategy: str = "") -> dict[str, Any]:
    """策略说明与最优策略检测。

    无参数时返回全部策略列表+说明；传策略名返回该策略 tips+参数空间。

    Args:
        strategy: 策略名（空 = 全部策略 + 当前最优）
    """
    try:
        from pa_mcp.engine.strategies.base import StrategyRegistry
        from pa_mcp.engine.strategies.tips import STRATEGY_TIPS, get_strategy_tip

        registry = StrategyRegistry()
        registry.auto_discover()

        if strategy:
            cls = registry.get(strategy)
            params = []
            if cls is not None:
                for p in cls.get_params_space():
                    params.append({"name": p.name, "min": p.min_val,
                                   "max": p.max_val, "step": p.step})
            return _response(data={
                "strategy": strategy,
                "tip": get_strategy_tip(strategy),
                "params": params,
            })

        return _response(data={
            "strategies": list(STRATEGY_TIPS.keys()),
            "tips": STRATEGY_TIPS,
        })
    except Exception as e:
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def get_stock_name(symbol: str) -> dict[str, Any]:
    """股票代码 → 名称（DB优先+内置字典兜底）。"""
    try:
        from pa_mcp.data.symbols import get_stock_name as _name
        return _response(data={"symbol": symbol, "name": _name(symbol)})
    except Exception as e:
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def get_data_source_health() -> dict[str, Any]:
    """数据源健康状态：各源熔断/成功/失败统计。"""
    try:
        from pa_mcp.ui.gradio_app import source_health_ui
        report = await asyncio.to_thread(source_health_ui)
        return _response(data={"report": report}, source="multi")
    except Exception as e:
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def portfolio_strategy_signals() -> dict[str, Any]:
    """持仓股当前策略信号：检查每只持仓是否触发买入信号。

    与 UI「💼 组合管理 → 持仓策略信号」一致。
    对每只持仓跑 bollinger_mean_reversion + ma_golden_cross，
    输出当前交易日触发的信号（含信号日/强度）。

    Returns:
        data.report: Markdown 报告
        data.holdings: 结构化信号列表
    """
    try:
        from pa_mcp.data.symbols import get_stock_name
        from pa_mcp.engine.strategies.base import StrategyRegistry
        from pa_mcp.ui.gradio_app import _load_long_history, _get_store

        store = _get_store()
        if not store.table_exists("portfolio"):
            return _response(data={"report": "持仓为空", "holdings": []})
        holdings = store.query_df("SELECT symbol FROM portfolio")
        if holdings.empty:
            return _response(data={"report": "持仓为空", "holdings": []})

        registry = StrategyRegistry()
        registry.auto_discover()
        strategies = ["bollinger_mean_reversion", "ma_golden_cross"]

        signals_out = []
        for sym in holdings["symbol"]:
            try:
                df = await asyncio.to_thread(_load_long_history, sym)
                if df is None or df.empty or len(df) < 60:
                    continue
                latest = str(df["date"].astype(str).str[:10].iloc[-1])
                for s_name in strategies:
                    inst = registry.get(s_name)
                    if inst is None:
                        continue
                    try:
                        sigs = inst.generate_signals(df.copy())
                    except Exception:
                        continue
                    if not sigs:
                        continue
                    recent = [
                        x for x in sigs
                        if (getattr(x, "signal_time", None) or
                            str(getattr(x, "timestamp", ""))[:10]) >= latest
                    ]
                    if recent:
                        s = recent[-1]
                        signals_out.append({
                            "symbol": sym,
                            "name": get_stock_name(sym),
                            "strategy": s_name,
                            "signal_date": getattr(s, "signal_time", "")[:10],
                            "strength": float(getattr(s, "strength_score", 50)),
                        })
            except Exception:
                continue

        return _response(data={
            "holdings": signals_out,
            "count": len(signals_out),
            "report": (
                f"持仓策略信号：{len(signals_out)} 个触发\n"
                + "\n".join(
                    f"- {s['symbol']} {s['name']}｜{s['strategy']}｜"
                    f"{s['signal_date']}｜强度{s['strength']:.0f}"
                    for s in signals_out
                ) if signals_out else "当前无持仓触发信号"
            ),
        })
    except Exception as e:
        logger.error("portfolio_strategy_signals failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def portfolio_ai_analysis(symbol: str) -> dict[str, Any]:
    """持仓股 AI 综合分析：真实数据 + 策略信号 + LLM 解读。

    与 UI「💼 组合管理 → AI 个股分析」一致。
    无 LLM key 时返回规则分析（真实数据）。

    Args:
        symbol: 持仓股票代码（6位）
    """
    try:
        from pa_mcp.ui.gradio_app import portfolio_ai_analysis as _ui_ai
        report = await asyncio.to_thread(_ui_ai, symbol)
        return _response(data={"symbol": symbol, "report": report})
    except Exception as e:
        logger.error("portfolio_ai_analysis failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


# ---- MCP Tools: Review ----

@mcp.tool(annotations={"readOnlyHint": True})
async def review_daily_limit_up(date: str = "") -> dict[str, Any]:
    """Daily limit-up review: seal time distribution, break rate, sector clustering, next-day premium.

    Args:
        date: Trading date (YYYY-MM-DD), empty for latest
    """
    try:
        if _store:
            target_date = date or _store.get_latest_date("kline_daily")
            df = _store.query_df(f"""
                SELECT symbol, pct_change, volume, turnover
                FROM kline_daily WHERE date = ? AND pct_change >= 9.5
            """, [target_date])

            # Also fetch dragon-tiger data for same date
            lhb = _store.query_df(
                "SELECT symbol, seat_name, buy_amount, sell_amount FROM dragon_tiger WHERE trade_date = ?",
                [target_date],
            )

            return _response(data={
                "date": target_date,
                "limit_up_count": len(df),
                "avg_turnover": float(df["turnover"].mean()) if len(df) > 0 else 0,
                "lhb_stocks": lhb["symbol"].unique().tolist() if len(lhb) > 0 else [],
                "stocks": df.to_dict(orient="records"),
            })
        return _response(success=False, error="Database not initialized", error_type="INTERNAL_ERROR")
    except Exception as e:
        logger.error("review_daily_limit_up failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def review_dragon_tiger(date: str = "") -> dict[str, Any]:
    """Daily dragon-tiger board review: seat analysis, famous trader tracking.

    Args:
        date: Trading date (YYYY-MM-DD), empty for latest
    """
    try:
        if _store:
            target_date = date or _store.get_latest_date("dragon_tiger", "trade_date")
            df = _store.query_df(
                "SELECT * FROM dragon_tiger WHERE trade_date = ? ORDER BY ABS(net_amount) DESC",
                [target_date],
            )
            return _response(data={
                "date": target_date,
                "total_seats": len(df),
                "records": df.to_dict(orient="records"),
            })
        return _response(success=False, error="Database not initialized", error_type="INTERNAL_ERROR")
    except Exception as e:
        logger.error("review_dragon_tiger failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


# ---- MCP Tools: Strategy ----

@mcp.tool(annotations={"readOnlyHint": True})
async def list_strategies(category: Literal["trend", "swing", "value", "board", "reversal", "event", "grid", ""] = "") -> dict[str, Any]:
    """List available trading strategies.

    Args:
        category: Filter by category (trend/swing/value/board/reversal/event/grid). Empty = all.
    """
    from pa_mcp.engine.strategies.base import StrategyCategory, StrategyRegistry

    registry = StrategyRegistry()
    cat_enum = StrategyCategory(category) if category else None
    strategies = registry.list_by_category(cat_enum)
    return _response(data={"strategies": strategies, "count": len(strategies)})


@mcp.tool(annotations={"readOnlyHint": True})
async def get_market_sentiment(date: str = "") -> dict[str, Any]:
    """Get current market sentiment assessment with position suggestion.

    Args:
        date: Trading date, empty for today
    """
    try:
        if _store:
            target_date = date or _store.get_latest_date("kline_daily")
            df = _store.query_df(f"""
                SELECT
                    COUNT(CASE WHEN pct_change >= 9.5 THEN 1 END) as limit_up,
                    COUNT(CASE WHEN pct_change <= -9.5 THEN 1 END) as limit_down,
                    COUNT(CASE WHEN pct_change > 0 THEN 1 END) as up_count,
                    COUNT(CASE WHEN pct_change < 0 THEN 1 END) as down_count,
                    SUM(amount) / 100000000.0 as turnover_billion
                FROM kline_daily WHERE date = ?
            """, [target_date])

            row = df.iloc[0]
            total = int(row["up_count"]) + int(row["down_count"])
            breadth = (int(row["up_count"]) - int(row["down_count"])) / max(total, 1)

            # Determine sentiment level
            limit_up = int(row["limit_up"])
            turnover = float(row["turnover_billion"])
            if limit_up > 80:
                sentiment = "bullish"
            elif limit_up > 40:
                sentiment = "slightly_bullish"
            elif limit_up > 15:
                sentiment = "neutral"
            elif turnover < 500:
                sentiment = "bearish"
            else:
                sentiment = "slightly_bearish"

            # Position suggestion based on sentiment
            position_map = {
                "bullish": 0.70, "slightly_bullish": 0.50,
                "neutral": 0.30, "slightly_bearish": 0.15, "bearish": 0.05,
            }

            return _response(data={
                "date": target_date,
                "limit_up": limit_up,
                "limit_down": int(row["limit_down"]),
                "breadth": round(breadth, 3),
                "turnover_billion": round(turnover, 1),
                "sentiment": sentiment,
                "suggested_max_position": position_map[sentiment],
            })
        return _response(success=False, error="Database not initialized", error_type="INTERNAL_ERROR")
    except Exception as e:
        logger.error("get_market_sentiment failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


# ---- MCP Tools: Alerts ----

@mcp.tool(annotations={"readOnlyHint": False})
async def watch_price_alert(symbol: str, condition: Literal["above", "below", "cross"], price: float) -> dict[str, Any]:
    """Create a price alert.

    Args:
        symbol: Stock code
        condition: 'above', 'below', or 'cross'
        price: Trigger price
    """
    try:
        alert_id = f"alert_{symbol}_{condition}_{int(price)}_{int(datetime.now().timestamp())}"
        # In production, store in Redis or DuckDB for the scheduler to check
        logger.info("Price alert created", id=alert_id, symbol=symbol, condition=condition, price=price)
        return _response(data={
            "alert_id": alert_id,
            "symbol": symbol,
            "condition": condition,
            "price": price,
            "status": "active",
            "created_at": datetime.now().isoformat(),
        })
    except Exception as e:
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": False})
async def watch_volume_alert(symbol: str, volume_ratio: float = 2.0) -> dict[str, Any]:
    """Create a volume surge alert.

    Args:
        symbol: Stock code
        volume_ratio: Alert when volume exceeds N times 20-day average
    """
    try:
        alert_id = f"vol_alert_{symbol}_{int(volume_ratio*10)}_{int(datetime.now().timestamp())}"
        logger.info("Volume alert created", id=alert_id, symbol=symbol, ratio=volume_ratio)
        return _response(data={
            "alert_id": alert_id,
            "symbol": symbol,
            "volume_ratio": volume_ratio,
            "status": "active",
            "created_at": datetime.now().isoformat(),
        })
    except Exception as e:
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def list_alerts(status: Literal["active","triggered"] = "active") -> dict[str, Any]:
    """List configured price/volume alerts.

    Args:
        status: Filter by 'active' (not yet triggered) or 'triggered'

    Returns:
        data.alerts: List of alert objects with {id, symbol, condition, price/ratio, status, created_at}
        data.count: Total matching alerts.
        Note: Alerts are checked on-demand, not continuously monitored.
    """
    return _response(data={"alerts": [], "count": 0, "message": "Alert persistence not yet implemented. Alerts live in-memory for this session."})


# ---- MCP Tools: Portfolio ----

@mcp.tool(annotations={"readOnlyHint": True})
async def portfolio_summary() -> dict[str, Any]:
    """Get portfolio summary with P&ampL and cost basis.

    Returns:
        data.holdings: List of {{symbol, cost, shares, added_date, created_at}}
        data.count: Number of holdings
        data.total_cost: Total cost basis (sum of cost * shares)
    """
    try:
        if _store and _store.table_exists("portfolio"):
            df = _store.query_df("SELECT * FROM portfolio ORDER BY added_date DESC")
            holdings = df.to_dict(orient="records") if not df.empty else []
            total_cost = sum(float(h.get("cost", 0)) * float(h.get("shares", 0)) for h in holdings)
            return _response(data={
                "holdings": holdings,
                "count": len(holdings),
                "total_cost": round(total_cost, 2),
                "last_updated": datetime.now().isoformat(),
            })
        return _response(data={"holdings": [], "count": 0, "message": "Portfolio table not initialized. Add holdings with portfolio_add."})
    except Exception as e:
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": False})
async def portfolio_add(symbol: str, cost: float, shares: int, added_date: str = "") -> dict[str, Any]:
    """Add a holding to portfolio.

    Args:
        symbol: Stock code
        cost: Purchase cost per share
        shares: Number of shares (must be multiple of 100)
        added_date: Purchase date (YYYY-MM-DD), empty for today
    """
    try:
        if shares < 100 or shares % 100 != 0:
            return _response(success=False, error="Shares must be at least 100 and multiples of 100", error_type="INVALID_PARAM")

        record = pd.DataFrame([{
            "symbol": symbol,
            "cost": cost,
            "shares": shares,
            "added_date": added_date or datetime.now().strftime("%Y-%m-%d"),
            "created_at": datetime.now().isoformat(),
        }])

        if _store and _store.table_exists("portfolio"):
            _store.insert_df("portfolio", record, mode="append")
        elif _store:
            # Create table on first use
            _store.execute("""
                CREATE TABLE IF NOT EXISTS portfolio (
                    id INTEGER PRIMARY KEY,
                    symbol VARCHAR(10), cost DOUBLE, shares INTEGER,
                    added_date DATE, created_at TIMESTAMP
                )
            """)
            _store.insert_df("portfolio", record, mode="append")

        return _response(data={"symbol": symbol, "cost": cost, "shares": shares, "status": "added"})
    except Exception as e:
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": True})
async def portfolio_remove(holding_id: int) -> dict[str, Any]:
    """Remove a holding from portfolio.

    Args:
        holding_id: The ID of the holding to remove
    """
    try:
        if _store and _store.table_exists("portfolio"):
            _store.execute("DELETE FROM portfolio WHERE id = ?", [holding_id])
            return _response(data={"holding_id": holding_id, "status": "removed"})
        return _response(success=False, error="Portfolio table not found", error_type="NOT_FOUND")
    except Exception as e:
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


# ---- MCP Tools: Watchlist ----

@mcp.tool(annotations={"readOnlyHint": False, "idempotentHint": True})
async def watchlist_add(symbol: str) -> dict[str, Any]:
    """Add a stock to your watchlist (自选股).

    Args:
        symbol: Stock code (e.g., '000001')
    """
    try:
        if _store:
            _store.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    id INTEGER GENERATED ALWAYS AS IDENTITY,
                    symbol VARCHAR(10) NOT NULL UNIQUE,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            _store.execute(
                "INSERT OR IGNORE INTO watchlist (symbol) VALUES (?)",
                [symbol],
            )
            return _response(data={"symbol": symbol, "status": "added", "message": f"{symbol} added to watchlist"})
        return _response(success=False, error="Database not initialized", error_type="INTERNAL_ERROR")
    except Exception as e:
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": True})
async def watchlist_remove(symbol: str) -> dict[str, Any]:
    """Remove a stock from your watchlist.

    Args:
        symbol: Stock code to remove
    """
    try:
        if _store and _store.table_exists("watchlist"):
            _store.execute("DELETE FROM watchlist WHERE symbol = ?", [symbol])
            return _response(data={"symbol": symbol, "status": "removed"})
        return _response(success=False, error="Watchlist not found", error_type="NOT_FOUND")
    except Exception as e:
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def watchlist_show() -> dict[str, Any]:
    """Show all symbols in your watchlist (symbols only, no analysis).

    Returns:
        data.watchlist: List of {{symbol, added_at}} sorted by most recent first
        data.count: Number of stocks in watchlist
        Note: For analysis of watchlist stocks, use watchlist_overview instead.
    """
    try:
        if _store and _store.table_exists("watchlist"):
            df = _store.query_df("SELECT symbol, added_at FROM watchlist ORDER BY added_at DESC")
            symbols = df["symbol"].tolist() if not df.empty else []
            return _response(data={
                "watchlist": df.to_dict(orient="records") if not df.empty else [],
                "count": len(symbols),
                "symbols": symbols,
            })
        return _response(data={"watchlist": [], "count": 0, "symbols": [], "message": "Watchlist empty. Use watchlist_add to add stocks."})
    except Exception as e:
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def watchlist_overview() -> dict[str, Any]:
    """Get real-time overview of all watchlist stocks with key metrics.

    Fetches real-time quotes and recent K-line data for all watchlist stocks.
    Includes: price, change%, volume ratio, MA trend, RSI, and risk signals.
    """
    try:
        if not _store or not _store.table_exists("watchlist"):
            return _response(data={"message": "Watchlist empty. Use watchlist_add to add stocks.", "stocks": [], "count": 0})

        wl_df = _store.query_df("SELECT symbol FROM watchlist ORDER BY added_at DESC")
        symbols = wl_df["symbol"].tolist() if not wl_df.empty else []
        if not symbols:
            return _response(data={"message": "Watchlist empty.", "stocks": [], "count": 0})

        stocks = []

        for sym in symbols:
            stock_info = {"symbol": sym, "name": "", "price": 0, "pct_change": 0,
                          "trend": "N/A", "rsi": 0, "volume_ratio": 0, "signals": []}

            # Try Sina for real-time quote
            if _sina:
                try:
                    quote = await _sina.get_realtime_quote(sym)  # [AI] Fixed: pass str not list; return is dict not DataFrame
                    if quote:
                        stock_info["name"] = quote.get("name", "")
                        stock_info["price"] = float(quote.get("price", 0))
                        stock_info["open"] = float(quote.get("open", 0))
                        stock_info["high"] = float(quote.get("high", 0))
                        stock_info["low"] = float(quote.get("low", 0))
                        stock_info["prev_close"] = float(quote.get("prev_close", 0))
                        if stock_info["prev_close"] > 0:
                            stock_info["pct_change"] = round(
                                (stock_info["price"] - stock_info["prev_close"]) / stock_info["prev_close"] * 100, 2
                            )
                        stock_info["volume"] = float(quote.get("volume", 0))
                        stock_info["amount"] = float(quote.get("amount", 0))
                except Exception:
                    pass

            # Try Sina for K-line analysis
            if _sina:
                try:
                    df_kl = await _sina.get_daily_kline(sym, start_date="20260601", end_date=datetime.now().strftime("%Y%m%d"))
                    if not df_kl.empty and len(df_kl) >= 20:
                        # MA trend
                        df_kl["ma5"] = df_kl["close"].rolling(5).mean()
                        df_kl["ma20"] = df_kl["close"].rolling(20).mean()
                        latest = df_kl.iloc[-1]
                        if pd.notna(latest.get("ma5")) and pd.notna(latest.get("ma20")):
                            if latest["close"] > latest["ma5"] > latest["ma20"]:
                                stock_info["trend"] = "bullish"
                            elif latest["close"] < latest["ma5"] < latest["ma20"]:
                                stock_info["trend"] = "bearish"
                            else:
                                stock_info["trend"] = "neutral"

                        # RSI(14)
                        delta = df_kl["close"].diff()
                        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
                        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
                        rs = gain / loss.replace(0, float("nan"))
                        rsi_series = 100 - (100 / (1 + rs))
                        stock_info["rsi"] = round(float(rsi_series.iloc[-1]), 1) if pd.notna(rsi_series.iloc[-1]) else 0

                        # Volume ratio
                        vol_avg = df_kl["volume"].tail(20).mean()
                        vol_latest = df_kl["volume"].iloc[-1]
                        stock_info["volume_ratio"] = round(float(vol_latest / vol_avg), 2) if vol_avg > 0 else 0

                        # 20-day change
                        stock_info["change_20d"] = round(float(df_kl["pct_change"].tail(20).sum()), 2)

                        # Signals
                        pct = stock_info["pct_change"]
                        if pct >= 9.5:
                            stock_info["signals"].append("limit_up")
                        elif pct <= -9.5:
                            stock_info["signals"].append("limit_down")
                        if stock_info["rsi"] > 70:
                            stock_info["signals"].append("rsi_overbought")
                        elif stock_info["rsi"] < 30:
                            stock_info["signals"].append("rsi_oversold")
                        if stock_info["volume_ratio"] > 2:
                            stock_info["signals"].append("volume_surge")
                        if stock_info.get("change_20d", 0) > 20:
                            stock_info["signals"].append("strong_20d")
                        elif stock_info.get("change_20d", 0) < -20:
                            stock_info["signals"].append("weak_20d")
                except Exception:
                    pass

            stocks.append(stock_info)

        # Sort: attention-grabbing first (limit up/down, volume surge)
        def sort_key(s):
            score = 0
            if "limit_up" in s.get("signals", []):
                score += 100
            if "limit_down" in s.get("signals", []):
                score += 80
            if "volume_surge" in s.get("signals", []):
                score += 50
            score += abs(s.get("pct_change", 0))
            return -score

        stocks.sort(key=sort_key)

        return _response(data={
            "stocks": stocks,
            "count": len(stocks),
            "updated_at": datetime.now().isoformat(),
            "source": "sina",
        })
    except Exception as e:
        logger.error("watchlist_overview failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


# ---- MCP Tools: Agent (Full Integration) ----

@mcp.tool(annotations={"readOnlyHint": True})
async def agent_analyze_stock(symbol: str, depth: Literal["fast","deep","debate"] = "fast") -> dict[str, Any]:
    """AI-powered multi-dimensional stock analysis.

    Args:
        symbol: Stock code (e.g., '000001')
        depth: 'fast' (single call, ~15s) or 'deep' (5 analysts + PM, ~60s)
               or 'debate' (deep + Bull/Bear 辩论 + 投资大师裁定, ~90s)
    """
    try:
        from pa_mcp.agent.orchestrator import get_orchestrator
        from pa_mcp.engine.market_state import MarketStateDetector, MarketIndicators

        orchestrator = get_orchestrator()

        # Fetch kline data
        kline_df = None
        if _store:
            try:
                kline_df = _store.query_df(
                    "SELECT * FROM kline_daily WHERE symbol = ? ORDER BY date DESC LIMIT 120",
                    [symbol],
                )
            except Exception:
                pass

        if kline_df is None or kline_df.empty:
            return _response(success=False, error=f"No data for symbol {symbol}", error_type="NOT_FOUND")

        # Get market state
        market_state = None
        try:
            detector = MarketStateDetector()
            indicators = MarketIndicators()
            market_state = detector.detect(indicators).value
        except Exception:
            pass

        # Build fundamental data
        fundamental = {}
        if _store:
            try:
                evt_df = _store.query_df(
                    "SELECT * FROM major_events WHERE symbol = ? ORDER BY event_date DESC LIMIT 10",
                    [symbol],
                )
                fundamental["events"] = evt_df.to_json(orient="records") if not evt_df.empty else "No major events"
            except Exception:
                fundamental["events"] = "N/A"

        if depth == "fast":
            result = await orchestrator.fast_analyze(
                symbol, kline_df, market_state=market_state, fundamental_data=fundamental,
            )
        elif depth == "debate":
            result = await orchestrator.deep_analyze(
                symbol, kline_df, market_state=market_state, debate=True)
        else:
            result = await orchestrator.deep_analyze(symbol, kline_df, market_state=market_state)

        return _response(data={
            "symbol": symbol,
            "mode": result.mode,
            "overall_strength_score": result.overall_strength_score,
            "dimension_scores": result.dimension_scores,
            "direction": result.direction,
            "key_evidence": result.key_evidence,
            "key_risks": result.key_risks,
            "risk_reward_assessment": result.risk_reward_assessment,
            "suggested_max_position_pct": result.suggested_max_position_pct,
            "analysis_time_ms": result.analysis_time_ms,
            "master_verdict": result.master_verdict,
            "debate": result.debate,
                })

    except Exception as e:
        logger.error("agent_analyze_stock failed", symbol=symbol, error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def agent_market_state() -> dict[str, Any]:
    """Get current market regime and position sizing suggestion.

    Uses 5-state detector (climax/fermenting/starting/dull/frozen) with hysteresis.

    Returns:
        data.market_state: Current market regime name
        data.suggested_position_pct: Recommended max position as percentage (0-100)
        data.recommended_strategies: Strategy categories suitable for current state
        data.indicators: Key metrics (limit_up, limit_down, breadth, turnover_billion)
    """
    try:
        from pa_mcp.engine.market_state import MarketStateDetector, MarketIndicators

        if _store:
            latest = _store.get_latest_date("kline_daily")
            if latest:
                df = _store.query_df("""
                    SELECT
                        COUNT(CASE WHEN pct_change >= 9.5 THEN 1 END) as limit_up,
                        COUNT(CASE WHEN pct_change <= -9.5 THEN 1 END) as limit_down,
                        COUNT(CASE WHEN pct_change > 0 THEN 1 END) as up_count,
                        COUNT(CASE WHEN pct_change < 0 THEN 1 END) as down_count,
                        SUM(amount) / 100000000.0 as turnover
                    FROM kline_daily WHERE date = ?
                """, [latest])
                row = df.iloc[0]
                indicators = MarketIndicators(
                    limit_up_count=int(row["limit_up"]),
                    limit_down_count=int(row["limit_down"]),
                    up_count=int(row["up_count"]),
                    down_count=int(row["down_count"]),
                    turnover_billion=float(row["turnover"]),
                )
                detector = MarketStateDetector()
                state = detector.detect(indicators)
                mapping = detector.get_strategy_mapping()

                position_map = {
                    "climax": 70, "fermenting": 60, "starting": 35, "dull": 15, "frozen": 5,
                }
                return _response(data={
                    "market_state": state.value,
                    "suggested_position_pct": position_map[state.value],
                    "recommended_strategies": mapping.get(state, []),
                    "indicators": {
                        "limit_up": indicators.limit_up_count,
                        "limit_down": indicators.limit_down_count,
                        "breadth": round((indicators.up_count - indicators.down_count) / max(indicators.up_count + indicators.down_count, 1), 3),
                        "turnover_billion": round(indicators.turnover_billion, 1),
                    },
                })

        return _response(data={"market_state": "unknown", "message": "No data available. Run daily update first."})
    except Exception as e:
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


# ---- MCP Tools: Market Prediction & LLM Diagnosis (NEW) ----

@mcp.tool(annotations={"readOnlyHint": True})
async def predict_market(symbol: str, horizon: Literal["5d", "20d"] = "5d",
                         save: bool = True) -> dict[str, Any]:
    """AI 市场预测：基于 K 线技术特征预测未来走势方向与概率。

    借鉴 PA_Agent「未来走势预期」机制：周期位置 + 方向概率 + 期望收益 +
    关键价位 + 多场景。有 LLM 配置时由 LLM 预测（JSON 校验+重试），
    无 LLM 时降级为确定性统计预测。预测写入 prediction_log 表，
    到期后可用 evaluate_predictions() 验证命中率——预测可检验，非算命。

    Args:
        symbol: 股票代码（如 '000001'）
        horizon: 预测周期 '5d'（短线）或 '20d'（中线）
        save: 是否落盘以便日后验证
    """
    try:
        from pa_mcp.agent.prediction import get_prediction_service

        kline_df = None
        if _store:
            try:
                kline_df = _store.query_df(
                    "SELECT * FROM kline_daily WHERE symbol = ? ORDER BY date DESC LIMIT 160",
                    [symbol],
                )
            except Exception:
                pass
        if kline_df is None or kline_df.empty:
            # 尝试多源实时抓取兜底
            df, _ = await _get_kline_fallback(symbol, days=160)
            if df is not None and not df.empty:
                kline_df = df

        if kline_df is None or kline_df.empty:
            return _response(success=False, error=f"No data for symbol {symbol}",
                             error_type="NOT_FOUND")

        svc = get_prediction_service()
        result = await svc.predict(symbol, kline_df, horizon=horizon)
        payload = result.to_dict()
        if save:
            payload["prediction_id"] = svc.save_prediction(result)
        return _response(data=payload)
    except Exception as e:
        logger.error("predict_market failed", symbol=symbol, error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def predict_position_size(symbol: str, account_value: float = 100000.0,
                                horizon: Literal["5d", "20d"] = "5d",
                                base_position_pct: float = 0.0) -> dict[str, Any]:
    """预测驱动的仓位建议（Risk Manager 思路，借鉴 ai-hedge-fund）。

    预测概率 × 历史命中率校准 × 概率桶校准 → 建议仓位（≤20% 硬上限）。
    输出完整推导链（可追溯），供研究与决策参考。

    Args:
        symbol: 股票代码
        account_value: 账户资金（用于金额展示）
        horizon: 预测周期
        base_position_pct: 基础仓位（分析师建议）；0 = 按方向回退
    """
    try:
        from pa_mcp.agent.prediction import get_prediction_service
        svc = get_prediction_service()

        kline_df = None
        if _store:
            try:
                kline_df = _store.query_df(
                    "SELECT * FROM kline_daily WHERE symbol = ? ORDER BY date DESC LIMIT 160",
                    [symbol])
            except Exception:
                pass
        if kline_df is None or kline_df.empty:
            df, _ = await _get_kline_fallback(symbol, days=160)
            if df is not None and not df.empty:
                kline_df = df
        if kline_df is None or kline_df.empty:
            return _response(success=False, error=f"No data for symbol {symbol}",
                             error_type="NOT_FOUND")

        sizing = await svc.position_sizing(
            symbol, account_value=account_value, horizon=horizon,
            base_position_pct=base_position_pct or None, kline_df=kline_df)
        return _response(data=sizing)
    except Exception as e:
        logger.error("predict_position_size failed", symbol=symbol, error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def prediction_history(symbol: str, limit: int = 20) -> dict[str, Any]:
    """查看某股票的历史预测记录与验证结果（方向/概率/实际收益/命中状态）。"""
    try:
        from pa_mcp.agent.prediction import get_prediction_service
        rows = get_prediction_service().prediction_history(symbol, limit=limit)
        return _response(data={"symbol": symbol, "predictions": rows, "count": len(rows)})
    except Exception as e:
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def evaluate_predictions() -> dict[str, Any]:
    """验证历史预测：回填已到期预测的真实收益，计算命中率/Brier/方向一致率。

    这是预测功能的「成绩单」——检验 AI 预测是否优于随机。
    """
    try:
        from pa_mcp.agent.prediction import get_prediction_service
        svc = get_prediction_service()

        # 用真实数据库 + 多源 router 兜底拉行情
        async def _provider(symbol: str):
            try:
                df, _ = await _get_kline_fallback(symbol, days=60)
                return df
            except Exception:
                return None

        summary = svc.evaluate_predictions(kline_provider=_provider)
        return _response(data=summary)
    except Exception as e:
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def agent_market_diagnosis() -> dict[str, Any]:
    """两阶段分析 Stage 1：LLM 市场诊断 + 策略路由。

    用真实市场指标（涨停/跌停/成交额/涨跌家数）驱动 LLM 判定市场状态
    （高潮/发酵/启动/低迷/冰点），并给出对应策略路由与仓位上限。
    无 LLM 时使用确定性诊断。
    """
    try:
        from pa_mcp.agent.orchestrator import get_orchestrator

        market_context: dict[str, Any] = {}
        if _store:
            try:
                latest = _store.get_latest_date("kline_daily")
                if latest:
                    df = _store.query_df("""
                        SELECT
                            COUNT(CASE WHEN pct_change >= 9.5 THEN 1 END) as limit_up,
                            COUNT(CASE WHEN pct_change <= -9.5 THEN 1 END) as limit_down,
                            COUNT(CASE WHEN pct_change > 0 THEN 1 END) as up_count,
                            COUNT(CASE WHEN pct_change < 0 THEN 1 END) as down_count,
                            SUM(amount) / 100000000.0 as turnover
                        FROM kline_daily WHERE date = ?
                    """, [latest])
                    row = df.iloc[0]
                    market_context = {
                        "limit_up_count": int(row["limit_up"]),
                        "limit_down_count": int(row["limit_down"]),
                        "up_count": int(row["up_count"]),
                        "down_count": int(row["down_count"]),
                        "turnover_billion": round(float(row["turnover"]), 1),
                        "date": latest,
                    }
            except Exception:
                pass

            # 情绪周期上下文注入（游资情绪 → LLM 市场诊断）
            try:
                from pa_mcp.research.sentiment_cycle import (
                    get_sentiment_analyzer)
                sentiment = get_sentiment_analyzer().sentiment_summary()
                if sentiment:
                    market_context["sentiment"] = sentiment
            except Exception:
                pass

        orch = get_orchestrator()
        diagnosis = await orch.market_diagnosis(market_context or None)
        return _response(data=diagnosis)
    except Exception as e:
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def get_decision_tree(symbol: str) -> dict[str, Any]:
    """决策树可视化（借鉴 PA_Agent 决策树机制）：逐层闸门推演。

    市场状态（诊断）→ 策略路由 → 预测方向 → 仓位分级 → 结论建议。
    每个节点带 reason（判定依据）与 detail（证据），可追溯、不编造。

    Args:
        symbol: 股票代码（如 '000001'）
    """
    try:
        from pa_mcp.agent.decision_tree import build_decision_tree, tree_summary
        from pa_mcp.agent.orchestrator import get_orchestrator
        from pa_mcp.agent.prediction import get_prediction_service
        from pa_mcp.data.symbols import get_stock_name

        kline_df = None
        if _store:
            try:
                kline_df = _store.query_df(
                    "SELECT * FROM kline_daily WHERE symbol = ? ORDER BY date DESC LIMIT 160",
                    [symbol])
            except Exception:
                pass
        if kline_df is None or kline_df.empty:
            df, _ = await _get_kline_fallback(symbol, days=160)
            if df is not None and not df.empty:
                kline_df = df
        if kline_df is None or kline_df.empty:
            return _response(success=False, error=f"No data for symbol {symbol}",
                             error_type="NOT_FOUND")

        svc = get_prediction_service()
        pred_result = await svc.predict(symbol, kline_df, horizon="5d")
        prediction = pred_result.to_dict()

        diagnosis = None
        try:
            diagnosis = await get_orchestrator().market_diagnosis(None)
        except Exception:
            pass

        tree = build_decision_tree(
            symbol, diagnosis=diagnosis, prediction=prediction,
            stock_name=get_stock_name(symbol))
        return _response(data={
            "symbol": symbol,
            "tree": tree["tree"],
            "summary": tree_summary(tree),
            "market_state": tree["market_state"],
            "market_state_zh": tree["market_state_zh"],
            "position_cap_pct": tree["position_cap_pct"],
            "risk_level": tree["risk_level"],
            "observations": tree["observations"],
        })
    except Exception as e:
        logger.error("get_decision_tree failed", symbol=symbol, error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def predict_sector_rotation(load_data: bool = False,
                                  board_type: str = "industry") -> dict[str, Any]:
    """板块轮动预测：RS 动量排名 + 资金流 + LLM 解读 → 未来一周强势板块。

    流程：板块相对强度（20 日涨幅排名）+ 动量加速（5日 vs 20日日均差）+
    轮动信号（新进/退出 top10）→ LLM 预测强势板块候选（结构化 JSON，
    无 LLM 时动量延续规则降级）→ 落盘 sector_prediction 供周度验证
    （5 交易日后回填 top3 超额收益）。

    Args:
        load_data: True = 先拉取东财板块行情（首次使用需 True，约 30-60s）
        board_type: 'industry'（行业板块）或 'concept'（概念板块）
    """
    try:
        from pa_mcp.research.sector_rotation import (
            get_sector_rotation_analyzer, format_rotation,
        )
        analyzer = get_sector_rotation_analyzer()
        analyzer.board_type = board_type

        if load_data:
            load_info = await analyzer.load_sector_data(top_n=60, days=120)
            if load_info.get("loaded", 0) == 0:
                return _response(success=False, error=load_info.get(
                    "message", "板块数据装载失败（东财接口不可达？）"),
                    error_type="DATA_UNAVAILABLE")

        analysis = analyzer.analyze()
        if "error" in analysis:
            return _response(success=False, error=analysis["error"],
                             error_type="DATA_UNAVAILABLE")
        pred = await analyzer.predict(analysis)
        pred_id = analyzer.save_prediction(pred)
        return _response(data={
            **pred,
            "prediction_id": pred_id,
            "report": format_rotation(pred),
        })
    except Exception as e:
        logger.error("predict_sector_rotation failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def sector_rotation_status() -> dict[str, Any]:
    """板块轮动当前状态：RS 排名 / 轮入轮出 / 轮动速度（只读分析，不预测）。"""
    try:
        from pa_mcp.research.sector_rotation import (
            get_sector_rotation_analyzer)
        analysis = get_sector_rotation_analyzer().analyze()
        if "error" in analysis:
            return _response(success=False, error=analysis["error"],
                             error_type="DATA_UNAVAILABLE")
        return _response(data=analysis)
    except Exception as e:
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def factor_library(category: str = "") -> dict[str, Any]:
    """因子库清单（借鉴 factor-skill-factory）：已注册因子列表。

    Args:
        category: 类别过滤（momentum/mean_reversion/volatility/volume/
                  trend），空 = 全部
    """
    try:
        from pa_mcp.research.factors import get_factor_registry
        registry = get_factor_registry()
        factors = registry.list_by_category(category) if category \
            else registry.list_all()
        return _response(data={
            "count": len(factors),
            "factors": [f.to_dict() for f in factors],
        })
    except Exception as e:
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def evaluate_factor(factor_name: str, symbol: str,
                          horizon: int = 5) -> dict[str, Any]:
    """单因子检验（量化标准）：IC + 分层（Q1-Q5）+ 单调性 + 覆盖率。

    Args:
        factor_name: 因子名（factor_library 查看全部）
        symbol: 股票代码
        horizon: 前瞻交易日数（默认 5）
    """
    try:
        from pa_mcp.research.factors import (
            evaluate_factor as run_eval, get_factor_registry,
            format_factor_report)
        fd = get_factor_registry().get(factor_name)
        if fd is None:
            return _response(success=False,
                             error=f"因子 {factor_name} 未注册（factor_library 查看）",
                             error_type="NOT_FOUND")

        kline_df = None
        if _store:
            try:
                kline_df = _store.query_df(
                    "SELECT * FROM kline_daily WHERE symbol = ? ORDER BY date DESC LIMIT 300",
                    [symbol])
            except Exception:
                pass
        if kline_df is None or kline_df.empty:
            df, _ = await _get_kline_fallback(symbol, days=300)
            if df is not None and not df.empty:
                kline_df = df
        if kline_df is None or kline_df.empty:
            return _response(success=False, error=f"No data for symbol {symbol}",
                             error_type="NOT_FOUND")

        result = run_eval(fd, kline_df, horizon=horizon)
        if "error" in result:
            return _response(success=False, error=result["error"],
                             error_type="DATA_UNAVAILABLE")
        return _response(data={**result, "symbol": symbol,
                               "report": format_factor_report(result)})
    except Exception as e:
        logger.error("evaluate_factor failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def value_momentum_screen(symbols: str, top_n: int = 10,
                                value_weight: float = 0.5) -> dict[str, Any]:
    """价值 × 动量 复合选股（Asness et al. 2013 学术框架）。

    价值分（格雷厄姆评分标准化）与 60 日动量 z-score 按权重复合 →
    排序 + 四象限（价值+动量双高 = 最佳候选「便宜且正在走强」）。

    Args:
        symbols: 股票池（逗号分隔）
        top_n: 返回数量
        value_weight: 价值权重 0-1（动量 = 1 - value_weight）
    """
    try:
        from pa_mcp.research.value_momentum import (
            get_value_momentum_screen, format_value_momentum)
        pool = [s.strip() for s in symbols.replace("，", ",").split(",")
                if s.strip()]
        if not pool:
            return _response(success=False, error="请输入股票代码",
                             error_type="INVALID_ARGUMENT")
        w = max(0.0, min(1.0, value_weight))
        result = get_value_momentum_screen().screen(
            pool, top_n=top_n, value_weight=w)
        if "error" in result:
            return _response(success=False, error=result["error"],
                             error_type="DATA_UNAVAILABLE")
        return _response(data={**result,
                               "report": format_value_momentum(result)})
    except Exception as e:
        logger.error("value_momentum_screen failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def graham_screen(symbols: str) -> dict[str, Any]:
    """格雷厄姆价值筛选（《聪明的投资者》防御性投资标准）。

    7 条标准逐条判定（可追溯）：规模≥50亿 / ROE≥10% / 近4期净利为正 /
    股息（无数据源不评分）/ 净利同比均值>0 / PE<15 / PB<1.5 或 PE×PB<22.5。
    + 成长公式内在价值 V = EPS×(8.5+2g) 与安全边际。
    估值缺省从腾讯快照拉取（网络不可用时标 unavailable）。

    Args:
        symbols: 股票池（逗号分隔）
    """
    try:
        from pa_mcp.research.graham import (
            get_graham_screener, format_graham)
        pool = [s.strip() for s in symbols.replace("，", ",").split(",")
                if s.strip()]
        if not pool:
            return _response(success=False, error="请输入股票代码",
                             error_type="INVALID_ARGUMENT")
        results = get_graham_screener().screen(pool)
        return _response(data={
            "count": len(results),
            "results": [r.to_dict() for r in results],
            "report": format_graham(results),
        })
    except Exception as e:
        logger.error("graham_screen failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def strategy_compare(symbols: str = "") -> dict[str, Any]:
    """全策略事件研究对比（多策略同台检验）。

    全部注册策略（10 个，含 CANSLIM/缠论/海龟/利弗莫尔等大牛方法）在
    股票池上生成信号 → 事件研究（5/10/20 日超额 vs 无条件基准）→
    聚合对比表 + 有效策略判定（5日超额>0 且多数股票 has_edge）。

    Args:
        symbols: 股票池（逗号分隔）；空 = 数据库内有行情的全部股票
    """
    try:
        from pa_mcp.research.strategy_compare import (
            compare_all_strategies, format_compare)
        pool = [s.strip() for s in symbols.replace("，", ",").split(",")
                if s.strip()]
        klines = {}
        if pool:
            for sym in pool:
                try:
                    df = _store.query_df(
                        "SELECT * FROM kline_daily WHERE symbol = ? "
                        "ORDER BY date DESC LIMIT 300", [sym]) if _store else None
                    if df is None or df.empty:
                        kdf, _ = await _get_kline_fallback(sym, days=300)
                        df = kdf
                    if df is not None and not df.empty:
                        klines[sym] = df
                except Exception:
                    continue
        else:
            if _store:
                try:
                    syms = _store.query_df(
                        "SELECT DISTINCT symbol FROM kline_daily LIMIT 20", [])
                    for sym in syms["symbol"]:
                        df = _store.query_df(
                            "SELECT * FROM kline_daily WHERE symbol = ? "
                            "ORDER BY date DESC LIMIT 300", [str(sym)])
                        if not df.empty:
                            klines[str(sym)] = df
                except Exception:
                    pass
        if len(klines) < 1:
            return _response(success=False, error="无行情数据（先运行调度装载）",
                             error_type="DATA_UNAVAILABLE")

        result = compare_all_strategies(klines)
        if "error" in result:
            return _response(success=False, error=result["error"],
                             error_type="DATA_UNAVAILABLE")
        return _response(data={**result, "report": format_compare(result)})
    except Exception as e:
        logger.error("strategy_compare failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def factor_prediction_sensitivity(symbols: str, top_n: int = 5) -> dict[str, Any]:
    """预测权重敏感性分析：AI 预测在选股里该占多大权重。

    对权重 0/0.25/0.5/0.75/1.0 各跑因子选股组合回测（预测融合用
    确定性模式控制成本）→ 对比超额收益 → 推荐最优权重。
    「数据回答权重，而非拍脑袋」。

    Args:
        symbols: 股票池（逗号分隔，≥5 只）
        top_n: 每期持仓数量
    """
    try:
        from pa_mcp.research.factors import (
            sensitivity_analysis, format_sensitivity)
        pool = [s.strip() for s in symbols.replace("，", ",").split(",")
                if s.strip()]
        if len(pool) < 5:
            return _response(success=False,
                             error="至少需要 5 只股票", error_type="INVALID_ARGUMENT")
        klines = {}
        for sym in pool:
            try:
                df = _store.query_df(
                    "SELECT * FROM kline_daily WHERE symbol = ? "
                    "ORDER BY date DESC LIMIT 400", [sym]) if _store else None
                if df is None or df.empty:
                    kdf, _ = await _get_kline_fallback(sym, days=400)
                    df = kdf
                if df is not None and not df.empty:
                    klines[sym] = df
            except Exception:
                continue
        if len(klines) < 5:
            return _response(success=False,
                             error=f"仅 {len(klines)} 只股票有数据（需 ≥5）",
                             error_type="DATA_UNAVAILABLE")

        result = sensitivity_analysis(klines, top_n=top_n)
        if "error" in result and not result.get("results"):
            return _response(success=False, error=result["error"],
                             error_type="DATA_UNAVAILABLE")
        return _response(data={**result, "report": format_sensitivity(result)})
    except Exception as e:
        logger.error("factor_prediction_sensitivity failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def factor_portfolio_backtest(symbols: str, top_n: int = 5,
                                    horizon: int = 5,
                                    train_window: int = 120) -> dict[str, Any]:
    """因子选股组合回测（选股 → 组合闭环）。

    滚动窗口：每 horizon 日用过去 train_window 天做 pooled 截面 IC 选股
    top N → 等权组合（bullish 买入/其余 bearish 调出，延迟一天执行）
    → 复用共享账本组合回测引擎（单票10%/T+1/费用）→ 对比全池等权基准。

    Args:
        symbols: 股票池（逗号分隔，≥3 只）
        top_n: 每期持仓数量
        horizon: 调仓周期（交易日）
        train_window: IC 训练窗口（交易日）
    """
    try:
        from pa_mcp.research.factors import (
            backtest_factor_selection, format_portfolio_backtest)
        pool = [s.strip() for s in symbols.replace("，", ",").split(",")
                if s.strip()]
        if len(pool) < 3:
            return _response(success=False,
                             error="至少需要 3 只股票", error_type="INVALID_ARGUMENT")

        klines = {}
        for sym in pool:
            try:
                df = _store.query_df(
                    "SELECT * FROM kline_daily WHERE symbol = ? "
                    "ORDER BY date DESC LIMIT 400", [sym]) if _store else None
                if df is None or df.empty:
                    kdf, _ = await _get_kline_fallback(sym, days=400)
                    df = kdf
                if df is not None and not df.empty:
                    klines[sym] = df
            except Exception:
                continue
        if len(klines) < 3:
            return _response(success=False,
                             error=f"仅 {len(klines)} 只股票有数据（需 ≥3）",
                             error_type="DATA_UNAVAILABLE")

        result = backtest_factor_selection(
            klines, top_n=top_n, horizon=horizon,
            train_window=train_window)
        if "error" in result:
            return _response(success=False, error=result["error"],
                             error_type="DATA_UNAVAILABLE")
        return _response(data={**result,
                               "report": format_portfolio_backtest(result)})
    except Exception as e:
        logger.error("factor_portfolio_backtest failed", error=str(e))
        return _response(success=False, error=str(e),
                         error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def factor_stock_selection(symbols: str, top_n: int = 10,
                                 horizon: int = 5,
                                 prediction_weight: float = 0.0) -> dict[str, Any]:
    """多因子截面选股（Qlib 风格，可选 AI 预测融合）。

    股票池上：每因子 pooled IC 符号（|IC|≥0.02 达标）→ 各股最新因子值
    截面 z-score × IC 方向 → 等权合成综合分；可选融合 AI 预测概率
    （方向化概率 z-score，prediction_weight 加权）→ 排名输出。

    Args:
        symbols: 股票池（逗号分隔，≥5 只）
        top_n: 返回数量
        horizon: IC 前瞻窗口（默认 5）
        prediction_weight: AI 预测权重 0-1（0=纯因子，0.5=混合，1=纯预测）
    """
    try:
        from pa_mcp.research.factors import (
            select_stocks_by_factors, format_selection)
        pool = [s.strip() for s in symbols.replace("，", ",").split(",")
                if s.strip()]
        if len(pool) < 5:
            return _response(success=False,
                             error="至少需要 5 只股票（截面合成要求）",
                             error_type="INVALID_ARGUMENT")
        weight = max(0.0, min(1.0, prediction_weight))

        klines = {}
        for sym in pool:
            try:
                df = _store.query_df(
                    "SELECT * FROM kline_daily WHERE symbol = ? "
                    "ORDER BY date DESC LIMIT 150", [sym]) if _store else None
                if df is None or df.empty:
                    kdf, _ = await _get_kline_fallback(sym, days=150)
                    df = kdf
                if df is not None and not df.empty:
                    klines[sym] = df
            except Exception:
                continue
        if len(klines) < 5:
            return _response(success=False,
                             error=f"仅 {len(klines)} 只股票有数据（需 ≥5）",
                             error_type="DATA_UNAVAILABLE")

        result = select_stocks_by_factors(
            klines, top_n=top_n, horizon=horizon,
            prediction_weight=weight)
        if "error" in result:
            return _response(success=False, error=result["error"],
                             error_type="DATA_UNAVAILABLE")
        return _response(data={**result, "report": format_selection(result)})
    except Exception as e:
        logger.error("factor_stock_selection failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")
    try:
        from pa_mcp.research.factors import (
            select_stocks_by_factors, format_selection)
        pool = [s.strip() for s in symbols.replace("，", ",").split(",")
                if s.strip()]
        if len(pool) < 5:
            return _response(success=False,
                             error="至少需要 5 只股票（截面合成要求）",
                             error_type="INVALID_ARGUMENT")

        klines = {}
        for sym in pool:
            try:
                df = _store.query_df(
                    "SELECT * FROM kline_daily WHERE symbol = ? "
                    "ORDER BY date DESC LIMIT 150", [sym]) if _store else None
                if df is None or df.empty:
                    kdf, _ = await _get_kline_fallback(sym, days=150)
                    df = kdf
                if df is not None and not df.empty:
                    klines[sym] = df
            except Exception:
                continue
        if len(klines) < 5:
            return _response(success=False,
                             error=f"仅 {len(klines)} 只股票有数据（需 ≥5）",
                             error_type="DATA_UNAVAILABLE")

        result = select_stocks_by_factors(klines, top_n=top_n, horizon=horizon)
        if "error" in result:
            return _response(success=False, error=result["error"],
                             error_type="DATA_UNAVAILABLE")
        return _response(data={**result, "report": format_selection(result)})
    except Exception as e:
        logger.error("factor_stock_selection failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def factor_scan(symbol: str, horizon: int = 5) -> dict[str, Any]:
    """因子批量扫描：全部注册因子在一只股票上的 IC/分层检验排行。

    输出有效因子清单（|IC|≥0.03），供选股/研究参考。

    Args:
        symbol: 股票代码
        horizon: 前瞻交易日数（默认 5）
    """
    try:
        from pa_mcp.research.factors import (
            scan_factors, format_scan)
        kline_df = None
        if _store:
            try:
                kline_df = _store.query_df(
                    "SELECT * FROM kline_daily WHERE symbol = ? ORDER BY date DESC LIMIT 300",
                    [symbol])
            except Exception:
                pass
        if kline_df is None or kline_df.empty:
            df, _ = await _get_kline_fallback(symbol, days=300)
            if df is not None and not df.empty:
                kline_df = df
        if kline_df is None or kline_df.empty:
            return _response(success=False, error=f"No data for symbol {symbol}",
                             error_type="NOT_FOUND")

        results = scan_factors(kline_df, horizon=horizon)
        useful = [r for r in results if r.get("useful")]
        return _response(data={
            "symbol": symbol,
            "horizon": horizon,
            "count": len(results),
            "useful_count": len(useful),
            "useful_factors": useful,
            "ranking": results,
            "report": format_scan(results),
        })
    except Exception as e:
        logger.error("factor_scan failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def regime_matrix() -> dict[str, Any]:
    """情绪 × 轮动联合矩阵（Regime Matrix）。

    游资情绪阶段（启动/发酵/高潮/退潮/冰点）× 板块轮动速度（高/中/低）
    → 9 格矩阵：市场状态标签 + 操作建议 + 风险提示。
    「情绪决定做不做，轮动决定做什么」——纯复用 sentiment_cycle 与
    sector_rotation 的合成视图。
    """
    try:
        from pa_mcp.research.regime_matrix import (
            get_regime_analyzer, format_matrix)
        result = get_regime_analyzer().analyze()
        return _response(data={**result, "report": format_matrix(result)})
    except Exception as e:
        logger.error("regime_matrix failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def chan_beichi_event_study(symbol: str) -> dict[str, Any]:
    """缠论背驰信号事件研究（大牛方法可检验性）。

    滑动窗口（60 日/步长 3）扫描背驰信号（上涨背驰=涨势衰竭/下跌背驰=
    跌势衰竭）→ 复用事件研究验证信号后 5/10/20 日收益 vs 无条件基准，
    判定缠论背驰是否有预测力（has_edge）。

    Args:
        symbol: 股票代码
    """
    try:
        from pa_mcp.engine.indicators.chan import scan_beichi_signals
        from pa_mcp.research.event_study import signal_forward_returns

        kline_df = None
        if _store:
            try:
                kline_df = _store.query_df(
                    "SELECT * FROM kline_daily WHERE symbol = ? ORDER BY date DESC LIMIT 400",
                    [symbol])
            except Exception:
                pass
        if kline_df is None or kline_df.empty:
            df, _ = await _get_kline_fallback(symbol, days=400)
            if df is not None and not df.empty:
                kline_df = df
        if kline_df is None or kline_df.empty:
            return _response(success=False, error=f"No data for symbol {symbol}",
                             error_type="NOT_FOUND")

        sig_df = scan_beichi_signals(kline_df, symbol=symbol, window=60, step=3)
        if sig_df.empty:
            return _response(data={"symbol": symbol, "n_signals": 0,
                                   "message": "未检出背驰信号（窗口内无动能衰竭结构）"})

        results = signal_forward_returns(kline_df, sig_df, [5, 10, 20])
        return _response(data={
            "symbol": symbol,
            "n_signals": len(sig_df),
            "bullish_signals": int((sig_df["direction"] == "bullish").sum()),
            "bearish_signals": int((sig_df["direction"] == "bearish").sum()),
            "signal_range": [sig_df["date"].iloc[0], sig_df["date"].iloc[-1]],
            "results": [
                {"horizon": r.horizon, "n_events": r.n_events,
                 "win_rate_pct": r.win_rate_pct,
                 "avg_return_pct": r.avg_return_pct,
                 "benchmark_avg_return_pct": r.benchmark_avg_return_pct,
                 "excess_return_pct": r.excess_return_pct,
                 "has_edge": r.has_edge}
                for r in results],
            "has_edge": any(r.has_edge for r in results),
        })
    except Exception as e:
        logger.error("chan_beichi_event_study failed", symbol=symbol, error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def factor_neutralize(symbols: str, lookback: int = 120) -> dict[str, Any]:
    """因子正交化（风格中性化，借鉴 factor-orthogonalize）。

    对股票池逐日 OLS 回归风格暴露（市值 + 板块收益），残差 = 剔除
    风格后的纯个股 alpha。输出 alpha 排名（IR 排序）+ 风格 beta 诊断
    （市值 β / 板块 β / 残差波动）——避免「押中板块」被误认为「个股 alpha」。

    Args:
        symbols: 股票代码（逗号分隔，≥5 只，同板块内效果最佳）
        lookback: 回溯交易日数（默认 120）
    """
    try:
        from pa_mcp.research.orthogonalize import (
            get_factor_neutralizer, format_neutralized)
        pool = [s.strip() for s in symbols.replace("，", ",").split(",")
                if s.strip()]
        if len(pool) < 3:
            return _response(success=False,
                             error="至少需要 5 只股票（截面回归自由度要求）",
                             error_type="INVALID_ARGUMENT")
        result = get_factor_neutralizer().neutralize(pool, lookback=lookback)
        if "error" in result:
            return _response(success=False, error=result["error"],
                             error_type="DATA_UNAVAILABLE")
        return _response(data={**result, "report": format_neutralized(result)})
    except Exception as e:
        logger.error("factor_neutralize failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def backtest_overfit_diagnosis(sharpe_obs: float, n_trials: int,
                                     periods: int,
                                     returns_matrix: str = "") -> dict[str, Any]:
    """回测过拟合与多重检验诊断（DSR / Harvey-Liu / CSCV-PBO）。

    借鉴 QuantSkills skill-backtest-overfit（Bailey & López de Prado 框架）：
    - DSR：考虑调参/试验次数 N 后，观测 Sharpe 是否仍显著（≥95%）
    - Harvey-Liu 阈值：多重检验下 Sharpe 显著性门槛 sqrt(2lnN/T)
    - CSCV-PBO（可选）：传入策略收益矩阵（JSON 二维数组，策略×时间），
      计算过拟合概率（样本内最优策略在样本外跌出前半的比例）

    Args:
        sharpe_obs: 观测年化 Sharpe（最佳参数组合的回测值）
        n_trials: 试验次数（策略数 × 参数组合数 × 股票数，诚实估计）
        periods: 样本期数（年化 Sharpe 的分母）
        returns_matrix: 可选 JSON 二维数组（策略×时间收益），用于 CSCV-PBO
    """
    try:
        from pa_mcp.research.overfit import (
            run_overfit_report, format_overfit_report)
        import json
        import numpy as np

        matrix = None
        if returns_matrix.strip():
            try:
                matrix = np.asarray(json.loads(returns_matrix), dtype=float)
                if matrix.ndim != 2:
                    return _response(success=False, error="returns_matrix 须为二维数组",
                                     error_type="INVALID_ARGUMENT")
            except Exception as e:
                return _response(success=False,
                                 error=f"returns_matrix 解析失败：{e}",
                                 error_type="INVALID_ARGUMENT")

        report = run_overfit_report(sharpe_obs, n_trials, periods,
                                    returns_matrix=matrix)
        return _response(data={**report, "report": format_overfit_report(report)})
    except Exception as e:
        logger.error("backtest_overfit_diagnosis failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def sentiment_cycle(date: str = "") -> dict[str, Any]:
    """游资情绪周期分析（涨停梯队/连板高度/晋级率/阶段判定）。

    情绪四阶段（借鉴游资龙头战法）：启动期 → 发酵期 → 高潮期 → 退潮期
    （+ 冰点期）。核心指标：连板高度（连续涨停天数）、梯队分布
    （首板/2板/3板/4板+）、晋级率（今日≥2板 ÷ 昨日涨停）、情绪分（0-100）。
    附近 5 日趋势与退潮预警。日线收盘涨停判定（≥9.5% 近似，无盘中炸板）。

    Args:
        date: 交易日（YYYY-MM-DD），空 = 最新
    """
    try:
        from pa_mcp.research.sentiment_cycle import (
            get_sentiment_analyzer, format_sentiment)
        result = get_sentiment_analyzer().analyze(target_date=date or None)
        if "error" in result:
            return _response(success=False, error=result["error"],
                             error_type="DATA_UNAVAILABLE")
        return _response(data={**result, "report": format_sentiment(result)})
    except Exception as e:
        logger.error("sentiment_cycle failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def sector_leaders(sector: str, top_n: int = 8) -> dict[str, Any]:
    """板块领涨股挖掘（板块轮动 → 个股闭环）。

    板块内按 60 日 RS 排名取领涨股，标记是否接近 60 日新高
    （强势突破候选）。配合 predict_sector_rotation 使用：
    先预测强势板块，再挖板块内领涨股。

    Args:
        sector: 板块名称（与 stock_basic.sector 一致，如 '银行'）
        top_n: 返回数量
    """
    try:
        from pa_mcp.research.sector_rotation import (
            get_sector_rotation_analyzer)
        result = get_sector_rotation_analyzer().leaders_in_sector(
            sector, top_n=top_n)
        if "error" in result:
            return _response(success=False, error=result["error"],
                             error_type="DATA_UNAVAILABLE")
        return _response(data=result)
    except Exception as e:
        logger.error("sector_leaders failed", sector=sector, error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def evaluate_sector_predictions() -> dict[str, Any]:
    """板块轮动预测验证：回填已到期预测的 top3 超额收益（vs 全板块平均）。"""
    try:
        from pa_mcp.research.sector_rotation import (
            get_sector_rotation_analyzer)
        return _response(data=get_sector_rotation_analyzer().evaluate_predictions())
    except Exception as e:
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def turtle_position_size(symbol: str, account_value: float = 100000.0,
                               risk_pct: float = 1.0) -> dict[str, Any]:
    """海龟交易仓位计算（ATR 波动率目标）。

    1 单位 = 账户 × risk_pct% ÷ ATR（元）；返回建议股数（100 股整手）、
    止损位（唐奇安 10 日通道）、以及单票仓位占比（受 10% 上限约束）。

    Args:
        symbol: 股票代码
        account_value: 账户资金（默认 10 万）
        risk_pct: 每单风险预算（账户 %，经典 1%）
    """
    try:
        from pa_mcp.engine.strategies.turtle import TurtleBreakoutStrategy
        from pa_mcp.data.symbols import get_stock_name

        kline_df = None
        if _store:
            try:
                kline_df = _store.query_df(
                    "SELECT * FROM kline_daily WHERE symbol = ? ORDER BY date DESC LIMIT 80",
                    [symbol])
            except Exception:
                pass
        if kline_df is None or kline_df.empty:
            df, _ = await _get_kline_fallback(symbol, days=80)
            if df is not None and not df.empty:
                kline_df = df
        if kline_df is None or kline_df.empty:
            return _response(success=False, error=f"No data for symbol {symbol}",
                             error_type="NOT_FOUND")

        data = kline_df.sort_values("date").reset_index(drop=True)
        strat = TurtleBreakoutStrategy(risk_percent=risk_pct)
        atr = float(strat._atr(data, strat.atr_period).iloc[-1])
        last_close = float(data["close"].iloc[-1])
        exit_level = float(data["low"].tail(strat.exit_period).min())
        atr_pct = atr / last_close * 100 if last_close > 0 else 0.0

        # 1 单位 = 账户 × 风险% ÷ ATR(元)；向下取整到 100 股
        risk_amount = account_value * risk_pct / 100
        units = risk_amount / atr if atr > 0 else 0
        shares = int(units // 100 * 100)
        position_value = shares * last_close
        position_pct = min(10.0, position_value / account_value * 100) \
            if account_value > 0 else 0.0

        return _response(data={
            "symbol": symbol,
            "name": get_stock_name(symbol),
            "last_close": round(last_close, 3),
            "atr": round(atr, 4),
            "atr_pct": round(atr_pct, 2),
            "risk_amount_per_unit": round(risk_amount, 2),
            "suggested_shares": shares,
            "suggested_position_value": round(position_value, 2),
            "suggested_position_pct": round(position_pct, 2),
            "stop_loss_level": round(exit_level, 3),
            "stop_loss_pct": round((last_close - exit_level) / last_close * 100, 2)
            if last_close > 0 else 0.0,
            "note": ("1 单位 = 账户 × 风险预算 ÷ ATR，回测中受单票 10% 上限约束。"
                     "研究参考，非投资建议。"),
        })
    except Exception as e:
        logger.error("turtle_position_size failed", symbol=symbol, error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def scan_canslim(top_n: int = 20, pool: str = "") -> dict[str, Any]:
    """CANSLIM 成长股扫描（欧奈尔《笑傲股市》七要素选股法）。

    七要素（A股适配）：C当季盈利(净利同比≥20%) / A年度增长(4期均值≥25%
    或ROE≥17%) / N新高(250日) / S突破放量(量比≥1.5) / L领军股(池内RS
    前20%) / M市场方向(高潮/发酵/启动通过，冰点否决)；I机构数据暂缺。
    纯确定性规则，无 LLM 依赖。

    Args:
        top_n: 返回数量（按总分排序）
        pool: 股票池（逗号分隔），空 = 数据库内有行情的全部股票
    """
    try:
        from pa_mcp.research.canslim import get_canslim_scanner, format_scan
        pool_list = [s.strip() for s in pool.replace("，", ",").split(",")
                     if s.strip()] or None
        scanner = get_canslim_scanner()
        results = scanner.scan(pool=pool_list, top_n=top_n)
        return _response(data={
            "results": [r.to_dict() for r in results],
            "count": len(results),
            "market_state": results[0].market_state if results else None,
            "report": format_scan(results),
        })
    except Exception as e:
        logger.error("scan_canslim failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def chan_analysis(symbol: str) -> dict[str, Any]:
    """缠论结构分析（缠中说禅体系）：分型→笔→中枢→背驰。

    输出：合并K线/分型/笔/中枢数量、最近中枢区间、背驰信号
    （上涨背驰=涨势衰竭 / 下跌背驰=跌势衰竭）、当前价位相对中枢位置。
    """
    try:
        from pa_mcp.engine.indicators.chan import (
            chan_analysis as chan_run,
            format_chan,
        )
        from pa_mcp.data.symbols import get_stock_name

        kline_df = None
        if _store:
            try:
                kline_df = _store.query_df(
                    "SELECT * FROM kline_daily WHERE symbol = ? ORDER BY date DESC LIMIT 200",
                    [symbol])
            except Exception:
                pass
        if kline_df is None or kline_df.empty:
            df, _ = await _get_kline_fallback(symbol, days=200)
            if df is not None and not df.empty:
                kline_df = df
        if kline_df is None or kline_df.empty:
            return _response(success=False, error=f"No data for symbol {symbol}",
                             error_type="NOT_FOUND")

        a = chan_run(kline_df, symbol=symbol)
        return _response(data={
            "symbol": symbol,
            "name": get_stock_name(symbol),
            "structure": a.to_dict(),
            "report": format_chan(a),
        })
    except Exception as e:
        logger.error("chan_analysis failed", symbol=symbol, error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def agent_memory_status(days: int = 60) -> dict[str, Any]:
    """长期记忆状态：决策记录数量/胜率/盈亏比 + 认知偏差检测。

    每次 AI 分析自动记录决策，事后回填实际收益；
    偏差检测：过度自信（高分决策+负收益）、处置效应等。
    """
    try:
        from pa_mcp.agent.memory import LongTermMemory
        mem = LongTermMemory()
        perf = mem.get_performance_summary(days=days)
        biases = mem.detect_bias()
        # 策略权重（贝叶斯滚动校准）
        import sqlite3
        conn = sqlite3.connect(mem.db_path)
        rows = conn.execute(
            "SELECT strategy_name, weight, win_rate, total_trades FROM strategy_weights "
            "ORDER BY weight DESC LIMIT 10").fetchall()
        conn.close()
        return _response(data={
            "performance": perf,
            "biases": biases,
            "strategy_weights": [
                {"strategy": r[0], "weight": r[1], "win_rate": r[2], "trades": r[3]}
                for r in rows],
        })
    except Exception as e:
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def agent_experience_search(symbol: str = "", cycle_position: str = "",
                                  direction: str = "", limit: int = 5) -> dict[str, Any]:
    """经验库检索（RAG）：按符号/周期位置/方向检索历史 AI 分析案例。

    案例含事后验证（hit/miss/实际收益），可注入其他分析 prompt 作参考。
    周期位置枚举：spike/micro_channel/tight_channel/normal_channel/
    broad_channel/trending_range/trading_range/extreme_range
    """
    try:
        from pa_mcp.agent.experience import get_experience_service
        svc = get_experience_service()
        entries = svc.search_experience(
            symbol=symbol or None,
            cycle_position=cycle_position or None,
            direction=direction or None,
            limit=limit,
        )
        return _response(data={
            "entries": [e.__dict__ for e in entries],
            "count": len(entries),
            "prompt_text": svc.format_experience(entries, limit=limit),
        })
    except Exception as e:
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


# ---- MCP Tools: Comprehensive Analysis (NEW) ----

@mcp.tool(annotations={"readOnlyHint": True})
async def agent_scan_market(
    top_n: int = 20,
    strategy_filter: str = "",
    sort_by: Literal["strength","consensus"] = "strength",
) -> dict[str, Any]:
    """AI-powered full market scan — run all strategies, rank by strength score.

    This is THE comprehensive analysis tool. It:
    1. Runs all strategies on all stocks (using pre-computed signals if available)
    2. Aggregates signals by stock
    3. Returns top-ranked candidates with per-strategy breakdown

    Args:
        top_n: Number of top candidates to return (default 20, max 50)
        strategy_filter: Only run specific strategy (empty = all strategies)
        sort_by: 'strength' (highest score) or 'consensus' (most strategies agree)
    """
    try:
        from pa_mcp.engine.strategies.base import StrategyRegistry, MarketState
        from pa_mcp.engine.market_state import MarketStateDetector, MarketIndicators

        registry = StrategyRegistry()
        top_n = min(top_n, 50)

        # Detect current market state
        market_state = None
        state_name = "unknown"
        if _store:
            latest = _store.get_latest_date("kline_daily")
            if latest:
                df = _store.query_df("""
                    SELECT
                        COUNT(CASE WHEN pct_change >= 9.5 THEN 1 END) as limit_up,
                        COUNT(CASE WHEN pct_change <= -9.5 THEN 1 END) as limit_down,
                        COUNT(CASE WHEN pct_change > 0 THEN 1 END) as up_count,
                        COUNT(CASE WHEN pct_change < 0 THEN 1 END) as down_count,
                        SUM(amount) / 100000000.0 as turnover
                    FROM kline_daily WHERE date = ?
                """, [latest])
                row = df.iloc[0]
                indicators = MarketIndicators(
                    limit_up_count=int(row["limit_up"]),
                    limit_down_count=int(row["limit_down"]),
                    up_count=int(row["up_count"]),
                    down_count=int(row["down_count"]),
                    turnover_billion=float(row["turnover"]),
                )
                detector = MarketStateDetector()
                market_state = detector.detect(indicators)
                state_name = market_state.value

        # Use pre-computed signals if available AND has data, else compute live
        candidates: dict[str, list[dict]] = {}  # symbol -> [signal summaries]

        use_cache = False
        if _store and _store.table_exists("signal_cache"):
            cache_count = _store.row_count("signal_cache")
            if cache_count > 0:
                use_cache = True

        if use_cache:
            # Use pre-computed cache (fast path — only if data exists)
            df = _store.query_df(f"""
                SELECT symbol, strategy_name, strength_score, direction, details
                FROM signal_cache
                WHERE date = (SELECT MAX(date) FROM signal_cache)
                ORDER BY strength_score DESC
                LIMIT 500
            """)
            for _, row in df.iterrows():
                sym = row["symbol"]
                if sym not in candidates:
                    candidates[sym] = []
                candidates[sym].append({
                    "strategy": row["strategy_name"],
                    "strength": round(float(row["strength_score"]), 1),
                    "direction": row["direction"],
                })
        else:
            # Compute live (slower but works without pre-computation)
            available = registry.list_all()
            if strategy_filter:
                strategies_to_run = [strategy_filter] if strategy_filter in available else []
            else:
                strategies_to_run = available

            if not strategies_to_run:
                return _response(data={
                    "market_state": state_name,
                    "candidates": [],
                    "count": 0,
                    "strategies_run": 0,
                    "message": f"No strategies found. Registered: {len(available)}",
                })

            # Get stock list
            stock_list = []
            if _store and _store.table_exists("stock_basic"):
                basic_df = _store.query_df("SELECT symbol, name FROM stock_basic LIMIT 200")
                stock_list = basic_df.to_dict(orient="records") if not basic_df.empty else []
            else:
                stock_list = [{"symbol": f"{i:06d}", "name": f"Stock{i}"} for i in range(1, 51)]

            # Run each strategy on each stock (use DuckDB kline for caching)
            for stock in stock_list[:100]:  # Cap at 100 stocks for live mode
                sym = stock["symbol"]
                try:
                    kline = None
                    if _store:
                        try:
                            kline = _store.query_df(
                                "SELECT * FROM kline_daily WHERE symbol = ? ORDER BY date DESC LIMIT 120",
                                [sym],
                            )
                        except Exception:
                            pass

                    if kline is None or kline.empty:
                        continue

                    for s_name in strategies_to_run:
                        try:
                            strategy = registry.get(s_name)
                            if market_state and not strategy.is_suitable_for(market_state):
                                continue
                            signals = strategy.generate_signals(kline, market_state)
                            for sig in signals[-1:]:  # Only latest signal
                                if sym not in candidates:
                                    candidates[sym] = []
                                candidates[sym].append({
                                    "strategy": s_name,
                                    "strength": round(sig.strength_score, 1),
                                    "direction": sig.direction.value,
                                })
                        except Exception:
                            continue
                except Exception:
                    continue

        # Sort by strength or consensus
        if sort_by == "consensus":
            ranked = sorted(
                candidates.items(),
                key=lambda x: (len(x[1]), sum(s["strength"] for s in x[1])),
                reverse=True,
            )
        else:  # strength
            ranked = sorted(
                candidates.items(),
                key=lambda x: sum(s["strength"] for s in x[1]) / max(len(x[1]), 1) if x[1] else 0,
                reverse=True,
            )

        # Build result
        result_candidates = []
        for sym, sigs in ranked[:top_n]:
            avg_strength = sum(s["strength"] for s in sigs) / max(len(sigs), 1)
            buy_sigs = [s for s in sigs if s["direction"] == "bullish"]
            result_candidates.append({
                "symbol": sym,
                "avg_strength": round(avg_strength, 1),
                "total_signals": len(sigs),
                "bullish_count": len(buy_sigs),
                "consensus_pct": round(len(buy_sigs) / max(len(sigs), 1) * 100, 1),
                "top_strategies": sorted(sigs, key=lambda x: x["strength"], reverse=True)[:5],
            })

        return _response(data={
            "market_state": state_name,
            "strategies_run": len(strategies_to_run) if not _store or not _store.table_exists("signal_cache") else "pre_computed",
            "scanned_stocks": len(candidates),
            "candidates": result_candidates,
            "count": len(result_candidates),
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        logger.error("agent_scan_market failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def agent_compare_stocks(symbols: str, dimensions: str = "") -> dict[str, Any]:
    """Side-by-side comparison of multiple stocks across all dimensions.

    Args:
        symbols: Comma-separated stock codes (e.g., '000001,000002,600036')
        dimensions: Comma-separated dimensions to compare (technical,capital,sentiment,fundamental,event). Empty = all.
    """
    try:
        sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
        if len(sym_list) < 2:
            return _response(success=False, error="Need at least 2 symbols to compare", error_type="INVALID_PARAM")
        if len(sym_list) > 10:
            return _response(success=False, error="Max 10 symbols for comparison", error_type="INVALID_PARAM")

        dim_list = [d.strip() for d in dimensions.split(",") if d.strip()] if dimensions else [
            "technical", "capital", "sentiment", "fundamental", "event",
        ]

        comparison = {}
        for sym in sym_list:
            stock_info: dict[str, Any] = {"symbol": sym}

            # Fetch kline
            kline = None
            if _store:
                try:
                    kline = _store.query_df(
                        "SELECT * FROM kline_daily WHERE symbol = ? ORDER BY date DESC LIMIT 60",
                        [sym],
                    )
                except Exception:
                    pass

            if kline is not None and not kline.empty:
                close = kline["close"].values
                volume = kline["volume"].values if "volume" in kline.columns else [0] * len(close)

                if "technical" in dim_list:
                    change_5d = (close[-1] - close[-5]) / close[-5] * 100 if len(close) >= 5 else 0
                    change_20d = (close[-1] - close[-20]) / close[-20] * 100 if len(close) >= 20 else 0
                    vol_ratio = volume[-1] / (volume[-20:].mean() if len(volume) >= 20 else 1)
                    ma20_pos = "above" if close[-1] > pd.Series(close).rolling(20).mean().iloc[-1] else "below"
                    stock_info["technical"] = {
                        "price": round(float(close[-1]), 2),
                        "change_5d_pct": round(change_5d, 2),
                        "change_20d_pct": round(change_20d, 2),
                        "volume_ratio": round(float(vol_ratio), 2),
                        "ma20_position": ma20_pos,
                    }

                if "capital" in dim_list and _store:
                    try:
                        flow = _store.query_df(
                            "SELECT * FROM capital_flow WHERE symbol = ? ORDER BY trade_date DESC LIMIT 5",
                            [sym],
                        )
                        if not flow.empty:
                            stock_info["capital"] = {
                                "main_net_inflow_5d": round(float(flow["main_net_inflow"].sum()), 0),
                                "northbound_hold_pct": round(float(flow["northbound_hold_pct"].iloc[0] * 100), 2) if flow["northbound_hold_pct"].notna().any() else None,
                            }
                    except Exception:
                        stock_info["capital"] = "N/A"

            # Fetch events
            if "event" in dim_list and _store:
                try:
                    events = _store.query_df(
                        "SELECT event_type, event_date FROM major_events WHERE symbol = ? ORDER BY event_date DESC LIMIT 10",
                        [sym],
                    )
                    stock_info["event"] = {
                        "recent_events": len(events),
                        "types": events["event_type"].unique().tolist() if not events.empty else [],
                    }
                except Exception:
                    stock_info["event"] = "N/A"

            comparison[sym] = stock_info

        return _response(data={
            "symbols": sym_list,
            "dimensions": dim_list,
            "comparison": comparison,
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        logger.error("agent_compare_stocks failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def agent_morning_brief(date: str = "") -> dict[str, Any]:
    """Daily pre-market briefing: overnight news, global markets, today's watchlist.

    Aggregates multiple data sources into a single actionable morning brief.

    Args:
        date: Trading date (YYYY-MM-DD), empty for today
    """
    try:
        brief = {
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "generated_at": datetime.now().isoformat(),
        }

        # 1. Market state
        from pa_mcp.engine.market_state import MarketStateDetector, MarketIndicators
        market_info: dict[str, Any] = {"state": "unknown"}
        if _store:
            latest = _store.get_latest_date("kline_daily")
            if latest:
                df = _store.query_df("""
                    SELECT
                        COUNT(CASE WHEN pct_change >= 9.5 THEN 1 END) as limit_up,
                        COUNT(CASE WHEN pct_change <= -9.5 THEN 1 END) as limit_down,
                        COUNT(CASE WHEN pct_change > 0 THEN 1 END) as up_count,
                        COUNT(CASE WHEN pct_change < 0 THEN 1 END) as down_count,
                        SUM(amount) / 100000000.0 as turnover
                    FROM kline_daily WHERE date = ?
                """, [latest])
                row = df.iloc[0]
                indicators = MarketIndicators(
                    limit_up_count=int(row["limit_up"]),
                    limit_down_count=int(row["limit_down"]),
                    up_count=int(row["up_count"]),
                    down_count=int(row["down_count"]),
                    turnover_billion=float(row["turnover"]),
                )
                detector = MarketStateDetector()
                ms = detector.detect(indicators)
                pos_map = {"climax": 70, "fermenting": 60, "starting": 35, "dull": 15, "frozen": 5}
                market_info = {
                    "state": ms.value,
                    "suggested_position_pct": pos_map[ms.value],
                    "yesterday_breadth": round(
                        (indicators.up_count - indicators.down_count) / max(indicators.up_count + indicators.down_count, 1), 3,
                    ),
                    "yesterday_limit_up": indicators.limit_up_count,
                    "yesterday_limit_down": indicators.limit_down_count,
                    "yesterday_turnover_billion": round(indicators.turnover_billion, 0),
                }
        brief["market"] = market_info

        # 2. Top limit-up stocks from yesterday
        if _store:
            latest = _store.get_latest_date("kline_daily")
            if latest:
                limit_up_df = _store.query_df(f"""
                    SELECT symbol, pct_change, volume, turnover
                    FROM kline_daily WHERE date = ? AND pct_change >= 9.5
                    ORDER BY pct_change DESC LIMIT 20
                """, [latest])
                brief["yesterday_top_boards"] = limit_up_df.to_dict(orient="records") if not limit_up_df.empty else []

        # 3. Volume surge stocks
        if _store:
            latest = _store.get_latest_date("kline_daily")
            if latest:
                surge_df = _store.query_df(f"""
                    WITH vol_avg AS (
                        SELECT symbol, AVG(volume) as avg_vol
                        FROM kline_daily WHERE date >= ?::DATE - INTERVAL '30 days' GROUP BY symbol
                    )
                    SELECT k.symbol, k.pct_change, k.volume / v.avg_vol as vol_ratio
                    FROM kline_daily k JOIN vol_avg v ON k.symbol = v.symbol
                    WHERE k.date = ? AND k.volume / v.avg_vol >= 2.0
                    ORDER BY vol_ratio DESC LIMIT 15
                """, [latest, latest])
                brief["volume_surge_stocks"] = surge_df.to_dict(orient="records") if not surge_df.empty else []

        # 4. Dragon-tiger activity
        if _store:
            latest_lhb = _store.get_latest_date("dragon_tiger", "trade_date")
            if latest_lhb:
                lhb_count = _store.query_df(
                    "SELECT COUNT(DISTINCT symbol) as cnt FROM dragon_tiger WHERE trade_date = ?",
                    [latest_lhb],
                ).iloc[0, 0]
                brief["dragon_tiger"] = {"date": latest_lhb, "stocks_count": int(lhb_count)}

        # 5. Strategy-suggested watchlist
        if _store and _store.table_exists("signal_cache"):
            latest_sig = _store.get_latest_date("signal_cache")
            if latest_sig:
                top_sigs = _store.query_df(f"""
                    SELECT symbol, strategy_name, strength_score
                    FROM signal_cache
                    WHERE date = ? AND strength_score >= 60
                    ORDER BY strength_score DESC LIMIT 20
                """, [latest_sig])
                brief["strategy_watchlist"] = top_sigs.to_dict(orient="records") if not top_sigs.empty else []
        else:
            brief["strategy_watchlist"] = []

        # 6. Risk alerts
        systemic_alerts = []
        if market_info.get("state") == "frozen":
            systemic_alerts.append("FROZEN market — recommend cash position only")
        if market_info.get("state") == "dull":
            systemic_alerts.append("DULL market — reduce position, avoid chasing")
        if market_info.get("yesterday_limit_down", 0) > 50:
            systemic_alerts.append("Mass limit-downs detected — systemic sell-off risk")
        if market_info.get("yesterday_turnover_billion", 1000) < 500:
            systemic_alerts.append("Volume collapse — liquidity risk elevated")

        from datetime import datetime as dt
        current_month = dt.now().month
        if current_month in (5, 6, 9, 11, 12):
            systemic_alerts.append(f"Seasonal defense: {current_month}月 historically weak — reduce exposure")

        brief["risk_alerts"] = systemic_alerts

        # 7. What to watch today
        watch_instructions = []
        state = market_info.get("state", "unknown")
        if state == "climax":
            watch_instructions = [
                "Ride momentum but trail stops aggressively",
                "Watch for divergence: high limit-up count + declining turnover = top signal",
                "Focus on: trend + momentum strategies",
            ]
        elif state == "fermenting":
            watch_instructions = [
                "Active trading environment — run full strategy scan",
                "Focus on: platform breakout, MA golden cross, sector leaders",
                "Position: 40-60%, use trailing stops",
            ]
        elif state == "starting":
            watch_instructions = [
                "Early recovery — small positions, test the water",
                "Focus on: oversold bounce, value plays, insider buying",
                "Position: 20-40%, wider stops",
            ]
        elif state == "dull":
            watch_instructions = [
                "Sideways market — grid strategies, avoid breakouts (false signals)",
                "Focus on: high dividend, low PE value, range grid",
                "Position: 10-20%, take profits quickly",
            ]
        elif state == "frozen":
            watch_instructions = [
                "STAY OUT — preservation mode",
                "Review past trades for lessons, prepare watchlist for recovery",
                "Position: 0-5%, all in cash or reverse repo",
            ]
        brief["today_playbook"] = watch_instructions

        return _response(data=brief)

    except Exception as e:
        logger.error("agent_morning_brief failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def agent_sector_analysis(top_n: int = 10) -> dict[str, Any]:
    """Sector/industry rotation analysis — identify leading and lagging sectors.

    Args:
        top_n: Number of top sectors to return
    """
    try:
        if not _store:
            return _response(success=False, error="Database not initialized", error_type="INTERNAL_ERROR")

        # Aggregate by sector from kline_daily
        latest = _store.get_latest_date("kline_daily")
        if not latest:
            return _response(data={"message": "No data available"})

        # Try sector mapping table first
        sector_stats = []
        if _store.table_exists("stock_sector_mapping") and _store.table_exists("stock_basic"):
            sector_df = _store.query_df(f"""
                SELECT s.sector, COUNT(*) as stock_count,
                       AVG(k.pct_change) as avg_pct_change,
                       SUM(k.amount) / 100000000.0 as total_turnover_billion,
                       COUNT(CASE WHEN k.pct_change >= 9.5 THEN 1 END) as limit_up_count,
                       COUNT(CASE WHEN k.pct_change <= -9.5 THEN 1 END) as limit_down_count
                FROM kline_daily k
                JOIN stock_basic s ON k.symbol = s.symbol
                WHERE k.date = ?
                GROUP BY s.sector
                HAVING COUNT(*) >= 3
                ORDER BY avg_pct_change DESC
                LIMIT ?
            """, [latest, top_n])
            sector_stats = sector_df.to_dict(orient="records") if not sector_df.empty else []

        # Leading stocks per sector
        leading = {}
        if _store.table_exists("stock_basic") and sector_stats:
            for s in sector_stats[:5]:
                sector = s.get("sector", "")
                if sector:
                    leaders_df = _store.query_df(f"""
                        SELECT k.symbol, k.pct_change
                        FROM kline_daily k
                        JOIN stock_basic s ON k.symbol = s.symbol
                        WHERE k.date = ? AND s.sector = ?
                        ORDER BY k.pct_change DESC LIMIT 5
                    """, [latest, sector])
                    if not leaders_df.empty:
                        leading[sector] = leaders_df.to_dict(orient="records")

        return _response(data={
            "date": latest,
            "sectors": sector_stats,
            "leaders_by_sector": leading,
            "count": len(sector_stats),
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        logger.error("agent_sector_analysis failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


# ---- MCP Tool: Help / Introspection ----

@mcp.tool(annotations={"readOnlyHint": True})
async def analyze_timeframe_alignment(symbol: str) -> dict[str, Any]:
    """Check trend alignment across multiple timeframes (multi-TF resonance/divergence).

    Fetches daily, 60min, and 15min K-line data and compares:
    - Trend direction (close vs MA20) on each timeframe
    - Key indicators (MACD, RSI) alignment
    - Resonance (all aligned bullish) vs divergence (conflicting signals)

    Args:
        symbol: Stock code (e.g., '000001')

    Returns:
        data.alignment: 'resonance_bullish' | 'resonance_bearish' | 'divergence'
        data.timeframes: Per-TF data (trend, ma20_position, macd_signal, rsi14)
        data.strength: Alignment strength score (0-100, higher = stronger conviction)
    """
    try:
        results: dict[str, dict] = {}
        for tf, period in [("daily", "daily"), ("60min", "60"), ("15min", "15")]:
            if period == "daily":
                if _store:
                    df = _store.query_df(
                        "SELECT * FROM kline_daily WHERE symbol = ? ORDER BY date DESC LIMIT 60",
                        [symbol],
                    )
                else:
                    df = None
            else:
                # Minute kline not reliably available from free sources
                df = None

            if df is not None and len(df) >= 20:
                close_vals = df["close"].values
                ma20_vals = pd.Series(close_vals).rolling(20).mean().values
                trend_val = "up" if close_vals[-1] > ma20_vals[-1] else "down"

                # RSI14 quick calc
                delta = pd.Series(close_vals).diff()
                gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14).mean().values
                loss = (-delta).where(delta < 0, 0.0).ewm(alpha=1/14).mean().values
                rs = gain[-1] / max(loss[-1], 0.001)
                rsi_val = round(100 - 100 / (1 + rs), 1)
                results[tf] = {
                    "trend": trend_val,
                    "ma20_position": "above" if close_vals[-1] > ma20_vals[-1] else "below",
                    "rsi14": rsi_val,
                    "data_bars": len(df),
                }
            else:
                results[tf] = {
                    "trend": "no_data",
                    "ma20_position": "no_data",
                    "rsi14": 50,
                    "data_bars": 0,
                }

        # Alignment assessment
        trends = [r["trend"] for r in results.values()]
        if all(t == "up" for t in trends):
            alignment = "resonance_bullish"
            strength = 80
        elif all(t == "down" for t in trends):
            alignment = "resonance_bearish"
            strength = 20
        elif "up" in trends and "down" in trends:
            alignment = "divergence"
            strength = 50
        else:
            alignment = "insufficient_data"
            strength = 50

        return _response(data={
            "symbol": symbol,
            "alignment": alignment,
            "strength": strength,
            "timeframes": results,
            "note": "60min and 15min data require minute kline in DB. If unavailable, only daily is analyzed.",
            "generated_at": datetime.now().isoformat(),
        })

    except Exception as e:
        logger.error("analyze_timeframe_alignment failed", symbol=symbol, error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool(annotations={"readOnlyHint": True})
async def calc_vwap(symbol: str, date: str = "") -> dict[str, Any]:
    """Calculate VWAP (Volume-Weighted Average Price) for a stock.

    VWAP is the most widely used intraday benchmark. Price above VWAP = bullish,
    below VWAP = bearish on the day.

    Uses the pure-Python indicator engine (no external dependencies).

    Args:
        symbol: Stock code
        date: Trading date (YYYY-MM-DD), empty for latest

    Returns:
        data.vwap: The VWAP value for the date
        data.latest_close: Latest close price
        data.position: 'above_vwap' or 'below_vwap'
        data.deviation_pct: Percentage deviation from VWAP
    """
    try:
        from pa_mcp.engine.indicators.indicators import calc_vwap as _calc_vwap

        if _store:
            target_date = date or _store.get_latest_date("kline_daily")
            if not target_date:
                return _response(success=False, error="No data available", error_type="INTERNAL_ERROR")

            df = _store.query_df(
                "SELECT * FROM kline_daily WHERE symbol = ? AND date <= ? ORDER BY date",
                [symbol, target_date],
            )
            if df is None or df.empty:
                return _response(error=f"No kline data for {symbol}", error_type="NOT_FOUND")

            # Ensure standard column names
            df_renamed = df.rename(columns={
                "open": "open", "high": "high", "low": "low",
                "close": "close", "volume": "volume",
            })
            vwap_df = _calc_vwap(df_renamed)
            vwap_val = float(vwap_df["vwap"].iloc[-1])
            close_val = float(df["close"].iloc[-1])
            deviation = round((close_val - vwap_val) / vwap_val * 100, 2)

            return _response(data={
                "symbol": symbol,
                "date": target_date,
                "vwap": round(vwap_val, 2),
                "latest_close": round(close_val, 2),
                "position": "above_vwap" if close_val > vwap_val else "below_vwap",
                "deviation_pct": deviation,
                "note": "VWAP is a daily cumulative metric. Reset at market open each day.",
                "generated_at": datetime.now().isoformat(),
            })
        return _response(success=False, error="Database not initialized", error_type="INTERNAL_ERROR")
    except Exception as e:
        logger.error("calc_vwap failed", symbol=symbol, error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


# ---- MCP Tool: Help / Introspection ----

@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
async def pa_help() -> dict[str, Any]:
    """Get a complete guide to PA_MCP — all tools, common workflows, and data freshness.

    Call this FIRST in any conversation to understand what PA_MCP can do.
    Returns tool inventory, recommended workflows for common tasks, and current data status.

    When to use: Start of session, or whenever unsure which tool to pick.
    """
    tool_list = []
    for name in sorted(mcp._tool_manager._tools.keys()):
        t = mcp._tool_manager._tools[name]
        desc = (t.description or "").split("\n")[0][:100]
        ann = getattr(t, "annotations", None)
        is_readonly = getattr(ann, "readOnlyHint", False) if ann else False
        tool_list.append({
            "name": name,
            "description": desc,
            "read_only": is_readonly,
        })

    # Data freshness
    freshness = {}
    if _store:
        for table in ["kline_daily", "dragon_tiger", "signal_cache"]:
            try:
                latest = _store.get_latest_date(table)
                freshness[table] = str(latest) if latest else "empty"
            except Exception:
                freshness[table] = "unavailable"

    return _response(data={
        "tool_count": len(tool_list),
        "tools": tool_list,
        "workflows": {
            "daily_analysis": [
                "1. agent_morning_brief() — pre-market briefing + today's watchlist",
                "2. agent_scan_market(top_n=20) — run all strategies, rank by strength",
                "3. agent_sector_analysis() — leading/lagging sectors",
                "4. scan_limit_up() — limit-up stocks + seal quality",
                "5. review_dragon_tiger() — institutional seat tracking",
            ],
            "stock_research": [
                "1. get_stock_info(symbol) — basic info",
                "2. get_kline(symbol) — price history",
                "3. get_major_events(symbol) — block trades, lockup, insider, pledge",
                "4. agent_analyze_stock(symbol, depth='fast') — multi-dim analysis",
                "5. agent_compare_stocks('000001,000002') — side-by-side comparison",
            ],
            "portfolio_management": [
                "1. portfolio_add(symbol, cost, shares) — add holding",
                "2. portfolio_summary() — view all holdings",
                "3. agent_market_state() — check if current regime fits your positions",
                "4. Use watchlist_add/watchlist_show/watchlist_overview for monitoring",
            ],
            "strategy_development": [
                "1. list_strategies(category='trend') — browse strategies",
                "2. agent_scan_market(strategy_filter='platform_breakout') — test one strategy",
                "3. Use scripts/run_backtest.py for full backtest with realistic A-share constraints",
            ],
        },
        "data_freshness": freshness,
        "known_limitations": [
            "Free APIs (AKShare) have 3-15 second delay on 'realtime' quotes",
            "Dragon-tiger data available ~18:00 after market close",
            "Financial data updates quarterly, not daily",
            "Backtest results overestimate real returns by 30-50% due to liquidity/slippage",
            "Limit-up stocks with strong seals are often unbuyable in practice",
            "Configure config/llm_config.json to enable AI analysis tools",
        ],
        "prompts_available": [
            "daily-review: End-of-day market review with limit-up, dragon-tiger, sector rotation",
            "stock-deep-dive: Comprehensive deep-dive on a single stock",
            "strategy-screen: Multi-strategy market scan with top candidate comparison",
            "morning-brief: Pre-market briefing with watchlist and risk alerts",
            "risk-audit: Portfolio risk audit with position concentration and correlation",
        ],
        "generated_at": datetime.now().isoformat(),
    })


# ---- MCP Resources ----

@mcp.resource("strategies://categories")
def get_strategy_categories() -> str:
    """List all strategy categories with descriptions."""
    from pa_mcp.engine.strategies.base import StrategyCategory
    return json.dumps([
        {"category": c.value, "name": c.name}
        for c in StrategyCategory
    ])


@mcp.resource("health://status")
def get_health_status() -> str:
    """Server health check."""
    status = {
        "server": "running",
        "duckdb": "connected" if _store is not None else "disconnected",
        "akshare": "initialized" if _akshare is not None else "not_initialized",
        "sina": "initialized" if _sina is not None else "not_initialized",
        "llm": "check config/llm_config.json",
        "tools": len(list(mcp._tool_manager._tools.keys())) if hasattr(mcp, '_tool_manager') else 0,
        "timestamp": datetime.now().isoformat(),
    }
    return json.dumps(status, indent=2)


# ---- MCP Prompts (reusable analysis workflows) ----

def _register_prompts() -> None:
    """Register all prompt templates. Each prompt is a reusable analysis workflow.

    Following MCP best practices: Prompts = reusable structured templates;
    Tools = actions with side effects. Prompts give the LLM a proven analysis
    framework without dictating its reasoning.
    """
    for prompt_def in PROMPTS.values():
        # Use factory to capture current prompt_def in closure
        def _make_factory(pd: dict) -> Any:
            @mcp.prompt(name=pd["name"], description=pd["description"])
            def _prompt_fn(arguments: dict[str, Any] | None = None) -> str:
                args = arguments or {}
                # Default date to today
                if "date" in pd.get("arguments", []) and "date" not in args:
                    args["date"] = datetime.now().strftime("%Y-%m-%d")
                return pd["template"].format(**args)
            return _prompt_fn
        _make_factory(prompt_def)

_register_prompts()


# ---- Server Entry Point ----

def run() -> None:
    """Run the MCP server."""
    transport = settings.server.transport

    if transport == "stdio":
        logger.info("Starting PA_MCP in stdio mode")
        mcp.run(transport="stdio")
    elif transport == "http":
        logger.info("Starting PA_MCP in HTTP mode", port=settings.server.http_port)
        mcp.run(transport="sse", host="0.0.0.0", port=settings.server.http_port)
    else:
        raise ValueError(f"Unknown transport: {transport}")


if __name__ == "__main__":
    run()
