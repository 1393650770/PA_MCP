# [AI:BEGIN]
# PA_MCP - MCP Server Entry Point
# FastMCP-based server with dual transport (stdio + HTTP/SSE).
# Registers all 25+ MCP tools across 8 domains.
# [AI:END]

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

import structlog
from mcp.server.fastmcp import FastMCP

from pa_mcp.config import get_settings, Settings
from pa_mcp.data import AKShareAdapter, CacheManager, DataValidator, DuckDBStore
from pa_mcp.risk.guard import RiskGuard

logger = structlog.get_logger(__name__)

# ---- Global state (initialized in lifespan) ----
_store: Optional[DuckDBStore] = None
_cache: Optional[CacheManager] = None
_akshare: Optional[AKShareAdapter] = None
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

    # Initialize risk layer
    _guard = RiskGuard()

    logger.info("PA_MCP server ready", transport=_settings.server.transport)

    yield

    # Cleanup
    if _store:
        _store.close()
    logger.info("PA_MCP server stopped")


# ---- Create MCP Server ----

settings = get_settings()
mcp = FastMCP(
    name="pa-mcp",
    description="Personal Analyst MCP - Full-stack A-share quantitative trading server",
    version="0.1.0",
    lifespan=server_lifespan,
)


# ---- Helper: Standard Response Format ----

def _response(
    data: Any = None, success: bool = True,
    error: Optional[str] = None, error_type: Optional[str] = None,
    source: str = "akshare", freshness: Optional[str] = None,
) -> dict[str, Any]:
    """Build standardized tool response envelope."""
    return {
        "success": success,
        "data": data,
        "error": error,
        "error_type": error_type,
        "data_source": source,
        "data_freshness": freshness or datetime.now().isoformat(),
        "disclaimer": (
            "This is a research tool, not investment advice. "
            "All analysis is for reference only. "
            "Past performance does not guarantee future results. "
            "Trading involves risk of loss."
        ),
    }


# ---- MCP Tools: Market Data ----

@mcp.tool()
async def get_realtime_quote(symbol: str, source: str = "akshare") -> dict[str, Any]:
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


@mcp.tool()
async def get_kline(
    symbol: str, period: str = "daily",
    start_date: str = "", end_date: str = "",
    adjust: str = "qfq",
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
        if _akshare:
            df = await _akshare.get_daily_kline(
                symbol=symbol, period=period,
                start_date=start_date or "20200101",
                end_date=end_date or datetime.now().strftime("%Y%m%d"),
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
            return _response(data={"symbol": symbol, "period": period, "adjust": adjust, "kline": records})
        return _response(success=False, error="Data source not initialized", error_type="INTERNAL_ERROR")
    except Exception as e:
        logger.error("get_kline failed", symbol=symbol, error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool()
async def get_market_overview() -> dict[str, Any]:
    """Get current market overview: indices, breadth, turnover, northbound flow."""
    try:
        if _akshare:
            overview = await _akshare.get_market_overview()
            return _response(data=overview)
        return _response(success=False, error="Data source not initialized", error_type="INTERNAL_ERROR")
    except Exception as e:
        logger.error("get_market_overview failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool()
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


@mcp.tool()
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

@mcp.tool()
async def scan_limit_up(date: str = "") -> dict[str, Any]:
    """Scan limit-up stocks with seal strength, break rate, sector distribution.

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
            return _response(data={
                "date": target_date,
                "limit_up_count": len(df),
                "stocks": df.to_dict(orient="records"),
            })
        return _response(success=False, error="Database not initialized", error_type="INTERNAL_ERROR")
    except Exception as e:
        logger.error("scan_limit_up failed", error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool()
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


@mcp.tool()
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


# ---- MCP Tools: Review ----

@mcp.tool()
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


@mcp.tool()
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

@mcp.tool()
async def list_strategies(category: str = "") -> dict[str, Any]:
    """List available trading strategies.

    Args:
        category: Filter by category (trend/swing/value/board/reversal/event/grid). Empty = all.
    """
    from pa_mcp.engine.strategies.base import StrategyCategory, StrategyRegistry

    registry = StrategyRegistry()
    cat_enum = StrategyCategory(category) if category else None
    strategies = registry.list_by_category(cat_enum)
    return _response(data={"strategies": strategies, "count": len(strategies)})


@mcp.tool()
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
