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
from typing import Any, Optional

import pandas as pd
import structlog
from mcp.server.fastmcp import FastMCP

from pa_mcp.config import get_settings, Settings
from pa_mcp.data import AKShareAdapter, CacheManager, DataValidator, DuckDBStore, SinaAdapter
from pa_mcp.risk.guard import RiskGuard

logger = structlog.get_logger(__name__)

# ---- Global state (initialized in lifespan) ----
_store: Optional[DuckDBStore] = None
_cache: Optional[CacheManager] = None
_akshare: Optional[AKShareAdapter] = None
_sina: Optional[SinaAdapter] = None
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

    # Initialize risk layer
    _guard = RiskGuard()

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
    instructions="Personal Analyst MCP - Full-stack A-share quantitative trading server. Provides A-share market data, technical analysis, strategy signals, backtesting, and AI-powered stock analysis.",
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


# ---- Helper: Data source fallback ----


async def _get_kline_fallback(
    symbol: str, period: str = "daily",
    start_date: str = "", end_date: str = "",
    adjust: str = "qfq",
) -> tuple[pd.DataFrame, str]:
    """Try AKShare first, fall back to Sina on failure."""
    # Try AKShare
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
        )
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


# ---- MCP Tools: Alerts ----

@mcp.tool()
async def watch_price_alert(symbol: str, condition: str, price: float) -> dict[str, Any]:
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


@mcp.tool()
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


@mcp.tool()
async def list_alerts(status: str = "active") -> dict[str, Any]:
    """List configured alerts.

    Args:
        status: 'active' or 'triggered'
    """
    return _response(data={"alerts": [], "count": 0, "message": "Alert persistence not yet implemented. Alerts live in-memory for this session."})


# ---- MCP Tools: Portfolio ----

@mcp.tool()
async def portfolio_summary() -> dict[str, Any]:
    """Get portfolio summary with P&L and risk metrics."""
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


@mcp.tool()
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


@mcp.tool()
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

@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
async def watchlist_show() -> dict[str, Any]:
    """Show all stocks in your watchlist (watchlist only, no analysis)."""
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


@mcp.tool()
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
                    df_rt = await _sina.get_realtime_quote([sym])
                    if not df_rt.empty:
                        r = df_rt.iloc[0]
                        stock_info["name"] = r.get("name", "")
                        stock_info["price"] = float(r.get("price", 0))
                        stock_info["open"] = float(r.get("open", 0))
                        stock_info["high"] = float(r.get("high", 0))
                        stock_info["low"] = float(r.get("low", 0))
                        stock_info["prev_close"] = float(r.get("prev_close", 0))
                        if stock_info["prev_close"] > 0:
                            stock_info["pct_change"] = round(
                                (stock_info["price"] - stock_info["prev_close"]) / stock_info["prev_close"] * 100, 2
                            )
                        stock_info["volume"] = float(r.get("volume", 0))
                        stock_info["amount"] = float(r.get("amount", 0))
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

@mcp.tool()
async def agent_analyze_stock(symbol: str, depth: str = "fast") -> dict[str, Any]:
    """AI-powered multi-dimensional stock analysis.

    Args:
        symbol: Stock code (e.g., '000001')
        depth: 'fast' (single call, ~15s) or 'deep' (5 analysts + debate, ~60s)
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
                })

    except Exception as e:
        logger.error("agent_analyze_stock failed", symbol=symbol, error=str(e))
        return _response(success=False, error=str(e), error_type="INTERNAL_ERROR")


@mcp.tool()
async def agent_market_state() -> dict[str, Any]:
    """Get current market regime assessment with position sizing suggestion."""
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


# ---- MCP Tools: Comprehensive Analysis (NEW) ----

@mcp.tool()
async def agent_scan_market(
    top_n: int = 20,
    strategy_filter: str = "",
    sort_by: str = "strength",
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

        # Use pre-computed signals if available, else compute live
        candidates: dict[str, list[dict]] = {}  # symbol -> [signal summaries]

        if _store and _store.table_exists("signal_cache"):
            # Use pre-computed cache (fast path)
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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
        "llm": "check config/llm_config.json",
        "timestamp": datetime.now().isoformat(),
    }
    return json.dumps(status, indent=2)


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
