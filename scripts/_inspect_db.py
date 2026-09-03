"""Read-only inspect: checkpoint table + kline latest dates."""
import duckdb
con = duckdb.connect(r"D:\Project\AI\PA_MCP\PA_MCP\data\pa_mcp.duckdb", read_only=True)
print("== checkpoint ==")
try:
    df = con.execute("SELECT job, COUNT(*) n FROM ingestion_checkpoint GROUP BY job").fetchall()
    for r in df:
        print(r)
except Exception as e:
    print("no checkpoint table:", e)
print("== kline latest dates per symbol (sample) ==")
try:
    df = con.execute("SELECT symbol, MAX(date) d FROM kline_daily GROUP BY symbol ORDER BY symbol LIMIT 10").fetchall()
    for r in df:
        print(r)
except Exception as e:
    print("err:", e)
print("== overall max ==")
try:
    print(con.execute("SELECT MAX(date) FROM kline_daily").fetchone())
except Exception as e:
    print("err:", e)
con.close()
