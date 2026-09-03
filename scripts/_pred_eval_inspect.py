import duckdb

con = duckdb.connect('data/pa_mcp.duckdb', read_only=True)
print('--- prediction_log status counts ---')
print(con.execute("SELECT status, COUNT(*) c FROM prediction_log GROUP BY status ORDER BY c DESC").fetchdf().to_string())
print()
print('--- recent predictions ---')
print(con.execute("SELECT id, symbol, predict_date, horizon, direction, status, actual_return_pct, mode FROM prediction_log ORDER BY id DESC LIMIT 20").fetchdf().to_string())
print()
print('--- tables ---')
print(con.execute("SHOW TABLES").fetchdf().to_string())
try:
    print()
    print('--- sector_prediction status counts ---')
    print(con.execute("SELECT status, COUNT(*) c FROM sector_prediction GROUP BY status ORDER BY c DESC").fetchdf().to_string())
    print()
    print('--- recent sector predictions ---')
    print(con.execute("SELECT * FROM sector_prediction ORDER BY id DESC LIMIT 10").fetchdf().to_string())
except Exception as e:\n    print('sector_prediction query failed:', e)
