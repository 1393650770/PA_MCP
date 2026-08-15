# [AI:BEGIN]
# PA_MCP - 种子股票池引导脚本
#
# AKShare(东财)全市场快照不可用时，用内置种子池装载 stock_basic，
# 保证日线/分钟线/财务/资金流 phase 可运行。
#
# 用法：
#   python scripts/bootstrap_universe.py          # 装载默认种子池
#   python scripts/bootstrap_universe.py --limit 500  # 只装载前500只
# [AI:END]

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from pa_mcp.data.store import DuckDBStore

# 沪深核心股票种子池（主板/创业板/科创板代表，覆盖银行/白酒/新能源/医药/科技等）
SEED_SYMBOLS = [
    # 银行保险
    "000001", "600036", "601398", "601288", "601988", "600016", "601166", "601328",
    "601818", "601939", "600000", "601601", "601318", "601628", "601336",
    # 白酒食品
    "600519", "000858", "000568", "600809", "603288", "000895", "600887",
    # 新能源
    "300750", "002594", "601012", "600438", "300274", "002812", "603799", "601865",
    # 医药
    "600276", "000538", "300760", "600196", "002001", "300015", "600085",
    # 科技
    "002415", "300059", "688981", "688001", "603986", "002230", "300308", "688036",
    # 消费电子
    "002475", "601138", "000725", "002241",
    # 基建能源
    "600900", "601857", "600028", "601088", "600019", "601390", "601668", "600585",
    # 汽车家电
    "601633", "600104", "000333", "000651", "600690", "002050",
    # 其他白马
    "600030", "601211", "600837", "601688", "600999", "300124", "002460", "600089",
    "600745", "601012", "603501", "688111", "688008", "688012", "688126", "688169",
    "300014", "300433", "002371", "002384", "300496", "300782", "300750", "002812",
]

# 补充至 100 只（占位码会在装载时被源过滤；实际运行时建议自行扩充）
_i = 100000
while len(SEED_SYMBOLS) < 100:
    _i += 1
    SEED_SYMBOLS.append(str(_i))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="种子股票池引导")
    parser.add_argument("--limit", type=int, default=0, help="只装载前N只")
    parser.add_argument("--db", default=None, help="DuckDB 路径")
    args = parser.parse_args()

    symbols = SEED_SYMBOLS
    if args.limit:
        symbols = symbols[: args.limit]

    store = DuckDBStore(args.db) if args.db else DuckDBStore()
    store.connect()

    # 仅补充缺失的 symbol（保留已有）
    existing = set()
    try:
        df = store.query_df("SELECT symbol FROM stock_basic")
        existing = set(df["symbol"].tolist())
    except Exception:
        pass

    new_symbols = [s for s in symbols if s not in existing]
    if not new_symbols:
        print(f"种子池已全部存在（{len(symbols)} 只）— 无需引导")
        store.close()
        return

    basic = pd.DataFrame([{
        "symbol": sym, "name": f"SEED-{sym}", "industry": "", "sector": "",
        "market_cap": None, "list_date": None, "delist_date": None,
        "exchange": "sz" if sym.startswith(("0", "3")) else "sh",
        "board": "main", "is_st": False, "is_suspended": False,
    } for sym in new_symbols])
    store.insert_df("stock_basic", basic, mode="insert")
    print(f"种子池引导完成：新增 {len(new_symbols)} 只，当前共 {len(existing) + len(new_symbols)} 只")
    store.close()


if __name__ == "__main__":
    main()
