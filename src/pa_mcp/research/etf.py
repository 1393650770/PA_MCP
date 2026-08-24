# [AI:BEGIN]
# PA_MCP - Research: ETF 数据支持（列表 / 行情 / IOPV 折溢价 / 扫描）
#
# 背景：项目此前只支持个股。ETF 差异点：无财务（无 PE/PB 单股财报）、
# 与指数强相关、有 IOPV（盘中参考净值）与折溢价、技术面分析完全适用。
#
# 数据源（实测结论）：
#   - ETF 列表：东财 push2delay clist `fs=b:MK0021`（push2 主域名时断时连，
#     delay 域名稳定）；新浪 etf_hq 节点返回空、腾讯排行接口 400，均不可用
#   - 行情+IOPV：腾讯 qt.gtimg.cn 批量（88 字段，idx51=IOPV），
#     折溢价 = (price - iopv) / iopv；实测正常（如 510300 折溢 -0.06%）
#   - K 线：腾讯/新浪/同花顺前缀已修复（5→sh、1→sz）
#
# 设计：
#   - is_etf / is_lof：代码段识别（510/588/56x 沪、159/16x 深）
#   - fetch_etf_list：东财列表（TTL 缓存，避免高频调用触发限流）
#   - fetch_etf_quotes：腾讯批量行情 + IOPV + 折溢价 + is_stale
#   - ETF 名称缓存：get_stock_name 回退查询（symbols.py 白马库不含 ETF）
#   - 扫描复用 scan_market_ui 的信号计算（etf 池模式）
# [AI:END]

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---- ETF 代码识别 ----

# 沪市 ETF 段：510xxx（宽基/行业）、511xxx（货币/债券）、512/513/515/516/517
#   /518/519/560/561/562/563/564/565/566/567/569/588（科创）
# 深市 ETF 段：159xxx、16xxxx
_ETF_PREFIXES_SH = ("510", "511", "512", "513", "515", "516", "517",
                    "518", "519", "560", "561", "562", "563", "564",
                    "565", "566", "567", "569", "588")
_ETF_PREFIXES_SZ = ("159", "160", "161", "162", "163", "164", "165",
                    "166", "167", "168", "169")


def is_etf(symbol: str) -> bool:
    """ETF/LOF 代码识别（5/1 开头号段）。"""
    code = (symbol or "").strip().lower()
    if code[:2] in ("sh", "sz"):
        code = code[2:]
    return (code.startswith(_ETF_PREFIXES_SH)
            or code.startswith(_ETF_PREFIXES_SZ))


def etf_exchange(symbol: str) -> str:
    """ETF 所属市场（sh/sz）。"""
    code = (symbol or "").strip().lower()
    if code[:2] in ("sh", "sz"):
        return code[:2]
    return "sh" if code.startswith(_ETF_PREFIXES_SH) else "sz"


# ---- ETF 名称缓存（get_stock_name 回退） ----

_name_cache: dict[str, str] = {}
_name_lock = threading.Lock()


def get_etf_name(symbol: str) -> str:
    """ETF 名称（fetch_etf_list 成功时自动填充的缓存，同步安全）。"""
    with _name_lock:
        return _name_cache.get(symbol, "")


# ---- ETF 列表（东财 push2delay clist） ----

# 5 个 ETF 板块合并 = 全市场（与 AKShare fund_etf_spot_em 同源）：
# MK0021 沪市ETF / MK0022 深市ETF / MK0023 沪市LOF / MK0024 深市LOF /
# MK0827 跨市场ETF。单板块（如 b:MK0021）仅 98 只精选，必须合并。
_ETF_FS = "b:MK0021,b:MK0022,b:MK0023,b:MK0024,b:MK0827"
_ETF_LIST_URL = (
    "https://push2delay.eastmoney.com/api/qt/clist/get"
    "?pn={page}&pz=100&po=1&np=1&fltt=2&invt=2"
    "&ut=bd1d9ddb04089700cf9c27f6f7426281&fid=f3"
    f"&fs={_ETF_FS}&fields=f12,f14,f2,f3,f6,f8")
_ETF_LIST_FALLBACK_URL = (
    "https://push2.eastmoney.com/api/qt/clist/get"
    "?pn={page}&pz=100&po=1&np=1&fltt=2&invt=2"
    "&ut=bd1d9ddb04089700cf9c27f6f7426281&fid=f3"
    f"&fs={_ETF_FS}&fields=f12,f14,f2,f3,f6,f8")

_list_cache: Optional[list[dict[str, Any]]] = None
_list_ts: float = 0.0
_list_lock = threading.Lock()
_LIST_TTL = 600.0  # 10 分钟


def clear_etf_list_cache() -> None:
    """清空 ETF 列表缓存（测试/手动刷新）。"""
    global _list_cache, _list_ts
    with _list_lock:
        _list_cache, _list_ts = None, 0.0


async def fetch_etf_list(limit: int = 300, use_cache: bool = True) -> list[dict[str, Any]]:
    """沪深 ETF/LOF 列表（东财，涨幅降序，TTL 缓存）。

    Returns:
        [{"symbol", "name", "price", "change_pct", "amount_billion"}]
        失败返回 []（调用方降级）。
    """
    global _list_cache, _list_ts
    if use_cache:
        with _list_lock:
            if _list_cache is not None and time.time() - _list_ts < _LIST_TTL:
                return _list_cache[:limit]

    import json
    import urllib.request

    rows: list[dict[str, Any]] = []
    for url_tpl in (_ETF_LIST_URL, _ETF_LIST_FALLBACK_URL):
        try:
            # 分页拉全量（每页 100；用东财 total 判断完整性，
            # 单页限流返空不提前停——重试一次再判）
            total = None
            for page in range(1, 12):
                raw = await _fetch_json(url_tpl.format(page=page))
                data = (raw or {}).get("data") or {}
                if total is None:
                    total = int(data.get("total") or 0)
                diff = data.get("diff") or []
                if not diff:
                    if total and len(rows) >= total:
                        break
                    raw = await _fetch_json(url_tpl.format(page=page))  # 限流重试
                    diff = ((raw or {}).get("data") or {}).get("diff") or []
                    if not diff:
                        break
                for r in diff:
                    code = str(r.get("f12", ""))
                    name = str(r.get("f14", ""))
                    if not (code.isdigit() and len(code) == 6 and name):
                        continue
                    if not is_etf(code):
                        continue  # 板块里偶发混入非 ETF
                    rows.append({
                        "symbol": code,
                        "name": name,
                        "price": _f(r, "f2"),
                        "change_pct": _f(r, "f3"),
                        "amount_billion": round(_f(r, "f6") / 1e8, 2),
                        "turnover_pct": _f(r, "f8"),
                    })
            if rows:
                break
        except Exception as e:  # noqa: BLE001
            logger.warning("ETF 列表拉取失败: %s", str(e)[:60])

    if rows:
        # 同时填充名称缓存（get_etf_name 同步查询）
        with _name_lock:
            _name_cache.update({e["symbol"]: e["name"] for e in rows})
    if use_cache and rows:
        with _list_lock:
            _list_cache, _list_ts = rows, time.time()
    return rows[:limit]


def _f(r: dict, key: str, default: float = 0.0) -> float:
    try:
        v = r.get(key)
        return float(v) if v not in (None, "-", "") else default
    except (TypeError, ValueError):
        return default


async def _fetch_json(url: str) -> dict:
    import asyncio
    import json
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = await asyncio.to_thread(urllib.request.urlopen, req, timeout=12)
    return json.loads(raw.read().decode("utf-8", errors="ignore"))


# ---- ETF 批量行情（腾讯 + IOPV 折溢价） ----

def _tencent_etf_code(symbol: str) -> str:
    code = (symbol or "").strip().lower()
    if code[:2] in ("sh", "sz"):
        return code
    return f"{etf_exchange(code)}{code}"


async def fetch_etf_quotes(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """腾讯批量 ETF 行情（含 IOPV / 折溢价 / is_stale）。

    Returns:
        {symbol: {"name", "price", "change_pct", "iopv", "premium_pct",
                  "amount_billion", "is_stale"}}
    """
    if not symbols:
        return {}
    raw = await _fetch_quotes_raw(symbols)
    return await _quotes_from_raw(symbols, raw)


async def _fetch_quotes_raw(symbols: list[str]) -> str:
    """腾讯批量行情原始文本（独立函数，便于测试 mock）。"""
    import asyncio
    import urllib.request

    codes = [_tencent_etf_code(s) for s in symbols]
    url = "https://qt.gtimg.cn/q=" + ",".join(codes)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = await asyncio.to_thread(
        lambda: urllib.request.urlopen(req, timeout=10).read().decode(
            "gbk", errors="ignore"))
    return raw


async def _quotes_from_raw(symbols: list[str], raw: str) -> dict[str, dict[str, Any]]:
    """解析腾讯批量行情文本（与拉取分离，便于测试注入固定响应）。"""
    out: dict[str, dict[str, Any]] = {}
    for line in raw.strip().split(";"):
        if '"' not in line or "=" not in line:
            continue
        key = line.split("=")[0].split("_")[-1]
        vals = line.split('"')[1].split("~")
        if len(vals) < 52:
            continue
        code = (key[2:] if key[:2].lower() in ("sh", "sz") else key)
        price = _f({"f": vals[3]}, "f")
        prev = _f({"f": vals[4]}, "f")
        iopv_raw = vals[51] if len(vals) > 51 else ""
        try:
            iopv = float(iopv_raw)
        except (TypeError, ValueError):
            iopv = 0.0
        premium = ((price / iopv - 1) * 100
                   if iopv > 0 and price > 0 else None)
        out[code] = {
            "symbol": code,
            "name": vals[1],
            "price": price,
            "prev_close": prev,
            "change_pct": _f({"f": vals[32]}, "f"),
            "open": _f({"f": vals[5]}, "f"),
            "high": _f({"f": vals[33]}, "f"),
            "low": _f({"f": vals[34]}, "f"),
            "amount_billion": round(_f({"f": vals[37]}, "f") / 1e4, 2),
            "turnover_pct": _f({"f": vals[38]}, "f"),
            "iopv": round(iopv, 4) if iopv else None,
            "premium_pct": round(premium, 2) if premium is not None else None,
            "is_stale": (price > 0 and price == prev and _f({"f": vals[37]}, "f") == 0),
        }
    return out


def format_etf_line(q: dict[str, Any]) -> str:
    """单只 ETF 行情 → 文本行（含折溢价标注）。"""
    prem = q.get("premium_pct")
    prem_s = (f"折溢{prem:+.2f}%" if prem is not None else "无IOPV")
    stale = " ⚠旧数据" if q.get("is_stale") else ""
    return (f"{q.get('symbol')} {q.get('name')} 价{q.get('price')} "
            f"{q.get('change_pct', 0):+.2f}% {prem_s}"
            f"{stale}")
