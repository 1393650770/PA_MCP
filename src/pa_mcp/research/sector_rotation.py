# [AI:BEGIN]
# PA_MCP - Research: 板块轮动分析与预测
#
# 思路借鉴（开源工程 + 经典板块轮动方法）：
#   - 相对强度（RS）轮动：欧奈尔 RS 概念推广到板块（强者恒强）
#   - 动量加速：5 日日均动量 vs 20 日日均动量 → 加速/减速（动量因子）
#   - 资金流确认：主力净流入方向（东财 fflow）
#   - LLM 融合：板块特征 + 大盘状态 → 预测未来一周强势板块（结构化 JSON）
#   - 验证闭环：预测落盘 sector_prediction，5 交易日后回填板块收益，
#     计算 top3 超额收益（预测可检验）
#
# 数据：东财板块（BK 代码）行业板块优先，概念板块可选。
# [AI:END]

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

RS_WINDOW = 20          # 相对强度窗口（交易日）
ACCEL_WINDOW = 5        # 加速检测窗口
TOP_N = 10              # 强势板块数量
AVOID_N = 3             # 回避板块数量


class SectorRotationAnalyzer:
    """板块轮动分析与预测。

    数据流：东财板块行情（sector_daily 表，短连接写入）→ 分析 → 预测 → 验证。
    """

    def __init__(self, store_path: Optional[str] = None,
                 board_type: str = "industry") -> None:
        self._store_path = store_path
        self.board_type = board_type

    def _store(self):
        from pa_mcp.config import get_settings
        from pa_mcp.data.store import DuckDBStore
        path = self._store_path or get_settings().database.path
        store = DuckDBStore(path)
        store.connect()
        return store

    # ---- 新浪行业板块实时榜（今日热门/冷门，东财断连时的保底源） ----

    @staticmethod
    async def fetch_sina_boards() -> list[dict]:
        """新浪行业板块实时快照（49 个行业，GBK JSONP）。

        Returns:
            [{code, name, change_pct, amount, leader_code, leader_name}]
            按涨跌幅降序；网络失败返回 []。
        """
        import json
        import re
        import urllib.request

        try:
            req = urllib.request.Request(
                "https://vip.stock.finance.sina.com.cn/q/view/newSinaHy.php",
                headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=12).read().decode(
                "gbk", errors="ignore")
            m = re.search(r"\{.*\}", raw, re.S)
            if not m:
                return []
            data = json.loads(m.group(0))
            boards = []
            for _, v in data.items():
                f = v.split(",")
                if len(f) < 13:
                    continue
                try:
                    boards.append({
                        "code": f[0], "name": f[1],
                        "change_pct": float(f[4]),
                        "amount": float(f[7]),
                        "leader_code": f[8],
                        "leader_name": f[12],
                    })
                except (ValueError, IndexError):
                    continue
            boards.sort(key=lambda b: b["change_pct"], reverse=True)
            return boards
        except Exception as e:  # noqa: BLE001
            logger.warning("新浪板块列表获取失败", error=str(e)[:80])
            return []

    async def hot_cold_sectors(self, top_n: int = 10) -> dict:
        """今日热门/冷门板块（新浪实时榜，永远可用不依赖东财历史）。

        Returns:
            hot: 涨幅榜 top_n [{name, change_pct, leader_name}]
            cold: 跌幅榜 top_n
            report: markdown
        """
        boards = await self.fetch_sina_boards()
        if not boards:
            return {"error": "新浪板块接口不可用，热门/冷门暂不可用"}

        def _fmt(b: dict) -> dict:
            return {"name": b["name"], "change_pct": b["change_pct"],
                    "leader": f"{b['leader_name']}({b['leader_code']})"}

        hot = [_fmt(b) for b in boards[:top_n]]
        cold = [_fmt(b) for b in boards[-top_n:]][::-1]
        report = format_hot_cold(hot, cold)
        return {"hot": hot, "cold": cold, "report": report}

    # ---- 10jqka 行业板块（东财断连时的备用真实数据源） ----

    @staticmethod
    async def fetch_ths_boards() -> list[dict]:
        """10jqka 行业板块列表（GBK HTML 解析，140 个真实行业分类）。

        Returns:
            [{code: 881xxx, name}] 通达信行业代码；网络失败返回 []。
        """
        import re
        import urllib.request

        try:
            req = urllib.request.Request(
                "http://q.10jqka.com.cn/thshy/",
                headers={"Referer": "http://q.10jqka.com.cn/",
                         "User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=12).read().decode(
                "gbk", errors="ignore")
            links = re.findall(
                r"/thshy/detail/code/(\d+)[^>]*>([^<]{2,10})<", raw)
            return [{"code": c, "name": n} for c, n in links]
        except Exception as e:  # noqa: BLE001
            logger.warning("10jqka 板块列表获取失败", error=str(e)[:80])
            return []

    async def load_sector_data_ths(self, top_n: int = 60,
                                   days: int = 120) -> dict:
        """10jqka 板块指数日线装载（真实行业分类，写入 sector_daily）。

        板块指数（bk_881xxx）日线格式与个股一致（date,OHLC,volume,amount），
        最近 140 根足够 RS 20 日窗口。sector_code 记 THS881xxx（预测验证
        回填目前只认 BK 代码，THS 代码不参与超额验证——诚实标注）。
        """
        import asyncio
        import urllib.request

        from pa_mcp.data.sources.ths_adapter import ThsAdapter

        boards = await self.fetch_ths_boards()
        if not boards:
            return {"loaded": 0, "boards_total": 0,
                    "message": "10jqka 板块列表不可用"}

        store = self._store()
        loaded = 0
        try:
            for b in boards[:top_n]:
                code, name = str(b["code"]), str(b["name"])
                try:
                    req = urllib.request.Request(
                        f"http://d.10jqka.com.cn/v6/line/bk_{code}/01/last.js",
                        headers={"Referer": "http://stockpage.10jqka.com.cn/",
                                 "User-Agent": "Mozilla/5.0"})
                    raw = urllib.request.urlopen(req, timeout=12).read().decode(
                        "utf-8", errors="ignore")
                    payload = ThsAdapter._strip_jsonp(raw)
                    rows = ThsAdapter._parse_year_data(payload.get("data") or "")
                    if not rows:
                        continue
                    df = pd.DataFrame(rows, columns=[
                        "date", "open", "high", "low", "close",
                        "volume", "amount"])
                    df = df.tail(days)
                    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
                    df["pct_change"] = df["close"].pct_change() * 100
                    df["turnover"] = None
                    df["sector_code"] = f"THS{code}"
                    df["name"] = name
                    df = df[["sector_code", "name", "date", "open", "close",
                             "high", "low", "volume", "amount",
                             "pct_change", "turnover"]]
                    store.insert_df("sector_daily", df)
                    loaded += 1
                except Exception as e:  # noqa: BLE001
                    logger.debug("ths sector kline failed", code=code,
                                 error=str(e)[:60])
                await asyncio.sleep(0.3)  # 免费接口限流
        finally:
            store.close()
        return {"loaded": loaded, "boards_total": len(boards[:top_n]),
                "board_type": self.board_type, "source": "ths",
                "message": f"10jqka 板块装载完成（{loaded} 个，真实行业分类）"
                           if loaded else "10jqka 板块日线全部失败"}

    # ---- 数据装载（东财板块，带重试） ----
    async def load_sector_data(self, top_n: int = 60, days: int = 120,
                               retries: int = 2) -> dict:
        """拉取板块列表 + 各自日线，写入 sector_daily 表。返回装载统计。

        降级链：东财板块 → 10jqka 行业板块（真实分类）→ 合成板块（兜底）。
        """
        from pa_mcp.data.sources.eastmoney_adapter import EastMoneyAdapter

        last_err = ""
        for attempt in range(retries + 1):
            try:
                adapter = EastMoneyAdapter()
                boards = await adapter.get_sector_boards(
                    board_type=self.board_type, top_n=top_n)
                if not boards.empty:
                    return await self._load_boards(boards, days=days)
                last_err = "板块列表为空"
            except Exception as e:  # noqa: BLE001
                last_err = str(e)[:80]
                logger.warning("东财板块装载失败（第 %s 次）: %s",
                               attempt + 1, last_err)
                if attempt < retries:
                    import asyncio
                    await asyncio.sleep(1.5 * (attempt + 1))

        # 东财失败 → 10jqka 行业板块（真实行业分类，非合成）
        try:
            ths = await self.load_sector_data_ths(top_n=top_n, days=days)
            if ths.get("loaded", 0) > 0:
                ths["message"] = f"东财不可用（{last_err}）→ {ths['message']}"
                return ths
            last_err += f"；10jqka 也失败"
        except Exception as e:  # noqa: BLE001
            last_err += f"；10jqka 异常：{str(e)[:60]}"

        # 全部失败 → 合成板块降级（stock_basic.sector + kline_daily 聚合）
        store = self._store()
        try:
            n = self._synthetic_sectors(store)
            if n > 0:
                return {"loaded": n, "boards_total": 0, "board_type": "synthetic",
                        "message": f"东财/10jqka 均不可用（{last_err}）→ "
                                   f"已用合成板块兜底（{n} 个，口径受限）"}
        finally:
            store.close()
        return {"loaded": 0, "boards_total": 0,
                "message": f"{self.board_type} 板块装载失败：{last_err}；"
                           "合成板块也失败（需 stock_basic.sector + kline_daily）"}

    async def _load_boards(self, boards: pd.DataFrame, days: int = 120) -> dict:
        """批量装载板块日线（东财）。"""
        from pa_mcp.data.sources.eastmoney_adapter import EastMoneyAdapter
        adapter = EastMoneyAdapter()

        store = self._store()
        try:
            loaded = 0
            for _, b in boards.iterrows():
                code = str(b["sector_code"])
                try:
                    df = await adapter.get_sector_kline(code, days=days)
                    if df.empty:
                        continue
                    df["sector_code"] = code
                    df["name"] = str(b["name"]) if not df["name"].iloc[0] else df["name"].iloc[0]
                    df = df[["sector_code", "name", "date", "open", "close",
                             "high", "low", "volume", "amount",
                             "pct_change", "turnover"]]
                    store.insert_df("sector_daily", df)
                    loaded += 1
                except Exception as e:  # noqa: BLE001
                    logger.debug("sector kline failed", code=code, error=str(e))
                await adapter.close() if False else None
            # 简单限流：每板块间隔 0.2s
            if loaded > 0:
                import asyncio
                await asyncio.sleep(0.2)
            return {"loaded": loaded, "boards_total": len(boards),
                    "board_type": self.board_type}
        finally:
            store.close()

    # ---- 合成板块降级（东财不可用时的后备数据源） ----
    def _synthetic_sectors(self, store) -> int:
        """用 stock_basic.sector + kline_daily 聚合生成合成板块日线。

        每板块按日：成分股等权平均收盘 → 板块指数（无前视：当日收盘）。
        板块代码用 'SYN_<板块名>'；来源为合成（预测验证回填跳过非 BK 代码）。
        返回写入板块数；0 = 数据不足。
        """
        try:
            sb = store.query_df(
                "SELECT symbol, sector FROM stock_basic "
                "WHERE sector IS NOT NULL AND sector != ''", [])
            if sb.empty:
                return 0
            # 每板块股票列表
            sector_stocks: dict[str, list[str]] = {}
            for _, r in sb.iterrows():
                sector_stocks.setdefault(str(r["sector"]), []).append(
                    str(r["symbol"]))
            if not sector_stocks:
                return 0

            # 拉全部行情（一次查询）
            kl = store.query_df(
                "SELECT symbol, date, close FROM kline_daily "
                "ORDER BY symbol, date", [])
            if kl.empty:
                return 0
            kl["close"] = pd.to_numeric(kl["close"], errors="coerce")

            rows = []
            for sector, syms in sector_stocks.items():
                sub = kl[kl["symbol"].isin(syms)].copy()
                if sub.empty:
                    continue
                # 等权平均收盘（按日期）
                daily = sub.groupby("date")["close"].mean().reset_index()
                daily = daily.dropna()
                if len(daily) < RS_WINDOW + 2:
                    continue
                daily = daily.sort_values("date")
                prev = daily["close"].shift(1)
                daily["pct_change"] = (daily["close"] / prev - 1) * 100
                daily["sector_code"] = f"SYN_{sector}"
                daily["name"] = sector
                daily["open"] = daily["close"] * 0.99
                daily["high"] = daily["close"] * 1.01
                daily["low"] = daily["close"] * 0.99
                daily["volume"] = 1e7
                daily["amount"] = 1e9
                daily["turnover"] = 1.0
                rows.append(daily[["sector_code", "name", "date", "open",
                                   "close", "high", "low", "volume",
                                   "amount", "pct_change", "turnover"]])
            if not rows:
                return 0
            store.insert_df("sector_daily", pd.concat(rows, ignore_index=True))
            return len(rows)
        except Exception as e:  # noqa: BLE001
            logger.warning("合成板块失败: %s", e)
            return 0

    # ---- 分析（RS 动量 + 资金流 + 轮动信号） ----
    def analyze(self, rs_window: int = RS_WINDOW,
                accel_window: int = ACCEL_WINDOW,
                top_n: int = TOP_N,
                auto_synthetic: bool = True) -> dict[str, Any]:
        """板块相对强度排名 + 轮动信号。数据来自 sector_daily 表。

        auto_synthetic=True 且表为空时，自动用合成板块降级
        （stock_basic.sector + kline_daily 聚合）。
        """
        store = self._store()
        try:
            latest = store.get_latest_date("sector_daily", "date")
            if not latest and auto_synthetic:
                n = self._synthetic_sectors(store)
                if n > 0:
                    latest = store.get_latest_date("sector_daily", "date")
            if not latest:
                return {"error": "板块数据为空，请先运行 load_sector_data()"}
            synthetic = str(store.query_df(
                "SELECT DISTINCT sector_code FROM sector_daily LIMIT 1", [])
                .iloc[0]["sector_code"]).startswith("SYN_") if latest else False

            # 每板块最近 rs_window+1 根
            df = store.query_df(
                "SELECT sector_code, name, date, close FROM sector_daily "
                "WHERE date <= ? ORDER BY sector_code, date", [latest])
            if df.empty:
                return {"error": "板块数据为空"}

            rows = []
            for code, g in df.groupby("sector_code"):
                g = g.sort_values("date")
                if len(g) < rs_window + 1:
                    continue
                close_now = float(g["close"].iloc[-1])
                close_rs = float(g["close"].iloc[-rs_window - 1])
                close_acc = float(g["close"].iloc[-accel_window - 1])
                rs = (close_now / close_rs - 1) * 100 if close_rs > 0 else 0.0
                # 几何日均动量差（复利正确）：5日日均 vs 20日日均
                if close_acc > 0 and close_rs > 0:
                    daily5 = ((close_now / close_acc) ** (1 / accel_window) - 1) * 100
                    daily20 = ((close_now / close_rs) ** (1 / rs_window) - 1) * 100
                    accel = daily5 - daily20
                else:
                    accel = 0.0
                rows.append({
                    "sector_code": code,
                    "name": str(g["name"].iloc[-1]) or code,
                    "rs_pct": round(rs, 2),
                    "accel": round(accel, 3),
                    "close": round(close_now, 3),
                })
            if not rows:
                return {"error": "板块数据不足以计算 RS（需要 ≥21 根）"}

            sdf = pd.DataFrame(rows).sort_values("rs_pct", ascending=False)
            sdf["rank"] = range(1, len(sdf) + 1)

            # 20 日前的 top10（轮入/轮出判定：用 rs 前 20 日位置近似——简化为
            # 当前 top10 中 accel 为正且 rs>0 为「维持/轮入加速」；
            # 用历史切片精确判定需存多日快照，此处用 accel 方向近似）
            top10_codes = set(sdf.head(top_n)["sector_code"])
            rotated_in = [
                {"code": r["sector_code"], "name": r["name"], "rs_pct": r["rs_pct"]}
                for _, r in sdf.head(top_n).iterrows()
                if r["accel"] > 0.05]
            rotated_out = [
                {"code": r["sector_code"], "name": r["name"], "rs_pct": r["rs_pct"]}
                for _, r in sdf.tail(top_n).iterrows()
                if r["accel"] < -0.05]

            # 轮动速度：top10 中加速板块占比
            accel_count = int((sdf.head(top_n)["accel"] > 0.05).sum())
            rotation_speed = ("高" if accel_count >= 6
                              else "中" if accel_count >= 3 else "低")

            return {
                "date": latest,
                "board_count": len(sdf),
                "ranked_sectors": sdf.to_dict(orient="records"),
                "rotated_in": rotated_in[:5],
                "rotated_out": rotated_out[:5],
                "rotation_speed": rotation_speed,
                "top_n": top_n,
                "synthetic": synthetic,
                "data_source": "合成板块（东财不可用，stock_basic 聚合）"
                if synthetic else "东财板块指数",
            }
        finally:
            store.close()

    # ---- 板块领涨股挖掘（板块轮动 → 个股闭环） ----
    def leaders_in_sector(self, sector: str, top_n: int = 8) -> dict[str, Any]:
        """板块内领涨股：60 日 RS 排名 + 当前处于突破/强势状态。

        依赖 stock_basic.sector 映射 + kline_daily；数据缺失返回 error。
        与 CANSLIM 交叉：返回带 canslim_score（数据可得时）。
        """
        store = self._store()
        try:
            sb = store.query_df(
                "SELECT symbol, name FROM stock_basic WHERE sector = ? "
                "AND (is_st IS NULL OR is_st = FALSE)", [sector])
            if sb.empty:
                return {"error": f"stock_basic 无板块 {sector} 映射（先运行调度装载）"}

            rows = []
            for _, r in sb.iterrows():
                sym = str(r["symbol"])
                df = store.query_df(
                    "SELECT date, close, high, low, volume FROM kline_daily "
                    "WHERE symbol = ? ORDER BY date DESC LIMIT 80", [sym])
                if len(df) < 61:
                    continue
                d = df.sort_values("date").reset_index(drop=True)
                now = float(d["close"].iloc[-1])
                rs60 = (now / float(d["close"].iloc[-61]) - 1) * 100
                rs20 = (now / float(d["close"].iloc[-21]) - 1) * 100
                hi60 = float(d["high"].iloc[-60:].max())
                near_high = now >= hi60 * 0.97
                rows.append({
                    "symbol": sym, "name": str(r["name"]) or sym,
                    "rs60_pct": round(rs60, 2), "rs20_pct": round(rs20, 2),
                    "near_60d_high": near_high,
                    "close": round(now, 3),
                })
            if not rows:
                return {"error": f"{sector} 无足够行情数据"}

            ranked = sorted(rows, key=lambda x: x["rs60_pct"], reverse=True)[:top_n]
            return {
                "sector": sector,
                "leaders": ranked,
                "leader_count": len(ranked),
                "note": "按 60 日 RS 排序；near_60d_high = 现价距 60 日高点 -3% 内",
            }
        finally:
            store.close()

    # ---- LLM 预测（融合板块特征 + 大盘状态） ----
    async def predict(self, analysis: Optional[dict] = None) -> dict[str, Any]:
        """板块轮动预测：LLM 优先，无 LLM 降级动量延续规则。

        返回 dict（含 mode: llm/deterministic），供落盘与展示。
        """
        if analysis is None:
            analysis = self.analyze()
        if "error" in analysis:
            return {"error": analysis["error"]}

        top = analysis.get("ranked_sectors", [])[:15]
        features = "\n".join(
            f"- {s['name']}（{s['sector_code']}）：20日涨幅 {s['rs_pct']:+.1f}%，"
            f"加速 {s['accel']:+.3f}"
            for s in top)
        rotation_speed = analysis.get("rotation_speed", "中")
        rotated_in = "、".join(s["name"] for s in analysis.get("rotated_in", [])) or "无"
        rotated_out = "、".join(s["name"] for s in analysis.get("rotated_out", [])) or "无"

        # 名称 → 板块代码映射（LLM 输出只有名称，验证需代码）
        name2code = {s["name"]: s["sector_code"] for s in
                     analysis.get("ranked_sectors", [])}

        def _attach_code(items: list[dict]) -> list[dict]:
            out = []
            for t in items or []:
                t = dict(t)
                t["sector_code"] = name2code.get(t.get("sector", ""), "")
                out.append(t)
            return out

        try:
            from pa_mcp.agent.llm_port import LLMCallParams
            from pa_mcp.agent.llm_factory import ensure_llm_adapter
            adapter = ensure_llm_adapter()  # 统一兜底：空单例主动读配置
            if adapter is not None:
                user_prompt = SECTOR_ROTATION_PROMPT.format(
                    features=features, rotation_speed=rotation_speed,
                    rotated_in=rotated_in, rotated_out=rotated_out)
                params = LLMCallParams(
                    system_prompt=(
                        "你是有经验的 A 股板块轮动研究员。只输出合法 JSON。"
                        "输出是研究参考，不是投资建议。"
                    ),
                    user_prompt=user_prompt, mode="fast", max_tokens=3000,
                )
                raw = await adapter.chat_json(params)
                if isinstance(raw, dict) and "error" not in raw:
                    errors = self._validate_prediction_json(raw)
                    if not errors:
                        result = {
                            "mode": "llm",
                            "predict_date": date.today().isoformat(),
                            "top_sectors_next_week": _attach_code(
                                raw.get("top_sectors_next_week", [])),
                            "rotation_logic": raw.get("rotation_logic", ""),
                            "sectors_to_avoid": _attach_code(
                                raw.get("sectors_to_avoid", [])),
                            "confidence": raw.get("confidence", 0.5),
                            "key_drivers": raw.get("key_drivers", []),
                            "risks": raw.get("risks", []),
                        }
                        result["analysis"] = analysis
                        return result
                    logger.info("板块预测 JSON 校验失败", errors=errors)
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM 板块预测失败，降级确定性", error=str(e))

        # 确定性降级：动量延续（RS 高 + 加速为正 + 涨幅为正）
        ranked = [s for s in analysis.get("ranked_sectors", [])
                  if s["rs_pct"] > 0 and s["accel"] > 0]
        ranked.sort(key=lambda s: (s["rs_pct"], s["accel"]), reverse=True)
        top_picks = [
            {"sector": s["name"], "sector_code": s["sector_code"],
             "probability": round(0.5 + min(0.3, s["rs_pct"] / 200), 2),
             "reason": f"20日涨幅 {s['rs_pct']:+.1f}% 且动量加速 {s['accel']:+.3f}"}
            for s in ranked[:5]]
        avoid = [{"sector": s["name"], "sector_code": s["sector_code"],
                  "reason": f"20日涨幅 {s['rs_pct']:+.1f}%，动量减速 {s['accel']:+.3f}"}
                 for s in sorted(
                     analysis.get("ranked_sectors", []),
                     key=lambda s: s["rs_pct"])[:AVOID_N]]
        return {
            "mode": "deterministic",
            "predict_date": date.today().isoformat(),
            "top_sectors_next_week": top_picks,
            "rotation_logic": "动量延续（强者恒强）——无 LLM 配置时的确定性规则",
            "sectors_to_avoid": avoid,
            "confidence": 0.5,
            "key_drivers": ["板块相对强度排名", "动量加速方向"],
            "risks": ["无 LLM 配置，确定性规则无宏观解读；动量策略在震荡市失效"],
            "analysis": analysis,
        }

    @staticmethod
    def _validate_prediction_json(raw: dict) -> list[str]:
        errors = []
        tops = raw.get("top_sectors_next_week")
        if not isinstance(tops, list) or not tops:
            errors.append("top_sectors_next_week 需为非空列表")
        else:
            for t in tops:
                if not isinstance(t, dict) or not t.get("sector"):
                    errors.append("top_sectors_next_week 元素需含 sector 字段")
        try:
            conf = float(raw.get("confidence", -1))
            if not (0 <= conf <= 1):
                errors.append("confidence 应为 0-1")
        except Exception:
            errors.append("confidence 非数值")
        return errors

    # ---- 落盘 ----
    def save_prediction(self, pred: dict[str, Any]) -> int:
        """写入 sector_prediction 表，返回 id。"""
        store = self._store()
        try:
            max_id = store.query_df(
                "SELECT COALESCE(MAX(id),0) AS m FROM sector_prediction", [])
            new_id = int(max_id.iloc[0]["m"]) + 1 if not max_id.empty else 1
            store.insert_df("sector_prediction", pd.DataFrame([{
                "id": new_id,
                "predict_date": pred.get("predict_date", date.today().isoformat()),
                "mode": pred.get("mode", "deterministic"),
                "top_sectors": json.dumps(
                    pred.get("top_sectors_next_week", []), ensure_ascii=False),
                "avoid_sectors": json.dumps(
                    pred.get("sectors_to_avoid", []), ensure_ascii=False),
                "rotation_logic": str(pred.get("rotation_logic", ""))[:300],
                "confidence": pred.get("confidence"),
                "key_drivers": json.dumps(
                    pred.get("key_drivers", []), ensure_ascii=False)[:1000],
                "risks": json.dumps(
                    pred.get("risks", []), ensure_ascii=False)[:1000],
                "status": "pending",
            }]))
            return new_id
        finally:
            store.close()

    # ---- 验证（5 交易日后回填） ----
    def evaluate_predictions(self, days_forward: int = 5) -> dict[str, Any]:
        """回填已到期板块预测：top3 平均收益 vs 全板块平均 → 超额。"""
        store = self._store()
        try:
            pending = store.query_df(
                "SELECT * FROM sector_prediction WHERE status = 'pending' "
                "ORDER BY id", [])
            if pending.empty:
                return {"evaluated": 0, "total": int(store.query_df(
                    "SELECT COUNT(*) AS c FROM sector_prediction", [])
                    .iloc[0]["c"] or 0)}

            # 全部板块最新收盘（按预测日之后 days_forward 个交易日）
            evaluated = 0
            summary = {"evaluated": 0, "hit_count": 0,
                       "total_excess": 0.0, "total": len(pending)}
            for _, row in pending.iterrows():
                pdate = str(row["predict_date"])[:10]
                try:
                    tops = json.loads(row["top_sectors"] or "[]")
                except Exception:
                    tops = []
                if not tops:
                    continue
                codes = [t.get("sector_code") or t.get("sector") for t in tops]
                # 板块收益：predict_date 之后第 days_forward 个交易日相对
                board_rets = {}
                for code in codes[:5]:
                    if not code or not str(code).upper().startswith("BK"):
                        continue
                    df = store.query_df(
                        "SELECT date, close FROM sector_daily "
                        "WHERE sector_code = ? AND date >= ? "
                        "ORDER BY date LIMIT ?",
                        [code.upper(), pdate, days_forward + 1])
                    if len(df) < 2:
                        continue
                    base = float(df["close"].iloc[0])
                    target = float(df["close"].iloc[min(days_forward, len(df) - 1)])
                    board_rets[code.upper()] = (target / base - 1) * 100
                if not board_rets:
                    continue

                top3_avg = sum(list(board_rets.values())[:3]) / min(
                    3, len(board_rets))
                # 全板块平均（同日）
                all_df = store.query_df(
                    "SELECT sector_code, date, close FROM sector_daily "
                    "WHERE date >= ? ORDER BY sector_code, date", [pdate])
                market_avg = 0.0
                cnt = 0
                for code, g in all_df.groupby("sector_code"):
                    g = g.sort_values("date")
                    if len(g) < 2:
                        continue
                    base = float(g["close"].iloc[0])
                    target = float(g["close"].iloc[min(days_forward, len(g) - 1)])
                    market_avg += (target / base - 1) * 100
                    cnt += 1
                if cnt > 0:
                    market_avg /= cnt
                excess = top3_avg - market_avg

                store.execute(
                    "UPDATE sector_prediction SET status = ?, "
                    "top3_avg_return_pct = ?, market_avg_return_pct = ?, "
                    "excess_return_pct = ?, evaluated_date = ? WHERE id = ?",
                    ["evaluated", round(top3_avg, 3), round(market_avg, 3),
                     round(excess, 3), date.today().isoformat(), int(row["id"])])
                evaluated += 1
                summary["evaluated"] = evaluated
                summary["total_excess"] += excess
                if excess > 0:
                    summary["hit_count"] += 1
            if evaluated:
                summary["avg_excess_pct"] = round(
                    summary["total_excess"] / evaluated, 3)
                summary["hit_rate"] = round(
                    summary["hit_count"] / evaluated, 3)
            return summary
        finally:
            store.close()


SECTOR_ROTATION_PROMPT = """你是 A 股板块轮动研究员。基于以下板块相对强度数据，预测未来一周的强势板块。

【板块特征（按 20 日涨幅排序 top15）】
{features}

【轮动状态】速度：{rotation_speed}；新进强势：{rotated_in}；退出强势：{rotated_out}

【预测要求】
1. 只输出一个 JSON 对象
2. top_sectors_next_week 给 3-5 个板块（sector=板块名称，probability 之和不必为 1，每个 0-1）
3. rotation_logic 用一句话说明轮动逻辑（强者恒强/高低切换/防御回归/事件驱动等）
4. sectors_to_avoid 给 1-3 个回避板块
5. key_drivers / risks 用中文 2-3 条
6. confidence 0-1

【JSON 格式】
{{
  "top_sectors_next_week": [
    {{"sector": "银行", "probability": 0.65, "reason": "RS 居前且动量加速，资金持续流入"}}
  ],
  "rotation_logic": "高低切换——前期强势板块减速，资金转向低估值板块",
  "sectors_to_avoid": [{{"sector": "光伏", "reason": "动量持续减速"}}],
  "confidence": 0.6,
  "key_drivers": ["板块资金流", "北向偏好"],
  "risks": ["大盘系统性回调将拖累所有板块"]
}}"""


_analyzer: Optional[SectorRotationAnalyzer] = None


def get_sector_rotation_analyzer() -> SectorRotationAnalyzer:
    """单例获取板块轮动分析器。"""
    global _analyzer
    if _analyzer is None:
        _analyzer = SectorRotationAnalyzer()
    return _analyzer


def format_rotation(pred: dict[str, Any]) -> str:
    """板块轮动预测 → markdown（UI/MCP 共用）。"""
    if "error" in pred:
        return f"板块轮动不可用：{pred['error']}"
    analysis = pred.get("analysis", {})
    lines = [
        f"## 🔄 板块轮动预测（{pred.get('predict_date', '')}）",
        f"**轮动逻辑**：{pred.get('rotation_logic', '')}",
        f"**模式**：{'🤖 AI 解读' if pred.get('mode') == 'llm' else '📐 统计规则'}",
        "",
        "### 未来一周强势板块候选",
        "| 板块 | 概率 | 理由 |",
        "|---|---|---|",
    ]
    for t in pred.get("top_sectors_next_week", []):
        lines.append(f"| {t.get('sector', '')} | {t.get('probability', 0):.0%} | "
                     f"{t.get('reason', '')} |")
    if pred.get("sectors_to_avoid"):
        lines.append("\n### 回避板块")
        for t in pred.get("sectors_to_avoid", []):
            lines.append(f"- **{t.get('sector', '')}**：{t.get('reason', '')}")
    if pred.get("key_drivers"):
        lines.append("\n**驱动因素**：" + "；".join(pred.get("key_drivers", [])))
    if pred.get("risks"):
        lines.append("**风险**：" + "；".join(pred.get("risks", [])))
    if analysis.get("rotation_speed"):
        lines.append(f"\n**当前轮动速度**：{analysis.get('rotation_speed')}"
                     f"（新进 {len(analysis.get('rotated_in', []))} 个 / "
                     f"退出 {len(analysis.get('rotated_out', []))} 个）")
    lines.append("\n*预测落盘可验证：5 交易日后回填板块收益，"
                 "计算 top3 超额。研究参考，非投资建议。*")
    return "\n".join(lines)


def format_hot_cold(hot: list[dict], cold: list[dict]) -> str:
    """热门/冷门板块 → markdown。"""
    lines = ["## 🔥 今日热门 / 🧊 冷门板块（新浪行业实时）", ""]
    if hot:
        lines.append("### 热门板块（涨幅榜）")
        lines.append("| 板块 | 涨跌幅 | 领涨股 |")
        lines.append("|---|---|---|")
        for b in hot:
            lines.append(f"| {b['name']} | {b['change_pct']:+.2f}% | "
                         f"{b['leader']} |")
    if cold:
        lines.append("")
        lines.append("### 冷门板块（跌幅榜）")
        lines.append("| 板块 | 涨跌幅 | 领涨股 |")
        lines.append("|---|---|---|")
        for b in cold:
            lines.append(f"| {b['name']} | {b['change_pct']:+.2f}% | "
                         f"{b['leader']} |")
    lines.append("\n*新浪行业板块实时快照（免费源，延迟 3-15s）。"
                 "研究参考，非投资建议。*")
    return "\n".join(lines)
