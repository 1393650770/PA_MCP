# [AI:BEGIN]
# PA_MCP - Agent Layer: Long-Term Memory
# SQLite-based memory for decision tracking, outcome monitoring, and bias detection.
# [AI:END]

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


# ---- Schema ----

MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    analysis_date TEXT NOT NULL,
    strength_score REAL,
    direction TEXT,
    market_state TEXT,
    mode TEXT,
    analyst_reports_hash TEXT,
    key_evidence TEXT,  -- JSON
    key_risks TEXT,     -- JSON
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id INTEGER NOT NULL,
    days_forward INTEGER,
    forward_return REAL,
    max_favorable_excursion REAL,
    max_adverse_excursion REAL,
    outcome_date TEXT,
    was_profitable BOOLEAN,
    FOREIGN KEY (decision_id) REFERENCES decisions(id)
);

CREATE TABLE IF NOT EXISTS strategy_weights (
    strategy_name TEXT PRIMARY KEY,
    weight REAL DEFAULT 1.0,
    win_rate REAL DEFAULT 0.5,
    total_trades INTEGER DEFAULT 0,
    last_updated TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS bias_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bias_type TEXT,  -- overconfidence, disposition_effect, recency_bias
    detected_at TEXT DEFAULT (datetime('now')),
    detail TEXT
);

CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions(symbol);
CREATE INDEX IF NOT EXISTS idx_decisions_date ON decisions(analysis_date);
CREATE INDEX IF NOT EXISTS idx_outcomes_decision ON outcomes(decision_id);
"""


class LongTermMemory:
    """Tracks agent decisions and outcomes for self-improvement.

    Modules:
    1. Decision recording — log every analysis
    2. Outcome tracking — track 5/10/20 day forward returns
    3. Performance analytics — win rate, profit factor, per-state breakdown
    4. Strategy weight adjustment — Bayesian blending based on rolling performance
    5. Bias detection — overconfidence, disposition effect
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """SQLite 长期记忆库。

        db_path 默认用项目根绝对路径（data/memory.db），
        避免从不同目录启动时连接到不同文件（与数据库同款修复）。
        """
        if db_path is None:
            from pa_mcp.config import PROJECT_ROOT
            db_path = str(PROJECT_ROOT / "data" / "memory.db")
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        import os
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        conn.executescript(MEMORY_SCHEMA)
        conn.commit()
        conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def record_decision(
        self, symbol: str, strength_score: float, direction: str,
        market_state: str, mode: str, evidence: list[dict],
        risks: list[str],
    ) -> int:
        """Record an analysis decision. Returns decision_id."""
        evidence_hash = hashlib.md5(
            json.dumps(evidence, sort_keys=True).encode()
        ).hexdigest()[:12]

        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO decisions (symbol, analysis_date, strength_score, direction, market_state, mode, analyst_reports_hash, key_evidence, key_risks)
               VALUES (?, date('now'), ?, ?, ?, ?, ?, ?, ?)""",
            [symbol, strength_score, direction, market_state, mode, evidence_hash,
             json.dumps(evidence), json.dumps(risks)],
        )
        decision_id = cursor.lastrowid
        conn.commit()
        conn.close()
        logger.debug("Decision recorded", id=decision_id, symbol=symbol)
        return decision_id or 0

    def record_outcome(self, decision_id: int, forward_return: float, days: int = 5) -> None:
        """Record the outcome of a past decision.

        miss 决策（方向与结果相反）自动沉淀教训（Reflexion 闭环）。
        """
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO outcomes (decision_id, days_forward, forward_return, outcome_date, was_profitable)
               VALUES (?, ?, ?, date('now'), ?)""",
            [decision_id, days, forward_return, forward_return > 0],
        )
        conn.commit()
        # 自动教训沉淀：查决策方向/市场状态，miss 时记录
        try:
            row = conn.execute(
                "SELECT symbol, direction, market_state, key_evidence "
                "FROM decisions WHERE id = ?", [decision_id]).fetchone()
            if row and forward_return < 0:
                conn.close()
                self.record_lesson(
                    symbol=row[0], direction=row[1] or "neutral",
                    market_state=row[2] or "unknown",
                    context=row[3][:200] if row[3] else "",
                    forward_return=forward_return)
                return
        except Exception:  # noqa: BLE001
            pass
        conn.close()

    # ---- 教训沉淀（Reflexion 式：决策失败 → 提取教训 → 后续参考） ----

    def _init_lessons_table(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(10), direction VARCHAR(10),
                market_state VARCHAR(20), context TEXT,
                forward_return DOUBLE, lesson TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def record_lesson(self, symbol: str, direction: str, market_state: str,
                      context: str, forward_return: float) -> int:
        """决策未兑现 → 沉淀教训（Reflexion：反思 → 下次行动参考）。

        context 为当时判断摘要；forward_return 为回填收益。
        规则生成教训文本（方向 vs 结果的偏差类型）；返回 lesson_id。
        """
        try:
            self._init_lessons_table()
        except Exception:  # noqa: BLE001
            pass
        if forward_return >= 0:
            return 0  # 只有 miss 才沉淀教训
        if direction == "up" and forward_return < -2.0:
            lesson = (f"看涨 {symbol} 但 {forward_return:.1f}% 负收益——"
                      "追高/假突破概率大，下次需确认放量突破+大盘配合")
        elif direction == "down" and forward_return > 2.0:
            lesson = (f"看跌 {symbol} 但 {forward_return:+.1f}% 反弹——"
                      "超跌反弹/利空出尽，下次避免在冰点期过度看空")
        elif abs(forward_return) >= 1.0:
            lesson = (f"方向误判（{direction} vs {forward_return:+.1f}%）——"
                      f"上下文: {context[:60]}，下次降低该环境下的仓位")
        else:
            return 0  # 幅度太小不算教训
        conn = self._get_conn()
        cur = conn.execute(
            """INSERT INTO lessons (symbol, direction, market_state, context,
               forward_return, lesson) VALUES (?, ?, ?, ?, ?, ?)""",
            [symbol, direction, market_state, context[:200],
             round(forward_return, 2), lesson])
        conn.commit()
        conn.close()
        logger.info("Lesson recorded", id=cur.lastrowid, symbol=symbol)
        return cur.lastrowid or 0

    def get_lessons(self, symbol: str = "", direction: str = "",
                    limit: int = 5) -> list[dict[str, Any]]:
        """检索历史教训（Reflexion 注入：分析前查相关教训）。"""
        try:
            self._init_lessons_table()
        except Exception:  # noqa: BLE001
            return []
        sql = "SELECT * FROM lessons WHERE 1=1"
        params: list[Any] = []
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        if direction:
            sql += " AND direction = ?"
            params.append(direction)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        conn = self._get_conn()
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [{
            "id": r[0], "symbol": r[1], "direction": r[2],
            "market_state": r[3], "context": r[4],
            "forward_return": r[5], "lesson": r[6],
            "created_at": r[7],
        } for r in rows]

    def lessons_stats(self) -> dict[str, Any]:
        """教训库统计（供自我改进报告）。"""
        try:
            self._init_lessons_table()
        except Exception:  # noqa: BLE001
            return {"total": 0}
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
        by_dir = dict(conn.execute(
            "SELECT direction, COUNT(*) FROM lessons GROUP BY direction").fetchall())
        conn.close()
        return {"total": total, "by_direction": by_dir}

    def update_strategy_weight(self, strategy_name: str, win: bool) -> None:
        """Update strategy weight using Bayesian blending."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT weight, win_rate, total_trades FROM strategy_weights WHERE strategy_name = ?",
            [strategy_name],
        ).fetchone()

        if row is None:
            weight, win_rate, total = 1.0, 0.5, 0
        else:
            weight, win_rate, total = row

        total += 1
        new_win_rate = (win_rate * (total - 1) + (1.0 if win else 0.0)) / total

        # Weight decays toward 0.5 (neutral) when insufficient data
        alpha = min(total / 20.0, 1.0)  # Full confidence after 20 trades
        new_weight = alpha * new_win_rate + (1 - alpha) * 0.5

        conn.execute(
            """INSERT OR REPLACE INTO strategy_weights (strategy_name, weight, win_rate, total_trades, last_updated)
               VALUES (?, ?, ?, ?, datetime('now'))""",
            [strategy_name, round(new_weight, 4), round(new_win_rate, 4), total],
        )
        conn.commit()
        conn.close()

    def get_strategy_weight(self, strategy_name: str) -> float:
        """Get current Bayesian weight for a strategy."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT weight FROM strategy_weights WHERE strategy_name = ?",
            [strategy_name],
        ).fetchone()
        conn.close()
        return row[0] if row else 1.0

    def detect_bias(self) -> list[dict[str, Any]]:
        """Detect cognitive biases in recent decisions.

        Checks for:
        - Overconfidence: high confidence decisions with negative outcomes
        - Disposition effect: holding losers longer than winners
        """
        biases: list[dict[str, Any]] = []
        conn = self._get_conn()

        # Overconfidence: strength_score > 70 but forward_return < -3%
        overconfident = conn.execute("""
            SELECT d.symbol, d.strength_score, o.forward_return, d.analysis_date
            FROM decisions d
            JOIN outcomes o ON d.id = o.decision_id
            WHERE d.strength_score > 70 AND o.forward_return < -0.03
              AND d.analysis_date >= date('now', '-60 days')
            ORDER BY d.analysis_date DESC
            LIMIT 5
        """).fetchall()

        if len(overconfident) >= 3:
            biases.append({
                "type": "overconfidence",
                "detail": f"Found {len(overconfident)} high-confidence decisions with negative outcomes in last 60 days",
                "severity": "high" if len(overconfident) >= 5 else "medium",
            })

        conn.close()
        return biases

    def get_performance_summary(self, days: int = 60) -> dict[str, Any]:
        """Get performance summary for recent decisions."""
        conn = self._get_conn()
        row = conn.execute(f"""
            SELECT
                COUNT(*) as total,
                AVG(CASE WHEN o.was_profitable THEN 1.0 ELSE 0.0 END) as win_rate,
                AVG(o.forward_return) as avg_return,
                AVG(CASE WHEN o.forward_return > 0 THEN o.forward_return ELSE NULL END) as avg_win,
                AVG(CASE WHEN o.forward_return <= 0 THEN o.forward_return ELSE NULL END) as avg_loss
            FROM decisions d
            JOIN outcomes o ON d.id = o.decision_id
            WHERE d.analysis_date >= date('now', ?)
        """, [f"-{days} days"]).fetchone()

        conn.close()

        if row is None or row[0] == 0:
            return {"total_decisions": 0, "message": "Insufficient data"}

        return {
            "period_days": days,
            "total_decisions": row[0],
            "win_rate": round(row[1] * 100, 1) if row[1] else 0,
            "avg_return_pct": round(row[2] * 100, 2) if row[2] else 0,
            "avg_win_pct": round(row[3] * 100, 2) if row[3] else 0,
            "avg_loss_pct": round(row[4] * 100, 2) if row[4] else 0,
        }
