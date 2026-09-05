# PA_MCP - DuckDB 排他锁与并发回归测试
#
# 背景（本次修复的线上事故）：OpenClaw 两个 cron 任务同一分钟触发，各起一个
# pa_mcp MCP Server 进程；DuckDB 单文件是**进程级排他锁**，后启动者在
# server_lifespan 里打开数据库时直接崩溃，导致该会话的 81 个工具全部不注册，
# 模型看不到工具就用 exec 写脚本兜底，把整轮任务搞成 error。
#
# 修复要点：
# 1. DuckDBStore 不在构造时连接（延迟到使用点），服务端启动不再因锁失败；
# 2. connect() 遇到锁冲突时排队重试（database.connect_timeout_seconds）；
# 3. 重试耗尽后抛可行动的 RuntimeError，而不是让进程崩溃；
# 4. 显式 read_only 配置生效，且只读模式下写操作给出明确报错。

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import duckdb
import pytest

from pa_mcp.data.store import DuckDBStore


# ---- 子进程脚本：模拟"另一个进程持有 DuckDB 锁" ----

_CHILD_WAIT_AND_QUERY = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {src!r})
    from pa_mcp.data.store import DuckDBStore

    path = sys.argv[1]
    start = time.monotonic()
    try:
        store = DuckDBStore(path)
        rows = store.query_df("SELECT count(*) AS c FROM probe")["c"].tolist()
        print("OK rows=%s waited=%.1f read_only=%s"
              % (rows, time.monotonic() - start, store.read_only))
        store.close()
    except Exception as e:
        print("FAIL %s: %s" % (type(e).__name__, str(e)[:90]))
    """
)


def _run_child(db_path: str, src_dir: str, timeout_seconds: str = "10") -> str:
    """在独立进程里用 DuckDBStore 连接同一个库，返回子进程输出。"""
    env = dict(os.environ)
    env["PA_MCP_DATABASE__CONNECT_TIMEOUT_SECONDS"] = timeout_seconds
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _CHILD_WAIT_AND_QUERY.format(src=src_dir),
            db_path,
        ],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    lines = [
        ln
        for ln in (proc.stdout or "").splitlines()
        if ln.startswith(("OK ", "FAIL "))
    ]
    return lines[-1] if lines else (proc.stdout or proc.stderr or "").strip()[-300:]


@pytest.fixture
def src_dir() -> str:
    """仓库 src 目录（供子进程 import pa_mcp）。"""
    return str(Path(__file__).resolve().parent.parent / "src")


@pytest.fixture
def locked_db():
    """创建一个带 probe 表的临时库，并由本进程持有读写锁。"""
    tmp = tempfile.mkdtemp(prefix="ducklock_test_")
    db_path = str(Path(tmp) / "probe.duckdb")
    con = duckdb.connect(db_path)
    con.execute("CREATE TABLE probe (a INTEGER)")
    con.execute("INSERT INTO probe VALUES (1), (2), (3)")
    yield db_path
    try:
        con.close()
    except Exception:
        pass


class TestStoreLazyConnect:
    """延迟连接：构造 DuckDBStore 不得触碰数据库文件。"""

    def test_constructor_does_not_open_db(self, locked_db):
        """构造不得打开数据库——否则 MCP Server 会在启动阶段因锁失败。"""
        store = DuckDBStore(locked_db)
        assert store._conn is None
        store.close()

    def test_missing_dir_is_created_only_on_connect(self):
        tmp = tempfile.mkdtemp(prefix="ducklazy_")
        db_path = str(Path(tmp) / "nested" / "sub" / "x.duckdb")
        store = DuckDBStore(db_path)
        assert store._conn is None
        store.connect()
        assert store._conn is not None
        assert Path(db_path).exists()
        store.close()


class TestLockContention:
    """锁冲突：排队重试 + 超时后干净报错。"""

    def test_queues_until_lock_released(self, src_dir):
        """持锁 5s 后释放：子进程应排队等到锁并成功读取（而不是直接失败）。"""
        import threading
        import time

        tmp = tempfile.mkdtemp(prefix="duckqueue_")
        db_path = str(Path(tmp) / "queue.duckdb")
        holder = duckdb.connect(db_path)
        holder.execute("CREATE TABLE probe (a INTEGER)")
        holder.execute("INSERT INTO probe VALUES (7), (8)")

        result: dict[str, str] = {}

        def worker():
            result["out"] = _run_child(db_path, src_dir, timeout_seconds="25")

        t = threading.Thread(target=worker)
        t.start()
        time.sleep(5)
        holder.close()  # 释放锁，子进程下一次重试即可拿到
        t.join(timeout=120)

        out = result.get("out", "")
        assert out.startswith("OK "), out
        assert "rows=[2]" in out, out
        assert "waited=" in out, out

    def test_reports_actionable_error_when_lock_never_released(
        self, locked_db, src_dir,
    ):
        """锁长期持有：应在预算内重试后抛 RuntimeError，不崩溃、不挂死。"""
        out = _run_child(locked_db, src_dir, timeout_seconds="6")
        assert out.startswith("FAIL RuntimeError"), out
        assert "排他锁" in out or "占用" in out, out

    def test_connects_immediately_when_unlocked(self, src_dir):
        """无竞争时零等待。"""
        tmp = tempfile.mkdtemp(prefix="duckfree_")
        db_path = str(Path(tmp) / "free.duckdb")
        con = duckdb.connect(db_path)
        con.execute("CREATE TABLE probe (a INTEGER)")
        con.close()

        out = _run_child(db_path, src_dir, timeout_seconds="5")
        assert out.startswith("OK "), out
        assert "waited=0.0" in out, out


class TestReadOnlyMode:
    """只读模式：配置生效且写操作被明确拒绝。"""

    def test_read_only_config_is_honored(self):
        tmp = tempfile.mkdtemp(prefix="duckro_")
        db_path = str(Path(tmp) / "ro.duckdb")
        seed = DuckDBStore(db_path)
        seed.connect()
        seed.query_df("SELECT 1")
        seed.close()

        store = DuckDBStore(db_path, read_only=True)
        store.connect()
        assert store.read_only is True
        assert store.is_writable() is False
        assert store.query_df("SELECT 1 AS x")["x"].tolist() == [1]
        store.close()

    def test_write_raises_clear_error_in_read_only(self):
        import pandas as pd

        tmp = tempfile.mkdtemp(prefix="duckro2_")
        db_path = str(Path(tmp) / "ro2.duckdb")
        seed = DuckDBStore(db_path)
        seed.connect()
        seed.close()

        store = DuckDBStore(db_path, read_only=True)
        store.connect()
        df = pd.DataFrame([{
            "symbol": "000001", "date": "2026-09-06",
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
            "volume": 1.0, "amount": 1.0, "amplitude": 0.0,
            "pct_change": 0.0, "change": 0.0, "turnover": 0.0,
            "adjust_factor": 1.0,
        }])
        with pytest.raises(RuntimeError, match="只读"):
            store.insert_df("kline_daily", df)
        store.close()

    def test_writable_store_still_works(self):
        """正常读写路径不受影响。"""
        tmp = tempfile.mkdtemp(prefix="duckrw_")
        store = DuckDBStore(str(Path(tmp) / "rw.duckdb"))
        store.connect()
        assert store.is_writable() is True
        assert store.read_only is False
        assert store.table_exists("kline_daily")
        store.close()


class TestLockErrorClassification:
    """锁冲突错误的识别（中英文 Windows / POSIX 文案）。"""

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("IO Error: Cannot open file: 另一个程序正在使用此文件", True),
            ("IO Error: The file is being used by another process", True),
            ("File is already open in C:\\Python314\\python.exe (PID 1)", True),
            ("IO Error: Could not set lock on file", True),
            ("duckdb.IOException: database is already open", True),
            ("Binder Error: Unknown column", False),
            ("Catalog Error: Table does not exist", False),
        ],
    )
    def test_is_lock_contention(self, message, expected):
        assert DuckDBStore._is_lock_contention(Exception(message)) is expected

    def test_open_modes_rw_then_ro(self):
        store = DuckDBStore("unused.duckdb", read_only=False)
        assert store._open_modes() == [False, True]

    def test_open_modes_ro_only(self):
        """只读模式下不再尝试读写，避免抢锁。"""
        store = DuckDBStore("unused.duckdb", read_only=True)
        assert store._open_modes() == [True]
