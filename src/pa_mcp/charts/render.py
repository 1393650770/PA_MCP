"""PNG 落盘与 OpenClaw 兼容输出。

输出目录优先级（满足 OpenClaw qqbot-media 安全策略）：
  1. 环境变量 PA_MCP_CHART_DIR 显式覆盖
  2. ~/.openclaw/media/qqbot/pamcp/charts/  ← OpenClaw 上传白名单根
  3. <PA_MCP_repo>/data/charts/                 ← 兜底（脱离 OpenClaw 也能用）

图片上限：单图 30MB（QQBot 限制）。文件名带时间戳 + 标题哈希，
避免 OpenClaw 重复推送时缓存命中旧图。
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
import structlog

logger = structlog.get_logger(__name__)

MAX_BYTES = 30 * 1024 * 1024     # 30MB，QQBot 图片上限
DEFAULT_RETENTION = 30           # 保留最近 N 张
WIDTH, HEIGHT, SCALE = 1280, 720, 2  # 默认 ~1600x900 高清

# 进程内常驻的 chrome 导出服务。plotly 6 + kaleido 1.x 的默认导出路径
# 每次 to_image 都会重新起停一个 chromium —— 实测单图 60~380s（并发抢
# 资源时更糟），足以拖垮定时任务。kaleido.start_sync_server() 起的是
# 单例服务，启动一次后后续出图回落到 0.1s 量级。
_ENGINE_LOCK = threading.Lock()
_ENGINE_READY = False


def _ensure_engine() -> None:
    """惰性启动 kaleido 常驻导出服务（幂等、线程安全）。"""
    global _ENGINE_READY
    if _ENGINE_READY:
        return
    with _ENGINE_LOCK:
        if _ENGINE_READY:
            return
        try:
            import kaleido
            kaleido.start_sync_server(silence_warnings=True)
            _ENGINE_READY = True
        except Exception as e:  # noqa: BLE001 - 起不来就退回默认逐次导出
            logger.warning("kaleido 常驻服务启动失败，回退默认导出",
                           error=str(e)[:200])
            _ENGINE_READY = True


def _candidate_dirs() -> list[Path]:
    candidates: list[Path] = []
    env = os.environ.get("PA_MCP_CHART_DIR")
    if env:
        candidates.append(Path(env))
    home = Path.home() / ".openclaw" / "media" / "qqbot" / "pamcp" / "charts"
    candidates.append(home)
    candidates.append(Path(__file__).resolve().parents[3] / "data" / "charts")
    return candidates


def _ensure_output_dir() -> Path:
    """第一个可创建/可写的目录。"""
    last_err: Optional[Exception] = None
    for d in _candidate_dirs():
        try:
            d.mkdir(parents=True, exist_ok=True)
            test = d / ".write_test"
            test.write_text("ok", encoding="utf-8")
            test.unlink(missing_ok=True)
            return d
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"无可用输出目录：{_candidate_dirs()} ({last_err})")


def _filename(prefix: str, title: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = hashlib.md5(title.encode("utf-8")).hexdigest()[:6]
    return f"{prefix}_{stamp}_{h}.png"


def _cleanup(dir_: Path, retain: int = DEFAULT_RETENTION) -> int:
    """保留最新 retain 张，超出删除。返回删除数量。"""
    try:
        files = sorted(dir_.glob("*.png"), key=lambda p: p.stat().st_mtime,
                       reverse=True)
        removed = 0
        for f in files[retain:]:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
        return removed
    except Exception:
        return 0


def _export_png(fig: go.Figure, target: Path,
                width: int = WIDTH, height: int = HEIGHT,
                scale: int = SCALE) -> int:
    """调 kaleido 导出 PNG，返回字节数（复用常驻 chrome）。"""
    _ensure_engine()
    png = fig.to_image(format="png", width=width, height=height, scale=scale)
    if len(png) > MAX_BYTES:
        raise ValueError(
            f"PNG {len(png)/1024/1024:.1f}MB 超过 QQBot 单图 30MB 上限，"
            "可缩小 width/height/scale 或减少数据点")
    target.write_bytes(png)
    return len(png)


def render(fig: go.Figure, prefix: str, title: str,
           width: int = WIDTH, height: int = HEIGHT,
           scale: int = SCALE) -> dict:
    """导出图到磁盘，返回 OpenClaw/QQBot 推送所需元数据。

    Returns:
        {
          "format": "png",
          "path": 绝对路径（OpenClaw qqbot-media 白名单内）,
          "qqmedia": "<qqmedia>path</qqmedia>" 标签,
          "bytes": int, "width": int, "height": int,
          "dir": 落落盘根目录（首个可写候选）,
          "cleanup": 旧文件清理数,
        }
    """
    out_dir = _ensure_output_dir()
    fname = _filename(prefix, title)
    target = out_dir / fname

    t0 = time.monotonic()
    bytes_written = _export_png(fig, target, width=width, height=height,
                                scale=scale)
    cleanup = _cleanup(out_dir)
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    return {
        "format": "png",
        "path": str(target),
        "qqmedia": f"<qqmedia>{target}</qqmedia>",
        "bytes": bytes_written,
        "width": width * scale,
        "height": height * scale,
        "dir": str(out_dir),
        "cleanup": cleanup,
        "elapsed_ms": elapsed_ms,
    }