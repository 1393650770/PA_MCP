# [AI:BEGIN]
# PA_MCP - Logging Setup（MCP stdio 协议纯净性守护）
#
# 问题背景：MCP stdio 传输要求进程 stdout 只输出 JSON-RPC 消息。
# structlog 默认（未 configure）输出到 sys.stdout → 任何日志行都会
# 污染协议流，客户端（OpenClaw/Claude）解析失败报
# "Invalid JSON: trailing characters"，表现为"MCP 经常报错"。
#
# 本模块把全部结构化日志重定向到 stderr（协议外通道），并统一
# 格式/级别。所有进程入口（server / ui / scheduler）调用一次即可。
# [AI:END]

from __future__ import annotations

import logging
import sys

import structlog

_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """初始化结构化日志（幂等）。

    关键：所有日志输出到 stderr——stdout 留给 MCP JSON-RPC。
    级别默认 INFO（过滤建表刷屏等 debug 日志）。
    """
    global _configured
    if _configured:
        return
    _configured = True

    # stdlib logging：根 logger 兜底也走 stderr（防第三方库污染 stdout）
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
