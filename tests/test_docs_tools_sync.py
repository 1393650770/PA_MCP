"""文档同步测试：docs/mcp-tools.md 必须与 server 源码里的工具定义一致。

背景：图表工具上线后文档停在 97 个（实际 116），落后 19 个没被发现。
本测试把「文档 = 工具清单」固化为回归项，新增工具时漏更新文档会直接红。

只做源码正则比对（不 import server、不碰 DB），保持轻量可移植。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "src" / "pa_mcp" / "server.py"
DOC = ROOT / "docs" / "mcp-tools.md"


def _defined_tools() -> set[str]:
    src = SERVER.read_text(encoding="utf-8")
    return set(re.findall(
        r"@mcp\.tool[^\n]*\n(?:\s*#[^\n]*\n)*\s*(?:async )?def ([a-z_0-9]+)", src))


def _doc_tools() -> set[str]:
    text = DOC.read_text(encoding="utf-8")
    return set(re.findall(r"^\| `([a-z_0-9]+)`", text, re.M))


def test_doc_covers_every_tool():
    defined, doc = _defined_tools(), _doc_tools()
    assert defined, "server.py 里没解析到任何 @mcp.tool —— 正则或文件路径变了"
    missing = sorted(defined - doc)
    assert not missing, (
        f"docs/mcp-tools.md 缺 {len(missing)} 个工具：{missing}；"
        "新增工具后请同步更新工具清单（含所属分类计数）")


def test_doc_has_no_stale_tool():
    defined, doc = _defined_tools(), _doc_tools()
    stale = sorted(doc - defined)
    assert not stale, f"文档里列了已不存在的工具：{stale}"


def test_doc_header_count_matches():
    defined, doc = _defined_tools(), _doc_tools()
    m = re.search(r"共 \*\*(\d+)\*\* 个 MCP 工具",
                  DOC.read_text(encoding="utf-8"))
    assert m, "文档头部缺少「共 N 个 MCP 工具」计数"
    assert int(m.group(1)) == len(defined), (
        f"文档头部写 {m.group(1)} 个，实际 {len(defined)} 个")


def test_section_counts_match_tables():
    """各章节标题的「（N）」必须等于该表实际行数。"""
    text = DOC.read_text(encoding="utf-8")
    mismatch = []
    for title, body in zip(*[iter(re.split(r"^## (.+)$", text, flags=re.M)[1:])] * 2):
        m = re.match(r"(.+?)（(\d+)）", title)
        if not m:
            continue
        actual = len(re.findall(r"^\| `", body, re.M))
        if int(m.group(2)) != actual:
            mismatch.append(f"{m.group(1)}: 标题 {m.group(2)} vs 实际 {actual}")
    assert not mismatch, "章节计数不符：" + "; ".join(mismatch)


def test_chart_tools_documented():
    """图表工具是定时任务的推送载体，漏一个就少一路推送。"""
    src = SERVER.read_text(encoding="utf-8")
    charts = set(re.findall(r"async def (chart_[a-z_0-9]+)", src))
    doc = _doc_tools()
    missing = sorted(charts - doc)
    assert charts, "没找到任何 chart_* 工具"
    assert not missing, f"图表工具未写入文档：{missing}"
