# [AI:BEGIN]
# PA_MCP - UI 按钮处理函数验证（58 个）
#
# 枚举 gradio_app 全部 .click/.submit 处理函数 → 最小参数逐个调用
# （种子库 + LLM/网络隔离）→ 分类 ✅ / ⚠️ 降级 / ❌ 崩溃。
# 运行：venv\\Scripts\\python.exe scripts/verify_ui.py [--llm]
# [AI:END]

from __future__ import annotations

import asyncio
import inspect
import io
import os
import re
import sys
import tempfile
from typing import Any, Optional, get_args, get_origin

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import verify_interfaces as vseed  # 复用种子库

# 处理函数清单（从 gradio_app 提取）
SRC = os.path.join(os.path.dirname(__file__), "..", "src", "pa_mcp",
                   "ui", "gradio_app.py")
text = open(SRC, encoding="utf-8").read()
HANDLERS = sorted(set(re.findall(
    r'\.(?:click|submit)\(\s*([a-zA-Z_]\w*)', text)))
# 过滤 lambda/gr 等
HANDLERS = [h for h in HANDLERS if not h.startswith(("gr.", "lambda"))]


def _gen_args(fn) -> list[Any]:
    """生成最小参数（兼容 from __future__ import annotations 的字符串注解）。"""
    sig = inspect.signature(fn)
    args = []
    for name, p in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if p.default is not inspect.Parameter.empty:
            # 默认 None → 按名字推断（UI 函数常见 symbol: str = None 之类）
            if p.default is None:
                args.append(_guess(name))
            else:
                args.append(p.default)
            continue
        args.append(_guess(name))
    return args


def _guess(name: str) -> Any:
    """按参数名推断最小合法值（复数 symbols 优先于单数 symbol）。"""
    if "symbols" in name or "pool" in name.lower() or "codes" in name:
        return "000001,000002,000003,000004,000005,000006"
    if "date" in name.lower():
        return "2026-08-14"
    if "symbol" in name or name in ("code",):
        return "000001"
    if "strategy" in name.lower():
        return "bollinger_mean_reversion"
    if "horizon" in name.lower():
        return "5d"
    if "days" in name.lower() or "size" in name.lower() \
            or "top" in name.lower() or "n_" in name.lower() \
            or "period" in name.lower() or "window" in name.lower():
        return 100 if "days" in name or "size" in name else 10
    if "cash" in name.lower() or "value" in name.lower() \
            or "weight" in name.lower() or "pct" in name.lower():
        return 100000.0
    if name in ("what",):
        return "selection"
    if name in ("depth",):
        return "fast"
    if name in ("load_data",) or "enable" in name or "llm" in name \
            or "use_" in name or "force" in name:
        return False
    return "x"


async def main() -> None:
    tmp = tempfile.mkdtemp(prefix="pa_mcp_verify_ui_")
    db = vseed._seed(tmp)

    import pa_mcp.config as cfg
    real = cfg.get_settings()
    d = real.model_dump()
    d["database"] = {"path": db, "read_only": False}
    cfg._settings = type(real)(**d)

    import pa_mcp.agent.llm_port as lp
    import pa_mcp.agent.llm_factory as lf
    import pa_mcp.agent.llm_client as lc
    if "--llm" in sys.argv:
        lf.init_llm_adapter("config/llm_config.json")
        print("LLM 模式（真实）")
    else:
        lp._adapter = None
        lf.init_llm_adapter = lambda *a, **k: None
        lc._client = None

    import pa_mcp.ui.gradio_app as g

    import concurrent.futures as cf

    print(f"=== UI 处理函数验证（{len(HANDLERS)} 个） ===")
    ok, warn, fail, missing = [], [], [], []
    for name in HANDLERS:
        fn = getattr(g, name, None)
        if fn is None or not callable(fn):
            missing.append(name)
            continue
        try:
            args = _gen_args(fn)
            # UI 函数内部 asyncio.run 需要无运行 loop 的线程
            # （真实 Gradio 线程环境同此；主线程 async 会嵌套崩）
            with cf.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(fn, *args)
                result = fut.result(timeout=45)
            if inspect.isawaitable(result):
                result = asyncio.run(result)
            # 元组结果（fig, text）
            if isinstance(result, tuple):
                result = result[1] if len(result) > 1 else ""
            if isinstance(result, str):
                # 仅当输出本身是失败/错误消息才降级（正常报告含
                # "无数据"段/表头"失败"列不算故障）
                head = result.strip()[:40]
                if head.startswith(("失败", "错误", "不可用", "无行情数据",
                                    "无数据", "未注册", "请先", "请输入")):
                    warn.append((name, f"⚠️ {result[:50]}"))
                elif result.strip():
                    ok.append((name, "✅"))
                else:
                    ok.append((name, "✅（空输出）"))
            else:
                ok.append((name, "✅"))
        except Exception as e:  # noqa: BLE001
            fail.append((name, f"❌ {type(e).__name__}: {str(e)[:70]}"))

    for name, st in ok:
        print(f"  {st} {name}")
    for name, st in warn:
        print(f"  {st} {name}")
    for name, st in fail:
        print(f"  {st} {name}")
    for name in missing:
        print(f"  ❌ 未找到函数 {name}")
    print(f"\n=== 汇总：✅ {len(ok)} / ⚠️ {len(warn)} / ❌ {len(fail)}"
          f" / 缺失 {len(missing)}（共 {len(HANDLERS)}） ===")
    sys.exit(1 if fail or missing else 0)


if __name__ == "__main__":
    asyncio.run(main())
