"""PA_MCP - 可视化图表输出（OpenClaw → QQ Bot 推送）。

复用/适配 `pa_mcp.ui.gradio_app` 已有的 plotly figure 构建逻辑，
导出 PNG 落到 `~/.openclaw/media/qqbot/pamcp/charts/` 下，OpenClaw
通过 `<qqmedia>绝对路径</qqmedia>` 标签推给 QQ Bot 即可（图片扩展名
自动被 qqbot-media 通道识别为图片）。
"""