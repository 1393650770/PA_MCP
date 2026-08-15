@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   PA_MCP 理财助手 - Web UI 一键启动
echo   http://127.0.0.1:7860
echo ============================================
echo.

:: 优先使用 venv，否则用系统 Python
set PYTHON=python
if exist "venv\Scripts\python.exe" set PYTHON=venv\Scripts\python.exe

:: 编码（避免 Windows 控制台中文乱码）
set PYTHONIOENCODING=utf-8

echo [INFO] 使用 %PYTHON%
echo [INFO] 正在启动 UI（首次约需 5-10 秒）...
echo [INFO] 启动后浏览器自动打开 http://127.0.0.1:7860
echo [INFO] 关闭本窗口即可停止服务
echo.

%PYTHON% -m pa_mcp.ui.gradio_app

echo.
echo [ERROR] 启动失败。请检查：
echo   1. 是否已安装依赖: pip install -e .
echo   2. 是否已配置 LLM: 复制 config\llm_config.example.json 为 config\llm_config.json 并填 key
echo   3. 完整报错信息见上方
pause
