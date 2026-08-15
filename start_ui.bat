@echo off
cd /d "%~dp0"

echo ============================================
echo   PA_MCP Wealth Assistant - Web UI Launcher
echo   http://127.0.0.1:7860
echo ============================================
echo.

set PYTHON=python
if exist "venv\Scripts\python.exe" set PYTHON=venv\Scripts\python.exe

:: Verify UI deps in chosen python; fall back to system python if missing
%PYTHON% -c "import gradio, plotly, duckdb" >nul 2>&1
if errorlevel 1 (
    if "%PYTHON%"=="venv\Scripts\python.exe" (
        echo [WARN] venv missing deps, falling back to system python.
        set PYTHON=python
    )
)

set PYTHONIOENCODING=utf-8

echo [INFO] Using: %PYTHON%
echo [INFO] Starting UI (first launch takes ~5-10s)...
echo [INFO] Browser will open at http://127.0.0.1:7860
echo [INFO] Close this window to stop the server.
echo.

%PYTHON% -m pa_mcp.ui.gradio_app

echo.
echo [ERROR] UI failed to start. Check:
echo   1. Dependencies installed:  pip install -e .
echo   2. LLM configured: copy config\llm_config.example.json config\llm_config.json
echo   3. Full error message above
pause
