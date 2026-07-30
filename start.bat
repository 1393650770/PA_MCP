@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   PA_MCP - Personal Analyst MCP Server
echo ============================================
echo.

:: Activate venv
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv not found. Run setup first.
    pause
    exit /b 1
)

:: Set encoding
set PYTHONIOENCODING=utf-8

:: Choose mode
echo Select transport mode:
echo   1. stdio (default, for MCP clients like OpenClaw)
echo   2. http  (SSE server on port 8080)
echo.
set /p MODE="Enter choice [1/2]: "

if "%MODE%"=="2" (
    echo.
    echo [INFO] Starting PA_MCP in HTTP/SSE mode on port 8080...
    set PA_MCP_SERVER__TRANSPORT=http
    venv\Scripts\python -m pa_mcp.server
) else (
    echo.
    echo [INFO] Starting PA_MCP in stdio mode...
    set PA_MCP_SERVER__TRANSPORT=stdio
    venv\Scripts\python -m pa_mcp.server
)

pause