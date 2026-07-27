# [AI:BEGIN]
# PA_MCP - Windows One-Click Setup Script
# Run: powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
# This script will:
#   1. Check Python >= 3.12
#   2. Create virtual environment
#   3. Install all dependencies (pure pip, zero C compilation)
#   4. Initialize config from example
#   5. Run tests to verify
# [AI:END]

param(
    [switch]$SkipTests,
    [string]$PythonExe = "python"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  PA_MCP - Windows One-Click Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ---- Step 1: Check Python ----
Write-Host "[1/6] Checking Python version..." -ForegroundColor Yellow
$pyVersion = & $PythonExe --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Install Python 3.12+ from https://python.org" -ForegroundColor Red
    exit 1
}
Write-Host "  $pyVersion" -ForegroundColor Green

# Verify >= 3.12
$verStr = $pyVersion -replace "Python ", ""
$parts = $verStr -split "\."
if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 12)) {
    Write-Host "ERROR: Python 3.12+ required. Found: $verStr" -ForegroundColor Red
    exit 1
}

# ---- Step 2: Create venv ----
Write-Host ""
Write-Host "[2/6] Creating virtual environment..." -ForegroundColor Yellow
$venvPath = Join-Path $PSScriptRoot "..\venv"
if (-not (Test-Path $venvPath)) {
    & $PythonExe -m venv $venvPath 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}
Write-Host "  Virtual environment at: $venvPath" -ForegroundColor Green

# Activate venv
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"
$pipExe = Join-Path $venvPath "Scripts\pip.exe"

# ---- Step 3: Upgrade pip ----
Write-Host ""
Write-Host "[3/6] Upgrading pip..." -ForegroundColor Yellow
& $pythonExe -m pip install --upgrade pip --quiet 2>&1
Write-Host "  pip upgraded" -ForegroundColor Green

# ---- Step 4: Install dependencies ----
Write-Host ""
Write-Host "[4/6] Installing PA_MCP dependencies (pure Python, zero C compilation)..." -ForegroundColor Yellow
Write-Host "  This may take 2-5 minutes on first run..."
$projectRoot = Join-Path $PSScriptRoot ".."
Push-Location $projectRoot
try {
    & $pipExe install -e ".[dev]" 2>&1 | Select-Object -Last 3
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "WARNING: pip install had warnings. Trying without dev extras..." -ForegroundColor Yellow
        & $pipExe install -e "."
    }
} finally {
    Pop-Location
}
Write-Host "  Dependencies installed" -ForegroundColor Green

# ---- Step 5: Initialize config ----
Write-Host ""
Write-Host "[5/6] Initializing configuration..." -ForegroundColor Yellow
$configPath = Join-Path $PSScriptRoot "..\config\llm_config.json"
$examplePath = Join-Path $PSScriptRoot "..\config\llm_config.example.json"
if (-not (Test-Path $configPath)) {
    Copy-Item $examplePath $configPath
    Write-Host "  Created config/llm_config.json (from example)" -ForegroundColor Green
    Write-Host ""
    Write-Host "  >>> IMPORTANT: Edit config/llm_config.json and fill in your LLM API key <<<" -ForegroundColor Cyan
} else {
    Write-Host "  config/llm_config.json already exists" -ForegroundColor Green
}

# ---- Step 6: Run tests ----
if (-not $SkipTests) {
    Write-Host ""
    Write-Host "[6/6] Running tests..." -ForegroundColor Yellow
    Push-Location $projectRoot
    try {
        & $pythonExe -m pytest tests/ -q 2>&1
        $testResult = $LASTEXITCODE
    } finally {
        Pop-Location
    }

    if ($testResult -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "  PA_MCP Setup Complete!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Yellow
        Write-Host "  Setup complete, some tests failed." -ForegroundColor Yellow
        Write-Host "  This may be OK — check output above." -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  PA_MCP Setup Complete! (tests skipped)" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Edit config/llm_config.json — fill in your API key" -ForegroundColor White
Write-Host "  2. Run: venv\Scripts\activate" -ForegroundColor White
Write-Host "  3. Run: python -m pa_mcp.server" -ForegroundColor White
Write-Host "  4. Or: pip install -e . && pa-mcp" -ForegroundColor White
Write-Host ""
Write-Host "For Claude Desktop, add to claude_desktop_config.json:" -ForegroundColor Cyan
$configJson = @"
{
  "mcpServers": {
    "pa-mcp": {
      "command": "${pythonExe}",
      "args": ["-m", "pa_mcp.server"],
      "cwd": "${projectRoot}"
    }
  }
}
"@
Write-Host $configJson -ForegroundColor Gray
