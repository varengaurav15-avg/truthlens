# ============================================================================
#  TruthLens — start backend + frontend  (Windows PowerShell)
#
#      powershell -ExecutionPolicy Bypass -File .\run.ps1
#
#  Opens the FastAPI backend in one window and the Vite dev server in another.
#  Run .\setup.ps1 first if you haven't already.
# ============================================================================

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

$venvPy = Join-Path $root "venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "venv not found. Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}

# --- sanity check: are the heavy deps actually installed? -------------------
& $venvPy -c "import torch, transformers, fastapi, uvicorn, cv2, timm" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Dependencies are missing. Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "Starting backend on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root'; & '$venvPy' -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload"
)

# --- wait for /health before launching the UI -------------------------------
Write-Host "Waiting for model warmup (can take 1-3 minutes on first run)..." -ForegroundColor Yellow
$ready = $false
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
    Write-Host "." -NoNewline
}
Write-Host ""

if ($ready) {
    Write-Host "Backend is up." -ForegroundColor Green
} else {
    Write-Host "Backend did not respond in time — check its window for errors." -ForegroundColor Yellow
}

Write-Host "Starting frontend on http://localhost:5173 ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root\frontend-app'; npm run dev"
)

Start-Sleep -Seconds 4
Start-Process "http://localhost:5173"

Write-Host "`nBoth services launched." -ForegroundColor Green
Write-Host "  Backend  http://127.0.0.1:8000/docs"
Write-Host "  Frontend http://localhost:5173"
