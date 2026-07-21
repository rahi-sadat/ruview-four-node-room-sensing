param([int]$Port = 8770)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-zone\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Run install-zone-dashboard.ps1 first."
}

Set-Location $ProjectRoot

try {
    $health = Invoke-RestMethod "http://localhost:3000/health" -TimeoutSec 3
    Write-Host "RuView health: $($health.status)"
}
catch {
    Write-Warning "RuView API is not responding at http://localhost:3000 yet."
}

Write-Host "Dashboard: http://127.0.0.1:$Port"
Write-Host "Press Ctrl+C to stop."
Start-Process "http://127.0.0.1:$Port"

& $Python -m uvicorn "app:app" `
    --app-dir (Join-Path $ProjectRoot "zone-dashboard") `
    --host 127.0.0.1 `
    --port $Port
