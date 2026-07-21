$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $ProjectRoot ".venv-zone"
$Python = Join-Path $Venv "Scripts\python.exe"

Set-Location $ProjectRoot

if (-not (Test-Path $Python)) {
    py -3 -m venv $Venv
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")

Write-Host ""
Write-Host "Installation complete."
Write-Host "Run:"
Write-Host "powershell -ExecutionPolicy Bypass -File .\zone-dashboard\run-zone-dashboard.ps1"
