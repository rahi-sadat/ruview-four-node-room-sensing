param(
    [ValidateSet("live","replay")]
    [string]$Mode = "live",
    [int]$Port = 8766
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-gpu\Scripts\python.exe"
$Model = Join-Path $ProjectRoot "models\room-pose-gpu\best_pose_tcn.pt"
$Replay = Join-Path $ProjectRoot "models\room-pose-gpu\eval_predictions.jsonl"
$TrainingRecord = Join-Path $ProjectRoot "data\recordings\rec_1783755595.jsonl"

foreach ($required in @($Python, $Model, $TrainingRecord)) {
    if (-not (Test-Path $required)) { throw "Required file not found: $required" }
}

# Install only the small web-dashboard dependencies. Existing torch is preserved.
& $Python -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Dashboard dependency installation failed." }

$env:DASHBOARD_MODE = $Mode
$env:POSE_MODEL = $Model
$env:POSE_REPLAY_FILE = $Replay
$env:POSE_TRAINING_RECORD = $TrainingRecord
$env:RUVIEW_WS_URL = "ws://localhost:3000/ws/sensing"
$env:EXPECTED_SUBCARRIERS = "56"
$env:EXPECTED_NODE_COUNT = "4"
$env:WINDOW_FRAMES = "20"
$env:POSE_SMOOTHING = "0.85"

Write-Host "Mode: $Mode"
Write-Host "Training recording: $TrainingRecord"
Write-Host "Dashboard: http://127.0.0.1:$Port"

Push-Location $ProjectRoot
try {
    & $Python -m uvicorn "app:app" `
        --app-dir (Join-Path $ProjectRoot "custom-pose-dashboard") `
        --host 127.0.0.1 `
        --port $Port
}
finally {
    Pop-Location
}
