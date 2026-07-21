$ErrorActionPreference = "Continue"

Write-Host "=== Docker ==="
docker ps --filter "name=ruview" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

Write-Host "`n=== RuView health ==="
try {
    Invoke-RestMethod "http://localhost:3000/health" -TimeoutSec 3 |
        ConvertTo-Json -Depth 6
}
catch {
    Write-Host "RuView health endpoint unavailable."
}

Write-Host "`n=== Nodes ==="
try {
    Invoke-RestMethod "http://localhost:3000/api/v1/nodes" -TimeoutSec 3 |
        ConvertTo-Json -Depth 8
}
catch {
    Write-Host "Node endpoint unavailable."
}

Write-Host "`n=== Zone dashboard ==="
try {
    Invoke-RestMethod "http://127.0.0.1:8770/health" -TimeoutSec 3 |
        ConvertTo-Json -Depth 6
}
catch {
    Write-Host "Zone dashboard is not running."
}
