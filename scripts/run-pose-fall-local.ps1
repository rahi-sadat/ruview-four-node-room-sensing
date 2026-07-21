[CmdletBinding()]
param(
  [string]$Image = 'ruview-local:pose-fall',
  [string]$Container = 'ruview',
  [int]$HttpPort = 3000,
  [int]$WsPort = 3001,
  [int]$UdpPort = 5005
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$DockerDataDir = Join-Path $RepoRoot 'data'
$DockerModelsDir = Join-Path $RepoRoot 'models'

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  throw 'Docker is required for this helper.'
}

foreach ($path in @($DockerDataDir, $DockerModelsDir)) {
  if (-not (Test-Path -LiteralPath $path)) {
    New-Item -ItemType Directory -Path $path | Out-Null
  }
}

$RuFieldCore = Join-Path $RepoRoot 'vendor\rufield\crates\rufield-core\Cargo.toml'
if (-not (Test-Path $RuFieldCore)) {
  throw "Missing $RuFieldCore. Hydrate the vendor/rufield submodule or copy the RuField vendor crates before building the Rust Docker image."
}

Write-Host "Building local image $Image from docker/Dockerfile.rust"
docker build -f docker/Dockerfile.rust -t $Image .

$existing = docker ps -a --filter "name=^/$Container$" --format "{{.Names}}"
if ($existing -eq $Container) {
  Write-Host "Stopping existing container $Container"
  docker stop $Container | Out-Null
  Write-Host "Removing stopped container $Container so it can be recreated with the local image"
  docker rm $Container | Out-Null
}

Write-Host "Starting $Container"
docker run -d --name $Container `
  -p "${HttpPort}:3000" `
  -p "${WsPort}:3001" `
  -p "${UdpPort}:5005/udp" `
  -v "${DockerDataDir}:/app/data" `
  -v "${DockerModelsDir}:/app/models" `
  -e CSI_SOURCE=esp32 `
  -e MODELS_DIR=/app/models `
  -e RECORDINGS_DIR=/app/data/recordings `
  -e RUVIEW_ALLOW_UNAUTHENTICATED=1 `
  -e RUVIEW_FALL_DETECTION=true `
  -e RUVIEW_FALL_IMPACT_Z=6.0 `
  -e RUVIEW_FALL_MIN_NODE_CONSENSUS=2 `
  -e RUVIEW_FALL_CONSENSUS_WINDOW_MS=750 `
  -e RUVIEW_FALL_STILLNESS_SECONDS=3 `
  -e RUVIEW_FALL_COOLDOWN_SECONDS=30 `
  -e RUVIEW_FALL_MIN_SIGNAL_QUALITY=0.25 `
  $Image | Out-Null

$base = "http://localhost:$HttpPort"
Write-Host "Waiting for $base/health"
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
  try {
    Invoke-RestMethod -Method Get -Uri "$base/health" -TimeoutSec 2 | Out-Null
    $ready = $true
    break
  } catch {
    Start-Sleep -Seconds 1
  }
}
if (-not $ready) {
  docker logs --tail 80 $Container
  throw 'RuView container did not become healthy within 60 seconds.'
}

Write-Host ''
Write-Host 'RuView local pose/fall demo is running.'
Write-Host "Dashboard: $base"
Write-Host "Pose:      $base/api/v1/pose/current"
Write-Host "Fall:      $base/api/v1/fall/status"
Write-Host "Nodes:     $base/api/v1/nodes"
Write-Host ''
Write-Host 'Smoke snapshots:'
Invoke-RestMethod -Method Get -Uri "$base/api/v1/pose/current" -TimeoutSec 5 | ConvertTo-Json -Depth 6
Invoke-RestMethod -Method Get -Uri "$base/api/v1/fall/status" -TimeoutSec 5 | ConvertTo-Json -Depth 6
try {
  Invoke-RestMethod -Method Get -Uri "$base/api/v1/nodes" -TimeoutSec 5 | ConvertTo-Json -Depth 6
} catch {
  Write-Warning "Node status unavailable yet: $($_.Exception.Message)"
}

Write-Host ''
Write-Host 'Rollback:'
Write-Host "  docker stop $Container"
Write-Host "  docker rm $Container"
Write-Host '  docker run -d --name ruview -p 3000:3000 -p 3001:3001 -p 5005:5005/udp -e CSI_SOURCE=esp32 -e RUVIEW_ALLOW_UNAUTHENTICATED=1 ruvnet/wifi-densepose:latest'
