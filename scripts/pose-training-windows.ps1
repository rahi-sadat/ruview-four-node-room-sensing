[CmdletBinding()]
param(
  [ValidateSet('check', 'collect', 'align', 'train', 'evaluate', 'install', 'run')]
  [string]$Command = 'check',

  [int]$Duration = 300,

  [ValidateSet('lite', 'base')]
  [string]$Scale = 'lite',

  [string]$Server = 'http://localhost:3000',
  [string]$Room = 'default-room',
  [string]$SubjectId = 'subject-001',
  [string]$Activity = 'mixed-safe-demo'
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Require-Command {
  param([string]$Name)
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "Missing required command '$Name'. Install it, then rerun with -Command check."
  }
}

function Latest-File {
  param([string]$Pattern)
  $item = Get-ChildItem -Path $Pattern -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if (-not $item) {
    throw "No file found for pattern: $Pattern"
  }
  return $item.FullName
}

function Ensure-Dir {
  param([string]$Path)
  if (-not (Test-Path $Path)) {
    New-Item -ItemType Directory -Path $Path | Out-Null
  }
}

function Get-ApiJson {
  param([string]$Path)
  Invoke-RestMethod -Method Get -Uri "$Server$Path" -TimeoutSec 5
}

function Post-ApiJson {
  param([string]$Path, [object]$Body = @{})
  Invoke-RestMethod -Method Post -Uri "$Server$Path" -Body ($Body | ConvertTo-Json -Depth 8) -ContentType 'application/json' -TimeoutSec 10
}

switch ($Command) {
  'check' {
    Require-Command python
    Require-Command node
    Require-Command docker
    python --version
    node --version

    if (-not (Test-Path 'scripts/collect-ground-truth.py')) { throw 'Missing scripts/collect-ground-truth.py' }
    if (-not (Test-Path 'scripts/align-ground-truth.js')) { throw 'Missing scripts/align-ground-truth.js' }
    if (-not (Test-Path 'scripts/train-wiflow-supervised.js')) { throw 'Missing scripts/train-wiflow-supervised.js' }
    if (-not (Test-Path 'scripts/eval-wiflow.js')) { throw 'Missing scripts/eval-wiflow.js' }

    try {
      $nodes = Get-ApiJson '/api/v1/nodes'
      $mesh = Get-ApiJson '/api/v1/mesh'
      $pose = Get-ApiJson '/api/v1/pose/current'
      $fall = Get-ApiJson '/api/v1/fall/status'
      Write-Host "RuView API reachable at $Server"
      Write-Host "Nodes:" ($nodes | ConvertTo-Json -Depth 4)
      Write-Host "Mesh:" ($mesh | ConvertTo-Json -Depth 4)
      Write-Host "Pose mode:" $pose.pose_mode
      Write-Host "Fall state:" $fall.state
    } catch {
      Write-Warning "RuView API check failed: $($_.Exception.Message)"
      Write-Warning "Start the local server/container before collection."
    }

    Ensure-Dir 'data/ground-truth'
    Ensure-Dir 'data/recordings'
    Ensure-Dir 'data/paired'
    Ensure-Dir 'models/wiflow-supervised'
    Write-Host 'Check complete.'
  }

  'collect' {
    Require-Command python
    Ensure-Dir 'data/ground-truth'
    Ensure-Dir 'data/recordings'

    $metadata = [ordered]@{
      room = $Room
      subject_id = $SubjectId
      activity = $Activity
      duration_seconds = $Duration
      server = $Server
      node_layout = 'four ESP32-S3 nodes, node IDs 1-4, TDM slots 0-3'
      safety = 'Use slow lying-down and safe staged fall-like motion onto a mattress only.'
      created_at = (Get-Date).ToUniversalTime().ToString('o')
    }
    $metaPath = "data/ground-truth/session-$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds()).metadata.json"
    $metadata | ConvertTo-Json -Depth 8 | Set-Content -Path $metaPath -Encoding UTF8
    Write-Host "Session metadata: $metaPath"

    python scripts/collect-ground-truth.py --server $Server --duration $Duration --preview --output data/ground-truth
  }

  'align' {
    Require-Command node
    Ensure-Dir 'data/paired'
    $gt = Latest-File 'data/ground-truth/*.jsonl'
    $csi = Latest-File 'data/recordings/*.jsonl'
    $out = "data/paired/aligned-$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds()).paired.jsonl"
    node scripts/align-ground-truth.js --gt $gt --csi $csi --output $out --window-ms 200 --window-frames 20 --min-camera-frames 3 --min-confidence 0.5
    Write-Host "Paired dataset: $out"
  }

  'train' {
    Require-Command node
    $paired = Latest-File 'data/paired/*.paired.jsonl'
    $epochs = if ($Scale -eq 'lite') { 80 } else { 300 }
    node scripts/train-wiflow-supervised.js --data $paired --scale $Scale --epochs $epochs --output models/wiflow-supervised
  }

  'evaluate' {
    Require-Command node
    $paired = Latest-File 'data/paired/*.paired.jsonl'
    $model = Latest-File 'models/wiflow-supervised/*.json'
    node scripts/eval-wiflow.js --model $model --data $paired --baseline --output "data/paired/eval-$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds()).json"
  }

  'install' {
    Ensure-Dir 'data/models'
    $rvf = Get-ChildItem -Path 'models/wiflow-supervised' -Filter '*.rvf' -File -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime -Descending |
      Select-Object -First 1
    if ($rvf) {
      Copy-Item -LiteralPath $rvf.FullName -Destination (Join-Path 'data/models' $rvf.Name) -Force
      Write-Host "Installed RVF model to data/models/$($rvf.Name)"
      try {
        Post-ApiJson '/api/v1/models/load' @{ id = [IO.Path]::GetFileNameWithoutExtension($rvf.Name) } | ConvertTo-Json -Depth 5
      } catch {
        Write-Warning "Model copied, but live load failed: $($_.Exception.Message)"
      }
    } else {
      Write-Warning 'No .rvf file found. The current JS supervised trainer usually emits JSON/SafeTensors artifacts; convert/export to RVF before live Rust loading.'
    }
  }

  'run' {
    & "$PSScriptRoot\run-pose-fall-local.ps1"
  }
}
