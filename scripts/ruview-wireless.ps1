[CmdletBinding()]
param(
    [ValidateSet("start", "recover", "status", "network", "ports", "flash", "provision", "install", "firewall")]
    [string]$Action = "status",

    [string]$Port,
    [int]$NodeId = 0,
    [int]$Slot = -1,
    [ValidateRange(1, 255)]
    [int]$TotalNodes = 4,
    [string]$Ssid,
    [string]$Password,
    [string]$TargetIp = "255.255.255.255",
    [ValidateRange(1, 65535)]
    [int]$TargetPort = 5005,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Firmware = Join-Path $Root "firmware_bins\v0.8.1-esp32"
$Provision = Join-Path $Root "firmware\esp32-csi-node\provision.py"
$Image = Join-Path $Firmware "esp32-csi-node-s3-8mb.bin"
$ProvisionStateDir = Join-Path $env:APPDATA "wifi-densepose\esp32-provision-state"
$RuntimeDir = Join-Path $Root "runtime"
$BridgeScript = Join-Path $Root "scripts\udp-5005-to-docker.py"
$BridgeListenPort = 5005
$DockerUdpHostPort = 5006
$BridgePidFile = Join-Path $RuntimeDir "ruview-udp-bridge.pid"
$BridgeOutLog = Join-Path $RuntimeDir "ruview-udp-bridge.out.log"
$BridgeErrLog = Join-Path $RuntimeDir "ruview-udp-bridge.err.log"
$DockerDataDir = Join-Path $Root "data"
$DockerModelsDir = Join-Path $Root "models"

function Assert-ExitCode {
    param([string]$Operation)
    if ($LASTEXITCODE -ne 0) {
        throw "$Operation failed with exit code $LASTEXITCODE."
    }
}

function Assert-NodeArguments {
    if ([string]::IsNullOrWhiteSpace($Port)) {
        throw "-Port is required for $Action (example: -Port COM5)."
    }
    if ($NodeId -lt 1 -or $NodeId -gt 255) {
        throw "-NodeId must be between 1 and 255."
    }
    if ($Slot -lt 0) {
        $script:Slot = $NodeId - 1
    }
    if ($Slot -lt 0 -or $Slot -ge $TotalNodes) {
        throw "-Slot must be between 0 and $($TotalNodes - 1)."
    }
    if (($Ssid -and -not $Password) -or ($Password -and -not $Ssid)) {
        throw "Supply -Ssid and -Password together."
    }
}

function Invoke-Flash {
    Assert-NodeArguments
    $required = @(
        (Join-Path $Firmware "bootloader.bin"),
        (Join-Path $Firmware "partition-table.bin"),
        (Join-Path $Firmware "ota_data_initial.bin"),
        $Image
    )
    foreach ($path in $required) {
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Offline firmware file is missing: $path"
        }
    }

    Write-Host "Flashing wireless RuView firmware to $Port..." -ForegroundColor Cyan
    & py -3 -m esptool --chip esp32s3 --port $Port --baud 460800 `
        write-flash --flash-mode dio --flash-size 8MB --flash-freq 80m `
        0x0 (Join-Path $Firmware "bootloader.bin") `
        0x8000 (Join-Path $Firmware "partition-table.bin") `
        0xf000 (Join-Path $Firmware "ota_data_initial.bin") `
        0x20000 $Image
    Assert-ExitCode "Firmware flash on $Port"
}

function Test-NodeIdAvailable {
    try {
        $nodes = Invoke-RestMethod "http://localhost:3000/api/v1/nodes" -TimeoutSec 3
        $collision = $nodes.nodes | Where-Object {
            $_.node_id -eq $NodeId -and $_.status -eq "active"
        }
        if ($collision -and -not $Force) {
            throw (
                "Node ID $NodeId is already active over Wi-Fi. Choose another ID, " +
                "power off the old node, or add -Force only for an intentional replacement."
            )
        }
    }
    catch {
        if ($_.Exception.Message -like "Node ID * is already active*") {
            throw
        }
        Write-Warning "Could not query live node IDs; continuing with the explicit NodeId $NodeId."
    }
}

function Invoke-Provision {
    param([switch]$ResetPortState)
    Assert-NodeArguments
    $arguments = @(
        "-3", $Provision,
        "--port", $Port,
        "--target-ip", $TargetIp,
        "--target-port", "$TargetPort",
        "--node-id", "$NodeId",
        "--tdm-slot", "$Slot",
        "--tdm-total", "$TotalNodes"
    )
    if ($ResetPortState) {
        $arguments += "--reset"
    }
    if ($Ssid) {
        $arguments += @("--ssid", $Ssid, "--password", $Password)
    }

    Write-Host (
        "Provisioning $Port as node $NodeId, slot $Slot/$TotalNodes " +
        "-> ${TargetIp}:${TargetPort}..."
    ) -ForegroundColor Cyan
    & py @arguments
    Assert-ExitCode "Provisioning on $Port"
}

function Show-Network {
    Write-Host "Active IPv4 addresses:" -ForegroundColor Cyan
    Get-NetIPConfiguration |
        Where-Object { $_.NetAdapter.Status -eq "Up" -and $_.IPv4Address } |
        ForEach-Object {
            [pscustomobject]@{
                Adapter = $_.InterfaceAlias
                IPv4 = ($_.IPv4Address.IPAddress -join ", ")
                Gateway = ($_.IPv4DefaultGateway.NextHop -join ", ")
            }
        } |
        Format-Table -AutoSize

    Write-Host "Target policy: 255.255.255.255:5005 (no PC IP update needed after DHCP changes)."
}

function Show-Status {
    Write-Host "Docker container:" -ForegroundColor Cyan
    & docker ps --filter "name=^/ruview$" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    Assert-ExitCode "Docker status"
    Show-UdpBridgeStatus

    try {
        $stream = Invoke-RestMethod "http://localhost:3000/api/v1/stream/status" -TimeoutSec 5
        $before = Invoke-RestMethod "http://localhost:3000/api/v1/pose/stats" -TimeoutSec 5
        Start-Sleep -Seconds 3
        $after = Invoke-RestMethod "http://localhost:3000/api/v1/pose/stats" -TimeoutSec 5
        $nodes = Invoke-RestMethod "http://localhost:3000/api/v1/nodes" -TimeoutSec 5
        $activeNodes = @($nodes.nodes | Where-Object { $_.status -eq "active" }).Count

        [pscustomobject]@{
            Source = $stream.source
            FPS = $stream.fps
            Frames = $after.frames_processed
            FramesIn3Seconds = $after.frames_processed - $before.frames_processed
            ActiveNodes = $activeNodes
            TotalNodes = $nodes.total
        } | Format-List
        $nodes.nodes |
            Select-Object node_id, status, last_seen_ms, rssi_dbm, motion_level |
            Sort-Object node_id |
            Format-Table -AutoSize
    }
    catch {
        throw "RuView API is unavailable: $($_.Exception.Message)"
    }
}

function Show-UdpBridgeStatus {
    $pidText = $null
    if (Test-Path -LiteralPath $BridgePidFile) {
        $pidText = (Get-Content -LiteralPath $BridgePidFile -Raw).Trim()
    }
    if ($pidText -and ($pidText -match "^\d+$")) {
        $proc = Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "UDP bridge: running PID $pidText ($BridgeListenPort -> $DockerUdpHostPort)" -ForegroundColor Cyan
            return
        }
    }
    Write-Host "UDP bridge: not tracked by pid file ($BridgeListenPort -> $DockerUdpHostPort)" -ForegroundColor Yellow
}

function Wait-RuViewApi {
    param([int]$TimeoutSeconds = 180)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        try {
            Invoke-RestMethod "http://localhost:3000/api/v1/stream/status" -TimeoutSec 2 *> $null
            return
        }
        catch {
            Start-Sleep -Seconds 1
        }
    } while ((Get-Date) -lt $deadline)

    throw "RuView API did not become ready within $TimeoutSeconds seconds."
}

function Test-RuViewContainerUsesBridgePort {
    $json = (& docker inspect ruview --format "{{json .HostConfig.PortBindings}}" 2>$null)
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($json)) {
        return $false
    }
    $bindings = $json | ConvertFrom-Json
    $udpProp = $bindings.PSObject.Properties["5005/udp"]
    if (-not $udpProp) {
        return $false
    }
    $hostPort = @($udpProp.Value)[0].HostPort
    return $hostPort -eq "$DockerUdpHostPort"
}

function Test-RuViewContainerHasHostMounts {
    $json = (& docker inspect ruview --format "{{json .Mounts}}" 2>$null)
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($json)) {
        return $false
    }
    $mounts = $json | ConvertFrom-Json
    $destinations = @($mounts | ForEach-Object { $_.Destination })
    return ($destinations -contains "/app/data") -and ($destinations -contains "/app/models")
}

function Test-RuViewApiQuick {
    try {
        Invoke-RestMethod "http://localhost:3000/api/v1/stream/status" -TimeoutSec 2 *> $null
        return $true
    }
    catch {
        return $false
    }
}

function Get-RuViewProcessState {
    $state = (& docker exec ruview sh -lc "awk '/^State:/ {print `$2}' /proc/1/status" 2>$null)
    if ($LASTEXITCODE -ne 0) {
        return ""
    }
    return (($state | Select-Object -First 1) -as [string]).Trim()
}

function Test-RuViewHasHttpListener {
    $result = (& docker exec ruview sh -lc "cat /proc/net/tcp /proc/net/tcp6 2>/dev/null | grep -qi ':0BB8' && echo yes || true" 2>$null)
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    return (($result | Select-Object -First 1) -eq "yes")
}

function Test-RuViewContainerWedged {
    $running = (& docker inspect -f "{{.State.Running}}" ruview 2>$null)
    if ($LASTEXITCODE -ne 0 -or $running -ne "true") {
        return $false
    }
    if (Test-RuViewApiQuick) {
        return $false
    }

    $state = Get-RuViewProcessState
    $hasHttp = Test-RuViewHasHttpListener
    if ($state -eq "D" -or -not $hasHttp) {
        Write-Warning (
            "ruview container is unhealthy: API is not answering, " +
            "process_state='$state', http_listener=$hasHttp."
        )
        return $true
    }
    return $false
}

function New-RuViewContainer {
    Write-Host (
        "Creating ruview as a minimal offline CSI receiver " +
        "(ESP/bridge 5005 -> Docker host $DockerUdpHostPort -> container 5005)..."
    ) -ForegroundColor Cyan
    & docker run -d --name ruview `
        -p 3000:3000 -p 3001:3001 -p ${DockerUdpHostPort}:5005/udp `
        -e CSI_SOURCE=esp32 `
        -e RUVIEW_ALLOW_UNAUTHENTICATED=1 `
        ruvnet/wifi-densepose:latest `
        --no-edge-registry
    Assert-ExitCode "Creating the ruview container"
}

function Start-UdpBridge {
    if (-not (Test-Path -LiteralPath $RuntimeDir)) {
        New-Item -ItemType Directory -Path $RuntimeDir | Out-Null
    }

    if (Test-Path -LiteralPath $BridgePidFile) {
        $pidText = (Get-Content -LiteralPath $BridgePidFile -Raw).Trim()
        if ($pidText -match "^\d+$") {
            $existing = Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue
            if ($existing) {
                Write-Host "UDP bridge already running: PID $pidText ($BridgeListenPort -> $DockerUdpHostPort)" -ForegroundColor Cyan
                return
            }
        }
    }

    if (-not (Test-Path -LiteralPath $BridgeScript)) {
        throw "Missing UDP bridge script: $BridgeScript"
    }

    Write-Host "Starting Windows UDP bridge: $BridgeListenPort -> Docker host port $DockerUdpHostPort..." -ForegroundColor Cyan
    $arguments = @(
        "-3",
        $BridgeScript,
        "--listen-port", "$BridgeListenPort",
        "--target-port", "$DockerUdpHostPort"
    )
    $process = Start-Process -FilePath "py" `
        -ArgumentList $arguments `
        -WindowStyle Hidden `
        -RedirectStandardOutput $BridgeOutLog `
        -RedirectStandardError $BridgeErrLog `
        -PassThru
    Set-Content -LiteralPath $BridgePidFile -Value $process.Id
    Start-Sleep -Seconds 2
    if ($process.HasExited) {
        throw "UDP bridge failed to start. Check $BridgeErrLog and $BridgeOutLog."
    }
}

function Ensure-DockerMountDirs {
    foreach ($path in @($DockerDataDir, $DockerModelsDir)) {
        if (-not (Test-Path -LiteralPath $path)) {
            New-Item -ItemType Directory -Path $path | Out-Null
        }
    }
}

function Start-RuView {
    param(
        [switch]$SkipStatus,
        [int]$ApiWaitSeconds = 20
    )

    & docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Desktop is not running. Start Docker Desktop, wait until it is ready, then rerun this command."
    }
    Ensure-DockerMountDirs

    & docker container inspect ruview *> $null
    if ($LASTEXITCODE -eq 0) {
        if ((-not (Test-RuViewContainerUsesBridgePort)) -or (Test-RuViewContainerHasHostMounts)) {
            Write-Host (
                "Recreating ruview for the minimal reliable Windows UDP bridge " +
                "(host $DockerUdpHostPort -> container 5005)..."
            ) -ForegroundColor Cyan
            & docker rm -f ruview
            Assert-ExitCode "Recreating ruview container"
            New-RuViewContainer
        }
        elseif ((& docker inspect -f "{{.State.Running}}" ruview 2>$null) -eq "true") {
            if ($Force) {
                Write-Host "Force-restarting ruview..." -ForegroundColor Cyan
                & docker restart ruview
                Assert-ExitCode "Restarting the ruview container"
            }
            elseif (Test-RuViewContainerWedged) {
                Write-Host "Recreating wedged ruview container..." -ForegroundColor Cyan
                & docker rm -f ruview
                Assert-ExitCode "Removing wedged ruview container"
                New-RuViewContainer
            }
            else {
                Write-Host "ruview is already running; keeping it up." -ForegroundColor Cyan
            }
        }
        else {
            & docker start ruview
            Assert-ExitCode "Starting the ruview container"
        }
    }
    else {
        & docker image inspect ruvnet/wifi-densepose:latest *> $null
        if ($LASTEXITCODE -ne 0) {
            throw (
                "The ruview container and cached Docker image are both missing. " +
                "An internet connection is required once to download the image."
            )
        }
        New-RuViewContainer
    }
    & docker update --restart unless-stopped ruview *> $null
    Assert-ExitCode "Setting the ruview restart policy"
    Start-UdpBridge

    $apiReady = $false
    try {
        Wait-RuViewApi -TimeoutSeconds $ApiWaitSeconds
        $apiReady = $true
    }
    catch {
        Write-Warning (
            "RuView is still booting after $ApiWaitSeconds seconds. " +
            "Docker and the UDP bridge are started; run scripts\ruview-wireless.cmd status in a minute."
        )
    }

    if (-not $SkipStatus -and $apiReady) {
        Show-Status
    }
}

function Get-EspSerialPorts {
    $code = "import serial.tools.list_ports as p; [print(f'{x.device}|{x.description}|{x.hwid}') for x in p.comports()]"
    $lines = & py -3 -c $code
    Assert-ExitCode "Serial-port listing"

    foreach ($line in $lines) {
        $parts = $line -split "\|", 3
        if ($parts.Count -lt 3) {
            continue
        }
        $device = $parts[0]
        $description = $parts[1]
        $hwid = $parts[2]
        $combined = "$description $hwid"

        if ($combined -match "Bluetooth|BTHENUM") {
            continue
        }
        if ($combined -match "CH343|CH340|CP210|USB.*SERIAL|VID:PID=1A86|VID:PID=10C4|VID:PID=303A") {
            [pscustomobject]@{
                Port = $device
                Description = $description
                HardwareId = $hwid
            }
        }
    }
}

function Get-SavedProvisionState {
    param([string]$StatePort)
    $path = Join-Path $ProvisionStateDir "$StatePort.json"
    if (-not (Test-Path -LiteralPath $path)) {
        return $null
    }
    Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
}

function Recover-PluggedNodes {
    Start-RuView -SkipStatus -ApiWaitSeconds 20

    $ports = @(Get-EspSerialPorts)
    if ($ports.Count -eq 0) {
        Write-Warning "No USB ESP32 boards are plugged in. Corner nodes may still recover over Wi-Fi; checking dashboard status."
        Show-Status
        return
    }

    Write-Host "Plugged ESP32 boards:" -ForegroundColor Cyan
    $ports | Select-Object Port, Description | Format-Table -AutoSize

    foreach ($esp in $ports) {
        $state = Get-SavedProvisionState -StatePort $esp.Port
        if (-not $state) {
            Write-Warning "No saved node config for $($esp.Port). Run install once for this board/port."
            continue
        }

        $script:Port = $esp.Port
        $script:NodeId = [int]$state.node_id
        $script:Slot = [int]$state.tdm_slot
        $script:TotalNodes = if ($state.tdm_total) { [int]$state.tdm_total } else { 4 }
        $script:Ssid = [string]$state.ssid
        $script:Password = [string]$state.password
        $script:TargetPort = $TargetPort

        Write-Host (
            "Recovering $Port from saved node config as node $NodeId. " +
            "Using target ${TargetIp}:${TargetPort}."
        ) -ForegroundColor Cyan
        Invoke-Provision
    }

    Write-Host "Waiting 15 seconds for Wi-Fi reconnect and UDP frames..." -ForegroundColor Cyan
    Start-Sleep -Seconds 15
    Wait-RuViewApi -TimeoutSeconds 240
    Show-Status
}

Set-Location $Root

switch ($Action) {
    "start" {
        Start-RuView
    }
    "recover" {
        Recover-PluggedNodes
    }
    "status" {
        Show-Status
    }
    "network" {
        Show-Network
    }
    "ports" {
        & py -3 -m serial.tools.list_ports -v
        Assert-ExitCode "Serial-port listing"
    }
    "flash" {
        Invoke-Flash
    }
    "provision" {
        Invoke-Provision
    }
    "install" {
        if (-not $Ssid -or -not $Password) {
            throw (
                "A fresh install requires both -Ssid and -Password. " +
                "The old configuration saved for this COM port will not be reused."
            )
        }
        Test-NodeIdAvailable
        Invoke-Flash
        Invoke-Provision -ResetPortState
        Write-Host "Wait 15 seconds, then run: scripts\ruview-wireless.cmd status"
    }
    "firewall" {
        $principal = New-Object Security.Principal.WindowsPrincipal(
            [Security.Principal.WindowsIdentity]::GetCurrent()
        )
        if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
            throw "Open PowerShell as Administrator, then run this firewall action again."
        }
        & netsh advfirewall firewall delete rule name="RuView ESP32 CSI UDP 5005" *> $null
        & netsh advfirewall firewall add rule `
            name="RuView ESP32 CSI UDP 5005" `
            dir=in action=allow protocol=UDP localport=5005
        Assert-ExitCode "Creating the Windows firewall rule"
    }
}
