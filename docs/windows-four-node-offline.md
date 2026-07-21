# Four-node wireless RuView on Windows

This setup uses four ESP32-S3 nodes, one per corner. Each node sends ADR-018
CSI wirelessly to Windows UDP port 5005. USB is needed only to flash or
change a node's stored Wi-Fi settings; normal operation needs power only.

On Windows, Docker Desktop can lose or ignore Wi-Fi/broadcast UDP after a
router disconnect. The startup script therefore uses a small local bridge:

```text
ESP32 nodes -> 255.255.255.255:5005
Windows UDP bridge -> 127.0.0.1:5006
Docker host port 5006 -> container UDP 5005
```

This is why Docker status should show `5006->5005/udp`. The ESP boards still
use `255.255.255.255:5005`; do not change them to 5006.

Internet access is not required after the Docker image, Python packages, and
firmware files are present. A local Wi-Fi access point is still required.
The Cudy router can operate without a WAN/internet connection.

## Stable node assignments

| Corner | Node ID | TDM slot | Example setup port |
|---|---:|---:|---|
| 1 | 1 | 0 | COM5 |
| 2 | 2 | 1 | assign when connected |
| 3 | 3 | 2 | COM10 |
| 4 | 4 | 3 | assign when connected |

COM port numbers are temporary Windows setup identifiers. The node ID and TDM
slot are stored on the board and do not change when USB is removed. A different
board may reuse COM5, COM10, or any other available port. The `install` action
clears the old per-port provisioning state before assigning the new board, so
reusing COM5 does not copy node 1's identity.

## Normal offline startup

1. Power the Cudy router. It does not need an internet/WAN connection.
2. Power all corner ESP32 nodes.
3. Start Docker Desktop and wait until it reports that the engine is running.
4. Run:

```powershell
cd C:\RuViewProject\RuView
scripts\ruview-wireless.cmd start
```

`start` starts the `ruview` container if it is stopped, starts the Windows
UDP bridge, and performs a short readiness check. It does not restart an
already-running container unless you pass `-Force`; this keeps daily startup
fast and avoids Docker's slow cold boot unless it is actually needed.

Open:

```text
http://localhost:3000/ui/index.html#demo
```

Check all nodes and verify the frame counter is increasing:

```powershell
scripts\ruview-wireless.cmd status
```

## Install each board separately

Connect only the board being installed. Substitute its current COM port and
use the stable assignment from the table.

```powershell
# Corner 1
scripts\ruview-wireless.cmd install -Port COM5 -NodeId 1 -Slot 0 `
  -Ssid "Cudy-8CC8" -Password "YOUR_WIFI_PASSWORD"

# Corner 2
scripts\ruview-wireless.cmd install -Port COM6 -NodeId 2 -Slot 1 `
  -Ssid "Cudy-8CC8" -Password "YOUR_WIFI_PASSWORD"

# Corner 3
scripts\ruview-wireless.cmd install -Port COM10 -NodeId 3 -Slot 2 `
  -Ssid "Cudy-8CC8" -Password "YOUR_WIFI_PASSWORD"

# Corner 4
scripts\ruview-wireless.cmd install -Port COM11 -NodeId 4 -Slot 3 `
  -Ssid "Cudy-8CC8" -Password "YOUR_WIFI_PASSWORD"
```

The default target is `255.255.255.255:5005`. Keep it unless the Wi-Fi
network blocks broadcast traffic. DHCP can change the PC's IPv4 address
without requiring node reconfiguration.

`install` refuses a node ID that is already active. Use another ID rather than
creating a collision. `-Force` is reserved for intentionally replacing a
failed board after the old board has been powered off.

## Wi-Fi or IP changes

Show current adapters, IPv4 addresses, and gateways:

```powershell
scripts\ruview-wireless.cmd network
ipconfig
```

If only the PC's DHCP address changed, do nothing to the nodes because they
use the broadcast target.

If the Wi-Fi SSID or password changed, reconnect each board by USB and
re-provision it. A firmware flash is not needed:

```powershell
scripts\ruview-wireless.cmd provision -Port COM5 -NodeId 1 -Slot 0 `
  -Ssid "NEW_SSID" -Password "NEW_PASSWORD"
```

Repeat with each node's ID and slot. If the access point isolates wireless
clients or blocks broadcast, use the PC's Wi-Fi IPv4 address explicitly:

```powershell
scripts\ruview-wireless.cmd provision -Port COM5 -NodeId 1 -Slot 0 `
  -TargetIp "192.168.1.50"
```

An explicit PC address must be updated on every node whenever DHCP changes it.
A DHCP reservation for the PC avoids that maintenance.

## Recovery and diagnostics

If anything small disconnects — router power, Wi-Fi reconnect, Docker restart,
or a USB replug — use this first:

```powershell
cd C:\RuViewProject\RuView
scripts\ruview-wireless.cmd start
```

If one or more boards are plugged into USB and you want the tool to reapply
their saved node settings without reflashing:

```powershell
scripts\ruview-wireless.cmd recover
```

`recover` restarts RuView, starts the UDP bridge, scans real USB ESP boards
(ignoring Bluetooth COM ports), and re-provisions plugged boards from their
saved node ID/slot/Wi-Fi settings.

List attached boards:

```powershell
scripts\ruview-wireless.cmd ports
```

Create the UDP firewall rule from an Administrator PowerShell:

```powershell
scripts\ruview-wireless.cmd firewall
```

Reapply configuration without reflashing:

```powershell
scripts\ruview-wireless.cmd provision -Port COM5 -NodeId 1 -Slot 0
```

Reflash only if the boot log does not identify the application as
`esp32-csi-node`, then provision it again.
