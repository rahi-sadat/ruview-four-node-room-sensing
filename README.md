# RuView Four-Node Pose and Fall-Like Detection Demo

This repository is our customized RuView project for a four-node ESP32-S3 WiFi CSI sensing demo. It builds on the original RuView / WiFi-DensePose codebase and adds a project-specific dashboard, trained-pose bridge, CSI recording workflow, coarse zone dashboard, and a fall-like event monitor for demonstration and research use.

## Project Team

| Name | Roll |
| --- | --- |
| Rahi Sadat Ruhan | 2207088 |
| Mehrab Hossen Sakal | 2207066 |

## What This Project Does

RuView uses Channel State Information (CSI) from WiFi hardware to estimate movement and room activity without using a camera at runtime. In this project, we focused on making a practical live demo with four ESP32-S3 nodes.

The system can:

- receive live CSI packets from ESP32-S3 nodes over UDP;
- show stable node status for a four-node room setup;
- display live sensing, pose, and zone information in the browser;
- collect CSI recordings for training and testing;
- use camera-labeled data only during training;
- run a Python PoseTCN bridge for trained 17-keypoint pose output;
- show a fall-like monitor based on multi-node CSI impact plus post-impact stillness;
- provide separate custom pose and coarse zone dashboards for demonstrations.

This is an experimental academic/project demo, not a medical or safety-certified fall detector.

## What We Added

Starting from the RuView repository, we explored the live sensing flow and added the pieces needed for our own four-node room demo.

### Dashboard updates

- Added a `Project Team` tab to the main UI.
- Added project team cards for Rahi Sadat Ruhan and Mehrab Hossen Sakal.
- Improved live sensing node stability so nodes do not disappear immediately during short message gaps.
- Added pose status labels:
  - `TRAINED POSE`
  - `HEURISTIC CSI POSE - NOT TRAINED`
  - `NO VALID POSE`
- Added a fall-like monitor card with confidence, node agreement, and acknowledge button.

### Backend updates

- Added fall-like detector state in the Rust sensing server.
- Added REST endpoints:
  - `GET /api/v1/fall/status`
  - `GET /api/v1/fall/events`
  - `POST /api/v1/fall/acknowledge`
- Added `/ws/fall` so fall messages are separated from normal sensing updates.
- Added support for configurable recordings and models directories through environment variables.
- Improved recording scan behavior for larger JSONL recordings.

### Pose model workflow

- Added `scripts/windows-live-bridge.py`, a Python bridge that can load the exported RVF/PoseTCN model and stream trained keypoints.
- Added `scripts/train_room_pose_gpu.py` for training a room-specific PoseTCN model.
- Added `scripts/export_pose_tcn_to_rvf.py` for packaging the trained checkpoint into an RVF bundle.
- Added `scripts/pose-training-windows.ps1` to help collect, align, train, evaluate, and run the pose workflow on Windows.

### Four-node and demo tooling

- Added `scripts/ruview-wireless.ps1` and `scripts/ruview-wireless.cmd` for the Windows wireless demo workflow.
- Added serial/UDP helper scripts for ESP32 data routing.
- Added `custom-pose-dashboard/` for a focused live pose dashboard.
- Added `zone-dashboard/` for a same-room coarse zone classifier:
  - empty room;
  - near node 1;
  - near node 2;
  - near node 3;
  - near node 4;
  - center;
  - uncertain.

### Documentation added

- `docs/POSE_FALL_DEMO_GUIDE.md`
- `docs/POSE_FALL_IMPLEMENTATION_AUDIT.md`
- `docs/MMFI_MODEL_COMPATIBILITY.md`
- `docs/windows-four-node-offline.md`
- `custom-pose-dashboard/README.md`
- `zone-dashboard/README.md`

## Repository Layout

| Path | Purpose |
| --- | --- |
| `ui/` | Main browser dashboard |
| `v2/crates/wifi-densepose-sensing-server/` | Rust live sensing server |
| `scripts/` | Windows demo, bridge, training, and export helpers |
| `custom-pose-dashboard/` | Focused pose dashboard |
| `zone-dashboard/` | Coarse four-node zone classifier |
| `docs/` | Demo notes, audit notes, and setup guides |
| `docker/` | Docker compose and entrypoint updates |

Local recordings, trained models, virtual environments, runtime files, firmware binary drops, and debug ZIP files are ignored so the GitHub repository stays source-focused.

## Hardware Used

Recommended demo setup:

- 4 x ESP32-S3 CSI nodes;
- 1 WiFi router / access point;
- Windows machine for running Docker, Python bridge scripts, and dashboard;
- optional webcam only for collecting pose labels during training.

Runtime pose/sensing does not require a camera. The camera is used only as a training-time teacher when collecting paired keypoint labels.

## Quick Start

### 1. Start the local RuView backend

```powershell
.\scripts\run-pose-fall-local.ps1
```

Open:

```text
http://localhost:3000
```

Useful pages:

```text
http://localhost:3000/ui/index.html#sensing
http://localhost:3000/ui/index.html#demo
http://localhost:3000/ui/index.html#training
http://localhost:3000/ui/index.html#project-team
```

### 2. Check live CSI nodes

```powershell
Invoke-RestMethod http://localhost:3000/api/v1/nodes | ConvertTo-Json -Depth 8
Invoke-RestMethod http://localhost:3000/api/v1/sensing/latest | ConvertTo-Json -Depth 8
```

Expected result: the four ESP32 nodes should appear as active after packets start arriving.

### 3. Check fall-like detector status

```powershell
Invoke-RestMethod http://localhost:3000/api/v1/fall/status | ConvertTo-Json -Depth 8
Invoke-RestMethod http://localhost:3000/api/v1/fall/events | ConvertTo-Json -Depth 8
```

To acknowledge the latest event:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:3000/api/v1/fall/acknowledge `
  -ContentType application/json `
  -Body '{}'
```

## Trained Pose Bridge

The Docker/Rust dashboard can show heuristic pose output. For trained PoseTCN keypoints, run the Python bridge:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\windows-live-bridge.py `
  --http-port 3100 `
  --ws-port 3101 `
  --no-udp `
  --upstream-sensing-url ws://localhost:3001/ws/sensing `
  --model room-pose-gpu
```

Open:

```text
http://localhost:3100/ui/index.html#demo
```

Check model status:

```powershell
Invoke-RestMethod http://localhost:3100/api/v1/models/active | ConvertTo-Json -Depth 8
```

Check trained keypoints:

```powershell
$pose = Invoke-RestMethod http://localhost:3100/api/v1/pose/current
$pose | Select-Object pose_source,pose_mode,total_persons
$pose.persons[0].keypoints.Count
```

## Training A Room Pose Model

The training workflow is documented in detail in `docs/POSE_FALL_DEMO_GUIDE.md`.

Short version:

```powershell
.\scripts\pose-training-windows.ps1 collect `
  -Duration 300 `
  -Room default-room `
  -SubjectId subject-001 `
  -Activity standing-walking-sitting
```

Then align and train:

```powershell
.\scripts\pose-training-windows.ps1 align

.\.venv-gpu\Scripts\python.exe scripts\train_room_pose_gpu.py `
  --data data\paired\aligned-<timestamp>.paired.jsonl `
  --output models\room-pose-gpu `
  --epochs 100 `
  --batch-size 256
```

Export to RVF:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\export_pose_tcn_to_rvf.py `
  --checkpoint models\room-pose-gpu\best_pose_tcn.pt `
  --report models\room-pose-gpu\training_report.json `
  --output data\models\room-pose-gpu.rvf `
  --model-id room-pose-gpu
```

## Zone Dashboard

Start the coarse zone dashboard:

```powershell
powershell -ExecutionPolicy Bypass -File .\zone-dashboard\run-zone-dashboard.ps1
```

Open:

```text
http://127.0.0.1:8770
```

See `zone-dashboard/README.md` for calibration order and showcase steps.

## Custom Pose Dashboard

Start the focused pose dashboard:

```powershell
powershell -ExecutionPolicy Bypass -File .\custom-pose-dashboard\run-dashboard.ps1 -Mode live
```

Open:

```text
http://127.0.0.1:8766
```

See `custom-pose-dashboard/README.md` for details.

## Verification We Ran

Before preparing this repository for GitHub, we checked:

- JavaScript syntax with `node --check` for the modified UI modules;
- Python syntax with `py -3 -m py_compile` for the new scripts and dashboard apps;
- shell syntax with `sh -n docker/docker-entrypoint.sh`;
- Git ignore rules so large local data, models, virtual environments, runtime files, firmware binary drops, and debug ZIP files are not accidentally published;
- a basic secret scan for obvious hardcoded tokens or credentials in the publishable source set.

Rust compile/tests were not run locally because `cargo` was not available on this Windows PATH during preparation.

## Important Safety Note

The fall-like detector is an experimental CSI demo. It is not a medical device, emergency alert system, or certified fall detector. Do not use it as the only way to monitor a person. Controlled demos should use safe staged motion only, such as a mattress or soft pad, and should not involve uncontrolled falls.

## Upstream Credit

This project is based on the original RuView / WiFi-DensePose repository by `ruvnet`. Our work adds the four-node live demo workflow, project dashboard changes, pose bridge workflow, zone dashboard, fall-like detector UI/API integration, and Windows-focused setup scripts for our project demonstration.

## License

This repository keeps the upstream license terms from RuView. See `LICENSE` for details.
