# Pose/Fall Live Demo Guide

Date: 2026-07-13

## Architecture

```text
Four ESP32-S3 CSI nodes
  -> UDP 5005
  -> Rust sensing server
      -> /api/v1/sensing/latest
      -> /api/v1/pose/current
      -> /api/v1/fall/status
      -> /api/v1/fall/events
      -> /ws/sensing
      -> /api/v1/stream/pose
  -> Dashboard
      -> pose badge: TRAINED / HEURISTIC / NONE
      -> dashed heuristic skeleton when only coordinates exist
      -> fall-like detector card and acknowledge button
```

## What Was Changed For The Room Pose Model

The live demo originally only showed a synthetic/demo skeleton or a heuristic CSI
skeleton. Loading an `.rvf` file in the Docker dashboard made the model appear
selected, but it did not run the `PoseTCN` network to produce live keypoints.

The current workspace now has a usable model path:

- `models/room-pose-gpu/best_pose_tcn.pt` is the trained PyTorch checkpoint.
- `data/models/room-pose-gpu.rvf` is the exported RVF bundle used by the UI/model
  library.
- `scripts/export_pose_tcn_to_rvf.py` packages the PyTorch checkpoint into the
  RVF segment format with manifest, metadata, weights, and training metrics.
- `scripts/windows-live-bridge.py` can now load the RVF, reconstruct `PoseTCN`,
  keep a 20-frame CSI window, and emit trained COCO-17 keypoints.
- `ui/services/model.service.js` normalizes model API responses so both
  `{ active: { id: ... } }` and `{ model_id: ... }` shapes show correctly.

Verified current model:

```text
model_id: room-pose-gpu
input: 56 CSI channels x 20 frames
output: 17 COCO keypoints, normalized x/y
parameters: 136,809
eval samples: 2,763
PCK@20: 47.0%
PCK@50: 80.8%
```

Verified live API result:

```json
{
  "pose_source": "model_inference",
  "pose_mode": "trained",
  "persons": 1,
  "keypoints": 17,
  "model": "room-pose-gpu",
  "ready": true
}
```

Important: the Rust/Docker app on port `3000` can list/load the RVF, but in this
checkout its model load endpoint is mostly a state switch. The actual trained
keypoint inference path is the Python/PyTorch bridge on port `3100`.

## Use The Current Trained Model Live

Start or keep the Docker receiver running on `3000/3001`. It receives ESP32 CSI
and publishes `/ws/sensing`.

```powershell
.\scripts\run-pose-fall-local.ps1
```

If you are using the already-running Docker container, you can skip that command.

Start the model-backed bridge in a second terminal:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\windows-live-bridge.py `
  --http-port 3100 `
  --ws-port 3101 `
  --no-udp `
  --upstream-sensing-url ws://localhost:3001/ws/sensing `
  --model room-pose-gpu
```

Keep this terminal open. Stop it with `Ctrl+C`.

Open the trained-pose live page:

```text
http://localhost:3100/ui/index.html#demo
```

Use `3100` for trained PoseTCN live pose. Use `3000` for the normal Docker
dashboard.

Check that the model is loaded:

```powershell
Invoke-RestMethod http://localhost:3100/api/v1/models/active | ConvertTo-Json -Depth 8
```

Check that trained keypoints are being emitted:

```powershell
$pose = Invoke-RestMethod http://localhost:3100/api/v1/pose/current
$pose | Select-Object pose_source,pose_mode,total_persons
$pose.persons[0].keypoints.Count
```

Expected after about 20 CSI frames:

```text
pose_source: model_inference
pose_mode: trained
keypoints: 17
```

If `input_window_ready` is false, wait a few seconds. If `pose_mode` stays
`heuristic` or `none`, check that `/api/v1/sensing/latest.nodes[0].amplitude`
contains CSI amplitudes.

## Train A New PoseTCN Model

The camera is used only while collecting labels. Runtime remains camera-free.

1. Start the live backend and make sure ESP32 CSI is arriving:

```powershell
Invoke-RestMethod http://localhost:3000/api/v1/sensing/latest | ConvertTo-Json -Depth 4
```

2. Collect camera-labeled training data:

```powershell
.\scripts\pose-training-windows.ps1 collect `
  -Duration 300 `
  -Room default-room `
  -SubjectId subject-001 `
  -Activity standing-walking-sitting
```

This writes camera keypoints under `data/ground-truth/` and CSI recordings under
`data/recordings/`.

3. Align CSI windows with camera keypoints:

```powershell
.\scripts\pose-training-windows.ps1 align
```

This creates a paired dataset like:

```text
data/paired/aligned-<timestamp>.paired.jsonl
```

4. Train the PoseTCN model with the GPU Python environment:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\train_room_pose_gpu.py `
  --data data\paired\aligned-<timestamp>.paired.jsonl `
  --output models\room-pose-gpu `
  --epochs 100 `
  --batch-size 256
```

Do not use plain `python` if it resolves to Python 2.7 on this machine. Use
`.venv-gpu\Scripts\python.exe` for PyTorch/Torch CUDA commands.

Training outputs:

```text
models/room-pose-gpu/best_pose_tcn.pt
models/room-pose-gpu/training_report.json
models/room-pose-gpu/eval_predictions.jsonl
```

5. Export the trained checkpoint to RVF:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\export_pose_tcn_to_rvf.py `
  --checkpoint models\room-pose-gpu\best_pose_tcn.pt `
  --report models\room-pose-gpu\training_report.json `
  --output data\models\room-pose-gpu.rvf `
  --model-id room-pose-gpu
```

6. Load or restart the model bridge:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:3100/api/v1/models/load `
  -ContentType application/json `
  -Body '{"model_id":"room-pose-gpu"}' | ConvertTo-Json -Depth 8
```

Or restart the bridge with `--model room-pose-gpu`.

### Training Quality Notes

- Keep ESP32 node placement fixed during collection and runtime.
- The camera should see the same area that the WiFi nodes cover.
- Collect standing, walking, sitting, turning, crouching, lying down, and
  empty-room segments.
- More diverse data usually matters more than more epochs.
- Do not claim full trained pose success only because the model loads. Confirm
  `pose_source` is `model_inference`, `pose_mode` is `trained`, and the evaluation
  beats a constant mean-pose baseline.

### Direct UDP Mode

If Docker is stopped and the bridge should receive ESP32 UDP directly, run:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\windows-live-bridge.py `
  --http-port 3000 `
  --ws-port 3001 `
  --udp-port 5005 `
  --model room-pose-gpu
```

Do not run this direct mode while Docker is also bound to the same ports.

## Build And Run Local Docker

Use the local image; the remote `ruvnet/wifi-densepose:latest` image will not contain these edits.

```powershell
.\scripts\run-pose-fall-local.ps1
```

Equivalent manual build:

```powershell
docker build -f docker/Dockerfile.rust -t ruview-local:pose-fall .
```

Equivalent manual run:

```powershell
docker stop ruview
docker rm ruview
docker run -d --name ruview `
  -p 3000:3000 `
  -p 3001:3001 `
  -p 5005:5005/udp `
  -e CSI_SOURCE=esp32 `
  -e RUVIEW_ALLOW_UNAUTHENTICATED=1 `
  -e RUVIEW_FALL_DETECTION=true `
  -e RUVIEW_FALL_IMPACT_Z=6.0 `
  -e RUVIEW_FALL_MIN_NODE_CONSENSUS=2 `
  -e RUVIEW_FALL_CONSENSUS_WINDOW_MS=750 `
  -e RUVIEW_FALL_STILLNESS_SECONDS=3 `
  -e RUVIEW_FALL_COOLDOWN_SECONDS=30 `
  -e RUVIEW_FALL_MIN_SIGNAL_QUALITY=0.25 `
  ruview-local:pose-fall
```

## Smoke Tests

```powershell
Invoke-RestMethod http://localhost:3000/health | ConvertTo-Json -Depth 6
Invoke-RestMethod http://localhost:3000/api/v1/nodes | ConvertTo-Json -Depth 8
Invoke-RestMethod http://localhost:3000/api/v1/mesh | ConvertTo-Json -Depth 8
Invoke-RestMethod http://localhost:3000/api/v1/sensing/latest | ConvertTo-Json -Depth 8
Invoke-RestMethod http://localhost:3000/api/v1/pose/current | ConvertTo-Json -Depth 8
Invoke-RestMethod http://localhost:3000/api/v1/fall/status | ConvertTo-Json -Depth 8
Invoke-RestMethod http://localhost:3000/api/v1/fall/events | ConvertTo-Json -Depth 8
```

Expected:

- `/api/v1/nodes` shows nodes 1-4 active after frames arrive.
- `/api/v1/sensing/latest.source` is `esp32`.
- `/api/v1/pose/current.pose_mode` is `heuristic`, `trained`, or `none`.
- `/api/v1/fall/status.state` is one of the documented detector states.

## Pose Demo Procedure

1. For the normal Docker dashboard, open `http://localhost:3000`.
2. For trained PoseTCN live pose, open `http://localhost:3100/ui/index.html#demo`.
3. Start the live pose view.
4. Empty room:
   - canvas should show `NO VALID POSE` instead of going blank.
5. One person walking before the model window is ready:
   - canvas may show a dashed skeleton labeled `HEURISTIC CSI POSE - NOT TRAINED`.
   - source confidence is preserved; visual confidence is separate.
6. Trained pose after the model window is ready:
   - `/api/v1/pose/current.pose_source` must be `model_inference`.
   - `/api/v1/pose/current.pose_mode` must be `trained`.
   - `/api/v1/pose/current.persons[0].keypoints` must contain 17 keypoints.
   - skeleton is solid and labeled `TRAINED POSE`.

Legacy checklist for the Docker-only page:

1. Open `http://localhost:3000`.
2. Start the live pose view.
3. Empty room:
   - canvas should show `NO VALID POSE` instead of going blank.
4. One person walking:
   - if trained model keypoints are unavailable, canvas should show a dashed skeleton labeled `HEURISTIC CSI POSE - NOT TRAINED`.
   - source confidence is preserved; visual confidence is separate.
5. Trained pose when available:
   - load a compatible trained model.
   - `/api/v1/pose/current.pose_mode` must be `trained`.
   - skeleton is solid and labeled `TRAINED POSE`.

## Live WiFi Sensing Page

The screenshot page is:

```text
http://localhost:3000/ui/index.html#sensing
```

It is a live CSI visualization page. It does not use the camera and it does not
collect pose labels. Use it to confirm the four-node stream is stable:

- banner says `LIVE - ESP32 HARDWARE`;
- node status stays at 4 active nodes;
- `/api/v1/sensing/latest.source` is `esp32`;
- `/api/v1/sensing/latest.nodes` contains node IDs 1-4.

If the node count flickers to 0, rebuild/restart the server and refresh the
browser. The expected fix is that fall messages use `/ws/fall`, while the
sensing page only renders `sensing_update` messages from `/ws/sensing`.

## Which Button Records Data?

For CSI-only recording from the dashboard:

1. Open `http://localhost:3000/ui/index.html#training`.
2. In `CSI Recordings`, press `Start Recording`.
3. Keep the ESP32 nodes fixed and perform the activity.
4. Press `Stop Recording`.

The Sensing page itself has no recording button.

For camera-labeled pose training data, use the script:

```powershell
.\scripts\pose-training-windows.ps1 collect -Duration 300 -Room default-room -SubjectId subject-001 -Activity mixed-safe-demo
```

That command starts the CSI recording through the backend and opens the webcam
preview for MediaPipe labels. The camera is only the training-time teacher; it
should face the same sensing area and see the full body. Runtime stays
camera-free.

## Project Team Page

Open:

```text
http://localhost:3000/ui/index.html#project-team
```

Or click the `Project Team` tab in the top dashboard navigation.

## Fall-Like Detector Configuration

Defaults:

```powershell
$env:RUVIEW_FALL_DETECTION='true'
$env:RUVIEW_FALL_IMPACT_Z='6.0'
$env:RUVIEW_FALL_MIN_NODE_CONSENSUS='2'
$env:RUVIEW_FALL_CONSENSUS_WINDOW_MS='750'
$env:RUVIEW_FALL_STILLNESS_SECONDS='3'
$env:RUVIEW_FALL_COOLDOWN_SECONDS='30'
$env:RUVIEW_FALL_MIN_SIGNAL_QUALITY='0.25'
```

Calibration/tuning commands:

```powershell
# Normal baseline check
Invoke-RestMethod http://localhost:3000/api/v1/fall/status | ConvertTo-Json -Depth 8

# Acknowledge the latest confirmed event
Invoke-RestMethod -Method Post http://localhost:3000/api/v1/fall/acknowledge -Body '{}' -ContentType 'application/json' | ConvertTo-Json -Depth 8

# Review recent events
Invoke-RestMethod http://localhost:3000/api/v1/fall/events | ConvertTo-Json -Depth 8
```

Safe staged fall-like demo:

1. Place a mattress or thick pad in the sensing area.
2. Start from standing.
3. Perform a controlled drop/lie-down motion onto the mattress.
4. Remain still for 3-5 seconds.
5. Confirm the dashboard shows `FALL-LIKE EVENT`.
6. Press acknowledge.

Do not perform uncontrolled falls.

## Legacy Training Helper Commands

These commands are kept for the older supervised WiFlow helper path. For the
current `room-pose-gpu.rvf` model, prefer the PoseTCN flow in
`Train A New PoseTCN Model` above.

```powershell
.\scripts\pose-training-windows.ps1 check
.\scripts\pose-training-windows.ps1 collect -Duration 300 -Room default-room -SubjectId subject-001 -Activity standing-walking-sitting
.\scripts\pose-training-windows.ps1 align
.\scripts\pose-training-windows.ps1 train -Scale lite
.\scripts\pose-training-windows.ps1 evaluate
.\scripts\pose-training-windows.ps1 install
.\scripts\pose-training-windows.ps1 run
```

Notes:

- The webcam is a training-time teacher only.
- Runtime deployment remains camera-free.
- Use the lite model first.
- The `train -Scale lite` command trains the older JS model path, not the
  `PoseTCN` checkpoint loaded by `windows-live-bridge.py`.
- Do not declare trained pose success unless evaluation beats the constant
  mean-pose baseline.

## Rollback

Return to the remote image:

```powershell
docker stop ruview
docker rm ruview
docker run -d --name ruview `
  -p 3000:3000 `
  -p 3001:3001 `
  -p 5005:5005/udp `
  -e CSI_SOURCE=esp32 `
  -e RUVIEW_ALLOW_UNAUTHENTICATED=1 `
  ruvnet/wifi-densepose:latest
```

Return to source-only state:

```powershell
git switch main
```

Do not use `git reset --hard` unless you intentionally want to discard local changes.

## Verification Performed In This Workspace

- `py -3 -m py_compile scripts\windows-live-bridge.py`: passed.
- `node --check ui/services/model.service.js`: passed.
- `.venv-gpu\Scripts\python.exe` loaded `data/models/room-pose-gpu.rvf` and
  produced one person with 17 keypoints from a synthetic CSI window.
- `http://localhost:3100/api/v1/pose/current` returned
  `pose_source=model_inference`, `pose_mode=trained`, `keypoints=17`.
- `node --check ui/utils/pose-renderer.js`: passed.
- `node --check ui/services/pose.service.js`: passed.
- `node --check ui/components/PoseDetectionCanvas.js`: passed.
- Rust `cargo` was not available on this Windows PATH, so Rust compile/tests were not run locally in this session.
- `docker build -f docker/Dockerfile.rust -t ruview-local:pose-fall .` was attempted with Docker access. It reached Cargo, then failed before compiling the changed server because this checkout's `vendor/rufield` directory is empty/missing `crates/rufield-core/Cargo.toml`, while `v2/crates/wifi-densepose-rufield/Cargo.toml` path-depends on it.
