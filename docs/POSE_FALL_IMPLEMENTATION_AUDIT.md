# Pose/Fall Implementation Audit

Date: 2026-07-09

## Repository State

- Repository: `C:\RuViewProject\RuView`
- Starting branch: `main`
- Working branch created: `feature/pose-fall-live-demo`
- Starting HEAD: `4bf88e12`
- Pre-existing local changes were present in `ui/services/api.service.js`, `ui/services/pose.service.js`, `ui/services/training.service.js`, `ui/sw.js`, plus several untracked files. These were preserved.

## Active Runtime Path

The active Docker/server path is the Rust sensing server:

- Server: `v2/crates/wifi-densepose-sensing-server/src/main.rs`
- Docker build: `docker/Dockerfile.rust`
- Runtime entrypoint: `docker/docker-entrypoint.sh`
- Static dashboard: `ui/`

Important endpoints and streams are implemented in `main.rs`:

- `/api/v1/sensing/latest`: returns `latest_update`
- `/api/v1/nodes`: built from per-node `NodeState`
- `/api/v1/mesh`: mesh/sync endpoint
- `/api/v1/pose/current`: `pose_current`
- `/api/v1/stream/pose`: pose WebSocket wrapper over sensing updates
- `/ws/sensing`: raw sensing broadcast

## Current Pose Path

The live pose returned by `/api/v1/pose/current` is not a trained neural pose estimate unless `SensingUpdate.pose_keypoints` is present and a trained model path really produced it.

Current fallback flow:

```text
ESP32 CSI frames
  -> FeatureInfo / ClassificationInfo
  -> derive_pose_from_sensing()
  -> derive_single_person_pose()
  -> PersonDetection with COCO-17 keypoints
  -> pose_current and pose WebSocket
  -> ui/utils/pose-renderer.js
```

`derive_single_person_pose()` creates template-like COCO keypoints from measured CSI features such as `motion_band_power`, `variance`, `breathing_band_power`, `dominant_freq_hz`, and `change_points`. This is useful as a demo fallback, but it is heuristic.

## Confidence/Blank Canvas Root Cause

The frontend renderer previously skipped:

- persons below `confidenceThreshold` (`0.3`)
- keypoints below `keypointConfidenceThreshold` (`0.1`)

When the running container emits zero-confidence keypoints, the coordinates can exist but no skeleton lines or keypoints are drawn. This explains the blank canvas.

The local source currently has a heuristic confidence clamp in `derive_single_person_pose()`, while the verified running container returns zero keypoint confidence. That strongly suggests the container may be using an older/remote image or a different built artifact. The fix does not depend on increasing backend confidence.

## Trained Model Status

The server has model loading fields:

- `model_loaded`
- `active_model_id`
- `progressive_loader`
- `pose_keypoints`
- `model_status`

However, for honest pose labeling, a loaded model flag alone is not enough. This implementation reports `TRAINED POSE` only when model-loaded updates actually contain model keypoints. Otherwise visible coordinates are labeled `HEURISTIC CSI POSE - NOT TRAINED`.

## Existing Fall-Related Code

Fall logic existed but was not exposed as a complete demo workflow:

- Firmware edge detector: `firmware/esp32-csi-node/main/edge_processing.c`
- Vitals packet parser: `parse_esp32_vitals()` in `main.rs`
- Vitals packet fall bit: `fall_detected: (flags & 0x02) != 0`
- ADR reference: `docs/adr/ADR-039-esp32-edge-intelligence.md`

The server parsed the edge fall flag but did not expose a stateful `/api/v1/fall/status`, event history, or dashboard fall card.

## Files Changed

- `v2/crates/wifi-densepose-sensing-server/src/main.rs`
- `ui/utils/pose-renderer.js`
- `ui/services/pose.service.js`
- `ui/components/PoseDetectionCanvas.js`
- `scripts/pose-training-windows.ps1`
- `scripts/run-pose-fall-local.ps1`
- `docs/POSE_FALL_IMPLEMENTATION_AUDIT.md`
- `docs/MMFI_MODEL_COMPATIBILITY.md`
- `docs/POSE_FALL_DEMO_GUIDE.md`

## Implementation Summary

Level 1:

- Added `pose_mode`, `pose_label`, `diagnostics`, and model status to `/api/v1/pose/current`.
- Added honest pose labels: `TRAINED POSE`, `HEURISTIC CSI POSE - NOT TRAINED`, `NO VALID POSE`.
- Changed frontend filtering so zero-confidence heuristic coordinates can draw without modifying source confidence.
- Added dashed/translucent heuristic rendering and explicit no-pose state.

Level 2:

- Added server-side CSI fall-like detector state machine:
  - `NORMAL`
  - `IMPACT_CANDIDATE`
  - `POST_IMPACT_OBSERVATION`
  - `FALL_LIKE_CONFIRMED`
  - `RECOVERY`
  - `COOLDOWN`
- Added robust per-node baseline scoring from motion power, variance, spectral power, and the firmware edge fall bit.
- Requires configurable multi-node consensus by default.
- Added `/api/v1/fall/status`, `/api/v1/fall/events`, and `/api/v1/fall/acknowledge`.
- Added WebSocket `fall_status` and `fall_event` messages.
- Persists fall-like events to `data/fall_events.jsonl`.
- Added dashboard fall status card.

Level 3:

- Added Windows workflow wrapper for camera-supervised pose training.
- Documented why the published MM-Fi model is not directly compatible with four-node ESP32 live CSI without a validated adapter.
