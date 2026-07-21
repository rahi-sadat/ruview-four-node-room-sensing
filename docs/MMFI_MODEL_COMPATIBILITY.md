# MM-Fi Pose Model Compatibility

Date: 2026-07-09

## Sources Checked

- Published model card: https://huggingface.co/ruvnet/wifi-densepose-mmfi-pose
- Published model files listing: https://huggingface.co/ruvnet/wifi-densepose-mmfi-pose/tree/main
- Pretrained CSI encoder card: https://huggingface.co/ruvnet/wifi-densepose-pretrained
- Local MM-Fi study: `docs/benchmarks/mmfi-wifi-sensing-study.md`
- Local camera training ADR: `docs/adr/ADR-079-camera-ground-truth-training.md`

## Published MM-Fi Pose Model Contract

The Hugging Face model card describes the MM-Fi pose model as:

- Task: WiFi CSI to 17-keypoint COCO skeleton.
- Input: `[3, 114, 10]` CSI amplitude.
- Meaning: 3 antenna pairs, 114 subcarriers, 10 time frames at 100 Hz.
- Output: `[17, 2]` keypoints in normalized `[0, 1]` coordinates.
- Architecture: temporal-token Transformer encoder, temporal attention pooling, MLP head, skeleton graph refinement.
- Model file: `pose_mmfi_best.pt` PyTorch pickle-style state dict.
- Companion files: `README.md`, `model.py`, `pose_mmfi_best.meta.json`.
- License: CC BY-NC 4.0.

The model card also explicitly warns that its headline score is controlled/in-domain and that cross-room generalization is the deployment frontier. It recommends in-room calibration/few-shot adaptation rather than silent zero-shot deployment.

## Live RuView ESP32 Data Contract

The verified live system is:

- Four ESP32-S3 nodes.
- Four node IDs and four TDM slots.
- UDP CSI on port `5005`.
- `/api/v1/sensing/latest` contains real amplitude arrays.
- Up to 192 subcarriers per node in the current deployment.
- Per-node RSSI, variance, motion, and mesh sync are live.
- Sampling is live ESP32 timing, not MM-Fi's fixed public benchmark tensor.

The server currently produces heuristic COCO-17 coordinates from extracted features. It does not currently produce a proven `[3,114,10]` MM-Fi tensor.

## Formal Mismatch

| Requirement | MM-Fi model | Live RuView |
|---|---:|---:|
| Antenna/link layout | 3 antenna pairs | 4 spatial ESP32 nodes |
| Subcarriers | 114 | up to 192 per node |
| Temporal window | 10 frames | live variable windows, server feature windows |
| Sample rate | 100 Hz in model card | live ESP32/mesh rate, measured from nodes |
| Feature type | amplitude tensor | amplitude plus derived features; phase available in raw packets |
| Output | 17x2 normalized | UI expects COCO-style person objects with confidences |
| Format | PyTorch `.pt` plus `model.py` | Rust server expects RVF-compatible live model path |
| License | CC BY-NC 4.0 | repo code is not enough to override model license |

## Decision

Do not silently reshape four-node ESP32 live CSI into `[3,114,10]` and call it trained pose.

Truncating 192 subcarriers to 114, dropping a node, duplicating antenna pairs, padding missing dimensions, or changing timing would create an unvalidated adapter. It might produce numbers, but those numbers would not be defensible as trained live pose output.

## Safe Integration Path

Path A can be revisited only behind an explicit experimental adapter with:

- declared input mapping,
- declared dropped/padded channels,
- per-room calibration data,
- test vectors,
- latency measurements,
- output coordinate checks,
- UI label such as `EXPERIMENTAL MM-FI ADAPTER`.

Until that exists, the correct path is Path B: camera-supervised room-specific training with the existing RuView pipeline.

## Path B Training Workflow

Use:

```powershell
.\scripts\pose-training-windows.ps1 check
.\scripts\pose-training-windows.ps1 collect -Duration 300 -Room default-room -SubjectId subject-001 -Activity mixed-safe-demo
.\scripts\pose-training-windows.ps1 align
.\scripts\pose-training-windows.ps1 train -Scale lite
.\scripts\pose-training-windows.ps1 evaluate
.\scripts\pose-training-windows.ps1 install
```

Recommended collection protocol:

- empty room,
- standing front,
- standing side,
- sitting,
- rising from chair,
- walking left/right,
- walking forward/backward,
- bending,
- crouching,
- lying down slowly,
- safe staged fall-like movement onto a mattress.

Do not perform uncontrolled falls.

## Success Criteria Before Calling Output Trained

- `/api/v1/pose/current.pose_mode == "trained"`.
- Model metadata is present.
- Keypoint confidences are nonzero because the model produced them.
- Evaluation beats constant mean-pose baseline.
- Report includes torso-normalized PCK@20, per-joint accuracy, temporal stability, and held-out validation split.
