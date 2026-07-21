#!/usr/bin/env python3
"""Package a room-pose PyTorch TCN checkpoint as a RuView RVF container.

This exporter preserves the trained tensor values in the standard RVF `SEG_VEC`
weight segment and stores tensor names/shapes plus evaluation metrics as JSON
metadata. It does not by itself add live Rust inference for this PyTorch
architecture; it makes the model discoverable and loadable as an RVF bundle.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
import zlib
from pathlib import Path
from typing import Any

import torch


SEGMENT_MAGIC = 0x5256_4653
SEGMENT_VERSION = 1
SEGMENT_ALIGNMENT = 64
SEGMENT_HEADER_SIZE = 64

SEG_VEC = 0x01
SEG_MANIFEST = 0x05
SEG_META = 0x07
SEG_WITNESS = 0x0A


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export best_pose_tcn.pt to a RuView .rvf bundle")
    parser.add_argument("--checkpoint", default="models/room-pose-gpu/best_pose_tcn.pt")
    parser.add_argument("--report", default="models/room-pose-gpu/training_report.json")
    parser.add_argument("--output", default="data/models/room-pose-gpu.rvf")
    parser.add_argument("--model-id", default="room-pose-gpu")
    parser.add_argument("--version", default="0.1.0")
    return parser.parse_args()


def align_up(size: int) -> int:
    return (size + SEGMENT_ALIGNMENT - 1) & ~(SEGMENT_ALIGNMENT - 1)


def content_hash(payload: bytes) -> bytes:
    crc = zlib.crc32(payload) & 0xFFFF_FFFF
    return struct.pack("<I", crc) + b"\x00" * 12


def segment(seg_type: int, segment_id: int, payload: bytes) -> bytes:
    raw_len = SEGMENT_HEADER_SIZE + len(payload)
    pad_len = align_up(raw_len) - raw_len
    header = bytearray(SEGMENT_HEADER_SIZE)
    header[0x00:0x04] = struct.pack("<I", SEGMENT_MAGIC)
    header[0x04] = SEGMENT_VERSION
    header[0x05] = seg_type
    header[0x06:0x08] = struct.pack("<H", 0)
    header[0x08:0x10] = struct.pack("<Q", segment_id)
    header[0x10:0x18] = struct.pack("<Q", len(payload))
    header[0x18:0x20] = struct.pack("<Q", time.time_ns())
    header[0x20] = 0
    header[0x21] = 0
    header[0x22:0x24] = struct.pack("<H", 0)
    header[0x24:0x28] = struct.pack("<I", 0)
    header[0x28:0x38] = content_hash(payload)
    header[0x38:0x3C] = struct.pack("<I", 0)
    header[0x3C:0x40] = struct.pack("<I", pad_len)
    return bytes(header) + payload + (b"\x00" * pad_len)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    report_path = Path(args.report)
    output_path = Path(args.output)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint does not contain a model_state dictionary")

    tensors: list[dict[str, Any]] = []
    flat_weights: list[float] = []
    for name, value in state.items():
        if not hasattr(value, "detach"):
            continue
        tensor = value.detach().cpu().contiguous().float()
        start = len(flat_weights)
        values = tensor.reshape(-1).tolist()
        flat_weights.extend(float(v) for v in values)
        tensors.append({
            "name": name,
            "shape": list(tensor.shape),
            "dtype": "f32",
            "offset": start,
            "length": len(values),
        })

    manifest = {
        "model_id": args.model_id,
        "version": args.version,
        "description": "Room-specific PoseTCN checkpoint packaged from best_pose_tcn.pt",
        "format": "wifi-densepose-rvf",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    report = load_json(report_path)
    metrics = checkpoint.get("metrics") or report.get("best_metrics") or {}
    metadata = {
        "task": "wifi-csi-to-coco17-pose",
        "source_format": "pytorch_checkpoint",
        "source_checkpoint": str(checkpoint_path),
        "architecture": "PoseTCN",
        "input": {
            "in_channels": int(checkpoint.get("in_channels", 0) or 0),
            "time_steps": int(checkpoint.get("time_steps", 0) or 0),
        },
        "output": {
            "keypoints": 17,
            "dimensions": 2,
            "coordinate_space": "normalized_0_1",
        },
        "training": {
            "data_file": checkpoint.get("data_file") or report.get("data"),
            "best_epoch": int(checkpoint.get("epoch", report.get("best_epoch", 0)) or 0),
            "best_pck": metrics.get("pck20"),
            "best_metrics": metrics,
            "report": report,
        },
        "tensors": tensors,
        "parameter_count": len(flat_weights),
        "live_inference_note": (
            "RVF bundle contains trained PyTorch PoseTCN weights. The current live server "
            "must implement this architecture's inference adapter before output can be "
            "honestly labeled TRAINED POSE."
        ),
    }
    witness = {
        "training_hash": f"crc32:{zlib.crc32(checkpoint_path.read_bytes()) & 0xFFFF_FFFF:08x}",
        "metrics": metrics,
    }

    weight_payload = struct.pack(f"<{len(flat_weights)}f", *flat_weights)
    payloads = [
        (SEG_MANIFEST, json.dumps(manifest, separators=(",", ":")).encode("utf-8")),
        (SEG_VEC, weight_payload),
        (SEG_META, json.dumps(metadata, separators=(",", ":")).encode("utf-8")),
        (SEG_WITNESS, json.dumps(witness, separators=(",", ":")).encode("utf-8")),
    ]

    output = bytearray()
    for segment_id, (seg_type, payload) in enumerate(payloads):
        output.extend(segment(seg_type, segment_id, payload))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(output))
    print(f"Wrote {output_path} ({output_path.stat().st_size} bytes)")
    print(f"Model id: {args.model_id}")
    print(f"Parameters: {len(flat_weights)}")
    if metrics:
        print(f"Metrics: {json.dumps(metrics, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
