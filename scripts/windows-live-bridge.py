#!/usr/bin/env python3
"""Local Windows bridge for the live ESP32 CSI dashboard.

This intentionally uses only the Python standard library:

    ESP32 broadcast UDP :5005
        -> sensing WebSocket :3001 (/ws/sensing)
        -> pose WebSocket/HTTP/UI :3000

It is a lightweight fallback for machines where the Rust sensing-server
binary and Docker are unavailable. It does not modify network adapters,
routes, DHCP, DNS, or firewall configuration.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import mimetypes
import os
import socket
import struct
import threading
import time
from collections import deque
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


CSI_MAGIC = 0xC511_0001
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

KEYPOINT_OFFSETS = [
    (0.0, -82.0),
    (-8.0, -91.0),
    (8.0, -91.0),
    (-17.0, -84.0),
    (17.0, -84.0),
    (-34.0, -48.0),
    (34.0, -48.0),
    (-48.0, -12.0),
    (48.0, -12.0),
    (-53.0, 25.0),
    (53.0, 25.0),
    (-22.0, 24.0),
    (22.0, 24.0),
    (-24.0, 76.0),
    (24.0, 76.0),
    (-26.0, 130.0),
    (26.0, 130.0),
]

SEGMENT_MAGIC = 0x5256_4653
SEG_VEC = 0x01
SEG_MANIFEST = 0x05
SEG_META = 0x07
SEG_WITNESS = 0x0A


def resolve_project_path(path_value: str | os.PathLike[str]) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def safe_model_id(model_id: str) -> str:
    return Path(model_id).name.replace(".rvf", "").replace(".pt", "")


def read_rvf(path: Path) -> dict:
    """Read the small RVF segment format emitted by export_pose_tcn_to_rvf.py."""
    data = path.read_bytes()
    offset = 0
    segments: dict[int, bytes] = {}
    while offset + 64 <= len(data):
        magic = struct.unpack_from("<I", data, offset)[0]
        if magic != SEGMENT_MAGIC:
            raise ValueError(f"bad RVF segment magic at offset {offset}")
        seg_type = data[offset + 5]
        payload_len = struct.unpack_from("<Q", data, offset + 0x10)[0]
        pad_len = struct.unpack_from("<I", data, offset + 0x3C)[0]
        start = offset + 64
        end = start + payload_len
        if end > len(data):
            raise ValueError("truncated RVF segment payload")
        segments[seg_type] = data[start:end]
        offset = end + pad_len

    manifest = json.loads(segments.get(SEG_MANIFEST, b"{}").decode("utf-8"))
    metadata = json.loads(segments.get(SEG_META, b"{}").decode("utf-8"))
    witness = json.loads(segments.get(SEG_WITNESS, b"{}").decode("utf-8"))
    return {
        "manifest": manifest,
        "metadata": metadata,
        "witness": witness,
        "weights": segments.get(SEG_VEC, b""),
    }


def build_pose_tcn(torch_module):
    nn = torch_module.nn

    class ResidualTCNBlock(nn.Module):
        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            padding = dilation
            self.net = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
                nn.BatchNorm1d(channels),
            )
            self.act = nn.GELU()

        def forward(self, x):
            return self.act(x + self.net(x))

    class PoseTCN(nn.Module):
        def __init__(self, in_channels: int) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.GELU(),
            )
            self.tcn = nn.Sequential(
                ResidualTCNBlock(64, 1),
                ResidualTCNBlock(64, 2),
                ResidualTCNBlock(64, 4),
                nn.Conv1d(64, 128, kernel_size=1),
                nn.GELU(),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(128, 256),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(256, len(KEYPOINT_NAMES) * 2),
                nn.Sigmoid(),
            )

        def forward(self, x):
            z = self.head(self.tcn(self.stem(x)))
            return z.view(-1, len(KEYPOINT_NAMES), 2)

    return PoseTCN


class PoseModelRuntime:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.models_dir = resolve_project_path(os.environ.get("MODELS_DIR", "data/models"))
        self.checkpoint_root = resolve_project_path(os.environ.get("POSE_CHECKPOINT_DIR", "models"))
        self.active_model_id: str | None = None
        self.active_model_info: dict | None = None
        self.model = None
        self.torch = None
        self.device = None
        self.in_channels = 0
        self.time_steps = 0
        self.window: deque[list[float]] = deque()
        self.frames_processed = 0
        self.avg_inference_ms = 0.0
        self.last_error: str | None = None

    def _import_torch(self):
        if self.torch is not None:
            return self.torch
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "PyTorch is required for live PoseTCN inference. "
                "Start this bridge with .venv-gpu\\Scripts\\python.exe."
            ) from exc
        try:
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        except Exception:
            pass
        device_name = os.environ.get("RUVIEW_POSE_DEVICE", "cpu")
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
        self.torch = torch
        self.device = torch.device(device_name)
        return torch

    def list_models(self) -> list[dict]:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        found: dict[str, dict] = {}
        for path in sorted(self.models_dir.glob("*.rvf")):
            model_id = safe_model_id(path.name)
            info = {
                "id": model_id,
                "model_id": model_id,
                "name": model_id,
                "format": "rvf",
                "filename": path.name,
                "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "size_bytes": path.stat().st_size,
                "modified_epoch": int(path.stat().st_mtime),
            }
            try:
                rvf = read_rvf(path)
                metadata = rvf.get("metadata") or {}
                manifest = rvf.get("manifest") or {}
                metrics = (metadata.get("training") or {}).get("best_metrics") or {}
                info.update({
                    "version": manifest.get("version"),
                    "architecture": metadata.get("architecture"),
                    "pck_score": metrics.get("pck20"),
                    "pck50": metrics.get("pck50"),
                    "parameter_count": metadata.get("parameter_count"),
                })
            except Exception as exc:
                info["warning"] = f"metadata unreadable: {exc}"
            found[model_id] = info

        for ckpt in sorted(self.checkpoint_root.glob("*/best_pose_tcn.pt")):
            model_id = safe_model_id(ckpt.parent.name)
            if model_id in found:
                found[model_id]["checkpoint_path"] = str(ckpt.relative_to(ROOT))
                continue
            found[model_id] = {
                "id": model_id,
                "model_id": model_id,
                "name": model_id,
                "format": "pt",
                "filename": ckpt.name,
                "path": str(ckpt.relative_to(ROOT)) if ckpt.is_relative_to(ROOT) else str(ckpt),
                "size_bytes": ckpt.stat().st_size,
                "modified_epoch": int(ckpt.stat().st_mtime),
                "architecture": "PoseTCN",
            }
        return list(found.values())

    def get_model_info(self, model_id: str) -> dict | None:
        model_id = safe_model_id(model_id)
        for model in self.list_models():
            if model.get("id") == model_id or model.get("model_id") == model_id:
                return model
        return None

    def _state_from_rvf(self, path: Path, torch_module) -> tuple[dict, dict]:
        rvf = read_rvf(path)
        metadata = rvf["metadata"]
        input_meta = metadata.get("input") or {}
        in_channels = int(input_meta.get("in_channels") or 0)
        if in_channels <= 0:
            raise RuntimeError("RVF metadata does not include input.in_channels")
        model_cls = build_pose_tcn(torch_module)
        model = model_cls(in_channels)
        template = model.state_dict()
        weights = rvf.get("weights") or b""
        tensors = metadata.get("tensors") or []
        state = {}
        for item in tensors:
            name = item.get("name")
            if name not in template:
                continue
            offset = int(item.get("offset") or 0)
            length = int(item.get("length") or 0)
            start = offset * 4
            end = start + length * 4
            if length <= 0 or end > len(weights):
                raise RuntimeError(f"RVF tensor {name} has invalid weight slice")
            values = struct.unpack(f"<{length}f", weights[start:end])
            tensor = torch_module.tensor(values, dtype=template[name].dtype).reshape(template[name].shape)
            state[name] = tensor
        model.load_state_dict(state, strict=False)
        return {"model": model, "metadata": metadata}, rvf

    def _state_from_checkpoint(self, path: Path, torch_module) -> tuple[dict, dict]:
        checkpoint = torch_module.load(path, map_location="cpu", weights_only=False)
        in_channels = int(checkpoint.get("in_channels") or 0)
        if in_channels <= 0:
            raise RuntimeError("checkpoint does not include in_channels")
        model_cls = build_pose_tcn(torch_module)
        model = model_cls(in_channels)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        metadata = {
            "architecture": "PoseTCN",
            "input": {
                "in_channels": in_channels,
                "time_steps": int(checkpoint.get("time_steps") or 20),
            },
            "training": {"best_metrics": checkpoint.get("metrics") or {}},
        }
        return {"model": model, "metadata": metadata}, {"manifest": {}, "metadata": metadata}

    def load(self, model_id: str) -> dict:
        model_id = safe_model_id(model_id)
        if not model_id:
            raise RuntimeError("missing model id")
        torch_module = self._import_torch()
        rvf_path = self.models_dir / f"{model_id}.rvf"
        ckpt_path = self.checkpoint_root / model_id / "best_pose_tcn.pt"
        if rvf_path.exists():
            loaded, rvf = self._state_from_rvf(rvf_path, torch_module)
            source_path = rvf_path
        elif ckpt_path.exists():
            loaded, rvf = self._state_from_checkpoint(ckpt_path, torch_module)
            source_path = ckpt_path
        else:
            raise RuntimeError(f"model '{model_id}' was not found in {self.models_dir}")

        metadata = loaded["metadata"]
        model = loaded["model"].to(self.device)
        model.eval()
        input_meta = metadata.get("input") or {}
        in_channels = int(input_meta.get("in_channels") or 56)
        time_steps = int(input_meta.get("time_steps") or 20)
        metrics = (metadata.get("training") or {}).get("best_metrics") or {}
        info = self.get_model_info(model_id) or {}
        info.update({
            "id": model_id,
            "model_id": model_id,
            "name": info.get("name") or model_id,
            "loaded": True,
            "path": str(source_path.relative_to(ROOT)) if source_path.is_relative_to(ROOT) else str(source_path),
            "architecture": metadata.get("architecture") or "PoseTCN",
            "pck_score": info.get("pck_score", metrics.get("pck20")),
            "pck50": info.get("pck50", metrics.get("pck50")),
        })
        with self.lock:
            self.active_model_id = model_id
            self.active_model_info = info
            self.model = model
            self.in_channels = in_channels
            self.time_steps = time_steps
            self.window = deque(maxlen=time_steps)
            self.frames_processed = 0
            self.avg_inference_ms = 0.0
            self.last_error = None
        print(
            f"[bridge] loaded PoseTCN model '{model_id}' "
            f"({in_channels} channels x {time_steps} frames) from {source_path}",
            flush=True,
        )
        return info

    def unload(self) -> dict:
        with self.lock:
            previous = self.active_model_id
            self.active_model_id = None
            self.active_model_info = None
            self.model = None
            self.window.clear()
        return {"success": True, "previous": previous}

    def status(self) -> dict:
        with self.lock:
            info = dict(self.active_model_info or {})
            if info:
                info.update({
                    "loaded": True,
                    "avg_inference_ms": self.avg_inference_ms,
                    "frames_processed": self.frames_processed,
                    "input_window_frames": len(self.window),
                    "input_window_ready": len(self.window) >= self.time_steps if self.time_steps else False,
                    "last_error": self.last_error,
                })
            return info

    def active(self) -> dict | None:
        info = self.status()
        return info if info else None

    def _resample_amplitude(self, amplitude: list[float], expected: int) -> list[float]:
        if not amplitude:
            return []
        values = [float(v) for v in amplitude]
        if len(values) == expected:
            return values
        if len(values) == 1:
            return values * expected
        out = []
        scale = (len(values) - 1) / max(1, expected - 1)
        for index in range(expected):
            pos = index * scale
            left = int(math.floor(pos))
            right = min(len(values) - 1, left + 1)
            frac = pos - left
            out.append(values[left] * (1.0 - frac) + values[right] * frac)
        return out

    def infer_persons(self, update: dict) -> list[dict] | None:
        with self.lock:
            if self.model is None or self.torch is None or self.active_model_id is None:
                return None
            model = self.model
            torch_module = self.torch
            device = self.device
            in_channels = self.in_channels
            time_steps = self.time_steps

        nodes = update.get("nodes") or []
        amplitude = nodes[0].get("amplitude") if nodes and isinstance(nodes[0], dict) else None
        if not isinstance(amplitude, list):
            return None
        frame = self._resample_amplitude(amplitude, in_channels)
        if len(frame) != in_channels:
            return None

        with self.lock:
            self.window.append(frame)
            if len(self.window) < time_steps:
                return None
            window = list(self.window)

        try:
            start = time.perf_counter()
            x = torch_module.tensor(window, dtype=torch_module.float32, device=device).transpose(0, 1)
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True).clamp_min(1e-5)
            x = ((x - mean) / std).unsqueeze(0)
            with torch_module.no_grad():
                pred = model(x)[0].detach().cpu().tolist()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            with self.lock:
                self.frames_processed += 1
                if self.frames_processed == 1:
                    self.avg_inference_ms = elapsed_ms
                else:
                    self.avg_inference_ms = self.avg_inference_ms * 0.92 + elapsed_ms * 0.08
                self.last_error = None
        except Exception as exc:
            with self.lock:
                self.last_error = str(exc)
            return None

        source_confidence = clamp(
            float((update.get("classification") or {}).get("confidence", 0.75) or 0.75),
            0.0,
            1.0,
        )
        kp_conf = round(clamp(0.35 + source_confidence * 0.45, 0.35, 0.86), 3)
        keypoints = []
        for name, pair in zip(KEYPOINT_NAMES, pred):
            x = clamp(float(pair[0]), 0.0, 1.0)
            y = clamp(float(pair[1]), 0.0, 1.0)
            keypoints.append({
                "name": name,
                "x": round(x, 5),
                "y": round(y, 5),
                "z": 0.0,
                "confidence": kp_conf,
            })

        xs = [kp["x"] for kp in keypoints]
        ys = [kp["y"] for kp in keypoints]
        margin = 0.025
        bbox = {
            "x": round(clamp(min(xs) - margin, 0.0, 1.0), 5),
            "y": round(clamp(min(ys) - margin, 0.0, 1.0), 5),
            "width": round(clamp(max(xs) - min(xs) + margin * 2, 0.02, 1.0), 5),
            "height": round(clamp(max(ys) - min(ys) + margin * 2, 0.02, 1.0), 5),
        }

        return [{
            "id": 1,
            "confidence": round(clamp(0.45 + source_confidence * 0.42, 0.45, 0.9), 3),
            "keypoints": keypoints,
            "bbox": bbox,
            "zone": "zone_1",
            "position": signal_field_peak_position(update),
            "motion_score": round(float((update.get("features") or {}).get("motion_band_power", 0.0) or 0.0) * 100.0, 3),
            "model_id": self.active_model_id,
        }]


def signal_field_peak_position(update: dict) -> list[float]:
    field = update.get("signal_field") or {}
    grid_size = field.get("grid_size") or [20, 1, 20]
    values = field.get("values") or []
    if not values:
        return [0.0, 0.0, 0.0]
    try:
        nx = max(1, int(grid_size[0]))
        nz = max(1, int(grid_size[2]))
        peak_index = max(range(len(values)), key=lambda index: values[index])
        px = peak_index % nx
        pz = peak_index // nx
        world_x = (px / max(1, nx - 1)) * 4.0 - 2.0
        world_z = (pz / max(1, nz - 1)) * 4.0 - 2.0
        return [round(world_x, 3), 0.0, round(world_z, 3)]
    except (TypeError, ValueError):
        return [0.0, 0.0, 0.0]


def derive_heuristic_persons(update: dict) -> list[dict]:
    classification = update.get("classification") or {}
    features = update.get("features") or {}
    if not classification.get("presence", False):
        return []

    confidence = clamp(float(classification.get("confidence", 0.0) or 0.0), 0.0, 1.0)
    motion = clamp(float(features.get("motion_band_power", 0.0) or 0.0), 0.0, 1.0)
    variance = max(0.0, float(features.get("variance", 0.0) or 0.0))
    tick = float(update.get("tick", 0) or 0)
    phase = tick * 0.11

    center_x = 400.0 + math.sin(phase * 0.35) * 70.0 * motion
    center_y = 292.0 - motion * 16.0
    lean_x = math.sin(phase * 0.22) * (6.0 + motion * 16.0)
    stride = math.sin(phase) * motion
    jitter_scale = min(math.sqrt(variance) / 120.0, 1.0) * (0.8 + motion * 2.2)
    kp_confidence = clamp(confidence * 0.52, 0.22, 0.65)

    keypoints = []
    for index, (name, (dx, dy)) in enumerate(zip(KEYPOINT_NAMES, KEYPOINT_OFFSETS)):
        x = center_x + dx + lean_x * (0.15 if dy < 0 else 0.04)
        y = center_y + dy

        if index in (7, 9):
            y -= stride * 20.0
            x -= abs(stride) * 8.0
        elif index in (8, 10):
            y += stride * 20.0
            x += abs(stride) * 8.0
        elif index in (13, 15):
            y += stride * 26.0
            x += stride * 8.0
        elif index in (14, 16):
            y -= stride * 26.0
            x -= stride * 8.0

        x += math.sin(phase + index * 1.73) * jitter_scale
        y += math.cos(phase * 0.9 + index * 1.41) * jitter_scale

        keypoints.append({
            "name": name,
            "x": round(x, 3),
            "y": round(y, 3),
            "z": round(lean_x * 0.02, 4),
            "confidence": round(kp_confidence, 3),
        })

    xs = [kp["x"] for kp in keypoints]
    ys = [kp["y"] for kp in keypoints]
    min_x = min(xs) - 12.0
    max_x = max(xs) + 12.0
    min_y = min(ys) - 12.0
    max_y = max(ys) + 12.0

    return [{
        "id": 1,
        "confidence": round(clamp(confidence * 0.72, 0.3, 0.78), 3),
        "keypoints": keypoints,
        "bbox": {
            "x": round(min_x, 3),
            "y": round(min_y, 3),
            "width": round(max(80.0, max_x - min_x), 3),
            "height": round(max(160.0, max_y - min_y), 3),
        },
        "zone": "zone_1",
        "position": signal_field_peak_position(update),
        "motion_score": round(motion * 100.0, 3),
    }]


def pose_mode(persons: list[dict], model_keypoints_present: bool = False) -> str:
    if model_keypoints_present and persons:
        return "trained"
    return "heuristic" if persons else "none"


def pose_label(mode: str) -> str:
    if mode == "trained":
        return "TRAINED POSE - PoseTCN"
    if mode == "heuristic":
        return "HEURISTIC CSI POSE - NOT TRAINED"
    return "NO VALID POSE"


def pose_diagnostics(
    update: dict | None,
    persons: list[dict],
    model_loaded: bool = False,
    model_keypoints_present: bool = False,
    input_window_frames: int = 0,
    input_window_ready: bool = False,
) -> dict:
    total_keypoints = sum(len(person.get("keypoints") or []) for person in persons)
    drawable_keypoints = sum(
        1
        for person in persons
        for keypoint in (person.get("keypoints") or [])
        if isinstance(keypoint.get("x"), (int, float)) and isinstance(keypoint.get("y"), (int, float))
    )
    positive_confidence = sum(
        1
        for person in persons
        for keypoint in (person.get("keypoints") or [])
        if float(keypoint.get("confidence", 0.0) or 0.0) > 0.0
    )
    nodes = update.get("nodes") or [] if update else []
    mode = pose_mode(persons, model_keypoints_present)
    return {
        "pose_mode": mode,
        "pose_label": pose_label(mode),
        "model_loaded": model_loaded,
        "model_keypoints_present": model_keypoints_present,
        "active_nodes": len(nodes),
        "input_window_frames": input_window_frames,
        "input_window_ready": input_window_ready,
        "persons": len(persons),
        "total_keypoints": total_keypoints,
        "drawable_keypoints": drawable_keypoints,
        "positive_confidence_keypoints": positive_confidence,
        "mean_source_confidence": round(float((update or {}).get("classification", {}).get("confidence", 0.0) or 0.0), 3),
        "reason": (
            "trained_pose_tcn_keypoints"
            if model_keypoints_present
            else "model_window_warming"
            if model_loaded and not input_window_ready
            else "single_node_csi_heuristic_pose"
            if persons
            else "no_drawable_pose"
        ),
    }


def recv_exact(stream, length: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < length:
        try:
            chunk = stream.read(length - len(chunks))
        except (OSError, socket.timeout):
            return None
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def websocket_frame(payload: str | bytes, opcode: int = 0x1) -> bytes:
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    first = 0x80 | opcode
    length = len(data)
    if length <= 125:
        header = bytes((first, length))
    elif length <= 0xFFFF:
        header = bytes((first, 126)) + struct.pack("!H", length)
    else:
        header = bytes((first, 127)) + struct.pack("!Q", length)
    return header + data


def read_websocket_frame(stream) -> tuple[int, bytes] | None:
    header = recv_exact(stream, 2)
    if header is None:
        return None
    opcode = header[0] & 0x0F
    masked = bool(header[1] & 0x80)
    length = header[1] & 0x7F
    if length == 126:
        raw = recv_exact(stream, 2)
        if raw is None:
            return None
        length = struct.unpack("!H", raw)[0]
    elif length == 127:
        raw = recv_exact(stream, 8)
        if raw is None:
            return None
        length = struct.unpack("!Q", raw)[0]
    mask = recv_exact(stream, 4) if masked else None
    if masked and mask is None:
        return None
    payload = recv_exact(stream, length)
    if payload is None:
        return None
    if mask:
        payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    return opcode, payload


class WsClient:
    def __init__(self, connection: socket.socket, channel: str):
        self.connection = connection
        self.channel = channel
        self.lock = threading.Lock()

    def send(self, payload: dict | str | bytes, opcode: int = 0x1) -> bool:
        if isinstance(payload, dict):
            payload = json.dumps(payload, separators=(",", ":"))
        try:
            with self.lock:
                self.connection.sendall(websocket_frame(payload, opcode))
            return True
        except OSError:
            return False

    def close(self) -> None:
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.connection.close()
        except OSError:
            pass


class BridgeState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.started = time.time()
        self.tick = 0
        self.raw_packets = 0
        self.last_received = 0.0
        self.latest: dict | None = None
        self.last_mean_amplitude: float | None = None
        self.amplitude_history: deque[float] = deque(maxlen=60)
        self.model_runtime = PoseModelRuntime()
        self.clients: dict[str, set[WsClient]] = {
            "sensing": set(),
            "pose": set(),
            "events": set(),
        }

    def add_client(self, client: WsClient) -> None:
        with self.lock:
            self.clients[client.channel].add(client)

    def remove_client(self, client: WsClient) -> None:
        with self.lock:
            self.clients[client.channel].discard(client)

    def client_count(self) -> int:
        with self.lock:
            return sum(len(group) for group in self.clients.values())

    def broadcast(self, channel: str, payload: dict) -> None:
        with self.lock:
            clients = list(self.clients[channel])
        stale = [client for client in clients if not client.send(payload)]
        if stale:
            with self.lock:
                for client in stale:
                    self.clients[channel].discard(client)
                    client.close()

    def status(self) -> dict:
        with self.lock:
            age = None if not self.last_received else round(time.time() - self.last_received, 3)
            source = "esp32" if age is not None and age < 3.0 else "esp32:offline"
            return {
                "status": "ok",
                "source": source,
                "tick": self.tick,
                "raw_packets": self.raw_packets,
                "clients": self.client_count(),
                "last_frame_age_s": age,
                "uptime_s": round(time.time() - self.started, 1),
                "model": self.model_runtime.status(),
            }

    def apply_pose_estimation(self, update: dict) -> dict:
        model_status = self.model_runtime.status()
        model_loaded = bool(model_status.get("loaded"))
        model_persons = self.model_runtime.infer_persons(update) if model_loaded else None
        model_keypoints_present = bool(model_persons)
        if model_keypoints_present:
            persons = model_persons or []
            source = "model_inference"
        else:
            persons = derive_heuristic_persons(update)
            source = "signal_derived"

        model_status = self.model_runtime.status()
        mode = pose_mode(persons, model_keypoints_present)
        diagnostics = pose_diagnostics(
            update,
            persons,
            model_loaded=bool(model_status.get("loaded")),
            model_keypoints_present=model_keypoints_present,
            input_window_frames=int(model_status.get("input_window_frames") or 0),
            input_window_ready=bool(model_status.get("input_window_ready")),
        )
        update["persons"] = persons
        update["pose_source"] = source
        update["pose_mode"] = mode
        update["pose_label"] = pose_label(mode)
        update["diagnostics"] = diagnostics
        update["model"] = {
            "loaded": bool(model_status.get("loaded")),
            "active_model_id": model_status.get("model_id"),
            "trained_keypoints_present": model_keypoints_present,
            "input_window_frames": model_status.get("input_window_frames", 0),
            "input_window_ready": model_status.get("input_window_ready", False),
            "avg_inference_ms": model_status.get("avg_inference_ms"),
            "frames_processed": model_status.get("frames_processed"),
            "last_error": model_status.get("last_error"),
        }
        return update

    def update_from_sensing(self, source_update: dict) -> dict | None:
        if not isinstance(source_update, dict):
            return None
        update = json.loads(json.dumps(source_update))
        if update.get("type") and update.get("type") != "sensing_update":
            return None
        now = time.time()
        with self.lock:
            self.raw_packets += 1
            self.tick += 1
            self.last_received = now
            update["type"] = "sensing_update"
            update.setdefault("timestamp", now)
            update.setdefault("source", source_update.get("source") or "upstream")
            update["tick"] = self.tick
        update = self.apply_pose_estimation(update)
        with self.lock:
            self.latest = update
        return update

    def update_csi(self, packet: bytes, peer: tuple[str, int]) -> dict | None:
        if len(packet) < 20:
            return None
        magic = struct.unpack_from("<I", packet, 0)[0]
        if magic != CSI_MAGIC:
            return None

        node_id = packet[4]
        n_antennas = packet[5]
        n_subcarriers = struct.unpack_from("<H", packet, 6)[0]
        freq_mhz = struct.unpack_from("<I", packet, 8)[0]
        sequence = struct.unpack_from("<I", packet, 12)[0]
        rssi = struct.unpack_from("<b", packet, 16)[0]
        noise_floor = struct.unpack_from("<b", packet, 17)[0]
        pair_count = n_antennas * n_subcarriers
        available_pairs = max(0, (len(packet) - 20) // 2)
        pair_count = min(pair_count, available_pairs)
        if pair_count <= 0:
            return None

        amplitudes: list[float] = []
        for index in range(pair_count):
            i_value, q_value = struct.unpack_from("<bb", packet, 20 + index * 2)
            amplitudes.append(math.hypot(i_value, q_value))

        mean_amp = sum(amplitudes) / len(amplitudes)
        variance = sum((value - mean_amp) ** 2 for value in amplitudes) / len(amplitudes)
        previous = self.last_mean_amplitude
        delta = 0.0 if previous is None else abs(mean_amp - previous)
        self.last_mean_amplitude = mean_amp
        self.amplitude_history.append(mean_amp)

        history_mean = sum(self.amplitude_history) / len(self.amplitude_history)
        temporal_variance = sum(
            (value - history_mean) ** 2 for value in self.amplitude_history
        ) / max(1, len(self.amplitude_history))
        motion_score = clamp(delta / 8.0 + math.sqrt(temporal_variance) / 12.0, 0.0, 1.0)
        confidence = clamp(0.65 + min(len(self.amplitude_history), 30) / 100.0, 0.65, 0.95)
        motion_level = "active" if motion_score > 0.18 else "present_still"

        grid_size = 20
        center = (grid_size - 1) / 2.0
        signal_values = []
        for z_pos in range(grid_size):
            for x_pos in range(grid_size):
                dx = x_pos - center
                dz = z_pos - center
                distance_sq = dx * dx + dz * dz
                base = math.exp(-distance_sq / 55.0) * 0.28
                motion_blob = math.exp(
                    -((dx - math.sin(sequence / 18.0) * 2.5) ** 2 + dz * dz) / 16.0
                ) * (0.2 + motion_score * 0.7)
                signal_values.append(clamp(base + motion_blob, 0.0, 1.0))

        now = time.time()
        with self.lock:
            self.raw_packets += 1
            self.tick += 1
            self.last_received = now
            update = {
                "type": "sensing_update",
                "timestamp": now,
                "source": "esp32",
                "tick": self.tick,
                "pose_source": "signal_derived",
                "nodes": [{
                    "node_id": node_id,
                    "rssi_dbm": rssi,
                    "position": [2.0, 0.0, 1.5],
                    "amplitude": [round(value, 4) for value in amplitudes],
                    "subcarrier_count": n_subcarriers,
                    "frequency_mhz": freq_mhz,
                    "sequence": sequence,
                    "noise_floor_dbm": noise_floor,
                    "peer_ip": peer[0],
                }],
                "node_features": [{
                    "node_id": node_id,
                    "rssi_dbm": rssi,
                    "frame_rate_hz": 0.0,
                    "last_seen_ms": 0,
                    "stale": False,
                    "features": {
                        "mean_rssi": rssi,
                        "variance": variance,
                        "motion_band_power": motion_score,
                    },
                    "classification": {
                        "presence": True,
                        "motion_level": motion_level,
                        "confidence": confidence,
                    },
                }],
                "features": {
                    "mean_rssi": rssi,
                    "variance": variance,
                    "std": math.sqrt(variance),
                    "motion_band_power": motion_score,
                    "breathing_band_power": 0.0,
                    "dominant_freq_hz": 0.0,
                    "change_points": 1 if motion_score > 0.25 else 0,
                    "spectral_power": variance,
                    "range": max(amplitudes) - min(amplitudes),
                },
                "classification": {
                    "presence": True,
                    "motion_level": motion_level,
                    "confidence": confidence,
                },
                "estimated_persons": 1,
                "signal_field": {
                    "grid_size": [grid_size, 1, grid_size],
                    "values": signal_values,
                },
            }
        update = self.apply_pose_estimation(update)
        with self.lock:
            self.latest = update
        return update


STATE = BridgeState()


def pose_message(update: dict) -> dict:
    classification = update.get("classification", {})
    persons = update.get("persons") or derive_heuristic_persons(update)
    source = update.get("pose_source") or "signal_derived"
    model = update.get("model") or {}
    model_keypoints_present = bool(model.get("trained_keypoints_present")) or source == "model_inference"
    mode = update.get("pose_mode") or pose_mode(persons, model_keypoints_present)
    diagnostics = update.get("diagnostics") or pose_diagnostics(
        update,
        persons,
        model_loaded=bool(model.get("loaded")),
        model_keypoints_present=model_keypoints_present,
        input_window_frames=int(model.get("input_window_frames") or 0),
        input_window_ready=bool(model.get("input_window_ready")),
    )
    note = (
        "Trained PoseTCN model inference from live CSI window."
        if model_keypoints_present
        else "Heuristic CSI skeleton for visualization until a trained model window is ready."
    )
    return {
        "type": "pose_data",
        "zone_id": "zone_1",
        "timestamp": utc_now(),
        "pose_source": source,
        "pose_mode": mode,
        "pose_label": update.get("pose_label") or pose_label(mode),
        "diagnostics": diagnostics,
        "data": {
            "pose": {"persons": persons},
            "pose_source": source,
            "pose_mode": mode,
            "pose_label": update.get("pose_label") or pose_label(mode),
            "diagnostics": diagnostics,
            "confidence": classification.get("confidence", 0.0),
            "activity": classification.get("motion_level", "unknown"),
            "metadata": {
                "frame_id": f"esp32_{update.get('tick', 0)}",
                "processing_time_ms": model.get("avg_inference_ms") or 0,
                "note": note,
            },
        },
    }


class UdpReceiver(threading.Thread):
    def __init__(self, host: str, port: int):
        super().__init__(name="udp-csi", daemon=True)
        self.host = host
        self.port = port

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.host, self.port))
        except OSError as exc:
            print(f"[bridge] UDP CSI bind failed on {self.host}:{self.port}: {exc}", flush=True)
            return
        print(f"[bridge] UDP CSI listening on {self.host}:{self.port}", flush=True)
        while True:
            packet, peer = sock.recvfrom(8192)
            update = STATE.update_csi(packet, peer)
            if update is None:
                continue
            STATE.broadcast("sensing", update)
            STATE.broadcast("pose", pose_message(update))
            tick = update["tick"]
            if tick <= 3 or tick % 100 == 0:
                node = update["nodes"][0]
                print(
                    f"[bridge] frame={tick} node={node['node_id']} "
                    f"rssi={node['rssi_dbm']} subcarriers={node['subcarrier_count']} "
                    f"from={peer[0]}",
                    flush=True,
                )


class UpstreamSensingClient(threading.Thread):
    def __init__(self, url: str):
        super().__init__(name="upstream-sensing", daemon=True)
        self.url = url
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._run_once()
            except Exception as exc:
                print(f"[bridge] upstream sensing disconnected: {exc}", flush=True)
            self.stop_event.wait(2.0)

    def _run_once(self) -> None:
        parsed = urlparse(self.url)
        if parsed.scheme not in {"ws", ""}:
            raise RuntimeError("only ws:// upstream sensing URLs are supported")
        host = parsed.hostname or "localhost"
        port = parsed.port or 80
        path = parsed.path or "/ws/sensing"
        if parsed.query:
            path += "?" + parsed.query

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        sock = socket.create_connection((host, port), timeout=5.0)
        try:
            request = (
                f"GET {path} HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n"
                "\r\n"
            ).encode("ascii")
            sock.sendall(request)
            stream = sock.makefile("rwb", buffering=0)
            header_bytes = bytearray()
            while b"\r\n\r\n" not in header_bytes and len(header_bytes) < 8192:
                chunk = stream.read(1)
                if not chunk:
                    raise RuntimeError("upstream closed during handshake")
                header_bytes.extend(chunk)
            header_text = header_bytes.decode("iso-8859-1", errors="replace")
            if " 101 " not in header_text.split("\r\n", 1)[0]:
                raise RuntimeError(header_text.split("\r\n", 1)[0])

            print(f"[bridge] upstream sensing connected: {self.url}", flush=True)
            while not self.stop_event.is_set():
                frame = read_websocket_frame(stream)
                if frame is None:
                    raise RuntimeError("no upstream frame")
                opcode, payload = frame
                if opcode == 0x8:
                    raise RuntimeError("upstream sent close")
                if opcode != 0x1:
                    continue
                try:
                    data = json.loads(payload.decode("utf-8"))
                except Exception:
                    continue
                update = STATE.update_from_sensing(data)
                if update is None:
                    continue
                STATE.broadcast("sensing", update)
                STATE.broadcast("pose", pose_message(update))
        finally:
            try:
                sock.close()
            except OSError:
                pass


class BridgeHandler(SimpleHTTPRequestHandler):
    server_version = "RuViewWindowsBridge/1.0"

    def log_message(self, fmt: str, *args) -> None:
        if self.path.startswith("/health") or self.path.startswith("/api/"):
            print(f"[http] {self.address_string()} {fmt % args}", flush=True)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def json_response(self, body: dict, status: int = HTTPStatus.OK) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/v1/models/load":
            try:
                body = self.read_json_body()
                model_id = body.get("id") or body.get("model_id") or ""
                info = STATE.model_runtime.load(str(model_id))
                self.json_response({"success": True, "model_id": info["model_id"], "model": info})
            except Exception as exc:
                self.json_response({"success": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/v1/models/unload":
            self.json_response(STATE.model_runtime.unload())
            return
        if path == "/api/v1/models/lora/activate":
            self.json_response({
                "success": False,
                "error": "LoRA profiles are not supported by windows-live-bridge.",
            }, HTTPStatus.NOT_IMPLEMENTED)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if self.headers.get("Upgrade", "").lower() == "websocket":
            self.handle_websocket(path)
            return

        if path == "/":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/ui/index.html#demo")
            self.end_headers()
            return

        if path in {"/health", "/api/v1/status"}:
            self.json_response(STATE.status())
            return
        if path in {"/health/live", "/health/ready"}:
            self.json_response({"status": "healthy", **STATE.status()})
            return
        if path == "/health/health":
            status = STATE.status()
            healthy = status["source"] == "esp32"
            self.json_response({
                "status": "healthy" if healthy else "degraded",
                "timestamp": utc_now(),
                "components": {
                    "api": {"status": "healthy"},
                    "hardware": {"status": "healthy" if healthy else "degraded"},
                    "streaming": {"status": "healthy"},
                    "inference": {
                        "status": "degraded",
                        "message": "Signal-derived mode; no full-pose model loaded",
                    },
                },
            })
            return
        if path == "/health/version":
            self.json_response({"version": "windows-live-bridge-1.0"})
            return
        if path == "/health/metrics":
            self.json_response(STATE.status())
            return
        if path == "/api/v1/info":
            self.json_response({
                "name": "RuView Windows Live Bridge",
                "version": "1.0",
                "source": "esp32",
            })
            return
        if path == "/api/v1/models":
            models = STATE.model_runtime.list_models()
            self.json_response({"models": models, "total": len(models)})
            return
        if path == "/api/v1/models/active":
            active = STATE.model_runtime.active()
            self.json_response({
                "active": active,
                "model_id": active.get("model_id") if active else None,
                "loaded": bool(active),
            })
            return
        if path == "/api/v1/models/lora/profiles":
            self.json_response({"profiles": []})
            return
        if path.startswith("/api/v1/models/"):
            model_id = path.rsplit("/", 1)[-1]
            info = STATE.model_runtime.get_model_info(model_id)
            if info is None:
                self.json_response({"error": "model not found"}, HTTPStatus.NOT_FOUND)
            else:
                active = STATE.model_runtime.active()
                if active and active.get("model_id") == info.get("model_id"):
                    info = {**info, **active}
                self.json_response(info)
            return
        if path == "/api/v1/sensing/latest":
            with STATE.lock:
                latest = STATE.latest
            if latest is None:
                self.json_response({"status": "no data yet"}, HTTPStatus.SERVICE_UNAVAILABLE)
            else:
                self.json_response(latest)
            return
        if path in {"/api/v1/stream/status", "/api/v1/stream/metrics"}:
            self.json_response(STATE.status())
            return
        if path == "/api/v1/pose/current":
            with STATE.lock:
                latest = STATE.latest
            persons = (latest or {}).get("persons") or ([] if latest is None else derive_heuristic_persons(latest))
            model = (latest or {}).get("model") or {}
            model_keypoints_present = bool(model.get("trained_keypoints_present"))
            mode = (latest or {}).get("pose_mode") or pose_mode(persons, model_keypoints_present)
            source = (latest or {}).get("pose_source") or "signal_derived"
            self.json_response({
                "timestamp": utc_now(),
                "persons": persons,
                "total_persons": len(persons),
                "zone_summary": {"zone_1": len(persons)} if persons else {},
                "pose_source": source,
                "pose_mode": mode,
                "pose_label": (latest or {}).get("pose_label") or pose_label(mode),
                "diagnostics": (latest or {}).get("diagnostics") or pose_diagnostics(latest, persons),
                "model": model,
                "warning": None if model_keypoints_present else ("HEURISTIC CSI POSE - NOT TRAINED" if persons else None),
                "source": (latest or {}).get("source", "esp32"),
            })
            return

        super().do_GET()

    def handle_websocket(self, path: str) -> None:
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing Sec-WebSocket-Key")
            return
        if path == "/ws/sensing":
            channel = "sensing"
        elif path == "/api/v1/stream/pose":
            channel = "pose"
        elif path == "/api/v1/stream/events":
            channel = "events"
        else:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        accept = base64.b64encode(
            hashlib.sha1((key + WS_GUID).encode("ascii")).digest()
        ).decode("ascii")
        self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()

        client = WsClient(self.connection, channel)
        STATE.add_client(client)
        if channel == "pose":
            client.send({"type": "connection_established", "timestamp": utc_now()})
        elif channel == "events":
            client.send({"type": "connection_established", "timestamp": utc_now()})
        else:
            with STATE.lock:
                latest = STATE.latest
            if latest is not None:
                client.send(latest)

        try:
            self.connection.settimeout(5.0)
            while True:
                frame = read_websocket_frame(self.rfile)
                if frame is None:
                    continue
                opcode, payload = frame
                if opcode == 0x8:
                    break
                if opcode == 0x9:
                    if not client.send(payload, opcode=0xA):
                        break
                elif opcode == 0x1 and payload == b"ping":
                    if not client.send("pong"):
                        break
        except (OSError, ValueError):
            pass
        finally:
            STATE.remove_client(client)
            client.close()
            self.close_connection = True


def start_http_server(host: str, port: int, label: str) -> ThreadingHTTPServer:
    mimetypes.add_type("application/javascript", ".js")
    handler = partial(BridgeHandler, directory=str(ROOT))
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(
        target=server.serve_forever,
        name=f"http-{port}",
        daemon=True,
    )
    thread.start()
    print(f"[bridge] {label} listening on http://{host}:{port}", flush=True)
    return server


def main() -> int:
    parser = argparse.ArgumentParser(description="RuView ESP32 Windows live bridge")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=3000)
    parser.add_argument("--ws-port", type=int, default=3001)
    parser.add_argument("--udp-port", type=int, default=5005)
    parser.add_argument("--no-udp", action="store_true", help="Do not bind the ESP32 UDP CSI port")
    parser.add_argument(
        "--upstream-sensing-url",
        default="",
        help="Optional ws://.../ws/sensing source to consume instead of, or in addition to, UDP",
    )
    parser.add_argument("--model", default="", help="Model id to load on startup, e.g. room-pose-gpu")
    args = parser.parse_args()

    if args.model:
        try:
            STATE.model_runtime.load(args.model)
        except Exception as exc:
            print(f"[bridge] model load failed: {exc}", flush=True)

    if not args.no_udp:
        UdpReceiver(args.bind, args.udp_port).start()
    if args.upstream_sensing_url:
        UpstreamSensingClient(args.upstream_sensing_url).start()
    http_server = start_http_server(args.bind, args.http_port, "UI/API/pose WebSocket")
    ws_server = start_http_server(args.bind, args.ws_port, "sensing WebSocket")
    print(
        f"[bridge] open http://localhost:{args.http_port}/ui/index.html#demo",
        flush=True,
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[bridge] stopping", flush=True)
        http_server.shutdown()
        ws_server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
