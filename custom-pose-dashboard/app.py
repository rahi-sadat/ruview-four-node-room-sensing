from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from torch import nn

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

MODEL_PATH = Path(os.getenv(
    "POSE_MODEL",
    str(PROJECT_ROOT / "models" / "room-pose-gpu" / "best_pose_tcn.pt"),
))
REPLAY_FILE = Path(os.getenv(
    "POSE_REPLAY_FILE",
    str(PROJECT_ROOT / "models" / "room-pose-gpu" / "eval_predictions.jsonl"),
))
TRAINING_RECORD = Path(os.getenv(
    "POSE_TRAINING_RECORD",
    str(PROJECT_ROOT / "data" / "recordings" / "rec_1783755595.jsonl"),
))
RUView_WS_URL = os.getenv("RUVIEW_WS_URL", "ws://localhost:3000/ws/sensing")
DASHBOARD_MODE = os.getenv("DASHBOARD_MODE", "live").lower()
EXPECTED_SUBCARRIERS = int(os.getenv("EXPECTED_SUBCARRIERS", "56"))
WINDOW_FRAMES = int(os.getenv("WINDOW_FRAMES", "20"))
EXPECTED_NODE_COUNT = int(os.getenv("EXPECTED_NODE_COUNT", "4"))
POSE_SMOOTHING = float(os.getenv("POSE_SMOOTHING", "0.85"))
ORDER_SCAN_UPDATES = int(os.getenv("ORDER_SCAN_UPDATES", "2000"))

COCO_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def detect_training_node_order(path: Path) -> tuple[list[int], str | None]:
    """Detect the modal ordered node tuple used in the original recording.

    align-ground-truth.js preserved the nodes[] order from each sensing_update,
    then flattened one amplitude frame per node. Reusing that order is therefore
    required for this already-trained checkpoint.
    """
    if not path.exists():
        return list(range(1, EXPECTED_NODE_COUNT + 1)), f"Training recording not found: {path}"

    counts: Counter[tuple[int, ...]] = Counter()
    accepted = 0
    malformed = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if '"sensing_update"' not in line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if obj.get("type") != "sensing_update":
                    continue
                ordered: list[int] = []
                for node in obj.get("nodes", []):
                    amp = node.get("amplitude")
                    node_id = node.get("node_id")
                    if isinstance(node_id, int) and isinstance(amp, list) and len(amp) == EXPECTED_SUBCARRIERS:
                        ordered.append(node_id)
                if len(ordered) == EXPECTED_NODE_COUNT and len(set(ordered)) == EXPECTED_NODE_COUNT:
                    counts[tuple(ordered)] += 1
                    accepted += 1
                    if accepted >= ORDER_SCAN_UPDATES:
                        break
    except OSError as exc:
        return list(range(1, EXPECTED_NODE_COUNT + 1)), f"Could not scan training recording: {exc}"

    if not counts:
        return list(range(1, EXPECTED_NODE_COUNT + 1)), (
            f"No full {EXPECTED_NODE_COUNT}-node/{EXPECTED_SUBCARRIERS}-subcarrier updates found "
            f"while scanning {path}"
        )

    order, count = counts.most_common(1)[0]
    warning = None
    if malformed:
        warning = f"Ignored {malformed} malformed recording lines while detecting node order."
    return list(order), warning


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            nn.Linear(256, 17 * 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.tcn(self.stem(x))).view(-1, 17, 2)


class PoseRuntime:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: PoseTCN | None = None
        self.model_error: str | None = None
        self.model_metrics: dict[str, Any] = {}
        self.training_node_order, self.order_warning = detect_training_node_order(TRAINING_RECORD)
        self.csi_buffer: list[np.ndarray] = []
        self.smoothed_pose: np.ndarray | None = None
        self.node_state: dict[str, dict[str, Any]] = {}
        self.pose_history: deque[tuple[float, float, float]] = deque(maxlen=40)
        self.clients: set[WebSocket] = set()
        self.ws_connected = False
        self.total_valid_frames = 0
        self.total_dropped_frames = 0
        self.accepted_updates = 0
        self.rejected_updates = 0
        self.last_live_order: list[int] = []
        self.last_fall_like = False
        self.last_fall_reason = "No pose yet"
        self.last_state: dict[str, Any] = {}
        self.load_model()
        self.last_state = self.build_state(source="initializing", warning=self.order_warning)

    def load_model(self) -> None:
        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")
            checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
            in_channels = int(checkpoint.get("in_channels", EXPECTED_SUBCARRIERS))
            time_steps = int(checkpoint.get("time_steps", WINDOW_FRAMES))
            if in_channels != EXPECTED_SUBCARRIERS:
                raise ValueError(f"Model expects {in_channels} subcarriers; configured {EXPECTED_SUBCARRIERS}.")
            if time_steps != WINDOW_FRAMES:
                raise ValueError(f"Model expects {time_steps} frames; configured {WINDOW_FRAMES}.")
            model = PoseTCN(in_channels)
            model.load_state_dict(checkpoint["model_state"])
            model.to(self.device).eval()
            self.model = model
            self.model_metrics = checkpoint.get("metrics", {})
        except Exception as exc:
            self.model_error = str(exc)

    def update_nodes(self, nodes: list[dict[str, Any]]) -> None:
        now = time.time()
        for node in nodes:
            node_id = str(node.get("node_id", "?"))
            amp = node.get("amplitude")
            self.node_state[node_id] = {
                "node_id": node.get("node_id"),
                "rssi_dbm": node.get("rssi_dbm", node.get("rssi")),
                "subcarrier_count": len(amp) if isinstance(amp, list) else node.get("subcarrier_count"),
                "last_seen": now,
                "online": True,
            }

    def append_sensing_update(self, payload: dict[str, Any]) -> bool:
        nodes = payload.get("nodes", [])
        if not isinstance(nodes, list):
            self.rejected_updates += 1
            return False
        self.update_nodes(nodes)

        valid_by_id: dict[int, np.ndarray] = {}
        raw_order: list[int] = []
        for node in nodes:
            node_id = node.get("node_id")
            amp = node.get("amplitude")
            if not isinstance(node_id, int) or not isinstance(amp, list) or len(amp) != EXPECTED_SUBCARRIERS:
                continue
            arr = np.asarray(amp, dtype=np.float32)
            if not np.isfinite(arr).all():
                continue
            valid_by_id[node_id] = arr
            raw_order.append(node_id)

        self.last_live_order = raw_order
        required = self.training_node_order
        if len(required) != EXPECTED_NODE_COUNT or any(node_id not in valid_by_id for node_id in required):
            self.rejected_updates += 1
            self.total_dropped_frames += max(1, EXPECTED_NODE_COUNT - len(valid_by_id))
            self.csi_buffer.clear()  # prevent phase mixing after an incomplete update
            return False

        # Critical fix: reproduce the original recording's modal node order,
        # rather than trusting the current Rust HashMap iteration order.
        for node_id in required:
            self.csi_buffer.append(valid_by_id[node_id])
            self.total_valid_frames += 1
        self.accepted_updates += 1

        # Training used non-overlapping groups of 20 flattened frames.
        # Never use a rolling window or infer at a different phase.
        if len(self.csi_buffer) > WINDOW_FRAMES:
            self.csi_buffer = self.csi_buffer[-WINDOW_FRAMES:]
        return True

    def preprocess_current_window(self) -> torch.Tensor:
        x = np.stack(self.csi_buffer, axis=0)  # [20, 56]
        x = x.T                              # [56, 20]
        mean = x.mean(axis=1, keepdims=True)
        std = np.maximum(x.std(axis=1, keepdims=True), 1e-5)
        x = (x - mean) / std
        return torch.from_numpy(x).unsqueeze(0).to(self.device)

    def infer_if_ready(self) -> tuple[np.ndarray | None, float | None]:
        if self.model is None or len(self.csi_buffer) != WINDOW_FRAMES:
            return None, None
        start = time.perf_counter()
        with torch.inference_mode():
            pred = self.model(self.preprocess_current_window()).detach().float().cpu().numpy()[0]
        latency_ms = (time.perf_counter() - start) * 1000.0
        self.csi_buffer.clear()  # exact non-overlapping alignment behavior
        pred = np.clip(pred, 0.0, 1.0)

        if self.smoothed_pose is None:
            self.smoothed_pose = pred
        else:
            a = POSE_SMOOTHING
            self.smoothed_pose = a * self.smoothed_pose + (1.0 - a) * pred
        self.last_fall_like, self.last_fall_reason = self.detect_fall_like(self.smoothed_pose)
        return self.smoothed_pose.copy(), latency_ms

    def detect_fall_like(self, pose: np.ndarray) -> tuple[bool, str]:
        now = time.time()
        shoulder = 0.5 * (pose[5] + pose[6])
        hip = 0.5 * (pose[11] + pose[12])
        torso = hip - shoulder
        torso_norm = float(np.linalg.norm(torso)) + 1e-6
        verticality = abs(float(torso[1])) / torso_norm
        hip_y = float(hip[1])
        self.pose_history.append((now, hip_y, verticality))
        recent = [item for item in self.pose_history if now - item[0] <= 1.5]
        if len(recent) < 4:
            return False, "Collecting pose history"
        downward_change = hip_y - min(item[1] for item in recent)
        if downward_change > 0.12 and verticality < 0.60:
            return True, f"Fall-like: hip Δ={downward_change:.2f}, torso verticality={verticality:.2f}"
        return False, f"No fall-like event: hip Δ={downward_change:.2f}, torso verticality={verticality:.2f}"

    def build_state(
        self,
        *,
        source: str,
        pose: np.ndarray | None = None,
        latency_ms: float | None = None,
        warning: str | None = None,
    ) -> dict[str, Any]:
        pose_to_show = pose if pose is not None else self.smoothed_pose
        pose_out: list[dict[str, Any]] = []
        if pose_to_show is not None:
            pose_out = [
                {"name": COCO_NAMES[i], "x": float(pose_to_show[i, 0]), "y": float(pose_to_show[i, 1])}
                for i in range(17)
            ]

        now = time.time()
        nodes = []
        for state in sorted(
            self.node_state.values(),
            key=lambda item: int(item["node_id"]) if str(item["node_id"]).isdigit() else 999,
        ):
            item = dict(state)
            item["online"] = now - float(item["last_seen"]) < 3.0
            item["age_ms"] = int((now - float(item["last_seen"])) * 1000)
            nodes.append(item)

        dynamic_warning = warning or self.order_warning
        if self.last_live_order and set(self.training_node_order) != set(self.last_live_order):
            dynamic_warning = (
                f"Incomplete/mismatched live nodes. Training order={self.training_node_order}; "
                f"live valid order={self.last_live_order}."
            )

        return {
            "mode": DASHBOARD_MODE,
            "source": source,
            "ruview_connected": self.ws_connected,
            "model_loaded": self.model is not None,
            "model_error": self.model_error,
            "model_metrics": self.model_metrics,
            "device": str(self.device),
            "model_path": str(MODEL_PATH),
            "training_record": str(TRAINING_RECORD),
            "training_node_order": self.training_node_order,
            "live_node_order": self.last_live_order,
            "nodes": nodes,
            "buffer_frames": len(self.csi_buffer),
            "window_frames": WINDOW_FRAMES,
            "subcarriers": EXPECTED_SUBCARRIERS,
            "pose": pose_out,
            "fall_like": self.last_fall_like,
            "fall_reason": self.last_fall_reason,
            "latency_ms": latency_ms,
            "valid_frames": self.total_valid_frames,
            "dropped_frames": self.total_dropped_frames,
            "accepted_updates": self.accepted_updates,
            "rejected_updates": self.rejected_updates,
            "updated_at": now,
            "warning": dynamic_warning,
        }

    async def publish(self, state: dict[str, Any]) -> None:
        self.last_state = state
        if not self.clients:
            return
        text = json.dumps(state)
        dead: list[WebSocket] = []
        for client in self.clients:
            try:
                await client.send_text(text)
            except Exception:
                dead.append(client)
        for client in dead:
            self.clients.discard(client)


runtime = PoseRuntime()
app = FastAPI(title="RuView Custom Pose Dashboard v2")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/state")
async def api_state() -> JSONResponse:
    return JSONResponse(runtime.last_state)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if runtime.model is not None else "degraded",
        "mode": DASHBOARD_MODE,
        "model_loaded": runtime.model is not None,
        "model_error": runtime.model_error,
        "ruview_connected": runtime.ws_connected,
        "training_node_order": runtime.training_node_order,
        "live_node_order": runtime.last_live_order,
        "accepted_updates": runtime.accepted_updates,
        "rejected_updates": runtime.rejected_updates,
    }


@app.websocket("/ws")
async def dashboard_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    runtime.clients.add(websocket)
    await websocket.send_text(json.dumps(runtime.last_state))
    try:
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        runtime.clients.discard(websocket)


async def live_loop() -> None:
    while True:
        try:
            async with websockets.connect(
                RUView_WS_URL,
                ping_interval=20,
                ping_timeout=20,
                max_size=16 * 1024 * 1024,
            ) as ws:
                runtime.ws_connected = True
                await runtime.publish(runtime.build_state(source="live"))
                async for message in ws:
                    payload = json.loads(message)
                    if payload.get("type") != "sensing_update":
                        continue
                    runtime.append_sensing_update(payload)
                    pose, latency = runtime.infer_if_ready()
                    await runtime.publish(runtime.build_state(source="live", pose=pose, latency_ms=latency))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            runtime.ws_connected = False
            runtime.csi_buffer.clear()
            await runtime.publish(runtime.build_state(
                source="live",
                warning=f"RuView WebSocket unavailable: {exc}",
            ))
            await asyncio.sleep(2.0)


async def replay_loop() -> None:
    runtime.ws_connected = False
    if not REPLAY_FILE.exists():
        await runtime.publish(runtime.build_state(
            source="replay",
            warning=f"Replay file not found: {REPLAY_FILE}",
        ))
        return
    while True:
        try:
            with REPLAY_FILE.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    pred = np.asarray(record["predicted"], dtype=np.float32)
                    runtime.smoothed_pose = pred
                    await runtime.publish(runtime.build_state(
                        source="replay",
                        pose=pred,
                        latency_ms=0.0,
                        warning="Replay mode: held-out predictions, not live CSI.",
                    ))
                    await asyncio.sleep(0.10)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await runtime.publish(runtime.build_state(source="replay", warning=f"Replay error: {exc}"))
            await asyncio.sleep(2.0)


@app.on_event("startup")
async def startup() -> None:
    app.state.worker = asyncio.create_task(replay_loop() if DASHBOARD_MODE == "replay" else live_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    worker = getattr(app.state, "worker", None)
    if worker:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
