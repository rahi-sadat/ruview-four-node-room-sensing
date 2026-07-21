from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import urllib.error
import urllib.request
import statistics
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import websockets
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
STATIC_DIR = APP_DIR / "static"
CONFIG_PATH = APP_DIR / "config.json"
MODEL_PATH = DATA_DIR / "zone_model.joblib"
CALIBRATION_PATH = DATA_DIR / "calibration.json"
LOG_PATH = DATA_DIR / "events.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CONFIG = json.load(f)

RUVIEW_WS_CANDIDATES = CONFIG.get(
    "ruview_websockets",
    ["ws://localhost:3000/ws/sensing", "ws://localhost:3001/ws/sensing"],
)
RUVIEW_BASE_URL = str(
    os.getenv("RUVIEW_BASE_URL", CONFIG.get("ruview_base_url", "http://127.0.0.1:3000"))
).rstrip("/")
CUSTOM_POSE_BASE_URL = str(
    os.getenv("CUSTOM_POSE_BASE_URL", CONFIG.get("custom_pose_base_url", "http://127.0.0.1:8766"))
).rstrip("/")
POSE_INDEX_PATH = APP_DIR.parent / "custom-pose-dashboard" / "static" / "index.html"
NODE_IDS = [int(x) for x in CONFIG.get("node_ids", [1, 2, 3, 4])]
STALE_SECONDS = float(CONFIG.get("node_stale_seconds", 2.5))
INFERENCE_HZ = float(CONFIG.get("inference_hz", 5.0))
MIN_CONFIDENCE = float(CONFIG.get("minimum_confidence", 0.52))
MIN_MARGIN = float(CONFIG.get("minimum_margin", 0.10))
PROBABILITY_SMOOTHING = int(CONFIG.get("probability_smoothing", 10))

ZONE_LABELS = {
    "EMPTY": "Empty room",
    "NODE_1": "Near Node 1",
    "NODE_2": "Near Node 2",
    "NODE_3": "Near Node 3",
    "NODE_4": "Near Node 4",
    "CENTER": "Center",
}

RECORDING_SESSIONS = {
    "empty": {"model_label": "EMPTY", "display": "Empty room"},
    "zone1": {"model_label": "NODE_1", "display": "Zone 1 / Near Node 1"},
    "zone2": {"model_label": "NODE_2", "display": "Zone 2 / Near Node 2"},
    "zone3": {"model_label": "NODE_3", "display": "Zone 3 / Near Node 3"},
    "zone4": {"model_label": "NODE_4", "display": "Zone 4 / Near Node 4"},
    "center": {"model_label": "CENTER", "display": "Center"},
}

RECORDING_ALIASES = {
    "empty_room": "empty",
    "empty": "empty",
    "node1": "zone1",
    "node_1": "zone1",
    "near_node_1": "zone1",
    "zone_1": "zone1",
    "zone1": "zone1",
    "node2": "zone2",
    "node_2": "zone2",
    "near_node_2": "zone2",
    "zone_2": "zone2",
    "zone2": "zone2",
    "node3": "zone3",
    "node_3": "zone3",
    "near_node_3": "zone3",
    "zone_3": "zone3",
    "zone3": "zone3",
    "node4": "zone4",
    "node_4": "zone4",
    "near_node_4": "zone4",
    "zone_4": "zone4",
    "zone4": "zone4",
    "center": "center",
}

FEATURE_NAMES_PER_NODE = [
    "baseline_abs_z",
    "baseline_rms_z",
    "correlation_drop",
    "shape_variation",
    "temporal_delta_z",
    "rssi_delta",
]


class CalibrationRequest(BaseModel):
    label: str
    duration_seconds: int


class DemoRequest(BaseModel):
    name: str | None = None


class ZoneRecordingRequest(BaseModel):
    session: str
    trial: int = Field(default=1, ge=1, le=99)
    prepare_seconds: int = Field(default=10, ge=0, le=300)
    duration_seconds: int = Field(default=30, ge=1, le=3600)


@dataclass
class NodeFrame:
    node_id: int
    amplitude: np.ndarray
    rssi: float | None
    timestamp: float
    subcarrier_count: int


@dataclass
class CalibrationSession:
    label: str
    duration_seconds: int
    started_at: float
    ends_at: float
    raw_snapshots: list[dict[int, dict[str, Any]]] = field(default_factory=list)
    feature_samples: list[list[float]] = field(default_factory=list)
    completed: bool = False


class ZoneRuntime:
    def __init__(self) -> None:
        self.nodes: dict[int, NodeFrame] = {}
        self.previous_snapshot: dict[int, NodeFrame] | None = None
        self.baseline: dict[int, dict[str, Any]] = {}
        self.samples: dict[str, list[list[float]]] = {k: [] for k in ZONE_LABELS}
        self.model: RandomForestClassifier | None = None
        self.model_labels: list[str] = []
        self.validation: dict[str, Any] = {}
        self.probability_history: deque[np.ndarray] = deque(maxlen=PROBABILITY_SMOOTHING)
        self.calibration: CalibrationSession | None = None
        self.calibration_history: list[dict[str, Any]] = []
        self.clients: set[WebSocket] = set()
        self.connected_url: str | None = None
        self.ruview_connected = False
        self.last_error: str | None = None
        self.last_update = 0.0
        self.last_inference = 0.0
        self.current_state: dict[str, Any] = {}
        self.last_zone = "UNAVAILABLE"
        self.last_confidence = 0.0
        self.last_probabilities: dict[str, float] = {}
        self.motion_threshold = 1.0
        self.high_motion_threshold = 3.0
        self.motion_history: deque[tuple[float, float]] = deque(maxlen=100)
        self.fall_candidate_at: float | None = None
        self.fall_alert_until = 0.0
        self.recording_file = None
        self.recording_path: Path | None = None
        self.replay_active = False
        self.load_saved_state()
        self.ensure_event_log()

    def ensure_event_log(self) -> None:
        if not LOG_PATH.exists():
            with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp_utc", "zone", "confidence", "presence",
                    "motion", "strongest_node", "fall_like", "source"
                ])

    def load_saved_state(self) -> None:
        if CALIBRATION_PATH.exists():
            try:
                data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
                self.baseline = {
                    int(node_id): {
                        "mean": np.asarray(v["mean"], dtype=np.float32),
                        "scale": np.asarray(v["scale"], dtype=np.float32),
                        "rssi_mean": v.get("rssi_mean"),
                        "subcarrier_count": int(v["subcarrier_count"]),
                    }
                    for node_id, v in data.get("baseline", {}).items()
                }
                self.samples = {
                    label: [[float(x) for x in row] for row in rows]
                    for label, rows in data.get("samples", {}).items()
                }
                for label in ZONE_LABELS:
                    self.samples.setdefault(label, [])
                self.motion_threshold = float(data.get("motion_threshold", 1.0))
                self.high_motion_threshold = float(data.get("high_motion_threshold", 3.0))
                self.calibration_history = data.get("history", [])
            except Exception as exc:
                self.last_error = f"Could not load calibration: {exc}"

        if MODEL_PATH.exists():
            try:
                bundle = joblib.load(MODEL_PATH)
                self.model = bundle["model"]
                self.model_labels = list(bundle["labels"])
                self.validation = dict(bundle.get("validation", {}))
            except Exception as exc:
                self.last_error = f"Could not load zone model: {exc}"

    def save_calibration(self) -> None:
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "node_ids": NODE_IDS,
            "baseline": {
                str(node_id): {
                    "mean": b["mean"].tolist(),
                    "scale": b["scale"].tolist(),
                    "rssi_mean": b.get("rssi_mean"),
                    "subcarrier_count": b["subcarrier_count"],
                }
                for node_id, b in self.baseline.items()
            },
            "samples": self.samples,
            "motion_threshold": self.motion_threshold,
            "high_motion_threshold": self.high_motion_threshold,
            "history": self.calibration_history,
        }
        CALIBRATION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def parse_node(self, obj: dict[str, Any], fallback_id: int | None = None) -> NodeFrame | None:
        node_id = obj.get("node_id", fallback_id)
        amplitude = obj.get("amplitude")
        if node_id is None or not isinstance(amplitude, list) or len(amplitude) < 8:
            return None
        try:
            arr = np.asarray(amplitude, dtype=np.float32)
            if arr.ndim != 1 or not np.isfinite(arr).all():
                return None
            rssi_raw = obj.get("rssi_dbm", obj.get("rssi"))
            rssi = float(rssi_raw) if rssi_raw is not None else None
            timestamp = float(obj.get("timestamp", time.time()))
            return NodeFrame(
                node_id=int(node_id),
                amplitude=arr,
                rssi=rssi,
                timestamp=timestamp,
                subcarrier_count=int(arr.size),
            )
        except Exception:
            return None

    def ingest_payload(self, payload: dict[str, Any]) -> bool:
        now = time.time()
        parsed: list[NodeFrame] = []
        nodes_obj = payload.get("nodes")
        if isinstance(nodes_obj, list):
            for obj in nodes_obj:
                if isinstance(obj, dict):
                    node = self.parse_node(obj)
                    if node:
                        parsed.append(node)
        else:
            node = self.parse_node(payload)
            if node:
                parsed.append(node)

        for node in parsed:
            if node.node_id in NODE_IDS:
                node.timestamp = now
                self.nodes[node.node_id] = node

        if parsed:
            self.last_update = now
        return bool(parsed)

    def complete_snapshot(self) -> dict[int, NodeFrame] | None:
        now = time.time()
        snapshot: dict[int, NodeFrame] = {}
        for node_id in NODE_IDS:
            node = self.nodes.get(node_id)
            if node is None or now - node.timestamp > STALE_SECONDS:
                return None
            snapshot[node_id] = node
        return snapshot

    @staticmethod
    def align_vector(arr: np.ndarray, target_length: int) -> np.ndarray:
        if arr.size == target_length:
            return arr.astype(np.float32, copy=False)
        old_x = np.linspace(0.0, 1.0, arr.size)
        new_x = np.linspace(0.0, 1.0, target_length)
        return np.interp(new_x, old_x, arr).astype(np.float32)

    def compute_features(
        self,
        snapshot: dict[int, NodeFrame],
        previous: dict[int, NodeFrame] | None,
    ) -> tuple[list[float], dict[int, float], float]:
        if len(self.baseline) != len(NODE_IDS):
            raise RuntimeError("Empty-room baseline has not been completed.")

        features: list[float] = []
        scores: dict[int, float] = {}
        temporal_values: list[float] = []

        for node_id in NODE_IDS:
            node = snapshot[node_id]
            base = self.baseline[node_id]
            target = int(base["subcarrier_count"])
            amp = self.align_vector(node.amplitude, target)
            mean = base["mean"]
            scale = base["scale"]
            z = (amp - mean) / scale

            abs_z = float(np.mean(np.abs(z)))
            rms_z = float(np.sqrt(np.mean(z * z)))

            amp_centered = amp - float(np.mean(amp))
            mean_centered = mean - float(np.mean(mean))
            denom = float(np.linalg.norm(amp_centered) * np.linalg.norm(mean_centered))
            corr = float(np.dot(amp_centered, mean_centered) / denom) if denom > 1e-8 else 1.0
            corr_drop = float(np.clip(1.0 - corr, 0.0, 2.0))

            amp_mean_abs = max(float(np.mean(np.abs(amp))), 1e-6)
            shape_variation = float(np.std(amp) / amp_mean_abs)

            temporal = 0.0
            if previous and node_id in previous:
                prev_amp = self.align_vector(previous[node_id].amplitude, target)
                temporal = float(np.mean(np.abs((amp - prev_amp) / scale)))
            temporal_values.append(temporal)

            rssi_mean = base.get("rssi_mean")
            if node.rssi is None or rssi_mean is None:
                rssi_delta = 0.0
            else:
                rssi_delta = min(abs(float(node.rssi) - float(rssi_mean)) / 12.0, 3.0)

            node_features = [
                abs_z, rms_z, corr_drop, shape_variation, temporal, rssi_delta
            ]
            features.extend(node_features)

            # Interpretable disturbance score for the dashboard.
            scores[node_id] = (
                0.40 * abs_z +
                0.22 * rms_z +
                0.15 * corr_drop +
                0.18 * temporal +
                0.05 * rssi_delta
            )

        # Add normalized cross-node disturbance pattern.
        raw_scores = np.asarray([scores[n] for n in NODE_IDS], dtype=np.float32)
        total = float(raw_scores.sum())
        normalized = raw_scores / total if total > 1e-8 else np.zeros_like(raw_scores)
        features.extend(normalized.tolist())

        global_motion = float(np.mean(temporal_values)) if temporal_values else 0.0
        return features, scores, global_motion

    def begin_calibration(self, label: str, duration_seconds: int) -> None:
        label = label.upper()
        if label not in ZONE_LABELS:
            raise ValueError(f"Unknown calibration label: {label}")
        if self.calibration and not self.calibration.completed:
            raise RuntimeError("Another calibration step is already running.")
        if label != "EMPTY" and len(self.baseline) != len(NODE_IDS):
            raise RuntimeError("Complete EMPTY calibration first.")
        if not self.complete_snapshot():
            raise RuntimeError("All four nodes must be online before calibration.")

        # Re-calibrating EMPTY invalidates all previous samples and model.
        if label == "EMPTY":
            self.baseline = {}
            self.samples = {k: [] for k in ZONE_LABELS}
            self.model = None
            self.model_labels = []
            self.validation = {}
            self.probability_history.clear()
            if MODEL_PATH.exists():
                MODEL_PATH.unlink()

        now = time.time()
        self.calibration = CalibrationSession(
            label=label,
            duration_seconds=duration_seconds,
            started_at=now,
            ends_at=now + duration_seconds,
        )

    def calibration_tick(self, snapshot: dict[int, NodeFrame]) -> None:
        session = self.calibration
        if not session or session.completed:
            return

        if session.label == "EMPTY":
            session.raw_snapshots.append({
                node_id: {
                    "amplitude": node.amplitude.astype(float).tolist(),
                    "rssi": node.rssi,
                }
                for node_id, node in snapshot.items()
            })
        else:
            try:
                features, _, _ = self.compute_features(snapshot, self.previous_snapshot)
                session.feature_samples.append(features)
            except Exception:
                pass

        if time.time() >= session.ends_at:
            try:
                self.finish_calibration()
            except Exception as exc:
                session.completed = True
                self.last_error = f"Calibration failed: {exc}"

    def finish_calibration(self) -> None:
        session = self.calibration
        if not session or session.completed:
            return

        if session.label == "EMPTY":
            if len(session.raw_snapshots) < 40:
                raise RuntimeError("Too few empty-room samples. Keep all nodes online and repeat.")
            baseline: dict[int, dict[str, Any]] = {}
            mode_lengths: dict[int, int] = {}
            for node_id in NODE_IDS:
                lengths = [
                    len(s[node_id]["amplitude"])
                    for s in session.raw_snapshots if node_id in s
                ]
                if not lengths:
                    raise RuntimeError(f"No empty data received for Node {node_id}.")
                mode_lengths[node_id] = Counter(lengths).most_common(1)[0][0]

            for node_id in NODE_IDS:
                target = mode_lengths[node_id]
                arrays = [
                    np.asarray(s[node_id]["amplitude"], dtype=np.float32)
                    for s in session.raw_snapshots
                    if node_id in s and len(s[node_id]["amplitude"]) == target
                ]
                stack = np.stack(arrays)
                mean = np.mean(stack, axis=0)
                std = np.std(stack, axis=0)
                floor = max(0.02 * float(np.median(np.abs(mean))), 1e-3)
                scale = np.maximum(std, floor)
                rssis = [
                    float(s[node_id]["rssi"])
                    for s in session.raw_snapshots
                    if node_id in s and s[node_id]["rssi"] is not None
                ]
                baseline[node_id] = {
                    "mean": mean,
                    "scale": scale,
                    "rssi_mean": float(np.mean(rssis)) if rssis else None,
                    "subcarrier_count": target,
                }
            self.baseline = baseline

            # Recreate EMPTY feature samples using the completed baseline.
            empty_features: list[list[float]] = []
            empty_motion: list[float] = []
            previous: dict[int, NodeFrame] | None = None
            for raw in session.raw_snapshots:
                snap = {
                    node_id: NodeFrame(
                        node_id=node_id,
                        amplitude=np.asarray(raw[node_id]["amplitude"], dtype=np.float32),
                        rssi=raw[node_id]["rssi"],
                        timestamp=0.0,
                        subcarrier_count=len(raw[node_id]["amplitude"]),
                    )
                    for node_id in NODE_IDS if node_id in raw
                }
                if len(snap) != len(NODE_IDS):
                    continue
                feat, _, motion = self.compute_features(snap, previous)
                empty_features.append(feat)
                empty_motion.append(motion)
                previous = snap
            self.samples["EMPTY"] = empty_features
            if empty_motion:
                self.motion_threshold = max(float(np.percentile(empty_motion, 97)), 0.10)
                self.high_motion_threshold = max(
                    float(np.percentile(empty_motion, 99.5)) * 4.0,
                    self.motion_threshold * 3.0,
                    0.75,
                )
        else:
            if len(session.feature_samples) < 30:
                raise RuntimeError(
                    f"Too few samples for {session.label}. Repeat while all nodes remain online."
                )
            self.samples[session.label] = session.feature_samples

        session.completed = True
        self.calibration_history.append({
            "label": session.label,
            "samples": len(
                session.raw_snapshots if session.label == "EMPTY"
                else session.feature_samples
            ),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        self.save_calibration()

    def train_model(self) -> dict[str, Any]:
        missing = [label for label in ZONE_LABELS if len(self.samples.get(label, [])) < 30]
        if missing:
            raise RuntimeError(
                "Missing calibration classes: " + ", ".join(missing)
            )

        # Time-block holdout: first 75% of each class train, final 25% validate.
        x_train, y_train, x_val, y_val = [], [], [], []
        all_x, all_y = [], []
        class_counts: dict[str, int] = {}

        for label in ZONE_LABELS:
            rows = self.samples[label]
            class_counts[label] = len(rows)
            split = max(1, int(len(rows) * 0.75))
            x_train.extend(rows[:split])
            y_train.extend([label] * split)
            x_val.extend(rows[split:])
            y_val.extend([label] * (len(rows) - split))
            all_x.extend(rows)
            all_y.extend([label] * len(rows))

        validation_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        validation_model.fit(np.asarray(x_train), np.asarray(y_train))
        val_pred = validation_model.predict(np.asarray(x_val))
        accuracy = float(accuracy_score(y_val, val_pred))
        labels = list(ZONE_LABELS)
        matrix = confusion_matrix(y_val, val_pred, labels=labels).tolist()

        final_model = RandomForestClassifier(
            n_estimators=400,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        final_model.fit(np.asarray(all_x), np.asarray(all_y))

        self.model = final_model
        self.model_labels = list(final_model.classes_)
        self.validation = {
            "time_block_accuracy": accuracy,
            "labels": labels,
            "confusion_matrix": matrix,
            "class_counts": class_counts,
            "note": (
                "Same-room calibration consistency only; this is not unseen-room "
                "localization accuracy."
            ),
        }
        joblib.dump(
            {
                "model": final_model,
                "labels": self.model_labels,
                "validation": self.validation,
                "node_ids": NODE_IDS,
                "feature_names_per_node": FEATURE_NAMES_PER_NODE,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            MODEL_PATH,
        )
        self.probability_history.clear()
        return self.validation

    def classify(
        self, features: list[float]
    ) -> tuple[str, float, dict[str, float]]:
        if self.model is None:
            return "UNTRAINED", 0.0, {}

        probs = self.model.predict_proba(np.asarray(features, dtype=np.float32).reshape(1, -1))[0]
        self.probability_history.append(probs)
        smoothed = np.mean(np.stack(self.probability_history), axis=0)
        order = np.argsort(smoothed)[::-1]
        top_idx = int(order[0])
        second_idx = int(order[1]) if len(order) > 1 else top_idx
        top_conf = float(smoothed[top_idx])
        margin = top_conf - float(smoothed[second_idx])
        raw_label = str(self.model.classes_[top_idx])
        probabilities = {
            str(label): float(prob)
            for label, prob in zip(self.model.classes_, smoothed)
        }
        if top_conf < MIN_CONFIDENCE or margin < MIN_MARGIN:
            return "UNCERTAIN", top_conf, probabilities
        return raw_label, top_conf, probabilities

    def motion_label(self, motion: float) -> str:
        if motion < self.motion_threshold:
            return "STILL"
        if motion < self.high_motion_threshold:
            return "MOVING"
        return "HIGH MOTION"

    def update_fall_like(self, motion: float, zone: str) -> tuple[bool, str]:
        now = time.time()
        self.motion_history.append((now, motion))
        if motion >= self.high_motion_threshold:
            self.fall_candidate_at = now

        if self.fall_candidate_at is not None:
            age = now - self.fall_candidate_at
            if 0.5 <= age <= 3.0 and motion < self.motion_threshold and zone not in ("EMPTY", "UNTRAINED"):
                self.fall_alert_until = now + 5.0
                self.fall_candidate_at = None
            elif age > 3.0:
                self.fall_candidate_at = None

        active = now < self.fall_alert_until
        if active:
            return True, "Experimental fall-like pattern: motion spike followed by stillness"
        return False, "No fall-like pattern"

    def node_status(self, scores: dict[int, float] | None = None) -> list[dict[str, Any]]:
        now = time.time()
        out = []
        scores = scores or {}
        for node_id in NODE_IDS:
            node = self.nodes.get(node_id)
            online = node is not None and now - node.timestamp <= STALE_SECONDS
            out.append({
                "node_id": node_id,
                "online": online,
                "age_ms": int((now - node.timestamp) * 1000) if node else None,
                "rssi": node.rssi if node else None,
                "subcarriers": node.subcarrier_count if node else None,
                "disturbance": float(scores.get(node_id, 0.0)),
            })
        return out

    def calibration_status(self) -> dict[str, Any]:
        session = self.calibration
        active = bool(session and not session.completed)
        remaining = max(0.0, session.ends_at - time.time()) if active else 0.0
        return {
            "active": active,
            "label": session.label if active else None,
            "remaining_seconds": remaining,
            "samples_current": (
                len(session.raw_snapshots) if active and session.label == "EMPTY"
                else len(session.feature_samples) if active else 0
            ),
            "completed_labels": {
                label: len(self.samples.get(label, [])) for label in ZONE_LABELS
            },
            "baseline_ready": len(self.baseline) == len(NODE_IDS),
            "model_ready": self.model is not None,
            "validation": self.validation,
        }

    def make_state(
        self,
        *,
        scores: dict[int, float] | None = None,
        motion: float = 0.0,
        source: str = "live",
    ) -> dict[str, Any]:
        strongest_node = None
        if scores:
            strongest_node = max(scores, key=scores.get)

        zone = self.last_zone
        confidence = self.last_confidence
        presence = (
            zone not in ("EMPTY", "UNCERTAIN", "UNTRAINED", "UNAVAILABLE")
        )
        fall_like, fall_reason = self.update_fall_like(motion, zone)
        state = {
            "timestamp": time.time(),
            "source": source,
            "ruview_connected": self.ruview_connected,
            "ruview_url": self.connected_url,
            "error": self.last_error,
            "zone": zone,
            "zone_display": ZONE_LABELS.get(zone, zone.title()),
            "confidence": confidence,
            "probabilities": self.last_probabilities,
            "presence": presence,
            "motion": self.motion_label(motion),
            "motion_score": motion,
            "motion_threshold": self.motion_threshold,
            "high_motion_threshold": self.high_motion_threshold,
            "strongest_node": strongest_node,
            "nodes": self.node_status(scores),
            "calibration": self.calibration_status(),
            "fall_like": fall_like,
            "fall_reason": fall_reason,
            "recording": str(self.recording_path) if self.recording_file else None,
            "replay_active": self.replay_active,
            "deployment_note": (
                "Room-specific, single-person, coarse zone estimate. "
                "Not exact coordinates or calibrated triangulation."
            ),
        }
        self.current_state = state
        return state

    async def publish(self, state: dict[str, Any]) -> None:
        self.current_state = state
        text = json.dumps(state)
        dead = []
        for client in self.clients:
            try:
                await client.send_text(text)
            except Exception:
                dead.append(client)
        for client in dead:
            self.clients.discard(client)

        if self.recording_file and state.get("source") == "live":
            self.recording_file.write(text + "\n")
            self.recording_file.flush()

    def write_event(self, state: dict[str, Any]) -> None:
        with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                state["zone"],
                f'{state["confidence"]:.4f}',
                state["presence"],
                state["motion"],
                state["strongest_node"],
                state["fall_like"],
                state["source"],
            ])

    def start_demo_recording(self, name: str | None) -> str:
        if self.recording_file:
            raise RuntimeError("A demo recording is already active.")
        safe_name = "".join(c for c in (name or "showcase") if c.isalnum() or c in "-_")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = DATA_DIR / f"demo_{safe_name}_{stamp}.jsonl"
        self.recording_file = path.open("w", encoding="utf-8")
        self.recording_path = path
        return str(path)

    def stop_demo_recording(self) -> str | None:
        path = str(self.recording_path) if self.recording_path else None
        if self.recording_file:
            self.recording_file.close()
        self.recording_file = None
        self.recording_path = None
        return path


def pose_ws_url() -> str:
    if CUSTOM_POSE_BASE_URL.startswith("https://"):
        return f"wss://{CUSTOM_POSE_BASE_URL.removeprefix('https://')}/ws"
    if CUSTOM_POSE_BASE_URL.startswith("http://"):
        return f"ws://{CUSTOM_POSE_BASE_URL.removeprefix('http://')}/ws"
    return f"ws://{CUSTOM_POSE_BASE_URL}/ws"


def post_json_blocking(url: str, payload: dict[str, Any], timeout: float = 15.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{exc.code} {detail}".strip()) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc

    if not text.strip():
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def get_json_blocking(url: str, timeout: float = 5.0) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{exc.code} {detail}".strip()) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc

    if not text.strip():
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


class ZoneRecordingController:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._state: dict[str, Any] = self._idle_state("Ready to record.")

    def _idle_state(self, message: str) -> dict[str, Any]:
        now = time.time()
        return {
            "active": False,
            "phase": "idle",
            "session": None,
            "session_display": None,
            "session_id": None,
            "trial": None,
            "prepare_seconds": 10,
            "duration_seconds": 30,
            "started_at": None,
            "phase_started_at": None,
            "phase_ends_at": None,
            "remaining_seconds": 0.0,
            "elapsed_seconds": 0.0,
            "recording_started": False,
            "stop_requested": False,
            "saved": False,
            "message": message,
            "error": None,
            "updated_at": now,
        }

    def _set_state(self, **updates: Any) -> None:
        self._state.update(updates)
        self._state["updated_at"] = time.time()

    def status(self) -> dict[str, Any]:
        now = time.time()
        state = dict(self._state)
        phase_ends_at = state.get("phase_ends_at")
        started_at = state.get("started_at")
        state["remaining_seconds"] = (
            max(0.0, float(phase_ends_at) - now) if phase_ends_at else 0.0
        )
        state["elapsed_seconds"] = (
            max(0.0, now - float(started_at)) if started_at else 0.0
        )
        state["ruview_base_url"] = RUVIEW_BASE_URL
        return state

    def normalize_session(self, raw_session: str) -> tuple[str, dict[str, str]]:
        key = raw_session.strip().lower().replace("-", "_").replace(" ", "_")
        session = RECORDING_ALIASES.get(key)
        if session is None or session not in RECORDING_SESSIONS:
            allowed = ", ".join(RECORDING_SESSIONS)
            raise ValueError(f"Session must be one of: {allowed}.")
        return session, RECORDING_SESSIONS[session]

    async def start(self, request: ZoneRecordingRequest) -> dict[str, Any]:
        async with self._lock:
            if self._task and not self._task.done():
                raise RuntimeError("Another zone recording is already active.")

            session, meta = self.normalize_session(request.session)
            session_id = f"train_{session}_trial_{request.trial:02d}"
            self._stop_event = asyncio.Event()
            self._set_state(
                active=True,
                phase="preparing",
                session=session,
                session_display=meta["display"],
                session_id=session_id,
                trial=request.trial,
                prepare_seconds=request.prepare_seconds,
                duration_seconds=request.duration_seconds,
                started_at=time.time(),
                phase_started_at=time.time(),
                phase_ends_at=time.time() + request.prepare_seconds,
                recording_started=False,
                stop_requested=False,
                saved=False,
                error=None,
                message=(
                    f"Move to {meta['display']}. "
                    "RuView recording has not started yet."
                ),
            )
            self._task = asyncio.create_task(
                self._run_recording(request, session, meta["display"], session_id, self._stop_event)
            )
            return self.status()

    async def stop(self) -> dict[str, Any]:
        async with self._lock:
            if not self._task or self._task.done():
                return self.status()
            if self._stop_event:
                self._stop_event.set()
            phase = str(self._state.get("phase") or "")
            self._set_state(
                stop_requested=True,
                message=(
                    "Stop requested. Waiting for RuView to finalize."
                    if phase in ("recording", "starting", "stopping")
                    else "Cancelled before recording started."
                ),
            )
            return self.status()

    async def shutdown(self) -> None:
        if self._task and not self._task.done():
            if self._stop_event:
                self._stop_event.set()
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except asyncio.TimeoutError:
                self._task.cancel()

    async def _sleep_or_stop(self, seconds: int, stop_event: asyncio.Event) -> bool:
        if seconds <= 0:
            return stop_event.is_set()
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=float(seconds))
            return True
        except asyncio.TimeoutError:
            return False

    async def _post_ruview(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(
            post_json_blocking,
            f"{RUVIEW_BASE_URL}{path}",
            payload,
        )

    async def _run_recording(
        self,
        request: ZoneRecordingRequest,
        session: str,
        display: str,
        session_id: str,
        stop_event: asyncio.Event,
    ) -> None:
        recording_started = False
        try:
            stopped_early = await self._sleep_or_stop(request.prepare_seconds, stop_event)
            if stopped_early:
                self._set_state(
                    active=False,
                    phase="stopped",
                    phase_ends_at=None,
                    stop_requested=True,
                    saved=False,
                    message=f"{display} trial {request.trial} cancelled before recording started.",
                )
                return

            self._set_state(
                phase="starting",
                phase_started_at=time.time(),
                phase_ends_at=None,
                message=f"Starting RuView recording for {display}.",
            )
            await self._post_ruview("/api/v1/recording/start", {"id": session_id})
            recording_started = True
            self._set_state(
                phase="recording",
                phase_started_at=time.time(),
                phase_ends_at=time.time() + request.duration_seconds,
                recording_started=True,
                message=f"Recording {display}. Remain in the selected position.",
            )

            stopped_early = await self._sleep_or_stop(request.duration_seconds, stop_event)
            self._set_state(
                phase="stopping",
                phase_started_at=time.time(),
                phase_ends_at=None,
                stop_requested=stopped_early,
                message="Stopping RuView recording.",
            )
            await self._post_ruview("/api/v1/recording/stop", {})
            self._set_state(
                active=False,
                phase="stopped" if stopped_early else "completed",
                phase_ends_at=None,
                saved=True,
                message=(
                    f"{display} trial {request.trial} stopped and saved."
                    if stopped_early
                    else f"{display} trial {request.trial} saved successfully."
                ),
            )
        except Exception as exc:
            stop_error = None
            if recording_started:
                try:
                    await self._post_ruview("/api/v1/recording/stop", {})
                except Exception as stop_exc:
                    stop_error = str(stop_exc)
            message = str(exc)
            if stop_error:
                message = f"{message}; additionally could not stop RuView cleanly: {stop_error}"
            self._set_state(
                active=False,
                phase="error",
                phase_ends_at=None,
                error=message,
                message=message,
            )


runtime = ZoneRuntime()
recording_controller = ZoneRecordingController()
app = FastAPI(title="RuView Four-Node Zone Dashboard")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/pose")
@app.get("/pose/")
async def pose_index() -> FileResponse:
    if not POSE_INDEX_PATH.exists():
        raise HTTPException(status_code=404, detail="Custom pose dashboard static page not found.")
    return FileResponse(POSE_INDEX_PATH)


@app.get("/health")
async def health() -> JSONResponse:
    all_online = all(n["online"] for n in runtime.node_status())
    return JSONResponse({
        "status": "ok" if runtime.ruview_connected else "degraded",
        "ruview_connected": runtime.ruview_connected,
        "all_four_nodes_online": all_online,
        "baseline_ready": len(runtime.baseline) == len(NODE_IDS),
        "model_ready": runtime.model is not None,
        "zone_recording": recording_controller.status(),
        "error": runtime.last_error,
    })


@app.get("/api/state")
async def api_state() -> JSONResponse:
    state = dict(runtime.current_state or runtime.make_state())
    state["zone_recording"] = recording_controller.status()
    return JSONResponse(state)


@app.get("/api/config")
async def api_config() -> JSONResponse:
    return JSONResponse(CONFIG)


@app.get("/api/record-zone/sessions")
async def record_zone_sessions() -> JSONResponse:
    sessions = [
        {
            "id": session,
            "display": meta["display"],
            "model_label": meta["model_label"],
        }
        for session, meta in RECORDING_SESSIONS.items()
    ]
    return JSONResponse({"sessions": sessions})


@app.get("/api/record-zone/status")
async def record_zone_status() -> JSONResponse:
    return JSONResponse(recording_controller.status())


@app.post("/api/record-zone")
@app.post("/api/record-zone/start")
async def record_zone_start(req: ZoneRecordingRequest) -> JSONResponse:
    try:
        state = await recording_controller.start(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return JSONResponse({"status": "started", "recording": state})


@app.post("/api/record-zone/stop")
async def record_zone_stop() -> JSONResponse:
    state = await recording_controller.stop()
    return JSONResponse({"status": state.get("phase", "idle"), "recording": state})


@app.get("/pose/health")
async def pose_health() -> JSONResponse:
    try:
        data = await asyncio.to_thread(get_json_blocking, f"{CUSTOM_POSE_BASE_URL}/health")
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Custom pose dashboard is not reachable at {CUSTOM_POSE_BASE_URL}: {exc}",
        )
    return JSONResponse(data)


@app.get("/pose/api/state")
async def pose_api_state() -> JSONResponse:
    try:
        data = await asyncio.to_thread(get_json_blocking, f"{CUSTOM_POSE_BASE_URL}/api/state")
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Custom pose dashboard is not reachable at {CUSTOM_POSE_BASE_URL}: {exc}",
        )
    return JSONResponse(data)


@app.post("/api/calibration/start")
async def calibration_start(req: CalibrationRequest) -> JSONResponse:
    duration = max(10, min(int(req.duration_seconds), 180))
    try:
        runtime.begin_calibration(req.label, duration)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse({"status": "started", "label": req.label.upper(), "duration": duration})


@app.post("/api/calibration/cancel")
async def calibration_cancel() -> JSONResponse:
    if runtime.calibration and not runtime.calibration.completed:
        runtime.calibration.completed = True
    return JSONResponse({"status": "cancelled"})


@app.post("/api/model/train")
async def model_train() -> JSONResponse:
    try:
        validation = await asyncio.to_thread(runtime.train_model)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse({"status": "trained", "validation": validation})


@app.post("/api/model/reset")
async def model_reset() -> JSONResponse:
    runtime.model = None
    runtime.model_labels = []
    runtime.validation = {}
    runtime.baseline = {}
    runtime.samples = {k: [] for k in ZONE_LABELS}
    runtime.probability_history.clear()
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    if CALIBRATION_PATH.exists():
        CALIBRATION_PATH.unlink()
    return JSONResponse({"status": "reset"})


@app.post("/api/demo/start")
async def demo_start(req: DemoRequest) -> JSONResponse:
    try:
        path = runtime.start_demo_recording(req.name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse({"status": "recording", "path": path})


@app.post("/api/demo/stop")
async def demo_stop() -> JSONResponse:
    path = runtime.stop_demo_recording()
    return JSONResponse({"status": "stopped", "path": path})


@app.get("/api/demo/list")
async def demo_list() -> JSONResponse:
    files = [
        {"name": p.name, "size_bytes": p.stat().st_size, "modified": p.stat().st_mtime}
        for p in sorted(DATA_DIR.glob("demo_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    ]
    return JSONResponse({"recordings": files})


@app.post("/api/demo/replay/{filename}")
async def demo_replay(filename: str) -> JSONResponse:
    path = (DATA_DIR / filename).resolve()
    if path.parent != DATA_DIR.resolve() or not path.exists():
        raise HTTPException(status_code=404, detail="Demo recording not found.")
    if runtime.replay_active:
        raise HTTPException(status_code=400, detail="Replay already active.")
    asyncio.create_task(replay_file(path))
    return JSONResponse({"status": "replaying", "file": filename})


@app.websocket("/pose/ws")
async def pose_dashboard_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    upstream_url = pose_ws_url()
    try:
        async with websockets.connect(
            upstream_url,
            ping_interval=20,
            ping_timeout=20,
            max_size=16 * 1024 * 1024,
        ) as upstream:
            async def browser_to_pose() -> None:
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        break
                    if message.get("text") is not None:
                        await upstream.send(message["text"])
                    elif message.get("bytes") is not None:
                        await upstream.send(message["bytes"])

            async def pose_to_browser() -> None:
                async for message in upstream:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)

            tasks = [
                asyncio.create_task(browser_to_pose()),
                asyncio.create_task(pose_to_browser()),
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                task.result()
    except WebSocketDisconnect:
        return
    except Exception as exc:
        fallback = {
            "mode": "unavailable",
            "source": "proxy",
            "ruview_connected": False,
            "model_loaded": False,
            "model_error": f"Custom pose dashboard unavailable at {CUSTOM_POSE_BASE_URL}: {exc}",
            "model_metrics": {},
            "device": "unknown",
            "training_node_order": [],
            "live_node_order": [],
            "nodes": [],
            "buffer_frames": 0,
            "window_frames": 20,
            "pose": [],
            "fall_like": False,
            "fall_reason": "Pose backend unavailable",
            "latency_ms": None,
            "valid_frames": 0,
            "dropped_frames": 0,
            "accepted_updates": 0,
            "rejected_updates": 0,
            "updated_at": time.time(),
            "warning": f"Start custom-pose-dashboard on {CUSTOM_POSE_BASE_URL} to use this view.",
        }
        try:
            await websocket.send_text(json.dumps(fallback))
            await websocket.close()
        except Exception:
            pass


@app.websocket("/ws")
async def dashboard_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    runtime.clients.add(websocket)
    await websocket.send_text(json.dumps(runtime.current_state or runtime.make_state()))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        runtime.clients.discard(websocket)
    except Exception:
        runtime.clients.discard(websocket)


async def replay_file(path: Path) -> None:
    runtime.replay_active = True
    try:
        with path.open("r", encoding="utf-8") as f:
            previous_ts = None
            for line in f:
                if not line.strip():
                    continue
                state = json.loads(line)
                ts = float(state.get("timestamp", 0))
                if previous_ts is not None:
                    await asyncio.sleep(min(max(ts - previous_ts, 0.05), 0.5))
                previous_ts = ts
                state["source"] = "replay"
                state["replay_active"] = True
                await runtime.publish(state)
    finally:
        runtime.replay_active = False


async def process_tick() -> None:
    snapshot = runtime.complete_snapshot()
    if snapshot is None:
        runtime.last_zone = "UNAVAILABLE"
        runtime.last_confidence = 0.0
        runtime.last_probabilities = {}
        await runtime.publish(runtime.make_state())
        return

    runtime.calibration_tick(snapshot)

    scores: dict[int, float] = {}
    motion = 0.0
    if len(runtime.baseline) == len(NODE_IDS):
        try:
            features, scores, motion = runtime.compute_features(snapshot, runtime.previous_snapshot)
            if runtime.model is not None and not (
                runtime.calibration and not runtime.calibration.completed
            ):
                zone, confidence, probabilities = runtime.classify(features)
                runtime.last_zone = zone
                runtime.last_confidence = confidence
                runtime.last_probabilities = probabilities
            elif runtime.calibration and not runtime.calibration.completed:
                runtime.last_zone = f"CALIBRATING_{runtime.calibration.label}"
                runtime.last_confidence = 0.0
            else:
                runtime.last_zone = "UNTRAINED"
                runtime.last_confidence = 0.0
        except Exception as exc:
            runtime.last_error = str(exc)

    runtime.previous_snapshot = {
        node_id: NodeFrame(
            node_id=node.node_id,
            amplitude=node.amplitude.copy(),
            rssi=node.rssi,
            timestamp=node.timestamp,
            subcarrier_count=node.subcarrier_count,
        )
        for node_id, node in snapshot.items()
    }

    state = runtime.make_state(scores=scores, motion=motion)
    runtime.write_event(state)
    await runtime.publish(state)


async def source_loop() -> None:
    delay = 1.0 / max(INFERENCE_HZ, 1.0)
    while True:
        connected = False
        for url in RUVIEW_WS_CANDIDATES:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=20,
                    max_size=32 * 1024 * 1024,
                ) as ws:
                    runtime.ruview_connected = True
                    runtime.connected_url = url
                    runtime.last_error = None
                    connected = True
                    last_process = 0.0

                    async for message in ws:
                        if runtime.replay_active:
                            continue
                        try:
                            payload = json.loads(message)
                        except Exception:
                            continue
                        if not isinstance(payload, dict):
                            continue
                        if not runtime.ingest_payload(payload):
                            continue

                        now = time.time()
                        if now - last_process >= delay:
                            last_process = now
                            await process_tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                runtime.last_error = f"{url}: {exc}"
                runtime.ruview_connected = False
                runtime.connected_url = None
                await runtime.publish(runtime.make_state())
                await asyncio.sleep(1.0)

            if connected:
                break

        if not connected:
            await asyncio.sleep(1.5)


@app.on_event("startup")
async def startup() -> None:
    runtime.current_state = runtime.make_state()
    app.state.source_task = asyncio.create_task(source_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    await recording_controller.shutdown()
    runtime.stop_demo_recording()
    task = getattr(app.state, "source_task", None)
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
