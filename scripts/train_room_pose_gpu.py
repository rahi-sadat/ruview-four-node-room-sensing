#!/usr/bin/env python3
"""
GPU trainer for RuView camera-supervised paired JSONL data.

Input record expected from scripts/align-ground-truth.js:
{
  "csi": [...],
  "csi_shape": [20, 56],   # frame-major [time, subcarriers]
  "kp": [[x, y], ... 17],
  "conf": 0.917,
  "ts_start": "...",
  ...
}

The loader transposes CSI to PyTorch Conv1d layout [subcarriers, time].
It uses a chronological 80/20 split, torso-normalized PCK@20/PCK@50,
mixed precision on CUDA, and saves the best checkpoint plus predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


COCO_KP = 17


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Paired JSONL file")
    p.add_argument("--output", default="models/room-pose-gpu", help="Output directory")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0, help="Use 0 on Windows")
    return p.parse_args()


def flatten_keypoints(obj: dict[str, Any]) -> list[float] | None:
    kp = obj.get("kp", obj.get("keypoints"))
    if not isinstance(kp, list) or len(kp) < COCO_KP:
        return None

    if kp and isinstance(kp[0], list):
        out: list[float] = []
        for pair in kp[:COCO_KP]:
            if not isinstance(pair, list) or len(pair) < 2:
                return None
            out.extend((float(pair[0]), float(pair[1])))
        return out

    if len(kp) < COCO_KP * 2:
        return None
    return [float(v) for v in kp[: COCO_KP * 2]]


def load_paired(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    ws: list[float] = []
    timestamps: list[str] = []

    expected_t: int | None = None
    expected_sc: int | None = None
    skipped = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                csi = obj.get("csi")
                shape = obj.get("csi_shape")
                kp = flatten_keypoints(obj)

                if not isinstance(csi, list) or not isinstance(shape, list) or len(shape) != 2 or kp is None:
                    skipped += 1
                    continue

                a, b = int(shape[0]), int(shape[1])
                if len(csi) != a * b:
                    skipped += 1
                    continue

                # Current aligner writes frame-major [time, subcarriers].
                # A normal capture is [20, 56]. Handle legacy [56, 20] too.
                flat = torch.tensor(csi, dtype=torch.float32)
                if a <= 32 and b > a:
                    t, sc = a, b
                    x = flat.view(t, sc).transpose(0, 1).contiguous()  # [sc, t]
                elif b <= 32 and a > b:
                    sc, t = a, b
                    x = flat.view(sc, t).contiguous()  # legacy [sc, t]
                else:
                    raise ValueError(f"Ambiguous CSI shape {shape}")

                if expected_t is None:
                    expected_t, expected_sc = t, sc
                if t != expected_t or sc != expected_sc:
                    skipped += 1
                    continue

                # Per-subcarrier temporal normalization.
                mean = x.mean(dim=1, keepdim=True)
                std = x.std(dim=1, keepdim=True).clamp_min(1e-5)
                x = (x - mean) / std

                conf = obj.get("conf", 1.0)
                if isinstance(conf, list) and conf:
                    weight = sum(float(v) for v in conf[:COCO_KP]) / min(len(conf), COCO_KP)
                else:
                    weight = float(conf)
                weight = max(0.05, min(1.0, weight))

                xs.append(x)
                ys.append(torch.tensor(kp, dtype=torch.float32).view(COCO_KP, 2))
                ws.append(weight)
                timestamps.append(str(obj.get("ts_start", obj.get("timestamp", line_no))))
            except Exception:
                skipped += 1

    if not xs:
        raise RuntimeError("No valid samples were loaded.")

    x_all = torch.stack(xs)             # [N, subcarriers, time]
    y_all = torch.stack(ys)             # [N, 17, 2]
    w_all = torch.tensor(ws, dtype=torch.float32)

    print(f"Loaded samples: {len(xs)}")
    print(f"CSI tensor: {tuple(x_all.shape)} [N, subcarriers, time]")
    print(f"Keypoints: {tuple(y_all.shape)}")
    print(f"Skipped records: {skipped}")
    return x_all, y_all, w_all, timestamps


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
            nn.Linear(256, COCO_KP * 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.head(self.tcn(self.stem(x)))
        return z.view(-1, COCO_KP, 2)


def torso_scale(kp: torch.Tensor) -> torch.Tensor:
    # COCO: shoulders 5/6, hips 11/12
    shoulder_mid = 0.5 * (kp[:, 5] + kp[:, 6])
    hip_mid = 0.5 * (kp[:, 11] + kp[:, 12])
    scale = torch.linalg.vector_norm(shoulder_mid - hip_mid, dim=-1)
    return scale.clamp_min(0.05)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    pck20_num = 0
    pck50_num = 0
    joint_count = 0
    mpjpe_sum = 0.0

    for xb, yb, wb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        wb = wb.to(device, non_blocking=True)

        pred = model(xb)
        per_sample = torch.nn.functional.smooth_l1_loss(pred, yb, reduction="none").mean((1, 2))
        total_loss += (per_sample * wb).sum().item()
        n += xb.size(0)

        dist = torch.linalg.vector_norm(pred - yb, dim=-1)
        scale = torso_scale(yb).unsqueeze(1)
        norm_dist = dist / scale

        pck20_num += (norm_dist <= 0.20).sum().item()
        pck50_num += (norm_dist <= 0.50).sum().item()
        joint_count += norm_dist.numel()
        mpjpe_sum += norm_dist.sum().item()

    return {
        "loss": total_loss / max(n, 1),
        "pck20": pck20_num / max(joint_count, 1),
        "pck50": pck50_num / max(joint_count, 1),
        "nmpjpe": mpjpe_sum / max(joint_count, 1),
    }


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"ERROR: file not found: {data_path}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available in this Python environment.", file=sys.stderr)
        return 3

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    x, y, w, timestamps = load_paired(data_path)
    n_total = x.size(0)
    n_eval = max(1, int(n_total * args.eval_split))
    n_train = n_total - n_eval
    if n_train < 100:
        raise RuntimeError("Too few training samples.")

    # Chronological split: last segment is held out.
    train_ds = TensorDataset(x[:n_train], y[:n_train], w[:n_train])
    eval_ds = TensorDataset(x[n_train:], y[n_train:], w[n_train:])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = PoseTCN(x.size(1)).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Train/eval: {n_train}/{n_eval}")
    print(f"Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = nn.SmoothL1Loss(reduction="none", beta=0.1)

    best_pck20 = -1.0
    best_metrics: dict[str, float] = {}
    history: list[dict[str, float | int]] = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for xb, yb, wb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)

            # Light CSI augmentation.
            if model.training:
                xb = xb + 0.01 * torch.randn_like(xb)
                mask = (torch.rand(xb.size(0), xb.size(1), 1, device=device) > 0.08).float()
                xb = xb * mask

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred = model(xb)
                per_sample = loss_fn(pred, yb).mean((1, 2))
                loss = (per_sample * wb).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * xb.size(0)
            train_n += xb.size(0)

        scheduler.step()
        metrics = evaluate(model, eval_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_n, 1),
            **metrics,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)

        if metrics["pck20"] > best_pck20:
            best_pck20 = metrics["pck20"]
            best_metrics = metrics.copy()
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "in_channels": x.size(1),
                    "time_steps": x.size(2),
                    "epoch": epoch,
                    "metrics": metrics,
                    "data_file": str(data_path),
                },
                out_dir / "best_pose_tcn.pt",
            )

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            elapsed = time.time() - start
            print(
                f"epoch {epoch:3d}/{args.epochs} "
                f"train={row['train_loss']:.5f} eval={metrics['loss']:.5f} "
                f"PCK20={metrics['pck20']*100:5.1f}% "
                f"PCK50={metrics['pck50']*100:5.1f}% "
                f"nMPJPE={metrics['nmpjpe']:.3f} "
                f"elapsed={elapsed/60:.1f}m"
            )

    # Restore best checkpoint for prediction export.
    checkpoint = torch.load(out_dir / "best_pose_tcn.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    predictions_path = out_dir / "eval_predictions.jsonl"
    pred_lines: list[str] = []
    offset = n_train
    with torch.no_grad():
        for i in range(n_eval):
            pred = model(x[offset + i : offset + i + 1].to(device)).cpu()[0]
            pred_lines.append(
                json.dumps(
                    {
                        "timestamp": timestamps[offset + i],
                        "predicted": pred.tolist(),
                        "ground_truth": y[offset + i].tolist(),
                    }
                )
            )
    predictions_path.write_text("\n".join(pred_lines) + "\n", encoding="utf-8")

    report = {
        "data": str(data_path),
        "samples": n_total,
        "train_samples": n_train,
        "eval_samples": n_eval,
        "epochs": args.epochs,
        "best_epoch": int(checkpoint["epoch"]),
        "best_metrics": best_metrics,
        "elapsed_seconds": time.time() - start,
        "gpu": torch.cuda.get_device_name(0),
        "note": "Same-room uncalibrated pilot. Metrics use chronological holdout and torso-normalized PCK.",
    }
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Training complete ===")
    print(f"Best epoch: {report['best_epoch']}")
    print(f"Best PCK@20: {best_metrics['pck20']*100:.2f}%")
    print(f"Best PCK@50: {best_metrics['pck50']*100:.2f}%")
    print(f"Best nMPJPE: {best_metrics['nmpjpe']:.4f}")
    print(f"Checkpoint: {out_dir / 'best_pose_tcn.pt'}")
    print(f"Report: {out_dir / 'training_report.json'}")
    print(f"Predictions: {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
