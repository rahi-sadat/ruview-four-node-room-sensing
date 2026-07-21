#!/usr/bin/env python3
"""Bridge Espressif ``csi_recv_router`` serial output to RuView UDP.

This is useful when an ESP32-S3 is running Espressif's receiver firmware,
which writes ``CSI_DATA`` CSV records to USB serial instead of emitting
RuView ADR-018 UDP frames itself.  Internet access is not required:

    ESP32 USB serial -> this bridge -> 127.0.0.1:5005/udp -> RuView

Example:

    py -3 scripts/esp32-serial-udp-bridge.py ^
        --node COM5:1 --node COM10:3
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass

import serial


ADR018_MAGIC = 0xC511_0001
ADR018_HEADER = struct.Struct("<IBBHIIbbBB")
MAX_SUBCARRIERS = 256


@dataclass(frozen=True)
class Node:
    port: str
    node_id: int


@dataclass
class Counters:
    serial_lines: int = 0
    frames_sent: int = 0
    parse_errors: int = 0
    serial_errors: int = 0


def channel_frequency_mhz(channel: int) -> int:
    if channel == 14:
        return 2484
    if 1 <= channel <= 13:
        return 2407 + channel * 5
    if 32 <= channel <= 177:
        return 5000 + channel * 5
    return 0


def parse_node(value: str) -> Node:
    try:
        port, raw_node_id = value.rsplit(":", 1)
        node_id = int(raw_node_id)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            f"expected PORT:NODE_ID, for example COM5:1 (got {value!r})"
        ) from exc
    if not port:
        raise argparse.ArgumentTypeError("serial port cannot be empty")
    if not 0 <= node_id <= 255:
        raise argparse.ArgumentTypeError("node ID must be between 0 and 255")
    return Node(port=port, node_id=node_id)


def clamp_i8(value: int) -> int:
    return max(-128, min(127, value))


def csi_csv_to_adr018(line: str, node_id: int) -> bytes | None:
    """Convert one Espressif CSI_DATA CSV row into an ADR-018 datagram."""
    if not line.startswith("CSI_DATA,"):
        return None

    row = next(csv.reader([line]))
    # Espressif csi_recv_router fields end with:
    # ..., channel, secondary_channel, timestamp, ant, sig_len, rx_state,
    # data_len, first_word_invalid, "[raw int8 CSI bytes]"
    if len(row) < 25 or row[0] != "CSI_DATA":
        raise ValueError(f"unexpected CSI_DATA field count: {len(row)}")

    sequence = int(row[1]) & 0xFFFF_FFFF
    rssi = clamp_i8(int(row[3]))
    noise_floor = clamp_i8(int(row[14]))
    channel = int(row[16])
    raw_values = json.loads(row[24])
    if not isinstance(raw_values, list):
        raise ValueError("CSI payload is not a list")

    declared_length = int(row[22])
    iq_values = [clamp_i8(int(value)) for value in raw_values[:declared_length]]
    if len(iq_values) < 2:
        raise ValueError("CSI payload is empty")
    if len(iq_values) % 2:
        iq_values.pop()

    n_subcarriers = min(len(iq_values) // 2, MAX_SUBCARRIERS)
    iq_values = iq_values[: n_subcarriers * 2]
    payload = struct.pack(f"<{len(iq_values)}b", *iq_values)
    header = ADR018_HEADER.pack(
        ADR018_MAGIC,
        node_id,
        1,  # ESP32-S3 receiver reports one CSI antenna
        n_subcarriers,
        channel_frequency_mhz(channel),
        sequence,
        rssi,
        noise_floor,
        0,  # HT/legacy PPDU; csi_recv_router does not expose ADR-110 tagging
        0,  # ADR-110 flags
    )
    return header + payload


def bridge_node(
    node: Node,
    baud: int,
    target: tuple[str, int],
    max_fps: float,
    stop: threading.Event,
    counters: Counters,
) -> None:
    minimum_interval = 0.0 if max_fps <= 0 else 1.0 / max_fps
    last_sent = 0.0
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    while not stop.is_set():
        try:
            with serial.Serial(node.port, baud, timeout=1) as stream:
                # CH343 boards can reset when the port opens.  Leave both
                # control lines inactive and wait for the app to boot.
                stream.dtr = False
                stream.rts = False
                print(
                    f"[{node.port}] connected as node {node.node_id} "
                    f"at {baud} baud -> {target[0]}:{target[1]}",
                    flush=True,
                )
                while not stop.is_set():
                    raw_line = stream.readline()
                    if not raw_line:
                        continue
                    counters.serial_lines += 1
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("CSI_DATA,"):
                        continue

                    now = time.monotonic()
                    if minimum_interval and now - last_sent < minimum_interval:
                        continue
                    try:
                        packet = csi_csv_to_adr018(line, node.node_id)
                    except (csv.Error, json.JSONDecodeError, ValueError, TypeError):
                        counters.parse_errors += 1
                        continue
                    if packet is None:
                        continue
                    udp.sendto(packet, target)
                    counters.frames_sent += 1
                    last_sent = now
        except (OSError, serial.SerialException) as exc:
            counters.serial_errors += 1
            if not stop.is_set():
                print(f"[{node.port}] serial error: {exc}; retrying in 2s", flush=True)
                stop.wait(2)

    udp.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bridge ESP32 csi_recv_router serial CSI to RuView ADR-018 UDP"
    )
    parser.add_argument(
        "--node",
        action="append",
        type=parse_node,
        required=True,
        metavar="PORT:NODE_ID",
        help="serial port and stable RuView node ID; repeat for multiple boards",
    )
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--target-ip", default="127.0.0.1")
    parser.add_argument("--target-port", type=int, default=5005)
    parser.add_argument(
        "--max-fps",
        type=float,
        default=10.0,
        help="maximum forwarded frames per second per node; 0 disables throttling",
    )
    parser.add_argument("--status-interval", type=float, default=5.0)
    args = parser.parse_args()

    duplicate_ports = len({node.port.upper() for node in args.node}) != len(args.node)
    duplicate_ids = len({node.node_id for node in args.node}) != len(args.node)
    if duplicate_ports or duplicate_ids:
        parser.error("every serial port and node ID must be unique")

    stop = threading.Event()
    counters = {node.port: Counters() for node in args.node}
    threads = [
        threading.Thread(
            target=bridge_node,
            name=f"serial-{node.port}",
            args=(
                node,
                args.baud,
                (args.target_ip, args.target_port),
                args.max_fps,
                stop,
                counters[node.port],
            ),
        )
        for node in args.node
    ]
    for thread in threads:
        thread.start()

    print("Press Ctrl+C to stop.", flush=True)
    try:
        while all(thread.is_alive() for thread in threads):
            stop.wait(max(1.0, args.status_interval))
            if stop.is_set():
                break
            summary = " | ".join(
                f"{node.port}: sent={counters[node.port].frames_sent} "
                f"parse_err={counters[node.port].parse_errors} "
                f"serial_err={counters[node.port].serial_errors}"
                for node in args.node
            )
            print(summary, flush=True)
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=3)

    return 0


if __name__ == "__main__":
    sys.exit(main())
