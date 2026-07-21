#!/usr/bin/env python3
"""
Windows UDP intake bridge for RuView ESP32 CSI.

Why this exists:
  On Docker Desktop for Windows, UDP packets arriving from Wi-Fi/broadcast can
  reach Windows itself but fail to pass through Docker's published UDP port.
  This process binds the real ESP intake port on Windows and forwards packets
  locally to Docker on a second host port, which is much more reliable.

Default path:
  ESP32 nodes -> 255.255.255.255:5005
  this bridge -> 127.0.0.1:5006
  Docker host port 5006 -> container UDP 5005
"""

from __future__ import annotations

import argparse
import socket
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forward RuView ESP32 UDP packets into Docker.")
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=5005)
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--target-port", type=int, default=5006)
    parser.add_argument("--stats-every", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.bind((args.listen_host, args.listen_port))

    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.target_host, args.target_port)

    print(
        f"RuView UDP bridge listening on {args.listen_host}:{args.listen_port} "
        f"-> {args.target_host}:{args.target_port}",
        flush=True,
    )

    packets = 0
    bytes_total = 0
    last_stats = time.monotonic()

    while True:
        packet, _addr = rx.recvfrom(65535)
        tx.sendto(packet, target)
        packets += 1
        bytes_total += len(packet)

        now = time.monotonic()
        if args.stats_every > 0 and now - last_stats >= args.stats_every:
            print(f"forwarded packets={packets} bytes={bytes_total}", flush=True)
            last_stats = now


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("RuView UDP bridge stopped", file=sys.stderr)
        raise SystemExit(130)
