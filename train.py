#!/usr/bin/env python
# /// script
# dependencies = [
#   "pyyaml>=6.0.2",
# ]
# ///

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys


def run(cmd: list[str]) -> int:
    print("[train.py]$", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def ensure_linux_system_deps() -> None:
    if platform.system().lower() != "linux":
        print("[train.py] Skipping NASM/binutils install: non-Linux runtime")
        return

    if shutil.which("nasm") and shutil.which("ld"):
        print("[train.py] NASM and ld already available")
        return

    if shutil.which("apt-get") is None:
        print("[train.py] apt-get not available; cannot auto-install nasm/binutils")
        return

    print("[train.py] Installing NASM/binutils via apt-get")
    subprocess.run(["apt-get", "update"], check=False)
    subprocess.run(["apt-get", "install", "-y", "nasm", "binutils"], check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo_config.yaml")
    parser.add_argument("--ensure-system-deps", action="store_true")
    parser.add_argument("--start-iter", type=int, default=0,
                        help="Resume from this iteration (loads iter_{N-1} checkpoint)")
    args = parser.parse_args()

    if args.ensure_system_deps:
        ensure_linux_system_deps()

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")

    cmd = [sys.executable, "-m", "src.trainer", "--config", args.config,
           "--start-iter", str(args.start_iter)]
    return subprocess.run(cmd, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
