#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str], check: bool = False) -> None:
    print("[bootstrap_kaggle] $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=check)


def main() -> None:
    run(["apt-get", "update", "-y"])
    run(["apt-get", "install", "-y", "nasm", "binutils"])
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--upgrade-strategy",
        "only-if-needed",
        "trl>=0.27.2",
        "peft>=0.18.1",
        "transformers>=5.2.0,<5.3.0",
        "accelerate>=1.11.0",
        "datasets>=4.6.1",
        "wandb>=0.25.0",
        "huggingface_hub>=0.36.2",
        "pyyaml>=6.0.2",
        "bitsandbytes>=0.49.2",
        "sentencepiece>=0.2.0",
        "tqdm>=4.67.0",
        "matplotlib>=3.9.0",
    ], check=True)
    print("[bootstrap_kaggle] done", flush=True)


if __name__ == "__main__":
    main()
