#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import load_tasks_jsonl
from src.utils import load_yaml
from src.verifier import ObjectiveVerifier


EXIT42 = """global _start
section .text
_start:
    mov rax, 60
    mov rdi, 42
    syscall"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--tasks", default="data/sample_tasks.jsonl")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    tasks = load_tasks_jsonl(args.tasks)
    verifier = ObjectiveVerifier(
        artifacts_dir=Path(cfg["runtime"]["artifacts_dir"]) / "smoke_test",
        timeout_seconds=int(cfg["runtime"]["verifier_timeout_sec"]),
        reward_weights=cfg.get("reward"),
    )
    result = verifier.evaluate(tasks[0], EXIT42, sample_id="exit42")
    print({
        "task_id": tasks[0].task_id,
        "reward": result.reward,
        "correct": result.correct,
        "stage_failed": result.stage_failed,
        "exit_code": result.exit_code,
    })


if __name__ == "__main__":
    main()
