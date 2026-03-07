#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import load_tasks_jsonl
from src.utils import load_yaml
from src.verifier import ObjectiveVerifier


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def parse_ks(raw: str) -> list[int]:
    return sorted({int(part.strip()) for part in raw.split(",") if part.strip()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ks", default="1,3,5")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    tasks = {task.task_id: task for task in load_tasks_jsonl(args.tasks)}
    preds = read_jsonl(Path(args.predictions))
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in preds:
        by_task.setdefault(str(row["task_id"]), []).append(row)
    for rows in by_task.values():
        rows.sort(key=lambda row: int(row.get("candidate_rank", 999999)))

    verifier = ObjectiveVerifier(
        artifacts_dir=Path(cfg["runtime"]["artifacts_dir"]) / "eval",
        timeout_seconds=int(cfg["runtime"]["verifier_timeout_sec"]),
        reward_weights=cfg.get("reward"),
    )

    ks = parse_ks(args.ks)
    top1_correct = 0
    top1_reward_sum = 0.0
    pass_at = {k: 0 for k in ks}

    for task_id, task in tasks.items():
        rows = by_task.get(task_id, [])
        results: list[bool] = []
        top1_reward = 0.0
        for idx, row in enumerate(rows):
            result = verifier.evaluate(task, str(row["asm"]), sample_id=f"{task_id}-eval-{idx}")
            results.append(result.correct)
            if idx == 0:
                top1_correct += int(result.correct)
                top1_reward = result.reward
        top1_reward_sum += top1_reward
        for k in ks:
            pass_at[k] += int(any(results[:k]))

    total = max(1, len(tasks))
    summary = {
        "tasks_total": len(tasks),
        "correct_at_1": top1_correct / total,
        "avg_reward_at_1": top1_reward_sum / total,
        "pass_at": {str(k): pass_at[k] / total for k in ks},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
