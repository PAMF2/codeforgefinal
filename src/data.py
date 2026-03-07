from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Task:
    task_id: str
    instruction: str
    tier: int = 1
    expected_stdout: str | None = None
    expected_exit_code: int | None = None
    hidden_tests: list[dict[str, Any]] = field(default_factory=list)
    task_kind: str = "generate"
    family_id: str = ""
    starter_code: str | None = None
    reference_solution: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_tasks_jsonl(path: str | Path) -> list[Task]:
    rows: list[Task] = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        known_keys = {
            "task_id",
            "instruction",
            "tier",
            "expected_stdout",
            "expected_exit_code",
            "hidden_tests",
            "task_kind",
            "family_id",
            "starter_code",
            "reference_solution",
        }
        rows.append(
            Task(
                task_id=str(obj["task_id"]),
                instruction=str(obj["instruction"]),
                tier=int(obj.get("tier", 1)),
                expected_stdout=obj.get("expected_stdout"),
                expected_exit_code=obj.get("expected_exit_code"),
                hidden_tests=list(obj.get("hidden_tests", [])),
                task_kind=str(obj.get("task_kind", "generate")),
                family_id=str(obj.get("family_id", "")),
                starter_code=obj.get("starter_code"),
                reference_solution=obj.get("reference_solution"),
                metadata={k: v for k, v in obj.items() if k not in known_keys},
            )
        )
    return rows
