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


def load_tasks_jsonl(path: str | Path) -> list[Task]:
    rows: list[Task] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rows.append(
            Task(
                task_id=str(obj["task_id"]),
                instruction=str(obj["instruction"]),
                tier=int(obj.get("tier", 1)),
                expected_stdout=obj.get("expected_stdout"),
                expected_exit_code=obj.get("expected_exit_code"),
                hidden_tests=list(obj.get("hidden_tests", [])),
            )
        )
    return rows
