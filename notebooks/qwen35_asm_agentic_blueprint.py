from __future__ import annotations

# %% [markdown]
# Qwen3.5-2B Assembly Agentic Blueprint

# %%
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass
class ExperimentConfig:
    repo_root: str = "/kaggle/working/codeforgefinal"
    model_id: str = "Qwen/Qwen3.5-2B"
    tasks_path: str = "data/sample_tasks.jsonl"
    artifacts_dir: str = "artifacts_notebook"
    num_candidates: int = 4
    repair_steps: int = 1
    verifier_timeout_sec: int = 6
    max_episode_steps: int = 3


CFG = ExperimentConfig()


def repo_path(*parts: str) -> Path:
    return Path(CFG.repo_root, *parts)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    tasks = read_jsonl(repo_path(CFG.tasks_path))
    print(f"loaded {len(tasks)} task(s)")
