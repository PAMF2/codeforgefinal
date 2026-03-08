from __future__ import annotations

import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "experiments" / "autoresearch_adapter" / "runs"
DATA_TEMPLATE = ROOT / "data" / "generated"


def run_cmd(cmd: list[str], cwd: Path, capture: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=str(cwd), check=True, text=True, capture_output=capture)
    if capture and result.stderr:
        print(result.stderr, file=sys.stderr)
    return result


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_data(out_dir: Path, core_size: int, dev_size: int, repair_size: int, hard_size: int) -> None:
    ensure_dir(out_dir)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "generate_synthetic_tasks.py"),
        "--out-dir",
        str(out_dir),
        "--core-size",
        str(core_size),
        "--dev-size",
        str(dev_size),
        "--repair-size",
        str(repair_size),
        "--hard-size",
        str(hard_size),
        "--validate-sample",
        "8",
    ]
    run_cmd(cmd, ROOT)


def run_ranked_sampling(tasks_path: Path, output_path: Path, args: dict[str, int]) -> dict[str, float]:
    ensure_dir(output_path.parent)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_ranked_sampling.py"),
        "--config",
        "configs/base.yaml",
        "--tasks",
        str(tasks_path),
        "--out",
        str(output_path),
        "--num-candidates",
        str(args["num_candidates"]),
        "--repair-steps",
        str(args["repair_steps"]),
    ]
    full = run_cmd(cmd, ROOT, capture=True)
    final_lines = [line for line in full.stdout.splitlines() if line.strip().startswith("{")]
    if not final_lines:
        raise RuntimeError("No JSON result from run_ranked_sampling")
    metrics = json.loads(final_lines[-1])
    return metrics


def run_experiments():
    ensure_dir(EXPERIMENT_DIR)
    variants = [
        {"dev_size": 30, "num_candidates": 2, "repair_steps": 0, "temperature": 0.4},
        {"dev_size": 40, "num_candidates": 2, "repair_steps": 1, "temperature": 0.45},
        {"dev_size": 50, "num_candidates": 3, "repair_steps": 1, "temperature": 0.5},
    ]
    best_metric = {"top1_correct_rate": 0.0}
    for idx, variant in enumerate(variants, start=1):
        run_dir = EXPERIMENT_DIR / f"run_{idx}_{datetime.utcnow():%Y%m%d%H%M%S}"
        data_dir = run_dir / "data"
        artifacts_dir = run_dir / "artifacts"
        ensure_dir(run_dir)
        generate_data(
            out_dir=data_dir,
            core_size=1200,
            dev_size=variant["dev_size"],
            repair_size=600,
            hard_size=150,
        )
        metrics = run_ranked_sampling(
            tasks_path=data_dir / "dev.jsonl",
            output_path=artifacts_dir / "predictions.jsonl",
            args={
                "num_candidates": variant["num_candidates"],
                "repair_steps": variant["repair_steps"],
            },
        )
        metrics["variant"] = variant
        log_path = run_dir / "metrics.json"
        log_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"run {idx} metrics:", json.dumps(metrics, indent=2))
        if metrics.get("top1_correct_rate", 0.0) >= best_metric.get("top1_correct_rate", 0.0):
            best_metric = metrics
            (EXPERIMENT_DIR / "best_metric.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run_experiments()
