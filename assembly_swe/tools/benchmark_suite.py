from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


def run(cmd: list[str], cwd: Path) -> None:
    print("[benchmark_suite] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_tasks(raw: str) -> list[str]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if not items:
        raise ValueError("No task files provided")
    return items


def load_aggregate(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "best_correct_rate_at_1": 0.0,
            "best_iter": None,
            "mean_correct_rate_at_1": 0.0,
            "std_correct_rate_at_1": 0.0,
            "mean_assembly_rate_at_1": 0.0,
            "mean_avg_reward_at_1": 0.0,
        }
    vals_correct = [float(r.get("correct_rate_at_1", 0.0) or 0.0) for r in rows]
    vals_assembly = [float(r.get("assembly_rate_at_1", 0.0) or 0.0) for r in rows]
    vals_reward = [float(r.get("avg_reward_at_1", 0.0) or 0.0) for r in rows]
    best = max(rows, key=lambda r: (float(r.get("correct_rate_at_1", 0.0) or 0.0), float(r.get("avg_reward_at_1", 0.0) or 0.0)))
    return {
        "n": len(rows),
        "best_correct_rate_at_1": float(best.get("correct_rate_at_1", 0.0) or 0.0),
        "best_iter": best.get("iter"),
        "mean_correct_rate_at_1": statistics.mean(vals_correct),
        "std_correct_rate_at_1": statistics.pstdev(vals_correct) if len(vals_correct) > 1 else 0.0,
        "mean_assembly_rate_at_1": statistics.mean(vals_assembly),
        "mean_avg_reward_at_1": statistics.mean(vals_reward),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Assembly-SWE benchmark on multiple task files and build suite summary.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--tasks", required=True, help="Comma-separated task jsonl files (e.g. dev_a.jsonl,dev_b.jsonl)")
    ap.add_argument("--iter-start", type=int, default=1)
    ap.add_argument("--iter-end", type=int, default=30)
    ap.add_argument("--ks", default="1,3,5")
    ap.add_argument("--hub-repo-id", default="mistral-hackaton-2026/codeforge")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--num-candidates", type=int, default=1)
    ap.add_argument("--verifier", choices=["none", "reward"], default="none")
    ap.add_argument("--verifier-timeout-sec", type=int, default=5)
    ap.add_argument("--repair-steps", type=int, default=0)
    ap.add_argument("--outdir", default="assembly_swe/results/suite")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    outdir = (repo_root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    task_files = parse_tasks(args.tasks)

    suite_rows: list[dict[str, Any]] = []
    for task in task_files:
        task_name = Path(task).stem
        task_out = outdir / task_name
        task_out.mkdir(parents=True, exist_ok=True)
        run_cmd = [
            sys.executable,
            "assembly_swe/tools/eval_all_iters.py",
            "--repo-root",
            str(repo_root),
            "--tasks",
            task,
            "--iter-start",
            str(args.iter_start),
            "--iter-end",
            str(args.iter_end),
            "--ks",
            args.ks,
            "--outdir",
            str(task_out.relative_to(repo_root)),
            "--hub-repo-id",
            args.hub_repo_id,
            "--num-candidates",
            str(args.num_candidates),
            "--verifier",
            args.verifier,
            "--verifier-timeout-sec",
            str(args.verifier_timeout_sec),
            "--repair-steps",
            str(args.repair_steps),
        ]
        if args.load_in_4bit:
            run_cmd.append("--load-in-4bit")
        run(run_cmd, cwd=repo_root)

        agg = load_aggregate(task_out / "aggregate.json")
        rows = agg.get("rows", [])
        summary = summarize_rows(rows)
        suite_rows.append(
            {
                "task_file": task,
                "task_name": task_name,
                "count_evaluated": agg.get("count_evaluated", 0),
                "count_skipped": agg.get("count_skipped", 0),
                **summary,
                "aggregate_path": str((task_out / "aggregate.json").resolve()),
            }
        )

    # Macro summary across datasets
    if suite_rows:
        macro_mean_correct = statistics.mean(float(x["mean_correct_rate_at_1"]) for x in suite_rows)
        macro_mean_assembly = statistics.mean(float(x["mean_assembly_rate_at_1"]) for x in suite_rows)
        macro_mean_reward = statistics.mean(float(x["mean_avg_reward_at_1"]) for x in suite_rows)
    else:
        macro_mean_correct = macro_mean_assembly = macro_mean_reward = 0.0

    suite = {
        "repo_root": str(repo_root),
        "iter_start": args.iter_start,
        "iter_end": args.iter_end,
        "tasks": task_files,
        "datasets": suite_rows,
        "macro": {
            "macro_mean_correct_rate_at_1": macro_mean_correct,
            "macro_mean_assembly_rate_at_1": macro_mean_assembly,
            "macro_mean_avg_reward_at_1": macro_mean_reward,
        },
    }
    (outdir / "suite_summary.json").write_text(json.dumps(suite, indent=2), encoding="utf-8")
    print(json.dumps(suite, indent=2))


if __name__ == "__main__":
    main()
