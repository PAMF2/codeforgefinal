#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path


def run(cmd, cwd):
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Run full paper eval pipeline for Assembly-SWE.")
    ap.add_argument("--repo-root", required=True, help="Repository root path.")
    ap.add_argument("--tasks", required=True, help="Tasks JSONL path (relative to repo root or absolute).")
    ap.add_argument("--iter-start", type=int, default=1)
    ap.add_argument("--iter-end", type=int, default=30)
    ap.add_argument("--ks", default="1,3,5")
    ap.add_argument("--outdir", default="assembly_swe/results/all_iters_paper")
    ap.add_argument("--hub-repo-id", default="mistral-hackaton-2026/codeforge")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--num-candidates", type=int, default=1)
    ap.add_argument("--verifier", choices=["none", "reward"], default="none")
    ap.add_argument("--verifier-timeout-sec", type=int, default=5)
    ap.add_argument("--repair-steps", type=int, default=0)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    outdir = (repo_root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    eval_cmd = [
        sys.executable,
        "assembly_swe/tools/eval_all_iters.py",
        "--repo-root",
        str(repo_root),
        "--tasks",
        args.tasks,
        "--iter-start",
        str(args.iter_start),
        "--iter-end",
        str(args.iter_end),
        "--ks",
        args.ks,
        "--outdir",
        args.outdir,
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
        eval_cmd.append("--load-in-4bit")

    run(eval_cmd, cwd=repo_root)

    aggregate_path = outdir / "aggregate.json"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"aggregate.json not found at {aggregate_path}")

    with aggregate_path.open("r", encoding="utf-8") as f:
        agg = json.load(f)
    rows = agg.get("rows", [])
    if not rows:
        raise RuntimeError("No rows in aggregate.json.")

    # Save CSV for paper tables
    csv_path = outdir / "results.csv"
    headers = [
        "iter",
        "correct_rate_at_1",
        "assembly_rate_at_1",
        "avg_reward_at_1",
        "pass_at_1",
        "pass_at_3",
        "pass_at_5",
        "summary_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(headers) + "\n")
        for r in sorted(rows, key=lambda x: x.get("iter", 0)):
            p = r.get("pass_at", {})
            vals = [
                r.get("iter", ""),
                r.get("correct_rate_at_1", ""),
                r.get("assembly_rate_at_1", ""),
                r.get("avg_reward_at_1", ""),
                p.get("1", ""),
                p.get("3", ""),
                p.get("5", ""),
                r.get("summary_path", ""),
            ]
            f.write(",".join(map(str, vals)) + "\n")

    # Package results folder
    tar_path = repo_root / "assembly_swe_results_paper.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(outdir, arcname=str(Path(args.outdir)))

    best = max(rows, key=lambda x: (x.get("correct_rate_at_1", -1), x.get("avg_reward_at_1", -1)))
    print("[done] aggregate:", aggregate_path)
    print("[done] csv:", csv_path)
    print("[done] tar:", tar_path)
    print(
        "[best] iter={iter} correct@1={cr:.4f} assembly@1={ar:.4f} reward@1={rw:.4f}".format(
            iter=best.get("iter"),
            cr=best.get("correct_rate_at_1", 0.0),
            ar=best.get("assembly_rate_at_1", 0.0),
            rw=best.get("avg_reward_at_1", 0.0),
        )
    )


if __name__ == "__main__":
    main()
