#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("[kaggle_agentic_pipeline] $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle pipeline for synthetic data + baseline + agentic GRPO.")
    parser.add_argument("--root", default="/kaggle/working/codeforgefinal")
    parser.add_argument("--bootstrap-deps", action="store_true")
    parser.add_argument("--core-size", type=int, default=5000)
    parser.add_argument("--repair-size", type=int, default=2000)
    parser.add_argument("--dev-size", type=int, default=500)
    parser.add_argument("--hard-size", type=int, default=800)
    parser.add_argument("--validate-sample", type=int, default=64)
    parser.add_argument("--agentic-config", default="configs/agentic_grpo.qwen35_2b.yaml")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--prompts-per-iteration", type=int, default=6)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--repair-steps", type=int, default=2)
    parser.add_argument("--max-episode-steps", type=int, default=3)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    if args.bootstrap_deps:
        run([sys.executable, str(root / "scripts" / "bootstrap_kaggle.py")], cwd=root)

    run(
        [
            sys.executable,
            str(root / "scripts" / "generate_synthetic_tasks.py"),
            "--out-dir",
            "data/generated",
            "--core-size",
            str(args.core_size),
            "--repair-size",
            str(args.repair_size),
            "--dev-size",
            str(args.dev_size),
            "--hard-size",
            str(args.hard_size),
            "--validate-sample",
            str(args.validate_sample),
        ],
        cwd=root,
    )

    if not args.skip_baseline:
        run(
            [
                sys.executable,
                str(root / "scripts" / "run_ranked_sampling.py"),
                "--config",
                "configs/base.yaml",
                "--tasks",
                "data/generated/dev.jsonl",
                "--out",
                "artifacts/predictions.jsonl",
                "--num-candidates",
                "4",
                "--repair-steps",
                "1",
            ],
            cwd=root,
        )
        run(
            [
                sys.executable,
                str(root / "scripts" / "eval.py"),
                "--config",
                "configs/base.yaml",
                "--tasks",
                "data/generated/dev.jsonl",
                "--predictions",
                "artifacts/predictions.jsonl",
                "--ks",
                "1,3,5",
            ],
            cwd=root,
        )

    run(
        [
            sys.executable,
            str(root / "scripts" / "run_agentic_grpo.py"),
            "--config",
            args.agentic_config,
            "--tasks",
            "data/generated/train.jsonl",
            "--iterations",
            str(args.iterations),
            "--prompts-per-iteration",
            str(args.prompts_per_iteration),
            "--num-candidates",
            str(args.num_candidates),
            "--repair-steps",
            str(args.repair_steps),
            "--max-episode-steps",
            str(args.max_episode_steps),
        ],
        cwd=root,
    )

    print(
        "[kaggle_agentic_pipeline] done | "
        "dataset=data/generated train=artifacts/agentic_grpo/metrics.jsonl",
        flush=True,
    )


if __name__ == "__main__":
    main()
