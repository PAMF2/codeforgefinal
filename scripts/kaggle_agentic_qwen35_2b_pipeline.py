#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.kaggle_runtime import load_kaggle_secrets


def run(cmd: list[str], cwd: Path) -> None:
    print("[kaggle_agentic_pipeline] $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle pipeline for synthetic data + baseline + agentic GRPO.")
    parser.add_argument("--root", default="/kaggle/working/codeforgefinal")
    parser.add_argument("--bootstrap-deps", action="store_true")
    parser.add_argument("--core-size", type=int, default=1200)
    parser.add_argument("--repair-size", type=int, default=400)
    parser.add_argument("--dev-size", type=int, default=120)
    parser.add_argument("--hard-size", type=int, default=120)
    parser.add_argument("--validate-sample", type=int, default=16)
    parser.add_argument("--agentic-config", default="configs/agentic_grpo.qwen35_2b.yaml")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--prompts-per-iteration", type=int, default=3)
    parser.add_argument("--num-candidates", type=int, default=2)
    parser.add_argument("--repair-steps", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=2)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--agentic-retries", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    load_kaggle_secrets(prefix="[kaggle_agentic_pipeline]")

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

    agentic_cmd = [
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
        "--resume-from",
        "none",
    ]

    try:
        run(agentic_cmd, cwd=root)
    except subprocess.CalledProcessError:
        if args.agentic_retries <= 0:
            raise

        retry_cmd = [
            sys.executable,
            str(root / "scripts" / "run_agentic_grpo.py"),
            "--config",
            args.agentic_config,
            "--tasks",
            "data/generated/train.jsonl",
            "--iterations",
            str(max(1, min(args.iterations, 2))),
            "--prompts-per-iteration",
            str(max(2, min(args.prompts_per_iteration, 3))),
            "--num-candidates",
            str(max(1, min(args.num_candidates, 2))),
            "--repair-steps",
            str(max(0, min(args.repair_steps, 1))),
            "--max-episode-steps",
            str(max(1, min(args.max_episode_steps, 2))),
            "--resume-from",
            "none",
        ]
        print(
            "[kaggle_agentic_pipeline] agentic stage failed once; retrying with safer settings",
            flush=True,
        )
        run(retry_cmd, cwd=root)

    print(
        "[kaggle_agentic_pipeline] done | "
        "dataset=data/generated train=artifacts/agentic_grpo/metrics.jsonl",
        flush=True,
    )


if __name__ == "__main__":
    main()
