#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("[qwen35_2b_pipeline] $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def latest_iter(checkpoints_dir: Path) -> int:
    if not checkpoints_dir.exists():
        return -1
    out = -1
    for path in checkpoints_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("iter_"):
            continue
        try:
            out = max(out, int(path.name.split("_", 1)[1]))
        except ValueError:
            continue
    return out


def bootstrap_deps() -> None:
    print("[qwen35_2b_pipeline] [bootstrap] installing system deps", flush=True)
    subprocess.run(["apt-get", "update", "-y"], check=False)
    subprocess.run(["apt-get", "install", "-y", "nasm", "binutils"], check=False)
    print("[qwen35_2b_pipeline] [bootstrap] installing qwen3.5-compatible python deps", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--upgrade-strategy",
            "only-if-needed",
            "trl>=0.23.1",
            "peft>=0.18.1",
            "transformers>=5.2.0,<5.3.0",
            "accelerate>=1.11.0",
            "datasets>=4.6.1",
            "wandb>=0.25.0",
            "huggingface_hub>=0.36.2",
            "pyyaml>=6.0.2",
            "bitsandbytes>=0.49.2",
            "sentencepiece>=0.2.0",
        ],
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5-2B Kaggle pipeline for CodeForge ASM")
    parser.add_argument("--root", default="/kaggle/working/codeforgefinal")
    parser.add_argument("--phase1-config", default="configs/grpo_config.qwen35_2b_phase1.yaml")
    parser.add_argument("--phase2-config", default="configs/grpo_config.qwen35_2b_phase2.yaml")
    parser.add_argument("--phase1-hours", type=float, default=8.0)
    parser.add_argument("--phase2-hours", type=float, default=10.0)
    parser.add_argument("--bench-tasks", default="assembly_swe/datasets/dev_v1_30.jsonl")
    parser.add_argument("--bench-ks", default="1,3,5")
    parser.add_argument("--bench-candidates", type=int, default=5)
    parser.add_argument("--bench-verifier", choices=["none", "reward"], default="reward")
    parser.add_argument("--bench-verifier-timeout-sec", type=int, default=6)
    parser.add_argument("--bench-repair-steps", type=int, default=1)
    parser.add_argument("--bench-last-iters", type=int, default=15)
    parser.add_argument("--bench-outdir", default="assembly_swe/results/qwen35_2b_pipeline")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--bootstrap-deps", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    if args.bootstrap_deps:
        bootstrap_deps()

    if not args.skip_training:
        start = time.time()
        print("[qwen35_2b_pipeline] [phase1] manual warm-start", flush=True)
        run(
            [
                sys.executable,
                str(root / "scripts" / "kaggle_autorun.py"),
                "--root",
                str(root),
                "--config",
                args.phase1_config,
                "--hours",
                str(args.phase1_hours),
                "--backend",
                "manual",
                "--safe-profile",
                "--batch-size",
                "1",
                "--generations-per-prompt",
                "6",
                "--prompts-per-iteration",
                "8",
                "--gradient-accumulation-steps",
                "6",
                "--max-new-tokens",
                "128",
                "--use-mcts-after-iteration",
                "999",
            ],
            cwd=root,
        )
        print(f"[qwen35_2b_pipeline] [phase1] done in {(time.time() - start) / 60:.1f} min", flush=True)

        start = time.time()
        print("[qwen35_2b_pipeline] [phase2] trl grpo refinement", flush=True)
        run(
            [
                sys.executable,
                str(root / "scripts" / "kaggle_autorun.py"),
                "--root",
                str(root),
                "--config",
                args.phase2_config,
                "--hours",
                str(args.phase2_hours),
                "--backend",
                "trl",
                "--safe-profile",
                "--batch-size",
                "1",
                "--generations-per-prompt",
                "4",
                "--prompts-per-iteration",
                "6",
                "--gradient-accumulation-steps",
                "8",
                "--max-new-tokens",
                "128",
                "--use-mcts-after-iteration",
                "999",
            ],
            cwd=root,
        )
        print(f"[qwen35_2b_pipeline] [phase2] done in {(time.time() - start) / 60:.1f} min", flush=True)

    if args.skip_benchmark:
        print("[qwen35_2b_pipeline] done | training finished (benchmark skipped)", flush=True)
        return

    last = latest_iter(root / "checkpoints")
    if last < 1:
        raise RuntimeError("No checkpoints found; benchmark cannot run")
    iter_start = max(1, last - max(1, int(args.bench_last_iters)) + 1)
    print(
        "[qwen35_2b_pipeline] [benchmark] starting | "
        f"iters={iter_start}..{last} tasks={args.bench_tasks} ks={args.bench_ks} "
        f"candidates={args.bench_candidates} verifier={args.bench_verifier} "
        f"repair_steps={args.bench_repair_steps}",
        flush=True,
    )

    run(
        [
            sys.executable,
            str(root / "assembly_swe" / "tools" / "eval_all_iters.py"),
            "--repo-root",
            str(root),
            "--tasks",
            args.bench_tasks,
            "--iter-start",
            str(iter_start),
            "--iter-end",
            str(last),
            "--ks",
            args.bench_ks,
            "--outdir",
            args.bench_outdir,
            "--load-in-4bit",
            "--hub-repo-id",
            "PAMF2/codeforgefinal-qwen35-2b",
            "--base-model",
            "Qwen/Qwen3.5-2B",
            "--max-new-tokens",
            "128",
            "--temperature",
            "0.20",
            "--top-p",
            "0.80",
            "--top-k",
            "20",
            "--repetition-penalty",
            "1.05",
            "--num-candidates",
            str(args.bench_candidates),
            "--verifier",
            args.bench_verifier,
            "--verifier-timeout-sec",
            str(args.bench_verifier_timeout_sec),
            "--repair-steps",
            str(args.bench_repair_steps),
        ],
        cwd=root,
    )

    print(
        "[qwen35_2b_pipeline] done | "
        f"iters={iter_start}..{last} | "
        f"summary={root / args.bench_outdir / 'aggregate.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()

