#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agentic import (
    EpisodeRecord,
    SamplingConfig,
    flatten_episode_candidates,
    flatten_episode_rows,
    run_repair_episode,
    summarize_episodes,
)
from src.data import load_tasks_jsonl
from src.trainer import (
    load_config,
    maybe_build_train_bundle,
    maybe_push_checkpoint_to_hub,
    run_grpo_update_manual,
)
from src.utils import ensure_dir
from src.verifier import ObjectiveVerifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compiler-guided repair GRPO-style training.")
    parser.add_argument("--config", default="configs/agentic_grpo.qwen35_2b.yaml")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--prompts-per-iteration", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--repair-steps", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume-from", default="latest")
    return parser.parse_args()


def latest_checkpoint_dir(checkpoints_root: Path) -> Path | None:
    if not checkpoints_root.exists():
        return None
    best_iter = -1
    best_path: Path | None = None
    for path in checkpoints_root.iterdir():
        if not path.is_dir() or not path.name.startswith("iter_"):
            continue
        try:
            iter_id = int(path.name.split("_", 1)[1])
        except ValueError:
            continue
        if not ((path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()):
            continue
        if iter_id > best_iter:
            best_iter = iter_id
            best_path = path
    return best_path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_checkpoint(bundle: Any, checkpoint_dir: Path, metrics: dict[str, Any]) -> None:
    ensure_dir(checkpoint_dir)
    bundle.model.save_pretrained(str(checkpoint_dir))
    bundle.tokenizer.save_pretrained(str(checkpoint_dir))
    (checkpoint_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def build_sft_rows(episodes: list[EpisodeRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        if episode.final_best is not None and episode.final_best.correct:
            rows.append(
                {
                    "task_id": episode.task_id,
                    "prompt": episode.steps[0].prompt_text if episode.steps else "",
                    "completion": episode.final_best.asm.strip() + "\n",
                    "source": "agentic_final",
                }
            )
        for step_idx, step in enumerate(episode.steps[1:], start=1):
            if not step.kept_improved:
                continue
            rows.append(
                {
                    "task_id": episode.task_id,
                    "prompt": step.prompt_text,
                    "completion": step.best.asm.strip() + "\n",
                    "source": f"agentic_repair_step_{step_idx}",
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    training = cfg.raw.setdefault("training", {})
    paths = cfg.raw.setdefault("paths", {})
    agentic = cfg.raw.setdefault("agentic", {})
    reward_cfg = cfg.raw.get("reward_weights", cfg.raw.get("reward"))

    if args.iterations is not None:
        training["iterations"] = int(args.iterations)
    if args.prompts_per_iteration is not None:
        training["prompts_per_iteration"] = int(args.prompts_per_iteration)
    if args.seed is not None:
        cfg.raw.setdefault("project", {})["seed"] = int(args.seed)

    tasks_path = Path(args.tasks or agentic.get("tasks_path") or paths.get("train_tasks") or "data/sample_tasks.jsonl")
    tasks = load_tasks_jsonl(tasks_path)
    if not tasks:
        raise RuntimeError(f"No tasks loaded from {tasks_path}")

    artifacts_root = Path(paths.get("artifacts_dir", "artifacts")) / "agentic_grpo"
    checkpoints_root = Path(paths.get("checkpoints_dir", "checkpoints")) / "agentic_grpo"
    ensure_dir(artifacts_root)
    ensure_dir(checkpoints_root)
    ensure_dir(artifacts_root / "trajectories")
    ensure_dir(artifacts_root / "sft")

    resume_from: str | None = None
    if args.resume_from and args.resume_from != "none":
        if args.resume_from == "latest":
            latest = latest_checkpoint_dir(checkpoints_root)
            resume_from = None if latest is None else str(latest)
        else:
            resume_from = args.resume_from

    bundle = maybe_build_train_bundle(cfg, resume_from=resume_from)
    if bundle is None:
        raise RuntimeError("Training bundle was not created. Set training.dry_run=false after bootstrap.")

    sampling = SamplingConfig(
        max_new_tokens=int(training.get("max_new_tokens", 128)),
        temperature=float(training.get("temperature", 0.55)),
        top_p=float(training.get("top_p", 0.80)),
        top_k=int(training["top_k"]) if training.get("top_k") is not None else None,
        min_p=float(training["min_p"]) if training.get("min_p") is not None else None,
        repetition_penalty=float(training.get("repetition_penalty", 1.03)),
    )

    verifier = ObjectiveVerifier(
        artifacts_dir=artifacts_root / "verify",
        timeout_seconds=int(agentic.get("verifier_timeout_sec", cfg.raw.get("runtime", {}).get("verifier_timeout_sec", 6))),
        reward_weights=reward_cfg,
    )

    iterations = int(training.get("iterations", 20))
    prompts_per_iteration = int(training.get("prompts_per_iteration", 4))
    num_candidates = max(1, int(args.num_candidates or agentic.get("num_candidates") or training.get("generations_per_prompt", 4)))
    repair_steps = max(0, int(args.repair_steps if args.repair_steps is not None else agentic.get("repair_steps", 1)))
    max_episode_steps = int(
        args.max_episode_steps
        if args.max_episode_steps is not None
        else agentic.get("max_episode_steps", cfg.raw.get("runtime", {}).get("max_episode_steps", 3))
    )
    repair_gain_weight = float(agentic.get("repair_gain_weight", 0.10))
    regress_penalty = float(cfg.raw.get("reward", {}).get("regress_penalty", -0.10))
    save_every = int(args.save_every or agentic.get("save_every", 5))
    seed = int(cfg.raw.get("project", {}).get("seed", 42))
    rng = random.Random(seed)

    history_path = artifacts_root / "metrics.jsonl"

    for iteration in range(1, iterations + 1):
        if len(tasks) <= prompts_per_iteration:
            batch_tasks = list(tasks)
            rng.shuffle(batch_tasks)
        else:
            batch_tasks = rng.sample(tasks, k=prompts_per_iteration)

        episodes = [
            run_repair_episode(
                task=task,
                verifier=verifier,
                generator=bundle.generator,
                sampling=sampling,
                num_candidates=num_candidates,
                repair_steps=repair_steps,
                max_episode_steps=max_episode_steps,
                repair_gain_weight=repair_gain_weight,
                regress_penalty=regress_penalty,
            )
            for task in batch_tasks
        ]

        train_rows = flatten_episode_rows(episodes, include_all_candidates=True)
        if not train_rows:
            raise RuntimeError("No training rows were produced from the agentic rollouts.")
        train_metrics = run_grpo_update_manual(train_rows, cfg, bundle)
        episode_metrics = summarize_episodes(episodes)
        metrics = {
            "iteration": iteration,
            "tasks_in_batch": len(batch_tasks),
            "train_rows": len(train_rows),
            "num_candidates": num_candidates,
            "repair_steps": repair_steps,
            "max_episode_steps": max_episode_steps,
            "repair_gain_weight": repair_gain_weight,
            **episode_metrics,
            **train_metrics,
        }

        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        trajectory_rows = flatten_episode_candidates(episodes)
        write_jsonl(artifacts_root / "trajectories" / f"iter_{iteration:04d}.jsonl", trajectory_rows)
        sft_rows = build_sft_rows(episodes)
        if sft_rows:
            write_jsonl(artifacts_root / "sft" / f"iter_{iteration:04d}.jsonl", sft_rows)

        print(
            "[run_agentic_grpo] "
            f"iter={iteration} solved_rate={metrics['solved_rate']:.4f} "
            f"avg_final_reward={metrics['avg_final_reward']:.4f} "
            f"repair_gain={metrics['avg_repair_gain']:.4f} "
            f"grpo_loss={metrics['grpo_loss']:.4f} "
            f"skipped_rows={int(metrics.get('skipped_rows', 0))}",
            flush=True,
        )

        if iteration % save_every == 0 or iteration == iterations:
            ckpt_dir = checkpoints_root / f"iter_{iteration}"
            save_checkpoint(bundle, ckpt_dir, metrics)
            maybe_push_checkpoint_to_hub(cfg, ckpt_dir, iteration)

    print(
        json.dumps(
            {
                "iterations": iterations,
                "history": str(history_path),
                "artifacts_dir": str(artifacts_root),
                "checkpoints_dir": str(checkpoints_root),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
