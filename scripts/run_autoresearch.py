from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoresearch loop for CodeForge GRPO.")
    parser.add_argument("--experiments", type=int, default=4)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "agentic_grpo.qwen35_2b.yaml")
    parser.add_argument(
        "--target-config",
        type=Path,
        default=REPO_ROOT / "experiments" / "autoresearch_grpo" / "target_config.yaml",
    )
    parser.add_argument("--tasks", type=Path, default=REPO_ROOT / "data" / "generated" / "train.jsonl")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--prompts-per-iteration", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--repair-steps", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-budget", type=int, default=60, help="Per-experiment timeout in minutes.")
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "artifacts" / "autoresearch" / "grpo")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} is not a YAML mapping.")
    return data


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = copy.deepcopy(cfg)
    training = updated.setdefault("training", {})
    agentic = updated.setdefault("agentic", {})
    if args.iterations is not None:
        training["iterations"] = int(args.iterations)
    if args.prompts_per_iteration is not None:
        training["prompts_per_iteration"] = int(args.prompts_per_iteration)
    if args.num_candidates is not None:
        agentic["num_candidates"] = int(args.num_candidates)
    if args.repair_steps is not None:
        agentic["repair_steps"] = int(args.repair_steps)
    if args.max_episode_steps is not None:
        agentic["max_episode_steps"] = int(args.max_episode_steps)
    updated.setdefault("project", {})["seed"] = int(args.seed)
    return updated


def ensure_target_config(target_config: Path, source_cfg: dict[str, Any]) -> dict[str, Any]:
    if target_config.exists():
        return load_yaml(target_config)
    dump_yaml(target_config, source_cfg)
    return copy.deepcopy(source_cfg)


def read_last_metric(metrics_path: Path) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    if not metrics_path.exists():
        return latest
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                latest = json.loads(line)
            except json.JSONDecodeError:
                continue
    return latest


def metric_tuple(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics.get("solved_rate", 0.0)),
        float(metrics.get("avg_final_reward", 0.0)),
        float(metrics.get("avg_repair_gain", 0.0)),
        -float(metrics.get("skipped_rows", 0.0)),
    )


def short_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def init_results_tsv(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "commit\tsolved_rate\tavg_final_reward\tstatus\tdescription\n",
        encoding="utf-8",
    )


def append_results_tsv(path: Path, commit: str, metrics: dict[str, Any], status: str, description: str) -> None:
    solved_rate = float(metrics.get("solved_rate", 0.0))
    avg_reward = float(metrics.get("avg_final_reward", 0.0))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{commit}\t{solved_rate:.6f}\t{avg_reward:.6f}\t{status}\t{description}\n")


def set_experiment_paths(cfg: dict[str, Any], exp_dir: Path) -> Path:
    exp_cfg = copy.deepcopy(cfg)
    paths = exp_cfg.setdefault("paths", {})
    paths["artifacts_dir"] = str(exp_dir / "artifacts")
    paths["checkpoints_dir"] = str(exp_dir / "checkpoints")
    cfg_path = exp_dir / "config.yaml"
    dump_yaml(cfg_path, exp_cfg)
    return cfg_path


def build_command(cfg_path: Path, tasks_path: Path, seed: int) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_agentic_grpo.py"),
        "--config",
        str(cfg_path),
        "--tasks",
        str(tasks_path),
        "--seed",
        str(seed),
        "--resume-from",
        "none",
    ]


def run_grpo_experiment(exp_dir: Path, cfg: dict[str, Any], tasks_path: Path, seed: int, timeout_minutes: int) -> tuple[str, dict[str, Any], Path]:
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = set_experiment_paths(cfg, exp_dir)
    run_log = exp_dir / "run.log"
    metrics_path = exp_dir / "artifacts" / "agentic_grpo" / "metrics.jsonl"
    command = build_command(cfg_path, tasks_path, seed)
    timeout_seconds = None if timeout_minutes <= 0 else timeout_minutes * 60

    try:
        with run_log.open("w", encoding="utf-8") as handle:
            subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout_seconds,
            )
    except subprocess.TimeoutExpired:
        return "crash", {}, run_log
    except subprocess.CalledProcessError:
        return "crash", read_last_metric(metrics_path), run_log

    return "ok", read_last_metric(metrics_path), run_log


def bounded_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def bounded_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def mutate_config(best_cfg: dict[str, Any], rng: random.Random) -> tuple[dict[str, Any], str]:
    mutated = copy.deepcopy(best_cfg)
    training = mutated.setdefault("training", {})
    agentic = mutated.setdefault("agentic", {})
    model = mutated.setdefault("model", {})
    reward = mutated.setdefault("reward", {})

    mutations = [
        "learning_rate",
        "prompts_per_iteration",
        "num_candidates",
        "repair_steps",
        "max_episode_steps",
        "temperature",
        "top_p",
        "lora_r",
        "hidden_test",
        "repair_gain_weight",
    ]
    choice = rng.choice(mutations)

    if choice == "learning_rate":
        current = float(training.get("learning_rate", 2e-6))
        factor = rng.choice([0.7, 0.85, 1.15, 1.3])
        training["learning_rate"] = float(f"{bounded_float(current * factor, 5e-7, 2e-5):.8f}")
        return mutated, f"mutate learning_rate -> {training['learning_rate']}"
    if choice == "prompts_per_iteration":
        current = int(training.get("prompts_per_iteration", 3))
        training["prompts_per_iteration"] = bounded_int(current + rng.choice([-1, 1]), 2, 8)
        return mutated, f"mutate prompts_per_iteration -> {training['prompts_per_iteration']}"
    if choice == "num_candidates":
        current = int(agentic.get("num_candidates", 2))
        agentic["num_candidates"] = bounded_int(current + rng.choice([-1, 1]), 1, 6)
        return mutated, f"mutate num_candidates -> {agentic['num_candidates']}"
    if choice == "repair_steps":
        current = int(agentic.get("repair_steps", 1))
        agentic["repair_steps"] = bounded_int(current + rng.choice([-1, 1]), 0, 4)
        return mutated, f"mutate repair_steps -> {agentic['repair_steps']}"
    if choice == "max_episode_steps":
        current = int(agentic.get("max_episode_steps", 2))
        agentic["max_episode_steps"] = bounded_int(current + rng.choice([-1, 1]), 1, 5)
        return mutated, f"mutate max_episode_steps -> {agentic['max_episode_steps']}"
    if choice == "temperature":
        current = float(training.get("temperature", 0.35))
        training["temperature"] = float(f"{bounded_float(current + rng.choice([-0.05, 0.05]), 0.15, 0.80):.2f}")
        return mutated, f"mutate temperature -> {training['temperature']}"
    if choice == "top_p":
        current = float(training.get("top_p", 0.75))
        training["top_p"] = float(f"{bounded_float(current + rng.choice([-0.05, 0.05]), 0.55, 0.95):.2f}")
        return mutated, f"mutate top_p -> {training['top_p']}"
    if choice == "lora_r":
        current = int(model.get("lora_r", 32))
        options = [16, 24, 32, 48, 64]
        options.sort(key=lambda value: (value != current, abs(value - current)))
        pick = rng.choice([value for value in options if value != current] or [current])
        model["lora_r"] = pick
        model["lora_alpha"] = pick * 2
        return mutated, f"mutate lora_r -> {model['lora_r']}"
    if choice == "hidden_test":
        current = float(reward.get("hidden_test", 0.60))
        reward["hidden_test"] = float(f"{bounded_float(current + rng.choice([-0.05, 0.05]), 0.40, 0.80):.2f}")
        reward["cleanliness"] = float(f"{bounded_float(1.0 - reward['hidden_test'] - float(reward.get('assemble', 0.10)) - float(reward.get('link', 0.05)) - float(reward.get('run', 0.05)) - float(reward.get('public_test', 0.10)), 0.02, 0.25):.2f}")
        return mutated, f"mutate hidden_test -> {reward['hidden_test']}"
    current = float(agentic.get("repair_gain_weight", 0.10))
    agentic["repair_gain_weight"] = float(f"{bounded_float(current + rng.choice([-0.05, 0.05]), 0.0, 0.30):.2f}")
    return mutated, f"mutate repair_gain_weight -> {agentic['repair_gain_weight']}"


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root_dir = args.log_dir / run_tag
    root_dir.mkdir(parents=True, exist_ok=True)
    results_tsv = root_dir / "results.tsv"
    init_results_tsv(results_tsv)

    base_cfg = apply_cli_overrides(load_yaml(args.config), args)
    best_cfg = ensure_target_config(args.target_config, base_cfg)
    best_cfg = apply_cli_overrides(best_cfg, args)
    dump_yaml(args.target_config, best_cfg)
    best_metrics: dict[str, Any] = {}
    best_exp_dir: Path | None = None
    records: list[dict[str, Any]] = []
    commit = short_head()

    baseline_dir = root_dir / "exp_000_baseline"
    print("[autoresearch] baseline run on real GRPO", flush=True)
    baseline_status, baseline_metrics, baseline_log = run_grpo_experiment(
        exp_dir=baseline_dir,
        cfg=best_cfg,
        tasks_path=args.tasks,
        seed=args.seed,
        timeout_minutes=args.time_budget,
    )
    baseline_keep = baseline_status == "ok" and bool(baseline_metrics)
    best_metrics = baseline_metrics if baseline_keep else {}
    best_exp_dir = baseline_dir if baseline_keep else None
    append_results_tsv(
        results_tsv,
        commit,
        baseline_metrics,
        "keep" if baseline_keep else "crash",
        "baseline",
    )
    records.append(
        {
            "experiment": 0,
            "status": "keep" if baseline_keep else "crash",
            "description": "baseline",
            "metrics": baseline_metrics,
            "run_log": str(baseline_log),
            "exp_dir": str(baseline_dir),
        }
    )

    for idx in range(1, args.experiments):
        candidate_cfg, description = mutate_config(best_cfg if best_metrics else base_cfg, rng)
        exp_dir = root_dir / f"exp_{idx:03d}"
        print(f"[autoresearch] exp={idx} {description}", flush=True)
        status, metrics, run_log = run_grpo_experiment(
            exp_dir=exp_dir,
            cfg=candidate_cfg,
            tasks_path=args.tasks,
            seed=args.seed + idx,
            timeout_minutes=args.time_budget,
        )

        keep = False
        final_status = "crash" if status != "ok" or not metrics else "discard"
        if status == "ok" and metrics:
            if not best_metrics or metric_tuple(metrics) > metric_tuple(best_metrics):
                keep = True
                final_status = "keep"
                best_cfg = candidate_cfg
                best_metrics = metrics
                best_exp_dir = exp_dir
                dump_yaml(args.target_config, best_cfg)
            else:
                dump_yaml(args.target_config, best_cfg)
        else:
            dump_yaml(args.target_config, best_cfg)

        append_results_tsv(results_tsv, commit, metrics, final_status, description)
        records.append(
            {
                "experiment": idx,
                "status": final_status,
                "description": description,
                "metrics": metrics,
                "run_log": str(run_log),
                "exp_dir": str(exp_dir),
            }
        )
        write_summary(root_dir / "runs.json", {"run_tag": run_tag, "records": records, "best_metrics": best_metrics})

    if best_exp_dir is not None:
        dump_yaml(root_dir / "best_config.yaml", best_cfg)
        dump_yaml(args.target_config, best_cfg)
        best_artifacts = best_exp_dir / "artifacts" / "agentic_grpo"
        if best_artifacts.exists():
            summary_target = root_dir / "best_artifacts"
            if summary_target.exists():
                shutil.rmtree(summary_target)
            shutil.copytree(best_artifacts, summary_target)

    write_summary(
        root_dir / "runs.json",
        {
            "run_tag": run_tag,
            "records": records,
            "best_metrics": best_metrics,
            "best_experiment_dir": str(best_exp_dir) if best_exp_dir is not None else None,
            "results_tsv": str(results_tsv),
        },
    )

    print(
        json.dumps(
            {
                "run_tag": run_tag,
                "log_dir": str(root_dir),
                "results_tsv": str(results_tsv),
                "best_metrics": best_metrics,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
