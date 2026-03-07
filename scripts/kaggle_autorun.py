#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import select
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}][kaggle_autorun] {msg}", flush=True)


def run_quick(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = (proc.stdout or proc.stderr or "").strip()
    return proc.returncode, out


def load_secret_candidates(names: list[str]) -> tuple[str | None, str | None]:
    for name in names:
        val = os.getenv(name)
        if val:
            return val, f"env:{name}"
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
    except Exception:
        return None, None
    client = UserSecretsClient()
    for name in names:
        try:
            val = client.get_secret(name)
            if val:
                return val, f"kaggle:{name}"
        except Exception:
            continue
    return None, None


def load_env() -> None:
    hf, hf_src = load_secret_candidates(
        ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_TOKEN", "HF_KEY", "hf_token"]
    )
    wandb, wb_src = load_secret_candidates(
        ["WANDB_API_KEY", "WANDB _API_KEY", "WANDB_KEY", "WANDB_TOKEN", "wandb_api_key"]
    )
    mistral, mi_src = load_secret_candidates(
        ["MISTRAL_API_KEY", "MISTRAL_KEY", "MISTRAL_TOKEN", "mistral_api_key"]
    )

    if hf:
        os.environ["HF_TOKEN"] = hf.strip()
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf.strip()
    if wandb:
        os.environ["WANDB_API_KEY"] = wandb.strip()
    if mistral:
        os.environ["MISTRAL_API_KEY"] = mistral.strip()

    log(f"HF_TOKEN loaded: {bool(os.getenv('HF_TOKEN'))} ({hf_src or 'not found'})")
    log(f"WANDB_API_KEY loaded: {bool(os.getenv('WANDB_API_KEY'))} ({wb_src or 'not found'})")
    log(f"MISTRAL_API_KEY loaded: {bool(os.getenv('MISTRAL_API_KEY'))} ({mi_src or 'not found'})")
    if not os.getenv("WANDB_API_KEY"):
        log("WARNING: W&B secret not found. Run will continue with use_wandb=false.")
    if not os.getenv("HF_TOKEN"):
        log("WARNING: HF token not found. Checkpoint push_to_hub will be disabled.")


def ensure_system_deps() -> None:
    if shutil.which("nasm") and shutil.which("ld"):
        log("System deps already present: nasm + ld")
        return
    log("Installing system deps: nasm + binutils")
    subprocess.run(["apt-get", "update", "-y"], check=False)
    subprocess.run(["apt-get", "install", "-y", "nasm", "binutils"], check=False)


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def tune_config(cfg_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    cfg = read_yaml(cfg_path)
    cfg.setdefault("training", {})
    cfg.setdefault("paths", {})
    tr = cfg["training"]

    tr["dry_run"] = False
    tr["grpo_backend"] = args.backend
    if args.iterations is not None:
        tr["iterations"] = int(args.iterations)
    tr["use_wandb"] = bool(os.getenv("WANDB_API_KEY"))
    tr["push_to_hub"] = bool(os.getenv("HF_TOKEN"))
    tr["use_random_sampling"] = True
    if args.prompt_dataset:
        cfg["paths"]["prompt_dataset"] = str(args.prompt_dataset)

    if args.safe_profile:
        tr["batch_size"] = int(args.batch_size)
        tr["generations_per_prompt"] = int(args.generations_per_prompt)
        tr["prompts_per_iteration"] = int(args.prompts_per_iteration)
        tr["gradient_accumulation_steps"] = int(args.gradient_accumulation_steps)
        tr["max_new_tokens"] = int(args.max_new_tokens)
        tr["use_mcts_after_iteration"] = int(args.use_mcts_after_iteration)

    write_yaml(cfg_path, cfg)
    log(
        "Config tuned: "
        f"backend={tr['grpo_backend']} iterations={tr['iterations']} "
        f"dataset={cfg.get('paths', {}).get('prompt_dataset')} "
        f"batch={tr.get('batch_size')} gens={tr.get('generations_per_prompt')} "
        f"prompts={tr.get('prompts_per_iteration')} max_new={tr.get('max_new_tokens')} "
        f"mcts_after={tr.get('use_mcts_after_iteration')} wandb={tr.get('use_wandb')}"
    )
    return cfg


def checkpoints_dir(root: Path, cfg: dict[str, Any]) -> Path:
    paths = cfg.get("paths", {})
    ck = paths.get("checkpoints_dir", "checkpoints")
    return root / str(ck)


def _has_lora_weights(iter_dir: Path) -> bool:
    return (iter_dir / "adapter_model.safetensors").exists() or (iter_dir / "adapter_model.bin").exists()


def latest_iter(ckpt_dir: Path) -> int:
    if not ckpt_dir.exists():
        return -1
    latest_valid = -1
    latest_any = -1
    for p in ckpt_dir.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"iter_(\d+)$", p.name)
        if m:
            iter_id = int(m.group(1))
            latest_any = max(latest_any, iter_id)
            # Prefer valid checkpoints, but keep latest folder as fallback.
            if _has_lora_weights(p):
                latest_valid = max(latest_valid, iter_id)
    return latest_valid if latest_valid >= 0 else latest_any


def gpu_line() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    rc, out = run_quick(cmd)
    if rc != 0 or not out:
        return "gpu: n/a"
    return " | ".join([x.strip() for x in out.splitlines()])


def run_train_once(root: Path, cfg_path: Path, start_iter: int, heartbeat_sec: int) -> int:
    cmd = [
        sys.executable,
        str(root / "train.py"),
        "--config",
        str(cfg_path),
        "--start-iter",
        str(start_iter),
        "--ensure-system-deps",
    ]
    log(f"Starting train process: {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None

    last_heartbeat = time.time()
    while True:
        if proc.poll() is not None:
            for line in proc.stdout:
                print(line.rstrip(), flush=True)
            break

        ready, _, _ = select.select([proc.stdout], [], [], 1.0)
        if ready:
            line = proc.stdout.readline()
            if line:
                print(line.rstrip(), flush=True)

        now = time.time()
        if now - last_heartbeat >= heartbeat_sec:
            last_heartbeat = now
            log(f"heartbeat | {gpu_line()}")

    rc = proc.returncode if proc.returncode is not None else 1
    log(f"Train process exited with code {rc}")
    return rc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kaggle 2-command autorun with verbose logs + auto-resume.")
    p.add_argument("--root", default="/kaggle/working/codeforgefinal")
    p.add_argument("--config", default="configs/grpo_config.yaml")
    p.add_argument("--hours", type=float, default=18.0)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--prompt-dataset", default="")
    p.add_argument("--backend", choices=["trl", "manual"], default="trl")
    p.add_argument("--retry-delay-sec", type=int, default=20)
    p.add_argument("--heartbeat-sec", type=int, default=30)
    p.add_argument("--safe-profile", action="store_true")
    p.add_argument("--no-safe-profile", dest="safe_profile", action="store_false")
    p.set_defaults(safe_profile=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--generations-per-prompt", type=int, default=4)
    p.add_argument("--prompts-per-iteration", type=int, default=6)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--use-mcts-after-iteration", type=int, default=999)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    cfg_path = root / args.config
    if not cfg_path.exists():
        log(f"Config not found: {cfg_path}")
        return 2

    os.chdir(root)
    load_env()
    ensure_system_deps()

    cfg = tune_config(cfg_path, args)
    ckpt_dir = checkpoints_dir(root, cfg)
    deadline = time.time() + args.hours * 3600
    log(f"Deadline in {args.hours:.2f}h | checkpoints_dir={ckpt_dir}")

    while time.time() < deadline:
        cfg = read_yaml(cfg_path)
        total_iterations = int(cfg.get("training", {}).get("iterations", args.iterations))
        last = latest_iter(ckpt_dir)
        start_iter = last + 1
        log(f"resume probe | last_iter={last} start_iter={start_iter} target_iterations={total_iterations}")

        if start_iter >= total_iterations:
            log("Training target reached. Exiting.")
            return 0

        rc = run_train_once(root, cfg_path, start_iter, args.heartbeat_sec)
        new_last = latest_iter(ckpt_dir)
        log(f"post-run checkpoint status | previous_last={last} new_last={new_last}")

        if rc == 0:
            time.sleep(3)
            continue

        log(f"Retrying in {args.retry_delay_sec}s...")
        time.sleep(args.retry_delay_sec)

    log("Deadline reached. Exiting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

