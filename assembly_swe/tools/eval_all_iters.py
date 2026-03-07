from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def run(cmd: list[str], cwd: Path) -> None:
    print(f"[eval_all_iters] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_hf_token() -> str | None:
    tok = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
    except Exception:
        return None
    client = UserSecretsClient()
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_TOKEN", "HF_KEY"):
        try:
            val = client.get_secret(name)
        except Exception:
            val = None
        if val:
            os.environ["HF_TOKEN"] = val.strip()
            os.environ["HUGGINGFACE_HUB_TOKEN"] = val.strip()
            print(f"[eval_all_iters] loaded HF token from kaggle:{name}", flush=True)
            return val.strip()
    return None


def ensure_checkpoint_local(repo_root: Path, iter_idx: int, hub_repo_id: str) -> bool:
    ckpt_dir = repo_root / "checkpoints" / f"iter_{iter_idx}"
    required = ("adapter_config.json", "adapter_model.safetensors")
    if ckpt_dir.exists():
        if all((ckpt_dir / name).exists() for name in required):
            return True
        print(
            f"[eval_all_iters] checkpoint iter_{iter_idx} exists but is incomplete; refetching from HF",
            flush=True,
        )

    token = ensure_hf_token()
    if not token:
        print(f"[eval_all_iters] skip iter_{iter_idx}: local checkpoint missing and HF token not found", flush=True)
        return False

    print(f"[eval_all_iters] fetching iter_{iter_idx} from HF repo {hub_repo_id}", flush=True)
    try:
        snapshot_download(
            repo_id=hub_repo_id,
            repo_type="model",
            token=token,
            local_dir=str(repo_root),
            allow_patterns=[f"checkpoints/iter_{iter_idx}/*"],
            resume_download=True,
        )
    except Exception as exc:
        print(f"[eval_all_iters] skip iter_{iter_idx}: HF fetch failed ({exc})", flush=True)
        return False
    return ckpt_dir.exists() and all((ckpt_dir / name).exists() for name in required)


def maybe_run_iter(
    repo_root: Path,
    tasks_path: Path,
    iter_idx: int,
    use_4bit: bool,
    ks: str,
    out_root: Path,
    hub_repo_id: str,
    num_candidates: int,
    verifier: str,
    verifier_timeout_sec: int,
    repair_steps: int,
    base_model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    min_p: float | None,
    repetition_penalty: float,
) -> dict | None:
    ckpt_dir = repo_root / "checkpoints" / f"iter_{iter_idx}"
    if not ensure_checkpoint_local(repo_root, iter_idx, hub_repo_id=hub_repo_id):
        print(f"[eval_all_iters] skip iter_{iter_idx}: checkpoint not available", flush=True)
        return None

    pred_path = out_root / f"preds_iter_{iter_idx}.jsonl"
    eval_dir = out_root / f"eval_iter_{iter_idx}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    gen_cmd = [
        sys.executable,
        "assembly_swe/tools/generate_predictions.py",
        "--tasks",
        str(tasks_path),
        "--checkpoint-dir",
        str(ckpt_dir),
        "--out",
        str(pred_path),
        "--base-model",
        base_model,
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--num-candidates",
        str(num_candidates),
        "--verifier",
        verifier,
        "--verifier-timeout-sec",
        str(verifier_timeout_sec),
        "--repair-steps",
        str(repair_steps),
        "--verifier-artifacts-dir",
        str((eval_dir / "verifier_artifacts").resolve()),
    ]
    if top_k is not None:
        gen_cmd.extend(["--top-k", str(top_k)])
    if min_p is not None:
        gen_cmd.extend(["--min-p", str(min_p)])
    if repetition_penalty != 1.0:
        gen_cmd.extend(["--repetition-penalty", str(repetition_penalty)])
    if use_4bit:
        gen_cmd.append("--load-in-4bit")
    run(gen_cmd, cwd=repo_root)

    eval_cmd = [
        sys.executable,
        "assembly_swe/tools/evaluate.py",
        "--tasks",
        str(tasks_path),
        "--predictions",
        str(pred_path),
        "--ks",
        ks,
        "--outdir",
        str(eval_dir),
    ]
    run(eval_cmd, cwd=repo_root)

    summary_path = eval_dir / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "iter": iter_idx,
        "summary_path": str(summary_path),
        "correct_rate_at_1": summary.get("correct_rate_at_1"),
        "assembly_rate_at_1": summary.get("assembly_rate_at_1"),
        "avg_reward_at_1": summary.get("avg_reward_at_1"),
        "pass_at": summary.get("pass_at", {}),
    }


def plot_curves(rows: list[dict], png_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[eval_all_iters] matplotlib unavailable, skip plot: {exc}", flush=True)
        return

    if not rows:
        return
    rows = sorted(rows, key=lambda x: x["iter"])
    xs = [r["iter"] for r in rows]
    ys_correct = [float(r.get("correct_rate_at_1", 0) or 0) for r in rows]
    ys_assembly = [float(r.get("assembly_rate_at_1", 0) or 0) for r in rows]
    ys_reward = [float(r.get("avg_reward_at_1", 0) or 0) for r in rows]

    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys_correct, marker="o", label="correct_rate@1")
    plt.plot(xs, ys_assembly, marker="o", label="assembly_rate@1")
    plt.plot(xs, ys_reward, marker="o", label="avg_reward@1")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title("Assembly-SWE over checkpoints")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Run Assembly-SWE eval for many checkpoints and build aggregate JSON + chart")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--tasks", default="assembly_swe/datasets/sample_dev.jsonl")
    p.add_argument("--iter-start", type=int, default=1)
    p.add_argument("--iter-end", type=int, default=30)
    p.add_argument("--ks", default="1,3,5")
    p.add_argument("--outdir", default="assembly_swe/results/all_iters")
    p.add_argument("--load-in-4bit", action="store_true", default=False)
    p.add_argument("--hub-repo-id", default="mistral-hackaton-2026/codeforge")
    p.add_argument("--base-model", default="mistralai/Ministral-8B-Instruct-2410")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--min-p", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--num-candidates", type=int, default=1, help="Candidates per task generated before evaluation")
    p.add_argument("--verifier", choices=["none", "reward"], default="none", help="Reranker used during generation")
    p.add_argument("--verifier-timeout-sec", type=int, default=5)
    p.add_argument("--repair-steps", type=int, default=0)
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    tasks_path = (repo_root / args.tasks).resolve()
    out_root = (repo_root / args.outdir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    ensure_hf_token()
    print(
        "[eval_all_iters] start | "
        f"tasks={tasks_path} range={args.iter_start}..{args.iter_end} ks={args.ks} "
        f"candidates={args.num_candidates} verifier={args.verifier} repair_steps={args.repair_steps}",
        flush=True,
    )

    rows: list[dict] = []
    skipped: list[dict] = []
    for i in range(args.iter_start, args.iter_end + 1):
        try:
            item = maybe_run_iter(
                repo_root=repo_root,
                tasks_path=tasks_path,
                iter_idx=i,
                use_4bit=args.load_in_4bit,
                ks=args.ks,
                out_root=out_root,
                hub_repo_id=args.hub_repo_id,
                num_candidates=args.num_candidates,
                verifier=args.verifier,
                verifier_timeout_sec=args.verifier_timeout_sec,
                repair_steps=args.repair_steps,
                base_model=args.base_model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                repetition_penalty=args.repetition_penalty,
            )
        except Exception as exc:
            print(f"[eval_all_iters] iter_{i} failed: {exc}", flush=True)
            item = None
        if item is None:
            skipped.append({"iter": i, "reason": "missing_or_failed"})
            continue
        rows.append(item)
        print(
            "[eval_all_iters] result "
            f"iter={item['iter']} "
            f"correct={float(item.get('correct_rate_at_1', 0) or 0):.4f} "
            f"assembly={float(item.get('assembly_rate_at_1', 0) or 0):.4f} "
            f"reward={float(item.get('avg_reward_at_1', 0) or 0):.4f}",
            flush=True,
        )

    rows = sorted(rows, key=lambda x: x["iter"])
    best = max(rows, key=lambda x: float(x.get("correct_rate_at_1", 0) or 0), default=None)
    agg = {
        "tasks": str(tasks_path),
        "iter_start": args.iter_start,
        "iter_end": args.iter_end,
        "count_evaluated": len(rows),
        "count_skipped": len(skipped),
        "best_by_correct_rate": best,
        "rows": rows,
        "skipped": skipped,
    }
    agg_path = out_root / "aggregate.json"
    agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")

    png_path = out_root / "curves.png"
    plot_curves(rows, png_path)

    print(f"[eval_all_iters] aggregate json: {agg_path}")
    print(f"[eval_all_iters] chart png: {png_path}")


if __name__ == "__main__":
    main()
