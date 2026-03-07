from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.prompt_engine import PromptItem
from src.reward import RewardPipeline, RewardResult


@dataclass
class Candidate:
    task_id: str
    asm: str
    candidate_id: str
    candidate_rank: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_tasks(path: Path) -> dict[str, PromptItem]:
    out: dict[str, PromptItem] = {}
    for row in _read_jsonl(path):
        task_id = str(row["task_id"])
        out[task_id] = PromptItem(
            id=task_id,
            tier=int(row["tier"]),
            instruction=str(row["instruction"]),
            expected_stdout=row.get("expected_stdout"),
            expected_exit_code=row.get("expected_exit_code"),
        )
    return out


def load_predictions(path: Path) -> dict[str, list[Candidate]]:
    grouped: dict[str, list[Candidate]] = {}
    for idx, row in enumerate(_read_jsonl(path)):
        cand = Candidate(
            task_id=str(row["task_id"]),
            asm=str(row["asm"]),
            candidate_id=str(row.get("candidate_id", f"c{idx}")),
            candidate_rank=int(row.get("candidate_rank", idx)),
        )
        grouped.setdefault(cand.task_id, []).append(cand)
    for task_id in grouped:
        grouped[task_id].sort(key=lambda c: c.candidate_rank)
    return grouped


def parse_ks(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k > 0:
            out.append(k)
    out = sorted(set(out))
    if not out:
        return [1]
    return out


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / total) + ((z * z) / (4 * total * total)))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (low, high)


def bootstrap_ci(values: list[float], n_bootstrap: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low_i = int((alpha / 2) * (n_bootstrap - 1))
    high_i = int((1 - alpha / 2) * (n_bootstrap - 1))
    return (means[low_i], means[high_i])


def leaderboard_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Assembly-SWE Leaderboard Summary",
        "",
        f"- tasks: **{summary['tasks_total']}**",
        f"- predictions: **{summary['predictions_total']}**",
        f"- pass@1: **{summary['pass_at'].get('1', 0.0):.4f}**",
        f"- assembly_rate@1: **{summary['assembly_rate_at_1']:.4f}**",
        f"- link_rate@1: **{summary['link_rate_at_1']:.4f}**",
        f"- run_rate@1: **{summary['run_rate_at_1']:.4f}**",
        f"- avg_reward@1: **{summary['avg_reward_at_1']:.4f}**",
        f"- correct@1 CI95 (Wilson): **[{summary['correct_rate_at_1_ci95']['low']:.4f}, {summary['correct_rate_at_1_ci95']['high']:.4f}]**",
        "",
        "## pass@k",
        "",
        "| k | pass@k |",
        "|---:|---:|",
    ]
    for k, v in summary["pass_at"].items():
        ci = summary.get("pass_at_ci95", {}).get(str(k), {})
        if ci:
            lines.append(f"| {k} | {v:.4f} ({ci['low']:.4f}..{ci['high']:.4f}) |")
        else:
            lines.append(f"| {k} | {v:.4f} |")

    lines += [
        "",
        "## Tier Breakdown",
        "",
        "| Tier | tasks | pass@1 | assembly@1 | link@1 | run@1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for tier, row in summary["tier_breakdown"].items():
        lines.append(
            f"| {tier} | {row['tasks']} | {row['pass_at_1']:.4f} | "
            f"{row['assembly_rate_at_1']:.4f} | {row['link_rate_at_1']:.4f} | {row['run_rate_at_1']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="Task JSONL path")
    parser.add_argument("--predictions", required=True, help="Predictions JSONL path")
    parser.add_argument("--ks", default="1,3,5", help="Comma-separated k values, e.g. 1,3,5")
    parser.add_argument("--outdir", default="assembly_swe/results/latest", help="Output directory")
    parser.add_argument("--timeout-sec", type=int, default=5, help="Per-sample execution timeout")
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    preds_path = Path(args.predictions)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ks = parse_ks(args.ks)
    tasks = load_tasks(tasks_path)
    preds = load_predictions(preds_path)
    reward = RewardPipeline(outdir / "artifacts", timeout_seconds=args.timeout_sec)

    tasks_total = len(tasks)
    predictions_total = sum(len(v) for v in preds.values())

    top1_rows: list[dict[str, Any]] = []
    correct_by_k = {k: 0 for k in ks}
    tier_rows: dict[int, list[dict[str, Any]]] = {}

    for task_id, prompt in tasks.items():
        cands = preds.get(task_id, [])
        cand_results: list[tuple[Candidate, RewardResult]] = []
        for i, cand in enumerate(cands):
            sample_id = f"{task_id}-cand{i}"
            rr = reward.evaluate(prompt, cand.asm, sample_id=sample_id)
            cand_results.append((cand, rr))

        top1 = cand_results[0][1] if cand_results else None
        if top1 is None:
            top1_payload = {
                "task_id": task_id,
                "tier": prompt.tier,
                "has_prediction": False,
                "assembled": False,
                "linked": False,
                "ran": False,
                "correct": False,
                "reward": 0.0,
                "stage_failed": "missing_prediction",
            }
        else:
            top1_payload = {
                "task_id": task_id,
                "tier": prompt.tier,
                "has_prediction": True,
                "assembled": top1.assembled,
                "linked": top1.linked,
                "ran": top1.ran,
                "correct": top1.correct,
                "reward": top1.reward,
                "stage_failed": top1.stage_failed,
            }
        top1_rows.append(top1_payload)
        tier_rows.setdefault(prompt.tier, []).append(top1_payload)

        for k in ks:
            subset = cand_results[:k]
            if any(rr.correct for _, rr in subset):
                correct_by_k[k] += 1

    assembled = sum(1 for r in top1_rows if r["assembled"])
    linked = sum(1 for r in top1_rows if r["linked"])
    ran = sum(1 for r in top1_rows if r["ran"])
    correct = sum(1 for r in top1_rows if r["correct"])
    avg_reward = sum(float(r["reward"]) for r in top1_rows) / max(1, tasks_total)

    pass_at = {str(k): correct_by_k[k] / max(1, tasks_total) for k in ks}
    tier_breakdown: dict[str, Any] = {}
    for tier, rows in sorted(tier_rows.items()):
        t_tasks = len(rows)
        t_pass1 = sum(1 for r in rows if r["correct"]) / max(1, t_tasks)
        t_assembly = sum(1 for r in rows if r["assembled"]) / max(1, t_tasks)
        t_link = sum(1 for r in rows if r["linked"]) / max(1, t_tasks)
        t_run = sum(1 for r in rows if r["ran"]) / max(1, t_tasks)
        tier_breakdown[str(tier)] = {
            "tasks": t_tasks,
            "pass_at_1": t_pass1,
            "assembly_rate_at_1": t_assembly,
            "link_rate_at_1": t_link,
            "run_rate_at_1": t_run,
        }

    stage_failed_counts: dict[str, int] = {}
    for r in top1_rows:
        key = str(r.get("stage_failed", "unknown"))
        stage_failed_counts[key] = stage_failed_counts.get(key, 0) + 1

    correct_ci = wilson_interval(correct, tasks_total)
    pass_at_ci95: dict[str, Any] = {}
    for k in ks:
        low, high = wilson_interval(correct_by_k[k], tasks_total)
        pass_at_ci95[str(k)] = {"low": low, "high": high}
    reward_ci = bootstrap_ci([float(r["reward"]) for r in top1_rows], n_bootstrap=1000, alpha=0.05, seed=42)

    summary = {
        "tasks_total": tasks_total,
        "predictions_total": predictions_total,
        "pass_at": pass_at,
        "assembly_rate_at_1": assembled / max(1, tasks_total),
        "link_rate_at_1": linked / max(1, tasks_total),
        "run_rate_at_1": ran / max(1, tasks_total),
        "correct_rate_at_1": correct / max(1, tasks_total),
        "correct_rate_at_1_ci95": {"low": correct_ci[0], "high": correct_ci[1]},
        "avg_reward_at_1": avg_reward,
        "avg_reward_at_1_bootstrap_ci95": {"low": reward_ci[0], "high": reward_ci[1]},
        "pass_at_ci95": pass_at_ci95,
        "stage_failed_counts_at_1": stage_failed_counts,
        "tier_breakdown": tier_breakdown,
        "inputs": {
            "tasks": str(tasks_path),
            "predictions": str(preds_path),
            "ks": ks,
            "timeout_sec": args.timeout_sec,
        },
    }

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / "leaderboard.md").write_text(leaderboard_markdown(summary), encoding="utf-8")
    with (outdir / "rows_top1.jsonl").open("w", encoding="utf-8") as f:
        for row in top1_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
