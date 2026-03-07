#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agentic import build_draft_prompt
from src.data import Task, load_tasks_jsonl
from src.modeling import generate_completion, load_model_and_tokenizer
from src.utils import ensure_dir, load_yaml
from src.verifier import ObjectiveVerifier, VerifyResult


SYS_PROMPT = (
    "You are an expert NASM x86-64 Linux programmer. "
    "Output only assembly code, no markdown fences, no explanations."
)


def build_repair_prompt(task: Task, previous_asm: str, result: VerifyResult | None) -> str:
    stage = "unknown" if result is None else (result.stage_failed or "correctness")
    stderr = "" if result is None else result.stderr.strip()
    if len(stderr) > 500:
        stderr = stderr[:500]
    return (
        f"{SYS_PROMPT}\n\n"
        f"Task: {task.instruction}\n\n"
        "The previous assembly candidate failed objective verification.\n"
        f"Failure stage: {stage}\n"
        f"Verifier stderr: {stderr}\n\n"
        "Rewrite and fix the assembly.\n"
        "Return only assembly code.\n\n"
        "Previous candidate:\n"
        f"{previous_asm}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--repair-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_cfg = cfg["model"]
    sampling_cfg = cfg["sampling"]
    runtime_cfg = cfg["runtime"]

    base_model = args.model or model_cfg["base_model"]
    adapter = args.adapter or model_cfg.get("adapter_path") or None
    num_candidates = int(args.num_candidates or sampling_cfg["num_candidates"])
    repair_steps = int(args.repair_steps if args.repair_steps is not None else sampling_cfg["repair_steps"])

    model, tokenizer = load_model_and_tokenizer(
        base_model=base_model,
        adapter_path=adapter,
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        torch_dtype=str(model_cfg.get("torch_dtype", "float16")),
    )

    tasks = load_tasks_jsonl(args.tasks)
    verifier = ObjectiveVerifier(
        artifacts_dir=Path(runtime_cfg["artifacts_dir"]) / "ranked_sampling",
        timeout_seconds=int(runtime_cfg["verifier_timeout_sec"]),
        reward_weights=cfg.get("reward"),
    )

    out_path = Path(args.out)
    ensure_dir(out_path.parent)

    written = 0
    solved = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            candidates: list[dict[str, Any]] = []
            initial_prompt = build_draft_prompt(task)
            for cand_idx in range(num_candidates):
                asm = generate_completion(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=initial_prompt,
                    max_new_tokens=int(sampling_cfg["max_new_tokens"]),
                    temperature=float(sampling_cfg["temperature"]),
                    top_p=float(sampling_cfg["top_p"]),
                    top_k=sampling_cfg.get("top_k"),
                    min_p=sampling_cfg.get("min_p"),
                    repetition_penalty=float(sampling_cfg.get("repetition_penalty", 1.0)),
                )
                result = verifier.evaluate(task, asm, sample_id=f"{task.task_id}-cand{cand_idx}")
                best_asm = asm
                best_result = result

                for repair_idx in range(repair_steps):
                    if best_result.correct:
                        break
                    repaired = generate_completion(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=build_repair_prompt(task, best_asm, best_result),
                        max_new_tokens=int(sampling_cfg["max_new_tokens"]),
                        temperature=max(0.2, float(sampling_cfg["temperature"]) * 0.8),
                        top_p=float(sampling_cfg["top_p"]),
                        top_k=sampling_cfg.get("top_k"),
                        min_p=sampling_cfg.get("min_p"),
                        repetition_penalty=float(sampling_cfg.get("repetition_penalty", 1.0)),
                    )
                    repaired_result = verifier.evaluate(
                        task,
                        repaired,
                        sample_id=f"{task.task_id}-cand{cand_idx}-repair{repair_idx + 1}",
                    )
                    if repaired_result.reward >= best_result.reward:
                        best_asm = repaired
                        best_result = repaired_result

                candidates.append({
                    "task_id": task.task_id,
                    "instruction": task.instruction,
                    "prompt_text": initial_prompt,
                    "candidate_id": f"{task.task_id}-c{cand_idx}",
                    "asm": best_asm,
                    "reward": best_result.reward,
                    "correct": best_result.correct,
                    "stage_failed": best_result.stage_failed,
                })

            candidates.sort(key=lambda row: float(row["reward"]), reverse=True)
            for rank, row in enumerate(candidates):
                row["candidate_rank"] = rank
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            solved += int(bool(candidates) and bool(candidates[0]["correct"]))
            print(
                f"[run_ranked_sampling] task={task.task_id} "
                f"best_reward={candidates[0]['reward']:.4f} correct={candidates[0]['correct']}",
                flush=True,
            )

    print({
        "tasks": len(tasks),
        "rows_written": written,
        "top1_correct": solved,
        "top1_correct_rate": solved / max(1, len(tasks)),
        "out": str(out_path),
    })


if __name__ == "__main__":
    main()
