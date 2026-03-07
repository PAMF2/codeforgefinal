from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.prompt_engine import PromptItem
from src.reward import RewardPipeline
from src.utils import SYS_PROMPT, sanitize_model_output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_model(
    base_model: str,
    adapter_path: Path,
    load_in_4bit: bool,
) -> tuple[Any, Any]:
    bnb = None
    if load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if bnb is not None:
        model_kwargs["quantization_config"] = bnb
    else:
        model_kwargs["torch_dtype"] = torch.float16

    base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    model.eval()

    # Prefer tokenizer from checkpoint (keeps special tokens aligned), fallback to base model.
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _generate_from_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    min_p: float | None,
    repetition_penalty: float,
) -> str:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            top_k=top_k if top_k is not None else 50,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Strip prompt echo if present.
    if text.startswith(prompt):
        text = text[len(prompt) :]
    return sanitize_model_output(text.strip())


def _build_base_prompt(instruction: str) -> str:
    return f"{SYS_PROMPT}\n\nTask: {instruction}"


def _build_repair_prompt(instruction: str, previous_asm: str, stage_failed: str | None, stderr: str) -> str:
    feedback = (stderr or "").strip()
    if len(feedback) > 500:
        feedback = feedback[:500]
    return (
        f"{SYS_PROMPT}\n\n"
        f"Task: {instruction}\n\n"
        "Previous candidate failed objective verification.\n"
        f"Failure stage: {stage_failed or 'unknown'}.\n"
        f"Verifier stderr: {feedback}\n\n"
        "Rewrite and fix the assembly so it is valid NASM x86-64 Linux and satisfies the task.\n"
        "Return only assembly code.\n\n"
        "Previous candidate:\n"
        f"{previous_asm}"
    )


def _candidate_score(asm: str, reward_result: Any | None) -> float:
    if reward_result is None:
        # Fallback verifier: short/clean candidates rank slightly better than huge rambles.
        return -float(len(asm))
    return float(reward_result.reward)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Assembly-SWE predictions from a CodeForge checkpoint")
    p.add_argument("--tasks", required=True, help="Task JSONL")
    p.add_argument("--checkpoint-dir", required=True, help="Local checkpoint dir (e.g., checkpoints/iter_30)")
    p.add_argument("--out", required=True, help="Output predictions JSONL")
    p.add_argument("--base-model", default="mistralai/Ministral-8B-Instruct-2410")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--min-p", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--load-in-4bit", action="store_true", default=False)
    p.add_argument("--num-candidates", type=int, default=1, help="Number of candidates per task")
    p.add_argument(
        "--verifier",
        choices=["none", "reward"],
        default="none",
        help="Candidate reranker. 'reward' runs compile/link/run verification and ranks by reward.",
    )
    p.add_argument("--verifier-timeout-sec", type=int, default=5, help="Per-candidate timeout for verifier=reward")
    p.add_argument("--repair-steps", type=int, default=0, help="Extra repair attempts per candidate using verifier feedback")
    p.add_argument(
        "--verifier-artifacts-dir",
        default="assembly_swe/results/verifier_artifacts",
        help="Artifact dir used by verifier=reward",
    )
    args = p.parse_args()

    tasks_path = Path(args.tasks)
    ckpt = Path(args.checkpoint_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.num_candidates < 1:
        raise ValueError("--num-candidates must be >= 1")
    if args.repair_steps < 0:
        raise ValueError("--repair-steps must be >= 0")

    tasks = read_jsonl(tasks_path)
    model, tok = load_model(args.base_model, ckpt, load_in_4bit=args.load_in_4bit)
    print(
        "[generate_predictions] start | "
        f"tasks={len(tasks)} checkpoint={ckpt} candidates={args.num_candidates} "
        f"verifier={args.verifier} repair_steps={args.repair_steps}",
        flush=True,
    )
    reward = None
    if args.verifier == "reward":
        reward = RewardPipeline(
            artifacts_dir=Path(args.verifier_artifacts_dir),
            timeout_seconds=int(args.verifier_timeout_sec),
        )

    with out_path.open("w", encoding="utf-8") as f:
        for i, task in enumerate(tasks):
            instruction = str(task["instruction"])
            task_id = str(task["task_id"])
            prompt_item = PromptItem(
                id=task_id,
                tier=int(task.get("tier", 1)),
                instruction=instruction,
                expected_stdout=task.get("expected_stdout"),
                expected_exit_code=task.get("expected_exit_code"),
            )

            candidates: list[dict[str, Any]] = []
            for j in range(args.num_candidates):
                prompt = _build_base_prompt(instruction)
                asm = _generate_from_prompt(
                    model=model,
                    tokenizer=tok,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
                    repetition_penalty=args.repetition_penalty,
                )
                rr = reward.evaluate(prompt_item, asm, sample_id=f"{task_id}-cand{j}") if reward is not None else None
                best_asm = asm
                best_rr = rr
                best_score = _candidate_score(best_asm, best_rr)

                for repair_idx in range(args.repair_steps):
                    if reward is None:
                        break
                    if best_rr is not None and best_rr.correct:
                        break
                    repair_prompt = _build_repair_prompt(
                        instruction=instruction,
                        previous_asm=best_asm,
                        stage_failed=best_rr.stage_failed if best_rr is not None else None,
                        stderr=best_rr.stderr if best_rr is not None else "",
                    )
                    repaired = _generate_from_prompt(
                        model=model,
                        tokenizer=tok,
                        prompt=repair_prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=max(0.2, float(args.temperature) * 0.8),
                        top_p=args.top_p,
                        top_k=args.top_k,
                        min_p=args.min_p,
                        repetition_penalty=max(1.0, float(args.repetition_penalty)),
                    )
                    repaired_rr = reward.evaluate(
                        prompt_item,
                        repaired,
                        sample_id=f"{task_id}-cand{j}-repair{repair_idx + 1}",
                    )
                    repaired_score = _candidate_score(repaired, repaired_rr)
                    if repaired_score >= best_score:
                        best_asm = repaired
                        best_rr = repaired_rr
                        best_score = repaired_score
                        print(
                            f"[generate_predictions] improved task={task_id} cand={j} "
                            f"repair={repair_idx + 1} score={best_score:.4f}",
                            flush=True,
                        )

                candidates.append(
                    {
                        "task_id": task_id,
                        "candidate_id": f"{task_id}-c{j}",
                        "asm": best_asm,
                        "_score": best_score,
                    }
                )

            candidates.sort(key=lambda x: float(x["_score"]), reverse=True)
            for rank, row in enumerate(candidates):
                payload = {
                    "task_id": row["task_id"],
                    "candidate_id": row["candidate_id"],
                    "candidate_rank": rank,
                    "asm": row["asm"],
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            if (i + 1) % 5 == 0 or (i + 1) == len(tasks):
                print(
                    f"[generate_predictions] progress {i + 1}/{len(tasks)} tasks done",
                    flush=True,
                )

    print(
        f"wrote {len(tasks)} tasks, {len(tasks) * int(args.num_candidates)} candidates "
        f"(verifier={args.verifier}) -> {out_path}"
    )


if __name__ == "__main__":
    main()
