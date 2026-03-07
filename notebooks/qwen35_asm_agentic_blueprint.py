from __future__ import annotations

# %% [markdown]
# Qwen3.5-2B Assembly Agentic RL Notebook Blueprint
#
# Copy this file into a Kaggle notebook cell-by-cell or open it in VS Code with
# notebook cell support. The goal is not to be production-complete in one pass.
# The goal is to give you a strong architecture-first starting point.

# %%
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import time

# %%
@dataclass
class ExperimentConfig:
    repo_root: str = "/kaggle/working/codeforgefinal"
    model_id: str = "Qwen/Qwen3.5-2B"
    train_tasks: str = "assembly_swe/datasets/dev_v1_30.jsonl"
    eval_tasks: str = "assembly_swe/datasets/dev_v1_30.jsonl"
    artifacts_dir: str = "artifacts_notebook"
    load_in_4bit: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    max_prompt_len: int = 1024
    max_new_tokens: int = 192
    num_candidates: int = 4
    repair_steps: int = 1
    verifier_timeout_sec: int = 6
    max_episode_steps: int = 3
    seed: int = 42
    public_test_weight: float = 0.10
    hidden_test_weight: float = 0.60
    assemble_weight: float = 0.10
    link_weight: float = 0.05
    run_weight: float = 0.05
    repair_gain_weight: float = 0.10
    cleanliness_weight: float = 0.10
    prose_penalty: float = -0.05
    regress_penalty: float = -0.10
    bad_abi_penalty: float = -0.05
    grpo_learning_rate: float = 5e-6
    grpo_num_generations: int = 4
    grpo_per_device_batch_size: int = 1
    grpo_gradient_accumulation_steps: int = 8
    grpo_beta: float = 0.04
    sample_temperature: float = 0.60
    sample_top_p: float = 0.95
    sample_top_k: int = 20
    sample_min_p: float | None = None
    repetition_penalty: float = 1.03


CFG = ExperimentConfig()
random.seed(CFG.seed)

# %% [markdown]
# Install cell
#
# Recommended manual cell for Kaggle:
#
# !pip install -U "transformers>=5.2.0,<5.3.0" "trl>=0.27.2" "peft>=0.18.1" \
#     "accelerate>=1.11.0" "datasets>=4.6.1" "huggingface_hub>=0.36.2" \
#     "bitsandbytes>=0.49.2" "sentencepiece>=0.2.0" "wandb>=0.25.0"
# !sudo apt-get update -y && sudo apt-get install -y nasm binutils

# %%
def repo_path(*parts: str) -> Path:
    return Path(CFG.repo_root, *parts)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# %%
@dataclass
class VerifyResult:
    reward: float
    assembled: bool
    linked: bool
    ran: bool
    correct: bool
    public_pass_rate: float
    hidden_pass_rate: float
    stdout: str
    stderr: str
    exit_code: int | None
    stage_failed: str | None
    cleanliness_bonus: float = 0.0
    penalties: list[str] = field(default_factory=list)


# %%
def run_cmd(cmd: list[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", f"timeout after {timeout_seconds}s")
    except Exception as exc:
        return subprocess.CompletedProcess(cmd, 125, "", str(exc))


# %%
def hidden_tests_for_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Placeholder for a real hidden-test generator.

    For the first notebook pass, keep this simple:
    - stdout tasks: single hidden test copied from target
    - exit-code tasks: single hidden test copied from target

    Replace later with task-family-specific generators.
    """
    return [
        {
            "expected_stdout": task.get("expected_stdout"),
            "expected_exit_code": task.get("expected_exit_code"),
        }
    ]


# %%
def evaluate_candidate(task: dict[str, Any], asm_code: str, sample_id: str) -> VerifyResult:
    artifacts_root = repo_path(CFG.artifacts_dir)
    workdir = artifacts_root / sample_id
    workdir.mkdir(parents=True, exist_ok=True)

    asm_path = workdir / "prog.asm"
    obj_path = workdir / "prog.o"
    bin_path = workdir / "prog"
    asm_path.write_text(asm_code + "\n", encoding="utf-8")

    penalties: list[str] = []
    reward = 0.0

    low = asm_code.lower()
    if "```" in asm_code or "explanation" in low:
        reward += CFG.prose_penalty
        penalties.append("prose")
    if "int 0x80" in low or "int 80h" in low:
        reward += CFG.bad_abi_penalty
        penalties.append("bad_abi")

    assemble = run_cmd(["nasm", "-f", "elf64", str(asm_path), "-o", str(obj_path)], CFG.verifier_timeout_sec)
    if assemble.returncode != 0:
        return VerifyResult(
            reward=reward,
            assembled=False,
            linked=False,
            ran=False,
            correct=False,
            public_pass_rate=0.0,
            hidden_pass_rate=0.0,
            stdout=assemble.stdout,
            stderr=assemble.stderr,
            exit_code=None,
            stage_failed="assemble",
            penalties=penalties,
        )
    reward += CFG.assemble_weight

    link = run_cmd(["ld", str(obj_path), "-o", str(bin_path)], CFG.verifier_timeout_sec)
    if link.returncode != 0:
        return VerifyResult(
            reward=reward,
            assembled=True,
            linked=False,
            ran=False,
            correct=False,
            public_pass_rate=0.0,
            hidden_pass_rate=0.0,
            stdout=link.stdout,
            stderr=link.stderr,
            exit_code=None,
            stage_failed="link",
            penalties=penalties,
        )
    reward += CFG.link_weight

    run = run_cmd([str(bin_path)], CFG.verifier_timeout_sec)
    if run.returncode in (124, 125, 126, 127) or run.returncode < 0:
        return VerifyResult(
            reward=reward,
            assembled=True,
            linked=True,
            ran=False,
            correct=False,
            public_pass_rate=0.0,
            hidden_pass_rate=0.0,
            stdout=run.stdout,
            stderr=run.stderr,
            exit_code=run.returncode,
            stage_failed="run",
            penalties=penalties,
        )
    reward += CFG.run_weight

    public_hits = 0
    public_total = 0
    if task.get("expected_stdout") is not None:
        public_total += 1
        public_hits += int(run.stdout == task.get("expected_stdout"))
    elif task.get("expected_exit_code") is not None:
        public_total += 1
        public_hits += int(run.returncode == task.get("expected_exit_code"))

    hidden_tests = hidden_tests_for_task(task)
    hidden_hits = 0
    for check in hidden_tests:
        if check.get("expected_stdout") is not None:
            hidden_hits += int(run.stdout == check.get("expected_stdout"))
        elif check.get("expected_exit_code") is not None:
            hidden_hits += int(run.returncode == check.get("expected_exit_code"))
        else:
            hidden_hits += int(run.returncode == 0)

    public_rate = public_hits / max(1, public_total)
    hidden_rate = hidden_hits / max(1, len(hidden_tests))
    reward += CFG.public_test_weight * public_rate
    reward += CFG.hidden_test_weight * hidden_rate

    cleanliness_bonus = 0.0
    if len(asm_code.splitlines()) <= 20 and "```" not in asm_code:
        cleanliness_bonus = CFG.cleanliness_weight
        reward += cleanliness_bonus

    correct = hidden_rate >= 1.0
    return VerifyResult(
        reward=reward,
        assembled=True,
        linked=True,
        ran=True,
        correct=correct,
        public_pass_rate=public_rate,
        hidden_pass_rate=hidden_rate,
        stdout=run.stdout,
        stderr=run.stderr,
        exit_code=run.returncode,
        stage_failed=None if correct else "correctness",
        cleanliness_bonus=cleanliness_bonus,
        penalties=penalties,
    )


# %%
@dataclass
class EnvState:
    task: dict[str, Any]
    draft: str = ""
    last_result: VerifyResult | None = None
    step_idx: int = 0
    done: bool = False


class AsmForgeEnv:
    def __init__(self, task: dict[str, Any]) -> None:
        self.task = task
        self.state = EnvState(task=task)

    def reset(self) -> dict[str, Any]:
        self.state = EnvState(task=self.task)
        return self.observe()

    def observe(self) -> dict[str, Any]:
        last = self.state.last_result
        return {
            "instruction": self.state.task["instruction"],
            "draft": self.state.draft,
            "step_idx": self.state.step_idx,
            "last_stage_failed": None if last is None else last.stage_failed,
            "last_stdout": "" if last is None else last.stdout,
            "last_stderr": "" if last is None else last.stderr,
            "last_hidden_pass_rate": 0.0 if last is None else last.hidden_pass_rate,
            "done": self.state.done,
        }

    def step(self, action_type: str, program: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.state.done:
            return self.observe(), 0.0, True, {"reason": "already_done"}

        prev_score = 0.0 if self.state.last_result is None else self.state.last_result.reward
        self.state.step_idx += 1
        self.state.draft = program
        result = evaluate_candidate(self.state.task, program, sample_id=f"{self.state.task['task_id']}-s{self.state.step_idx}")
        self.state.last_result = result

        reward = result.reward
        if result.reward > prev_score and self.state.step_idx > 1:
            reward += CFG.repair_gain_weight * (result.reward - prev_score)
        elif result.reward < prev_score and self.state.step_idx > 1:
            reward += CFG.regress_penalty

        if action_type == "submit" or result.correct or self.state.step_idx >= CFG.max_episode_steps:
            self.state.done = True

        info = {
            "assembled": result.assembled,
            "linked": result.linked,
            "ran": result.ran,
            "correct": result.correct,
            "public_pass_rate": result.public_pass_rate,
            "hidden_pass_rate": result.hidden_pass_rate,
            "stage_failed": result.stage_failed,
        }
        return self.observe(), reward, self.state.done, info

    def close(self) -> None:
        return None


# %% [markdown]
# Model-loading cell sketch
#
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import LoraConfig, get_peft_model
#
# bnb = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.float16,
# )
# tokenizer = AutoTokenizer.from_pretrained(CFG.model_id, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     CFG.model_id,
#     trust_remote_code=True,
#     device_map="auto",
#     quantization_config=bnb,
#     torch_dtype=torch.float16,
# )
# lora_cfg = LoraConfig(
#     r=CFG.lora_r,
#     lora_alpha=CFG.lora_alpha,
#     lora_dropout=CFG.lora_dropout,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
#     bias="none",
#     task_type="CAUSAL_LM",
# )
# model = get_peft_model(model, lora_cfg)

# %%
def ranking_baseline(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Placeholder ranking baseline.

    In the real notebook, replace `dummy_generate` with model sampling.
    """
    rows: list[dict[str, Any]] = []
    for task in tasks:
        best_program = ""
        best_result: VerifyResult | None = None
        for cand_idx in range(CFG.num_candidates):
            dummy_generate = "global _start\nsection .text\n_start:\n    mov rax, 60\n    xor rdi, rdi\n    syscall"
            result = evaluate_candidate(task, dummy_generate, sample_id=f"{task['task_id']}-rank-{cand_idx}")
            if best_result is None or result.reward > best_result.reward:
                best_program = dummy_generate
                best_result = result
        rows.append(
            {
                "task_id": task["task_id"],
                "instruction": task["instruction"],
                "asm": best_program,
                "reward": 0.0 if best_result is None else best_result.reward,
                "correct": False if best_result is None else best_result.correct,
            }
        )
    return rows


# %%
def build_sft_pairs_from_ranked(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for row in rows:
        if row.get("correct"):
            pairs.append(
                {
                    "prompt": f"Task: {row['instruction']}",
                    "completion": row["asm"],
                }
            )
    return pairs


# %% [markdown]
# GRPO wiring sketch
#
# from datasets import Dataset
# from trl import GRPOConfig, GRPOTrainer
#
# def reward_fn(completions, prompts=None, **kwargs):
#     ...
#
# def rollout_func(prompts, tokenizer, model, **kwargs):
#     # Option A: one-shot with repair traces.
#     # Option B: true environment rollout using AsmForgeEnv.
#     ...
#
# train_dataset = Dataset.from_list([...])
# args = GRPOConfig(
#     output_dir="/kaggle/working/out-grpo",
#     learning_rate=CFG.grpo_learning_rate,
#     per_device_train_batch_size=CFG.grpo_per_device_batch_size,
#     gradient_accumulation_steps=CFG.grpo_gradient_accumulation_steps,
#     num_generations=CFG.grpo_num_generations,
#     max_completion_length=CFG.max_new_tokens,
#     beta=CFG.grpo_beta,
#     temperature=CFG.sample_temperature,
#     top_p=CFG.sample_top_p,
#     top_k=CFG.sample_top_k,
#     repetition_penalty=CFG.repetition_penalty,
#     report_to="none",
# )
# trainer = GRPOTrainer(
#     model=model,
#     args=args,
#     train_dataset=train_dataset,
#     reward_funcs=[reward_fn],
#     processing_class=tokenizer,
# )
# trainer.train()

# %%
def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = max(1, len(rows))
    correct = sum(1 for row in rows if row.get("correct"))
    return {
        "count": float(len(rows)),
        "correct_at_1": correct / total,
    }


# %%
if __name__ == "__main__":
    tasks_path = repo_path(CFG.train_tasks)
    if tasks_path.exists():
        tasks = read_jsonl(tasks_path)
        ranked = ranking_baseline(tasks[: min(4, len(tasks))])
        print(json.dumps(evaluate_rows(ranked), indent=2))
    else:
        print("Blueprint loaded. Run cell-by-cell inside a notebook.")

