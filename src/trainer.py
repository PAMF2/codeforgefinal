from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .best_of_n import BestOfN, BestOfNConfig
from .mcts import MCTSConfig, MCTSLineSearch
from .prompt_engine import PromptEngine, PromptItem
from .reward import RewardPipeline
from .utils import SYS_PROMPT, ensure_dir, sanitize_model_output

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None

try:
    from peft import LoraConfig, get_peft_model
except Exception:  # pragma: no cover
    LoraConfig = None
    get_peft_model = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None

try:
    from datasets import Dataset
except Exception:  # pragma: no cover
    Dataset = None

try:
    from huggingface_hub import HfApi, snapshot_download, upload_folder
except Exception:  # pragma: no cover
    HfApi = None
    snapshot_download = None
    upload_folder = None

try:
    from trl import GRPOConfig, GRPOTrainer
except Exception:  # pragma: no cover
    GRPOConfig = None
    GRPOTrainer = None


@dataclass
class RuntimeConfig:
    raw: dict[str, Any]

    @property
    def iterations(self) -> int:
        return int(self.raw["training"]["iterations"])

    @property
    def prompts_per_iteration(self) -> int:
        return int(self.raw["training"]["prompts_per_iteration"])


def load_config(path: str | Path) -> RuntimeConfig:
    return RuntimeConfig(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(callable_obj)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


class DummyGenerator:
    """MVP generator to validate the full RL data path before model wiring."""

    def __call__(
        self,
        prompt: str,
        n: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int | None = None,
        min_p: float | None = None,
        repetition_penalty: float = 1.0,
    ) -> list[str]:
        del prompt, max_new_tokens, temperature, top_p, top_k, min_p, repetition_penalty
        return [
            """global _start
section .text
_start:
    mov rax, 60
    mov rdi, 0
    syscall"""
            for _ in range(n)
        ]


class HFTextGenerator:
    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._ensure_lm_head_dtype()

    def _find_lm_head(self) -> Any | None:
        # Plain HF model path
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            return lm_head

        # PEFT wrapper path
        base = getattr(self.model, "base_model", None)
        if base is not None:
            lm_head = getattr(base, "lm_head", None)
            if lm_head is not None and hasattr(lm_head, "weight"):
                return lm_head
            inner = getattr(base, "model", None)
            lm_head = getattr(inner, "lm_head", None) if inner is not None else None
            if lm_head is not None and hasattr(lm_head, "weight"):
                return lm_head

        # Fallback search
        for name, module in self.model.named_modules():
            if name.endswith("lm_head") and hasattr(module, "weight"):
                return module
        return None

    def _ensure_lm_head_dtype(self) -> None:
        """Keep lm_head dtype aligned with hidden states for 4-bit + PEFT generation."""
        if torch is None:
            return
        lm_head = self._find_lm_head()
        if lm_head is None or not hasattr(lm_head, "weight"):
            return
        try:
            # T4 + 4-bit QLoRA path should run generation in fp16.
            target_dtype = torch.float16 if torch.cuda.is_available() else next(self.model.parameters()).dtype
            if lm_head.weight.dtype != target_dtype:
                old_dtype = lm_head.weight.dtype
                lm_head.to(dtype=target_dtype)
                if getattr(lm_head, "bias", None) is not None and lm_head.bias.dtype != target_dtype:
                    lm_head.bias.data = lm_head.bias.data.to(target_dtype)
                print(
                    f"[CodeForge] Aligned lm_head dtype: "
                    f"{old_dtype} -> {target_dtype}"
                )
            # Some PEFT + accelerate combinations keep lm_head in fp32 while activations are fp16.
            # Guard forward to cast activations to lm_head dtype and avoid runtime dtype mismatch.
            if not getattr(lm_head, "_codeforge_dtype_guard", False):
                old_forward = lm_head.forward

                def guarded_forward(x: Any) -> Any:
                    w = getattr(lm_head, "weight", None)
                    if w is not None and hasattr(x, "dtype") and x.dtype != w.dtype:
                        x = x.to(w.dtype)
                    return old_forward(x)

                lm_head.forward = guarded_forward  # type: ignore[assignment]
                lm_head._codeforge_dtype_guard = True

            # Extra safety: install pre-hook on every lm_head module found.
            # This survives most wrapper paths and keeps inputs in the expected dtype.
            for name, module in self.model.named_modules():
                if not name.endswith("lm_head"):
                    continue
                if not hasattr(module, "weight"):
                    continue
                if getattr(module, "_codeforge_pre_hook_guard", False):
                    continue

                def _pre_hook(mod: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
                    if not args:
                        return args
                    x = args[0]
                    w = getattr(mod, "weight", None)
                    if w is not None and hasattr(x, "dtype") and x.dtype != w.dtype:
                        x = x.to(w.dtype)
                        return (x, *args[1:])
                    return args

                module.register_forward_pre_hook(_pre_hook)
                module._codeforge_pre_hook_guard = True
        except Exception as exc:
            print(f"[CodeForge] lm_head dtype alignment skipped: {exc}")

    def __call__(
        self,
        prompt: str,
        n: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int | None = None,
        min_p: float | None = None,
        repetition_penalty: float = 1.0,
    ) -> list[str]:
        self._ensure_lm_head_dtype()
        encoded = self.tokenizer(
            [prompt] * n,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

        # Generation does not need gradient checkpointing and can trigger noisy warnings.
        # Temporarily disable it for faster/cleaner sampling, then restore.
        gc_was_enabled = bool(getattr(self.model, "is_gradient_checkpointing", False))
        use_cache_prev = getattr(getattr(self.model, "config", None), "use_cache", None)
        try:
            if gc_was_enabled and hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()
            if use_cache_prev is not None and hasattr(self.model, "config"):
                self.model.config.use_cache = True

            with torch.no_grad():
                outputs = self.model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k if top_k is not None else 50,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    remove_invalid_values=True,
                    renormalize_logits=True,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            if use_cache_prev is not None and hasattr(self.model, "config"):
                self.model.config.use_cache = use_cache_prev
            if gc_was_enabled and hasattr(self.model, "gradient_checkpointing_enable"):
                try:
                    self.model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except Exception:
                    self.model.gradient_checkpointing_enable()

        prompt_len = encoded["input_ids"].shape[1]
        completions = outputs[:, prompt_len:]
        return self.tokenizer.batch_decode(completions, skip_special_tokens=True)


@dataclass
class TrainBundle:
    model: Any
    tokenizer: Any
    optimizer: Any
    generator: Any


@dataclass
class TRLBundle:
    trainer: Any


def _build_mcts_searcher(cfg: RuntimeConfig, generator: Any, reward_pipeline: RewardPipeline) -> MCTSLineSearch:
    mcts_raw = cfg.raw.get("mcts", {})
    mcts_cfg = MCTSConfig(
        simulations=int(mcts_raw.get("simulations", 32)),
        max_lines=int(mcts_raw.get("max_lines", 30)),
        branch_factor=int(mcts_raw.get("branch_factor", 4)),
        exploration_constant=float(mcts_raw.get("exploration_constant", 1.414)),
        max_depth=int(mcts_raw.get("max_depth", 15)),
    )
    return MCTSLineSearch(cfg=mcts_cfg, generator=generator, reward_pipeline=reward_pipeline)


def _group_relative_weights(rows: list[dict[str, Any]]) -> list[float]:
    by_prompt: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_prompt.setdefault(row["prompt_id"], []).append(i)

    weights = [0.0 for _ in rows]
    for _, indices in by_prompt.items():
        rewards = [float(rows[i]["reward"]) for i in indices]
        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / max(1, len(rewards))
        std = variance ** 0.5
        advantages = [(r - mean) / (std + 1e-6) for r in rewards]

        pos = [max(0.0, a) for a in advantages]
        if sum(pos) == 0.0:
            best_idx = max(range(len(indices)), key=lambda j: rewards[j])
            pos[best_idx] = 1.0

        s = sum(pos)
        norm = [p / s for p in pos]
        for local_j, row_idx in enumerate(indices):
            weights[row_idx] = norm[local_j]

    return weights


def run_grpo_update_manual(rows: list[dict[str, Any]], cfg: RuntimeConfig, bundle: TrainBundle | None) -> dict[str, float]:
    if bundle is None:
        return {"grpo_loss": 0.0, "kl": 0.0, "skipped_rows": 0.0}

    if torch is None or F is None:
        raise RuntimeError("Torch is required for non-dry training mode.")

    model = bundle.model
    tokenizer = bundle.tokenizer
    optimizer = bundle.optimizer
    model.train()

    grad_acc = int(cfg.raw["training"].get("gradient_accumulation_steps", 4))
    max_len = int(cfg.raw["training"].get("train_max_seq_len", 1024))
    grad_clip = float(cfg.raw["training"].get("grad_clip_norm", 1.0))

    weights = _group_relative_weights(rows)
    total_loss = 0.0
    used = 0
    skipped = 0

    optimizer.zero_grad(set_to_none=True)
    for i, row in enumerate(rows):
        weight = float(weights[i])
        if weight <= 0.0 or not math.isfinite(weight):
            continue

        prompt_text = str(row.get("prompt_text") or f"{SYS_PROMPT}\n\nTask: {row['instruction']}\n")
        completion = row["asm"].strip() + "\n"
        if not completion.strip():
            skipped += 1
            continue
        full_text = prompt_text + completion

        full_enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_len)
        prompt_enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = full_enc["input_ids"].to(model.device)
        attention_mask = full_enc["attention_mask"].to(model.device)
        labels = input_ids.clone()

        prompt_len = min(prompt_enc["input_ids"].shape[1], labels.shape[1])
        labels[:, :prompt_len] = -100
        if prompt_len >= labels.shape[1]:
            skipped += 1
            continue

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :].contiguous()
        target = labels[:, 1:].contiguous()
        valid_targets = int((target != -100).sum().item())
        if valid_targets == 0:
            skipped += 1
            continue

        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            target.view(-1),
            ignore_index=-100,
            reduction="mean",
        )
        if not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        weighted_loss = loss * weight
        if not torch.isfinite(weighted_loss):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        (weighted_loss / grad_acc).backward()

        total_loss += float(weighted_loss.detach().cpu())
        used += 1

        if used % grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if used > 0 and used % grad_acc != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    model.eval()

    avg_loss = total_loss / max(1, used)
    return {"grpo_loss": avg_loss, "kl": 0.0, "skipped_rows": float(skipped)}


def _extract_completion_text(raw_completion: Any) -> str:
    if isinstance(raw_completion, str):
        return raw_completion

    if isinstance(raw_completion, list):
        parts: list[str] = []
        for msg in raw_completion:
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    parts.append(content)
        if parts:
            return "\n".join(parts)

    return str(raw_completion)


def _build_trl_reward_fn(
    reward_pipeline: RewardPipeline,
    prompt_by_instruction: dict[str, PromptItem],
) -> Any:
    counter = {"n": 0}

    def reward_fn(completions: Any, prompts: Any = None, **_: Any) -> list[float]:
        if prompts is None:
            prompts = [""] * len(completions)

        rewards: list[float] = []
        for prompt_text, completion in zip(prompts, completions):
            instruction = str(prompt_text).split("Task:", 1)[-1].strip()
            prompt_item = prompt_by_instruction.get(instruction)
            if prompt_item is None:
                rewards.append(0.0)
                continue

            asm = sanitize_model_output(_extract_completion_text(completion))
            sample_id = f"trl-{counter['n']}"
            counter["n"] += 1
            result = reward_pipeline.evaluate(prompt_item, asm, sample_id)
            rewards.append(float(result.reward))

        return rewards

    return reward_fn


def maybe_build_trl_bundle(
    cfg: RuntimeConfig,
    train_bundle: TrainBundle | None,
    reward_pipeline: RewardPipeline,
    prompt_items: list[PromptItem],
    artifacts_dir: Path,
) -> TRLBundle | None:
    training = cfg.raw["training"]
    backend = str(training.get("grpo_backend", "manual")).lower()
    if backend != "trl":
        return None
    if bool(training.get("dry_run", True)):
        return None

    if train_bundle is None:
        raise RuntimeError("TRL backend requires non-dry mode with a loaded model.")
    if GRPOTrainer is None or GRPOConfig is None or Dataset is None:
        raise RuntimeError("TRL/Datasets dependency missing. Install trl and datasets.")

    prompt_rows = [{"prompt": f"{SYS_PROMPT}\n\nTask: {p.instruction}"} for p in prompt_items]
    train_dataset = Dataset.from_list(prompt_rows)

    output_dir = artifacts_dir / "trl"
    ensure_dir(output_dir)

    use_vllm = bool(training.get("use_vllm", False))
    base_args: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": float(training.get("learning_rate", 5e-6)),
        "per_device_train_batch_size": int(training.get("batch_size", 2)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 4)),
        "num_train_epochs": int(training.get("num_train_epochs", 1)),
        "num_generations": int(training.get("generations_per_prompt", 16)),
        "max_completion_length": int(training.get("max_new_tokens", 256)),
        "temperature": float(training.get("temperature", 0.8)),
        "top_p": float(training.get("top_p", 0.95)),
        "top_k": (
            int(training["top_k"])
            if training.get("top_k") is not None
            else None
        ),
        "min_p": (
            float(training["min_p"])
            if training.get("min_p") is not None
            else None
        ),
        "repetition_penalty": float(training.get("repetition_penalty", 1.0)),
        "beta": float(training.get("kl_beta", 0.1)),
        "logging_steps": 1,
        "report_to": "wandb" if bool(training.get("use_wandb", True)) and wandb is not None else "none",
        # vLLM fast generation — 5-10x faster than HF generate on large batches.
        # With tensor_parallel_size=2 on 2x T4: vLLM shards across both GPUs (8GB each),
        # leaving ~7GB per GPU for the 4-bit training model.
        "use_vllm": use_vllm,
        "vllm_tensor_parallel_size": int(training.get("vllm_tensor_parallel_size", 1)),
        "vllm_gpu_memory_utilization": float(training.get("vllm_gpu_memory_utilization", 0.4)),
        "vllm_max_model_len": int(training.get("vllm_max_model_len", 512)),
    }
    per_device_bs = int(training.get("batch_size", 2))
    num_gens = int(training.get("generations_per_prompt", 16))
    if per_device_bs % max(1, num_gens) != 0:
        gen_bs = int(math.ceil(per_device_bs / max(1, num_gens)) * max(1, num_gens))
        base_args["generation_batch_size"] = gen_bs
        print(
            f"[CodeForge] Adjusted generation_batch_size to {gen_bs} "
            f"(batch_size={per_device_bs}, num_generations={num_gens})"
        )
    if torch is not None and not torch.cuda.is_available():
        base_args.update({"use_cpu": True, "bf16": False, "fp16": False})
    elif torch is not None and torch.cuda.is_available():
        bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        base_args.update({"use_cpu": False, "bf16": bf16_supported, "fp16": not bf16_supported})
        if torch.cuda.device_count() > 1:
            base_args["ddp_find_unused_parameters"] = False

    grpo_cfg_kwargs = _filter_kwargs(GRPOConfig, base_args)
    grpo_cfg = GRPOConfig(**grpo_cfg_kwargs)

    prompt_by_instruction = {p.instruction: p for p in prompt_items}
    reward_fn = _build_trl_reward_fn(reward_pipeline, prompt_by_instruction)

    trainer_kwargs = {
        "model": train_bundle.model,
        "args": grpo_cfg,
        "train_dataset": train_dataset,
        "reward_funcs": [reward_fn],
        "processing_class": train_bundle.tokenizer,
    }

    # TRL may expect `model.warnings_issued` on some transformers/peft versions.
    # Ensure this exists on wrapped and base models before trainer construction.
    model_ref = train_bundle.model
    for candidate in (
        model_ref,
        getattr(model_ref, "base_model", None),
        getattr(getattr(model_ref, "base_model", None), "model", None),
    ):
        if candidate is not None and not hasattr(candidate, "warnings_issued"):
            try:
                setattr(candidate, "warnings_issued", {})
            except Exception:
                pass

    filtered_trainer_kwargs = _filter_kwargs(GRPOTrainer.__init__, trainer_kwargs)
    trainer = GRPOTrainer(**filtered_trainer_kwargs)
    return TRLBundle(trainer=trainer)


def run_grpo_update_trl(trl_bundle: TRLBundle | None) -> dict[str, float]:
    if trl_bundle is None:
        return {"grpo_loss": 0.0, "kl": 0.0}

    result = trl_bundle.trainer.train()
    log_history = getattr(result, "metrics", {}) or {}
    return {
        "grpo_loss": float(log_history.get("train_loss", 0.0)),
        "kl": float(log_history.get("kl", 0.0)),
    }


def _hf_token_from_env() -> str | None:
    return (
        None
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )


def maybe_push_checkpoint_to_hub(
    cfg: RuntimeConfig,
    ckpt_path: Path,
    iteration: int,
) -> None:
    training = cfg.raw["training"]
    if not bool(training.get("push_to_hub", False)):
        return
    if HfApi is None or upload_folder is None:
        print("[CodeForge] huggingface_hub not available; skipping push_to_hub")
        return

    token = _hf_token_from_env()
    repo_id = str(training.get("hub_repo_id", "")).strip()
    if not token or not repo_id:
        print("[CodeForge] Missing HF token or hub_repo_id; skipping push_to_hub")
        return

    private = bool(training.get("hub_private", True))
    fallback_repo_id = str(training.get("hub_fallback_repo_id", "")).strip()
    api = HfApi(token=token)

    target_repo = repo_id
    try:
        api.create_repo(repo_id=target_repo, repo_type="model", private=private, exist_ok=True)
    except Exception as exc:
        if not fallback_repo_id:
            print(f"[CodeForge] Cannot create repo {repo_id}: {exc}")
            return
        print(f"[CodeForge] Cannot create repo {repo_id}; using fallback {fallback_repo_id}")
        target_repo = fallback_repo_id
        api.create_repo(repo_id=target_repo, repo_type="model", private=private, exist_ok=True)

    upload_folder(
        repo_id=target_repo,
        repo_type="model",
        folder_path=str(ckpt_path),
        path_in_repo=f"checkpoints/iter_{iteration}",
        token=token,
        commit_message=f"Add checkpoint iter_{iteration}",
    )
    print(f"[CodeForge] Pushed checkpoint iter_{iteration} to {target_repo}")


def evaluate_candidates(
    reward_pipeline: RewardPipeline,
    prompt_item: PromptItem,
    candidates: list[str],
    sample_prefix: str,
) -> list[dict[str, Any]]:
    # Build batch and evaluate in parallel (ThreadPoolExecutor inside evaluate_batch)
    items = [(prompt_item, asm, f"{sample_prefix}-{idx}") for idx, asm in enumerate(candidates)]
    results = reward_pipeline.evaluate_batch(items, workers=min(32, len(items)))
    return [
        {
            "prompt_id": prompt_item.id,
            "instruction": prompt_item.instruction,
            "tier": prompt_item.tier,
            "asm": asm,
            "reward": result.reward,
            "assembled": result.assembled,
            "linked": result.linked,
            "ran": result.ran,
            "correct": result.correct,
            "stage_failed": result.stage_failed,
            "source": "bon",
        }
        for asm, result in zip(candidates, results)
    ]


def _per_tier_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute assembly/correctness rates per tier for W&B logging."""
    by_tier: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        t = int(r.get("tier", 0))
        by_tier.setdefault(t, []).append(r)

    metrics: dict[str, float] = {}
    for tier, tier_rows in sorted(by_tier.items()):
        n = max(1, len(tier_rows))
        metrics[f"reward/tier{tier}_assemble"] = sum(1 for r in tier_rows if r["assembled"]) / n
        metrics[f"reward/tier{tier}_correct"] = sum(1 for r in tier_rows if r["correct"]) / n
        metrics[f"reward/tier{tier}_avg_reward"] = sum(r["reward"] for r in tier_rows) / n
    return metrics


def maybe_build_train_bundle(cfg: RuntimeConfig, resume_from: str | None = None) -> TrainBundle | None:
    training = cfg.raw["training"]
    if bool(training.get("dry_run", True)):
        return None

    if AutoModelForCausalLM is None or AutoTokenizer is None or torch is None:
        raise RuntimeError("Missing ML dependencies. Install requirements and rerun.")

    model_cfg = cfg.raw["model"]
    model_name = model_cfg["name_or_path"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    load_in_4bit = bool(model_cfg.get("load_in_4bit", True))
    use_unsloth = bool(training.get("use_unsloth", False))

    # Try Unsloth fast path first.
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=int(training.get("train_max_seq_len", 1024)),
                dtype=None,
                load_in_4bit=load_in_4bit,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=int(model_cfg.get("lora_r", 16)),
                lora_alpha=int(model_cfg.get("lora_alpha", 32)),
                lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=int(cfg.raw["project"]["seed"]),
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            learning_rate = float(training.get("learning_rate", 5e-6))
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            print("[CodeForge] Using Unsloth fast path.")
            return TrainBundle(model=model, tokenizer=tokenizer, optimizer=optimizer, generator=HFTextGenerator(model=model, tokenizer=tokenizer))
        except ImportError:
            print("[CodeForge] Unsloth not available; falling back to standard HF.")

    quant_cfg = None
    if load_in_4bit:
        if BitsAndBytesConfig is None:
            raise RuntimeError("bitsandbytes/transformers quantization config unavailable.")
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Under DDP (accelerate launch --num_processes N), LOCAL_RANK is set.
    # In that case accelerate handles device placement — don't use device_map.
    is_ddp = os.environ.get("LOCAL_RANK") is not None
    device_map = None if is_ddp else model_cfg.get("device_map", "auto")
    max_memory = None
    if not is_ddp and torch is not None and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"[CodeForge] CUDA detected: {gpu_count} GPU(s)")
        if gpu_count > 1:
            per_gpu_gb = int(model_cfg.get("max_memory_per_gpu_gb", 14))
            max_memory = {i: f"{per_gpu_gb}GiB" for i in range(gpu_count)}
            max_memory["cpu"] = "64GiB"
            print(f"[CodeForge] Multi-GPU (model parallel) device_map={device_map}, max_memory={max_memory}")
    elif is_ddp:
        print(f"[CodeForge] DDP mode detected (LOCAL_RANK={os.environ['LOCAL_RANK']}); accelerate handles devices")

    # SDPA (scaled-dot-product attention) — faster than eager, works on all GPUs
    # Flash Attention 2 requires Ampere+ (SM80+); T4 is SM75, so we use sdpa.
    model_load_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
        "max_memory": max_memory,
        "quantization_config": quant_cfg,
    }

    # torch_dtype: must be set when using 4-bit QLoRA with device_map (multi-GPU).
    # Without it, non-quantized layers (e.g. lm_head) default to fp32 while
    # activations are fp16 → RuntimeError: expected scalar type Float but found Half.
    # T4 does NOT support bfloat16 — use float16.
    torch_dtype_str = model_cfg.get("torch_dtype", None)
    if torch_dtype_str and torch is not None:
        model_load_kwargs["torch_dtype"] = getattr(torch, torch_dtype_str)
    elif load_in_4bit and torch is not None:
        # Sane default for 4-bit QLoRA: always use float16
        model_load_kwargs["torch_dtype"] = torch.float16

    attn_impl = str(model_cfg.get("attn_implementation", "sdpa"))
    if attn_impl != "eager":
        model_load_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_load_kwargs)

    if resume_from:
        from peft import PeftModel
        print(f"[CodeForge] Resuming LoRA from: {resume_from}")
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
        model.print_trainable_parameters()
    elif get_peft_model is not None and LoraConfig is not None:
        lora_cfg = LoraConfig(
            r=int(model_cfg.get("lora_r", 16)),
            lora_alpha=int(model_cfg.get("lora_alpha", 32)),
            lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # Gradient checkpointing: trades compute for memory, enables larger effective batch
    if bool(training.get("gradient_checkpointing", True)):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("[CodeForge] Gradient checkpointing enabled")
        except Exception as gc_err:
            print(f"[CodeForge] Gradient checkpointing skipped: {gc_err}")

    learning_rate = float(training.get("learning_rate", 5e-6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = HFTextGenerator(model=model, tokenizer=tokenizer)

    return TrainBundle(model=model, tokenizer=tokenizer, optimizer=optimizer, generator=generator)


def _has_lora_weights(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "adapter_model.safetensors").exists() or (checkpoint_dir / "adapter_model.bin").exists()


def _hydrate_checkpoint_from_hub(cfg: RuntimeConfig, checkpoints_dir: Path, iter_id: int) -> bool:
    if snapshot_download is None:
        return False

    training = cfg.raw.get("training", {})
    repo_candidates = [
        str(training.get("hub_repo_id", "")).strip(),
        str(training.get("hub_fallback_repo_id", "")).strip(),
    ]
    repo_ids = [r for r in repo_candidates if r]
    if not repo_ids:
        return False

    token = _hf_token_from_env()
    local_root = checkpoints_dir.parent
    prefix = f"{checkpoints_dir.name}/iter_{iter_id}/"
    pattern = f"{prefix}*"

    for repo_id in repo_ids:
        try:
            print(f"[CodeForge] Attempting checkpoint hydration from {repo_id} ({pattern})")
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                token=token,
                local_dir=str(local_root),
                allow_patterns=[pattern],
            )
            if _has_lora_weights(checkpoints_dir / f"iter_{iter_id}"):
                print(f"[CodeForge] Hydrated iter_{iter_id} from {repo_id}")
                return True
        except Exception as exc:
            print(f"[CodeForge] Hydration failed from {repo_id}: {exc}")
            continue

    return False


def _resolve_resume_checkpoint(cfg: RuntimeConfig, checkpoints_dir: Path, start_iter: int) -> tuple[str | None, int]:
    if start_iter <= 0:
        return None, start_iter

    requested_prev = start_iter - 1
    for it in range(requested_prev, -1, -1):
        ckpt_candidate = checkpoints_dir / f"iter_{it}"
        if not ckpt_candidate.exists() or not _has_lora_weights(ckpt_candidate):
            _hydrate_checkpoint_from_hub(cfg, checkpoints_dir, it)
        if not ckpt_candidate.exists():
            continue
        if _has_lora_weights(ckpt_candidate):
            if it != requested_prev:
                print(
                    f"[CodeForge] WARNING: checkpoint iter_{requested_prev} incomplete/missing; "
                    f"falling back to iter_{it}"
                )
            return str(ckpt_candidate), it + 1
        print(f"[CodeForge] WARNING: checkpoint iter_{it} exists but has no adapter_model.*; skipping")

    print("[CodeForge] WARNING: no valid LoRA checkpoint found; starting fresh")
    return None, start_iter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo_config.yaml")
    parser.add_argument(
        "--start-iter", type=int, default=0,
        help="Resume training from this iteration index. Loads the LoRA checkpoint "
             "from iter_{N-1} in checkpoints_dir. (default: 0 = start fresh)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    start_iter: int = args.start_iter
    random.seed(int(cfg.raw["project"]["seed"]))

    training = cfg.raw["training"]
    paths = cfg.raw["paths"]
    artifacts_dir = Path(paths["artifacts_dir"])
    checkpoints_dir = Path(paths["checkpoints_dir"])
    ensure_dir(artifacts_dir)

    prompts = PromptEngine(paths["prompt_dataset"])
    reward_pipeline = RewardPipeline(
        artifacts_dir=artifacts_dir,
        timeout_seconds=int(cfg.raw["reward"]["timeout_seconds"]),
        stage_weights=cfg.raw["reward"].get("stage_weights"),
    )

    bon_cfg = BestOfNConfig(
        n=int(training["generations_per_prompt"]),
        max_new_tokens=int(training["max_new_tokens"]),
        temperature=float(training["temperature"]),
        top_p=float(training["top_p"]),
        top_k=(
            int(training["top_k"])
            if training.get("top_k") is not None
            else None
        ),
        min_p=(
            float(training["min_p"])
            if training.get("min_p") is not None
            else None
        ),
        repetition_penalty=float(training.get("repetition_penalty", 1.0)),
    )

    # Resolve resume checkpoint from iter_{start_iter - 1}, with fallback for partial checkpoints.
    resume_path, adjusted_start_iter = _resolve_resume_checkpoint(cfg, checkpoints_dir, start_iter)
    if resume_path is not None:
        print(f"[CodeForge] Resuming LoRA from checkpoint: {resume_path}")
    if adjusted_start_iter != start_iter:
        print(f"[CodeForge] Adjusted start iteration: {start_iter} -> {adjusted_start_iter}")
        start_iter = adjusted_start_iter

    train_bundle = maybe_build_train_bundle(cfg, resume_from=resume_path)
    generator = train_bundle.generator if train_bundle is not None else DummyGenerator()
    best_of_n = BestOfN(generator=generator, cfg=bon_cfg)
    mcts_searcher = _build_mcts_searcher(cfg, generator, reward_pipeline)
    use_mcts_after = int(training.get("use_mcts_after_iteration", 999))
    use_random_sampling = bool(training.get("use_random_sampling", True))

    batch_prompts_for_init = prompts.sample(cfg.prompts_per_iteration)
    trl_bundle = maybe_build_trl_bundle(
        cfg=cfg,
        train_bundle=train_bundle,
        reward_pipeline=reward_pipeline,
        prompt_items=batch_prompts_for_init,
        artifacts_dir=artifacts_dir,
    )

    backend = str(training.get("grpo_backend", "manual")).lower()
    actual_mode = "dry_run" if train_bundle is None else f"real:{backend}"
    tier_counts = prompts.tier_counts()
    print(f"[CodeForge] Training mode: {actual_mode}")
    print(f"[CodeForge] Dataset: {len(prompts.all_items())} prompts, tiers: {tier_counts}")
    if start_iter > 0:
        print(f"[CodeForge] Resuming from iteration {start_iter} / {cfg.iterations}")

    use_wandb = bool(training.get("use_wandb", True)) and wandb is not None
    if use_wandb and not os.environ.get("WANDB_API_KEY"):
        print("[CodeForge] WANDB_API_KEY not set — disabling W&B to avoid interactive prompt")
        use_wandb = False
    run = None
    if use_wandb:
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
        run = wandb.init(project=cfg.raw["project"]["name"], config=cfg.raw, resume="allow")

    print("[CodeForge] Starting loop")
    for it in range(start_iter, cfg.iterations):
        if use_random_sampling:
            batch_prompts = prompts.sample_random(cfg.prompts_per_iteration)
        else:
            batch_prompts = prompts.sample(cfg.prompts_per_iteration)

        all_rows: list[dict[str, Any]] = []
        using_mcts = it >= use_mcts_after
        mcts_depths: list[float] = []

        for p in batch_prompts:
            user_prompt = f"{SYS_PROMPT}\n\nTask: {p.instruction}"
            # MCTS only for high-tier prompts (tier >= mcts_searcher.cfg.min_tier)
            if using_mcts and mcts_searcher.should_apply(p):
                mcts_rows = mcts_searcher.search(
                    task=user_prompt,
                    prompt_item=p,
                    sample_prefix=f"it{it}-{p.id}",
                )
                for row in mcts_rows:
                    mcts_depths.append(row.pop("mcts_avg_depth", 0.0))
                all_rows.extend(mcts_rows)
            else:
                candidates = best_of_n.generate(user_prompt)
                rows = evaluate_candidates(reward_pipeline, p, candidates, sample_prefix=f"it{it}-{p.id}")
                all_rows.extend(rows)

        rewards = [r["reward"] for r in all_rows]
        n_rows = max(1, len(all_rows))
        avg_reward = sum(rewards) / n_rows
        success_assemble = sum(1 for r in all_rows if r["assembled"]) / n_rows
        success_correct = sum(1 for r in all_rows if r["correct"]) / n_rows
        mcts_rows_count = sum(1 for r in all_rows if r.get("source") == "mcts")
        avg_mcts_depth = sum(mcts_depths) / max(1, len(mcts_depths))

        if backend == "trl":
            grpo_metrics = run_grpo_update_trl(trl_bundle)
        else:
            grpo_metrics = run_grpo_update_manual(all_rows, cfg, bundle=train_bundle)

        tier_metrics = _per_tier_metrics(all_rows)

        iter_metrics = {
            "iteration": it,
            "reward/mean": avg_reward,
            "reward/assembly_rate": success_assemble,
            "reward/correct_rate": success_correct,
            "mcts/rows": mcts_rows_count,
            "mcts/avg_depth": avg_mcts_depth,
            "source": "mcts" if using_mcts else "bon",
            **grpo_metrics,
            **tier_metrics,
        }

        print(json.dumps(iter_metrics, ensure_ascii=False))
        if run is not None:
            run.log(iter_metrics)

        out_path = artifacts_dir / f"iteration_{it}.json"
        out_path.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

        if train_bundle is not None:
            ensure_dir(checkpoints_dir)
            ckpt_path = checkpoints_dir / f"iter_{it}"
            ensure_dir(ckpt_path)
            train_bundle.model.save_pretrained(ckpt_path)
            train_bundle.tokenizer.save_pretrained(ckpt_path)
            maybe_push_checkpoint_to_hub(cfg=cfg, ckpt_path=ckpt_path, iteration=it)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
