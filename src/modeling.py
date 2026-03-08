from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import sanitize_model_output


def load_model_and_tokenizer(
    base_model: str,
    adapter_path: str | None = None,
    load_in_4bit: bool = True,
    trust_remote_code: bool = True,
    torch_dtype: str = "float16",
) -> tuple[Any, Any]:
    quant_cfg = None
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "device_map": "auto",
    }
    if quant_cfg is not None:
        model_kwargs["quantization_config"] = quant_cfg
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if adapter_path:
        adapter_path_obj = Path(adapter_path)
        if adapter_path_obj.exists():
            model = PeftModel.from_pretrained(model, str(adapter_path_obj), is_trainable=False)
    model.eval()
    return model, tokenizer


def generate_completion(
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
    enc = {key: value.to(model.device) for key, value in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            top_k=top_k if top_k is not None else 50,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            remove_invalid_values=True,
            renormalize_logits=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt):]
    return sanitize_model_output(text)
