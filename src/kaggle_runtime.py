from __future__ import annotations

import os


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
        except Exception:
            continue
        if val:
            return val, f"kaggle:{name}"
    return None, None


def load_kaggle_secrets(prefix: str = "[kaggle_runtime]") -> dict[str, str | bool | None]:
    hf, hf_src = load_secret_candidates(
        ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_TOKEN", "HF_KEY", "hf_token"]
    )
    wandb, wandb_src = load_secret_candidates(
        ["WANDB_API_KEY", "WANDB_KEY", "WANDB_TOKEN", "wandb_api_key"]
    )
    mistral, mistral_src = load_secret_candidates(
        ["MISTRAL_API_KEY", "MISTRAL_KEY", "MISTRAL_TOKEN", "mistral_api_key"]
    )

    if hf:
        os.environ["HF_TOKEN"] = hf.strip()
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf.strip()
    if wandb:
        os.environ["WANDB_API_KEY"] = wandb.strip()
    if mistral:
        os.environ["MISTRAL_API_KEY"] = mistral.strip()

    summary = {
        "hf_loaded": bool(os.getenv("HF_TOKEN")),
        "hf_source": hf_src,
        "wandb_loaded": bool(os.getenv("WANDB_API_KEY")),
        "wandb_source": wandb_src,
        "mistral_loaded": bool(os.getenv("MISTRAL_API_KEY")),
        "mistral_source": mistral_src,
    }
    print(
        f"{prefix} secrets | "
        f"HF_TOKEN={summary['hf_loaded']} ({hf_src or 'not found'}) "
        f"WANDB_API_KEY={summary['wandb_loaded']} ({wandb_src or 'not found'}) "
        f"MISTRAL_API_KEY={summary['mistral_loaded']} ({mistral_src or 'not found'})",
        flush=True,
    )
    return summary
