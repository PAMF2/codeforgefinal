# Architecture

## Direction

This repo is built for Kaggle, not for local services.

Core idea:

1. sample multiple candidates
2. compile and execute all of them
3. rerank with objective reward
4. run one repair step with verifier feedback
5. keep solved traces for SFT and later RL

## Why This Layout

- `scripts/run_ranked_sampling.py` is the cheapest strong baseline.
- `src/verifier.py` is the core asset.
- `src/env.py` exists so we can move into agentic RL cleanly.
- `notebooks/qwen35_asm_agentic_blueprint.py` is the starting point for GRPO/OpenEnv experiments.
