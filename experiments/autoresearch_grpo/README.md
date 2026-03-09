# Autoresearch GRPO

This folder adapts the operating model of `karpathy/autoresearch` to the real
CodeForge GRPO experiment.

Files:

- `program.md`: operating instructions for the autonomous research loop.
- `target_config.yaml`: the single mutable experiment target. This is the
  GRPO equivalent of `train.py` in the original repo.

The fixed harness lives outside this folder:

- `scripts/run_agentic_grpo.py`
- `scripts/generate_synthetic_tasks.py`
- `src/verifier.py`
- `src/trainer.py`

The loop is:

1. Keep a baseline `target_config.yaml`.
2. Mutate only `target_config.yaml`.
3. Run the real GRPO experiment.
4. Measure the real GRPO metrics.
5. Keep the new target only if metrics improved.
6. Otherwise restore the previous best target.
