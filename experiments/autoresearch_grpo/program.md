# autoresearch for CodeForge GRPO

This is the CodeForge adaptation of the `karpathy/autoresearch` operating model.

## Scope

The fixed experiment harness is:

- `scripts/run_agentic_grpo.py`
- `scripts/generate_synthetic_tasks.py`
- verifier and reward logic in `src/`

Do not edit the harness during the autonomous loop.

The single mutable target is:

- `experiments/autoresearch_grpo/target_config.yaml`

This file is the GRPO equivalent of `train.py` in the original autoresearch repo.

## Goal

Improve the real CodeForge GRPO metrics on the real task set.

Primary metric priority:

1. `solved_rate`
2. `avg_final_reward`
3. `avg_repair_gain`
4. lower `skipped_rows`

## Loop

1. Start from the current `target_config.yaml`.
2. Mutate only `target_config.yaml`.
3. Run the real GRPO experiment.
4. Read the real metrics.
5. Mark the run as `keep`, `discard`, or `crash`.
6. If the run improved, keep the mutated target.
7. Otherwise restore the previous best target.

## Constraints

- Do not create a toy model.
- Do not replace the GRPO training loop with a proxy model.
- The experiment must run through `scripts/run_agentic_grpo.py`.
- Keep changes simple and measurable.

## Outputs

Each autoresearch run should produce:

- `results.tsv`
- `runs.json`
- `best_config.yaml`
- `best_artifacts/`
- per-experiment `run.log`
