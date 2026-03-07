# Assembly-SWE (CodeForge)

Assembly-SWE is a SWE-bench-inspired benchmark protocol for NASM x86-64 tasks.
It standardizes task format, deterministic verification, and leaderboard metrics.

## Layout

```text
assembly_swe/
  README.md
  schema.task.json
  datasets/
    sample_dev.jsonl
    dev_v1_30.jsonl
  examples/
    sample_predictions.jsonl
  tools/
    evaluate.py
    eval_all_iters.py
    benchmark_suite.py
    validate_dataset.py
```

## Task format (JSONL)

One task per line:

```json
{
  "task_id": "asm_hello_001",
  "tier": 1,
  "instruction": "Write NASM x86-64 Linux program that prints Hello\\n and exits 0.",
  "expected_stdout": "Hello\n",
  "expected_exit_code": 0
}
```

Supported keys:
- `task_id` (string, required)
- `tier` (int, required)
- `instruction` (string, required)
- `expected_stdout` (string, optional)
- `expected_exit_code` (int, optional)
- `tags` (list[string], optional)

## Predictions format (JSONL)

One candidate per line:

```json
{
  "task_id": "asm_hello_001",
  "candidate_id": "c0",
  "candidate_rank": 0,
  "asm": "global _start\nsection .text\n_start:\n..."
}
```

Supported keys:
- `task_id` (string, required)
- `asm` (string, required)
- `candidate_id` (string, optional)
- `candidate_rank` (int, optional, lower is better)

If `candidate_rank` is absent, file order is used.

## Metrics

- `pass@k`: fraction of tasks with at least one correct candidate in top-k.
- `assembly_rate@1`: fraction of top-1 candidates that assemble.
- `link_rate@1`: fraction of top-1 candidates that link.
- `run_rate@1`: fraction of top-1 candidates that execute.
- `correct_rate@1`: same as `pass@1`.
- `avg_reward@1`: average reward of top-1 candidates.
- Tier metrics: `tier_<n>_pass@1`, `tier_<n>_pass@k`.

## Run evaluation

```bash
python assembly_swe/tools/evaluate.py \
  --tasks assembly_swe/datasets/sample_dev.jsonl \
  --predictions assembly_swe/examples/sample_predictions.jsonl \
  --ks 1,3,5 \
  --outdir assembly_swe/results/latest
```

## Real model evaluation (checkpoint)

Generate real predictions from a trained checkpoint and evaluate:

```bash
python assembly_swe/tools/generate_predictions.py \
  --tasks assembly_swe/datasets/sample_dev.jsonl \
  --checkpoint-dir checkpoints/iter_30 \
  --out assembly_swe/results/iter30_predictions.jsonl \
  --load-in-4bit \
  --num-candidates 5 \
  --verifier reward \
  --repair-steps 1

python assembly_swe/tools/evaluate.py \
  --tasks assembly_swe/datasets/sample_dev.jsonl \
  --predictions assembly_swe/results/iter30_predictions.jsonl \
  --ks 1,3,5 \
  --outdir assembly_swe/results/iter30_eval
```

Outputs:
- `summary.json`
- `leaderboard.md`
- `rows_top1.jsonl`

## Validate a dataset before runs

```bash
python assembly_swe/tools/validate_dataset.py \
  --tasks assembly_swe/datasets/sample_dev.jsonl
```


## Notes

- Evaluator uses the same deterministic verification core (`nasm` + `ld` + run)
  via `src.reward.RewardPipeline`.
- This protocol is intentionally simple and reproducible for local and Kaggle runs.

## Paper-grade workflow

1. Validate tasks:
```bash
python assembly_swe/tools/validate_dataset.py \
  --tasks assembly_swe/datasets/dev_v1_30.jsonl
```

2. Evaluate checkpoints for one dataset:
```bash
python assembly_swe/tools/eval_all_iters.py \
  --repo-root . \
  --tasks assembly_swe/datasets/dev_v1_30.jsonl \
  --iter-start 1 --iter-end 30 \
  --ks 1,3,5 \
  --outdir assembly_swe/results/dev_v1_30 \
  --load-in-4bit \
  --num-candidates 5 \
  --verifier reward \
  --repair-steps 1 \
  --hub-repo-id mistral-hackaton-2026/codeforge
```

3. Evaluate multiple datasets/splits:
```bash
python assembly_swe/tools/benchmark_suite.py \
  --repo-root . \
  --tasks assembly_swe/datasets/sample_dev.jsonl,assembly_swe/datasets/dev_v1_30.jsonl \
  --iter-start 1 --iter-end 30 \
  --ks 1,3,5 \
  --outdir assembly_swe/results/suite \
  --load-in-4bit \
  --num-candidates 5 \
  --verifier reward \
  --repair-steps 1 \
  --hub-repo-id mistral-hackaton-2026/codeforge
```
