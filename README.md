# codeforgefinal

Kaggle-first repo for verifier-driven assembly generation around `Qwen/Qwen3.5-2B`.

Main tracks:

1. `baseline`: sample, verify, rerank, repair, evaluate
2. `warm-start + GRPO`: old one-shot training path
3. `agentic repair GRPO`: compiler-guided repair loop with grouped candidates per step

## Repo layout

```text
codeforgefinal/
  assembly_swe/
  configs/
    agentic_grpo.qwen35_2b.yaml
    base.yaml
    grpo_config.yaml
    grpo_config.qwen35_2b_phase1.yaml
    grpo_config.qwen35_2b_phase2.yaml
  data/
  docs/
  notebooks/
    qwen35_asm_agentic_blueprint.py
  prompts/
  scripts/
    bootstrap_kaggle.py
    generate_synthetic_tasks.py
    run_ranked_sampling.py
    run_agentic_grpo.py
    kaggle_agentic_qwen35_2b_pipeline.py
    build_sft_dataset.py
    eval.py
    smoke_test.py
    kaggle_autorun.py
    kaggle_qwen35_2b_pipeline.py
  src/
    agentic.py
    best_of_n.py
    data.py
    env.py
    modeling.py
    reward.py
    trainer.py
    utils.py
    verifier.py
  train.py
```

## Kaggle secrets

Add in Kaggle Secrets:

- `HF_TOKEN`
- `WANDB_API_KEY`

The Kaggle pipeline now loads these automatically at startup.

## Bootstrap

```python
!git -C /kaggle/working/codeforgefinal pull || git clone https://github.com/PAMF2/codeforgefinal.git /kaggle/working/codeforgefinal
%cd /kaggle/working/codeforgefinal
!python scripts/bootstrap_kaggle.py
```

## One Command

Safe Kaggle preset:

```python
!git -C /kaggle/working/codeforgefinal pull || git clone https://github.com/PAMF2/codeforgefinal.git /kaggle/working/codeforgefinal
%cd /kaggle/working/codeforgefinal
!python scripts/kaggle_agentic_qwen35_2b_pipeline.py --bootstrap-deps
```

This now defaults to a conservative first run:

- `core_size=1200`
- `repair_size=400`
- `dev_size=120`
- `hard_size=120`
- `iterations=2`
- `prompts_per_iteration=3`
- `num_candidates=2`
- `repair_steps=1`
- `max_episode_steps=2`

If the first agentic pass fails once, the pipeline retries automatically with the same safe regime.

## Legacy adapter

`experiments/autoresearch_adapter/` is legacy exploratory material and is not the main GRPO
autoresearch path. The active path is `scripts/run_autoresearch.py` plus
`experiments/autoresearch_grpo/target_config.yaml`.

## Smoke test

```python
!python scripts/smoke_test.py --config configs/base.yaml
```

## Generate synthetic data

This is the first thing to run before longer training.

```python
!python scripts/generate_synthetic_tasks.py \
  --out-dir data/generated \
  --core-size 5000 \
  --repair-size 2000 \
  --dev-size 500 \
  --hard-size 800 \
  --validate-sample 64
```

Outputs:

- `data/generated/train.jsonl`
- `data/generated/dev.jsonl`
- `data/generated/hard.jsonl`
- `data/generated/private_eval.jsonl`
- `data/generated/generation_only.jsonl`
- `data/generated/repair_only.jsonl`
- `data/generated/manifest.json`

## Fast baseline

```python
!python scripts/run_ranked_sampling.py \
  --config configs/base.yaml \
  --tasks data/generated/dev.jsonl \
  --out artifacts/predictions.jsonl \
  --num-candidates 4 \
  --repair-steps 1
```

```python
!python scripts/build_sft_dataset.py \
  --tasks data/generated/dev.jsonl \
  --predictions artifacts/predictions.jsonl \
  --out artifacts/sft_pairs.jsonl
```

```python
!python scripts/eval.py \
  --config configs/base.yaml \
  --tasks data/generated/dev.jsonl \
  --predictions artifacts/predictions.jsonl \
  --ks 1,3,5
```

## Agentic repair GRPO

This is the new Kaggle path for compiler-guided repair RL.

One command:

```python
!python scripts/kaggle_agentic_qwen35_2b_pipeline.py --bootstrap-deps --iterations 12
```

```python
!python scripts/run_agentic_grpo.py \
  --config configs/agentic_grpo.qwen35_2b.yaml \
  --tasks data/generated/train.jsonl \
  --iterations 12 \
  --prompts-per-iteration 6 \
  --num-candidates 4 \
  --repair-steps 2 \
  --max-episode-steps 3
```

Artifacts written under:

- `artifacts/agentic_grpo/metrics.jsonl`
- `artifacts/agentic_grpo/trajectories/`
- `artifacts/agentic_grpo/sft/`
- `checkpoints/agentic_grpo/`

### Autoresearch GRPO

Passing `--run-autoresearch` now makes autoresearch the main GRPO stage. Instead of one direct
`run_agentic_grpo.py` call, the pipeline launches `scripts/run_autoresearch.py`, which:

1. runs a baseline GRPO experiment with the current config
2. mutates the live GRPO config
3. re-runs the real GRPO experiment
4. marks it `keep`, `discard`, or `crash`
5. advances only if the metrics improved

Outputs are written to `artifacts/autoresearch/grpo/<run_tag>/`, including:

- `results.tsv`
- `runs.json`
- `best_config.yaml`
- `best_artifacts/`
- per-experiment `run.log`

`--autoresearch-time-budget` is the per-experiment timeout in minutes.

## One-shot warm-start + GRPO pipeline

This preserves the older path from the previous repo.

```python
!python scripts/kaggle_qwen35_2b_pipeline.py --bootstrap-deps --phase1-hours 8 --phase2-hours 10
```

Manual phase 1:

```python
!python scripts/kaggle_autorun.py \
  --root /kaggle/working/codeforgefinal \
  --config configs/grpo_config.qwen35_2b_phase1.yaml \
  --hours 8 \
  --backend manual \
  --safe-profile \
  --batch-size 1 \
  --generations-per-prompt 6 \
  --prompts-per-iteration 8 \
  --gradient-accumulation-steps 6 \
  --max-new-tokens 128 \
  --use-mcts-after-iteration 999
```

Manual phase 2:

```python
!python scripts/kaggle_autorun.py \
  --root /kaggle/working/codeforgefinal \
  --config configs/grpo_config.qwen35_2b_phase2.yaml \
  --hours 10 \
  --backend trl \
  --safe-profile \
  --batch-size 1 \
  --generations-per-prompt 4 \
  --prompts-per-iteration 6 \
  --gradient-accumulation-steps 8 \
  --max-new-tokens 128 \
  --use-mcts-after-iteration 999
```

## Benchmark

```python
!python assembly_swe/tools/eval_all_iters.py \
  --repo-root . \
  --tasks assembly_swe/datasets/dev_v1_30.jsonl \
  --iter-start 1 \
  --iter-end 30 \
  --ks 1,3,5 \
  --outdir assembly_swe/results/qwen35_2b_eval \
  --load-in-4bit \
  --hub-repo-id PAMF2/codeforgefinal-qwen35-2b \
  --base-model Qwen/Qwen3.5-2B \
  --max-new-tokens 128 \
  --temperature 0.20 \
  --top-p 0.80 \
  --top-k 20 \
  --repetition-penalty 1.05 \
  --num-candidates 5 \
  --verifier reward \
  --verifier-timeout-sec 6 \
  --repair-steps 1
```

## First run on Kaggle

If you want the shortest path that actually tests the new stack:

1. `python scripts/kaggle_agentic_qwen35_2b_pipeline.py --bootstrap-deps`
2. if that works, scale up the explicit flags gradually

Then scale iterations and dataset size once the small run completes.
