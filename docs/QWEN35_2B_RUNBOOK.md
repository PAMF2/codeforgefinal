# Qwen3.5-2B Runbook

## Goal

Run `Qwen/Qwen3.5-2B` on Kaggle with three layers:

1. synthetic task generation
2. verifier-first baseline
3. compiler-guided repair GRPO

The old one-shot warm-start pipeline is still in the repo, but the preferred path is now the agentic repair loop.

## Required files

- `configs/base.yaml`
- `configs/agentic_grpo.qwen35_2b.yaml`
- `scripts/bootstrap_kaggle.py`
- `scripts/generate_synthetic_tasks.py`
- `scripts/run_ranked_sampling.py`
- `scripts/run_agentic_grpo.py`
- `scripts/kaggle_agentic_qwen35_2b_pipeline.py`

## Kaggle bootstrap

```python
!git -C /kaggle/working/codeforgefinal pull || git clone https://github.com/PAMF2/codeforgefinal.git /kaggle/working/codeforgefinal
%cd /kaggle/working/codeforgefinal
!python scripts/bootstrap_kaggle.py
```

## One-command run

```python
!git -C /kaggle/working/codeforgefinal pull || git clone https://github.com/PAMF2/codeforgefinal.git /kaggle/working/codeforgefinal
%cd /kaggle/working/codeforgefinal
!python scripts/kaggle_agentic_qwen35_2b_pipeline.py --bootstrap-deps
```

This is the default safe preset for Kaggle now. It uses smaller dataset and rollout sizes, retries the agentic stage once with conservative settings if the first pass fails, and loads `HF_TOKEN` / `WANDB_API_KEY` from `Kaggle Secrets` automatically.

## Colab autoresearch helper

Run this to iterate quick experiments:

```python
%cd /kaggle/working/codeforgefinal
python experiments/autoresearch_adapter/colab_runner.py
```

Each iteration generates a tiny dataset, runs `run_ranked_sampling.py` with lightweight candidates, and stores metrics in `experiments/autoresearch_adapter/runs`.

## Dataset generation

```python
!python scripts/generate_synthetic_tasks.py \
  --out-dir data/generated \
  --core-size 5000 \
  --repair-size 2000 \
  --dev-size 500 \
  --hard-size 800 \
  --validate-sample 64
```

## Baseline signal

```python
!python scripts/run_ranked_sampling.py \
  --config configs/base.yaml \
  --tasks data/generated/dev.jsonl \
  --out artifacts/predictions.jsonl \
  --num-candidates 4 \
  --repair-steps 1
```

```python
!python scripts/eval.py \
  --config configs/base.yaml \
  --tasks data/generated/dev.jsonl \
  --predictions artifacts/predictions.jsonl \
  --ks 1,3,5
```

## Agentic GRPO

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

Expected outputs:

- `artifacts/agentic_grpo/metrics.jsonl`
- `artifacts/agentic_grpo/trajectories/iter_*.jsonl`
- `artifacts/agentic_grpo/sft/iter_*.jsonl`
- `checkpoints/agentic_grpo/iter_*`

## Old one-shot pipeline

Use this only if you want the original warm-start path:

```python
!python scripts/kaggle_qwen35_2b_pipeline.py --bootstrap-deps --phase1-hours 8 --phase2-hours 10
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

## What not to do first

- do not start with `8B`
- do not start with MCTS
- do not spend the first Kaggle session on pure token-level RL
- do not skip dataset generation and expect the repair loop to generalize
