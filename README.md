# codeforgefinal

Kaggle-first repo for assembly generation and training with:

- `Qwen3.5-2B`
- objective verification (`nasm + ld + run`)
- ranked sampling + repair
- `assembly_swe` benchmark
- `GRPO` training pipeline
- notebook blueprint for compiler-guided repair RL

This repo now has two tracks:

1. fast baseline: generate, verify, rerank, repair, evaluate
2. full training: warm-start + GRPO + checkpoint benchmark on Kaggle

## Structure

```text
codeforgefinal/
  assembly_swe/
  configs/
    base.yaml
    grpo_config.yaml
    grpo_config.qwen35_2b_phase1.yaml
    grpo_config.qwen35_2b_phase2.yaml
  data/
  docs/
  notebooks/
  prompts/
  scripts/
    bootstrap_kaggle.py
    smoke_test.py
    run_ranked_sampling.py
    build_sft_dataset.py
    eval.py
    kaggle_autorun.py
    kaggle_qwen35_2b_pipeline.py
  src/
    best_of_n.py
    data.py
    env.py
    mcts.py
    modeling.py
    prompt_engine.py
    reward.py
    trainer.py
    utils.py
    verifier.py
  train.py
```

## Kaggle Secrets

Add these in Kaggle Secrets:

- `HF_TOKEN`
- `WANDB_API_KEY`

## Quick Baseline

Clone:

```python
!git clone https://github.com/PAMF2/codeforgefinal.git /kaggle/working/codeforgefinal
%cd /kaggle/working/codeforgefinal
```

Bootstrap:

```python
!python scripts/bootstrap_kaggle.py
```

Smoke test:

```python
!python scripts/smoke_test.py --config configs/base.yaml
```

Generate + rerank + repair:

```python
!python scripts/run_ranked_sampling.py \
  --config configs/base.yaml \
  --tasks data/sample_tasks.jsonl \
  --out artifacts/predictions.jsonl \
  --num-candidates 4 \
  --repair-steps 1
```

Build SFT pairs:

```python
!python scripts/build_sft_dataset.py \
  --tasks data/sample_tasks.jsonl \
  --predictions artifacts/predictions.jsonl \
  --out artifacts/sft_pairs.jsonl
```

Evaluate:

```python
!python scripts/eval.py \
  --config configs/base.yaml \
  --tasks data/sample_tasks.jsonl \
  --predictions artifacts/predictions.jsonl \
  --ks 1,3,5
```

## Full Kaggle Training Pipeline

One command for warm-start + GRPO + benchmark:

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

Checkpoint benchmark on `assembly_swe`:

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

## Main Files

- fast baseline reward/verifier: `src/verifier.py`
- GRPO reward pipeline: `src/reward.py`
- trainer: `src/trainer.py`
- benchmark protocol: `assembly_swe/`
- Kaggle runner: `scripts/kaggle_qwen35_2b_pipeline.py`
- notebook blueprint: `notebooks/qwen35_asm_agentic_blueprint.py`
- research blueprint: `docs/QWEN35_ASM_AGENTIC_RL_BLUEPRINT.md`

## What To Run First

If you want the shortest path to signal on Kaggle:

1. `python scripts/bootstrap_kaggle.py`
2. `python scripts/smoke_test.py --config configs/base.yaml`
3. `python scripts/kaggle_qwen35_2b_pipeline.py --bootstrap-deps --phase1-hours 2 --phase2-hours 2 --bench-last-iters 3`

Then scale up once the small run works.
