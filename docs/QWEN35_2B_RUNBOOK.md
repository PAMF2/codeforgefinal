# Qwen3.5-2B Runbook

## Goal

Use `Qwen/Qwen3.5-2B` as the main actor for CodeForge ASM with:

- manual warm-start (`phase1`)
- moderate GRPO refinement (`phase2`)
- verifier-first evaluation (`best-of-n + reward + repair`)

This setup is tuned for a Kaggle T4 budget and avoids heavy MCTS.

## Files

- `configs/grpo_config.qwen35_2b_phase1.yaml`
- `configs/grpo_config.qwen35_2b_phase2.yaml`
- `scripts/kaggle_qwen35_2b_pipeline.py`

## Kaggle Commands

```python
!git -C /kaggle/working/codeforgefinal pull || git clone https://github.com/PAMF2/codeforgefinal.git /kaggle/working/codeforgefinal
!python /kaggle/working/codeforgefinal/scripts/kaggle_qwen35_2b_pipeline.py --bootstrap-deps --phase1-hours 8 --phase2-hours 10
```

## Manual Training

Phase 1:

```bash
python scripts/kaggle_autorun.py \
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

Phase 2:

```bash
python scripts/kaggle_autorun.py \
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

## Evaluation

```bash
python assembly_swe/tools/eval_all_iters.py \
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

## RL Guidance

- Use more data and better verifier signals before using more RL.
- Keep RL moderate on T4:
  - `manual` warm-start first
  - `trl` refinement second
  - no MCTS in the main 30h run
- If you increase RL pressure too early, you usually improve `assembly_rate` faster than `correct_rate`.

