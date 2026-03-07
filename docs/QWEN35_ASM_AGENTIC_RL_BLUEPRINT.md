# Qwen3.5-2B Assembly Agentic RL Blueprint

## Goal

Build a new Kaggle notebook from scratch around `Qwen/Qwen3.5-2B` that is:

- verifier-first
- repair-centric
- RL-ready
- small-compute aware

This blueprint is intentionally not a copy of the current repo pipeline. It is a cleaner architecture for a notebook-first experiment.

## Why this architecture

The strongest signals from the current literature and official docs point in the same direction:

1. Qwen3.5-2B is explicitly positioned for prototyping and task-specific fine-tuning, not as a frontier base model replacement.
2. TRL's `GRPOTrainer` now supports custom `reward_funcs`, `rollout_func`, and `environment_factory`, which is exactly what we need for an assembly environment.
3. OpenEnv formalizes a simple environment interface around `step()`, `reset()`, and `close()`.
4. Open-R1 uses executable code rewards and dataset filtering by pass rate on verifiable tasks.
5. PanGu-Coder2 shows that ranking generated responses with unit tests is a strong low-compute training signal.
6. VRpilot shows that reasoning plus patch-validation feedback materially improves repair quality.

The implication is straightforward:

- do not treat assembly as plain text generation only
- do not spend the whole budget on pure token-level RL
- use an environment with compile/run feedback and explicit repair traces

## Architecture

### 1. Three loops, not one

Use three training loops in the notebook.

#### Loop A: Ranked sampling warm start

- Prompt -> sample `k` candidates.
- Verify all candidates with `nasm + ld + run + hidden tests`.
- Keep:
  - best correct candidate
  - best repairing trace
  - top negative examples with informative failure
- Distill these into SFT-style data.

Why:
- This follows the ranking-feedback direction from PanGu-Coder2.
- It is cheaper and more stable than jumping straight into full RL.

#### Loop B: Agentic GRPO

Train the model in an assembly environment where the model can:

- write a full draft
- ask for verifier output
- patch the program
- submit final answer

Why:
- TRL now supports environment-driven rollouts.
- Assembly is a rare domain where the environment gives objective, cheap rewards.

#### Loop C: Distill the winners back into a fast one-shot policy

Every N GRPO steps:

- collect solved trajectories
- convert them into compact `(instruction -> final assembly)` pairs
- optionally also create `(instruction + verifier feedback + failed asm -> repaired asm)` pairs
- run a short SFT refresh

Why:
- RL agents get strong but slow.
- Distillation recovers inference speed and stabilizes gains.

### 2. Environment design

Create a local environment class in the notebook, then optionally wrap it behind OpenEnv later.

#### State

State should include:

- natural-language task
- current assembly draft
- compile status
- linker status
- runtime status
- public test summary
- hidden test score
- last stderr/stdout
- step count

#### Actions

Do not give the model an unrestricted forever-loop.

Use these action types:

- `draft_full(program)`
- `patch_block(program)`
- `request_verifier()`
- `submit(program)`

If you want a more structured version later, split `patch_block` into:

- `replace_lines(start, end, block)`
- `append_block(block)`
- `rewrite_full(program)`

#### Episode budget

Hard-cap each episode.

Recommended:

- max steps: `3` or `4`
- max program length: `160-220` tokens for early runs
- hard timeout per verification: `4-8s`

This matters because small models can get trapped in repetitive repair loops.

### 3. Reward design

Use additive rewards, but make correctness dominant.

Recommended default:

- `0.10` assemble
- `0.05` link
- `0.05` run
- `0.60` hidden-test pass fraction
- `0.10` repair improvement bonus
- `0.10` exact/public-test and cleanliness bonus

Penalties:

- `-0.05` prose or markdown fences
- `-0.05` repeated invalid retries
- `-0.05` known bad ABI patterns like `int 0x80` for x86-64 Linux tasks
- `-0.10` if the repair step makes the score worse after a previously better state

The key innovation here is `repair improvement bonus`:

- reward delta from previous verifier state
- not just absolute correctness

That makes the model learn to repair, not only to guess perfect code from scratch.

### 4. Data mixture

The notebook should not rely on one tiny dataset.

Use a mixture with explicit curriculum:

#### Tier 1

- exit codes
- fixed stdout
- basic syscalls

#### Tier 2

- arithmetic
- branches
- short loops

#### Tier 3

- memory access
- arrays
- decimal conversion
- string traversal

#### Tier 4

- bug-fix tasks
- constrained optimization tasks
- multi-test tasks
- partial programs that must be repaired

Add a second dataset family that matters more than people think:

- `failure -> repair` examples

These should come from your own verifier logs.

### 5. Notebook-first innovation

The most promising novel angle is not "more RL" by itself.

It is this:

## Compiler-Guided Repair RL

A small policy learns in an environment where progress is measured by verifier deltas, not just terminal success.

That means the model is rewarded for:

- going from not assembling -> assembling
- assembling -> linking
- linking -> running
- running -> passing more tests
- bad draft -> better repair

This is stronger than normal one-shot GRPO because it turns sparse terminal reward into a short-horizon repair game.

That is the architecture I would call genuinely interesting.

## Recommended notebook order

### Cell 1

Define experiment config.

- model id
- paths
- timeouts
- max episode steps
- number of candidates
- SFT and GRPO toggles

### Cell 2

Install deps.

Use versions compatible with Qwen3.5 and TRL environment rollouts.

### Cell 3

Load tokenizer/model.

Start with `Qwen/Qwen3.5-2B`.

### Cell 4

Load task mixture.

Keep both:

- generation tasks
- repair tasks

### Cell 5

Implement verifier.

- write file
- assemble
- link
- run
- compare tests
- return detailed diagnostics

### Cell 6

Implement `AsmForgeEnv`.

Methods:

- `reset()`
- `step(action)`
- `close()`

### Cell 7

Run ranked sampling baseline.

- best-of-k
- reward rerank
- optional one repair attempt

This becomes your first baseline.

### Cell 8

Build SFT dataset from winners.

- direct solves
- repair traces

### Cell 9

Run short SFT.

### Cell 10

Wire GRPO.

Use:

- custom `reward_funcs`
- custom `rollout_func`
- optionally `environment_factory`

### Cell 11

Evaluate.

Track at least:

- `compile@1`
- `run@1`
- `correct@1`
- `pass@5`
- `repair_gain`
- `avg_steps_to_success`

### Cell 12

Distill solved trajectories and rerun eval.

## Hyperparameter guidance

### Warm start / ranked sampling

- model: `Qwen/Qwen3.5-2B`
- load in 4-bit
- LoRA rank: `32`
- lr: `1e-5`
- seq len: `1024`
- candidates per task: `4-8`
- temperature: `0.6`
- top_p: `0.95`
- top_k: `20`

These are close to Qwen's own precise-coding guidance for thinking mode, but I am adapting them for short assembly rollouts.

### GRPO stage

Start conservative.

- generations per prompt: `4`
- per-device batch: `1`
- grad accumulation: `8`
- lr: `3e-6` to `5e-6`
- max completion length: `192`
- beta / KL: low-to-moderate
- max episode steps: `3`

Do not begin with long contexts or large rollouts.

## What I would not do

- No MCTS in the first notebook.
- No 8B model first.
- No pure token-only RL with no repair loop.
- No reward model learned from preferences this early.
- No giant hidden dataset before proving the environment works.

## Success criteria

The notebook is good if it proves these three things:

1. `best-of-k + verifier + repair` beats plain one-shot generation.
2. short-horizon agentic GRPO beats ranked sampling alone.
3. distilling solved trajectories recovers most of the gain in one-shot inference.

If you prove those three, then you have something real.

## Sources

- Qwen3.5-2B model card: https://huggingface.co/Qwen/Qwen3.5-2B
- Qwen3.5 Transformers docs: https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5
- TRL GRPO docs: https://huggingface.co/docs/trl/en/grpo_trainer
- TRL OpenEnv docs: https://huggingface.co/docs/trl/openenv
- OpenEnv intro blog: https://huggingface.co/blog/openenv
- Open-R1 repo: https://github.com/huggingface/open-r1
- PanGu-Coder2 ranking feedback: https://arxiv.org/pdf/2307.14936
- Automatic Unit Test Data Generation and Actor-Critic RL for Code Synthesis: https://aclanthology.org/2023.findings-emnlp.28/
- VRpilot / reasoning + patch validation feedback: https://arxiv.org/pdf/2405.15690
