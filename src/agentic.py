from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .data import Task
from .utils import SYS_PROMPT, sanitize_model_output
from .verifier import ObjectiveVerifier, VerifyResult


GeneratorFn = Callable[..., list[str]]


@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float = 1.0


@dataclass
class CandidateRecord:
    prompt_id: str
    prompt_text: str
    task_id: str
    instruction: str
    step_idx: int
    candidate_idx: int
    action_type: str
    asm: str
    base_reward: float
    reward: float
    assembled: bool
    linked: bool
    ran: bool
    correct: bool
    public_pass_rate: float
    hidden_pass_rate: float
    stage_failed: str | None
    stdout: str
    stderr: str
    penalties: list[str] = field(default_factory=list)

    @classmethod
    def from_verify_result(
        cls,
        *,
        prompt_id: str,
        prompt_text: str,
        task: Task,
        step_idx: int,
        candidate_idx: int,
        action_type: str,
        asm: str,
        result: VerifyResult,
    ) -> "CandidateRecord":
        return cls(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            task_id=task.task_id,
            instruction=task.instruction,
            step_idx=step_idx,
            candidate_idx=candidate_idx,
            action_type=action_type,
            asm=asm,
            base_reward=float(result.reward),
            reward=float(result.reward),
            assembled=bool(result.assembled),
            linked=bool(result.linked),
            ran=bool(result.ran),
            correct=bool(result.correct),
            public_pass_rate=float(result.public_pass_rate),
            hidden_pass_rate=float(result.hidden_pass_rate),
            stage_failed=result.stage_failed,
            stdout=result.stdout,
            stderr=result.stderr,
            penalties=list(result.penalties),
        )

    def to_train_row(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "task_id": self.task_id,
            "instruction": self.instruction,
            "step_idx": self.step_idx,
            "candidate_idx": self.candidate_idx,
            "action_type": self.action_type,
            "asm": self.asm,
            "base_reward": self.base_reward,
            "reward": self.reward,
            "correct": self.correct,
            "assembled": self.assembled,
            "linked": self.linked,
            "ran": self.ran,
            "public_pass_rate": self.public_pass_rate,
            "hidden_pass_rate": self.hidden_pass_rate,
            "stage_failed": self.stage_failed,
            "penalties": self.penalties,
        }

    def to_json(self) -> dict[str, Any]:
        payload = self.to_train_row()
        payload["stdout"] = self.stdout
        payload["stderr"] = self.stderr
        return payload


@dataclass
class StepRecord:
    prompt_id: str
    prompt_text: str
    step_idx: int
    action_type: str
    candidates: list[CandidateRecord]
    best_idx: int
    kept_improved: bool

    @property
    def best(self) -> CandidateRecord:
        return self.candidates[self.best_idx]


@dataclass
class EpisodeRecord:
    task_id: str
    instruction: str
    steps: list[StepRecord] = field(default_factory=list)
    final_best: CandidateRecord | None = None

    def to_summary(self) -> dict[str, Any]:
        initial_best = self.steps[0].best if self.steps else None
        final_best = self.final_best
        initial_reward = 0.0 if initial_best is None else initial_best.reward
        final_reward = 0.0 if final_best is None else final_best.reward
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "steps": len(self.steps),
            "solved": False if final_best is None else final_best.correct,
            "initial_reward": initial_reward,
            "final_reward": final_reward,
            "repair_gain": final_reward - initial_reward,
            "final_stage_failed": None if final_best is None else final_best.stage_failed,
            "assembled": False if final_best is None else final_best.assembled,
            "linked": False if final_best is None else final_best.linked,
            "ran": False if final_best is None else final_best.ran,
        }


def build_draft_prompt(task: Task) -> str:
    if task.task_kind == "repair" and task.starter_code:
        return (
            f"{SYS_PROMPT}\n\n"
            "You are fixing an existing NASM x86-64 Linux program.\n"
            f"Task: {task.instruction}\n\n"
            "Buggy program:\n"
            f"{task.starter_code}\n\n"
            "Return only the corrected full program."
        )
    return f"{SYS_PROMPT}\n\nTask: {task.instruction}\nReturn only assembly code."


def build_repair_prompt(task: Task, previous: CandidateRecord, step_idx: int) -> str:
    stage = previous.stage_failed or "correctness"
    stderr = previous.stderr.strip()
    stdout = previous.stdout.strip()
    if len(stderr) > 400:
        stderr = stderr[:400]
    if len(stdout) > 200:
        stdout = stdout[:200]
    return (
        f"{SYS_PROMPT}\n\n"
        f"Task: {task.instruction}\n\n"
        f"Repair round: {step_idx}\n"
        f"Failure stage: {stage}\n"
        f"Previous reward: {previous.reward:.4f}\n"
        f"Hidden pass rate: {previous.hidden_pass_rate:.4f}\n"
        f"Last stdout: {stdout}\n"
        f"Last stderr: {stderr}\n\n"
        "Rewrite the full program and improve it.\n"
        "Return only the corrected full program.\n\n"
        "Previous candidate:\n"
        f"{previous.asm}"
    )


def _generate_samples(
    generator: GeneratorFn,
    prompt_text: str,
    num_candidates: int,
    sampling: SamplingConfig,
) -> list[str]:
    outputs = generator(
        prompt=prompt_text,
        n=num_candidates,
        max_new_tokens=sampling.max_new_tokens,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=sampling.top_k,
        min_p=sampling.min_p,
        repetition_penalty=sampling.repetition_penalty,
    )
    return [sanitize_model_output(str(text)) for text in outputs]


def score_prompt_group(
    *,
    task: Task,
    verifier: ObjectiveVerifier,
    generator: GeneratorFn,
    prompt_id: str,
    prompt_text: str,
    step_idx: int,
    action_type: str,
    num_candidates: int,
    sampling: SamplingConfig,
) -> StepRecord:
    candidates = _generate_samples(generator, prompt_text, num_candidates, sampling)
    rows: list[CandidateRecord] = []
    for cand_idx, asm in enumerate(candidates):
        result = verifier.evaluate(
            task,
            asm,
            sample_id=f"{task.task_id}-s{step_idx}-c{cand_idx}",
        )
        rows.append(
            CandidateRecord.from_verify_result(
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                task=task,
                step_idx=step_idx,
                candidate_idx=cand_idx,
                action_type=action_type,
                asm=asm,
                result=result,
            )
        )

    best_idx = max(
        range(len(rows)),
        key=lambda idx: (
            rows[idx].reward,
            rows[idx].correct,
            rows[idx].hidden_pass_rate,
            rows[idx].assembled,
        ),
    )
    return StepRecord(
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        step_idx=step_idx,
        action_type=action_type,
        candidates=rows,
        best_idx=best_idx,
        kept_improved=True,
    )


def _recompute_best_idx(rows: list[CandidateRecord]) -> int:
    return max(
        range(len(rows)),
        key=lambda idx: (
            rows[idx].reward,
            rows[idx].correct,
            rows[idx].hidden_pass_rate,
            rows[idx].assembled,
        ),
    )


def run_repair_episode(
    *,
    task: Task,
    verifier: ObjectiveVerifier,
    generator: GeneratorFn,
    sampling: SamplingConfig,
    num_candidates: int,
    repair_steps: int,
    max_episode_steps: int,
    repair_gain_weight: float = 0.10,
    regress_penalty: float = -0.10,
) -> EpisodeRecord:
    episode = EpisodeRecord(task_id=task.task_id, instruction=task.instruction)
    best_so_far: CandidateRecord | None = None
    total_steps = max(1, min(max_episode_steps, 1 + max(0, repair_steps)))

    for step_idx in range(total_steps):
        if step_idx == 0:
            prompt_text = build_draft_prompt(task)
            action_type = "draft_full"
        else:
            if best_so_far is None:
                break
            prompt_text = build_repair_prompt(task, best_so_far, step_idx)
            action_type = "patch_block"

        prompt_id = f"{task.task_id}:step{step_idx}"
        step = score_prompt_group(
            task=task,
            verifier=verifier,
            generator=generator,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            step_idx=step_idx,
            action_type=action_type,
            num_candidates=num_candidates,
            sampling=sampling,
        )

        if best_so_far is not None:
            for candidate in step.candidates:
                delta = candidate.base_reward - best_so_far.base_reward
                if delta > 0.0:
                    candidate.reward += repair_gain_weight * delta
                elif delta < 0.0:
                    candidate.reward += regress_penalty
            step.best_idx = _recompute_best_idx(step.candidates)

        group_best = step.best
        improved = best_so_far is None or group_best.reward >= best_so_far.reward
        step.kept_improved = improved
        if improved:
            best_so_far = group_best
        episode.steps.append(step)

        if best_so_far is not None and best_so_far.correct:
            break

    episode.final_best = best_so_far
    return episode


def flatten_episode_rows(
    episodes: list[EpisodeRecord],
    *,
    include_all_candidates: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        for step in episode.steps:
            candidates = step.candidates if include_all_candidates else [step.best]
            for candidate in candidates:
                rows.append(candidate.to_train_row())
    return rows


def flatten_episode_candidates(episodes: list[EpisodeRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        for step in episode.steps:
            for candidate in step.candidates:
                rows.append(candidate.to_json())
    return rows


def summarize_episodes(episodes: list[EpisodeRecord]) -> dict[str, float]:
    total = max(1, len(episodes))
    finals = [ep.final_best for ep in episodes]
    initials = [ep.steps[0].best for ep in episodes if ep.steps]
    return {
        "tasks": float(len(episodes)),
        "solved_rate": sum(1 for row in finals if row is not None and row.correct) / total,
        "assemble_rate": sum(1 for row in finals if row is not None and row.assembled) / total,
        "link_rate": sum(1 for row in finals if row is not None and row.linked) / total,
        "run_rate": sum(1 for row in finals if row is not None and row.ran) / total,
        "avg_final_reward": sum(0.0 if row is None else row.reward for row in finals) / total,
        "avg_initial_reward": sum(row.reward for row in initials) / max(1, len(initials)),
        "avg_repair_gain": sum(
            0.0
            if ep.final_best is None or not ep.steps
            else ep.final_best.reward - ep.steps[0].best.reward
            for ep in episodes
        ) / total,
        "avg_steps": sum(len(ep.steps) for ep in episodes) / total,
    }
