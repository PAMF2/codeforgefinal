from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .data import Task
from .verifier import ObjectiveVerifier, VerifyResult


@dataclass
class EnvState:
    task: Task
    draft: str = ""
    last_result: VerifyResult | None = None
    step_idx: int = 0
    done: bool = False


class AsmForgeEnv:
    def __init__(self, task: Task, verifier: ObjectiveVerifier, max_episode_steps: int = 3) -> None:
        self.task = task
        self.verifier = verifier
        self.max_episode_steps = max_episode_steps
        self.state = EnvState(task=task)

    def reset(self) -> dict[str, Any]:
        self.state = EnvState(task=self.task)
        return self.observe()

    def observe(self) -> dict[str, Any]:
        last = self.state.last_result
        return {
            "task_id": self.task.task_id,
            "instruction": self.task.instruction,
            "draft": self.state.draft,
            "step_idx": self.state.step_idx,
            "last_reward": 0.0 if last is None else last.reward,
            "last_stage_failed": None if last is None else last.stage_failed,
            "done": self.state.done,
        }

    def step(self, action_type: str, program: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.state.done:
            return self.observe(), 0.0, True, {"reason": "already_done"}

        previous_reward = 0.0 if self.state.last_result is None else self.state.last_result.reward
        self.state.step_idx += 1
        self.state.draft = program
        result = self.verifier.evaluate(self.task, program, sample_id=f"{self.task.task_id}-s{self.state.step_idx}")
        self.state.last_result = result

        reward = result.reward
        if self.state.step_idx > 1 and reward < previous_reward:
            reward += self.verifier.weights.regress_penalty

        if action_type == "submit" or result.correct or self.state.step_idx >= self.max_episode_steps:
            self.state.done = True

        return self.observe(), reward, self.state.done, {
            "assembled": result.assembled,
            "linked": result.linked,
            "ran": result.ran,
            "correct": result.correct,
        }
