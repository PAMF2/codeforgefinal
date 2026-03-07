from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Mapping

from .prompt_engine import PromptItem
from .utils import ensure_dir, run_cmd


@dataclass
class RewardResult:
    reward: float
    assembled: bool
    linked: bool
    ran: bool
    correct: bool
    stdout: str
    stderr: str
    exit_code: int | None
    stage_failed: str | None


@dataclass(frozen=True)
class RewardWeights:
    assemble: float = 0.25
    link: float = 0.25
    run: float = 0.20
    correctness: float = 0.30

    @classmethod
    def from_mapping(cls, raw: Mapping[str, float] | None) -> "RewardWeights":
        if raw is None:
            return cls()
        return cls(
            assemble=float(raw.get("assemble", cls.assemble)),
            link=float(raw.get("link", cls.link)),
            run=float(raw.get("run", cls.run)),
            correctness=float(raw.get("correctness", cls.correctness)),
        )


_ASM_MNEMONICS = frozenset({
    "mov", "push", "pop", "jmp", "je", "jne", "jl", "jg", "jle", "jge",
    "jz", "jnz", "call", "ret", "xor", "add", "sub", "cmp", "inc", "dec",
    "lea", "imul", "idiv", "mul", "div", "and", "or", "not", "shl", "shr",
    "test", "nop", "neg", "cbw", "cdq", "cqo", "movzx", "movsx", "cmov",
    "cmovl", "cmovg", "cmovle", "cmovge", "cmove", "cmovne",
})


def _structural_score(asm_code: str) -> float:
    """
    Partial reward for structurally-correct NASM even when assembly fails.

    Max score is 0.12 in the default reward scale.
    """
    if not asm_code.strip():
        return 0.0

    low = asm_code.lower()
    score = 0.0

    if "global _start" in low:
        score += 0.025
    if "section .text" in low:
        score += 0.025
    if "_start:" in low:
        score += 0.025

    has_syscall = "syscall" in low
    has_int80 = "int 0x80" in low or "int 80h" in low
    if has_syscall and not has_int80:
        score += 0.020
    if has_int80:
        score -= 0.020

    has_exit60 = any(
        part in low
        for part in (
            "mov rax, 60",
            "mov eax, 60",
            "mov rax,60",
            "mov eax,60",
            "mov rax, 0x3c",
            "mov eax, 0x3c",
        )
    )
    if has_exit60:
        score += 0.015

    found = sum(1 for mnemonic in _ASM_MNEMONICS if mnemonic in low)
    if found >= 2:
        score += 0.010
    if found >= 5:
        score += 0.010
    if found == 0:
        score = 0.0

    return max(0.0, min(score, 0.12))


def _stdout_partial(got: str, expected: str, exact_reward: float) -> float:
    if not expected:
        return 0.0
    if got == expected:
        return exact_reward
    sim = SequenceMatcher(None, got, expected).ratio()
    if sim >= 0.95:
        return exact_reward * (2.0 / 3.0)
    return 0.0


class RewardPipeline:
    def __init__(
        self,
        artifacts_dir: str | Path,
        timeout_seconds: int = 5,
        stage_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.timeout_seconds = timeout_seconds
        self.weights = RewardWeights.from_mapping(stage_weights)
        ensure_dir(self.artifacts_dir)

    def _assemble_partial_reward(self, asm_code: str) -> float:
        if self.weights.assemble <= 0:
            return 0.0
        scaled = _structural_score(asm_code) * (self.weights.assemble / 0.25)
        return max(0.0, min(scaled, self.weights.assemble))

    def evaluate(self, prompt: PromptItem, asm_code: str, sample_id: str) -> RewardResult:
        workdir = self.artifacts_dir / sample_id
        ensure_dir(workdir)

        asm_path = workdir / "prog.asm"
        obj_path = workdir / "prog.o"
        bin_path = workdir / "prog"
        asm_path.write_text(asm_code + "\n", encoding="utf-8")

        assemble = run_cmd(
            ["nasm", "-f", "elf64", str(asm_path), "-o", str(obj_path)],
            self.timeout_seconds,
        )
        if assemble.returncode != 0:
            return RewardResult(
                reward=self._assemble_partial_reward(asm_code),
                assembled=False,
                linked=False,
                ran=False,
                correct=False,
                stdout=assemble.stdout,
                stderr=assemble.stderr,
                exit_code=None,
                stage_failed="assemble",
            )

        reward = self.weights.assemble

        link = run_cmd(["ld", str(obj_path), "-o", str(bin_path)], self.timeout_seconds)
        if link.returncode != 0:
            return RewardResult(
                reward=reward,
                assembled=True,
                linked=False,
                ran=False,
                correct=False,
                stdout=link.stdout,
                stderr=link.stderr,
                exit_code=None,
                stage_failed="link",
            )
        reward += self.weights.link

        try:
            run = run_cmd([str(bin_path)], self.timeout_seconds)
        except subprocess.TimeoutExpired:
            return RewardResult(
                reward=reward,
                assembled=True,
                linked=True,
                ran=False,
                correct=False,
                stdout="",
                stderr=f"Command timed out after {self.timeout_seconds} seconds",
                exit_code=124,
                stage_failed="run",
            )
        if run.returncode in (124, 126, 127) or run.returncode < 0:
            return RewardResult(
                reward=reward,
                assembled=True,
                linked=True,
                ran=False,
                correct=False,
                stdout=run.stdout,
                stderr=run.stderr,
                exit_code=run.returncode,
                stage_failed="run",
            )
        reward += self.weights.run

        correct = False
        correctness_bonus = 0.0

        if prompt.expected_stdout is not None:
            if run.stdout == prompt.expected_stdout:
                correct = True
                correctness_bonus = self.weights.correctness
            else:
                correctness_bonus = _stdout_partial(
                    run.stdout,
                    prompt.expected_stdout,
                    self.weights.correctness,
                )
        elif prompt.expected_exit_code is not None:
            if run.returncode == prompt.expected_exit_code:
                correct = True
                correctness_bonus = self.weights.correctness
        else:
            if run.returncode == 0:
                correct = True
                correctness_bonus = self.weights.correctness

        if (
            not correct
            and prompt.expected_stdout is None
            and prompt.expected_exit_code is None
            and run.returncode == 0
        ):
            correctness_bonus = self.weights.correctness * 0.5

        reward += correctness_bonus

        return RewardResult(
            reward=reward,
            assembled=True,
            linked=True,
            ran=True,
            correct=correct,
            stdout=run.stdout,
            stderr=run.stderr,
            exit_code=run.returncode,
            stage_failed=None if correct else "correctness",
        )

    def evaluate_batch(
        self,
        items: list[tuple[PromptItem, str, str]],
        workers: int = 32,
    ) -> list[RewardResult]:
        if not items:
            return []

        zero = RewardResult(
            reward=0.0,
            assembled=False,
            linked=False,
            ran=False,
            correct=False,
            stdout="",
            stderr="evaluation_error",
            exit_code=None,
            stage_failed="error",
        )

        n_workers = min(workers, len(items))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(self.evaluate, prompt, asm, sample_id) for prompt, asm, sample_id in items]
            results: list[RewardResult] = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception:
                    results.append(zero)
            return results
