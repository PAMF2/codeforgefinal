from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .data import Task
from .utils import ensure_dir, run_cmd


@dataclass(frozen=True)
class RewardWeights:
    assemble: float = 0.10
    link: float = 0.05
    run: float = 0.05
    public_test: float = 0.10
    hidden_test: float = 0.60
    cleanliness: float = 0.10
    prose_penalty: float = -0.05
    bad_abi_penalty: float = -0.05
    regress_penalty: float = -0.10

    @classmethod
    def from_mapping(cls, raw: Mapping[str, float] | None) -> "RewardWeights":
        if raw is None:
            return cls()
        return cls(
            assemble=float(raw.get("assemble", cls.assemble)),
            link=float(raw.get("link", cls.link)),
            run=float(raw.get("run", cls.run)),
            public_test=float(raw.get("public_test", cls.public_test)),
            hidden_test=float(raw.get("hidden_test", cls.hidden_test)),
            cleanliness=float(raw.get("cleanliness", cls.cleanliness)),
            prose_penalty=float(raw.get("prose_penalty", cls.prose_penalty)),
            bad_abi_penalty=float(raw.get("bad_abi_penalty", cls.bad_abi_penalty)),
            regress_penalty=float(raw.get("regress_penalty", cls.regress_penalty)),
        )


@dataclass
class VerifyResult:
    reward: float
    assembled: bool
    linked: bool
    ran: bool
    correct: bool
    public_pass_rate: float
    hidden_pass_rate: float
    stdout: str
    stderr: str
    exit_code: int | None
    stage_failed: str | None
    penalties: list[str]


class ObjectiveVerifier:
    def __init__(
        self,
        artifacts_dir: str | Path,
        timeout_seconds: int = 6,
        reward_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.timeout_seconds = timeout_seconds
        self.weights = RewardWeights.from_mapping(reward_weights)
        ensure_dir(self.artifacts_dir)

    def _evaluate_checks(self, stdout: str, exit_code: int, checks: list[dict[str, Any]]) -> float:
        if not checks:
            return 1.0 if exit_code == 0 else 0.0
        hits = 0
        for check in checks:
            expected_stdout = check.get("expected_stdout")
            expected_exit = check.get("expected_exit_code")
            if expected_stdout is not None:
                hits += int(stdout == expected_stdout)
            elif expected_exit is not None:
                hits += int(exit_code == expected_exit)
            else:
                hits += int(exit_code == 0)
        return hits / max(1, len(checks))

    def evaluate(self, task: Task, asm_code: str, sample_id: str) -> VerifyResult:
        workdir = self.artifacts_dir / sample_id
        ensure_dir(workdir)

        asm_path = workdir / "prog.asm"
        obj_path = workdir / "prog.o"
        bin_path = workdir / "prog"
        asm_path.write_text(asm_code + "\n", encoding="utf-8")

        reward = 0.0
        penalties: list[str] = []
        low = asm_code.lower()
        if "```" in asm_code or "explanation" in low:
            reward += self.weights.prose_penalty
            penalties.append("prose")
        if "int 0x80" in low or "int 80h" in low:
            reward += self.weights.bad_abi_penalty
            penalties.append("bad_abi")

        assemble = run_cmd(["nasm", "-f", "elf64", str(asm_path), "-o", str(obj_path)], self.timeout_seconds)
        if assemble.returncode != 0:
            return VerifyResult(reward, False, False, False, False, 0.0, 0.0, assemble.stdout, assemble.stderr, None, "assemble", penalties)
        reward += self.weights.assemble

        link = run_cmd(["ld", str(obj_path), "-o", str(bin_path)], self.timeout_seconds)
        if link.returncode != 0:
            return VerifyResult(reward, True, False, False, False, 0.0, 0.0, link.stdout, link.stderr, None, "link", penalties)
        reward += self.weights.link

        run = run_cmd([str(bin_path)], self.timeout_seconds)
        if run.returncode in (124, 125, 126, 127) or run.returncode < 0:
            return VerifyResult(reward, True, True, False, False, 0.0, 0.0, run.stdout, run.stderr, run.returncode, "run", penalties)
        reward += self.weights.run

        public_checks: list[dict[str, Any]] = []
        if task.expected_stdout is not None:
            public_checks.append({"expected_stdout": task.expected_stdout})
        elif task.expected_exit_code is not None:
            public_checks.append({"expected_exit_code": task.expected_exit_code})
        public_pass_rate = self._evaluate_checks(run.stdout, run.returncode, public_checks)
        hidden_pass_rate = self._evaluate_checks(run.stdout, run.returncode, task.hidden_tests)

        reward += self.weights.public_test * public_pass_rate
        reward += self.weights.hidden_test * hidden_pass_rate

        if len(asm_code.splitlines()) <= 20 and "```" not in asm_code:
            reward += self.weights.cleanliness

        return VerifyResult(
            reward=reward,
            assembled=True,
            linked=True,
            ran=True,
            correct=hidden_pass_rate >= 1.0,
            public_pass_rate=public_pass_rate,
            hidden_pass_rate=hidden_pass_rate,
            stdout=run.stdout,
            stderr=run.stderr,
            exit_code=run.returncode,
            stage_failed=None if hidden_pass_rate >= 1.0 else "correctness",
            penalties=penalties,
        )
