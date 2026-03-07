#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import string
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import Task
from src.verifier import ObjectiveVerifier


def bytes_db_literal(text: str) -> str:
    return ", ".join(str(b) for b in text.encode("utf-8"))


def asm_print_stdout(stdout: str) -> str:
    data = bytes_db_literal(stdout)
    return (
        "global _start\n"
        "section .data\n"
        f"    msg db {data}\n"
        "    len equ $ - msg\n"
        "section .text\n"
        "_start:\n"
        "    mov rax, 1\n"
        "    mov rdi, 1\n"
        "    mov rsi, msg\n"
        "    mov rdx, len\n"
        "    syscall\n"
        "    mov rax, 60\n"
        "    xor rdi, rdi\n"
        "    syscall"
    )


def asm_exit(code: int) -> str:
    return (
        "global _start\n"
        "section .text\n"
        "_start:\n"
        "    mov rax, 60\n"
        f"    mov rdi, {code}\n"
        "    syscall"
    )


def rand_token(rng: random.Random, min_len: int = 3, max_len: int = 12) -> str:
    n = rng.randint(min_len, max_len)
    alpha = string.ascii_letters + string.digits
    return "".join(rng.choice(alpha) for _ in range(n))


def build_hidden_tests(stdout: str | None, exit_code: int | None) -> list[dict[str, Any]]:
    if stdout is not None:
        return [{"expected_stdout": stdout}, {"expected_stdout": stdout}]
    if exit_code is not None:
        return [{"expected_exit_code": exit_code}, {"expected_exit_code": exit_code}]
    return []


def make_exit_task(i: int, rng: random.Random) -> dict[str, Any]:
    code = rng.randint(0, 255)
    return {
        "task_id": f"cfv2_exit_{i:06d}",
        "tier": 1,
        "task_kind": "generate",
        "family_id": "exit_code",
        "instruction": f"Write a NASM x86-64 Linux program that exits with code {code}.",
        "expected_exit_code": code,
        "expected_stdout": None,
        "hidden_tests": build_hidden_tests(None, code),
        "reference_solution": asm_exit(code),
    }


def make_print_task(i: int, rng: random.Random, tier: int = 2) -> dict[str, Any]:
    token_a = rand_token(rng, 2, 10)
    token_b = rand_token(rng, 2, 10)
    text = f"{token_a}_{token_b}"
    stdout = text + "\n"
    return {
        "task_id": f"cfv2_print_{i:06d}",
        "tier": tier,
        "task_kind": "generate",
        "family_id": "print_literal",
        "instruction": f'Write a NASM x86-64 Linux program that prints exactly "{text}" followed by a newline.',
        "expected_exit_code": None,
        "expected_stdout": stdout,
        "hidden_tests": build_hidden_tests(stdout, None),
        "reference_solution": asm_print_stdout(stdout),
    }


def make_math_task(i: int, rng: random.Random) -> dict[str, Any]:
    a = rng.randint(-99, 199)
    b = rng.randint(-99, 199)
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        out = a + b
    elif op == "-":
        out = a - b
    else:
        out = a * b
    stdout = f"{out}\n"
    return {
        "task_id": f"cfv2_math_{i:06d}",
        "tier": 3,
        "task_kind": "generate",
        "family_id": "const_math_stdout",
        "instruction": (
            "Write a NASM x86-64 Linux program that computes the constant expression "
            f"{a} {op} {b} and prints the result followed by a newline."
        ),
        "expected_exit_code": None,
        "expected_stdout": stdout,
        "hidden_tests": build_hidden_tests(stdout, None),
        "reference_solution": asm_print_stdout(stdout),
    }


def make_two_line_task(i: int, rng: random.Random, tier: int = 3) -> dict[str, Any]:
    token_a = rand_token(rng, 3, 14)
    token_b = rand_token(rng, 3, 14)
    line1 = f"head-{token_a}"
    line2 = f"tail-{token_b}"
    stdout = f"{line1}\n{line2}\n"
    return {
        "task_id": f"cfv2_twoline_{i:06d}",
        "tier": tier,
        "task_kind": "generate",
        "family_id": "two_line_stdout",
        "instruction": (
            "Write a NASM x86-64 Linux program that prints exactly two lines: "
            f'"{line1}" and "{line2}", each terminated by newline, and then exits with code 0.'
        ),
        "expected_exit_code": None,
        "expected_stdout": stdout,
        "hidden_tests": build_hidden_tests(stdout, None),
        "reference_solution": asm_print_stdout(stdout),
    }


def generate_core(total: int, rng: random.Random) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(total):
        p = rng.random()
        if p < 0.30:
            rows.append(make_exit_task(i, rng))
        elif p < 0.65:
            rows.append(make_print_task(i, rng, tier=2))
        elif p < 0.90:
            rows.append(make_math_task(i, rng))
        else:
            rows.append(make_two_line_task(i, rng, tier=3))
    return rows


def _replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        return text
    return text.replace(old, new, 1)


def corrupt_bad_syscall(program: str) -> tuple[str, str]:
    return "bad_syscall", _replace_once(program, "syscall", "syscal")


def corrupt_wrong_exit(program: str) -> tuple[str, str]:
    if "mov rdi, 0" in program:
        return "wrong_exit", _replace_once(program, "mov rdi, 0", "mov rdi, 7")
    if "mov rdi," in program:
        before, after = program.split("mov rdi,", 1)
        line, rest = after.split("\n", 1)
        return "wrong_exit", before + "mov rdi, 0\n" + rest
    return "wrong_exit", program + "\n    mov rdi, 0"


def corrupt_wrong_len(program: str) -> tuple[str, str]:
    if "mov rdx, len" in program:
        return "wrong_len", _replace_once(program, "mov rdx, len", "mov rdx, 1")
    return "wrong_len", program


def corrupt_wrong_rax(program: str) -> tuple[str, str]:
    if "mov rax, 60" in program:
        return "wrong_rax", _replace_once(program, "mov rax, 60", "mov rax, 1")
    if "mov rax, 1" in program:
        return "wrong_rax", _replace_once(program, "mov rax, 1", "mov rax, 60")
    return "wrong_rax", program


CORRUPTORS: list[Callable[[str], tuple[str, str]]] = [
    corrupt_bad_syscall,
    corrupt_wrong_exit,
    corrupt_wrong_len,
    corrupt_wrong_rax,
]


def make_repair_task(i: int, base_task: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    bug_type, starter = rng.choice(CORRUPTORS)(str(base_task["reference_solution"]))
    return {
        "task_id": f"cfv2_repair_{i:06d}",
        "tier": max(4, int(base_task["tier"]) + 1),
        "task_kind": "repair",
        "family_id": f"repair_{base_task['family_id']}",
        "instruction": (
            "Fix the NASM x86-64 Linux program below so that it satisfies this requirement: "
            f"{base_task['instruction']}"
        ),
        "expected_exit_code": base_task.get("expected_exit_code"),
        "expected_stdout": base_task.get("expected_stdout"),
        "hidden_tests": list(base_task.get("hidden_tests", [])),
        "starter_code": starter,
        "reference_solution": base_task["reference_solution"],
        "bug_type": bug_type,
    }


def generate_repair(rows: list[dict[str, Any]], total: int, rng: random.Random, offset: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    out: list[dict[str, Any]] = []
    for i in range(total):
        base = rows[rng.randrange(len(rows))]
        out.append(make_repair_task(offset + i, base, rng))
    return out


def split_train_dev(rows: list[dict[str, Any]], dev_size: int, rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    dev_size = max(1, min(dev_size, len(shuffled) - 1))
    return shuffled[dev_size:], shuffled[:dev_size]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tiers: dict[str, int] = {}
    families: dict[str, int] = {}
    kinds: dict[str, int] = {}
    for row in rows:
        tier = str(row.get("tier", "0"))
        family = str(row.get("family_id", "unknown"))
        kind = str(row.get("task_kind", "generate"))
        tiers[tier] = tiers.get(tier, 0) + 1
        families[family] = families.get(family, 0) + 1
        kinds[kind] = kinds.get(kind, 0) + 1
    return {
        "count": len(rows),
        "tiers": dict(sorted(tiers.items())),
        "families": dict(sorted(families.items())),
        "task_kinds": dict(sorted(kinds.items())),
    }


def to_task(row: dict[str, Any]) -> Task:
    return Task(
        task_id=str(row["task_id"]),
        instruction=str(row["instruction"]),
        tier=int(row.get("tier", 1)),
        expected_stdout=row.get("expected_stdout"),
        expected_exit_code=row.get("expected_exit_code"),
        hidden_tests=list(row.get("hidden_tests", [])),
        task_kind=str(row.get("task_kind", "generate")),
        family_id=str(row.get("family_id", "")),
        starter_code=row.get("starter_code"),
        reference_solution=row.get("reference_solution"),
        metadata={k: v for k, v in row.items() if k not in {
            "task_id",
            "instruction",
            "tier",
            "expected_stdout",
            "expected_exit_code",
            "hidden_tests",
            "task_kind",
            "family_id",
            "starter_code",
            "reference_solution",
        }},
    )


def maybe_validate(rows: list[dict[str, Any]], sample: int, seed: int, out_dir: Path) -> dict[str, Any]:
    if sample <= 0:
        return {"enabled": False}
    verifier = ObjectiveVerifier(artifacts_dir=out_dir / "validation_artifacts", timeout_seconds=8)
    rng = random.Random(seed + 991)
    sample_rows = list(rows)
    rng.shuffle(sample_rows)
    sample_rows = sample_rows[: min(sample, len(sample_rows))]

    passed = 0
    failed: list[dict[str, Any]] = []
    for row in sample_rows:
        task = to_task(row)
        result = verifier.evaluate(task, str(row.get("reference_solution") or ""), sample_id=f"validate-{task.task_id}")
        if result.correct:
            passed += 1
        else:
            failed.append({
                "task_id": task.task_id,
                "stage_failed": result.stage_failed,
                "stderr": result.stderr[:200],
            })
    return {
        "enabled": True,
        "ok": len(failed) == 0,
        "sample_size": len(sample_rows),
        "passed": passed,
        "failed_count": len(failed),
        "failed_examples": failed[:20],
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CodeForgeFinal synthetic assembly tasks.")
    parser.add_argument("--out-dir", default="data/generated")
    parser.add_argument("--core-size", type=int, default=5000)
    parser.add_argument("--repair-size", type=int, default=2000)
    parser.add_argument("--dev-size", type=int, default=500)
    parser.add_argument("--hard-size", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-sample", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    generation_rows = generate_core(total=int(args.core_size), rng=rng)
    repair_rows = generate_repair(generation_rows, total=int(args.repair_size), rng=rng, offset=len(generation_rows) + 1)
    hard_rows = generate_repair(generation_rows, total=int(args.hard_size), rng=rng, offset=len(generation_rows) + len(repair_rows) + 1)
    mixed = generation_rows + repair_rows
    train_rows, dev_rows = split_train_dev(mixed, dev_size=int(args.dev_size), rng=rng)
    private_eval = hard_rows[: max(1, len(hard_rows) // 2)]
    full_rows = mixed + hard_rows

    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "dev.jsonl", dev_rows)
    write_jsonl(out_dir / "hard.jsonl", hard_rows)
    write_jsonl(out_dir / "private_eval.jsonl", private_eval)
    write_jsonl(out_dir / "generation_only.jsonl", generation_rows)
    write_jsonl(out_dir / "repair_only.jsonl", repair_rows)
    write_jsonl(out_dir / "full.jsonl", full_rows)

    validation = maybe_validate(full_rows, sample=int(args.validate_sample), seed=int(args.seed), out_dir=out_dir)
    manifest = {
        "dataset": "codeforgefinal_synth_v1",
        "seed": int(args.seed),
        "paths": {
            "train": str((out_dir / "train.jsonl").as_posix()),
            "dev": str((out_dir / "dev.jsonl").as_posix()),
            "hard": str((out_dir / "hard.jsonl").as_posix()),
            "private_eval": str((out_dir / "private_eval.jsonl").as_posix()),
            "generation_only": str((out_dir / "generation_only.jsonl").as_posix()),
            "repair_only": str((out_dir / "repair_only.jsonl").as_posix()),
            "full": str((out_dir / "full.jsonl").as_posix()),
        },
        "summary": {
            "train": summarize(train_rows),
            "dev": summarize(dev_rows),
            "hard": summarize(hard_rows),
            "private_eval": summarize(private_eval),
            "generation_only": summarize(generation_rows),
            "repair_only": summarize(repair_rows),
            "full": summarize(full_rows),
        },
        "validation": validation,
    }
    write_json(out_dir / "manifest.json", manifest)

    print(
        "[generate_synthetic_tasks] done | "
        f"train={len(train_rows)} dev={len(dev_rows)} hard={len(hard_rows)} "
        f"repair={len(repair_rows)} out={out_dir}"
    )
    if validation.get("enabled"):
        print(
            "[generate_synthetic_tasks] validation | "
            f"ok={validation.get('ok')} sample={validation.get('sample_size')} "
            f"failed={validation.get('failed_count')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
