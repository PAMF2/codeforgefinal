from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = ("task_id", "tier", "instruction")


def main() -> None:
    p = argparse.ArgumentParser(description="Validate Assembly-SWE task dataset JSONL")
    p.add_argument("--tasks", required=True, help="Path to tasks JSONL")
    args = p.parse_args()

    path = Path(args.tasks).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    lines = path.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    errors: list[str] = []
    tiers: dict[int, int] = {}
    rows = 0

    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        rows += 1
        try:
            row = json.loads(line)
        except Exception as exc:
            errors.append(f"line {i}: invalid json ({exc})")
            continue
        for k in REQUIRED:
            if k not in row:
                errors.append(f"line {i}: missing required key '{k}'")
        task_id = str(row.get("task_id", "")).strip()
        if not task_id:
            errors.append(f"line {i}: empty task_id")
        elif task_id in seen:
            errors.append(f"line {i}: duplicate task_id '{task_id}'")
        else:
            seen.add(task_id)

        tier = row.get("tier")
        if not isinstance(tier, int) or tier < 1:
            errors.append(f"line {i}: tier must be int >= 1")
        else:
            tiers[tier] = tiers.get(tier, 0) + 1

        if not isinstance(row.get("instruction", None), str) or not row["instruction"].strip():
            errors.append(f"line {i}: instruction must be non-empty string")
        if "expected_stdout" in row and not isinstance(row["expected_stdout"], str):
            errors.append(f"line {i}: expected_stdout must be string when provided")
        if "expected_exit_code" in row and not isinstance(row["expected_exit_code"], int):
            errors.append(f"line {i}: expected_exit_code must be int when provided")

    print(f"[validate_dataset] file: {path}")
    print(f"[validate_dataset] parsed_rows: {rows}")
    print(f"[validate_dataset] unique_task_ids: {len(seen)}")
    print(f"[validate_dataset] tier_distribution: {dict(sorted(tiers.items()))}")

    if errors:
        print(f"[validate_dataset] errors: {len(errors)}")
        for e in errors[:50]:
            print(" -", e)
        raise SystemExit(1)

    print("[validate_dataset] OK")


if __name__ == "__main__":
    main()
