#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pred_rows = read_jsonl(Path(args.predictions))
    pred_rows.sort(key=lambda row: (row["task_id"], row.get("candidate_rank", 999999)))

    by_task: dict[str, list[dict]] = {}
    for row in pred_rows:
        by_task.setdefault(str(row["task_id"]), []).append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for task_id, rows in by_task.items():
            rows.sort(key=lambda row: int(row.get("candidate_rank", 999999)))
            best = rows[0]
            if not bool(best.get("correct")):
                continue
            prompt = f"Task: {best['instruction']}"
            completion = str(best["asm"]).strip() + "\n"
            handle.write(json.dumps({
                "task_id": task_id,
                "prompt": prompt,
                "completion": completion,
            }, ensure_ascii=False) + "\n")
            written += 1

    print({"written": written, "out": str(out_path)})


if __name__ == "__main__":
    main()
