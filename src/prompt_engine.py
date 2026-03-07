from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PromptItem:
    id: str
    tier: int
    instruction: str
    expected_stdout: str | None = None
    expected_exit_code: int | None = None
    hint_lines: int | None = None
    reference_solution: str | None = None


class PromptEngine:
    def __init__(self, dataset_path: str | Path) -> None:
        self.dataset_path = Path(dataset_path)
        self._items = self._load()

    def _load(self) -> list[PromptItem]:
        raw = json.loads(self.dataset_path.read_text(encoding="utf-8-sig"))
        items = []
        for item in raw:
            items.append(
                PromptItem(
                    id=item["id"],
                    tier=item["tier"],
                    instruction=item["instruction"],
                    expected_stdout=item.get("expected_stdout"),
                    expected_exit_code=item.get("expected_exit_code"),
                    hint_lines=item.get("hint_lines"),
                    reference_solution=item.get("reference_solution"),
                )
            )
        return items

    def sample(self, n: int) -> list[PromptItem]:
        """Deterministic sample (first N). Use sample_random() for training."""
        return self._items[:n]

    def sample_random(self, n: int) -> list[PromptItem]:
        """Random sample without replacement. Wraps if n > len(items)."""
        if n >= len(self._items):
            return list(self._items)
        return random.sample(self._items, n)

    def sample_by_tier(self, n: int, tier: int) -> list[PromptItem]:
        """Random sample from a specific tier."""
        tier_items = [p for p in self._items if p.tier == tier]
        if not tier_items:
            return []
        return random.sample(tier_items, min(n, len(tier_items)))

    def sample_min_tier(self, n: int, min_tier: int) -> list[PromptItem]:
        """Random sample from prompts with tier >= min_tier."""
        eligible = [p for p in self._items if p.tier >= min_tier]
        if not eligible:
            return []
        return random.sample(eligible, min(n, len(eligible)))

    def all_items(self) -> list[PromptItem]:
        return list(self._items)

    def tier_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for item in self._items:
            counts[item.tier] = counts.get(item.tier, 0) + 1
        return counts

    @staticmethod
    def from_mistral_generation(_: dict[str, Any]) -> list[dict[str, Any]]:
        """Placeholder for automatic prompt generation via Mistral API."""
        raise NotImplementedError("Use prompts/generate_prompts.py instead.")
