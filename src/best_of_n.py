from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .utils import sanitize_model_output


class TextGenerator(Protocol):
    def __call__(
        self,
        prompt: str,
        n: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int | None = None,
        min_p: float | None = None,
        repetition_penalty: float = 1.0,
    ) -> list[str]:
        ...


@dataclass
class BestOfNConfig:
    n: int = 16
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float = 1.0


class BestOfN:
    def __init__(self, generator: TextGenerator, cfg: BestOfNConfig) -> None:
        self.generator = generator
        self.cfg = cfg

    def generate(self, task_prompt: str) -> list[str]:
        raw_outputs = self.generator(
            prompt=task_prompt,
            n=self.cfg.n,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            min_p=self.cfg.min_p,
            repetition_penalty=self.cfg.repetition_penalty,
        )
        return [sanitize_model_output(x) for x in raw_outputs]
