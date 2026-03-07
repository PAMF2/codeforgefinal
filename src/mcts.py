from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .prompt_engine import PromptItem
    from .reward import RewardPipeline

_SYS_CONT = (
    "You are an expert NASM x86-64 Linux programmer. "
    "Output only assembly code, no markdown fences, no explanations. "
    "Given a partial program, output only the REMAINING lines to complete it."
)

# Patterns that indicate the model is generating non-assembly text.
_NON_ASM_PATTERNS = re.compile(
    r"^(def |class |import |function |#include|int main|public |private |return |print\(|console\.)",
    re.IGNORECASE,
)
# Lines that are plausible NASM — instructions, directives, labels, comments.
_ASM_HINT = re.compile(
    r"^\s*(global|section|extern|mov|add|sub|imul|idiv|div|xor|and|or|not|shl|shr|cmp|"
    r"jmp|je|jne|jl|jg|jle|jge|ja|jb|jz|jnz|call|ret|push|pop|lea|test|inc|dec|neg|"
    r"syscall|int|db|dw|dd|dq|resb|equ|nop|_start|\.|\w+:|\s*;)",
    re.IGNORECASE,
)


@dataclass
class MCTSConfig:
    simulations: int = 32
    max_lines: int = 30
    branch_factor: int = 4
    exploration_constant: float = 1.414
    max_depth: int = 15
    min_tier: int = 3  # Only apply MCTS to prompts with tier >= min_tier


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
    ) -> list[str]: ...


@dataclass
class _Node:
    lines: tuple[str, ...]
    parent: _Node | None = field(default=None, repr=False)
    children: dict[str, _Node] = field(default_factory=dict)
    visits: int = 0
    total_reward: float = 0.0
    best_reward: float = 0.0
    best_program: str | None = None

    def ucb(self, c: float) -> float:
        if self.visits == 0:
            return float("inf")
        parent_v = self.parent.visits if self.parent else self.visits
        exploitation = self.total_reward / self.visits
        exploration = c * math.sqrt(math.log(max(1, parent_v)) / self.visits)
        return exploitation + exploration

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def prefix_text(self) -> str:
        return "\n".join(self.lines)

    def depth(self) -> int:
        d, n = 0, self
        while n.parent is not None:
            d += 1
            n = n.parent
        return d


class MCTSLineSearch:
    """
    Prefix-tree MCTS for assembly generation (tier 3+ prompts).

    Selection:   UCB1 traversal to a leaf or terminal node.
    Expansion:   Generator produces branch_factor completions from the prefix;
                 plausible first lines become child nodes.
    Simulation:  Complete the program from the selected node and score it.
    Backprop:    Propagate reward up to root.
    """

    def __init__(
        self,
        cfg: MCTSConfig,
        generator: TextGenerator,
        reward_pipeline: RewardPipeline,
    ) -> None:
        self.cfg = cfg
        self.generator = generator
        self.reward_pipeline = reward_pipeline

    # ------------------------------------------------------------------ helpers

    def _make_prompt(self, task: str, prefix_lines: tuple[str, ...]) -> str:
        base = f"{_SYS_CONT}\n\nTask: {task}"
        if prefix_lines:
            prefix = "\n".join(prefix_lines)
            base += f"\n\nPartial program:\n{prefix}\n\nContinue:"
        return base

    def _is_terminal(self, lines: tuple[str, ...]) -> bool:
        if len(lines) >= self.cfg.max_lines:
            return True
        text = "".join(lines).replace(" ", "").replace("\t", "").lower()
        has_exit = "movrax,60" in text or "moveax,60" in text
        has_syscall = "syscall" in text
        has_int80 = "int0x80" in text
        return (has_exit and has_syscall) or has_int80

    @staticmethod
    def _is_plausible_line(line: str) -> bool:
        """Reject lines that are clearly not assembly (model hallucinating prose)."""
        stripped = line.strip()
        if not stripped:
            return False
        if _NON_ASM_PATTERNS.match(stripped):
            return False
        return bool(_ASM_HINT.match(stripped)) or stripped.startswith(";")

    @staticmethod
    def _parse_lines(code: str) -> tuple[str, ...]:
        from .utils import sanitize_model_output

        cleaned = sanitize_model_output(code)
        return tuple(line for line in cleaned.splitlines() if line.strip())

    # ------------------------------------------------------------------ MCTS phases

    def _select(self, root: _Node) -> _Node:
        node = root
        while not node.is_leaf() and not self._is_terminal(node.lines):
            node = max(
                node.children.values(),
                key=lambda n: n.ucb(self.cfg.exploration_constant),
            )
        return node

    def _expand(self, node: _Node, task: str) -> list[_Node]:
        if self._is_terminal(node.lines) or node.depth() >= self.cfg.max_depth:
            return [node]

        prompt = self._make_prompt(task, node.lines)
        completions = self.generator(
            prompt=prompt,
            n=self.cfg.branch_factor,
            max_new_tokens=256,
            temperature=0.9,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.01,
        )

        new_children: list[_Node] = []
        for comp in completions:
            new_lines = self._parse_lines(comp)
            if not new_lines:
                continue
            next_line = new_lines[0]
            # Plausibility filter: skip if line looks like prose/non-assembly.
            if not self._is_plausible_line(next_line):
                continue
            if next_line in node.children:
                new_children.append(node.children[next_line])
            else:
                child = _Node(lines=node.lines + (next_line,), parent=node)
                node.children[next_line] = child
                new_children.append(child)

        return new_children or [node]

    def _simulate(
        self,
        node: _Node,
        task: str,
        prompt_item: PromptItem,
        sample_id: str,
    ) -> tuple[float, str, Any]:
        from .utils import sanitize_model_output

        prompt = self._make_prompt(task, node.lines)
        completions = self.generator(
            prompt=prompt,
            n=1,
            max_new_tokens=384,
            temperature=0.7,
            top_p=0.95,
            top_k=24,
            repetition_penalty=1.02,
        )
        tail = sanitize_model_output(completions[0])
        full = (node.prefix_text() + "\n" + tail).strip() if node.lines else tail
        result = self.reward_pipeline.evaluate(prompt_item, full, sample_id)
        return result.reward, full, result

    def _backpropagate(self, node: _Node, reward: float, program: str) -> None:
        cur: _Node | None = node
        while cur is not None:
            cur.visits += 1
            cur.total_reward += reward
            if reward > cur.best_reward:
                cur.best_reward = reward
                cur.best_program = program
            cur = cur.parent

    # ------------------------------------------------------------------ public API

    def should_apply(self, prompt_item: PromptItem) -> bool:
        """Returns True if MCTS should be applied to this prompt (tier >= min_tier)."""
        return prompt_item.tier >= self.cfg.min_tier

    def search(
        self,
        task: str,
        prompt_item: PromptItem,
        sample_prefix: str,
    ) -> list[dict[str, Any]]:
        """
        Run MCTS and return row dicts compatible with evaluate_candidates().
        Rows are sorted best-reward-first.
        Also returns avg_depth as metadata in each row.
        """
        root = _Node(lines=())
        all_rows: list[dict[str, Any]] = []
        seen_programs: set[str] = set()
        depths: list[int] = []

        for sim_idx in range(self.cfg.simulations):
            node = self._select(root)
            children = self._expand(node, task)
            node = children[0]
            depths.append(node.depth())

            sample_id = f"{sample_prefix}-s{sim_idx}"
            reward, program, rr = self._simulate(node, task, prompt_item, sample_id)
            self._backpropagate(node, reward, program)

            if program not in seen_programs:
                seen_programs.add(program)
                all_rows.append(
                    {
                        "prompt_id": prompt_item.id,
                        "instruction": prompt_item.instruction,
                        "tier": prompt_item.tier,
                        "asm": program,
                        "reward": reward,
                        "assembled": rr.assembled,
                        "linked": rr.linked,
                        "ran": rr.ran,
                        "correct": rr.correct,
                        "stage_failed": rr.stage_failed,
                        "source": "mcts",
                    }
                )

        avg_depth = sum(depths) / max(1, len(depths))
        # Attach avg_depth to each row for W&B logging aggregation.
        for row in all_rows:
            row["mcts_avg_depth"] = avg_depth

        all_rows.sort(key=lambda r: r["reward"], reverse=True)
        return all_rows

    def generate_candidates(self, _task_prompt: str) -> list[str]:
        """Stub for legacy interface. Use search() for full MCTS."""
        return []
