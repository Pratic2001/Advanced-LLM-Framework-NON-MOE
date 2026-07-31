#!/usr/bin/env python3
"""
recipe.py

Single source of truth for training recipe configuration — mode, chat template,
special tokens, and model_name. Every script in the framework imports from
here instead of hardcoding its own copy of template strings, token lists, or
the architecture name.

Usage:
    from recipe import TrainingRecipe, get_recipe

    # Default reasoning recipe
    recipe = TrainingRecipe()
    print(recipe.special_tokens)
    print(recipe.format_assistant_turn(thinking="2+2=4", answer="4"))

    # Load from a saved recipe.json
    recipe = get_recipe("/path/to/checkpoint/dir")

    # JSON serialisation (saved alongside checkpoints)
    recipe.to_json("/path/to/recipe.json")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional


# ---------------------------------------------------------------------------
# TrainingRecipe — the one place mode / template / tokens are defined
# ---------------------------------------------------------------------------


@dataclass
class TrainingRecipe:
    """
    Central configuration for a training run. Every script reads its
    template, special tokens, reward behaviour, and model name from this
    object rather than hardcoding its own copies.

    Modes:
        reasoning      — expects <think>...</think> blocks; GRPO reward
                         includes a think-format tier. Default for math/code.
        non_reasoning  — no thinking tags expected; GRPO reward skips the
                         think-format check entirely. Use for general
                         instruction-following or chat.
        hybrid         — each example can independently request thinking or
                         not via the `want_thinking` flag in the record.
                         The packer reads `want_thinking` from the JSONL;
                         GRPO reward conditions on it.
    """

    mode: Literal["reasoning", "non_reasoning", "hybrid"] = "reasoning"

    # ------------------------------------------------------------------
    # Chat template
    # ------------------------------------------------------------------
    chat_template: Literal["chatml", "raw", "custom"] = "chatml"

    turn_prefix_user: str = "<|im_start|>user\n"
    turn_suffix_user: str = "<|im_end|>\n"
    turn_prefix_assistant: str = "<|im_start|>assistant\n"
    turn_suffix_assistant: str = "<|im_end|>\n"

    # ------------------------------------------------------------------
    # Reasoning-specific tags (ignored when mode == "non_reasoning")
    # ------------------------------------------------------------------
    think_open: str = "<think>"
    think_close: str = "</think>"

    # Hybrid-mode control tokens — the user puts these in prompt text to
    # toggle thinking on/off per-example (à la Qwen3's enable_thinking).
    hybrid_think_token: str = "<|think_on|>"
    hybrid_nothink_token: str = "<|think_off|>"

    # ------------------------------------------------------------------
    # Base special tokens — the ONE place this list is defined
    # ------------------------------------------------------------------
    base_special_tokens: List[str] = field(default_factory=lambda: [
        "<|endoftext|>",
        "<|pad|>",
        "<|im_start|>",
        "<|im_end|>",
    ])

    # ------------------------------------------------------------------
    # Model name — replaces hardcoded "Qwen3" throughout the framework
    # ------------------------------------------------------------------
    model_name: str = "DenseLLM"

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def special_tokens(self) -> List[str]:
        """Complete list of special tokens for this recipe, derived from
        mode, base tokens, and thinking tags."""
        toks = list(self.base_special_tokens)
        if self.mode in ("reasoning", "hybrid"):
            toks += [self.think_open, self.think_close]
        if self.mode == "hybrid":
            toks += [self.hybrid_think_token, self.hybrid_nothink_token]
        return toks

    @property
    def eos_token(self) -> str:
        return "<|endoftext|>"

    @property
    def pad_token(self) -> str:
        return "<|pad|>"

    # ------------------------------------------------------------------
    # Template formatting
    # ------------------------------------------------------------------

    def format_user_turn(self, prompt: str) -> str:
        """Wrap a user message in the chat template."""
        return f"{self.turn_prefix_user}{prompt}{self.turn_suffix_user}"

    def format_assistant_turn(
        self,
        thinking: str = "",
        answer: str = "",
        want_thinking: Optional[bool] = None,
    ) -> str:
        """Format the assistant turn.

        Args:
            thinking: The reasoning/chain-of-thought text (empty = no thinking).
            answer: The final answer text.
            want_thinking: Only used in hybrid mode. If None, defaults to
                           ``bool(thinking)``. In reasoning mode this is
                           forced True; in non_reasoning mode forced False.

        Returns:
            The formatted assistant turn string, including prefix/suffix.
        """
        if self.mode == "non_reasoning":
            want = False
        elif self.mode == "reasoning":
            want = True
        else:  # hybrid
            want = want_thinking if want_thinking is not None else bool(thinking)

        if want:
            body = f"{self.think_open}\n{thinking}\n{self.think_close}\n{answer}"
        else:
            body = answer
        return f"{self.turn_prefix_assistant}{body}{self.turn_suffix_assistant}"

    def format_system_turn(self, system: str) -> str:
        """Format a system message with the same prefix/suffix pair as a user
        turn but with the role tag rewritten. Default ChatML layout uses
        ``system\\n{system}\\n``. Subclasses / custom templates should
        override this rather than relying on string substitution.
        """
        if not system:
            return ""
        # Explicit rewrite — do NOT use ``replace('user', 'system')`` because
        # if the prefix ever contains the substring ``user`` inside the
        # content (e.g. content references "user") it would corrupt output.
        prefix = self.turn_prefix_user.replace("user", "system")
        return f"{prefix}{system}{self.turn_suffix_user}"

    def format_full_conversation(
        self,
        prompt: str,
        thinking: str = "",
        answer: str = "",
        want_thinking: Optional[bool] = None,
        system: str = "",
    ) -> str:
        """Format a full conversation: optional system message, user turn,
        assistant turn. This is the string that gets tokenised and packed."""
        parts: List[str] = []
        if system:
            parts.append(self.format_system_turn(system))
        parts.append(self.format_user_turn(prompt))
        parts.append(self.format_assistant_turn(
            thinking=thinking, answer=answer, want_thinking=want_thinking,
        ))
        return "".join(parts)

    # ------------------------------------------------------------------
    # Reward behaviour: should we check for a <think> block?
    # ------------------------------------------------------------------

    def reward_should_check_thinking(self, want_thinking: Optional[bool] = None) -> bool:
        """Whether the GRPO reward function should check for balanced
        <think>...</think> in the completion."""
        if self.mode == "reasoning":
            return True
        if self.mode == "non_reasoning":
            return False
        # hybrid: check only if this prompt wanted thinking
        return bool(want_thinking)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrainingRecipe:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> TrainingRecipe:
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Recipe loader utility — used identically by every script
# ---------------------------------------------------------------------------

def get_recipe(
    checkpoint_dir_or_path: Optional[str] = None,
    mode: Optional[str] = None,
) -> TrainingRecipe:
    """
    Load a TrainingRecipe from a checkpoint directory or a recipe.json path.

    Resolution order:
        1. If checkpoint_dir_or_path is a path to a recipe.json, load it.
        2. If checkpoint_dir_or_path is a directory containing recipe.json,
           load from there.
        3. If checkpoint_dir_or_path is a checkpoint .pt file, look for
           recipe.json next to it or in the parent directory.
        4. Fall back to a default recipe constructed from ``mode`` (or
           "reasoning" if neither is given).

    This is the ONE function every script calls to answer "what recipe am
    I using?" — ensuring a single code path and no duplicated defaults.
    """
    # Case 1: explicit recipe.json path
    if checkpoint_dir_or_path and os.path.isfile(checkpoint_dir_or_path):
        if checkpoint_dir_or_path.endswith(".json"):
            return TrainingRecipe.from_json(checkpoint_dir_or_path)
        # Case 3: .pt checkpoint — look for sibling recipe.json
        sibling = os.path.splitext(checkpoint_dir_or_path)[0] + "_recipe.json"
        if os.path.isfile(sibling):
            return TrainingRecipe.from_json(sibling)
        parent_dir = os.path.dirname(os.path.abspath(checkpoint_dir_or_path))
        parent_recipe = os.path.join(parent_dir, "recipe.json")
        if os.path.isfile(parent_recipe):
            return TrainingRecipe.from_json(parent_recipe)

    # Case 2: directory with recipe.json
    if checkpoint_dir_or_path and os.path.isdir(checkpoint_dir_or_path):
        recipe_path = os.path.join(checkpoint_dir_or_path, "recipe.json")
        if os.path.isfile(recipe_path):
            return TrainingRecipe.from_json(recipe_path)

    # Case 4: fallback to defaults
    return TrainingRecipe(mode=mode or "reasoning")


# ---------------------------------------------------------------------------
# Mode-aware argument helper (reduces argparse boilerplate in training scripts)
# ---------------------------------------------------------------------------

def add_recipe_args(parser) -> None:
    """Add --recipe and --mode arguments to an argparse parser."""
    parser.add_argument(
        "--recipe", default=None,
        help="Path to a recipe.json file. If omitted, uses --mode to "
             "construct a default recipe.",
    )
    parser.add_argument(
        "--mode", default=None,
        choices=["reasoning", "non_reasoning", "hybrid"],
        help="Training mode (only used when --recipe is not provided)."
             " Default: reasoning.",
    )


def recipe_from_args(args) -> TrainingRecipe:
    """Build a TrainingRecipe from parsed CLI args (--recipe / --mode)."""
    if getattr(args, "recipe", None):
        return get_recipe(args.recipe)
    mode = getattr(args, "mode", None) or "reasoning"
    return TrainingRecipe(mode=mode)
