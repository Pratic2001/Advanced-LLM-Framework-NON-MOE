#!/usr/bin/env python3
"""
tokenizer_train.py

Byte-Pair Encoding (BBPE) tokenizer training with memory-safe streaming.
Reads special tokens from the TrainingRecipe so the token set is defined
in exactly one place.

Usage:
    python tokenizer_train.py --data-dir ./data --recipe ./recipe.json
    python tokenizer_train.py --data-dir ./data --mode reasoning  # defaults
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import List, Optional

from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


# ---------------------------------------------------------------------------
# Recipe integration
# ---------------------------------------------------------------------------

def _load_recipe_special_tokens(recipe_path: Optional[str], mode: Optional[str]) -> List[str]:
    """Get the special-token list from a recipe, falling back to defaults."""
    if recipe_path and os.path.isfile(recipe_path):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from recipe import get_recipe
        recipe = get_recipe(recipe_path)
        return recipe.special_tokens

    # Fallback: construct from mode
    base = ["<|endoftext|>", "<|pad|>", "<|im_start|>", "<|im_end|>"]
    if mode in ("reasoning", "hybrid"):
        base += ["<think>", "</think>"]
    if mode == "hybrid":
        base += ["<|think_on|>", "<|think_off|>"]
    return base


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_path: str):
    """
    Load a trained tokenizer from a directory or file.

    Accepts either:
        - A directory containing ``tokenizer.json``
        - A direct path to a ``.json`` tokenizer file

    Returns:
        A ``tokenizers.Tokenizer`` instance ready for encoding.
    """
    from tokenizers import Tokenizer

    # If a directory, look for tokenizer.json inside
    if os.path.isdir(tokenizer_path):
        tok_file = os.path.join(tokenizer_path, "tokenizer.json")
    else:
        tok_file = tokenizer_path

    if not os.path.isfile(tok_file):
        raise FileNotFoundError(
            f"Tokenizer file not found: {tok_file}\n"
            f"Train one first with: python tokenizer_train.py --data-dir ./data --output-dir ./tokenizer"
        )

    tokenizer = Tokenizer.from_file(tok_file)
    print(f"[tokenizer] loaded from {tok_file} (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tokenizer(
    data_dir: str,
    output_dir: str,
    special_tokens: List[str],
    vocab_size: int = 65536,
    min_frequency: int = 2,
) -> Tokenizer:
    """
    Train a Byte-level BPE tokenizer on all .jsonl files under ``data_dir``.

    The training is memory-safe: ``train_from_iterator`` streams from a
    generator that lazily reads each file line by line, so peak RAM is
    independent of corpus size.

    Args:
        data_dir: Directory tree containing .jsonl files with a "text" field.
        output_dir: Where to save the trained tokenizer.json.
        special_tokens: List of special tokens (from TrainingRecipe).
        vocab_size: Target vocabulary size.
        min_frequency: Minimum token frequency to be included.

    Returns:
        The trained Tokenizer instance.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialise BBPE tokenizer
    tok = Tokenizer(BPE(unk_token=None, byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    # Trainer with special tokens
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    # Stream corpus line-by-line (memory-safe even for multi-TB datasets)
    def line_stream():
        files = sorted(glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True))
        if not files:
            files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
        if not files:
            raise FileNotFoundError(
                f"No .jsonl files found under {data_dir}. "
                f"Run build_dataset.py first."
            )
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Extract text field — support pretrain {"text": ...}
                    # and SFT {"prompt": ..., "answer": ...} formats
                    text = rec.get("text") or rec.get("prompt") or ""
                    if "answer" in rec:
                        text += "\n" + rec.get("answer", "")
                    if text:
                        yield text

    # Train
    t0 = time.time()
    print(f"[tokenizer] training BBPE with vocab_size={vocab_size:,}, "
          f"min_frequency={min_frequency}, "
          f"{len(special_tokens)} special tokens …")
    tok.train_from_iterator(line_stream(), trainer=trainer)
    print(f"[tokenizer] trained in {time.time() - t0:.1f}s")

    # Save
    out_path = os.path.join(output_dir, "tokenizer.json")
    tok.save(out_path)
    print(f"[tokenizer] saved to {out_path}")
    print(f"[tokenizer] vocab_size={tok.get_vocab_size()}")
    print(f"[tokenizer] special tokens: {special_tokens}")

    return tok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a Byte-level BPE tokenizer for the dense LLM framework.",
    )
    p.add_argument("--data-dir", default="./data",
                    help="Directory of .jsonl files with 'text' or 'prompt' fields.")
    p.add_argument("--output-dir", default="./tokenizer",
                    help="Where to save tokenizer.json.")
    p.add_argument("--vocab-size", type=int, default=65536,
                    help="Target vocabulary size.")
    p.add_argument("--min-frequency", type=int, default=2,
                    help="Minimum token frequency to include.")
    p.add_argument("--recipe", default=None,
                    help="Path to recipe.json (defines special tokens and mode).")
    p.add_argument("--mode", default=None,
                    choices=["reasoning", "non_reasoning", "hybrid"],
                    help="Training mode (used when --recipe is not given).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    special_tokens = _load_recipe_special_tokens(args.recipe, args.mode)
    print(f"[tokenizer] special tokens: {special_tokens}")

    train_tokenizer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        special_tokens=special_tokens,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    print("\nDone.")
