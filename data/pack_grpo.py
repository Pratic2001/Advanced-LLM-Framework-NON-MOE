#!/usr/bin/env python3
"""
data/pack_grpo.py

Pack {prompt, answer} JSONL pairs into memmap cache + raw answer strings
for GRPO reward computation.

The memmap stores prompt tokens (including assistant turn prefix) + EOS separator.
The answer strings are preserved in a JSON list per shard for reward functions.

During GRPO training:
    1. Load prompt tokens from memmap (batch of prompts)
    2. Model generates completions from each prompt
    3. Reward function compares completions against the stored answer strings

Usage:
    python data/pack_grpo.py --data-dir ./data --tokenizer ./tokenizer --cache-dir ./packed
    python data/pack_grpo.py --data-dir ./data --mode reasoning --worker 0 --num-workers 4
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tokenizer_train import load_tokenizer  # noqa: E402
from recipe import TrainingRecipe, get_recipe, recipe_from_args, add_recipe_args  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[pack_grpo] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_jsonl_files(data_dir: str) -> List[str]:
    """
    Discover every JSONL shard under <data_dir>/<category>/*.jsonl.
    Sorting first makes the assignment deterministic across processes.
    """
    all_shards = sorted(glob.glob(os.path.join(data_dir, "*", "*.jsonl")))
    if not all_shards:
        # Fallback: try flat structure
        all_shards = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    if not all_shards:
        raise FileNotFoundError(
            f"No .jsonl shards found under {data_dir}/<category>/. "
            "Expected JSONL with 'prompt' and 'answer' fields."
        )
    return all_shards


def assign_files_to_worker(
    files: List[str], worker: int, num_workers: int
) -> List[str]:
    """Deterministic round-robin file assignment."""
    if num_workers <= 1:
        return files
    return [f for i, f in enumerate(files) if i % num_workers == worker]


# ---------------------------------------------------------------------------
# Line streaming
# ---------------------------------------------------------------------------

def _read_records(files: List[str]):
    """Yield (file_path, line_number, record_dict) for every valid JSONL line."""
    for fp in files:
        with open(fp, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError as exc:
                    log.warning("Skipping malformed JSON at %s:%d — %s", fp, lineno, exc)
                    continue
                prompt = rec.get("prompt", "").strip()
                answer = rec.get("answer", "").strip()
                if not prompt or not answer:
                    log.warning("Skipping record at %s:%d — missing prompt or answer", fp, lineno)
                    continue
                yield fp, lineno, rec


# ---------------------------------------------------------------------------
# Tokenize a single GRPO prompt
# ---------------------------------------------------------------------------

def tokenize_prompt(
    rec: Dict[str, Any],
    recipe: TrainingRecipe,
    tokenizer,
    eos_id: int,
    max_len: int = 0,
) -> List[int]:
    """
    Tokenize the prompt portion of a GRPO record.

    The tokenized sequence is:
        [user_turn] [assistant_turn_prefix] [EOS]

    The assistant turn prefix (e.g. "<|im_start|>assistant\n")
    is appended to guide the model into assistant generation mode.
    The EOS marks the end of the prompt; the model generates from there.

    IMPORTANT: the prompt format must match the EXACT same ChatML template
    used by pack_sft.py and seen by the model during SFT training. The model
    only knows how to produce well-formed <think>...</think> completions when
    it sees a prompt in the format it was fine-tuned on. Using an abbreviated
    template like "user\n{prompt}\nassistant\n" (without im_start/im_end tags)
    would be a different, unseen prompt shape — GRPO rollouts would sample
    the model off-distribution, corrupting the reward signal from the very
    first step.

    If max_len > 0, the prompt tokens are truncated to max_len before
    the EOS separator is appended.

    Returns:
        List of token ids.
    """
    prompt = rec.get("prompt", "").strip()

    # Format user turn using recipe template (e.g. <|im_start|>user\n{prompt}<|im_end|>\n)
    user_text = recipe.format_user_turn(prompt)

    # Append assistant turn prefix — this signals the model to start generating
    # an assistant response. No body is included; the model fills that in.
    assistant_prefix = recipe.turn_prefix_assistant  # e.g. "<|im_start|>assistant\n"

    full_text = user_text + assistant_prefix

    token_ids = tokenizer.encode(full_text).ids

    # Truncate if necessary
    if max_len > 0:
        token_ids = token_ids[:max_len]

    # NOTE: no uint16 clamping here — tokens are written as uint32 so the
    # full vocab is representable (silently clipping token IDs >= 65536 would
    # corrupt data for any tokenizer with a vocab larger than 65k).

    # Append EOS separator
    token_ids.append(eos_id)

    return token_ids


# ---------------------------------------------------------------------------
# Two-pass packing
# ---------------------------------------------------------------------------

def count_tokens(files: List[str], recipe: TrainingRecipe, tokenizer, eos_id: int, max_len: int = 0) -> Tuple[int, List[int]]:
    """
    Pass 1: count total tokens and per-record lengths.

    Returns:
        (total_tokens, per_record_lengths)
    """
    per_record: List[int] = []
    for _fp, _ln, rec in _read_records(files):
        token_ids = tokenize_prompt(rec, recipe, tokenizer, eos_id, max_len)
        per_record.append(len(token_ids))
    total = sum(per_record)
    return total, per_record


def write_shard(
    files: List[str],
    recipe: TrainingRecipe,
    tokenizer,
    eos_id: int,
    tokens_path: str,
    answers_path: str,
    total_tokens: int,
    record_indices: Optional[List[int]] = None,
    max_len: int = 0,
) -> Tuple[int, int]:
    """
    Pass 2: write prompt tokens in the length-prefixed uint32 format and
    save answer strings to a JSON list.

    Each record is a 4-byte little-endian uint32 length prefix followed by
    that many uint32 token IDs — the exact format consumed by
    ``PackedGRPODataLoader._load_bin`` (train_grpo.py). This matches the
    reference writer in train_grpo's smoke test.

    If *record_indices* is provided, only those records are written.

    Returns:
        (records_written, total_answers_written)
    """
    answers: List[str] = []

    want = set(record_indices) if record_indices is not None else None
    records_written = 0
    record_counter = 0
    tokens_written = 0

    with open(tokens_path, "wb") as f:
        for _fp, _ln, rec in _read_records(files):
            if want is not None and record_counter not in want:
                record_counter += 1
                continue

            token_ids = tokenize_prompt(rec, recipe, tokenizer, eos_id, max_len)
            f.write(len(token_ids).to_bytes(4, "little"))
            f.write(np.asarray(token_ids, dtype=np.uint32).tobytes())
            tokens_written += len(token_ids)
            # The loader (PackedGRPODataLoader._load_json) expects a list of
            # dicts with "answer" + "want_thinking" keys — NOT a bare string
            # list. Writing dicts keeps records aligned 1:1 with the .bin.
            answers.append({
                "answer": rec.get("answer", ""),
                "want_thinking": rec.get("want_thinking", True),
            })
            records_written += 1
            record_counter += 1

    assert tokens_written == total_tokens, (
        f"Token count mismatch: expected {total_tokens}, wrote {tokens_written}"
    )

    # Write answer strings
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False)

    return records_written, len(answers)


# ---------------------------------------------------------------------------
# Train / val split (deterministic interleaving)
# ---------------------------------------------------------------------------

def _is_val(record_idx: int, val_fraction: float) -> bool:
    """Deterministic train/val split: every Nth record goes to val."""
    if val_fraction <= 0:
        return False
    period = max(1, round(1.0 / val_fraction))
    return (record_idx % period) == 0


def split_by_record(
    per_record_lengths: List[int], val_fraction: float
) -> Tuple[List[int], List[int]]:
    """Deterministic interleaving split: every Nth record goes to val."""
    train_idx = []
    val_idx = []
    for i in range(len(per_record_lengths)):
        if _is_val(i, val_fraction):
            val_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pack_grpo(args: argparse.Namespace) -> None:
    """Main GRPO packing routine."""
    t0 = time.time()

    # --- Recipe ---
    recipe = recipe_from_args(args)
    log.info("Recipe mode: %s", recipe.mode)

    # --- Tokenizer ---
    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()

    # --- EOS token id ---
    eos_id = vocab_size - 1  # fallback
    try:
        tid = tokenizer.token_to_id(recipe.eos_token)
        if tid is not None:
            eos_id = tid
    except Exception:
        pass
    log.info("EOS token id: %d ('%s')", eos_id, recipe.eos_token)

    # --- Discover files ---
    all_files = discover_jsonl_files(args.data_dir)
    log.info("Found %d .jsonl files under %s", len(all_files), args.data_dir)

    # --- Worker assignment ---
    files = assign_files_to_worker(all_files, args.worker, args.num_workers)
    log.info("Worker %d/%d assigned %d files", args.worker, args.num_workers, len(files))

    # --- Max length per example ---
    max_len = args.max_len_per_example

    # --- Output directory ---
    os.makedirs(args.cache_dir, exist_ok=True)

    # --- Pass 1: count ---
    log.info("Pass 1: counting tokens ...")
    total_tokens, per_record_lengths = count_tokens(files, recipe, tokenizer, eos_id, max_len)
    num_records = len(per_record_lengths)
    log.info("Total tokens: %d (%d records)", total_tokens, num_records)

    if total_tokens == 0:
        log.error("No tokens found — check your data files.")
        sys.exit(1)

    # --- Train / val split ---
    val_fraction = args.val_fraction
    if val_fraction > 0 and args.num_workers > 1 and args.worker != 0:
        val_fraction = 0.0
        log.info("Worker %d: skipping val split (only worker 0 splits)", args.worker)

    if val_fraction > 0:
        train_idx, val_idx = split_by_record(per_record_lengths, val_fraction)
        train_tokens = sum(per_record_lengths[i] for i in train_idx)
        val_tokens = sum(per_record_lengths[i] for i in val_idx)
        log.info("Split: %d train records (%d tokens), %d val records (%d tokens)",
                 len(train_idx), train_tokens, len(val_idx), val_tokens)
    else:
        train_idx = list(range(num_records))
        val_idx = []
        train_tokens = total_tokens
        val_tokens = 0

    # --- Pass 2: write ---
    suffix = f"w{args.worker}-of-{args.num_workers}"

    # Write training shard
    log.info("Pass 2: writing training shard ...")
    train_tok_path = os.path.join(args.cache_dir, f"grpo_prompt_tokens.{suffix}.bin")
    train_ans_path = os.path.join(args.cache_dir, f"grpo_answers.{suffix}.json")
    n_written, n_answers = write_shard(
        files, recipe, tokenizer, eos_id,
        train_tok_path, train_ans_path, train_tokens,
        record_indices=train_idx,
        max_len=max_len,
    )

    # Write training manifest
    manifest = {
        "total_tokens": int(train_tokens),
        "num_prompts": n_written,
        "num_answers": n_answers,
        "vocab_size": vocab_size,
        "eos_token_id": eos_id,
        "dtype": "uint32",
        "format": "length-prefixed-uint32",
        "mode": recipe.mode,
        "worker": args.worker,
        "num_workers": args.num_workers,
        "split": "train",
        "token_file": os.path.basename(train_tok_path),
        "answers_file": os.path.basename(train_ans_path),
    }
    manifest_path = os.path.join(args.cache_dir, f"grpo_manifest.{suffix}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Wrote manifest: %s", manifest_path)

    # Write validation shard (if requested)
    if val_fraction > 0 and val_tokens > 0 and val_idx:
        log.info("Pass 2: writing validation shard ...")
        val_tok_path = os.path.join(args.cache_dir, f"grpo_val_prompt_tokens.{suffix}.bin")
        val_ans_path = os.path.join(args.cache_dir, f"grpo_val_answers.{suffix}.json")
        n_written_val, n_answers_val = write_shard(
            files, recipe, tokenizer, eos_id,
            val_tok_path, val_ans_path, val_tokens,
            record_indices=val_idx,
            max_len=max_len,
        )

        # Update manifest with val files
        manifest["val_tokens"] = int(val_tokens)
        manifest["val_prompts"] = n_written_val
        manifest["val_answers"] = n_answers_val
        manifest["val_token_file"] = os.path.basename(val_tok_path)
        manifest["val_answers_file"] = os.path.basename(val_ans_path)
        manifest["val_fraction"] = val_fraction

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        log.info("Updated manifest with val files: %s", manifest_path)

    elapsed = time.time() - t0
    log.info("Done in %.1fs", elapsed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pack {prompt, answer} JSONL into memmap + answer strings for GRPO training.",
    )
    p.add_argument(
        "--data-dir", required=True,
        help="Directory tree containing .jsonl files with 'prompt' and 'answer' fields.",
    )
    p.add_argument(
        "--tokenizer", required=True,
        help="Path to tokenizer directory or tokenizer.json file.",
    )
    p.add_argument(
        "--cache-dir", default="./packed",
        help="Output directory for packed files.",
    )
    p.add_argument(
        "--val-fraction", type=float, default=0.0,
        help="Fraction of records to hold out for validation (0 = no split).",
    )
    p.add_argument(
        "--worker", type=int, default=0,
        help="Worker index for multi-process packing (0-indexed).",
    )
    p.add_argument(
        "--num-workers", type=int, default=1,
        help="Total number of parallel workers.",
    )
    p.add_argument(
        "--max-len-per-example", type=int, default=2048,
        help="Max tokens per individual GRPO example before truncation.",
    )
    p.add_argument(
        "--seq-length", type=int, default=None,
        help="Shorthand to set --max-len-per-example to a specific value. "
             "Useful for short-context training (e.g. 256, 512). "
             "Overrides --max-len-per-example when both are given.",
    )
    add_recipe_args(p)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # --seq-length overrides --max-len-per-example
    if args.seq_length is not None:
        log.info("--seq-length %d overrides --max-len-per-example %d",
                 args.seq_length, args.max_len_per_example)
        args.max_len_per_example = args.seq_length

    if args.worker >= args.num_workers:
        log.error("--worker (%d) must be < --num-workers (%d)", args.worker, args.num_workers)
        sys.exit(1)

    if not (0.0 <= args.val_fraction < 1.0):
        log.error("--val-fraction must be in [0, 1)")
        sys.exit(1)

    pack_grpo(args)
