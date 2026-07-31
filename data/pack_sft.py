#!/usr/bin/env python3
"""
data/pack_sft.py

Stage 0 of SFT: read raw JSONL SFT records ({prompt, thinking, answer}),
apply the recipe-aware ChatML template, tokenise, and write the result to
packed memmap .bin files that train_sft.py reads directly.

Supports reasoning, non_reasoning, and hybrid modes via TrainingRecipe.

Template (reasoning/hybrid with thinking):
    <|im_start|>user
    {prompt}<|im_end|>
    <|im_start|>assistant
    <think>
    {thinking}
    </think>
    {answer}<|im_end|>

Output layout (per worker):
    <cache_dir>/sft_train_tokens.w{worker}-of-{num_workers}.bin
    <cache_dir>/sft_train_mask.w{worker}-of-{num_workers}.bin
    <cache_dir>/sft_val_tokens.w{worker}-of-{num_workers}.bin
    <cache_dir>/sft_val_mask.w{worker}-of-{num_workers}.bin
    <cache_dir>/sft_manifest.w{worker}-of-{num_workers}.json

train_sft.py discovers every worker's manifest and concatenates shards
via mmap (no RAM copy) at load time.

Usage:
    # Single process
    python data/pack_sft.py --data-dir ./data --tokenizer ./tokenizer \\
        --cache-dir ./sft_packed

    # Multi-worker parallel
    python data/pack_sft.py --worker 0 --num-workers 4 --data-dir ./data \\
        --tokenizer ./tokenizer --cache-dir ./sft_packed
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
from recipe import TrainingRecipe, recipe_from_args, add_recipe_args  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[pack_sft] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Special token helpers
# ---------------------------------------------------------------------------

def get_special_token_id(tok, token: str) -> int:
    """Look up a special token id, with fallback to vocab_size - 1."""
    tid = tok.token_to_id(token)
    if tid is not None:
        return tid
    added = tok.get_added_vocabulary()
    if token in added:
        return added[token]
    return tok.get_vocab_size() - 1


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_jsonl_files(data_dir: str) -> List[str]:
    """Discover every JSONL shard under <data_dir>/<category>/*.jsonl."""
    all_shards = sorted(glob.glob(os.path.join(data_dir, "*", "*.jsonl")))
    if not all_shards:
        all_shards = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    if not all_shards:
        raise FileNotFoundError(
            f"No .jsonl shards found under {data_dir}/<category>/. "
            "Expected JSONL with 'prompt', 'thinking' (optional), and 'answer' fields."
        )
    return all_shards


def assign_files_to_worker(files: List[str], worker: int, num_workers: int) -> List[str]:
    """Deterministic round-robin file assignment."""
    if num_workers <= 1:
        return files
    return [f for i, f in enumerate(files) if i % num_workers == worker]


# ---------------------------------------------------------------------------
# Line streaming
# ---------------------------------------------------------------------------

def _safe_load_line(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _read_records(files: List[str]):
    """Yield (file_path, record_dict) for every valid JSONL line."""
    for fp in files:
        with open(fp, "r", encoding="utf-8") as fh:
            for raw in fh:
                rec = _safe_load_line(raw)
                if rec is None:
                    continue
                prompt = rec.get("prompt", "").strip()
                answer = rec.get("answer", "").strip()
                if not prompt or not answer:
                    continue
                yield fp, rec


# ---------------------------------------------------------------------------
# Tokenize a single SFT record via recipe
# ---------------------------------------------------------------------------

def tokenize_record(
    rec: Dict[str, Any],
    recipe: TrainingRecipe,
    tokenizer,
    eos_id: int,
    max_len: int = 0,
) -> Optional[Tuple[List[int], List[int]]]:
    """
    Tokenize one SFT record into (token_ids, loss_mask).

    Layout:
        [user_turn_tokens] [assistant_turn_tokens] [EOS]
        [      0 ... 0   ] [      1 ... 1        ] [ 0 ]

    Returns None if the formatted example is empty or exceeds max_len.
    """
    prompt = rec.get("prompt", "").strip()
    thinking = rec.get("thinking", "").strip()
    answer = rec.get("answer", "").strip()
    want_thinking = rec.get("want_thinking", None)

    if not prompt or not answer:
        return None

    # Format turns using recipe templates
    user_text = recipe.format_user_turn(prompt)
    assistant_text = recipe.format_assistant_turn(
        thinking=thinking, answer=answer, want_thinking=want_thinking,
    )

    # Tokenize each turn separately
    user_ids = tokenizer.encode(user_text).ids
    assistant_ids = tokenizer.encode(assistant_text).ids

    # NOTE: no uint16 clamp here. The on-disk dtype is chosen from
    # ``vocab_size`` (uint16 when vocab <= 65536, else uint32), so clamping
    # to 65535 would silently corrupt token ids above that range for large
    # vocabs. Writing the full ids relies on the caller's dtype selection.

    # Truncate if necessary, preserving at least a few answer tokens
    if max_len > 0:
        total = len(user_ids) + len(assistant_ids)
        if total > max_len:
            budget = max_len - len(user_ids)
            if budget < 32:
                user_ids = user_ids[:max_len - min(32, len(assistant_ids))]
                assistant_ids = assistant_ids[:32]
            else:
                assistant_ids = assistant_ids[:budget]

    # Build token sequence: user + assistant + EOS
    token_ids = user_ids + assistant_ids + [eos_id]
    loss_mask = [0] * len(user_ids) + [1] * len(assistant_ids) + [0]

    if len(token_ids) < 4:
        return None

    assert len(token_ids) == len(loss_mask), (
        f"Length mismatch: {len(token_ids)} tokens vs {len(loss_mask)} masks"
    )

    return token_ids, loss_mask


# ---------------------------------------------------------------------------
# Train / val split (deterministic interleaving)
# ---------------------------------------------------------------------------

def _is_val(record_idx: int, val_fraction: float) -> bool:
    """Deterministic train/val split: every Nth record goes to val."""
    if val_fraction <= 0:
        return False
    period = max(1, round(1.0 / val_fraction))
    return (record_idx % period) == 0


# ---------------------------------------------------------------------------
# Pack: stream JSONL -> tokenise -> write memmap .bin files for this worker
# ---------------------------------------------------------------------------

def pack_worker_shard(
    data_dir: str,
    tokenizer,
    cache_dir: str,
    max_len_per_example: int,
    val_fraction: float,
    worker: int,
    num_workers: int,
    recipe: Optional[TrainingRecipe] = None,
    vocab_size: Optional[int] = None,
) -> dict:
    """
    Stream JSONL records assigned to this worker, tokenise, and write to
    this worker's memmap files. Peak RAM is constant regardless of dataset size.

    Returns the manifest dict that was also written to disk.
    """
    os.makedirs(cache_dir, exist_ok=True)

    shard_paths = assign_files_to_worker(
        discover_jsonl_files(data_dir), worker, num_workers
    )
    all_total = len(glob.glob(os.path.join(data_dir, "*", "*.jsonl")))
    log.info("Worker %d/%d: %d shard(s) assigned out of %d total",
             worker + 1, num_workers, len(shard_paths), all_total)

    if not shard_paths:
        log.warning("Worker %d/%d: no shards assigned — writing empty output",
                     worker + 1, num_workers)

    if recipe is None:
        recipe = TrainingRecipe(mode="reasoning")

    eos_id = get_special_token_id(tokenizer, recipe.eos_token)
    vocab_size = vocab_size or tokenizer.get_vocab_size()
    dtype_t = np.uint16 if vocab_size <= 65536 else np.uint32
    dtype_m = np.uint8

    suffix = f"w{worker}-of-{num_workers}"
    train_tok_path = os.path.join(cache_dir, f"sft_train_tokens.{suffix}.bin")
    train_mask_path = os.path.join(cache_dir, f"sft_train_mask.{suffix}.bin")
    val_tok_path = os.path.join(cache_dir, f"sft_val_tokens.{suffix}.bin")
    val_mask_path = os.path.join(cache_dir, f"sft_val_mask.{suffix}.bin")
    manifest_path = os.path.join(cache_dir, f"sft_manifest.{suffix}.json")

    # ---- first pass: count tokens so we can pre-allocate mmaps
    t0 = time.time()
    total_train = 0
    total_val = 0
    n_records = 0

    for _fp, rec in _read_records(shard_paths):
        result = tokenize_record(rec, recipe, tokenizer, eos_id, max_len=max_len_per_example)
        if result is None:
            continue
        ids, _ = result
        n_tok = len(ids)
        if _is_val(n_records, val_fraction):
            total_val += n_tok
        else:
            total_train += n_tok
        n_records += 1

    log.info("Worker %d/%d: counted %d records in %.1fs",
             worker + 1, num_workers, n_records, time.time() - t0)
    log.info("Worker %d/%d: train tokens: %d  val tokens: %d",
             worker + 1, num_workers, total_train, total_val)

    # ---- allocate memmap files on disk (no RAM)
    if total_train > 0:
        np.memmap(train_tok_path, dtype=dtype_t, mode="w+", shape=(total_train,))
        np.memmap(train_mask_path, dtype=dtype_m, mode="w+", shape=(total_train,))
        train_tok = np.memmap(train_tok_path, dtype=dtype_t, mode="r+")
        train_mask = np.memmap(train_mask_path, dtype=dtype_m, mode="r+")
    else:
        train_tok = None
        train_mask = None

    if total_val > 0:
        np.memmap(val_tok_path, dtype=dtype_t, mode="w+", shape=(total_val,))
        np.memmap(val_mask_path, dtype=dtype_m, mode="w+", shape=(total_val,))
        val_tok = np.memmap(val_tok_path, dtype=dtype_t, mode="r+")
        val_mask = np.memmap(val_mask_path, dtype=dtype_m, mode="r+")
    else:
        val_tok = None
        val_mask = None

    # ---- second pass: write tokens + masks directly into the mmaps
    train_ptr = 0
    val_ptr = 0
    n_records = 0
    last_print = time.time()

    for _fp, rec in _read_records(shard_paths):
        result = tokenize_record(rec, recipe, tokenizer, eos_id, max_len=max_len_per_example)
        if result is None:
            continue

        ids, lmask = result
        tok_arr = np.array(ids, dtype=dtype_t)
        mask_arr = np.array(lmask, dtype=dtype_m)

        if _is_val(n_records, val_fraction):
            n = len(tok_arr)
            if val_tok is not None and val_mask is not None:
                val_tok[val_ptr:val_ptr + n] = tok_arr
                val_mask[val_ptr:val_ptr + n] = mask_arr
                val_ptr += n
        else:
            n = len(tok_arr)
            if train_tok is not None and train_mask is not None:
                train_tok[train_ptr:train_ptr + n] = tok_arr
                train_mask[train_ptr:train_ptr + n] = mask_arr
                train_ptr += n

        n_records += 1
        if time.time() - last_print > 5:
            log.info("Worker %d/%d: packing … %d records written",
                     worker + 1, num_workers, n_records)
            last_print = time.time()

    if train_tok is not None:
        train_tok.flush()
        train_mask.flush()
    if val_tok is not None:
        val_tok.flush()
        val_mask.flush()

    log.info("Worker %d/%d: packed %d records in %.1fs total",
             worker + 1, num_workers, n_records, time.time() - t0)

    manifest = {
        "worker": worker,
        "num_workers": num_workers,
        "n_records": n_records,
        "train_tokens": total_train,
        "val_tokens": total_val,
        "dtype_t": str(np.dtype(dtype_t)),
        "dtype_m": str(np.dtype(dtype_m)),
        "vocab_size": vocab_size,
        "max_len_per_example": max_len_per_example,
        "val_fraction": val_fraction,
        "recipe_mode": recipe.mode,
        "train_tokens_file": os.path.basename(train_tok_path) if total_train > 0 else None,
        "train_mask_file": os.path.basename(train_mask_path) if total_train > 0 else None,
        "val_tokens_file": os.path.basename(val_tok_path) if total_val > 0 else None,
        "val_mask_file": os.path.basename(val_mask_path) if total_val > 0 else None,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Worker %d/%d: wrote manifest to %s", worker + 1, num_workers, manifest_path)

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pack SFT JSONL into memmap .bin + loss-mask .bin with recipe-aware formatting.",
    )
    p.add_argument(
        "--data-dir", required=True,
        help="Directory tree containing .jsonl files with 'prompt', 'thinking', 'answer'.",
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
        "--val-fraction", type=float, default=0.01,
        help="Fraction of records for validation (default 0.01 = 1%%).",
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
        help="Max tokens per individual SFT example before truncation.",
    )
    p.add_argument(
        "--seq-length", type=int, default=None,
        help="Shorthand to set --max-len-per-example to a specific value.",
    )
    add_recipe_args(p)
    return p.parse_args()


def pack_sft(args: argparse.Namespace) -> None:
    """Main SFT packing routine."""
    t0 = time.time()

    # Recipe
    recipe = recipe_from_args(args)
    log.info("Recipe mode: %s", recipe.mode)
    log.info("Special tokens: %s", recipe.special_tokens)

    # Tokenizer
    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()

    # Max length
    max_len = args.seq_length if args.seq_length is not None else args.max_len_per_example
    log.info("Max tokens per example: %d", max_len)

    if args.worker >= args.num_workers:
        log.error("--worker (%d) must be < --num-workers (%d)", args.worker, args.num_workers)
        sys.exit(1)

    pack_worker_shard(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        cache_dir=args.cache_dir,
        max_len_per_example=max_len,
        val_fraction=args.val_fraction,
        worker=args.worker,
        num_workers=args.num_workers,
        recipe=recipe,
        vocab_size=vocab_size,
    )

    elapsed = time.time() - t0
    log.info("Done in %.1fs", elapsed)
    if args.num_workers > 1:
        log.info("Run the remaining %d worker(s) before training, "
                 "then point train_sft.py --cache-dir at the same directory.",
                 args.num_workers - 1)


if __name__ == "__main__":
    args = parse_args()
    pack_sft(args)
