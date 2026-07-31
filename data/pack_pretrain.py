#!/usr/bin/env python3
"""
data/pack_pretrain.py

Pack raw text JSONL into uint16 memmap .bin for pretraining.
Each record is tokenized and appended with an EOS separator.

Usage:
    python data/pack_pretrain.py --data-dir ./data --tokenizer ./tokenizer --cache-dir ./packed
    python data/pack_pretrain.py --data-dir ./data --worker 0 --num-workers 4  # parallel
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
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup so we can import from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tokenizer_train import load_tokenizer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[pack_pretrain] %(asctime)s %(levelname)s %(message)s",
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
            "Each line should be a JSON object with a 'text' field."
        )
    return all_shards


def assign_files_to_worker(
    files: List[str], worker: int, num_workers: int
) -> List[str]:
    """Deterministically assign files to a worker via round-robin (index % num_workers)."""
    if num_workers <= 1:
        return files
    return [f for i, f in enumerate(files) if i % num_workers == worker]


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _read_lines(files: List[str]):
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
                text = rec.get("text", "")
                if not text:
                    continue
                yield fp, lineno, rec


# ---------------------------------------------------------------------------
# Two-pass packing
# ---------------------------------------------------------------------------

def count_tokens(files: List[str], tokenizer, eos_id: int, max_seq_len: int = 0) -> Tuple[int, List[int]]:
    """
    Pass 1: count total tokens across all files.

    If max_seq_len > 0, each document's tokens are truncated to max_seq_len
    before the EOS separator is added.

    Returns:
        (total_tokens, per_record_lengths) — per_record_lengths is
        a list of ints giving the token count (including EOS) of each record.
    """
    per_record: List[int] = []
    for _fp, _ln, rec in _read_lines(files):
        text = rec["text"]
        ids = tokenizer.encode(text).ids
        if max_seq_len > 0:
            ids = ids[:max_seq_len]
        # +1 for the EOS separator after each record
        per_record.append(len(ids) + 1)
    total = sum(per_record)
    return total, per_record


def write_tokens(
    files: List[str],
    tokenizer,
    eos_id: int,
    memmap_path: str,
    total_tokens: int,
    per_record_lengths: List[int],
    max_seq_len: int = 0,
    dtype_t: np.dtype = np.uint16,
) -> None:
    """
    Pass 2: write token ids into a pre-allocated memmap.

    ``dtype_t`` is ``np.uint32`` for vocabs > 65536 (set by the caller from
    ``vocab_size``); token ids are never clamped, so ids above 65535 survive.
    """
    log.info("Allocating memmap %s (%d tokens, %.1f MiB, %s)",
             memmap_path, total_tokens,
             total_tokens * dtype_t.itemsize / (1024 ** 2),
             np.dtype(dtype_t).name)
    mmap = np.memmap(memmap_path, dtype=dtype_t, mode="w+", shape=(total_tokens,))

    offset = 0
    records_written = 0
    for _fp, _ln, rec in _read_lines(files):
        text = rec["text"]
        ids = tokenizer.encode(text).ids
        if max_seq_len > 0:
            ids = ids[:max_seq_len]
        # Append EOS separator (no uint16 clamp — see dtype_t above)
        ids = ids + [eos_id]
        n = len(ids)
        mmap[offset: offset + n] = np.array(ids, dtype=dtype_t)
        offset += n
        records_written += 1

    assert offset == total_tokens, (
        f"Token count mismatch: expected {total_tokens}, wrote {offset}"
    )
    mmap.flush()
    log.info("Wrote %d tokens from %d records to %s", total_tokens, records_written, memmap_path)


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
    """
    Deterministic interleaving split: every Nth record goes to val.
    """
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

def pack_pretrain(args: argparse.Namespace) -> None:
    """Main packing routine."""
    t0 = time.time()

    # --- Tokenizer ---
    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    # Token ids are stored uint16 for vocabs <= 65536, else uint32 so ids
    # above 65535 are never truncated (the loader reads this from meta).
    dtype_t = np.uint16 if vocab_size <= 65536 else np.uint32
    dtype_name = np.dtype(dtype_t).name  # "uint16" | "uint32"
    # EOS token id: try recipe special tokens first, fallback to tokenizer's eos
    eos_id = vocab_size - 1  # convention: last token is EOS for BBPE
    # Try to get it from the tokenizer's special tokens map if available
    try:
        eos_id = tokenizer.token_to_id("</s>")
        if eos_id is None:
            eos_id = vocab_size - 1
    except Exception:
        pass
    log.info("Using EOS token id: %d (vocab_size=%d)", eos_id, vocab_size)

    # --- Discover files ---
    all_files = discover_jsonl_files(args.data_dir)
    log.info("Found %d .jsonl files under %s", len(all_files), args.data_dir)

    # --- Worker assignment ---
    files = assign_files_to_worker(all_files, args.worker, args.num_workers)
    log.info("Worker %d/%d assigned %d files", args.worker, args.num_workers, len(files))

    # --- Output directory ---
    os.makedirs(args.cache_dir, exist_ok=True)

    # --- Pass 1: count ---
    max_seq_len = getattr(args, "max_seq_len_pretrain", 0) or 0
    log.info("Pass 1: counting tokens ...")
    total_tokens, per_record_lengths = count_tokens(files, tokenizer, eos_id, max_seq_len)
    log.info("Total tokens: %d (%d records)", total_tokens, len(per_record_lengths))

    if total_tokens == 0:
        log.error("No tokens found — check your data files.")
        sys.exit(1)

    # --- Train / val split ---
    val_fraction = args.val_fraction
    if val_fraction > 0 and args.num_workers > 1:
        # In multi-worker mode, val split is only applied on worker 0
        # to avoid overlapping val sets. Other workers pack everything as "train".
        if args.worker != 0:
            val_fraction = 0.0
            log.info("Worker %d: skipping val split (only worker 0 splits)", args.worker)

    if val_fraction > 0:
        train_idx, val_idx = split_by_record(per_record_lengths, val_fraction)
        train_lengths = [per_record_lengths[i] for i in train_idx]
        val_lengths = [per_record_lengths[i] for i in val_idx]
        train_tokens = sum(train_lengths)
        val_tokens = sum(val_lengths)
        log.info("Split: %d train records (%d tokens), %d val records (%d tokens)",
                 len(train_idx), train_tokens, len(val_idx), val_tokens)
    else:
        train_lengths = per_record_lengths
        train_tokens = total_tokens
        val_lengths = []
        val_tokens = 0
        train_idx = list(range(len(per_record_lengths)))
        val_idx = []

    # --- Pass 2: write ---
    suffix = f"w{args.worker}-of-{args.num_workers}"

    if val_fraction > 0 and val_tokens > 0:
        # Write train shard
        log.info("Pass 2: writing training tokens ...")
        train_bin = os.path.join(args.cache_dir, f"pretrain_tokens.{suffix}_train.bin")
        write_tokens_train(files, tokenizer, eos_id, train_bin, train_tokens, train_idx, per_record_lengths, max_seq_len, dtype_t)

        # Write val shard
        log.info("Pass 2: writing validation tokens ...")
        val_bin = os.path.join(args.cache_dir, f"pretrain_tokens.{suffix}_val.bin")
        write_tokens_val(files, tokenizer, eos_id, val_bin, val_tokens, val_idx, per_record_lengths, max_seq_len, dtype_t)

        # Meta for train
        meta_train = {
            "total_tokens": train_tokens,
            "num_records": len(train_idx),
            "vocab_size": vocab_size,
            "eos_token_id": eos_id,
            "dtype": dtype_name,
            "worker": args.worker,
            "num_workers": args.num_workers,
            "split": "train",
        }
        meta_path_train = os.path.join(args.cache_dir, f"meta.{suffix}_train.json")
        with open(meta_path_train, "w") as f:
            json.dump(meta_train, f, indent=2)
        log.info("Wrote meta: %s", meta_path_train)

        # Meta for val
        meta_val = {
            "total_tokens": val_tokens,
            "num_records": len(val_idx),
            "vocab_size": vocab_size,
            "eos_token_id": eos_id,
            "dtype": dtype_name,
            "worker": args.worker,
            "num_workers": args.num_workers,
            "split": "val",
        }
        meta_path_val = os.path.join(args.cache_dir, f"meta.{suffix}_val.json")
        with open(meta_path_val, "w") as f:
            json.dump(meta_val, f, indent=2)
        log.info("Wrote meta: %s", meta_path_val)
    else:
        # No split — write single shard
        log.info("Pass 2: writing tokens ...")
        bin_path = os.path.join(args.cache_dir, f"pretrain_tokens.{suffix}.bin")
        write_tokens(files, tokenizer, eos_id, bin_path, total_tokens, per_record_lengths, max_seq_len, dtype_t)

        meta = {
            "total_tokens": total_tokens,
            "num_records": len(per_record_lengths),
            "vocab_size": vocab_size,
            "eos_token_id": eos_id,
            "dtype": dtype_name,
            "worker": args.worker,
            "num_workers": args.num_workers,
            "split": "all",
        }
        meta_path = os.path.join(args.cache_dir, f"meta.{suffix}.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        log.info("Wrote meta: %s", meta_path)

    elapsed = time.time() - t0
    log.info("Done in %.1fs", elapsed)


# ---------------------------------------------------------------------------
# Split-aware writers
# ---------------------------------------------------------------------------

def write_tokens_train(
    files: List[str],
    tokenizer,
    eos_id: int,
    memmap_path: str,
    total_tokens: int,
    train_idx: List[int],
    per_record_lengths: List[int],
    max_seq_len: int = 0,
    dtype_t: np.dtype = np.uint16,
) -> None:
    """Write only training-split records into a pre-allocated memmap.

    ``dtype_t`` is ``np.uint32`` for vocabs > 65536; ids are never clamped.
    """
    mmap = np.memmap(memmap_path, dtype=dtype_t, mode="w+", shape=(total_tokens,))
    train_set = set(train_idx)
    offset = 0
    record_counter = 0
    for _fp, _ln, rec in _read_lines(files):
        if record_counter in train_set:
            text = rec["text"]
            ids = tokenizer.encode(text).ids
            if max_seq_len > 0:
                ids = ids[:max_seq_len]
            ids = ids + [eos_id]
            n = len(ids)
            mmap[offset: offset + n] = np.array(ids, dtype=dtype_t)
            offset += n
        record_counter += 1

    assert offset == total_tokens, (
        f"Train token count mismatch: expected {total_tokens}, wrote {offset}"
    )
    mmap.flush()
    log.info("Wrote %d train tokens to %s", total_tokens, memmap_path)


def write_tokens_val(
    files: List[str],
    tokenizer,
    eos_id: int,
    memmap_path: str,
    total_tokens: int,
    val_idx: List[int],
    per_record_lengths: List[int],
    max_seq_len: int = 0,
    dtype_t: np.dtype = np.uint16,
) -> None:
    """Write only validation-split records into a pre-allocated memmap.

    ``dtype_t`` is ``np.uint32`` for vocabs > 65536; ids are never clamped.
    """
    mmap = np.memmap(memmap_path, dtype=dtype_t, mode="w+", shape=(total_tokens,))
    val_set = set(val_idx)
    offset = 0
    record_counter = 0
    for _fp, _ln, rec in _read_lines(files):
        if record_counter in val_set:
            text = rec["text"]
            ids = tokenizer.encode(text).ids
            if max_seq_len > 0:
                ids = ids[:max_seq_len]
            ids = ids + [eos_id]
            n = len(ids)
            mmap[offset: offset + n] = np.array(ids, dtype=dtype_t)
            offset += n
        record_counter += 1

    assert offset == total_tokens, (
        f"Val token count mismatch: expected {total_tokens}, wrote {offset}"
    )
    mmap.flush()
    log.info("Wrote %d val tokens to %s", total_tokens, memmap_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pack raw text JSONL into uint16 memmap .bin for pretraining.",
    )
    p.add_argument(
        "--data-dir", required=True,
        help="Directory tree containing .jsonl files with a 'text' field.",
    )
    p.add_argument(
        "--tokenizer", required=True,
        help="Path to tokenizer directory or tokenizer.json file.",
    )
    p.add_argument(
        "--cache-dir", default="./packed",
        help="Output directory for packed .bin and meta.json files.",
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
        help="Total number of parallel workers (files are split by index %% num_workers).",
    )
    p.add_argument(
        "--max-seq-len-pretrain", type=int, default=4096,
        help="Maximum tokens per document before appending the EOS separator. "
             "Documents are truncated to this many tokens, which means each "
             "training window starts at a clean document boundary. This helps "
             "RoPE learn position resets at EOS tokens. "
             "Should be >= training --seq-len to avoid wasted padding. "
             "Defaults to 4096.",
    )
    p.add_argument(
        "--seq-length", type=int, default=None,
        help="Shorthand to set --max-seq-len-pretrain to a specific value. "
             "Useful for short-context training (e.g. 256, 512). "
             "Overrides --max-seq-len-pretrain when both are given. "
             "If training with --seq-len N, set this to N or slightly higher "
             "so each training window contains one complete document.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # --seq-length overrides --max-seq-len-pretrain
    if args.seq_length is not None:
        log.info("--seq-length %d overrides --max-seq-len-pretrain %d",
                 args.seq_length, args.max_seq_len_pretrain)
        args.max_seq_len_pretrain = args.seq_length

    # Warn if pack cap is smaller than typical training seq_len
    if args.max_seq_len_pretrain < 1024:
        log.warning("--max-seq-len-pretrain=%d is small. Each document is "
                     "capped here, so training windows will start at document "
                     "boundaries. This helps RoPE learn position resets at EOS. "
                     "Make sure training --seq-len <= %d to avoid padding waste.",
                     args.max_seq_len_pretrain, args.max_seq_len_pretrain)

    if args.worker >= args.num_workers:
        log.error("--worker (%d) must be < --num-workers (%d)", args.worker, args.num_workers)
        sys.exit(1)

    if not (0.0 <= args.val_fraction < 1.0):
        log.error("--val-fraction must be in [0, 1)")
        sys.exit(1)

    pack_pretrain(args)
