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
    """Recursively find all .jsonl files under *data_dir*, sorted for determinism."""
    patterns = [
        os.path.join(data_dir, "**", "*.jsonl"),
        os.path.join(data_dir, "**", "*.jsonl.gz"),
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    # Fallback: non-recursive
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    if not files:
        raise FileNotFoundError(
            f"No .jsonl files found under {data_dir}. "
            "Each line should be a JSON object with a 'text' field."
        )
    files.sort()
    return files


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

def count_tokens(files: List[str], tokenizer, eos_id: int) -> Tuple[int, List[int]]:
    """
    Pass 1: count total tokens across all files.

    Returns:
        (total_tokens, per_record_lengths) — per_record_lengths is
        a list of ints giving the token count (including EOS) of each record.
    """
    per_record: List[int] = []
    for _fp, _ln, rec in _read_lines(files):
        text = rec["text"]
        ids = tokenizer.encode(text).ids
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
) -> None:
    """
    Pass 2: write token ids into a pre-allocated uint16 memmap.
    """
    log.info("Allocating memmap %s (%d tokens, %.1f MiB)",
             memmap_path, total_tokens, total_tokens * 2 / (1024 ** 2))
    mmap = np.memmap(memmap_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))

    offset = 0
    records_written = 0
    for _fp, _ln, rec in _read_lines(files):
        text = rec["text"]
        ids = tokenizer.encode(text).ids
        # Clamp to uint16 range — tokens > 65535 are truncated (safe for most vocab sizes)
        safe_ids = [min(tid, 65535) for tid in ids]
        # Append EOS separator
        safe_ids.append(eos_id)
        n = len(safe_ids)
        mmap[offset: offset + n] = np.array(safe_ids, dtype=np.uint16)
        offset += n
        records_written += 1

    assert offset == total_tokens, (
        f"Token count mismatch: expected {total_tokens}, wrote {offset}"
    )
    mmap.flush()
    log.info("Wrote %d tokens from %d records to %s", total_tokens, records_written, memmap_path)


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def split_by_record(
    per_record_lengths: List[int], val_fraction: float
) -> Tuple[List[int], List[int]]:
    """
    Split record indices into train / val sets.

    Uses a contiguous tail split: the last ceil(N * val_fraction) records
    become the validation set (deterministic, preserves ordering).
    """
    n = len(per_record_lengths)
    n_val = max(1, math.ceil(n * val_fraction)) if val_fraction > 0 else 0
    n_train = n - n_val
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n))
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
    log.info("Pass 1: counting tokens ...")
    total_tokens, per_record_lengths = count_tokens(files, tokenizer, eos_id)
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
    worker_tag = f".w{args.worker}" if args.num_workers > 1 else ""

    if val_fraction > 0 and val_tokens > 0:
        # Write train shard
        log.info("Pass 2: writing training tokens ...")
        train_bin = os.path.join(args.cache_dir, f"pretrain_tokens{worker_tag}_train.bin")
        write_tokens_train(files, tokenizer, eos_id, train_bin, train_tokens, train_idx, per_record_lengths)

        # Write val shard
        log.info("Pass 2: writing validation tokens ...")
        val_bin = os.path.join(args.cache_dir, f"pretrain_tokens{worker_tag}_val.bin")
        write_tokens_val(files, tokenizer, eos_id, val_bin, val_tokens, val_idx, per_record_lengths)

        # Meta for train
        meta_train = {
            "total_tokens": train_tokens,
            "num_records": len(train_idx),
            "vocab_size": vocab_size,
            "eos_token_id": eos_id,
            "dtype": "uint16",
            "worker": args.worker,
            "num_workers": args.num_workers,
            "split": "train",
        }
        meta_path_train = os.path.join(args.cache_dir, f"meta{worker_tag}_train.json")
        with open(meta_path_train, "w") as f:
            json.dump(meta_train, f, indent=2)
        log.info("Wrote meta: %s", meta_path_train)

        # Meta for val
        meta_val = {
            "total_tokens": val_tokens,
            "num_records": len(val_idx),
            "vocab_size": vocab_size,
            "eos_token_id": eos_id,
            "dtype": "uint16",
            "worker": args.worker,
            "num_workers": args.num_workers,
            "split": "val",
        }
        meta_path_val = os.path.join(args.cache_dir, f"meta{worker_tag}_val.json")
        with open(meta_path_val, "w") as f:
            json.dump(meta_val, f, indent=2)
        log.info("Wrote meta: %s", meta_path_val)
    else:
        # No split — write single shard
        log.info("Pass 2: writing tokens ...")
        bin_path = os.path.join(args.cache_dir, f"pretrain_tokens{worker_tag}.bin")
        write_tokens(files, tokenizer, eos_id, bin_path, total_tokens, per_record_lengths)

        meta = {
            "total_tokens": total_tokens,
            "num_records": len(per_record_lengths),
            "vocab_size": vocab_size,
            "eos_token_id": eos_id,
            "dtype": "uint16",
            "worker": args.worker,
            "num_workers": args.num_workers,
            "split": "all",
        }
        meta_path = os.path.join(args.cache_dir, f"meta{worker_tag}.json")
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
) -> None:
    """Write only training-split records into a pre-allocated memmap."""
    mmap = np.memmap(memmap_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))
    train_set = set(train_idx)
    offset = 0
    record_counter = 0
    for _fp, _ln, rec in _read_lines(files):
        if record_counter in train_set:
            text = rec["text"]
            ids = tokenizer.encode(text).ids
            safe_ids = [min(tid, 65535) for tid in ids]
            safe_ids.append(eos_id)
            n = len(safe_ids)
            mmap[offset: offset + n] = np.array(safe_ids, dtype=np.uint16)
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
) -> None:
    """Write only validation-split records into a pre-allocated memmap."""
    mmap = np.memmap(memmap_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))
    val_set = set(val_idx)
    offset = 0
    record_counter = 0
    for _fp, _ln, rec in _read_lines(files):
        if record_counter in val_set:
            text = rec["text"]
            ids = tokenizer.encode(text).ids
            safe_ids = [min(tid, 65535) for tid in ids]
            safe_ids.append(eos_id)
            n = len(safe_ids)
            mmap[offset: offset + n] = np.array(safe_ids, dtype=np.uint16)
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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.worker >= args.num_workers:
        log.error("--worker (%d) must be < --num-workers (%d)", args.worker, args.num_workers)
        sys.exit(1)

    if not (0.0 <= args.val_fraction < 1.0):
        log.error("--val-fraction must be in [0, 1)")
        sys.exit(1)

    pack_pretrain(args)
