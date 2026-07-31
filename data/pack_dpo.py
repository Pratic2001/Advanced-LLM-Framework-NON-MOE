#!/usr/bin/env python3
"""
data/pack_dpo.py

Pack DPO preference pairs JSONL into packed binary format for the DPO training
scripts (train_dpo.py / train_dpo_deepspeed.py).

Input JSONL format (one record per line):
    {"prompt": "...", "chosen": "...", "rejected": "...",
     "chosen_thinking": "...", "rejected_thinking": "..."}

The `thinking` fields are optional and describe the reasoning that led to the
answer (analogous to the GRPO/thinking format). If present, they are formatted
as <think>...</think> blocks inside the assistant turn.

Output files in --cache-dir:
    dpo_prompts.bin     — length-prefixed uint32 array of prompt token IDs
    dpo_chosen.bin      — length-prefixed uint32 array of chosen completion tokens
    dpo_rejected.bin    — length-prefixed uint32 array of rejected completion tokens
    dpo_prompt_lens*.json — per-record prompt token lengths (for completion masking)
    dpo_manifest.json   — metadata

Usage:
    # Basic
    python data/pack_dpo.py --data-dir ./dpo_data --tokenizer ./tokenizer \\
        --cache-dir ./dpo_packed

    # With recipe mode awareness
    python data/pack_dpo.py --data-dir ./dpo_data --tokenizer ./tokenizer \\
        --cache-dir ./dpo_packed --mode reasoning

    # Multi-worker
    python data/pack_dpo.py --data-dir ./dpo_data --tokenizer ./tokenizer \\
        --worker 0 --num-workers 4 --cache-dir ./dpo_packed &
    python data/pack_dpo.py --data-dir ./dpo_data --tokenizer ./tokenizer \\
        --worker 1 --num-workers 4 --cache-dir ./dpo_packed &
    ...

    # Generate preference pairs from GRPO-style data using Ollama judge
    python data/pack_dpo.py --data-dir ./grpo_data --tokenizer ./tokenizer \\
        --generate-pairs --ollama-url http://remote:11434 --gen-model qwen2.5:7b \\
        --judge-model qwen2.5:7b-instruct --num-candidates 4
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tokenizer_train import load_tokenizer  # noqa: E402
from recipe import TrainingRecipe, add_recipe_args, recipe_from_args  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[pack_dpo] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def discover_jsonl_files(data_dir: str) -> List[str]:
    """Recursively find all .jsonl files under data_dir, sorted."""
    patterns = [os.path.join(data_dir, "**", "*.jsonl")]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    if not files:
        raise FileNotFoundError(
            f"No .jsonl files found under {data_dir}. "
            "Expected JSONL with 'prompt', 'chosen', and 'rejected' fields."
        )
    files.sort()
    return files


def assign_files_to_worker(
    files: List[str], worker: int, num_workers: int,
) -> List[str]:
    """Deterministic round-robin file assignment."""
    if num_workers <= 1:
        return files
    return [f for i, f in enumerate(files) if i % num_workers == worker]


# ---------------------------------------------------------------------------
# Line streaming
# ---------------------------------------------------------------------------


def _read_records(files: List[str]):
    """
    Yield (file_path, line_number, record_dict) for every valid JSONL line
    that has the required 'prompt', 'chosen', 'rejected' fields.
    """
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
                chosen = rec.get("chosen", "").strip()
                rejected = rec.get("rejected", "").strip()
                if not prompt or not chosen or not rejected:
                    log.warning(
                        "Skipping record at %s:%d — missing prompt, chosen, or rejected",
                        fp, lineno,
                    )
                    continue
                yield fp, lineno, rec


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def tokenize_preference_triple(
    rec: Dict[str, Any],
    recipe: TrainingRecipe,
    tokenizer,
    eos_id: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Tokenize a single preference triple into (prompt_ids, chosen_ids, rejected_ids).

    The prompt is formatted as:  user_turn + assistant_prefix + EOS
    The chosen/rejected are the completion BODY ONLY (no prompt, no assistant
    prefix — the prompt is stored separately and concatenated at load time by
    PackedDPODataLoader as ``cat([prompt_ids, completion_ids])``).

    IMPORTANT: previously the packer wrote the FULL sequence (prompt + body)
    into the chosen/rejected fields while the loader ALSO concatenated the
    prompt — producing a duplicated prompt and a completion mask that covered
    prompt tokens. The body is now written alone so the loader's
    ``cat([pids, cids])`` yields exactly one prompt + one completion.

    Returns:
        (prompt_token_ids, chosen_completion_token_ids, rejected_completion_token_ids)
    """
    prompt = rec.get("prompt", "").strip()
    chosen = rec.get("chosen", "").strip()
    rejected = rec.get("rejected", "").strip()
    chosen_thinking = rec.get("chosen_thinking", "").strip()
    rejected_thinking = rec.get("rejected_thinking", "").strip()

    # --- prompt tokens (for reference / concatenation in training) ---
    prompt_text = recipe.format_user_turn(prompt) + recipe.turn_prefix_assistant
    prompt_ids = tokenizer.encode(prompt_text).ids

    # --- chosen completion (body only, EOS-terminated) ---
    chosen_body = _format_completion(chosen, chosen_thinking, recipe)
    chosen_ids = tokenizer.encode(chosen_body).ids
    chosen_ids.append(eos_id)

    # --- rejected completion (body only, EOS-terminated) ---
    rejected_body = _format_completion(rejected, rejected_thinking, recipe)
    rejected_ids = tokenizer.encode(rejected_body).ids
    rejected_ids.append(eos_id)

    return prompt_ids, chosen_ids, rejected_ids


def _format_completion(
    answer: str, thinking: str, recipe: TrainingRecipe,
) -> str:
    """
    Format a single completion body using the recipe's format_assistant_turn.
    This produces the assistant-turn body (without prefix) in the format:
        <think>\n{thinking}\n</think>\n{answer}
    or just answer if there's no thinking.
    """
    # Use the recipe's own formatting (gives us think_open/close wrapping)
    full_turn = recipe.format_assistant_turn(
        thinking=thinking,
        answer=answer,
        want_thinking=bool(thinking),
    )
    # Strip the prefix and suffix to get just the body
    prefix = recipe.turn_prefix_assistant
    suffix = recipe.turn_suffix_assistant
    if full_turn.startswith(prefix):
        full_turn = full_turn[len(prefix):]
    if full_turn.endswith(suffix):
        full_turn = full_turn[:-len(suffix)]
    return full_turn.strip()


# ---------------------------------------------------------------------------
# Two-pass packing
# ---------------------------------------------------------------------------


def _count_token_lengths(
    files: List[str],
    recipe: TrainingRecipe,
    tokenizer,
    eos_id: int,
) -> Tuple[int, int, int, List[int], List[int], List[int]]:
    """
    Pass 1: count total tokens and per-record token lengths.

    Returns:
        (total_prompt_tokens, total_chosen_tokens, total_rejected_tokens,
         per_record_prompt_lens, per_record_chosen_lens, per_record_rejected_lens)
    """
    prompt_lens: List[int] = []
    chosen_lens: List[int] = []
    rejected_lens: List[int] = []

    for _fp, _ln, rec in _read_records(files):
        p_ids, c_ids, r_ids = tokenize_preference_triple(rec, recipe, tokenizer, eos_id)
        prompt_lens.append(len(p_ids))
        chosen_lens.append(len(c_ids))
        rejected_lens.append(len(r_ids))

    return (
        sum(prompt_lens), sum(chosen_lens), sum(rejected_lens),
        prompt_lens, chosen_lens, rejected_lens,
    )


def _write_field(
    files: List[str],
    recipe: TrainingRecipe,
    tokenizer,
    eos_id: int,
    output_path: str,
    field_name: str,
    total_tokens: int,
    record_indices: Optional[List[int]] = None,
) -> int:
    """
    Write the specified field as length-prefixed uint32 records.

    Each record is a 4-byte little-endian uint32 length prefix followed by
    that many uint32 token IDs — the exact format consumed by
    ``PackedDPODataLoader._load_bin`` (train_dpo.py).

    Args:
        field_name: one of "prompt_ids", "chosen_ids", "rejected_ids"
                    (returned by tokenize_preference_triple)

    Returns:
        Number of records written.
    """
    want = set(record_indices) if record_indices is not None else None
    records_written = 0
    record_counter = 0
    tokens_written = 0

    with open(output_path, "wb") as f:
        for _fp, _ln, rec in _read_records(files):
            if want is not None and record_counter not in want:
                record_counter += 1
                continue

            p_ids, c_ids, r_ids = tokenize_preference_triple(rec, recipe, tokenizer, eos_id)
            if field_name == "prompt_ids":
                ids = p_ids
            elif field_name in ("chosen_ids", "chosen_all_ids"):
                ids = c_ids
            elif field_name in ("rejected_ids", "rejected_all_ids"):
                ids = r_ids
            else:
                raise ValueError(f"Unknown field name: {field_name}")

            f.write(len(ids).to_bytes(4, "little"))
            f.write(np.asarray(ids, dtype=np.uint32).tobytes())
            tokens_written += len(ids)
            records_written += 1
            record_counter += 1

    assert tokens_written == total_tokens, (
        f"Token count mismatch for {field_name}: expected {total_tokens}, "
        f"wrote {tokens_written}"
    )
    return records_written


def _write_lengths_file(
    lengths: List[int],
    output_path: str,
) -> None:
    """
    Write per-record lengths as a JSON array.
    Used by the DataLoader to determine where completions start.
    """
    with open(output_path, "w") as f:
        json.dump(lengths, f)


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------


def split_by_record(
    per_record_lengths: Dict[str, List[int]],
    val_fraction: float,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """Contiguous tail split across all fields."""
    n = len(next(iter(per_record_lengths.values())))
    n_val = max(1, math.ceil(n * val_fraction)) if val_fraction > 0 else 0
    n_train = n - n_val
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n))

    train_indices: Dict[str, List[int]] = {}
    val_indices: Dict[str, List[int]] = {}
    for key, lengths in per_record_lengths.items():
        train_indices[key] = train_idx[:len(train_idx)]
        val_indices[key] = val_idx[:len(val_idx)]

    return train_indices, val_indices


# ---------------------------------------------------------------------------
# Main packing logic
# ---------------------------------------------------------------------------


def pack_dpo(args: argparse.Namespace) -> None:
    """Main DPO packing routine."""
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

    # --- Output directory ---
    os.makedirs(args.cache_dir, exist_ok=True)

    # --- Pass 1: count ---
    log.info("Pass 1: counting tokens ...")
    try:
        total_p, total_c, total_r, lens_p, lens_c, lens_r = _count_token_lengths(
            files, recipe, tokenizer, eos_id,
        )
    except StopIteration:
        log.error("No valid preference records found — check your data files.")
        sys.exit(1)

    num_records = len(lens_p)
    log.info(
        "Total: %d records | prompt=%d tok | chosen=%d tok | rejected=%d tok",
        num_records, total_p, total_c, total_r,
    )

    if total_p == 0:
        log.error("No tokens found — check your data files.")
        sys.exit(1)

    # --- Train / val split ---
    val_fraction = args.val_fraction
    if val_fraction > 0 and args.num_workers > 1 and args.worker != 0:
        val_fraction = 0.0
        log.info("Worker %d: skipping val split (only worker 0 splits)", args.worker)

    if val_fraction > 0:
        # Split indices
        n_val = max(1, math.ceil(num_records * val_fraction))
        n_train = num_records - n_val
        train_idx = list(range(n_train))
        val_idx = list(range(n_train, num_records))
        log.info(
            "Split: %d train + %d val records", len(train_idx), len(val_idx),
        )
    else:
        train_idx = list(range(num_records))
        val_idx = []

    # --- Helper: total tokens for a subset ---
    def _subset_total(lens: List[int], indices: List[int]) -> int:
        return sum(lens[i] for i in indices if i < len(lens))

    train_p_tok = _subset_total(lens_p, train_idx)
    train_c_tok = _subset_total(lens_c, train_idx)
    train_r_tok = _subset_total(lens_r, train_idx)
    val_p_tok = _subset_total(lens_p, val_idx)
    val_c_tok = _subset_total(lens_c, val_idx)
    val_r_tok = _subset_total(lens_r, val_idx)

    # --- Pass 2: write ---
    worker_tag = f".w{args.worker}" if args.num_workers > 1 else ""

    def _write_split(
        split: str,
        rec_indices: List[int],
        p_tok: int, c_tok: int, r_tok: int,
    ):
        suffix = f"_{split}{worker_tag}"
        p_path = os.path.join(args.cache_dir, f"dpo_prompts{suffix}.bin")
        c_path = os.path.join(args.cache_dir, f"dpo_chosen{suffix}.bin")
        r_path = os.path.join(args.cache_dir, f"dpo_rejected{suffix}.bin")
        plen_path = os.path.join(args.cache_dir, f"dpo_prompt_lens{suffix}.json")

        log.info("Pass 2: writing %s split ...", split)

        n_p = _write_field(
            files, recipe, tokenizer, eos_id,
            p_path, "prompt_ids", p_tok,
            record_indices=rec_indices,
        )
        n_c = _write_field(
            files, recipe, tokenizer, eos_id,
            c_path, "chosen_all_ids", c_tok,
            record_indices=rec_indices,
        )
        n_r = _write_field(
            files, recipe, tokenizer, eos_id,
            r_path, "rejected_all_ids", r_tok,
            record_indices=rec_indices,
        )

        # Write prompt lengths so the DataLoader can mask prompt tokens
        plens = [lens_p[i] for i in rec_indices if i < len(lens_p)]
        _write_lengths_file(plens, plen_path)

        # Manifest
        manifest = {
            "split": split,
            "num_records": n_p,
            "total_prompt_tokens": int(p_tok),
            "total_chosen_tokens": int(c_tok),
            "total_rejected_tokens": int(r_tok),
            "vocab_size": vocab_size,
            "eos_token_id": eos_id,
            "dtype": "uint32",
            "format": "length-prefixed-uint32",
            "mode": recipe.mode,
            "worker": args.worker,
            "num_workers": args.num_workers,
            "prompt_file": os.path.basename(p_path),
            "chosen_file": os.path.basename(c_path),
            "rejected_file": os.path.basename(r_path),
            "prompt_lens_file": os.path.basename(plen_path),
        }
        manifest_path = os.path.join(
            args.cache_dir, f"dpo_manifest{suffix}.json",
        )
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        log.info("Wrote manifest: %s", manifest_path)
        log.info(
            "  %s: %d records | prompt=%d tok | chosen=%d tok | rejected=%d tok",
            split, n_p, p_tok, c_tok, r_tok,
        )

    # Write training split
    if train_idx:
        _write_split("train", train_idx, train_p_tok, train_c_tok, train_r_tok)

    # Write validation split
    if val_idx:
        _write_split("val", val_idx, val_p_tok, val_c_tok, val_r_tok)

    elapsed = time.time() - t0
    log.info("Done in %.1fs", elapsed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pack DPO preference pairs into packed binary format.",
    )
    p.add_argument(
        "--data-dir", required=True,
        help="Directory tree containing .jsonl files with 'prompt', 'chosen', "
             "and 'rejected' fields.",
    )
    p.add_argument(
        "--tokenizer", required=True,
        help="Path to tokenizer directory or tokenizer.json file.",
    )
    p.add_argument(
        "--cache-dir", default="./dpo_packed",
        help="Output directory for packed files (default: ./dpo_packed).",
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
    add_recipe_args(p)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    if args.worker >= args.num_workers:
        log.error("--worker (%d) must be < --num-workers (%d)", args.worker, args.num_workers)
        sys.exit(1)
    if not (0.0 <= args.val_fraction < 1.0):
        log.error("--val-fraction must be in [0, 1)")
        sys.exit(1)
    pack_dpo(args)
