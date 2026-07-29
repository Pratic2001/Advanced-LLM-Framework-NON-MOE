#!/usr/bin/env python3
"""
train_grpo.py

Group Relative Policy Optimization (GRPO) for the dense LLM framework.
Stage 2 of post-training: consumes an SFT checkpoint + prompt-answer data
and applies RL to improve reasoning.

Key features:
  - Three-tier rule-based reward (think format + correctness)
  - Mode-aware reward via TrainingRecipe (B3): reasoning checks <think> tags,
    non_reasoning skips them, hybrid per-prompt want_thinking
  - Batched rollout generation with KV-cache
  - GRPO loss with group-normalized advantages, PPO clip, KL penalty
  - Single-model / two-model reference policy
  - LoRA support (reuses peft.lora module)
  - PackedGRPODataLoader for GRPO-specific packed data
  - TrainingRecipe integration for all template decisions

Usage:
    python train_grpo.py \\
        --data-dir ./grpo_packed \\
        --checkpoint ./sft_checkpoints/latest.pt \\
        --tokenizer ./tokenizer

    python train_grpo.py --smoke-test
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tokenizers import Tokenizer

from model import ModelConfig, TransformerForCausalLM, count_parameters
from recipe import TrainingRecipe, get_recipe, add_recipe_args, recipe_from_args
from train_sft import SFTDataset

# ---------------------------------------------------------------------------
# Hardware TFLOPS table (for throughput estimation)
# ---------------------------------------------------------------------------

GPU_PEAK_TFLOPS = {
    "A100-80G":  312,  "A100-40G":  312,
    "H100":      989,  "H100-PCIe": 756,  "H200": 989,
    "RTX-4090":  165,  "RTX-3090":  165,
    "RTX-5090":  260,  "RTX-5090D": 260,
    "RTX-A6000": 155,  "RTX-A6000-48gb": 155,
    "RTX-4080":  120,  "RTX-4070Ti": 82,
    "L40S":      362,
    "MI250X":    383,  "MI300X":     1307,
    "TPU-v4":    275,  "TPU-v5e":    197,
}

def _detect_gpu_peak_tflops() -> float:
    """Return the peak BF16 TFLOPS of the detected GPU, or 120 as fallback."""
    if not torch.cuda.is_available():
        return 0.0
    name = torch.cuda.get_device_name(0)
    for key, val in GPU_PEAK_TFLOPS.items():
        if key.lower() in name.lower():
            return val
    return 120.0

# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, local_rank, world_size, device

def is_master(rank: int) -> bool:
    return rank == 0

def _raw(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP / torch.compile wrappers."""
    m = model.module if isinstance(model, DDP) else model
    return m._orig_mod if hasattr(m, "_orig_mod") else m

# ---------------------------------------------------------------------------
# Tokenizer helper
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_dir: str) -> Tokenizer:
    """Load a HuggingFace `tokenizers` Tokenizer from a directory."""
    path = os.path.join(tokenizer_dir, "tokenizer.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"tokenizer.json not found in {tokenizer_dir}")
    return Tokenizer.from_file(path)

def get_special_token_id(tokenizer: Tokenizer, token: str) -> int:
    """Look up a special token id, with fallback."""
    tid = tokenizer.token_to_id(token)
    if tid is not None:
        return tid
    # fallback: decode the special token map
    added = tokenizer.get_added_vocabulary()
    if token in added:
        return added[token]
    return tokenizer.get_vocab_size() - 1

# ---------------------------------------------------------------------------
# Reward function — three-tier, recipe-aware
# ---------------------------------------------------------------------------

# \boxed{...} capture
ANSWER_RE = re.compile(r"\\boxed\{([^}]+)\}")

# Numeric fallback
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_answer(field: str, think_close: str = "</think>") -> Tuple[Optional[float], str]:
    """
    Pull a comparable answer out of a completion string.

    Args:
        field: The completion text to extract answer from.
        think_close: The closing think tag (from recipe).

    Returns:
        (numeric_value, raw_string).
    """
    if field is None:
        return None, ""

    # Try \boxed{} first
    m = ANSWER_RE.search(field)
    if m:
        return _try_float(m.group(1).strip()), m.group(1).strip()

    # Slice to "post-think" portion for final-answer extraction
    c = field.rfind(think_close)
    answer_zone = field[c + len(think_close):] if c != -1 else field

    # Last numeric match in the answer zone
    nums = list(NUM_RE.finditer(answer_zone))
    if nums:
        last = nums[-1]
        return _try_float(last.group(0)), last.group(0)

    toks = answer_zone.strip().split()
    return None, toks[-1] if toks else ""


def _try_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _has_balanced_think(text: str, think_open: str, think_close: str) -> bool:
    """True iff the text contains think_open...think_close in order."""
    o = text.find(think_open)
    c = text.find(think_close)
    return o != -1 and c != -1 and o < c


def reward_fn(
    completion: str,
    expected_answer: str,
    want_thinking: Optional[bool],
    recipe: TrainingRecipe,
    max_new_tokens: int,
    correct_weight: float = 1.0,
    format_weight: float = 0.3,
) -> Tuple[float, Dict[str, int]]:
    """
    Three-tier reward with mode awareness via TrainingRecipe.

    reward_should_check_thinking(want_thinking) decides whether think-format
    checking applies:

        reasoning     → ALWAYS check (want_thinking ignored → True)
        non_reasoning → NEVER check  (want_thinking ignored → False)
        hybrid        → check only if want_thinking is True

    Tiers:
        1.0 (correct_weight): correct answer AND balanced think (if checked)
        0.5 (0.5 * correct_weight): correct but missing/bad think format
        0.0: wrong answer
    """
    check_think = recipe.reward_should_check_thinking(want_thinking)
    has_think = (
        _has_balanced_think(completion, recipe.think_open, recipe.think_close)
        if check_think else True
    )
    truncated = len(completion) >= max_new_tokens * 4

    gold_num, gold_str = extract_answer(expected_answer, recipe.think_close)
    pred_num, pred_str = extract_answer(completion, recipe.think_close)

    if gold_num is not None and pred_num is not None:
        correct = math.isclose(gold_num, pred_num, rel_tol=1e-3, abs_tol=1e-4)
    else:
        correct = bool(gold_str) and gold_str.lower() == pred_str.lower()

    has_answer = bool(pred_str)

    info = {
        "correct": int(correct),
        "has_think": int(has_think),
        "has_answer": int(has_answer),
        "truncated": int(truncated),
    }

    if truncated and not has_answer:
        return 0.0, info
    if correct and has_think:
        return correct_weight, info
    if correct and not has_think:
        return 0.5 * correct_weight, info
    if has_think and has_answer and not truncated:
        return format_weight, info
    return 0.0, info


# ---------------------------------------------------------------------------
# GRPOPromptDataset — prompt dataset backed by packed SFT memmaps
# ---------------------------------------------------------------------------

class GRPOPromptDataset:
    """
    Streams ``{prompt_ids, ground_truth_answer}`` pairs from the **packed
    memmaps** written by ``data/pack_sft.py``. Re-tokenisation is unnecessary;
    the tokens are already on disk in two mmap'd arrays:

        tokens[i]  : token id at position i
        mask[i]    : 1 = assistant token (in loss), 0 = prompt / EOS sep

    Walking sample boundaries:

        As ``data/pack_sft.py`` writes the file, each record is followed by
        a single EOS token (mask = 0). Sample boundaries are the positions
        where mask transitions from 1 -> 0 -> EOS. We find every
        (prompt_start, prompt_end, answer_end) triple during a single linear
        sweep over the memmap, store them as ``(offset, prompt_len, answer_len)``
        triples in RAM, and never re-touch the disk during training.

    Mode selection:
        default  -> ``--cache_dir ./sft_packed`` + ``--data_dir ./sft_data``.
                     The packed memmaps drive tokenisation; ``./sft_data`` JSONL
                     shards are read once to recover ``answer`` strings.
        override -> ``--prompts_file`` path. A plain JSONL of ``{prompt, answer}``
                     is read in full (still small enough for typical eval sets).

    Either way the per-step sample() call returns random indices into a
    pre-built list — RAM stays flat.
    """

    def __init__(
        self,
        cache_dir: Optional[str],        # --cache_dir (packed memmaps)
        data_dir: Optional[str],         # --data_dir  (raw JSONL, for answer text)
        prompts_file: Optional[str],     # --prompts_file override
        tokenizer: Tokenizer,
        max_prompt_len: int,
        eos_id: int,
    ):
        self.tokenizer      = tokenizer
        self.max_prompt_len = max_prompt_len
        self.eos_id         = eos_id

        if prompts_file:
            self._init_from_jsonl(prompts_file)
        else:
            assert cache_dir, "--cache_dir is required when --prompts_file is not given"
            self._init_from_packed(cache_dir, data_dir)

        if not self._prompts:
            raise RuntimeError(
                f"No usable prompts after filtering. Check that "
                f"{'--cache_dir' if prompts_file is None else prompts_file} "
                f"contains valid records."
            )

    # ----------------------------------------------------------------------
    # Path A: packed SFT memmaps + raw JSONL for answer text
    # ----------------------------------------------------------------------
    def _init_from_packed(self, cache_dir: str, data_dir: Optional[str]):
        """
        Open the same memmap files train_sft.py reads, locate sample
        boundaries, and pre-build (prompt_ids, ground_truth) pairs.
        """
        # Reuse SFTDataset for manifest discovery + mmap concatenation.
        # We pass a tiny seq_len so .__len__==0 in the rank/world_size
        # sense doesn't matter — we use the underlying _ConcatMemmaps
        # directly.
        probe = SFTDataset(
            cache_dir=cache_dir,
            seq_len=2**30,                  # floor: every "window" is the full file
            rank=0, world_size=1,
            split="train",
        )
        self._tokens_memmap = probe.tokens   # lazy _ConcatMemmap (mmap-backed, no copy)
        self._mask_memmap   = probe.mask     # lazy _ConcatMemmap (mmap-backed, no copy)
        self._n_shards      = probe.n_shards
        total = len(self._tokens_memmap)

        # ---- find (prompt_start, prompt_end, assistant_end) by scanning
        #      the mask array. Sample boundary = mask==0 position that is
        #      also an EOS token (the separator pack_sft_data.py writes).
        #
        # We scan each underlying worker-shard memmap directly (still
        # disk-/page-cache-backed, evictable under memory pressure)
        # instead of materializing the whole concatenated dataset into
        # one permanent, non-reclaimable RAM buffer via _contiguous_view.
        # Each worker shard from pack_sft_data.py holds complete records
        # (no record straddles two shard files), so per-shard scanning
        # finds the exact same boundaries as a single global scan would.
        tok_shards  = self._tokens_memmap.arrays
        mask_shards = self._mask_memmap.arrays

        # (shard_idx, local_start, local_end) — kept per-shard rather
        # than flattened into one global offset, so prompt extraction
        # below can index straight into that shard's own memmap.
        boundaries: List[Tuple[int, int, int]] = []
        for shard_idx, (tok_arr, mask_arr) in enumerate(zip(tok_shards, mask_shards)):
            for s, e in self._scan_boundaries(mask_arr, tok_arr):
                boundaries.append((shard_idx, s, e))

        # ---- recover the original `answer` strings from the JSONL pool
        answers_text: List[Optional[str]] = self._load_answer_strings(data_dir, len(boundaries))

        # ---- build per-record slices
        prompts: List[List[int]] = []
        ground_truths: List[str] = []
        prompt_texts: List[str]  = []
        eos_id = self.eos_id

        skipped_long = 0
        skipped_no_gt = 0
        for i, (shard_idx, s, e) in enumerate(boundaries):
            gt = answers_text[i] if i < len(answers_text) else None
            if not gt:
                skipped_no_gt += 1
                continue
            tok_arr  = tok_shards[shard_idx]
            mask_arr = mask_shards[shard_idx]
            # Find the assistant-turn start: the first mask==1 within [s, e).
            p_start = s
            while p_start < e and mask_arr[p_start] == 0:
                p_start += 1
            prompt_ids = tok_arr[s:p_start].tolist()
            if len(prompt_ids) >= self.max_prompt_len:
                skipped_long += 1
                continue
            prompts.append(prompt_ids)
            ground_truths.append(gt)
            try:
                prompt_texts.append(
                    self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
                )
            except Exception:
                prompt_texts.append("")

        if skipped_long:
            print(f"[PackedDataset] skipped {skipped_long} record(s) "
                  f"with prompt > --max_prompt_len={self.max_prompt_len}")
        if skipped_no_gt:
            print(f"[PackedDataset] {skipped_no_gt} record(s) had no "
                  f"answer text in JSONL pool — skipped")

        self._prompts     = prompts
        self._answers     = ground_truths
        self._prompt_text = prompt_texts

    # ----------------------------------------------------------------------
    @staticmethod
    def _scan_boundaries(mask_arr: np.ndarray, tok_arr: np.ndarray) -> List[Tuple[int, int]]:
        """
        Walk the mask array and return sample boundaries.

        A boundary ends at a position where mask is 0 *and* the
        corresponding token is the EOS special id (the separator
        written between records).

        Returns: list of (start, end_exclusive) tuples.

        Implementation: vectorized search with numpy for candidates,
        then loop over candidates (O(n_records), not O(n_tokens)).
        """
        n = len(mask_arr)
        if n == 0:
            return []

        # Candidate separator positions: mask==0 and token==EOS(0).
        candidates = np.flatnonzero((mask_arr == 0) & (tok_arr == 0))

        # Positions where mask==1, needed to confirm "in_record" was
        # true since the last boundary.
        mask1_positions = np.flatnonzero(mask_arr == 1)

        boundaries: List[Tuple[int, int]] = []
        rec_start = 0
        m1_ptr = 0
        n_m1 = len(mask1_positions)

        for c in candidates:
            if c < rec_start:
                continue
            while m1_ptr < n_m1 and mask1_positions[m1_ptr] < rec_start:
                m1_ptr += 1
            in_record = m1_ptr < n_m1 and mask1_positions[m1_ptr] < c
            if in_record:
                boundaries.append((rec_start, c + 1))
                rec_start = c + 1

        if rec_start < n:
            while m1_ptr < n_m1 and mask1_positions[m1_ptr] < rec_start:
                m1_ptr += 1
            if m1_ptr < n_m1 and mask1_positions[m1_ptr] < n:
                boundaries.append((rec_start, n))

        return boundaries

    # ----------------------------------------------------------------------
    @staticmethod
    def _load_answer_strings(data_dir: Optional[str], n_expected: int) -> List[Optional[str]]:
        """
        Stream JSONL shards under data_dir and collect each record's
        ``answer`` field, in the same order ``data/pack_sft.py`` saw them.
        """
        if not data_dir:
            return [""] * n_expected
        paths = sorted(glob.glob(os.path.join(data_dir, "*", "*.jsonl")))
        if not paths:
            paths = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
        if not paths:
            print(f"[PackedDataset] no JSONL found under {data_dir}; "
                  f"every prompt will get an empty ground truth.")
            return [""] * n_expected

        answers: List[str] = []
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    answers.append((rec.get("answer") or "").strip())
        if len(answers) < n_expected:
            answers.extend([""] * (n_expected - len(answers)))
        else:
            answers = answers[:n_expected]
        return answers

    # ----------------------------------------------------------------------
    # Path B: plain JSONL override
    # ----------------------------------------------------------------------
    def _init_from_jsonl(self, path: str):
        """Fallback: read every record's prompt + answer from a single file."""
        prompts: List[List[int]] = []
        ground_truths: List[str] = []
        prompt_texts: List[str]  = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = (rec.get("prompt") or "").strip()
                answer = (rec.get("answer") or "").strip()
                if not prompt or not answer:
                    continue
                ids = self._format_prompt(prompt)
                if ids is None:
                    continue
                prompts.append(ids)
                ground_truths.append(answer)
                prompt_texts.append(prompt)

        self._prompts     = prompts
        self._answers     = ground_truths
        self._prompt_text = prompt_texts

    # ----------------------------------------------------------------------
    def _format_prompt(self, prompt: str) -> Optional[List[int]]:
        """ChatML user-turn prefix, matches data/pack_sft.py formatting."""
        text = f"user\n{prompt}\nassistant\n"
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) >= self.max_prompt_len:
            return None
        return ids

    # ----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._prompts)

    def sample_batch(
        self, batch_size: int, rng: random.Random,
    ) -> Tuple[List[List[int]], List[str], List[str]]:
        """Return (prompt_ids, ground_truth, prompt_text) for `batch_size` prompts."""
        idxs = [rng.randrange(len(self._prompts)) for _ in range(batch_size)]
        return (
            [self._prompts[i]     for i in idxs],
            [self._answers[i]     for i in idxs],
            [self._prompt_text[i] for i in idxs],
        )


def _contiguous_view(concat_memmap) -> np.ndarray:
    """
    Materialise a ``_ConcatMemmap`` view as a single in-RAM ndarray.
    Memory cost = one full pass over the dataset.
    """
    return np.asarray(concat_memmap[:])


# ---------------------------------------------------------------------------
# PackedGRPODataLoader — reads grpo_prompt_tokens*.bin + grpo_answers*.json
# ---------------------------------------------------------------------------
# This is the GRPO-specific data path. Packed data from pack_grpo.py is
# simple: each .bin file contains flat uint32 token arrays, and the .json
# sidecar carries answer strings and want_thinking flags.
#
# Format of .bin files: flat uint32 numpy memmap, records concatenated with
# length prefixes [n_tokens: uint32][tokens: uint32 * n_tokens].
#
# Format of .json files: JSON array of {"answer": "...", "want_thinking": bool}


class PackedGRPODataLoader:
    """
    Streams (prompt_ids, answers, want_thinking) from GRPO-packed data.

    Data layout uses length-prefixed records packed into flat binary + JSON
    sidecar file:

        grpo_prompt_tokens_000.bin   — uint32 flat array of token IDs
        grpo_answers_000.json        — {"answer": "...", "want_thinking": bool}

    Each .bin record has a 4-byte uint32 length prefix followed by that
    many uint32 token IDs. Records align 1:1 with entries in the .json file.

    Multi-worker data: filenames follow *.bin / *.json pattern; all files
    matching are discovered and concatenated.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.rng = random.Random(seed + rank)

        # Discover data files
        bin_files = sorted(glob.glob(os.path.join(data_dir, "grpo_prompt_tokens*.bin")))
        json_files = sorted(glob.glob(os.path.join(data_dir, "grpo_answers*.json")))

        if not bin_files:
            raise FileNotFoundError(
                f"No grpo_prompt_tokens*.bin files found in {data_dir}. "
                f"Run data/pack_grpo.py first."
            )

        # Load all records
        self._prompt_ids: List[torch.Tensor] = []
        self._answers: List[str] = []
        self._want_thinking: List[bool] = []

        for bin_fp, json_fp in zip(bin_files, json_files):
            prompts = self._load_bin(bin_fp)
            answers_meta = self._load_json(json_fp)

            if len(prompts) != len(answers_meta):
                print(f"[PackedGRPODataLoader] WARNING: {bin_fp} has {len(prompts)} "
                      f"records but {json_fp} has {len(answers_meta)} — truncating")
                n = min(len(prompts), len(answers_meta))
                prompts = prompts[:n]
                answers_meta = answers_meta[:n]

            for pids, meta in zip(prompts, answers_meta):
                self._prompt_ids.append(torch.tensor(pids, dtype=torch.long))
                self._answers.append(meta.get("answer", ""))
                self._want_thinking.append(meta.get("want_thinking", True))

        # Rank shard
        total = len(self._prompt_ids)
        shard_size = total // world_size
        start = rank * shard_size
        end = start + shard_size if rank < world_size - 1 else total
        self._prompt_ids = self._prompt_ids[start:end]
        self._answers = self._answers[start:end]
        self._want_thinking = self._want_thinking[start:end]

        if rank == 0:
            print(f"[PackedGRPODataLoader] {total:,} total prompts, "
                  f"{end - start:,} per rank")

    # ------------------------------------------------------------------
    @staticmethod
    def _load_bin(path: str) -> List[List[int]]:
        """Load length-prefixed uint32 records from a .bin file."""
        with open(path, "rb") as f:
            data = f.read()
        records: List[List[int]] = []
        offset = 0
        while offset < len(data):
            n_tokens = int.from_bytes(data[offset:offset + 4], "little")
            offset += 4
            tokens = []
            for i in range(n_tokens):
                tok = int.from_bytes(
                    data[offset + i * 4:offset + (i + 1) * 4], "little"
                )
                tokens.append(tok)
            offset += n_tokens * 4
            records.append(tokens)
        return records

    @staticmethod
    def _load_json(path: str) -> List[Dict[str, Any]]:
        with open(path, "r") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._prompt_ids)

    def __iter__(self):
        """Yield (prompt_ids, answers, want_thinking) batches."""
        indices = list(range(len(self._prompt_ids)))
        self.rng.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            prompt_ids = torch.stack([self._prompt_ids[j] for j in batch_idx])
            answers = [self._answers[j] for j in batch_idx]
            want_thinking = [self._want_thinking[j] for j in batch_idx]
            yield prompt_ids, answers, want_thinking


# ---------------------------------------------------------------------------
# Attention mask builder for left-padded batches
# ---------------------------------------------------------------------------

@torch.no_grad()
def _build_attn_mask(
    prompt_pad_mask: torch.Tensor,   # (B, P) additive float: 0.0 real, -inf pad
    seq_len: int,
    past_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build a (B, 1, seq_len, past_len+seq_len) additive attention mask
    combining standard causal restriction with the prompt's left-padding.
    Passing an explicit attn_mask disables the model's internal
    `is_causal` SDPA fast path, so this mask must encode causality itself.
    """
    B, P = prompt_pad_mask.shape
    device = prompt_pad_mask.device
    total_len = past_len + seq_len
    gen_len = total_len - P
    if gen_len > 0:
        gen_pad = torch.zeros((B, gen_len), dtype=prompt_pad_mask.dtype, device=device)
        full_pad = torch.cat([prompt_pad_mask, gen_pad], dim=1)
    else:
        full_pad = prompt_pad_mask[:, :total_len]
    # (B, 1, 1, total_len) broadcast — only padding columns are masked
    pad_mask = full_pad[:, None, None, :]   # 0 real, -inf pad
    # causal: positions at index > seq_offset are masked
    seq_offset = past_len
    causal = torch.triu(
        torch.full((seq_len, total_len), float("-inf"), device=device, dtype=dtype),
        diagonal=seq_offset + 1,
    )
    return (pad_mask + causal).to(dtype)  # (B, 1, seq_len, total_len)


# ---------------------------------------------------------------------------
# Batched rollout generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_rollouts(
    model: TransformerForCausalLM,
    prompt_ids_list: List[List[int]],
    recipe: TrainingRecipe,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_id: int,
    pad_id: int,
    rng: random.Random,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate G completions per prompt.

    Args:
        model: The policy model.
        prompt_ids_list: List of prompt token id lists (one per prompt).
        recipe: TrainingRecipe for think_open/close detection.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        eos_id: End-of-sequence token id.
        pad_id: Padding token id.
        rng: Per-replica RNG for diverse samples.

    Returns:
        full_ids:       (B*G, P+T) token ids.
        gen_mask:       (B*G, T) 1 for generated positions, 0 for padding.
        sampled_lp:     (B*G, T) per-token log-prob of the sampled token.
        prompt_pad_mask: (B*G, P) additive left-padding mask (0 real, -inf pad).
    """
    device = next(model.parameters()).device

    B = len(prompt_ids_list)
    P = max(len(p) for p in prompt_ids_list)
    prompt_ids = torch.full((B, P), pad_id, dtype=torch.long, device=device)
    prompt_pad_mask = torch.zeros((B, P), dtype=torch.float, device=device)
    for i, pids in enumerate(prompt_ids_list):
        offset = P - len(pids)
        prompt_ids[i, offset:] = torch.tensor(pids, dtype=torch.long, device=device)
        if offset > 0:
            prompt_pad_mask[i, :offset] = float("-inf")

    full_ids = torch.full((B, P + max_new_tokens), pad_id, dtype=torch.long, device=device)
    full_ids[:, :P] = prompt_ids
    gen_mask = torch.zeros((B, max_new_tokens), dtype=torch.float, device=device)
    sampled_lp = torch.zeros((B, max_new_tokens), dtype=torch.float, device=device)

    model_dtype = next(model.parameters()).dtype
    past_kv = None
    inp = prompt_ids

    g = torch.Generator(device=device)
    g.manual_seed(rng.randrange(2**31))

    already_done = torch.zeros(B, dtype=torch.bool, device=device)

    for t in range(max_new_tokens):
        past_len = 0 if past_kv is None else past_kv[0][0].shape[2]
        attn_mask = _build_attn_mask(prompt_pad_mask, inp.shape[1], past_len, model_dtype)

        out = model(inp, attention_mask=attn_mask, past_key_values=past_kv, use_cache=True)
        logits = out["logits"][:, -1, :].float()
        past_kv = out["past_key_values"]

        logits = logits / max(temperature, 1e-5)

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            sp = sorted_logits.softmax(dim=-1)
            cumsp = sp.cumsum(dim=-1)
            keep = cumsp <= top_p
            keep[..., 0] = True
            mask = torch.full_like(logits, False)
            mask.scatter_(-1, sorted_idx, keep)
            logits = torch.where(mask, logits, torch.full_like(logits, float("-inf")))

        probs = logits.softmax(dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1, generator=g).squeeze(-1)
        tok_lp = logits.log_softmax(dim=-1).gather(-1, next_tok.unsqueeze(-1)).squeeze(-1)

        active = (~already_done).float()
        full_ids[:, P + t] = next_tok
        gen_mask[:, t] = active
        sampled_lp[:, t] = tok_lp * active

        finished_now = (next_tok == eos_id) & (~already_done)
        already_done = already_done | finished_now

        if already_done.all():
            break

    actual_T = max_new_tokens
    if gen_mask.any():
        active_lens = gen_mask.sum(dim=1).long()
        actual_T = int(active_lens.max().item())
        actual_T = max(1, min(actual_T, max_new_tokens))

    return full_ids[:, :P + actual_T], gen_mask[:, :actual_T], sampled_lp[:, :actual_T], prompt_pad_mask


# ---------------------------------------------------------------------------
# Log-prob computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_logprobs(
    model: TransformerForCausalLM,
    full_ids: torch.Tensor,
    gen_mask: torch.Tensor,
    prompt_pad_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Per-token log-prob of generated positions.

    When `prompt_pad_mask` is given, builds an explicit causal+padding
    attention mask so rows with a shorter prompt aren't scored against
    corrupted (pad-attending) hidden states.
    """
    attn_mask = None
    if prompt_pad_mask is not None:
        seq_len = full_ids.shape[1]
        model_dtype = next(model.parameters()).dtype
        attn_mask = _build_attn_mask(prompt_pad_mask, seq_len, 0, model_dtype)
    T = gen_mask.shape[1]
    out = model(full_ids, attention_mask=attn_mask, use_cache=False,
                num_logits_to_keep=T)
    logits = out["logits"].float()
    targets = full_ids[:, -T:]
    logp = logits.log_softmax(dim=-1)
    tok_lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return tok_lp * gen_mask


# ---------------------------------------------------------------------------
# GRPO loss
# ---------------------------------------------------------------------------

def grpo_loss(
    policy_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    rewards: torch.Tensor,
    gen_mask: torch.Tensor,
    group_size: int,
    kl_coef: float,
    clip_ratio: float,
    entropy_coef: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    GRPO objective with group-normalized advantages, PPO clip, KL penalty,
    and optional entropy bonus.

    Args:
        policy_logp: (N, T) log-probs of the policy.
        ref_logp: (N, T) log-probs of the reference model.
        rewards: (N,) per-completion rewards.
        gen_mask: (N, T) mask for generated positions.
        group_size: G — number of completions per prompt.
        kl_coef: KL penalty coefficient.
        clip_ratio: PPO clipping range.
        entropy_coef: Entropy bonus coefficient (0 = disabled).

    Returns:
        (loss, metrics_dict)
    """
    N, T = policy_logp.shape
    assert N % group_size == 0
    n_prompts = N // group_size

    r_g = rewards.view(n_prompts, group_size)
    mean = r_g.mean(dim=1, keepdim=True)
    std = r_g.std(dim=1, keepdim=True).clamp(min=1e-4)
    advantages = ((r_g - mean) / std).view(-1)

    log_ratio = (policy_logp - ref_logp) * gen_mask.float()
    ratio = log_ratio.exp()

    adv_b = advantages.unsqueeze(1)
    surr1 = adv_b * ratio
    surr2 = adv_b * ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    pg_per_tok = -torch.min(surr1, surr2) * gen_mask.float()
    pg_loss = pg_per_tok.sum() / gen_mask.sum().clamp(min=1.0)

    # KL penalty — k3 estimator (unbiased, matches TRL/HF GRPO)
    kl_per_tok = (ratio - log_ratio - 1.0) * gen_mask.float()
    kl_loss = kl_per_tok.sum() / gen_mask.sum().clamp(min=1.0)

    # Optional entropy bonus (encourages exploration)
    if entropy_coef > 0:
        # Compute policy entropy from logits
        logits = policy_logp / (policy_logp.exp().clamp(min=1e-12))  # approx
        # More precise: policy_logp is per-token, so entropy = -p*log(p)
        probs = policy_logp.exp().clamp(min=1e-12)
        entropy_per_tok = -(probs * policy_logp)  # p * log(p) = prob * logp
        entropy = (entropy_per_tok * gen_mask).sum() / gen_mask.sum().clamp(min=1.0)
        entropy_loss = -entropy_coef * entropy
    else:
        entropy_loss = 0.0

    loss = pg_loss + kl_coef * kl_loss + entropy_loss

    metrics = {
        "pg": float(pg_loss.detach().item()),
        "kl": float(kl_loss.detach().item()),
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(rewards.std().item()),
        "advantage_abs_mean": float(advantages.abs().mean().item()),
        "ratio_mean": float(ratio.detach().mean().item()),
    }
    if entropy_coef > 0:
        metrics["entropy"] = float(entropy_loss.detach().item())

    return loss, metrics


# ---------------------------------------------------------------------------
# Reference policy setup
# ---------------------------------------------------------------------------

def build_reference(
    ref_policy: str,
    config: ModelConfig,
    sft_ckpt_path: Optional[str],
    device: torch.device,
) -> Optional[TransformerForCausalLM]:
    """
    Build a frozen reference model for KL penalty.

    'single' — reuses the trainable model under no_grad (None returned).
    'two'    — loads a frozen second copy of the model.
    """
    if ref_policy == "single":
        return None
    if ref_policy == "two":
        if sft_ckpt_path is None or not os.path.exists(sft_ckpt_path):
            raise FileNotFoundError(
                "--ref_policy two requires --checkpoint pointing at the SFT checkpoint."
            )
        ref_model = TransformerForCausalLM(config).to(device)
        ckpt = torch.load(sft_ckpt_path, map_location=device, weights_only=False)
        ref_model.load_state_dict(ckpt["model_state"])
        if hasattr(ref_model, "tie_weights"):
            ref_model.tie_weights()
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        n_ref = count_parameters(ref_model)
        print(f"[RefPolicy] two-model: frozen reference ({n_ref/1e9:.3f}B)")
        return ref_model

    raise ValueError(f"Unknown --ref_policy {ref_policy!r}; expected 'single' or 'two'")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def prune_checkpoints(out_dir: str, keep: int = 3):
    ckpts = sorted(
        Path(out_dir).glob("grpo_step*.pt"),
        key=lambda p: int(p.stem.replace("grpo_step", "")),
    )
    for old in ckpts[:-keep]:
        old.unlink()


def save_grpo_checkpoint(
    out_dir: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    args_dict: dict,
    is_lora: bool,
    recipe: TrainingRecipe,
):
    """Save GRPO checkpoint with recipe.json sidecar."""
    raw = _raw(model)
    ckpt = {
        "step": step,
        "model_state": raw.state_dict(),  # Always full state for RL
        "optimizer_state": optimizer.state_dict(),
        "config": vars(config),
        "args": args_dict,
        "is_lora": is_lora,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"grpo_step{step:07d}.pt")
    torch.save(ckpt, path)
    latest = os.path.join(out_dir, "latest.pt")
    if os.path.islink(latest):
        os.remove(latest)
    try:
        os.symlink(os.path.abspath(path), latest)
    except OSError:
        pass

    # Save recipe alongside checkpoint
    recipe_path = os.path.join(out_dir, "recipe.json")
    recipe.to_json(recipe_path)

    print(f"[Checkpoint] saved {path}")
    return path


def load_grpo_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    is_lora: bool,
) -> int:
    """Load GRPO checkpoint and return step."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw = _raw(model)
    state = ckpt["model_state"]
    if is_lora:
        missing, unexpected = raw.load_state_dict(state, strict=False)
        print(f"[Checkpoint] loaded LoRA checkpoint from {path}")
    else:
        raw.load_state_dict(state)
        if hasattr(raw, "tie_weights"):
            raw.tie_weights()
        print(f"[Checkpoint] loaded full checkpoint from {path}")
    if optimizer and "optimizer_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception as e:
            print(f"[Checkpoint] optimizer state load skipped: {e}")
    step = ckpt.get("step", 0)
    return step


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: TransformerForCausalLM,
    val_dataset: PackedGRPODataLoader,
    recipe: TrainingRecipe,
    max_new_tokens: int,
    eos_id: int,
    pad_id: int,
    num_generations: int,
    temperature: float = 0.7,
    top_p: float = 0.95,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Run generation-based validation over a subset of prompts.

    Returns metrics dict with accuracy, avg reward, think rate.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    rng = random.Random(0)

    all_rewards: List[float] = []
    all_correct: List[int] = []
    all_think: List[int] = []
    n_batches = max(1, min(10, len(val_dataset) // max(1, val_dataset.batch_size)))

    for bidx, (prompt_ids, answers, want_thinking) in enumerate(val_dataset):
        if bidx >= n_batches:
            break

        prompt_ids_list = [p.tolist() for p in prompt_ids]
        expanded_p = []
        expanded_a = []
        expanded_wt = []
        for p, a, wt in zip(prompt_ids_list, answers, want_thinking):
            for _ in range(num_generations):
                expanded_p.append(p)
                expanded_a.append(a)
                expanded_wt.append(wt)

        full_ids, gen_mask, _prompt_pad_mask = generate_rollouts(
            model, expanded_p, recipe,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_id=eos_id,
            pad_id=pad_id,
            rng=rng,
        )

        P_max = max(len(p) for p in expanded_p)
        for i, (pids, answer, wt) in enumerate(zip(expanded_p, expanded_a, expanded_wt)):
            start = P_max - len(pids)
            active = int(gen_mask[i].sum().item())
            gen_ids = full_ids[i, start + len(pids):start + len(pids) + active].tolist()
            completion_text = val_dataset._prompt_ids[0].new_zeros(1).tolist()  # placeholder
            # Decode via tokenizer (must be available externally)
            # In practice, we need the tokenizer. For simplicity, decode in the caller.
            all_correct.append(0)  # placeholder

    model.train()
    return {
        "avg_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "accuracy": float(np.mean(all_correct)) if all_correct else 0.0,
        "think_rate": float(np.mean(all_think)) if all_think else 0.0,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test():
    """
    Self-contained smoke test using tiny model, synthetic data.
    Tests the full GRPO pipeline: data loading, rollout, reward, loss.
    """
    print("\n=== GRPO smoke test ===")
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp()
    data_dir = os.path.join(tmp, "grpo_data")
    ckpt_dir = os.path.join(tmp, "ckpts")
    tok_dir = os.path.join(tmp, "tokenizer")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tok_dir, exist_ok=True)

    # ---- minimal tokenizer
    from tokenizers import Tokenizer as _Tok
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers import pre_tokenizers, decoders

    SPECIAL = ["<|endoftext|>", "<|pad|>", "<|im_start|>", "<|im_end|>",
               "<think>", "</think>"]
    tok = _Tok(BPE(unk_token=None, byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(vocab_size=512, special_tokens=SPECIAL,
                         initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                         show_progress=False)
    corpus = [
        "<|im_start|>user\nSolve 2+2<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n2+2=4\n</think>\n4<|im_end|>\n",
    ] * 30
    tok.train_from_iterator(corpus, trainer=trainer)
    tok.save(os.path.join(tok_dir, "tokenizer.json"))

    # ---- create GRPO data (packed format)
    recipe = TrainingRecipe(mode="reasoning")
    eos_id = tok.token_to_id("<|endoftext|>") or tok.get_vocab_size() - 1
    records = [
        {"prompt": "Solve: 2+2", "thinking": "2 plus 2 equals 4", "answer": "4"},
        {"prompt": "What is 10-3?", "thinking": "10 minus 3 is 7", "answer": "7"},
    ] * 25

    # Write packed data in the length-prefixed format
    prompt_ids_list = []
    answers_list = []
    for r in records:
        text = f"user\n{r['prompt']}\nassistant\n"
        ids = tok.encode(text).ids
        prompt_ids_list.append(ids)
        answers_list.append(r["answer"])

    # Write .bin
    bin_path = os.path.join(data_dir, "grpo_prompt_tokens_000.bin")
    with open(bin_path, "wb") as f:
        for ids in prompt_ids_list:
            f.write(len(ids).to_bytes(4, "little"))
            for tid in ids:
                f.write(min(tid, 65535).to_bytes(4, "little"))

    # Write .json
    json_path = os.path.join(data_dir, "grpo_answers_000.json")
    with open(json_path, "w") as f:
        meta = [{"answer": a, "want_thinking": True} for a in answers_list]
        json.dump(meta, f)

    # ---- tiny model
    config = ModelConfig(
        vocab_size=512, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, max_position_embeddings=256,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerForCausalLM(config).to(device)

    # Save fake SFT checkpoint
    sft_ckpt_path = os.path.join(ckpt_dir, "sft.pt")
    torch.save({"model_state": model.state_dict(), "config": vars(config)}, sft_ckpt_path)

    # ---- inject LoRA for lower memory
    from peft.lora import inject_lora, lora_state_dict
    n_replaced = inject_lora(model, rank=4, alpha=8.0)
    print(f"[smoke] LoRA: {n_replaced} adapters injected")

    # ---- dataset
    ds = PackedGRPODataLoader(
        data_dir=data_dir, batch_size=2,
        rank=0, world_size=1,
    )
    print(f"[smoke] dataset size: {len(ds)}")

    # ---- reference
    ref = build_reference("single", config, sft_ckpt_path, device)

    # ---- generate rollouts
    rng = random.Random(0)
    for batch_prompts, batch_answers, batch_wt in ds:
        prompt_list = [p.tolist() for p in batch_prompts]
        # Generate G=2 per prompt
        expanded_p = [p for p in prompt_list for _ in range(2)]
        expanded_a = [a for a in batch_answers for _ in range(2)]
        expanded_wt = [wt for wt in batch_wt for _ in range(2)]

        full_ids, gen_mask, _prompt_pad_mask = generate_rollouts(
            model, expanded_p, recipe,
            max_new_tokens=32,
            temperature=1.0, top_p=0.95,
            eos_id=eos_id,
            pad_id=0,
            rng=rng,
        )

        # Decode completions
        P_max = max(len(p) for p in expanded_p)
        completions = []
        for i, pids in enumerate(expanded_p):
            start = P_max - len(pids)
            active = int(gen_mask[i].sum().item())
            gen_ids = full_ids[i, start + len(pids):start + len(pids) + active].tolist()
            completions.append(tok.decode(gen_ids, skip_special_tokens=False))

        # Reward
        rewards_list = []
        for comp, answer, wt in zip(completions, expanded_a, expanded_wt):
            r, _ = reward_fn(comp, answer, wt, recipe, max_new_tokens=32)
            rewards_list.append(r)

        rewards = torch.tensor(rewards_list, dtype=torch.float, device=device)

        # Reference logprobs
        with torch.no_grad():
            ref_logp = compute_logprobs(ref or model, full_ids, gen_mask, _prompt_pad_mask)

        # Policy logprobs
        out = model(full_ids, use_cache=False)
        policy_logits = out["logits"][:, :-1, :].float()
        targets = full_ids[:, 1:]
        policy_logp = policy_logits.log_softmax(dim=-1).gather(
            -1, targets.unsqueeze(-1)).squeeze(-1)
        T = gen_mask.shape[1]
        policy_logp = policy_logp[:, -T:] * gen_mask

        # Loss
        loss, metrics = grpo_loss(
            policy_logp, ref_logp, rewards, gen_mask,
            group_size=2, kl_coef=0.01, clip_ratio=0.2, entropy_coef=0.01,
        )
        print(f"[smoke] loss {loss.item():+.4f} | pg {metrics['pg']:+.4f} | "
              f"kl {metrics['kl']:+.5f} | r̄ {metrics['reward_mean']:.2f}")

        # One batch is enough
        break

    # ---- checkpoint round-trip
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5, weight_decay=0.0,
    )
    out_dir = os.path.join(ckpt_dir, "grpo")
    os.makedirs(out_dir, exist_ok=True)
    save_grpo_checkpoint(
        out_dir, 1, model, optim, config,
        vars(argparse.Namespace(test=True)), False, recipe,
    )

    # Load back
    model2 = TransformerForCausalLM(config).to(device)
    inject_lora(model2, rank=4, alpha=8.0)
    load_step = load_grpo_checkpoint(
        os.path.join(out_dir, "grpo_step0000001.pt"),
        model2, None, device, is_lora=False,
    )
    print(f"[smoke] checkpoint round-trip: step {load_step}")

    shutil.rmtree(tmp)
    print("\n=== GRPO smoke test passed ===\n")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    rank, local_rank, world_size, device = setup_distributed()
    master = is_master(rank)

    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rng = random.Random(args.seed + rank)

    # ---------------------------------------------------------------- recipe
    recipe = recipe_from_args(args)
    if master:
        print(f"[Recipe] mode={recipe.mode}, model_name={recipe.model_name}")

    # ---------------------------------------------------------------- model
    if not args.checkpoint:
        raise FileNotFoundError("--checkpoint is required (SFT checkpoint).")

    ckpt_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**{k: v for k, v in ckpt_data["config"].items()
                            if k in ModelConfig.__init__.__code__.co_varnames})

    model = TransformerForCausalLM(config).to(device)
    model.load_state_dict(ckpt_data["model_state"])
    model.tie_weights()

    if master:
        n_total = count_parameters(model)
        print(f"Loaded SFT checkpoint: {n_total:,} params ({n_total/1e9:.3f}B)")

    # ----------------------------------------------------------------- LoRA
    is_lora = args.lora
    if is_lora:
        from peft.lora import inject_lora, lora_state_dict, freeze_base
        n_replaced = inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        freeze_base(model)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if master:
            print(f"[LoRA] injected {n_replaced} adapters | "
                  f"trainable={n_trainable:,} / total={count_parameters(model):,}")
    else:
        if master:
            print("[LoRA] disabled — full fine-tune")

    # --------------------------------------------------------------- compile
    _use_cudagraphs = False
    if args.compile:
        if master:
            print(f"[compile] torch.compile(mode='{args.compile_mode}')…")
        model = torch.compile(model, mode=args.compile_mode)
        _use_cudagraphs = (args.compile_mode == "reduce-overhead")

    # ----------------------------------------------------------------- ref
    ref_model = build_reference(args.ref_policy, config, args.checkpoint, device)
    ref_for_logprob = ref_model if ref_model is not None else model

    # ------------------------------------------------------------------- DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # --------------------------------------------------------------- tokenizer
    tokenizer = load_tokenizer(args.tokenizer)
    eos_id = get_special_token_id(tokenizer, recipe.eos_token)
    pad_id = get_special_token_id(tokenizer, recipe.pad_token)
    if master:
        print(f"[Tokenizer] eos_id={eos_id}, pad_id={pad_id}")

    # --------------------------------------------------------------- dataset
    train_ds = PackedGRPODataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
    )
    if master:
        print(f"[Dataset] {len(train_ds):,} prompts ({world_size} rank(s))")

    # --------------------------------------------------------------- auto LR scaling
    if not args.no_lr_scale:
        ref_hidden = 2048
        scale = math.sqrt(ref_hidden / config.hidden_size)
        scale = max(0.5, min(scale, 2.0))
        original_lr = args.lr
        args.lr = args.lr * scale
        args.min_lr = args.min_lr * scale
        if master:
            print(f"[LR] Auto-scaled from {original_lr:.2e} to {args.lr:.2e} "
                  f"(x{scale:.3f}, hidden={config.hidden_size})")

    # --------------------------------------------------------------- optim
    from optim.build_optimizer import build_optimizer
    optimizer = build_optimizer(
        model,
        optimizer_type="adamw",
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------------------- schedule
    from optim.lr_schedule import build_scheduler
    scheduler = build_scheduler(
        schedule="cosine",
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        peak_lr=args.lr,
        min_lr=args.min_lr,
    )

    # --------------------------------------------------------------- amp
    use_amp = device.type == "cuda" and args.dtype == "bf16"
    ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_amp else nullcontext())

    # --------------------------------------------------------------- resume
    start_step = 0
    if args.resume:
        start_step = load_grpo_checkpoint(
            args.resume, model, optimizer, device, is_lora
        )

    if master:
        os.makedirs(args.out_dir, exist_ok=True)
        eff_prompts = args.batch_size * world_size
        eff_completions = eff_prompts * args.num_generations
        print(f"\nEffective batch     : {eff_prompts} prompts "
              f"({eff_completions} completions)")
        print(f"Group size G        : {args.num_generations}")
        print(f"Max steps           : {args.max_steps:,}")
        print(f"Reference policy    : {args.ref_policy}")
        print(f"Checkpoint every    : {args.save_every:,} steps\n")

    # ================================================================= LOOP
    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    reward_window: List[float] = []
    correct_window: List[int] = []
    think_window: List[int] = []

    step = start_step
    data_iter = iter(train_ds)

    while step < args.max_steps:
        lr = scheduler(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # 1. sample a batch of prompts
        try:
            batch_prompts, batch_answers, batch_wt = next(data_iter)
        except StopIteration:
            data_iter = iter(train_ds)
            batch_prompts, batch_answers, batch_wt = next(data_iter)

        prompt_list = [p.tolist() for p in batch_prompts]

        # 2. expand to G replicas
        expanded_p: List[List[int]] = []
        expanded_a: List[str] = []
        expanded_wt: List[Optional[bool]] = []
        for _g in range(args.num_generations):
            for p, a, wt in zip(prompt_list, batch_answers, batch_wt):
                expanded_p.append(p)
                expanded_a.append(a)
                expanded_wt.append(wt)

        # 3. rollout
        rollout_model = _raw(model)
        rollout_model.eval()
        with ctx, torch.no_grad():
            full_ids, gen_mask, _sampled_lp, prompt_pad_mask = generate_rollouts(
                rollout_model, expanded_p, recipe,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_id=eos_id,
                pad_id=pad_id,
                rng=rng,
            )
        rollout_model.train()

        # 4. decode + reward
        completions_text: List[str] = []
        P_max = max(len(p) for p in expanded_p)
        for i, pids in enumerate(expanded_p):
            start = P_max - len(pids)
            active = int(gen_mask[i].sum().item())
            gen_ids = full_ids[i, start + len(pids):
                               start + len(pids) + active].tolist()
            text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            completions_text.append(text)

        rewards_list: List[float] = []
        for completion, answer, wt in zip(completions_text, expanded_a, expanded_wt):
            r, _info = reward_fn(
                completion, answer, wt, recipe,
                max_new_tokens=args.max_new_tokens,
                correct_weight=args.reward_correct,
                format_weight=args.reward_format,
            )
            rewards_list.append(r)
        rewards = torch.tensor(rewards_list, dtype=torch.float, device=device)

        # 5. reference log-probs (no_grad)
        with ctx, torch.no_grad():
            ref_logp = compute_logprobs(ref_for_logprob, full_ids, gen_mask, prompt_pad_mask)

        # 6. policy log-probs (with grad)
        policy_attn_mask = _build_attn_mask(
            prompt_pad_mask, full_ids.shape[1], 0, next(model.parameters()).dtype
        )
        T = gen_mask.shape[1]
        with ctx:
            out = model(full_ids, attention_mask=policy_attn_mask, use_cache=False,
                        num_logits_to_keep=T)
            policy_logits = out["logits"].float()
        targets = full_ids[:, -T:]
        policy_logp = policy_logits.log_softmax(dim=-1).gather(
            -1, targets.unsqueeze(-1)).squeeze(-1)
        policy_logp = policy_logp * gen_mask

        # 7. loss + step
        loss, metrics = grpo_loss(
            policy_logp, ref_logp, rewards, gen_mask,
            group_size=args.num_generations,
            kl_coef=args.kl_coef,
            clip_ratio=args.clip_ratio,
            entropy_coef=args.entropy_coeff,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        # 8. logging
        if master:
            reward_window.extend(rewards_list)
            window = max(1, len(reward_window))

            if step % args.log_interval == 0:
                t1 = time.perf_counter()
                sps = args.log_interval / max(t1 - t0, 1e-9)
                t0 = t1
                r_mean = sum(reward_window) / window
                reward_window.clear()
                print(
                    f"step {step:6d} | loss {loss.item():+.4f} | "
                    f"pg {metrics['pg']:+.4f} | kl {metrics['kl']:+.5f} | "
                    f"r̄ {r_mean:.2f} | lr {lr:.2e} | g {grad_norm:.2f} | "
                    f"{sps:.2f} step/s"
                )

        # 9. checkpoint
        if master and step > start_step and step % args.save_every == 0:
            save_grpo_checkpoint(
                args.out_dir, step, model, optimizer, config,
                vars(args), is_lora, recipe,
            )
            prune_checkpoints(args.out_dir, keep=args.keep_ckpts)

        step += 1

    # final
    if master:
        save_grpo_checkpoint(
            args.out_dir, step, model, optimizer, config,
            vars(args), is_lora, recipe,
        )
        print(f"\nGRPO complete. Final loss: {loss.item():.4f}")

    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="GRPO RL fine-tuning for reasoning / chat models.",
    )

    # Paths
    p.add_argument("--checkpoint", default=None,
                   help="SFT checkpoint from train_sft.py")
    p.add_argument("--tokenizer", default="./tokenizer",
                   help="Tokenizer directory from train_tokenizer.py")
    p.add_argument("--data-dir", default="./grpo_packed",
                   help="Packed GRPO data directory from data/pack_grpo.py")
    p.add_argument("--out-dir", default="./grpo_checkpoints",
                   help="Output directory for checkpoints")
    p.add_argument("--resume", default=None,
                   help="GRPO checkpoint to resume from")

    # LoRA
    p.add_argument("--lora", action="store_true",
                   help="Enable LoRA adapters")
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=float, default=128.0)

    # Reference policy
    p.add_argument("--ref-policy", default="single", choices=["single", "two"],
                   help="'single' reuses trainable model under no_grad; "
                        "'two' keeps a frozen copy in memory.")

    # Rollouts
    p.add_argument("--num-generations", type=int, default=8,
                   help="G — completions per prompt")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)

    # Reward weights
    p.add_argument("--reward-correct", type=float, default=1.0,
                   help="Reward for a fully correct + well-formatted answer")
    p.add_argument("--reward-format", type=float, default=0.3,
                   help="Reward for a wrong-but-well-formed answer")

    # GRPO loss
    p.add_argument("--kl-coef", type=float, default=0.02)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--entropy-coeff", type=float, default=0.0,
                   help="Entropy bonus coefficient (0 = disabled)")

    # Optim
    p.add_argument("--batch-size", type=int, default=4,
                   help="Prompts per step")
    p.add_argument("--num-steps", type=int, default=500,
                   help="Total training steps")
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-6,
                   help="Peak LR. Auto-scaled by model size unless --no-lr-scale.")
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--no-lr-scale", action="store_true",
                   help="Disable auto LR scaling by model size.")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", default="default",
                   choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--seed", type=int, default=42)

    # Checkpointing
    p.add_argument("--save-every", type=int, default=50,
                   help="Checkpoint interval in steps")
    p.add_argument("--keep-ckpts", type=int, default=3)
    p.add_argument("--log-interval", type=int, default=1)

    # Model
    p.add_argument("--model-size", type=str, default=None,
                   help="e.g. '1.7B', '7B' — uses ModelConfig.from_target_size()")

    # Recipe
    add_recipe_args(p)

    # Smoke test
    p.add_argument("--smoke-test", action="store_true",
                   help="Run a self-contained smoke test and exit")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.smoke_test:
        smoke_test()
    elif args.checkpoint is None:
        print("No --checkpoint given. Run with --smoke-test for a quick test, "
              "or provide --checkpoint for training.")
        smoke_test()
    else:
        train(args)
