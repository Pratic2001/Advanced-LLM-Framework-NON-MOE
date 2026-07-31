#!/usr/bin/env python3
"""
train_sft.py

Supervised Fine-Tuning (SFT) script — Stage 1 of reasoning post-training.

Loads the pretrained checkpoint produced by train.py and the packed memmap
files produced by pack_sft_data.py, and fine-tunes the model using a
ChatML + <think>...</think> template, before the RL stage (grpo.py).

This script does NOT read or tokenise raw JSONL — that is pack_sft_data.py's
job. Run pack_sft_data.py first (optionally with --worker/--num-workers to
parallelise packing across processes), then point --data-dir here at the
same directory.

Key features:
  - Loss masking: prompt tokens are masked, loss computed only on the
    assistant turn (thinking + answer), so the model learns to generate
    reasoning, not to predict the question
  - Sample packing: multiple variable-length examples are packed into one
    fixed-length window; a position-level mask prevents loss from bleeding
    across sample boundaries
  - LoRA / DoRA (optional): inject low-rank adapters into Q/K/V/O projections
    so fine-tuning a large model fits on a single GPU; adapters can be merged
    back into full weights for deployment
  - NEFTune noise injection for regularisation
  - DDP multi-GPU (torchrun) support
  - Checkpoint save/resume with LoRA-aware state handling
  - Training recipe system for mode / template / special-token management

Usage:
    # Full fine-tune (small model)
    python train_sft.py \\
        --checkpoint-dir ./pretrained \\
        --data-dir ./sft_packed \\
        --output-dir ./sft_checkpoints

    # LoRA fine-tune (recommended for 1B+ on a single GPU)
    python train_sft.py \\
        --checkpoint-dir ./pretrained \\
        --data-dir ./sft_packed \\
        --lora-rank 64 --lora-alpha 128 \\
        --output-dir ./sft_checkpoints

    # DoRA fine-tune
    python train_sft.py \\
        --checkpoint-dir ./pretrained \\
        --data-dir ./sft_packed \\
        --lora-rank 64 --lora-alpha 128 --lora-type dora \\
        --output-dir ./sft_checkpoints

    # Train from scratch with a target model size
    python train_sft.py \\
        --data-dir ./sft_packed \\
        --model-size 1.7B \\
        --output-dir ./sft_checkpoints

    # Multi-GPU
    torchrun --nproc_per_node=4 train_sft.py --checkpoint-dir ... --lora-rank 64

    # Merge LoRA weights back into the base model after training
    python train_sft.py --merge-and-save \\
        --checkpoint-dir ./sft_checkpoints \\
        --output-dir ./sft_merged
"""

import argparse
import atexit
import glob
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from model import ModelConfig, TransformerForCausalLM, add_architecture_args, apply_architecture_args, compute_mtp_loss, count_parameters
from optim.build_optimizer import build_optimizer
from optim.lr_schedule import build_scheduler
from recipe import TrainingRecipe, get_recipe, add_recipe_args, recipe_from_args

from atomic_io import atomic_symlink, atomic_torch_save, load_torch_checkpoint
from shutdown import install_signal_handlers, should_stop
from logging_utils import setup_logging, get_logger, log_event
from peft.lora import (
    inject_lora, merge_lora, lora_state_dict,
    freeze_base, build_lora_optimizer, neftune_noise, register_neftune_hook,
)


# ---------------------------------------------------------------------------
# GPU peak TFLOPS (bf16 Tensor Core, per card)
# ---------------------------------------------------------------------------
# Used for MFU estimation.  Add your GPU here if missing.

GPU_PEAK_TFLOPS = {
    # NVIDIA consumer
    "NVIDIA GeForce RTX 4090":    165.2,
    "NVIDIA GeForce RTX 4080":    97.5,
    "NVIDIA GeForce RTX 4070 Ti": 80.8,
    "NVIDIA GeForce RTX 4070":    59.8,
    "NVIDIA GeForce RTX 3090":    71.0,
    "NVIDIA GeForce RTX 3080":    59.4,
    # NVIDIA data-centre
    "NVIDIA A100-SXM4-80GB":      312.0,
    "NVIDIA A100-SXM4-40GB":      312.0,
    "NVIDIA A100-PCIE-40GB":      312.0,
    "NVIDIA H100 SXM5":           989.5,
    "NVIDIA H100 PCIe":           756.0,
    "NVIDIA L40S":                362.1,
    "NVIDIA L4":                  121.0,
    "NVIDIA A10G":                125.0,
    "NVIDIA V100-SXM2-16GB":      28.0,
    # AMD
    "AMD Instinct MI300X":        1307.4,
    "AMD Instinct MI250X":        383.0,
}

_FALLBACK_TFLOPS = 100.0


def get_gpu_peak_tflops(device: torch.device) -> float:
    if device.type != "cuda":
        return _FALLBACK_TFLOPS
    name = torch.cuda.get_device_name(device)
    for key, val in GPU_PEAK_TFLOPS.items():
        if key.lower() in name.lower() or name.lower() in key.lower():
            return val
    print(f"[MFU] Unknown GPU '{name}', using {_FALLBACK_TFLOPS} TFLOP/s fallback. "
          f"Add it to GPU_PEAK_TFLOPS for accurate MFU.")
    return _FALLBACK_TFLOPS


def estimate_mfu(model, tok_per_sec: float, gpu_peak_tflops: float) -> float:
    """Estimate model FLOP utilisation from throughput."""
    n_params = sum(p.numel() for p in model.parameters())
    # Approx: 6 * N params forward + backward per token
    model_tflops = 6.0 * n_params * tok_per_sec / 1e12
    return model_tflops / gpu_peak_tflops


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """Init DDP if launched with torchrun, else single-process."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank       = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank       = 0
        local_rank = 0
        world_size = 1
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, local_rank, world_size, device


def is_master(rank: int) -> bool:
    return rank == 0


def destroy_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# _ConcatMemmap — lazy zero-copy concatenation of memmap shards
# ---------------------------------------------------------------------------

class _ConcatMemmap:
    """
    Read-only view that makes several np.memmap arrays look like one
    contiguous array, without copying any of them into RAM.

    This avoids the multi-GB anonymous memory copies that
    ``np.concatenate(memmaps)`` forces, breaking the "RAM stays flat"
    promise when the dataset is large or world_size=1 (full dataset
    slice).
    """

    def __init__(self, arrays):
        self.arrays = [a for a in arrays if len(a) > 0]
        self.lengths = [len(a) for a in self.arrays]
        self.offsets = np.cumsum([0] + self.lengths)
        self.total = int(self.offsets[-1])

    def __len__(self):
        return self.total

    def __array__(self, dtype=None):
        # Supports np.asarray(concat_memmap) / np.array(concat_memmap).
        # Materializes exactly the pieces this instance holds — for a
        # per-step training window (this class's main use case via
        # __getitem__ below) that's tiny (seq_len-ish elements, usually
        # just 1-2 pieces). Callers holding a _ConcatMemmap spanning a
        # much larger range should avoid calling this unless they
        # specifically want the one-time full-materialization cost.
        if not self.arrays:
            return np.array([], dtype=dtype or np.uint16)
        pieces = [np.asarray(a) for a in self.arrays]
        out = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        return out.astype(dtype) if dtype is not None else out

    def _locate(self, idx):
        arr_i = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        return arr_i, idx - self.offsets[arr_i]

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.total)
            assert step == 1, "strided slicing not supported"
            if start >= stop:
                return _ConcatMemmap([])
            arr_i_start, local_start = self._locate(start)
            arr_i_end,   local_end   = self._locate(stop - 1)
            if arr_i_start == arr_i_end:
                return _ConcatMemmap([self.arrays[arr_i_start][local_start: local_end + 1]])
            pieces = []
            cur = start
            while cur < stop:
                arr_i, local = self._locate(cur)
                arr = self.arrays[arr_i]
                take = min(len(arr) - local, stop - cur)
                pieces.append(arr[local: local + take])
                cur += take
            return _ConcatMemmap(pieces)
        else:
            arr_i, local = self._locate(key)
            return self.arrays[arr_i][local]


# ---------------------------------------------------------------------------
# SFTDataset — manifest-based memmap reader (needed by GRPOPromptDataset)
# ---------------------------------------------------------------------------

class SFTDataset:
    """
    Reads packed memmap files produced by data/pack_sft.py:

        <cache_dir>/sft_train_manifest*.json   — per-worker metadata
        <cache_dir>/sft_train_tokens*.bin      — uint16/uint32 token ids
        <cache_dir>/sft_train_mask*.bin        — uint8 loss mask (1 = compute loss)

    All worker shards found for the requested split are discovered,
    sorted by worker index, and concatenated (still mmap-backed, no RAM
    copy). The result is sharded across DDP ranks by token count.

    This class is used by GRPOPromptDataset to scan sample boundaries;
    for SFT training itself, use PackedSFTDataLoader instead.
    """

    def __init__(
        self,
        cache_dir: str,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
        split: str = "train",
    ):
        self.seq_len    = seq_len
        self.rank       = rank
        self.world_size = world_size
        self.split      = split

        manifests = self._discover_manifests(cache_dir, split)
        if not manifests:
            raise FileNotFoundError(
                f"No packed manifests found for split={split!r} in {cache_dir}. "
                f"Run data/pack_sft.py first."
            )

        # pack_sft_data.py writes one manifest per worker with both train and
        # val file names, and picks the on-disk token dtype from vocab size
        # (uint16 when vocab <= 65536, else uint32). Read both from the
        # manifest instead of assuming uint16.
        dtype_t = np.dtype(manifests[0].get("dtype_t", "uint16"))
        dtype_m = np.dtype(manifests[0].get("dtype_m", "uint8"))

        tok_key  = f"{split}_tokens_file"   # e.g. train_tokens_file
        mask_key = f"{split}_mask_file"
        token_arrays = []
        mask_arrays  = []
        total_records = 0
        for m in manifests:
            tok_name  = m.get(tok_key)
            mask_name = m.get(mask_key)
            if not tok_name or not mask_name:
                continue  # this worker wrote no {split} records
            tok_path  = os.path.join(cache_dir, tok_name)
            mask_path = os.path.join(cache_dir, mask_name)
            token_arrays.append(np.memmap(tok_path,  dtype=dtype_t, mode="r"))
            mask_arrays.append(np.memmap(mask_path, dtype=dtype_m, mode="r"))
            total_records += m.get("n_records", 0)

        if not token_arrays:
            raise FileNotFoundError(
                f"No {split} shards found in the pack manifests under "
                f"{cache_dir}. Run data/pack_sft.py first."
            )

        self.tokens = _ConcatMemmap(token_arrays)
        self.mask   = _ConcatMemmap(mask_arrays)
        self.n_shards = len(token_arrays)

        if rank == 0:
            print(f"[SFTDataset] {split}: discovered {len(manifests)} worker "
                  f"shard(s) in {cache_dir} ({total_records:,} records total)")

        # Shard across DDP ranks by token count
        total = len(self.tokens)
        shard_size = total // world_size
        start = rank * shard_size
        end   = start + shard_size if rank < world_size - 1 else total
        self.tokens = self.tokens[start:end]
        self.mask   = self.mask[start:end]

        n_windows = max(0, (len(self.tokens) - 1) // seq_len)
        print(f"[SFTDataset rank {rank}] {split}: {len(self.tokens):,} tokens "
              f"-> {n_windows:,} windows of {seq_len}")

    # ------------------------------------------------------------------
    @staticmethod
    def _discover_manifests(cache_dir: str, split: str):
        """Find every sft_manifest*.json in cache_dir, sorted by worker.

        pack_sft_data.py writes one manifest per worker (``sft_manifest
        .w{i}-of-{n}.json``) containing both train and val file names, so
        no split filtering happens here — the caller picks the
        ``{split}_tokens_file`` / ``{split}_mask_file`` keys. ``split`` is
        kept as an argument for call-site clarity.
        """
        pattern = os.path.join(cache_dir, "sft_manifest*.json")
        manifest_paths = sorted(glob.glob(pattern))
        manifests = []
        for p in manifest_paths:
            with open(p, "r") as f:
                manifests.append(json.load(f))
        manifests.sort(key=lambda m: m.get("worker", 0))
        return manifests

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return max(0, (len(self.tokens) - 1) // self.seq_len)

    def get_batch(self, batch_size: int, device: torch.device):
        """Sample `batch_size` random windows; return (x, y, loss_mask)."""
        n = len(self)
        if n == 0:
            raise RuntimeError(
                "SFT dataset has no complete windows. "
                "Try a smaller --seq-len."
            )
        starts = torch.randint(0, n, (batch_size,)) * self.seq_len
        xs, ys, ms = [], [], []
        for s in starts.tolist():
            s = min(int(s), len(self.tokens) - self.seq_len - 1)
            xs.append(torch.from_numpy(
                np.asarray(self.tokens[s     : s + self.seq_len    ]).astype(np.int64)))
            ys.append(torch.from_numpy(
                np.asarray(self.tokens[s + 1 : s + self.seq_len + 1]).astype(np.int64)))
            ms.append(torch.from_numpy(
                np.asarray(self.mask  [s + 1 : s + self.seq_len + 1]).astype(np.float32)))
        x = torch.stack(xs)
        y = torch.stack(ys)
        m = torch.stack(ms)
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
            m = m.pin_memory().to(device, non_blocking=True)
        else:
            x, y, m = x.to(device), y.to(device), m.to(device)
        return x, y, m


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class PackedSFTDataLoader:
    """
    Streams fixed-length windows from flat memmap token + mask files.

    Reads ``sft_train_tokens*.bin`` and ``sft_train_mask*.bin`` from
    ``data_dir``, each shard sharded across DDP ranks by token count so
    no two GPUs see the same data in the same step.

    Yields (x, y, loss_mask, num_valid) per batch:
        x          : int64  (B, seq_len)  input token ids
        y          : int64  (B, seq_len)  shifted target ids
        loss_mask  : float32 (B, seq_len) 1 = compute loss, 0 = ignore
        num_valid  : int    total valid (non-masked) tokens in the batch
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        batch_size: int,
        pad_id: int,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.seq_len    = seq_len
        self.batch_size = batch_size
        self.pad_id     = pad_id
        self.rank       = rank

        # --- discover & mmap-concat token shards ---
        token_files = sorted(glob.glob(os.path.join(data_dir, "sft_train_tokens*.bin")))
        mask_files  = sorted(glob.glob(os.path.join(data_dir, "sft_train_mask*.bin")))
        if not token_files:
            raise FileNotFoundError(
                f"No sft_train_tokens*.bin files found in {data_dir}. "
                f"Run pack_sft_data.py first."
            )

        # pack_sft_data.py stores uint32 token ids when vocab_size > 65536;
        # read the dtype from the manifest instead of assuming uint16.
        dtype_t = np.dtype("uint16")
        manifests = sorted(glob.glob(os.path.join(data_dir, "sft_manifest*.json")))
        if manifests:
            with open(manifests[0], "r") as _f:
                dtype_t = np.dtype(json.load(_f).get("dtype_t", "uint16"))

        token_arrs = [np.memmap(f, dtype=dtype_t, mode="r") for f in token_files]
        mask_arrs  = [np.memmap(f, dtype=np.uint8,  mode="r") for f in mask_files]

        # Use _ConcatMemmap to lazily stitch memmap shards without copying
        # them into anonymous RAM. The rank-sharding slice below
        # (self.tokens[start:end]) can cover the *entire* dataset when
        # world_size=1 — eagerly concatenating would force a multi-GB
        # anonymous-memory copy that the OS can't reclaim like mmap'd pages.
        self.tokens = _ConcatMemmap(token_arrs)

        if mask_arrs:
            self.masks = _ConcatMemmap(mask_arrs)
        else:
            # Fallback: generate an all-ones mask (no masking)
            self.masks = np.ones(len(self.tokens), dtype=np.uint8)

        # --- shard across DDP ranks ---
        total = len(self.tokens)
        shard_size = total // world_size
        start = rank * shard_size
        end   = start + shard_size if rank < world_size - 1 else total
        self.tokens = self.tokens[start:end]
        self.masks  = self.masks[start:end]

        # number of complete windows
        self.n_windows = max(0, (len(self.tokens) - 1) // seq_len)

        if rank == 0:
            print(f"[DataLoader] {len(token_files)} token shard(s) in {data_dir} "
                  f"({total:,} tokens) -> {self.n_windows:,} windows "
                  f"(seq_len={seq_len})")

    def __len__(self) -> int:
        """Number of full batches this loader can yield."""
        return self.n_windows // self.batch_size

    def __iter__(self):
        """Yield (x, y, loss_mask, num_valid) batches sequentially."""
        n_batches = self.n_windows // self.batch_size
        for b in range(n_batches):
            xs, ys, ms = [], [], []
            valid_sum = 0
            for j in range(self.batch_size):
                idx = (b * self.batch_size + j) * self.seq_len
                x = np.asarray(self.tokens[idx     : idx + self.seq_len]).astype(np.int64)
                y = np.asarray(self.tokens[idx + 1 : idx + self.seq_len + 1]).astype(np.int64)
                m = np.asarray(self.masks[idx + 1  : idx + self.seq_len + 1]).astype(np.float32)
                xs.append(x)
                ys.append(y)
                ms.append(m)
                valid_sum += int(m.sum())

            xs = torch.from_numpy(np.stack(xs))
            ys = torch.from_numpy(np.stack(ys))
            ms = torch.from_numpy(np.stack(ms))

            if torch.cuda.is_available():
                xs = xs.pin_memory().to("cuda", non_blocking=True)
                ys = ys.pin_memory().to("cuda", non_blocking=True)
                ms = ms.pin_memory().to("cuda", non_blocking=True)

            yield xs, ys, ms, valid_sum


# ---------------------------------------------------------------------------
# Masked loss (only assistant tokens contribute)
# ---------------------------------------------------------------------------

def masked_cross_entropy(
    logits: torch.Tensor,    # (B, T, V)
    targets: torch.Tensor,   # (B, T)
    mask: torch.Tensor,      # (B, T) float, 1 = compute loss, 0 = ignore
) -> torch.Tensor:
    B, T, V = logits.shape
    logits_flat  = logits.reshape(B * T, V)
    targets_flat = targets.reshape(B * T)
    mask_flat    = mask.reshape(B * T)

    # token-level NLL
    nll = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    # mask and mean over active positions only
    denom = mask_flat.sum().clamp(min=1.0)
    return (nll * mask_flat).sum() / denom


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _raw(model):
    m = model.module if isinstance(model, DDP) else model
    return m._orig_mod if hasattr(m, "_orig_mod") else m


def save_sft_checkpoint(
    model,
    step: int,
    total_tokens: int,
    recipe: TrainingRecipe,
    is_lora: bool,
    base_weights: Optional[dict],
    args,
    best_val_loss: float = float("inf"),
    optimizer=None,
):
    """Save SFT checkpoint, recipe.json, and LoRA state dict if applicable."""
    raw = _raw(model)
    ckpt = {
        "step":          step,
        "total_tokens":  total_tokens,
        "model_state":   lora_state_dict(raw) if is_lora else raw.state_dict(),
        "is_lora":       is_lora,
        "best_val_loss": best_val_loss,
        "args":          vars(args),
    }
    if base_weights is not None:
        ckpt["base_weights_meta"] = {k: v.shape for k, v in base_weights.items()}
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f"sft_step{step:07d}.pt")
    # Atomic save — a crash mid-write leaves a tmp file, never a torn .pt
    # that latest.pt could point at.
    atomic_torch_save(ckpt, path)

    # Save recipe alongside checkpoint
    recipe.to_json(os.path.join(out_dir, "recipe.json"))

    # Symlink latest (atomic — no window where latest.pt is missing)
    latest = os.path.join(out_dir, "latest.pt")
    atomic_symlink(path, latest)

    if is_lora:
        lora_path = os.path.join(out_dir, f"sft_step{step:07d}_lora.pt")
        atomic_torch_save(lora_state_dict(raw), lora_path)
        print(f"[Checkpoint] saved LoRA state dict to {lora_path}")

    print(f"[Checkpoint] saved {path}")
    log_event(get_logger(), "checkpoint_saved", step=step, path=path,
              is_lora=is_lora)
    return path


def load_sft_checkpoint(
    path: str,
    model,
    optimizer,
    device,
    is_lora: bool,
):
    """
    Load SFT checkpoint.  Handles both full-FT and LoRA checkpoints.

    For LoRA: base weights must already be loaded; this loads only the
    adapter weights + optimizer state.
    """
    ckpt  = load_torch_checkpoint(path, map_location=device)
    raw   = _raw(model)
    state = ckpt["model_state"]

    if is_lora:
        missing, unexpected = raw.load_state_dict(state, strict=False)
        lora_keys = [k for k in state if "lora_A" in k or "lora_B" in k]
        print(f"[Checkpoint] loaded {len(lora_keys)} LoRA tensors from {path}")
    else:
        raw.load_state_dict(state)
        if hasattr(raw, "tie_weights"):
            raw.tie_weights()

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    step          = ckpt.get("step", 0)
    total_tokens  = ckpt.get("total_tokens", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    print(f"[Checkpoint] resumed from step {step}  ({total_tokens:,} tokens) "
          f"best_val={best_val_loss:.4f}")
    return step, total_tokens, best_val_loss


def prune_checkpoints(out_dir: str, keep: int = 3):
    ckpts = sorted(
        Path(out_dir).glob("sft_step*.pt"),
        key=lambda p: int(p.stem.replace("sft_step", "")),
    )
    # also remove corresponding _lora.pt files
    for old in ckpts[:-keep]:
        old.unlink()
        lora_path = old.with_name(old.stem + "_lora.pt")
        if lora_path.exists():
            lora_path.unlink()
        print(f"[Checkpoint] pruned {old.name}")


# ---------------------------------------------------------------------------
# Eval pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_loader, pad_id: int) -> float:
    """
    Sequential evaluation: walk through val windows in order.
    Returns the mean masked cross-entropy loss.
    """
    model.eval()
    total_loss  = 0.0
    total_valid = 0

    for x, y, m, num_valid in val_loader:
        out    = model(x)
        logits = out["logits"]
        B, T, V = logits.shape

        logits_flat  = logits.reshape(B * T, V)
        targets_flat = y.reshape(B * T)
        mask_flat    = m.reshape(B * T)

        nll = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        weighted_nll = (nll * mask_flat).sum()
        total_loss  += weighted_nll.item()
        total_valid += num_valid

    model.train()
    return total_loss / max(total_valid, 1)


# ---------------------------------------------------------------------------
# Merge-only mode: fuse LoRA into base weights and save
# ---------------------------------------------------------------------------

def merge_and_save_mode(args, rank, world_size, device):
    """Merge LoRA weights into base model and save the full model."""
    ckpt_dir = args.checkpoint_dir

    # Load latest checkpoint
    latest = os.path.join(ckpt_dir, "latest.pt")
    if not os.path.exists(latest):
        # Find latest by globbing
        candidates = sorted(Path(ckpt_dir).glob("sft_step*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        latest = str(candidates[-1])

    ckpt = load_torch_checkpoint(latest, map_location="cpu")
    is_lora = ckpt.get("is_lora", False)
    if not is_lora:
        print("[Merge] Checkpoint is not LoRA; nothing to merge. Saving as-is.")
        out_path = os.path.join(args.output_dir, "merged_model.pt")
        os.makedirs(args.output_dir, exist_ok=True)
        atomic_torch_save({"model_state": ckpt["model_state"], "config": ckpt.get("args", {})}, out_path)
        print(f"[Merge] saved to {out_path}")
        return

    # Resolve model config
    saved_args = ckpt.get("args", {})
    config = _resolve_model_config(saved_args)

    # Build base model
    model = TransformerForCausalLM(config).to(device)

    # Load base weights from the original pretrain checkpoint
    base_ckpt_path = saved_args.get("checkpoint_dir")
    if base_ckpt_path and os.path.isdir(base_ckpt_path):
        base_latest = os.path.join(base_ckpt_path, "latest.pt")
        if os.path.exists(base_latest):
            base_ckpt = load_torch_checkpoint(base_latest, map_location=device)
            if "model_state" in base_ckpt:
                model.load_state_dict(base_ckpt["model_state"])
                if hasattr(model, "tie_weights"):
                    model.tie_weights()
                print(f"[Merge] loaded base weights from {base_latest}")
            else:
                print("[Merge] WARNING: base checkpoint has no 'model_state' key")
        else:
            print("[Merge] WARNING: base latest.pt not found; "
                  "LoRA will be merged onto random weights.")
    else:
        print("[Merge] WARNING: --checkpoint-dir not found; "
              "LoRA will be merged onto random weights.")

    # Inject LoRA with the same hyper-parameters as training
    inject_lora(
        model,
        rank=saved_args.get("lora_rank", 64),
        alpha=saved_args.get("lora_alpha", 128.0),
        target_modules=tuple(saved_args.get("lora_target_modules", "q_proj,k_proj,v_proj,o_proj").split(",")),
        lora_type=saved_args.get("lora_type", "lora"),
    )

    # Load LoRA weights
    model.load_state_dict(ckpt["model_state"], strict=False)

    # Merge
    model = merge_lora(model)
    if hasattr(model, "tie_weights"):
        model.tie_weights()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "merged_model.pt")
    atomic_torch_save({"model_state": model.state_dict(), "config": vars(config)}, out_path)
    print(f"[Merge] saved merged model to {out_path}")


def _resolve_model_config(saved_args: dict) -> ModelConfig:
    """Reconstruct a ModelConfig from saved checkpoint args."""
    # Check if there is a saved config dict (from ModelConfig)
    config_fields = {}
    model_size_keys = [
        "vocab_size", "hidden_size", "intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "max_position_embeddings", "rms_norm_eps", "rope_theta",
        "tie_word_embeddings", "scale_emb", "norm_type", "mlp_type",
        "use_qk_norm", "attn_type",
    ]
    for k in model_size_keys:
        if k in saved_args:
            config_fields[k] = saved_args[k]

    if "model_size" in saved_args and saved_args["model_size"]:
        target = ModelConfig.parse_param_count(saved_args["model_size"])
        return ModelConfig.from_target_size(target_params=target, **config_fields)

    return ModelConfig(**config_fields)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def smoke_test():
    """
    Self-contained smoke test using tiny model and synthetic packed data.
    Exercises the real SFT path: checkpoint load, data loading, masked CE
    loss, backward, optimizer step, and checkpoint round-trip.
    """
    print("\n=== SFT smoke test ===")
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp()
    data_dir = os.path.join(tmp, "sft_data")
    ckpt_dir = os.path.join(tmp, "ckpts")
    out_dir  = os.path.join(tmp, "out")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- tiny model + fake pretrained checkpoint ----
    config = ModelConfig(
        vocab_size=512, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, max_position_embeddings=256,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerForCausalLM(config).to(device)
    sft_ckpt_path = os.path.join(ckpt_dir, "latest.pt")
    atomic_torch_save(
        {"model_state": model.state_dict(), "config": vars(config)},
        sft_ckpt_path,
    )

    # ---- synthetic packed data (flat token + mask memmaps) ----
    seq_len = 64
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 512, size=seq_len * 8, dtype=np.uint16)
    toks.tofile(os.path.join(data_dir, "sft_train_tokens_000.bin"))
    np.ones(seq_len * 8, dtype=np.uint8).tofile(
        os.path.join(data_dir, "sft_train_mask_000.bin")
    )
    with open(os.path.join(data_dir, "sft_manifest_000.json"), "w") as f:
        json.dump({"dtype_t": "uint16"}, f)

    # ---- dataset ----
    ds = PackedSFTDataLoader(
        data_dir=data_dir, seq_len=seq_len, batch_size=2,
        pad_id=0, rank=0, world_size=1,
    )
    print(f"[smoke] dataset size: {len(ds)} batches")

    # ---- load pretrained weights (real code path) ----
    model2 = TransformerForCausalLM(config).to(device)
    load_sft_checkpoint(sft_ckpt_path, model2, None, device, is_lora=False)
    model2.train()

    # ---- one real training step ----
    opt = build_optimizer(model2, lr=1e-4, weight_decay=0.0)
    for x, y, m, num_valid in ds:
        x, y, m = x.to(device), y.to(device), m.to(device)
        out = model2(x)
        loss = masked_cross_entropy(out["logits"], y, m)
        print(f"[smoke] loss {loss.item():.4f} | num_valid {num_valid}")
        loss.backward()
        nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        print("[smoke] backward + optimizer step: OK")
        break  # one batch is enough

    # ---- checkpoint round-trip ----
    args = argparse.Namespace(output_dir=out_dir)
    save_sft_checkpoint(
        model2, 1, num_valid, TrainingRecipe(mode="reasoning"),
        is_lora=False, base_weights=None, args=args,
        best_val_loss=1.0, optimizer=opt,
    )
    model3 = TransformerForCausalLM(config).to(device)
    step, total_tokens, _best = load_sft_checkpoint(
        os.path.join(out_dir, "sft_step0000001.pt"),
        model3, None, device, is_lora=False,
    )
    print(f"[smoke] checkpoint round-trip: step {step} ({total_tokens} tokens)")

    shutil.rmtree(tmp)
    print("\n=== SFT smoke test passed ===\n")


def main():
    # ====================================================================
    # Argument parser
    # ====================================================================
    p = argparse.ArgumentParser(description="SFT fine-tuning (DDP + LoRA/DoRA)")

    # --- mode ---
    p.add_argument("--merge-and-save", action="store_true",
                   help="Merge LoRA into base model and save; skip training")

    # --- paths ---
    p.add_argument("--checkpoint-dir", default=None,
                   help="Pretrained checkpoint directory (from train.py)")
    p.add_argument("--data-dir",       default="./sft_packed",
                   help="Packed memmap files produced by pack_sft_data.py")
    p.add_argument("--output-dir",     default="./sft_checkpoints")
    p.add_argument("--resume",         default=None,
                   help="SFT checkpoint to resume from")

    # --- model ---
    p.add_argument("--model-size",     default=None,
                   help="Target model size (e.g. '1.7B', '70B', '300B', '1T'); creates model from "
                        "scratch using ModelConfig.from_target_size when no "
                        "--checkpoint-dir is given")

    # --- LoRA / DoRA ---
    p.add_argument("--lora-rank",              type=int,   default=64)
    p.add_argument("--lora-alpha",             type=float, default=128.0)
    p.add_argument("--lora-target-modules",    type=str,
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                   help="Comma-separated list of target module suffixes")
    p.add_argument("--lora-type",              type=str, default="lora",
                   choices=["lora", "dora"],
                   help="Adapter type: standard LoRA or DoRA")
    p.add_argument("--use-rslora",             action="store_true",
                   help="Use rank-stabilised LoRA (rsLoRA)")
    p.add_argument("--lora-lr-ratio",          type=float, default=1.0,
                   help="LR multiplier for LoRA params relative to --lr")

    # --- NEFTune ---
    p.add_argument("--neftune-alpha", type=float, default=0.0,
                   help="NEFTune noise alpha (0 = disabled)")

    # --- recipe / mode ---
    add_recipe_args(p)

    # --- training ---
    p.add_argument("--seq-len",        type=int,   default=2048)
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="Max position embeddings (default 8192). Separate from --seq-len.")
    add_architecture_args(p)
    p.add_argument("--batch-size",     type=int,   default=4)
    p.add_argument("--num-steps",      type=int,   default=10_000)
    p.add_argument("--grad-accum",     type=int,   default=4)
    p.add_argument("--lr",             type=float, default=2e-5,
                   help="Peak LR (typically 1e-5 to 5e-5 for SFT). "
                        "Auto-scaled by model size unless --no-lr-scale.")
    p.add_argument("--min-lr",         type=float, default=2e-6)
    p.add_argument("--no-lr-scale",    action="store_true",
                   help="Disable auto LR scaling by model size.")
    p.add_argument("--z-loss-weight",  type=float, default=1e-4,
                   help="Z-loss coefficient (0=disabled). "
                        "Penalises large logit magnitudes for stable training.")
    p.add_argument("--weight-decay",   type=float, default=0.01)
    p.add_argument("--warmup-steps",   type=int,   default=200)
    p.add_argument("--grad-clip",      type=float, default=1.0)
    p.add_argument("--dtype",          default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--compile",        action="store_true")
    p.add_argument("--compile-mode",   default="default",
                   choices=["default", "reduce-overhead", "max-autotune"])

    # --- checkpointing / logging ---
    p.add_argument("--ckpt-interval",  type=int, default=1_000)
    p.add_argument("--keep-ckpts",     type=int, default=3)
    p.add_argument("--log-interval",   type=int, default=10)
    p.add_argument("--eval-interval",  type=int, default=200)
    p.add_argument("--eval-steps",     type=int, default=20)

    # --- validation ---
    p.add_argument("--val-fraction", type=float, default=0.05,
                   help="Fraction of training data to hold out for validation "
                        "(used when no separate val files exist)")

    # --- smoke test ---
    p.add_argument("--smoke-test", action="store_true",
                   help="Run a self-contained smoke test and exit")

    args = p.parse_args()

    # Namespace of pure argparse defaults, used to detect which architecture
    # flags the user explicitly set (see apply_architecture_args).
    arch_defaults = p.parse_args([])

    # ------------------------------------------------------------------
    # Smoke test mode
    # ------------------------------------------------------------------
    if args.smoke_test:
        smoke_test()
        return

    # ====================================================================
    # Merge-and-save mode
    # ====================================================================
    if args.merge_and_save:
        merge_and_save_mode(args, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return

    # ====================================================================
    # Distributed setup
    # ====================================================================
    rank, local_rank, world_size, device = setup_distributed()
    master = is_master(rank)

    # Ensure the process group is torn down even if an exception (e.g. OOM)
    # unwinds main(). Idempotent: destroy_distributed checks is_initialized().
    atexit.register(destroy_distributed)

    # Graceful shutdown on SIGINT/SIGTERM (scheduler preemption, Ctrl+C).
    install_signal_handlers()
    log = setup_logging()
    log_event(log, "sft_start",
              model_size=args.model_size, num_steps=args.num_steps,
              lora_rank=args.lora_rank, resume=args.resume)

    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32        = True

    # ====================================================================
    # Recipe
    # ====================================================================
    recipe = recipe_from_args(args)

    # ====================================================================
    # Model
    # ====================================================================
    use_pretrained = args.checkpoint_dir is not None and not args.model_size

    if use_pretrained:
        # Load config from pretrained checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Run train.py first to produce a pretrained checkpoint."
            )
        ckpt_data = load_torch_checkpoint(ckpt_path, map_location="cpu")
        config    = ModelConfig(**ckpt_data["config"])

    elif args.model_size:
        # Build config from scratch using target size
        target_params = ModelConfig.parse_param_count(args.model_size)
        max_pos = getattr(args, "max_seq_len", None) or 8192
        config = ModelConfig.from_target_size(
            target_params=target_params,
            max_position_embeddings=max_pos,
        )

    else:
        raise ValueError(
            "Either --checkpoint-dir or --model-size must be provided."
        )

    # Apply architecture CLI flags BEFORE constructing the model so --arch,
    # --num-mtp-heads, --use-mla etc. take effect. Only flags the user
    # explicitly set are applied, preserving the checkpoint's architecture
    # (e.g. jamba / MTP heads) unless deliberately overridden.
    apply_architecture_args(config, args, defaults=arch_defaults)

    model = TransformerForCausalLM(config).to(device)

    if use_pretrained:
        model.load_state_dict(ckpt_data["model_state"])
        if hasattr(model, "tie_weights"):
            model.tie_weights()

        if master:
            n_total = count_parameters(model)
            print(f"Loaded pretrained model: {n_total:,} params ({n_total/1e9:.3f}B)")

    elif args.model_size:
        if master:
            n_total = count_parameters(model)
            print(f"Created model from --model-size {args.model_size}: "
                  f"{n_total:,} params ({n_total/1e9:.3f}B)")
            print(f"  config: H={config.hidden_size} L={config.num_hidden_layers} "
                  f"heads={config.num_attention_heads} kv={config.num_key_value_heads} "
                  f"I={config.intermediate_size}")

    # Store config fields in args for checkpoint round-trip
    args.vocab_size          = config.vocab_size
    args.hidden_size         = config.hidden_size
    args.intermediate_size   = config.intermediate_size
    args.num_hidden_layers   = config.num_hidden_layers
    args.num_attention_heads = config.num_attention_heads
    args.num_key_value_heads = config.num_key_value_heads
    args.head_dim            = config.head_dim
    args.max_position_embeddings = config.max_position_embeddings
    args.rms_norm_eps        = config.rms_norm_eps
    args.rope_theta          = config.rope_theta
    args.tie_word_embeddings = config.tie_word_embeddings
    args.scale_emb           = config.scale_emb
    args.norm_type           = config.norm_type
    args.mlp_type            = config.mlp_type
    args.use_qk_norm         = config.use_qk_norm
    args.attn_type           = config.attn_type

    # ====================================================================
    # LoRA / DoRA
    # ====================================================================
    has_lora = args.lora_rank > 0
    if has_lora:
        target_modules = tuple(m.strip() for m in args.lora_target_modules.split(","))
        n_replaced = inject_lora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_type=args.lora_type,
        )
        freeze_base(model)

        # Count LoRA params
        n_lora = sum(p.numel() for n, p in model.named_parameters()
                     if "lora_A" in n or "lora_B" in n)
        n_total = count_parameters(model)
        if master:
            print(f"[{args.lora_type.upper()}] injected {n_replaced} adapters | "
                  f"lora_params={n_lora:,} / total={n_total:,} "
                  f"({100*n_lora/n_total:.2f}%)")
    else:
        if master:
            print("[LoRA] disabled — full fine-tune")

    # ====================================================================
    # torch.compile
    # ====================================================================
    _use_cudagraphs = False
    if args.compile:
        mode = args.compile_mode
        if master:
            print(f"[compile] torch.compile(mode='{mode}')...")
        model = torch.compile(model, mode=mode)
        _use_cudagraphs = (mode == "reduce-overhead")

    # ====================================================================
    # DDP
    # ====================================================================
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # ====================================================================
    # NEFTune
    # ====================================================================
    if args.neftune_alpha > 0:
        if master:
            print(f"[NEFTune] alpha={args.neftune_alpha}")
        register_neftune_hook(model, args.neftune_alpha)

    # ====================================================================
    # Data
    # ====================================================================
    if master:
        print(f"\nReading packed SFT data from {args.data_dir} ...")

    pad_id = 0  # default pad token id
    train_loader = PackedSFTDataLoader(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        pad_id=pad_id,
        rank=rank,
        world_size=world_size,
    )

    # Build a validation loader if val files exist, otherwise use train
    val_data_dir = args.data_dir  # same dir; look for val files
    val_token_files = glob.glob(os.path.join(val_data_dir, "sft_val_tokens*.bin"))
    if val_token_files:
        val_loader = PackedSFTDataLoader(
            data_dir=val_data_dir,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            pad_id=pad_id,
            rank=rank,
            world_size=world_size,
        )
        # Override to use val files
        val_token_files_sorted = sorted(val_token_files)
        val_mask_files = sorted(glob.glob(os.path.join(val_data_dir, "sft_val_mask*.bin")))
        # Token dtype is set by pack_sft_data.py from vocab size (uint32 for
        # vocabs > 65536) — mirror PackedSFTDataLoader's manifest lookup.
        val_dtype_t = np.dtype("uint16")
        val_manifests = sorted(glob.glob(os.path.join(val_data_dir, "sft_manifest*.json")))
        if val_manifests:
            with open(val_manifests[0], "r") as _f:
                val_dtype_t = np.dtype(json.load(_f).get("dtype_t", "uint16"))
        val_token_arrs = [np.memmap(f, dtype=val_dtype_t, mode="r") for f in val_token_files_sorted]
        val_mask_arrs  = [np.memmap(f, dtype=np.uint8, mode="r") for f in val_mask_files] if val_mask_files else None
        if val_token_arrs:
            val_loader.tokens = _ConcatMemmap(val_token_arrs)
            if val_mask_arrs:
                val_loader.masks = _ConcatMemmap(val_mask_arrs)
            else:
                val_loader.masks = np.ones(len(val_loader.tokens), dtype=np.uint8)
            total_val = len(val_loader.tokens)
            shard_size = total_val // world_size
            start_v = rank * shard_size
            end_v   = start_v + shard_size if rank < world_size - 1 else total_val
            val_loader.tokens = val_loader.tokens[start_v:end_v]
            val_loader.masks  = val_loader.masks[start_v:end_v]
            val_loader.n_windows = max(0, (len(val_loader.tokens) - 1) // args.seq_len)
            if master:
                print(f"[DataLoader] val: {len(val_token_files_sorted)} shard(s) "
                      f"({total_val:,} tokens) -> {val_loader.n_windows:,} windows")
    else:
        if master:
            print("[DataLoader] No separate val files found; "
                  "validation will run on a fraction of training data.")
        # Use the training loader for validation
        val_loader = train_loader

    # ====================================================================
    # Auto LR scaling
    # ====================================================================
    if args.model_size and not args.no_lr_scale:
        ref_hidden = 2048
        scale = math.sqrt(ref_hidden / config.hidden_size)
        scale = max(0.5, min(scale, 2.0))
        original_lr = args.lr
        args.lr = args.lr * scale
        args.min_lr = args.min_lr * scale
        if master:
            print(f"[LR] Auto-scaled from {original_lr:.2e} to {args.lr:.2e} "
                  f"(×{scale:.3f}, hidden={config.hidden_size})")

    if len(train_loader) == 0:
        raise RuntimeError(
            "Training dataset has no complete windows. "
            "Try a smaller --seq-len, or re-run pack_sft_data.py with more data."
        )

    # ====================================================================
    # Optimizer
    # ====================================================================
    if has_lora and args.lora_lr_ratio != 1.0:
        optimizer = build_lora_optimizer(
            model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lora_lr_ratio=args.lora_lr_ratio,
        )
    else:
        optimizer = build_optimizer(
            model,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    # ====================================================================
    # LR scheduler
    # ====================================================================
    scheduler = build_scheduler(
        schedule="cosine",
        warmup_steps=args.warmup_steps,
        max_steps=args.num_steps,
        peak_lr=args.lr,
        min_lr=args.min_lr,
    )

    # ====================================================================
    # AMP
    # ====================================================================
    # bf16 needs autocast but no GradScaler (bf16 has the same exponent range
    # as fp32, so dynamic-range scaling is unnecessary). fp16 still needs
    # GradScaler to avoid underflow. The GradScaler is therefore only
    # enabled for fp16, and is a no-op scaler.step/update for bf16.
    use_amp   = device.type == "cuda" and args.dtype in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    ctx       = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if use_amp
        else nullcontext()
    )
    use_scaler = use_amp and args.dtype == "fp16"
    scaler     = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ====================================================================
    # Resume
    # ====================================================================
    start_step     = 0
    total_tokens   = 0
    best_val_loss  = float("inf")
    base_weights   = None

    if args.resume:
        start_step, total_tokens, best_val_loss = load_sft_checkpoint(
            args.resume, model, optimizer, device, has_lora,
        )

    # Save base weights reference for LoRA (so we can re-inject on resume)
    if has_lora:
        base_weights = {}
        for k, v in _raw(model).state_dict().items():
            if "lora_A" not in k and "lora_B" not in k:
                base_weights[k] = v.clone().cpu()

    if master:
        os.makedirs(args.output_dir, exist_ok=True)

    tokens_per_step = (
        args.batch_size * args.seq_len * args.grad_accum * world_size
    )
    gpu_peak = get_gpu_peak_tflops(device)

    if master:
        print(f"\nTokens / step    : {tokens_per_step:,}")
        print(f"Effective batch  : {args.batch_size * args.grad_accum * world_size}")
        print(f"Max steps        : {args.num_steps:,}")
        print(f"Checkpoint every : {args.ckpt_interval:,} steps")
        print(f"Peak LR          : {args.lr:.2e}")
        print(f"Min LR           : {args.min_lr:.2e}")
        print(f"Warmup steps     : {args.warmup_steps:,}")
        print(f"Weight decay     : {args.weight_decay}")
        print(f"Grad clip        : {args.grad_clip}")
        print(f"LoRA             : {'on' if has_lora else 'off'}")
        if has_lora:
            print(f"  type           : {args.lora_type}")
            print(f"  rank           : {args.lora_rank}")
            print(f"  alpha          : {args.lora_alpha}")
            print(f"  lr_ratio       : {args.lora_lr_ratio}")
        print(f"NEFTune alpha    : {args.neftune_alpha}")
        print(f"Recipe mode      : {recipe.mode}\n")

    # ====================================================================
    # Training loop
    # ====================================================================
    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0          = time.perf_counter()
    loss_accum  = 0.0
    tokens_accum = 0

    train_iter = iter(train_loader)

    interrupted = False
    for step in range(start_step, args.num_steps):
        # --- graceful shutdown (collective-safe: all ranks agree to stop)
        if should_stop(device, world_size):
            interrupted = True
            if master:
                print(f"\n[Shutdown] requested at step {step} — "
                      f"saving final checkpoint …")
                log_event(log, "shutdown_requested", step=step)
            break

        # --- LR schedule
        lr = scheduler(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # --- gradient accumulation
        for micro in range(args.grad_accum):
            try:
                x, y, m, num_valid = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y, m, num_valid = next(train_iter)

            sync    = (micro == args.grad_accum - 1)
            ctx_ddp = nullcontext() if (world_size == 1 or sync) else model.no_sync()

            with ctx_ddp:
                with ctx:
                    if _use_cudagraphs:
                        torch.compiler.cudagraph_mark_step_begin()
                    out    = model(x)
                    logits = out["logits"]
                # masked loss: only assistant tokens contribute
                loss = masked_cross_entropy(logits, y, m) / args.grad_accum
                # Z-loss: penalise large logits for training stability
                if args.z_loss_weight > 0:
                    loss = loss + (args.z_loss_weight * logits.float().square().mean() / args.grad_accum)
                # MTP loss
                if "mtp_logits" in out:
                    pad_id_val = pad_id if isinstance(pad_id, int) else 0
                    mtp_loss = compute_mtp_loss(
                        out["mtp_logits"], y,
                        discount=model.config.mtp_discount if hasattr(model, 'config') else 0.5,
                        ignore_index=pad_id_val,
                    )
                    loss = loss + mtp_loss / args.grad_accum
                # MoD loss
                loss += out.get("mod_aux_loss", 0.0) / args.grad_accum

            # Backward (GradScaler only meaningful for fp16).
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_accum  += loss.item()
            tokens_accum += num_valid

        # --- optimiser step
        if use_scaler:
            scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        ).item()
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # --- logging
        if master and step % args.log_interval == 0:
            t1  = time.perf_counter()
            elapsed = max(t1 - t0, 1e-9)
            tok_per_sec = tokens_accum / elapsed
            mfu = estimate_mfu(model, tok_per_sec, gpu_peak)
            loss_display = loss_accum / args.log_interval
            loss_accum   = 0.0
            tokens_accum = 0
            print(
                f"step {step:7d} | loss {loss_display:.4f} | lr {lr:.2e} | "
                f"grad {grad_norm:.3f} | {tok_per_sec/1e3:.1f}k tok/s | "
                f"mfu {mfu*100:.2f}%"
            )
            t0 = t1

        # --- validation
        if step % args.eval_interval == 0 and step > start_step:
            val_loss = validate(model, val_loader, pad_id)
            if world_size > 1:
                vl = torch.tensor(val_loss, device=device)
                dist.all_reduce(vl, op=dist.ReduceOp.AVG)
                val_loss = vl.item()
            if master:
                improved = " [BEST]" if val_loss < best_val_loss else ""
                print(f"  [eval] step {step:7d} | val_loss {val_loss:.4f}{improved}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

        # --- checkpoint
        if master and step % args.ckpt_interval == 0 and step > start_step:
            total_tokens += tokens_per_step * args.log_interval  # approximate
            save_sft_checkpoint(
                model, step, total_tokens, recipe,
                has_lora, base_weights, args,
                best_val_loss=best_val_loss,
                optimizer=optimizer,
            )
            prune_checkpoints(args.output_dir, keep=args.keep_ckpts)

    # ====================================================================
    # Final checkpoint
    # ====================================================================
    if master:
        # On graceful shutdown `step` is exactly the next optimizer step to
        # run (we broke at the top of iteration `step`), so saving it makes
        # resume continue where we stopped — no lost progress.
        final_step = step if interrupted else args.num_steps
        save_sft_checkpoint(
            model, final_step, total_tokens, recipe,
            has_lora, base_weights, args,
            best_val_loss=best_val_loss,
            optimizer=optimizer,
        )
        if interrupted:
            print(f"\n[Shutdown] checkpoint saved at step {final_step}. "
                  f"Resume with --resume {os.path.join(args.output_dir, 'latest.pt')}")
            log_event(log, "shutdown_checkpoint_saved", step=final_step)
        else:
            print(f"\nSFT complete. Best val loss: {best_val_loss:.4f}")
            log_event(log, "training_complete", step=final_step,
                      best_val_loss=best_val_loss)
            if has_lora:
                print(f"\nTo merge LoRA into base weights for deployment:")
                print(f"  python train_sft.py --merge-and-save "
                      f"--checkpoint-dir {args.output_dir} "
                      f"--output-dir ./sft_merged")

    destroy_distributed()


# ---------------------------------------------------------------------------
# DDP launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
