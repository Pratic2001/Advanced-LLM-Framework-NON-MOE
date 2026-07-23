#!/usr/bin/env python3
"""
train_pretrain.py

Production pretraining loop for the generic dense transformer built in model.py.
Reads packed memmap token files (pretrain_tokens*.bin) and trains with:

  - bf16 mixed precision
  - torch.compile for kernel fusion (+25-40% throughput)
  - Gradient accumulation (simulate large batch on few GPUs)
  - Cosine or WSD LR schedule with linear warmup
  - AdamW or Muon optimizer
  - Gradient clipping
  - DDP multi-GPU
  - Periodic validation loss
  - Checkpoint save/resume (step_XXXXX.pt + config.json + meta.json + recipe.json)
  - MFU logging with architecture-aware ops count
  - Optional Weights & Biases logging

Single GPU:
    python train_pretrain.py --model-size 0.6B --data-dir ./packed --checkpoint-dir ./checkpoints

Multi-GPU (e.g. 4 GPUs on one node):
    torchrun --nproc_per_node=4 train_pretrain.py --model-size 0.6B --data-dir ./packed

Resume from checkpoint:
    python train_pretrain.py --resume /path/to/step_00000.pt

Resume from checkpoint directory (latest step):
    python train_pretrain.py --resume /path/to/checkpoints

Recommended command for RTX 4090 (0.3B model):
    python train_pretrain.py --model-size 0.3B --data-dir ./packed \\
        --seq-len 2048 --batch-size 32 --grad-accum 4 \\
        --jit --checkpoint-dir ./checkpoints
"""

import argparse
import glob
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from model import ModelConfig, TransformerForCausalLM, count_parameters
from optim.build_optimizer import build_optimizer
from optim.lr_schedule import build_scheduler


# ---------------------------------------------------------------------------
# ── LOSS (pre-shifted targets) ───────────────────────────────────────────────
# ---------------------------------------------------------------------------
# The PackedDataLoader already builds y as data[i+1 : i+1+seq_len] -- i.e.
# y[t] IS the correct next-token target for logits[t] (computed from context
# x[0..t]). model.py's forward(), however, expects `labels` to be UNSHIFTED
# (same alignment as input_ids) and does its own internal shift. Calling
# model(x, labels=y) with our already-shifted y therefore shifts TWICE, which
# produces an unlearnable target at every position.
#
# Fix: call the model WITHOUT labels and compute the loss here ourselves,
# directly against the already-shifted y, using pad_id as `ignore_index` so
# padding positions (where no valid token exists) do not contribute.

def pretrain_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """
    Compute pretrain cross-entropy loss with pre-shifted targets.

    Args:
        model: The transformer model (called without labels).
        x: Input token IDs, shape (B, T).
        y: Pre-shifted target token IDs, shape (B, T).
           y[t] is the next-token target for logits[t].
        pad_id: Token ID used for padding; masked out of the loss.

    Returns:
        Scalar loss tensor (cross-entropy, averaged over non-pad positions).
    """
    out = model(x)
    logits = out["logits"]  # (B, T, V)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        ignore_index=pad_id,
    )
    return loss


# ---------------------------------------------------------------------------
# GPU peak FLOP/s table  (bf16 Tensor Core, per card)
# ---------------------------------------------------------------------------
# Used for MFU estimation. Add your GPU here if missing.
GPU_PEAK_TFLOPS: Dict[str, float] = {
    # NVIDIA data-center
    "NVIDIA A100-SXM4-80GB":      312.0,
    "NVIDIA A100-SXM4-40GB":      312.0,
    "NVIDIA A100-PCIE-40GB":      312.0,
    "NVIDIA H100 SXM5":           989.0,
    "NVIDIA H100 PCIe":           756.0,
    "NVIDIA H200":                989.0,
    "NVIDIA L40S":                362.0,
    # NVIDIA consumer
    "NVIDIA GeForce RTX 4090":    165.0,
    "NVIDIA GeForce RTX 3090":    165.0,
    "NVIDIA GeForce RTX 5090":    260.0,
    "NVIDIA GeForce RTX 4080":    120.0,
    "NVIDIA GeForce RTX 4070 Ti": 82.0,
    "NVIDIA RTX A6000":           155.0,
    # AMD
    "AMD Instinct MI250X":        383.0,
    "AMD Instinct MI300X":        1307.0,
    # TPU reference values (no CUDA device-name match)
    "TPU-v4":                     275.0,
    "TPU-v5e":                    197.0,
}

_FALLBACK_TFLOPS: float = 120.0


def get_gpu_peak_tflops(device: torch.device) -> float:
    """Look up the GPU's theoretical bf16 peak TFLOPS from the table."""
    if device.type != "cuda":
        return _FALLBACK_TFLOPS
    name = torch.cuda.get_device_name(device)
    for key, val in GPU_PEAK_TFLOPS.items():
        if key.lower() in name.lower() or name.lower() in key.lower():
            return val
    print(f"[MFU] Unknown GPU '{name}', using {_FALLBACK_TFLOPS} TFLOPS fallback. "
          f"Add it to GPU_PEAK_TFLOPS for accurate MFU.")
    return _FALLBACK_TFLOPS


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_ddp(local_rank: int, world_size: int) -> Tuple[int, int, int, torch.device]:
    """
    Initialise the DDP process group.

    Works with both:
      - torchrun (RANK/LOCAL_RANK/WORLD_SIZE env vars are pre-set)
      - torch.multiprocessing.spawn (env vars may need to be set beforehand)

    Returns:
        (global_rank, local_rank, world_size, device)
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    effective_local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
    torch.cuda.set_device(effective_local_rank)
    device = torch.device(f"cuda:{effective_local_rank}")
    return rank, effective_local_rank, world_size, device


def is_master(rank: int) -> bool:
    """True only for the rank-0 process (sole writer of logs/checkpoints)."""
    return rank == 0


def cleanup_ddp() -> None:
    """Destroy the DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class PackedDataLoader:
    """
    Streams random fixed-length windows from packed memmap token file(s).

    Each rank gets a non-overlapping shard of the document list so no two
    GPUs see the same tokens in the same step.

    The loader supports three file layouts (checked in order):
      1. ``pretrain_tokens_{split}*.bin``  (split-specific glob)
      2. ``{split}.bin``                   (single-file per split)
      3. ``pretrain_tokens*.bin``           (generic glob — all tokens)

    Documents are delimited by EOS tokens (``eos_id``). Each sample starts
    at a document boundary (position 0 or immediately after an EOS) so that
    every training window fits inside a single document.

    Yields ``(x, y, num_valid)`` where:
      - ``x, y`` are ``(batch_size, seq_len)`` int64 tensors on GPU
      - ``y`` is pre-shifted (``y[t] = next token after x[t]``)
      - ``num_valid`` is the total number of non-pad tokens in the batch
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        batch_size: int,
        pad_id: int,
        rank: int = 0,
        world_size: int = 1,
        split: str = "train",
        eos_id: int = 1,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.eos_id = eos_id

        # ---- find data files ------------------------------------------------
        patterns: List[str] = sorted(
            glob.glob(os.path.join(data_dir, f"pretrain_tokens_{split}*.bin"))
        )
        if not patterns:
            fallback = os.path.join(data_dir, f"{split}.bin")
            if os.path.exists(fallback):
                patterns = [fallback]
            else:
                # Try generic glob as last resort
                patterns = sorted(
                    glob.glob(os.path.join(data_dir, "pretrain_tokens*.bin"))
                )
        if not patterns:
            raise FileNotFoundError(
                f"No data files matching pretrain_tokens_{split}*.bin, "
                f"{split}.bin, or pretrain_tokens*.bin in {data_dir}"
            )

        # ---- memmap all files, build cumulative offset table  ---------------
        self.file_paths = patterns
        self.memmaps: List[np.memmap] = []
        self.cumulative_offsets: List[int] = []
        total_tokens = 0
        for p in patterns:
            mm = np.memmap(p, dtype=np.uint16, mode="r")
            self.memmaps.append(mm)
            self.cumulative_offsets.append(total_tokens)
            total_tokens += len(mm)
        self.total_tokens = total_tokens
        self.cumulative_offsets.append(total_tokens)  # sentinel

        # ---- find document starts  ------------------------------------------
        # A document start is either position 0 or the position just after an
        # EOS token. We build the list globally, then rank-shard it.
        doc_starts: List[int] = []
        for file_idx, mm in enumerate(self.memmaps):
            base = self.cumulative_offsets[file_idx]
            eos_positions = np.where(mm == eos_id)[0]
            candidates = np.concatenate([[0], eos_positions + 1]).astype(np.int64)
            for s in candidates:
                global_pos = base + int(s)
                if global_pos + seq_len <= total_tokens:
                    doc_starts.append(global_pos)

        # ---- rank shard  ----------------------------------------------------
        n_docs = len(doc_starts)
        shard_size = n_docs // world_size
        start_idx = rank * shard_size
        end_idx = start_idx + shard_size if rank < world_size - 1 else n_docs
        self.doc_starts = doc_starts[start_idx:end_idx]

        # Per-loader RNG for reproducible random sampling
        self.gen = torch.Generator().manual_seed(seed * 1_000_003 + rank * 31)

        if is_master(rank if rank is not None else 0):
            print(f"[DataLoader] {split}: {n_docs:,} documents across "
                  f"{len(patterns)} file(s), {total_tokens:,} total tokens, "
                  f"shard [{start_idx}:{end_idx}) = {len(self.doc_starts):,} documents")

    # ------------------------------------------------------------------
    # Internal: read a window spanning possibly multiple memmap files
    # ------------------------------------------------------------------

    def _read_window(self, global_start: int, length: int) -> np.ndarray:
        """
        Read ``length`` tokens starting at ``global_start``, handling
        cross-file boundaries.  Short windows are padded with ``pad_id``.
        """
        result = np.empty(length, dtype=np.uint16)
        written = 0
        pos = global_start

        for file_idx in range(len(self.file_paths)):
            file_start = self.cumulative_offsets[file_idx]
            file_end = self.cumulative_offsets[file_idx + 1]

            if pos >= file_end:
                continue
            if pos < file_start:
                continue

            local_pos = pos - file_start
            available = file_end - pos
            take = min(length - written, available)
            result[written : written + take] = self.memmaps[file_idx][
                local_pos : local_pos + take
            ]
            written += take
            pos += take

            if written >= length:
                break

        if written < length:
            result[written:] = self.pad_id

        return result

    # ------------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------------

    def __iter__(self) -> "PackedDataLoader":
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return the next (x, y, num_valid) from randomly sampled documents."""
        if not self.doc_starts:
            raise StopIteration("No document starts available in this shard.")

        indices = torch.randint(
            0, len(self.doc_starts), (self.batch_size,), generator=self.gen
        )

        x_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        num_valid = 0

        for idx in indices:
            start = self.doc_starts[idx.item()]
            tokens = self._read_window(start, self.seq_len + 1)  # +1 for shift
            x = tokens[: self.seq_len]
            y = tokens[1 : self.seq_len + 1]  # pre-shifted
            x_list.append(x)
            y_list.append(y)
            num_valid += int((y != self.pad_id).sum())

        x_np = np.stack(x_list).astype(np.int64)
        y_np = np.stack(y_list).astype(np.int64)

        device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', '0')}")
        x_gpu = torch.from_numpy(x_np).to(device, non_blocking=True)
        y_gpu = torch.from_numpy(y_np).to(device, non_blocking=True)

        return x_gpu, y_gpu, num_valid

    next_batch = __next__  # convenience alias used by validate()

    # ------------------------------------------------------------------
    # Sequential iteration (for validation — no overlap, no replacement)
    # ------------------------------------------------------------------

    def iter_sequential(
        self, max_batches: int
    ) -> "PackedDataLoaderSequential":
        """
        Yield ``(x, y)`` by walking the data in document order with no
        overlap and no replacement.  Used by ``validate()`` to produce a
        deterministic held-out loss.
        """
        return PackedDataLoaderSequential(self, max_batches)


class PackedDataLoaderSequential:
    """
    Sequential (non-random) window iterator.  Walks document starts in order,
    yielding one window per document.  Stops after ``max_batches`` or when
    data is exhausted.
    """

    def __init__(self, loader: PackedDataLoader, max_batches: int):
        self.loader = loader
        self.max_batches = max_batches
        self._cursor = 0
        self._device = None

    def __iter__(self) -> "PackedDataLoaderSequential":
        self._cursor = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cursor >= len(self.loader.doc_starts) or self.max_batches <= 0:
            raise StopIteration

        # Build one batch from consecutive document starts
        offsets: List[int] = []
        for _ in range(self.loader.batch_size):
            if self._cursor >= len(self.loader.doc_starts) or self.max_batches <= 0:
                break
            offsets.append(self.loader.doc_starts[self._cursor])
            self._cursor += 1
            self.max_batches -= 1

        if not offsets:
            raise StopIteration

        x_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        for start in offsets:
            tokens = self.loader._read_window(start, self.loader.seq_len + 1)
            x_list.append(tokens[: self.loader.seq_len])
            y_list.append(tokens[1 : self.loader.seq_len + 1])

        x_np = np.stack(x_list).astype(np.int64)
        y_np = np.stack(y_list).astype(np.int64)

        if self._device is None:
            self._device = torch.device(
                f"cuda:{os.environ.get('LOCAL_RANK', '0')}"
            )
        x_gpu = torch.from_numpy(x_np).to(self._device, non_blocking=True)
        y_gpu = torch.from_numpy(y_np).to(self._device, non_blocking=True)
        return x_gpu, y_gpu


# ---------------------------------------------------------------------------
# LR schedule (delegates to optim/lr_schedule.py)
# ---------------------------------------------------------------------------

def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
    schedule: str = "cosine",
    stable_ratio: float = 0.8,
) -> float:
    """
    Compute the learning rate at ``step`` using ``build_scheduler()``.

    Args:
        step: Current training step.
        warmup_steps: Linear warmup length.
        max_steps: Total training steps.
        max_lr: Peak learning rate after warmup.
        min_lr: Floor learning rate (cosine decay minimum).
        schedule: ``"cosine"`` or ``"wsd"`` (warmup-stable-decay).
        stable_ratio: (WSD only) fraction of remaining steps in stable phase.

    Returns:
        The learning rate at ``step``.
    """
    scheduler = build_scheduler(
        schedule=schedule,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        peak_lr=max_lr,
        min_lr=min_lr,
        stable_ratio=stable_ratio,
    )
    return scheduler(step)


# ---------------------------------------------------------------------------
# Optimizer (delegates to optim/build_optimizer.py)
# ---------------------------------------------------------------------------

def build_optimizer_groups(
    model: nn.Module,
    lr: float = 5e-4,
    weight_decay: float = 0.1,
    optimizer_type: str = "adamw",
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    **kwargs,
) -> List[torch.optim.Optimizer]:
    """
    Build the optimizer(s) for training, delegating to
    ``optim.build_optimizer.build_optimizer()``.

    Returns a list of optimizers so the training loop can iterate uniformly
    over single (AdamW) and dual (Muon + AdamW) cases.

    Args:
        model: The transformer model.
        lr: Peak learning rate.
        weight_decay: Weight decay coefficient.
        optimizer_type: ``"adamw"`` (default) or ``"muon"``.
        betas: Adam/W beta coefficients.
        eps: Adam epsilon.

    Returns:
        List of one or two ``torch.optim.Optimizer`` instances.
    """
    result = build_optimizer(
        model,
        optimizer_type=optimizer_type,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        **kwargs,
    )
    if isinstance(result, tuple):
        # Muon returns (muon_optimizer, adamw_optimizer)
        print(f"[Optimizer] Muon + AdamW ({len(result)} optimizers)")
        return list(result)
    else:
        print(f"[Optimizer] AdamW")
        return [result]


# ---------------------------------------------------------------------------
# MFU (model FLOPs utilization) — calibrated per GPU
# ---------------------------------------------------------------------------

def estimate_mfu(
    model: nn.Module,
    tokens_per_sec: float,
    gpu_peak_tflops: float,
) -> float:
    """
    Estimate what fraction of the GPU's theoretical bf16 peak we are using.

    Formula: each token costs 6 FLOPs per non-embedding parameter for forward
    + backward (Chinchilla / PaLM approximation).

    Architecture-aware ops count:
      - GQA (num_kv_heads < num_attention_heads): 30 ops/element → 6 FLOPs/param/token
      - MHA (num_kv_heads == num_attention_heads): 24 ops/element → 4.8 FLOPs/param/token

    Attention FLOPs (quadratic in seq_len) are omitted here because they are a
    small fraction for typical seq_len << hidden_size * n_layers.
    """
    raw = model.module if isinstance(model, DDP) else model
    # Unwrap torch.compile wrapper if present
    raw = getattr(raw, "_orig_mod", raw)

    # Count non-embedding parameters
    n_params = sum(
        p.numel()
        for name, p in raw.named_parameters()
        if "embed_tokens" not in name
    )

    # Architecture-aware FLOPs per parameter per token
    if hasattr(raw, "config") and hasattr(raw.config, "num_key_value_heads"):
        if raw.config.num_key_value_heads < raw.config.num_attention_heads:
            flops_per_param = 6.0   # GQA  (30 ops/element)
        else:
            flops_per_param = 4.8   # MHA  (24 ops/element)
    else:
        flops_per_param = 6.0

    flops_per_sec = flops_per_param * n_params * tokens_per_sec
    return flops_per_sec / (gpu_peak_tflops * 1e12)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    model: nn.Module,
    optimizers: List[torch.optim.Optimizer],
    config: ModelConfig,
    total_tokens: int = 0,
    best_val_loss: float = float("inf"),
    recipe: Optional[Dict[str, Any]] = None,
    train_args: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a training checkpoint.

    Saves:
      - ``step_XXXXX.pt`` — model state, optimizer state, step, total_tokens
      - ``config.json``   — serialized ModelConfig
      - ``meta.json``     — step and total_tokens metadata
      - ``recipe.json``   — training recipe (if provided)
      - ``latest_checkpoint`` symlink pointing to the newest ``step_XXXXX.pt``

    Args:
        checkpoint_dir: Output directory.
        step: Current optimizer step.
        model: The model (may be wrapped in DDP / torch.compile).
        optimizers: List of optimizers (single AdamW or Muon+AdamW).
        config: ModelConfig used for this run.
        total_tokens: Cumulative tokens processed so far.
        best_val_loss: Best validation loss seen so far.
        recipe: Optional dict of recipe/training configuration.
        train_args: Optional dict of CLI args for reproducibility.

    Returns:
        Path to the saved checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    raw_model = model.module if isinstance(model, DDP) else model
    state_model = getattr(raw_model, "_orig_mod", raw_model)

    # ---- model + optimizer state  ------------------------------------------
    ckpt: Dict[str, Any] = {
        "step": step,
        "model_state": state_model.state_dict(),
        "total_tokens": total_tokens,
        "best_val_loss": best_val_loss,
    }
    if len(optimizers) == 1:
        ckpt["optimizer_state"] = optimizers[0].state_dict()
    else:
        ckpt["optimizer_state"] = [opt.state_dict() for opt in optimizers]

    step_path = os.path.join(checkpoint_dir, f"step_{step:05d}.pt")
    torch.save(ckpt, step_path)

    # ---- config.json  -------------------------------------------------------
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2, default=str)

    # ---- meta.json  --------------------------------------------------------
    meta_path = os.path.join(checkpoint_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "step": step,
                "total_tokens": total_tokens,
                "best_val_loss": best_val_loss,
            },
            f,
            indent=2,
        )

    # ---- recipe.json  ------------------------------------------------------
    if recipe is not None:
        recipe_path = os.path.join(checkpoint_dir, "recipe.json")
        with open(recipe_path, "w") as f:
            json.dump(recipe, f, indent=2, default=str)

    # ---- latest_checkpoint symlink  -----------------------------------------
    latest = os.path.join(checkpoint_dir, "latest_checkpoint")
    if os.path.islink(latest) or os.path.exists(latest):
        if os.path.islink(latest):
            os.remove(latest)
        else:
            # Regular file or directory — remove it so the symlink can be placed
            os.remove(latest)
    os.symlink(os.path.abspath(step_path), latest)

    print(f"[Checkpoint] saved {step_path}")
    return step_path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizers: Optional[List[torch.optim.Optimizer]],
    device: torch.device,
) -> Tuple[int, float, int]:
    """
    Load a training checkpoint, handling both old-style (single .pt with
    embedded config) and new recipe-based formats.

    Args:
        path: Path to a ``step_XXXXX.pt`` file, a ``latest_checkpoint``
              symlink, or a directory containing a ``step_*.pt`` file.
        model: The model (may be wrapped in DDP / torch.compile).
        optimizers: List of optimizers (or None to skip optimizer restore).
        device: Device to load the state dict onto.

    Returns:
        (step, best_val_loss, total_tokens) restored from the checkpoint.
    """
    # Resolve path
    if os.path.isdir(path):
        step_files = sorted(glob.glob(os.path.join(path, "step_*.pt")))
        if not step_files:
            raise FileNotFoundError(f"No step_*.pt files in directory {path}")
        ckpt_path = step_files[-1]
    elif os.path.islink(path) or os.path.isfile(path):
        ckpt_path = path
    else:
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    raw_model = model.module if isinstance(model, DDP) else model
    state_model = getattr(raw_model, "_orig_mod", raw_model)

    # Load model state
    state_model.load_state_dict(ckpt["model_state"])

    # Re-tie weights after state_dict restore (loading replaces the weight
    # tensor object, breaking the tie).
    if hasattr(state_model, "tie_weights"):
        state_model.tie_weights()

    # Load optimizer state
    if optimizers is not None and "optimizer_state" in ckpt:
        opt_state = ckpt["optimizer_state"]
        if isinstance(opt_state, list):
            # Multi-optimizer (Muon + AdamW)
            for opt, state in zip(optimizers, opt_state):
                opt.load_state_dict(state)
        else:
            # Single optimizer (AdamW)
            optimizers[0].load_state_dict(opt_state)

    step = ckpt.get("step", 0)
    total_tokens = ckpt.get("total_tokens", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))

    print(f"[Checkpoint] resumed from {ckpt_path} at step {step} "
          f"(total_tokens={total_tokens:,})")
    return step, best_val_loss, total_tokens


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: PackedDataLoader,
    pad_id: int,
    max_batches: int = 50,
    ctx: Optional[Any] = nullcontext(),
    use_cudagraphs: bool = False,
) -> Tuple[float, float]:
    """
    Evaluate the model on validation data.

    Returns ``(loss_random, loss_sequential)``:
      - ``loss_random``:    average over random windows (comparable to train loss)
      - ``loss_sequential``: walk the data in document order (simulates a true
                             held-out perplexity on coherent text)

    Args:
        model: The transformer model.
        val_loader: Data loader for the validation set.
        pad_id: Token ID used for padding (masked in loss).
        max_batches: Number of batches to evaluate.
        ctx: Autocast context manager (or ``nullcontext()``).
        use_cudagraphs: Whether to call ``cudagraph_mark_step_begin()``.

    Returns:
        (loss_random, loss_sequential)
    """
    model.eval()

    # ---- random-window loss  ------------------------------------------------
    losses_random: List[float] = []
    for _ in range(max_batches):
        x, y, _ = val_loader.next_batch() if hasattr(val_loader, 'next_batch') else next(val_loader.__iter__())
        with ctx:  # type: ignore[union-attr]
            if use_cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()
            loss = pretrain_loss(model, x, y, pad_id)
        losses_random.append(loss.item())
    loss_random = float(np.mean(losses_random)) if losses_random else 0.0

    # ---- sequential loss (document-order)  ----------------------------------
    losses_seq: List[float] = []
    for x, y in val_loader.iter_sequential(max_batches=max_batches):
        with ctx:  # type: ignore[union-attr]
            if use_cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()
            loss = pretrain_loss(model, x, y, pad_id)
        losses_seq.append(loss.item())
    loss_seq = float(np.mean(losses_seq)) if losses_seq else loss_random

    model.train()
    return loss_random, loss_seq


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test() -> None:
    """
    Quick end-to-end sanity check that runs a few steps on synthetic data,
    saves and reloads a checkpoint, and verifies the training loop works.
    """
    print("\n=== smoke test (no real data found) ===")
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp()
    data_dir = os.path.join(tmp, "packed")
    os.makedirs(data_dir)
    ckpt_dir = os.path.join(tmp, "ckpts")

    vocab_size = 1024
    n_tokens = 50_000
    arr = np.random.randint(0, vocab_size, n_tokens, dtype=np.uint16)
    arr.tofile(os.path.join(data_dir, "pretrain_tokens_train.bin"))
    arr[:5000].tofile(os.path.join(data_dir, "pretrain_tokens_val.bin"))
    with open(os.path.join(data_dir, "meta.json"), "w") as f:
        json.dump(
            {
                "vocab_size": vocab_size,
                "dtype": "uint16",
                "train_tokens": n_tokens,
                "val_tokens": 5000,
                "total_tokens": n_tokens,
                "category_token_counts": {},
            },
            f,
        )

    config = ModelConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=256,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerForCausalLM(config).to(device)
    print(f"Smoke-test model: {count_parameters(model):,} params on {device}")

    optimizers = build_optimizer_groups(model, lr=3e-4, weight_decay=0.1)
    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    # Data loaders — use pad_id=0 matching nn.Embedding(padding_idx=0)
    train_loader = PackedDataLoader(
        data_dir, seq_len=64, batch_size=2, pad_id=0,
        rank=0, world_size=1, split="train", seed=42,
    )
    val_loader = PackedDataLoader(
        data_dir, seq_len=64, batch_size=2, pad_id=0,
        rank=0, world_size=1, split="val", seed=42,
    )
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    gpu_peak = get_gpu_peak_tflops(device)
    model.train()
    os.makedirs(ckpt_dir, exist_ok=True)
    t0 = time.perf_counter()

    for step in range(5):
        lr = get_lr(step, warmup_steps=2, max_steps=5, max_lr=3e-4, min_lr=3e-5)
        for opt in optimizers:
            for pg in opt.param_groups:
                pg["lr"] = lr

        x, y, num_valid = next(train_iter)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        with ctx:
            loss = pretrain_loss(model, x, y, pad_id=0)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()

        mfu = estimate_mfu(
            model, 2 * 64 / max(time.perf_counter() - t0, 1e-6), gpu_peak,
        )
        print(f"  step {step} | loss {loss.item():.4f} | "
              f"lr {lr:.2e} | mfu {mfu * 100:.2f}% | valid {num_valid}")

    # Validate
    val_random, val_seq = validate(
        model, val_loader, pad_id=0,
        max_batches=3, ctx=ctx,
    )
    print(f"  val_random: {val_random:.4f} | val_seq: {val_seq:.4f}")

    # Save checkpoint
    save_checkpoint(
        ckpt_dir, step=5, model=model, optimizers=optimizers,
        config=config, total_tokens=5 * 2 * 64, best_val_loss=val_seq,
        train_args={"model_size": "smoke_test"},
    )

    # Load checkpoint
    model2 = TransformerForCausalLM(config).to(device)
    optimizers2 = build_optimizer_groups(model2, lr=3e-4, weight_decay=0.1)
    step_r, _, _ = load_checkpoint(
        ckpt_dir, model2, optimizers2, device,
    )
    assert step_r == 5, f"Expected step 5, got {step_r}"

    shutil.rmtree(tmp)
    print(f"\n=== smoke test passed in {time.perf_counter() - t0:.2f}s ===\n")


# ---------------------------------------------------------------------------
# Optional W&B
# ---------------------------------------------------------------------------

def try_init_wandb(
    args: argparse.Namespace,
    config: ModelConfig,
    n_params: int,
) -> bool:
    """Initialise Weights & Biases logging (no-op on import error)."""
    try:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"dense-{args.model_size}",
            config={
                **vars(config),
                "n_params": n_params,
                **{k: v for k, v in vars(args).items() if k != "wandb_project" and k != "wandb_run_name"},
            },
        )
        return True
    except Exception as e:
        print(f"[W&B] disabled: {e}")
        return False


def log_wandb(metrics: Dict[str, float], step: int) -> None:
    """Log metrics to W&B (no-op on import error)."""
    try:
        import wandb
        wandb.log(metrics, step=step)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Checkpoint pruning
# ---------------------------------------------------------------------------

def _prune_checkpoints(checkpoint_dir: str, keep: int = 3) -> None:
    """Remove oldest step_*.pt files, keeping only the ``keep`` most recent."""
    ckpts = sorted(
        Path(checkpoint_dir).glob("step_*.pt"),
        key=lambda p: int(p.stem.replace("step_", "")),
    )
    for old in ckpts[:-keep]:
        old.unlink()
        print(f"[Checkpoint] pruned {old}")


# ---------------------------------------------------------------------------
# Data availability check
# ---------------------------------------------------------------------------

def _has_data(data_dir: str) -> bool:
    """Return True if any training data files exist under ``data_dir``."""
    if not os.path.isdir(data_dir):
        return False
    if glob.glob(os.path.join(data_dir, "pretrain_tokens*.bin")):
        return True
    if os.path.isfile(os.path.join(data_dir, "train.bin")):
        return True
    return False


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def _train(
    rank: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Inner training routine (called by ``main_ddp`` per process)."""
    master = is_master(rank)

    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------ meta
    meta_path = os.path.join(args.data_dir, "meta.json")
    meta: Dict[str, Any] = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    vocab_size = meta.get("vocab_size", args.vocab_size or 65536)

    # ------------------------------------------------------------------ model
    if args.resume:
        # Attempt to load config from checkpoint
        if os.path.isdir(args.resume):
            config_path = os.path.join(args.resume, "config.json")
            if os.path.isfile(config_path):
                with open(config_path) as f:
                    ckpt_config = json.load(f)
                config = ModelConfig(**ckpt_config)
                if master:
                    print(f"[Resume] loaded config from {config_path}")
            else:
                # Fallback: build from model-size or CLI args
                config = _build_config(args, vocab_size)
        elif os.path.isfile(args.resume) or os.path.islink(args.resume):
            # Try loading directly from checkpoint file
            ckpt_raw = torch.load(
                args.resume, map_location="cpu", weights_only=False,
            )
            if "config" in ckpt_raw:
                config = ModelConfig(**ckpt_raw["config"])
                if master:
                    print(f"[Resume] loaded config from checkpoint file")
            else:
                config = _build_config(args, vocab_size)
        else:
            config = _build_config(args, vocab_size)
    else:
        config = _build_config(args, vocab_size)

    model = TransformerForCausalLM(config).to(device)
    n_params = count_parameters(model)

    if master:
        print(f"Model params : {n_params:,}  ({n_params / 1e9:.3f}B)")
        if device.type == "cuda":
            _print_vram_estimate(args, config, n_params, device)

    # ---- gradient checkpointing  -------------------------------------------
    if args.gradient_checkpointing:
        model.model.enable_gradient_checkpointing()

    # ---- torch.compile / JIT  ----------------------------------------------
    if args.jit:
        if master:
            print(f"[compile] torch.compile(mode='{args.compile_mode}')…")
            if args.compile_mode == "reduce-overhead":
                print("          Using CUDAGraphs (reduce-overhead). If you see tensor")
                print("          overwrite errors, switch to --compile-mode default.")
            print("          First step will be slow (~60-120s). Subsequent steps are fast.")
        model = torch.compile(model, mode=args.compile_mode)

    _use_cudagraphs = args.jit and args.compile_mode == "reduce-overhead"

    # ------------------------------------------------------------------ DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # ------------------------------------------------------------------ data
    # pad_id=0 matches nn.Embedding(padding_idx=0) in model.py
    train_loader = PackedDataLoader(
        args.data_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        pad_id=0,
        rank=rank,
        world_size=world_size,
        split="train",
        seed=args.seed,
    )
    val_loader = PackedDataLoader(
        args.data_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        pad_id=0,
        rank=rank,
        world_size=world_size,
        split="val",
        seed=args.seed,
    )
    train_iter = iter(train_loader)

    # ------------------------------------------------------------------ optim
    optimizers = build_optimizer_groups(
        model,
        optimizer_type=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ------------------------------------------------------------------ LR schedule
    scheduler = build_scheduler(
        schedule=args.schedule,
        warmup_steps=args.warmup_steps,
        max_steps=args.num_steps,
        peak_lr=args.lr,
        min_lr=args.min_lr,
        stable_ratio=args.stable_ratio,
    )

    # ------------------------------------------------------------------ amp
    use_amp = device.type == "cuda" and args.dtype == "bf16"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp
        else nullcontext()
    )

    # ------------------------------------------------------------------ resume
    start_step = 0
    total_tokens = 0
    best_val_loss = float("inf")
    if args.resume:
        start_step, best_val_loss, total_tokens = load_checkpoint(
            args.resume, model, optimizers, device,
        )
        # Re-tie weights after loading state_dict
        raw = model.module if isinstance(model, DDP) else model
        raw_model = getattr(raw, "_orig_mod", raw)
        if hasattr(raw_model, "tie_weights"):
            raw_model.tie_weights()

    # ------------------------------------------------------------------ MFU
    gpu_peak_tflops = get_gpu_peak_tflops(device)
    if master:
        print(f"GPU peak bf16: {gpu_peak_tflops:.1f} TFLOPS")

    # ------------------------------------------------------------------ W&B
    if master:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    use_wandb = (
        master
        and args.wandb_project
        and try_init_wandb(args, config, n_params)
    )

    # ------------------------------------------------------------------ tokens accounting
    tokens_per_step = (
        args.batch_size * args.seq_len * args.grad_accum * world_size
    )
    if master:
        print(f"\nTokens / optimizer step : {tokens_per_step:,}")
        print(f"Effective batch size    : {args.batch_size * args.grad_accum * world_size}")
        print(f"Total steps             : {args.num_steps:,}")
        print(f"Checkpoint every        : {args.save_every:,} steps")
        print(f"Validate every          : {args.val_every:,} steps")
        print(f"Total tokens (planned)  : {args.num_steps * tokens_per_step:,}\n")

    # ================================================================== LOOP
    model.train()
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)

    t0 = time.perf_counter()
    # GPU-side accumulators (avoids GPU->CPU sync inside the micro-step loop)
    loss_accum = torch.zeros((), device=device)
    grad_norm_accum = torch.zeros((), device=device)
    tokens_accum = 0

    for step in range(start_step, args.num_steps):

        # ---- LR
        lr = scheduler(step)
        for opt in optimizers:
            for pg in opt.param_groups:
                pg["lr"] = lr

        # ---- gradient accumulation micro-steps
        for micro_step in range(args.grad_accum):
            x, y, num_valid = next(train_iter)

            sync = micro_step == args.grad_accum - 1
            ctx_ddp = (
                nullcontext()
                if (world_size == 1 or sync)
                else model.no_sync()
            )

            with ctx_ddp:
                with amp_ctx:
                    if _use_cudagraphs:
                        torch.compiler.cudagraph_mark_step_begin()
                    loss = pretrain_loss(model, x, y, pad_id=0) / args.grad_accum
                loss.backward()

            loss_accum = loss_accum + loss.detach()
            tokens_accum += num_valid

        # ---- gradient clipping  (accumulate on GPU, sync at log time)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        grad_norm_accum = grad_norm_accum + grad_norm.detach()

        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)

        total_tokens += tokens_per_step * world_size

        # ---- CUDA sync for accurate timing
        if device.type == "cuda":
            torch.cuda.synchronize()

        # ---- logging
        if master and step % args.log_interval == 0:
            t1 = time.perf_counter()
            dt = max(t1 - t0, 1e-9)
            tok_per_sec = tokens_per_step * args.log_interval / dt
            mfu = estimate_mfu(model, tok_per_sec, gpu_peak_tflops)

            loss_display = (loss_accum / args.log_interval).item()
            grad_norm_display = (grad_norm_accum / args.log_interval).item()
            loss_accum = torch.zeros((), device=device)
            grad_norm_accum = torch.zeros((), device=device)

            print(
                f"step {step:7d} | loss {loss_display:.4f} | lr {lr:.2e} | "
                f"grad {grad_norm_display:.3f} | "
                f"{tok_per_sec / 1e3:.1f}k tok/s | "
                f"mfu {mfu * 100:.2f}%"
            )

            if use_wandb:
                log_wandb(
                    {
                        "train/loss": loss_display,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm_display,
                        "perf/tokens_per_sec": tok_per_sec,
                        "perf/mfu_pct": mfu * 100,
                    },
                    step=step,
                )

            t0 = t1

        # ---- validation
        if master and step % args.val_every == 0 and step > start_step + 1:
            val_random, val_seq = validate(
                model, val_loader, pad_id=0,
                max_batches=args.eval_steps, ctx=amp_ctx,
                use_cudagraphs=_use_cudagraphs,
            )
            # Use sequential loss for best-checkpoint tracking (it is not
            # biased by the random-window artifact).
            val_loss = val_seq

            if world_size > 1:
                vr = torch.tensor(val_random, device=device)
                vs = torch.tensor(val_seq, device=device)
                dist.all_reduce(vr, op=dist.ReduceOp.AVG)
                dist.all_reduce(vs, op=dist.ReduceOp.AVG)
                val_random = vr.item()
                val_seq = vs.item()
                val_loss = val_seq

            print(
                f"  [eval] step {step:7d} | val_random {val_random:.4f} | "
                f"val_seq {val_seq:.4f}"
            )
            if use_wandb:
                log_wandb(
                    {
                        "val/loss": val_loss,
                        "val/loss_random": val_random,
                        "val/loss_seq": val_seq,
                    },
                    step=step,
                )
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        # ---- checkpoint
        if master and step % args.save_every == 0 and step > start_step + 1:
            recipe = _build_recipe(args, config)
            save_checkpoint(
                checkpoint_dir=args.checkpoint_dir,
                step=step,
                model=model,
                optimizers=optimizers,
                config=config,
                total_tokens=total_tokens,
                best_val_loss=best_val_loss,
                recipe=recipe,
                train_args=vars(args),
            )
            _prune_checkpoints(args.checkpoint_dir, keep=args.keep_ckpts)

    # ---- final checkpoint
    if master:
        recipe = _build_recipe(args, config)
        save_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            step=args.num_steps,
            model=model,
            optimizers=optimizers,
            config=config,
            total_tokens=total_tokens,
            best_val_loss=best_val_loss,
            recipe=recipe,
            train_args=vars(args),
        )
        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


def _build_config(args: argparse.Namespace, vocab_size: int) -> ModelConfig:
    """Build a ModelConfig from CLI args."""
    if args.model_size:
        target_params = ModelConfig.parse_param_count(args.model_size)
        config = ModelConfig.from_target_size(
            target_params,
            vocab_size=vocab_size,
            max_position_embeddings=args.seq_len,
        )
    else:
        hidden_size = args.hidden_size or 2048
        num_heads = args.num_heads or max(4, hidden_size // 128)
        num_kv_heads = args.num_kv_heads or num_heads // 4
        num_layers = args.num_layers or 28
        intermediate_size = args.intermediate_size or hidden_size * 3
        config = ModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=args.head_dim or 128,
            max_position_embeddings=args.seq_len,
        )
    return config


def _build_recipe(
    args: argparse.Namespace,
    config: ModelConfig,
) -> Dict[str, Any]:
    """Build a recipe dict for checkpoint reproducibility."""
    return {
        "model": vars(config),
        "training": {
            "lr": args.lr,
            "min_lr": args.min_lr,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "num_steps": args.num_steps,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "seq_len": args.seq_len,
            "seed": args.seed,
            "schedule": args.schedule,
            "optimizer": args.optimizer,
            "dtype": args.dtype,
        },
        "data": {
            "data_dir": args.data_dir,
            "val_fraction": args.val_fraction,
        },
    }


def _print_vram_estimate(
    args: argparse.Namespace,
    config: ModelConfig,
    n_params: int,
    device: torch.device,
) -> None:
    """Print VRAM usage estimates (master process only)."""
    vram_total = torch.cuda.get_device_properties(device).total_memory
    vram_gb = vram_total / 1024**3
    static_gb = n_params * (2 + 2 + 8) / 1024**3  # bf16 w + bf16 g + fp32 m+v
    cfg = config
    bytes_per_token_per_layer = (
        cfg.hidden_size * 2  # hidden states (bf16)
        + cfg.intermediate_size * 2 * 2  # gate + up proj (bf16)
        + cfg.num_attention_heads * cfg.head_dim * 2  # attn output (bf16)
    )
    act_gb_per_step = (
        args.batch_size
        * args.seq_len
        * cfg.num_hidden_layers
        * bytes_per_token_per_layer
        / 1024**3
    )
    total_est = static_gb + act_gb_per_step
    headroom_gb = vram_gb - total_est

    print(f"GPU          : {torch.cuda.get_device_name(device)}")
    print(f"VRAM         : {vram_gb:.1f} GB total")
    print(f"  static     : ~{static_gb:.1f} GB  (weights + grads + Adam)")
    print(f"  activations: ~{act_gb_per_step:.1f} GB  "
          f"(batch={args.batch_size}, seq={args.seq_len})")
    print(f"  headroom   : ~{headroom_gb:.1f} GB")

    if headroom_gb < 1.5:
        safe_batch = max(
            1,
            int(
                (vram_gb - static_gb - 2.0) * 1024**3
                / (
                    args.seq_len
                    * cfg.num_hidden_layers
                    * bytes_per_token_per_layer
                )
            ),
        )
        print(f"\n  WARNING: estimated VRAM is tight (<1.5 GB headroom).")
        print(f"  Likely OOM at batch={args.batch_size}, seq={args.seq_len}.")
        print(f"  Suggestions:")
        print(f"    --batch-size {safe_batch}  (estimated safe)")
        print(f"    --gradient-checkpointing   (cuts activation VRAM ~35%)")
        print(f"    --seq-len {args.seq_len // 2}  (halves activation VRAM)\n")


# ---------------------------------------------------------------------------
# DDP entry point
# ---------------------------------------------------------------------------

def main_ddp(local_rank: int, world_size: int, args: argparse.Namespace) -> None:
    """
    Entry point for each DDP process (called by either ``torchrun`` or
    ``torch.multiprocessing.spawn``).
    """
    # Set environment variables for the DDP process group.
    os.environ.setdefault("RANK", str(local_rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")

    rank, effective_local_rank, effective_world_size, device = setup_ddp(
        local_rank, world_size,
    )
    try:
        _train(rank, effective_local_rank, effective_world_size, device, args)
    finally:
        cleanup_ddp()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Pretrain a dense transformer LLM.",
    )

    # ---- model
    p.add_argument("--model-size", default=None,
                   help="Target model size (e.g. '0.6B', '1.7B', '600M'). "
                        "When set, model architecture is auto-derived.")
    p.add_argument("--vocab-size", type=int, default=None,
                   help="Override vocabulary size (read from meta.json if not set).")
    p.add_argument("--hidden-size", type=int, default=None,
                   help="Hidden dimension (used when --model-size is not set).")
    p.add_argument("--num-layers", type=int, default=None,
                   help="Number of decoder layers.")
    p.add_argument("--num-heads", type=int, default=None,
                   help="Number of attention heads.")
    p.add_argument("--num-kv-heads", type=int, default=None,
                   help="Number of key-value heads (GQA).")
    p.add_argument("--intermediate-size", type=int, default=None,
                   help="MLP intermediate size.")
    p.add_argument("--head-dim", type=int, default=128,
                   help="Head dimension (default 128).")

    # ---- data
    p.add_argument("--data-dir", default="./packed",
                   help="Directory containing pretrain_tokens*.bin files.")
    p.add_argument("--seq-len", type=int, default=2048,
                   help="Training sequence length.")
    p.add_argument("--val-fraction", type=float, default=0.01,
                   help="Fraction of data reserved for validation (informational).")

    # ---- training
    p.add_argument("--batch-size", type=int, default=8,
                   help="Per-GPU batch size.")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation steps.")
    p.add_argument("--num-steps", type=int, default=100_000,
                   help="Total training steps.")
    p.add_argument("--warmup-steps", type=int, default=2_000,
                   help="LR warmup steps.")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Peak learning rate.")
    p.add_argument("--min-lr", type=float, default=3e-5,
                   help="Minimum LR (cosine decay floor).")
    p.add_argument("--weight-decay", type=float, default=0.1,
                   help="AdamW weight decay.")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Gradient clipping max norm.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                   help="Training precision.")

    # ---- schedule
    p.add_argument("--schedule", default="cosine", choices=["cosine", "wsd"],
                   help="LR schedule type: cosine (default) or WSD.")
    p.add_argument("--stable-ratio", type=float, default=0.8,
                   help="WSD only: fraction of non-warmup steps in stable phase.")

    # ---- optimizer
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"],
                   help="Optimizer type: adamw (default) or muon.")

    # ---- compilation
    p.add_argument("--jit", action="store_true",
                   help="Enable torch.compile for kernel fusion (+25-40% throughput).")
    p.add_argument("--compile-mode", default="default",
                   choices=["default", "reduce-overhead", "max-autotune"],
                   help=("torch.compile mode. "
                         "'default' — safe, good speedup. "
                         "'reduce-overhead' — uses CUDAGraphs. "
                         "'max-autotune' — exhaustive kernel search."))

    # ---- data loading
    p.add_argument("--num-workers", type=int, default=2,
                   help="Not used directly (PackedDataLoader uses in-process loading).")

    # ---- memory
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Enable gradient checkpointing (saves ~35% VRAM, costs ~30% compute).")

    # ---- checkpointing
    p.add_argument("--checkpoint-dir", default="./checkpoints",
                   help="Directory for saving checkpoints.")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint (file or dir) to resume from.")
    p.add_argument("--save-every", type=int, default=5_000,
                   help="Save checkpoint every N steps.")
    p.add_argument("--keep-ckpts", type=int, default=3,
                   help="Number of recent checkpoints to keep.")

    # ---- logging / eval
    p.add_argument("--log-interval", type=int, default=10,
                   help="Log metrics every N steps.")
    p.add_argument("--val-every", "--eval-every", type=int, default=500,
                   help="Run validation every N steps.")
    p.add_argument("--eval-steps", type=int, default=50,
                   help="Number of batches per validation.")
    p.add_argument("--wandb-project", default=None,
                   help="W&B project name (omit to disable W&B).")
    p.add_argument("--wandb-run-name", default=None,
                   help="W&B run name.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _args = parse_args()

    if not _has_data(_args.data_dir):
        smoke_test()
    else:
        _world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

        if _world_size > 1 and "RANK" not in os.environ:
            # Launch via torch.multiprocessing.spawn
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "12355")
            torch.multiprocessing.spawn(
                main_ddp,
                nprocs=_world_size,
                args=(_world_size, _args),
                join=True,
            )
        else:
            # Single GPU or already launched via torchrun
            _local_rank = int(os.environ.get("LOCAL_RANK", 0))
            _world_size_from_env = int(os.environ.get("WORLD_SIZE", 1))
            main_ddp(_local_rank, _world_size_from_env, _args)
