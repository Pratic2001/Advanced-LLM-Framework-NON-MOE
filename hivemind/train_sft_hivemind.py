#!/usr/bin/env python3
"""
train_sft_hivemind.py

Supervised fine-tuning with Hivemind — distributed across heterogeneous
nodes (laptops, gaming PCs, cloud instances) with async averaging.

Usage
-----
  # Bootstrap (first node):
      python hivemind/train_sft_hivemind.py \\
          --hivemind --initial-peers "" --port 5678 \\
          --model-size 300M --data-dir ./sft_packed \\
          --checkpoint-dir ./sft_ckpts --lora-rank 64

  # Worker nodes:
      python hivemind/train_sft_hivemind.py \\
          --hivemind --initial-peers "192.168.1.100:5678" \\
          --model-size 300M --data-dir ./sft_packed \\
          --checkpoint-dir ./sft_ckpts_worker1 --lora-rank 64

  # Resume:
      python hivemind/train_sft_hivemind.py --hivemind ... \\
          --resume ./sft_ckpts/step_00050.pt

Supports full fine-tune, LoRA, DoRA, and rsLoRA — same as ``train_sft.py``
but with Hivemind's DecentralizedOptimizer replacing DDP for parameter
averaging across heterogeneous peers.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from atomic_io import load_torch_checkpoint

from model import ModelConfig, TransformerForCausalLM, add_architecture_args, apply_architecture_args, compute_mtp_loss, count_parameters
from optim.build_optimizer import build_optimizer
from optim.lr_schedule import build_scheduler

from hivemind.hivemind_utils import (
    setup_hivemind_peer,
    build_hivemind_optimizer,
    average_checkpoints_via_hivemind,
    average_checkpoint_during_training,
    get_initial_peers_from_args,
    get_peer_seed,
    get_swarm_info,
    wrap_optimizers_for_hivemind,
    save_hivemind_checkpoint,
    load_hivemind_checkpoint,
    maybe_average_final_checkpoint,
    add_hivemind_args,
    check_hivemind_args,
    measure_peer_throughputs,
    compute_adaptive_target_batch_size,
    get_fast_peer_subset,
    try_init_wandb,
    log_wandb,
    _HIVEMIND_AVAILABLE,
)


# ---------------------------------------------------------------------------
# LoRA/DoRA wrapper (import from parent project)
# ---------------------------------------------------------------------------

def _maybe_wrap_lora(
    model: nn.Module,
    config: ModelConfig,
    lora_rank: int = 0,
    lora_alpha: float = 1.0,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    use_rslora: bool = False,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Apply LoRA/DoRA/rsLoRA adapters if lora_rank > 0."""
    if lora_rank <= 0:
        return model
    try:
        from peft.lora import LoRAWrapper
    except ImportError:
        print("[LoRA] Could not import peft.lora. Make sure it's in your PYTHONPATH.")
        raise

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    model = LoRAWrapper(
        model,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        use_dora=use_dora,
        use_rslora=use_rslora,
        target_modules=target_modules,
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[LoRA] rank={lora_rank} alpha={lora_alpha} dora={use_dora} rslora={use_rslora}")
    print(f"       trainable params: {n_trainable:,} ({n_trainable / 1e6:.2f}M)")
    return model


# ---------------------------------------------------------------------------
# SFT loss (with optional LoRA)
# ---------------------------------------------------------------------------

def sft_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    mtp_discount: float = 0.5,
) -> torch.Tensor:
    """SFT cross-entropy loss with optional MTP and MoD auxiliary losses."""
    out = model(input_ids, attention_mask=attention_mask)
    logits = out["logits"]  # (B, T, V)
    # Shift: logits[t] predicts token at t+1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )
    # Multi-Token Prediction (MTP) auxiliary loss
    if "mtp_logits" in out:
        mtp_loss = compute_mtp_loss(
            out["mtp_logits"], labels,
            discount=mtp_discount, ignore_index=-100,
        )
        loss += mtp_loss
    # Mixture-of-Depth (MoD) auxiliary loss
    loss += out.get("mod_aux_loss", 0.0)
    return loss


# ---------------------------------------------------------------------------
# Packed SFT data loader (reads packed_sft_*.bin + packed_sft_mask_*.bin)
# ---------------------------------------------------------------------------

class PackedSFTLoader:
    """
    Streams packed SFT token arrays and loss masks from disk.

    Expects files:
      - ``packed_sft_{split}_*.bin``  — token IDs (uint16)
      - ``packed_sft_mask_{split}_*.bin`` — loss masks (uint8, 1 = compute loss)

    Yields (input_ids, labels, attention_mask) per batch.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        batch_size: int,
        pad_id: int = 0,
        rank: int = 0,
        world_size: int = 1,
        split: str = "train",
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pad_id = pad_id

        # Find token files
        token_patterns = sorted(glob.glob(
            os.path.join(data_dir, f"packed_sft_{split}*.bin")
        ))
        mask_patterns = sorted(glob.glob(
            os.path.join(data_dir, f"packed_sft_mask_{split}*.bin")
        ))
        if not token_patterns:
            # Try without split prefix
            token_patterns = sorted(glob.glob(
                os.path.join(data_dir, "packed_sft*.bin")
            ))
            mask_patterns = sorted(glob.glob(
                os.path.join(data_dir, "packed_sft_mask*.bin")
            ))

        if not token_patterns:
            raise FileNotFoundError(f"No packed SFT data found in {data_dir}")

        # Load memmaps
        self.token_memmaps = [np.memmap(p, dtype=np.uint16, mode="r") for p in token_patterns]
        if mask_patterns:
            self.mask_memmaps = [np.memmap(p, dtype=np.uint8, mode="r") for p in mask_patterns]
        else:
            self.mask_memmaps = [np.ones_like(mm, dtype=np.uint8) for mm in self.token_memmaps]

        # Compute cumulative sizes and document starts
        self.cumulative_offsets: List[int] = []
        total = 0
        for i, mm in enumerate(self.token_memmaps):
            self.cumulative_offsets.append(total)
            total += len(mm)
        self.cumulative_offsets.append(total)

        # Each sample = a random start position + seq_len
        self.total_starts = total // seq_len
        self.doc_starts = list(range(0, max(1, total - seq_len), seq_len))
        n = len(self.doc_starts)
        shard_size = n // max(1, world_size)
        start_idx = rank * shard_size
        end_idx = start_idx + shard_size if rank < world_size - 1 else n
        self.doc_starts = self.doc_starts[start_idx:end_idx]

        self.gen = torch.Generator().manual_seed(seed * 1_000_003 + rank * 31)
        print(f"[SFTPack] {split}: {len(self.doc_starts)} starts from {total:,} tokens, "
              f"shard [{start_idx}:{end_idx})")

    def _read_window(self, start: int, length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Read tokens and mask starting at position ``start``."""
        result_tok = np.full(length, self.pad_id, dtype=np.uint16)
        result_mask = np.zeros(length, dtype=np.uint8)
        written = 0
        pos = start
        for fi in range(len(self.token_memmaps)):
            fs = self.cumulative_offsets[fi]
            fe = self.cumulative_offsets[fi + 1]
            if pos >= fe or pos < fs:
                continue
            local = pos - fs
            take = min(length - written, fe - pos)
            result_tok[written:written + take] = self.token_memmaps[fi][local:local + take]
            result_mask[written:written + take] = self.mask_memmaps[fi][local:local + take]
            written += take
            pos += take
            if written >= length:
                break
        return result_tok, result_mask

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.doc_starts:
            raise StopIteration
        idxs = torch.randint(0, len(self.doc_starts), (self.batch_size,), generator=self.gen)
        input_ids_list = []
        labels_list = []
        mask_list = []
        for idx in idxs:
            start = self.doc_starts[idx.item()]
            tok, mask = self._read_window(start, self.seq_len + 1)
            input_ids_list.append(tok[:self.seq_len])
            # Shuffle target: token t should predict token t+1
            labels_list.append(tok[1:self.seq_len + 1])
            mask_list.append(mask[:self.seq_len])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = torch.from_numpy(np.stack(input_ids_list).astype(np.int64)).to(device)
        labels = torch.from_numpy(np.stack(labels_list).astype(np.int64)).to(device)
        attn_mask = torch.from_numpy(np.stack(mask_list).astype(np.int64)).to(device)

        # Replace padding positions in labels with -100 (ignore in CE loss)
        labels[attn_mask == 0] = -100

        return input_ids, labels, attn_mask


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_sft(
    model: nn.Module,
    val_loader: PackedSFTLoader,
    max_batches: int = 50,
    ctx: Any = nullcontext(),
) -> float:
    """Evaluate SFT validation loss."""
    model.eval()
    losses: List[float] = []
    for _ in range(max_batches):
        try:
            input_ids, labels, attn_mask = next(val_loader)
            with ctx:
                loss = sft_loss(model, input_ids, labels, attn_mask)
            losses.append(loss.item())
        except StopIteration:
            break
    model.train()
    return float(np.mean(losses)) if losses else 0.0


# ---------------------------------------------------------------------------
# Checkpoint save/load (SFT-specific)
# ---------------------------------------------------------------------------

def save_sft_checkpoint(
    checkpoint_dir: str,
    step: int,
    model: nn.Module,
    optimizers: List[torch.optim.Optimizer],
    config: ModelConfig,
    train_args: Optional[Dict] = None,
    lora_rank: int = 0,
) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    raw = getattr(model, "_orig_mod", model)

    ckpt: Dict[str, Any] = {
        "step": step,
        "model_state": raw.state_dict(),
        "lora_rank": lora_rank,
    }
    if len(optimizers) == 1:
        # Handle Hivemind wrapper
        inner = optimizers[0].opt if hasattr(optimizers[0], "opt") else optimizers[0]
        ckpt["optimizer_state"] = inner.state_dict()
    else:
        ckpt["optimizer_state"] = [
            opt.opt.state_dict() if hasattr(opt, "opt") else opt.state_dict()
            for opt in optimizers
        ]

    step_path = os.path.join(checkpoint_dir, f"sft_step_{step:05d}.pt")
    torch.save(ckpt, step_path)

    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2, default=str)

    # Symlink
    latest = os.path.join(checkpoint_dir, "latest_checkpoint")
    if os.path.islink(latest) or os.path.exists(latest):
        if os.path.islink(latest):
            os.remove(latest)
        else:
            os.remove(latest)
    os.symlink(os.path.abspath(step_path), latest)

    print(f"[SFT] saved {step_path}")
    return step_path


def load_sft_checkpoint(
    path: str,
    model: Optional[nn.Module] = None,
    optimizers: Optional[List] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, int]:
    """Load SFT checkpoint. Returns (step, lora_rank)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.isdir(path):
        step_files = sorted(glob.glob(os.path.join(path, "sft_step_*.pt")))
        if not step_files:
            raise FileNotFoundError(f"No sft_step_*.pt in {path}")
        ckpt_path = step_files[-1]
    else:
        ckpt_path = path

    ckpt = load_torch_checkpoint(ckpt_path, map_location=device)

    if model is not None:
        raw = getattr(model, "_orig_mod", model)
        raw.load_state_dict(ckpt["model_state"])
        if hasattr(raw, "tie_weights"):
            raw.tie_weights()

    if optimizers is not None and "optimizer_state" in ckpt:
        opt_state = ckpt["optimizer_state"]
        if isinstance(opt_state, list):
            for opt, st in zip(optimizers, opt_state):
                inner = opt.opt if hasattr(opt, "opt") else opt
                inner.load_state_dict(st)
        else:
            inner = optimizers[0].opt if hasattr(optimizers[0], "opt") else optimizers[0]
            inner.load_state_dict(opt_state)

    step = ckpt.get("step", 0)
    lora_rank = ckpt.get("lora_rank", 0)
    print(f"[SFT] resumed from {ckpt_path} at step {step}")
    return step, lora_rank


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train(
    args: argparse.Namespace,
    hivemind_info: Any = None,
) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if hivemind_info is not None:
        peer_seed = get_peer_seed(args, hivemind_info.endpoint)
    else:
        peer_seed = args.seed

    torch.manual_seed(peer_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- config and model ---------------------------------------------------
    max_pos = getattr(args, "max_seq_len", None) or 8192
    if args.resume:
        resume_dir = args.resume if os.path.isdir(args.resume) else os.path.dirname(args.resume)
        config_path = os.path.join(resume_dir, "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            config = ModelConfig(**cfg)
        else:
            config = _build_config(args, args.vocab_size or 65536)
    elif args.model_size:
        target = ModelConfig.parse_param_count(args.model_size)
        config = ModelConfig.from_target_size(target, vocab_size=args.vocab_size or 65536,
                                              max_position_embeddings=max_pos)
    elif args.checkpoint_dir or args.checkpoint:
        # Load from pretrained checkpoint
        ckpt_source = args.checkpoint_dir or args.checkpoint
        config_path = os.path.join(ckpt_source, "config.json") if os.path.isdir(ckpt_source) else ""
        if os.path.isfile(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            config = ModelConfig(**cfg)
        else:
            raise ValueError("Could not find config.json in checkpoint source")
    else:
        config = _build_config(args, args.vocab_size or 65536)

    model = TransformerForCausalLM(config).to(device)

    # ---- LoRA ---------------------------------------------------------------
    model = _maybe_wrap_lora(
        model, config,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_dora=args.use_dora,
        use_rslora=args.use_rslora,
    )

    n_params = count_parameters(model)
    print(f"Model params: {n_params:,} ({n_params / 1e6:.2f}M)")

    if args.gradient_checkpointing:
        model.model.enable_gradient_checkpointing()

    if args.jit:
        print(f"[compile] torch.compile(mode='{args.compile_mode}')…")
        model = torch.compile(model, mode=args.compile_mode)

    # ---- data ---------------------------------------------------------------
    if hivemind_info is not None:
        swarm_size, peer_idx = get_swarm_info(
            hivemind_info.peer, args.target_group_size, hivemind_info.endpoint
        )
    else:
        swarm_size = 1
        peer_idx = 0

    train_loader = PackedSFTLoader(
        args.data_dir, args.seq_len, args.batch_size,
        pad_id=0, rank=peer_idx, world_size=swarm_size,
        split="train", seed=args.seed,
    )
    val_loader = PackedSFTLoader(
        args.data_dir, args.seq_len, args.batch_size,
        pad_id=0, rank=peer_idx, world_size=swarm_size,
        split="val", seed=args.seed,
    )
    train_iter = iter(train_loader)

    # ---- LR scaling ---------------------------------------------------------
    if args.model_size and not args.no_lr_scale:
        ref_hidden = 2048
        scale = math.sqrt(ref_hidden / config.hidden_size)
        scale = max(0.5, min(scale, 2.0))
        args.lr = args.lr * scale
        args.min_lr = args.min_lr * scale
        print(f"[LR] Auto-scaled to {args.lr:.2e} (x{scale:.3f})")

    # ---- resume -------------------------------------------------------------
    start_step = 0
    if args.resume:
        start_step, _ = load_sft_checkpoint(args.resume, model, None, device)

    # ---- optimizer ----------------------------------------------------------
    local_opts = build_optimizer(
        model,
        optimizer_type=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if isinstance(local_opts, tuple):
        local_opts = list(local_opts)
    else:
        local_opts = [local_opts]

    if args.hivemind:
        if hivemind_info is None:
            raise RuntimeError("Hivemind flag set but no peer info.")
        initial_sync = (start_step == 0)
        optimizers = wrap_optimizers_for_hivemind(
            model=model,
            local_optimizers=local_opts,
            peer=hivemind_info.peer,
            args=args,
            prefix="sft_hivemind",
            verbose=True,
            initial_sync=initial_sync,
        )
    else:
        optimizers = local_opts

    # ---- load optimizer state after Hivemind wrapping (if resuming) ---------
    if args.resume:
        _, _ = load_sft_checkpoint(args.resume, None, optimizers, device)

    # ---- LR schedule --------------------------------------------------------
    scheduler = build_scheduler(
        schedule=args.schedule,
        warmup_steps=args.warmup_steps,
        max_steps=args.num_steps,
        peak_lr=args.lr,
        min_lr=args.min_lr,
        stable_ratio=args.stable_ratio,
    )

    # ---- AMP ----------------------------------------------------------------
    use_amp = device.type == "cuda" and args.dtype == "bf16"
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    # ---- W&B ----------------------------------------------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    use_wandb = args.wandb_project and try_init_wandb(
        args, config, n_params,
    )

    # ---- tokens accounting --------------------------------------------------
    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
    print(f"\nTokens / local step : {tokens_per_step:,}")
    print(f"Total steps : {args.num_steps:,}")
    if args.hivemind:
        print(f"Swarm               : {args.target_group_size}+ peers (async)")
        print(f"Hivemind accum steps: {args.hivemind_accumulation_steps}")
        print(f"Adaptive batch      : {args.hivemind_adaptive_batch}")
    print()

    # ================================================================== LOOP
    model.train()
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)

    t0 = time.perf_counter()
    loss_accum = 0.0
    step_local = start_step

    # Hivemind async averaging state
    avg_future = None
    local_accum_counter = 0

    # Bandwidth-aware peer weighting state
    peer_throughputs = {}  # peer_id -> tokens/sec
    last_throughput_update = 0

    try:
        while step_local < args.num_steps:
            lr_val = scheduler(step_local)
            for opt in optimizers:
                inner = opt.opt if hasattr(opt, "opt") else opt
                for pg in inner.param_groups:
                    pg["lr"] = lr_val

            # gradient accumulation
            for micro_step in range(args.grad_accum):
                input_ids, labels, attn_mask = next(train_iter)
                with amp_ctx:
                    loss = sft_loss(
                        model, input_ids, labels, attn_mask,
                        mtp_discount=args.mtp_discount,
                    ) / args.grad_accum
                loss.backward()
                loss_accum += loss.item()

            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)

            step_local += 1
            local_accum_counter += 1

            # ---- Hivemind: adaptive target_batch_size based on swarm size ----
            if args.hivemind and args.hivemind_adaptive_batch and local_accum_counter == 1:
                # Update target_batch_size on first step and periodically
                try:
                    visible_peers = len(hivemind_info.peer.get_visible_peers())
                    swarm_size = max(visible_peers, args.target_group_size)
                    new_target = compute_adaptive_target_batch_size(
                        args.batch_size,
                        swarm_size,
                        min_peers=args.hivemind_min_peers,
                        max_scale=args.hivemind_max_scale,
                    )
                    if new_target != args.batch_size:
                        for opt in optimizers:
                            if hasattr(opt, 'target_batch_size'):
                                opt.target_batch_size = new_target
                        print(f"[Hivemind] Adaptive target_batch_size: {new_target} (swarm={swarm_size})")
                except Exception as e:
                    print(f"[Hivemind] Adaptive batch sizing skipped: {e}")

            # ---- Hivemind: bandwidth-aware peer throughput measurement ------
            if args.hivemind and args.hivemind_bandwidth_aware and step_local % args.hivemind_throughput_window == 0:
                try:
                    # Measure local throughput
                    local_tok_per_sec = tokens_per_step * args.log_interval / max(time.perf_counter() - t0, 1e-9)
                    peer_throughputs = measure_peer_throughputs(
                        hivemind_info.peer,
                        local_tok_per_sec,
                        window=args.hivemind_throughput_window,
                    )
                    # Get fast peer subset for potential weighted averaging
                    fast_peers = get_fast_peer_subset(
                        peer_throughputs,
                        min_throughput=1000.0,
                        max_peers=32,
                    )
                    print(f"[Hivemind] Local throughput: {local_tok_per_sec/1e3:.1f}k tok/s, fast peers: {len(fast_peers)}")
                except Exception as e:
                    print(f"[Hivemind] Throughput measurement skipped: {e}")

            # ---- Hivemind: async averaging with compute/comm overlap ---------
            if args.hivemind and local_accum_counter >= args.hivemind_accumulation_steps:
                # Wait for previous async average to complete (non-blocking check)
                if avg_future is not None:
                    if hasattr(avg_future, 'ready') and avg_future.ready():
                        try:
                            avg_future.result()  # raises if error
                        except Exception as e:
                            print(f"[Hivemind] Async average error: {e}")
                        avg_future = None
                    # If not ready, skip this round - don't queue up

                # Launch new async average
                if avg_future is None:
                    try:
                        for opt in optimizers:
                            if hasattr(opt, 'average_parameters'):
                                avg_future = opt.average_parameters(async_op=True)
                                break
                    except Exception as e:
                        print(f"[Hivemind] Failed to launch async average: {e}")

                local_accum_counter = 0

            # logging
            if step_local % args.log_interval == 0 or step_local == 1:
                t1 = time.perf_counter()
                dt = max(t1 - t0, 1e-9)
                tok_per_sec = tokens_per_step * args.log_interval / dt
                peer_tag = f"[{hivemind_info.endpoint}] " if hivemind_info else ""
                print(
                    f"{peer_tag}sft step {step_local:7d} | loss "
                    f"{loss_accum / args.log_interval:.4f} | lr {lr_val:.2e} | "
                    f"{tok_per_sec / 1e3:.1f}k tok/s"
                )
                if use_wandb:
                    log_wandb({"train/loss": loss_accum / args.log_interval,
                               "train/lr": lr_val}, step=step_local)
                loss_accum = 0.0
                t0 = t1

            # validation
            if step_local % args.val_every == 0 and step_local > start_step + 1:
                val_loss = validate_sft(
                    model, val_loader, max_batches=args.eval_steps, ctx=amp_ctx,
                )
                print(f"  [eval] step {step_local:7d} | val_loss {val_loss:.4f}")
                if use_wandb:
                    log_wandb({"val/loss": val_loss}, step=step_local)

            # checkpoint
            if step_local % args.save_every == 0 and step_local > start_step:
                save_sft_checkpoint(
                    args.checkpoint_dir, step_local, model, optimizers,
                    config, train_args=vars(args), lora_rank=args.lora_rank,
                )
                _prune_ckpts(args.checkpoint_dir, args.keep_ckpts)

    except KeyboardInterrupt:
        print(f"\n[SFT] Interrupted at step {step_local}.")

    # If Hivemind enabled, trigger cross-swarm averaging BEFORE saving final checkpoint
    if args.hivemind and hivemind_info is not None:
        print(f"\n[Hivemind] Running cross-swarm averaging before final checkpoint...")
        try:
            # Trigger averaging with current model state
            for opt in optimizers:
                if hasattr(opt, 'average_parameters'):
                    # Blocking wait for convergence
                    opt.average_parameters(async_op=False)
                    time.sleep(2.0)  # allow propagation
                    break
        except Exception as e:
            print(f"[Hivemind] Cross-swarm averaging error: {e}")

    save_sft_checkpoint(
        args.checkpoint_dir, step_local, model, optimizers,
        config, train_args=vars(args), lora_rank=args.lora_rank,
    )
    print(f"SFT complete at step {step_local}.")

    if args.hivemind and args.average_checkpoints and hivemind_info is not None:
        avg_state = average_checkpoints_via_hivemind(
            model, hivemind_info.peer,
            target_group_size=min(args.target_group_size, 8),
            num_rounds=args.checkpoint_average_rounds,
        )
        torch.save({"model_state": avg_state, "config": vars(config)},
                   os.path.join(args.checkpoint_dir, "sft_averaged_final.pt"))


def _prune_ckpts(checkpoint_dir: str, keep: int = 3) -> None:
    ckpts = sorted(
        Path(checkpoint_dir).glob("sft_step_*.pt"),
        key=lambda p: int(p.stem.replace("sft_step_", "")),
    )
    for old in ckpts[:-keep]:
        old.unlink()


def _build_config(args, vocab_size):
    max_pos = getattr(args, "max_seq_len", None) or 8192
    hidden_size = args.hidden_size or 2048
    config = ModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=args.intermediate_size or hidden_size * 3,
        num_hidden_layers=args.num_layers or 28,
        num_attention_heads=args.num_heads or max(4, hidden_size // 128),
        num_key_value_heads=args.num_kv_heads or max(1, hidden_size // 512),
        head_dim=args.head_dim or 128,
        max_position_embeddings=max_pos,
    )
    apply_architecture_args(config, args)
    return config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hivemind-based SFT training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-size", default=None,
                   help="Target model size (e.g. '300M', '1.7B').")
    p.add_argument("--checkpoint", "--checkpoint-dir", default=None, dest="checkpoint_dir",
                   help="Path to pretrained checkpoint directory (for continued SFT).")
    p.add_argument("--vocab-size", type=int, default=65536)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--num-kv-heads", type=int, default=None)
    p.add_argument("--intermediate-size", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=None)

    add_architecture_args(p)

    p.add_argument("--data-dir", default="./sft_packed")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=50000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--min-lr", type=float, default=2e-6)
    p.add_argument("--no-lr-scale", action="store_true")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--schedule", default="cosine", choices=["cosine", "wsd"])
    p.add_argument("--stable-ratio", type=float, default=0.8)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    p.add_argument("--jit", action="store_true")
    p.add_argument("--compile-mode", default="default")
    p.add_argument("--gradient-checkpointing", action="store_true")
    # Checkpoint directory already defined as --checkpoint/--checkpoint-dir alias above
    p.add_argument("--resume", default=None)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--keep-ckpts", type=int, default=3)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--seed", type=int, default=42)

    # LoRA
    p.add_argument("--lora-rank", type=int, default=0,
                   help="LoRA rank (0 = full fine-tune).")
    p.add_argument("--lora-alpha", type=float, default=1.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--use-dora", action="store_true", help="Enable DoRA.")
    p.add_argument("--use-rslora", action="store_true", help="Enable rsLoRA.")

    # Hivemind
    add_hivemind_args(p)
    p.add_argument("--average-checkpoints", action="store_true", default=False,
                   help="After training, average parameters across the swarm "
                        "to produce a merged evaluation checkpoint.")
    # Async averaging overlap
    p.add_argument("--hivemind-accumulation-steps", type=int, default=8,
                   help="Local gradient accumulation steps before triggering "
                        "all-reduce (reduces communication frequency).")
    p.add_argument("--hivemind-adaptive-batch", action="store_true", default=True,
                   help="Adapt target_batch_size based on swarm size.")
    p.add_argument("--hivemind-min-peers", type=int, default=4,
                   help="Minimum peers for adaptive batch scaling.")
    p.add_argument("--hivemind-max-scale", type=float, default=4.0,
                   help="Maximum batch size scaling factor.")
    p.add_argument("--hivemind-bandwidth-aware", action="store_true", default=False,
                   help="Weight peer contributions by measured throughput (experimental).")
    p.add_argument("--hivemind-throughput-window", type=int, default=100,
                   help="Window size for peer throughput measurement.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    hivemind_info = None
    if args.hivemind:
        check_hivemind_args(args)
        initial = get_initial_peers_from_args(args)
        hivemind_info = setup_hivemind_peer(initial, host=args.host, port=args.port)
        print(f"[Hivemind] Peer endpoint: {hivemind_info.endpoint}")

    try:
        _train(args, hivemind_info=hivemind_info)
    finally:
        if hivemind_info is not None:
            try:
                hivemind_info.peer.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
