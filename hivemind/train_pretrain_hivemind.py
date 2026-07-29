#!/usr/bin/env python3
"""
train_pretrain_hivemind.py

Decentralized pretraining with Hivemind — pool laptops, gaming PCs, and
cloud GPUs into a single asynchronous training swarm.

How it works
------------
  1. Start a **bootstrap peer** (one machine):
       python hivemind/train_pretrain_hivemind.py --hivemind --initial-peers ""
  2. Every other machine joins by pointing to any already-connected peer:
       python hivemind/train_pretrain_hivemind.py --hivemind --initial-peers <bootstrap_ip>:<port>
  3. Each peer trains locally at its own speed and asynchronously averages
     parameters with ``--target-group-size`` random peers on every step.
  4. Faster GPUs contribute more updates; slower ones still help.

Usage
-----
  # Bootstrap (first node, also trains):
      python hivemind/train_pretrain_hivemind.py \\
          --hivemind --initial-peers "" --port 5678 \\
          --model-size 300M --data-dir ./packed --checkpoint-dir ./ckpts_bootstrap

  # Worker nodes (laptop, second desktop, etc.):
      python hivemind/train_pretrain_hivemind.py \\
          --hivemind --initial-peers "192.168.1.100:5678" \\
          --model-size 300M --data-dir ./packed --checkpoint-dir ./ckpts_worker1

  # Resume and average final checkpoint:
      python hivemind/train_pretrain_hivemind.py --hivemind ... --resume ./ckpts_worker1
      python hivemind/train_pretrain_hivemind.py \\
          --hivemind --average-checkpoints --initial-peers "..." \\
          --checkpoint-dir ./final_model

Key differences from ``train_pretrain.py``
------------------------------------------
  - No ``torchrun`` / DDP required — each peer runs independently.
  - ``--hivemind`` flag enables the decentralized optimizer.
  - ``--initial-peers`` controls swarm discovery.
  - ``--target-group-size`` sets averaging fan-out (more = slower but stabler).
  - Checkpoints are local per peer; use ``--average-checkpoints`` to
    produce a merged evaluation checkpoint from the full swarm.
  - MFU / tok/s are reported **per peer** — expect slower peers to show
    lower throughput; that is normal.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import socket
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import parent project modules
import sys
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from model import ModelConfig, TransformerForCausalLM, count_parameters
from optim.build_optimizer import build_optimizer
from optim.lr_schedule import build_scheduler

# Import Hivemind utilities
from hivemind.hivemind_utils import (
    setup_hivemind_peer,
    build_hivemind_optimizer,
    average_checkpoints_via_hivemind,
    get_initial_peers_from_args,
    add_hivemind_args,
    check_hivemind_args,
    PackedDataLoader,
    pretrain_loss,
    validate,
    estimate_mfu,
    save_checkpoint,
    load_checkpoint,
    get_lr,
    build_optimizer_groups,
    try_init_wandb,
    log_wandb,
    _HIVEMIND_AVAILABLE,
)

# ---------------------------------------------------------------------------
# W&B (optional)
# ---------------------------------------------------------------------------

def try_init_wandb_hivemind(
    args: argparse.Namespace,
    config: ModelConfig,
    n_params: int,
    peer_endpoint: str = "",
) -> bool:
    """Initialise W&B with Hivemind-specific tags."""
    try:
        import wandb
        run_name = args.wandb_run_name or f"hivemind-{args.model_size or 'custom'}"
        if peer_endpoint:
            run_name += f"-{peer_endpoint.replace(':', '_')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                **vars(config),
                "n_params": n_params,
                "hivemind": True,
                "target_group_size": args.target_group_size,
                "averaging_period": args.averaging_period,
                "average_parameters": args.average_parameters,
                "initial_peers": args.initial_peers,
                **{k: v for k, v in vars(args).items()
                   if k not in ("wandb_project", "wandb_run_name", "initial_peers")},
            },
        )
        return True
    except Exception as e:
        print(f"[W&B] disabled: {e}")
        return False


# ---------------------------------------------------------------------------
# Data availability check
# ---------------------------------------------------------------------------

def _has_data(data_dir: str) -> bool:
    if not os.path.isdir(data_dir):
        return False
    if glob.glob(os.path.join(data_dir, "pretrain_tokens*.bin")):
        return True
    if os.path.isfile(os.path.join(data_dir, "train.bin")):
        return True
    return False


# ---------------------------------------------------------------------------
# Config builder (same as train_pretrain.py)
# ---------------------------------------------------------------------------

def _build_config(args: argparse.Namespace, vocab_size: int) -> ModelConfig:
    max_pos = getattr(args, "max_seq_len", None) or 8192
    if args.model_size:
        target_params = ModelConfig.parse_param_count(args.model_size)
        config = ModelConfig.from_target_size(
            target_params,
            vocab_size=vocab_size,
            max_position_embeddings=max_pos,
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
            max_position_embeddings=max_pos,
        )
    return config


# ---------------------------------------------------------------------------
# Hivemind peer index (stable hash of endpoint for data sharding)
# ---------------------------------------------------------------------------

def _peer_shard_index(endpoint: str, world_size: int) -> int:
    """Deterministic peer index from its endpoint, used for data sharding."""
    return hash(endpoint) % world_size if world_size > 1 else 0


# ---------------------------------------------------------------------------
# Checkpoint pruning
# ---------------------------------------------------------------------------

def _prune_checkpoints(checkpoint_dir: str, keep: int = 3) -> None:
    ckpts = sorted(
        Path(checkpoint_dir).glob("step_*.pt"),
        key=lambda p: int(p.stem.replace("step_", "")),
    )
    for old in ckpts[:-keep]:
        old.unlink()
        print(f"[Checkpoint] pruned {old}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def _train(
    args: argparse.Namespace,
    hivemind_info: Any = None,
) -> None:
    """
    Inner training routine (single process — no DDP).

    Args:
        args: Parsed CLI arguments.
        hivemind_info: ``HivemindPeerInfo`` from ``setup_hivemind_peer()``,
            or ``None`` if not using Hivemind (single-machine mode).
    """
    master = True   # every peer is its own "master" for local logging

    # Reproducibility (each peer has its own seed based on endpoint hash)
    if hivemind_info is not None:
        peer_seed = args.seed + _peer_shard_index(hivemind_info.endpoint, 65536)
    else:
        peer_seed = args.seed
    torch.manual_seed(peer_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- device -------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if hivemind_info is not None:
            print(f"[Peer {hivemind_info.endpoint}] Using GPU: "
                  f"{torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Using Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU — training will be slow.")

    # ---- meta ---------------------------------------------------------------
    meta_path = os.path.join(args.data_dir, "meta.json")
    meta: Dict[str, Any] = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    vocab_size = meta.get("vocab_size", args.vocab_size or 65536)

    # ---- model --------------------------------------------------------------
    if args.resume:
        if os.path.isdir(args.resume):
            config_path = os.path.join(args.resume, "config.json")
            if os.path.isfile(config_path):
                with open(config_path) as f:
                    ckpt_config = json.load(f)
                config = ModelConfig(**ckpt_config)
            else:
                config = _build_config(args, vocab_size)
        else:
            config = _build_config(args, vocab_size)
    else:
        config = _build_config(args, vocab_size)

    model = TransformerForCausalLM(config).to(device)
    n_params = count_parameters(model)

    print(f"Model params : {n_params:,}  ({n_params / 1e9:.3f}B)")
    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU VRAM     : {vram:.1f} GB")

    # ---- gradient checkpointing ---------------------------------------------
    if args.gradient_checkpointing:
        model.model.enable_gradient_checkpointing()

    # ---- torch.compile ------------------------------------------------------
    if args.jit:
        print(f"[compile] torch.compile(mode='{args.compile_mode}')…")
        model = torch.compile(model, mode=args.compile_mode)

    _use_cudagraphs = args.jit and args.compile_mode == "reduce-overhead"

    # ---- data ---------------------------------------------------------------
    # Each peer gets its own shard of the data based on its endpoint hash.
    # If no Hivemind, fall back to rank=0, world_size=1.
    if hivemind_info is not None:
        # Estimate swarm size from target_group_size (conservative — use actual
        # visible peers if available, else default to target_group_size)
        try:
            visible = len(hivemind_info.peer.get_visible_peers())
            swarm_size = max(visible, args.target_group_size)
        except Exception:
            swarm_size = args.target_group_size
        peer_idx = _peer_shard_index(hivemind_info.endpoint, swarm_size)
    else:
        swarm_size = 1
        peer_idx = 0

    train_loader = PackedDataLoader(
        args.data_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        pad_id=0,
        rank=peer_idx,
        world_size=swarm_size,
        split="train",
        seed=args.seed,
    )
    val_loader = PackedDataLoader(
        args.data_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        pad_id=0,
        rank=peer_idx,
        world_size=swarm_size,
        split="val",
        seed=args.seed,
    )
    train_iter = iter(train_loader)

    # ---- auto LR scaling ----------------------------------------------------
    if args.model_size and not args.no_lr_scale:
        ref_hidden = 2048
        scale = math.sqrt(ref_hidden / config.hidden_size)
        scale = max(0.5, min(scale, 2.0))
        original_lr = args.lr
        args.lr = args.lr * scale
        args.min_lr = args.min_lr * scale
        print(f"[LR] Auto-scaled from {original_lr:.2e} to {args.lr:.2e} "
              f"(x{scale:.3f}, hidden={config.hidden_size})")

    # ---- build local optimizer(s) -------------------------------------------
    # We need to build the optimizer BEFORE wrapping with Hivemind.
    # build_optimizer_groups returns a list — for Hivemind we unwrap single-opitmizer
    # lists and wrap the optimizer directly.
    local_opts = build_optimizer_groups(
        model,
        optimizer_type=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ---- wrap with Hivemind DecentralizedOptimizer --------------------------
    if args.hivemind:
        if hivemind_info is None:
            raise RuntimeError("Hivemind flag set but no peer info provided.")
        # For simplicity: wrap the first optimizer (AdamW or Muon main).
        # Muon+AdamW dual optimiser case: wrap both separately.
        hivemind_opts = []
        for i, opt in enumerate(local_opts):
            hopt = build_hivemind_optimizer(
                model=model,
                base_optimizer=opt,
                peer=hivemind_info.peer,
                target_group_size=args.target_group_size,
                averaging_period=args.averaging_period,
                average_parameters=args.average_parameters,
                prefix=f"hivemind_{i}",
                verbose=True,
            )
            hivemind_opts.append(hopt)
        optimizers = hivemind_opts   # use wrapped optimizers
        print(f"[Hivemind] Wrapped {len(optimizers)} optimizer(s) for async averaging.")
    else:
        optimizers = local_opts   # plain local optimizers (no Hivemind)

    # ---- LR scheduler -------------------------------------------------------
    scheduler = build_scheduler(
        schedule=args.schedule,
        warmup_steps=args.warmup_steps,
        max_steps=args.num_steps,
        peak_lr=args.lr,
        min_lr=args.min_lr,
        stable_ratio=args.stable_ratio,
    )

    # ---- AMP context --------------------------------------------------------
    use_amp = device.type == "cuda" and args.dtype == "bf16"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp
        else nullcontext()
    )

    # ---- resume -------------------------------------------------------------
    start_step = 0
    total_tokens = 0
    best_val_loss = float("inf")
    if args.resume:
        start_step, best_val_loss, total_tokens = load_checkpoint(
            args.resume, model, optimizers, device,
        )
        raw = getattr(model, "_orig_mod", model)  # unwrap torch.compile
        if hasattr(raw, "tie_weights"):
            raw.tie_weights()

    # ---- MFU ----------------------------------------------------------------
    gpu_peak_tflops = _get_peak_tflops(device)
    print(f"GPU peak bf16: {gpu_peak_tflops:.1f} TFLOPS")

    # ---- W&B (per peer — each peer logs independently) ----------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    use_wandb = (
        args.wandb_project
        and try_init_wandb_hivemind(
            args, config, n_params,
            peer_endpoint=hivemind_info.endpoint if hivemind_info else "",
        )
    )

    # ---- tokens accounting --------------------------------------------------
    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
    print(f"\nTokens / local step : {tokens_per_step:,}")
    print(f"Local batch size    : {args.batch_size} × {args.grad_accum} accum")
    print(f"Total steps         : {args.num_steps:,}")
    if args.hivemind:
        print(f"Swarm               : {args.target_group_size}+ peers (async)")
    print(f"Checkpoint every    : {args.save_every:,} steps")
    print()

    # ================================================================== LOOP
    model.train()
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)

    t0 = time.perf_counter()
    loss_accum = torch.zeros((), device=device)
    grad_norm_accum = torch.zeros((), device=device)
    tokens_accum = 0
    step_local = start_step  # local step counter

    try:
        while step_local < args.num_steps:

            # ---- LR ---------------------------------------------------------
            lr_val = scheduler(step_local)
            for opt in optimizers:
                # Hivemind optimizers wrap a local opt — access via opt.opt
                inner_opt = opt.opt if hasattr(opt, "opt") else opt
                for pg in inner_opt.param_groups:
                    pg["lr"] = lr_val

            # ---- gradient accumulation --------------------------------------
            for micro_step in range(args.grad_accum):
                x, y, num_valid = next(train_iter)

                with amp_ctx:
                    if _use_cudagraphs:
                        torch.compiler.cudagraph_mark_step_begin()
                    loss = pretrain_loss(
                        model, x, y, pad_id=0,
                        z_loss_weight=args.z_loss_weight,
                    ) / args.grad_accum
                loss.backward()

                loss_accum = loss_accum + loss.detach()
                tokens_accum += num_valid

            # ---- gradient clipping ------------------------------------------
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            grad_norm_accum = grad_norm_accum + grad_norm.detach()

            # ---- optimizer step (local + async all-reduce for Hivemind) ------
            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)

            total_tokens += tokens_per_step
            step_local += 1

            # ---- CUDA sync for accurate timing -----------------------------
            if device.type == "cuda":
                torch.cuda.synchronize()

            # ---- logging ----------------------------------------------------
            if (step_local % args.log_interval == 0) or (step_local == 1):
                t1 = time.perf_counter()
                dt = max(t1 - t0, 1e-9)
                tok_per_sec = tokens_per_step * args.log_interval / dt
                mfu = estimate_mfu(model, tok_per_sec, gpu_peak_tflops)

                loss_display = (loss_accum / args.log_interval).item()
                grad_norm_display = (grad_norm_accum / args.log_interval).item()
                loss_accum = torch.zeros((), device=device)
                grad_norm_accum = torch.zeros((), device=device)

                peer_tag = (f"[{hivemind_info.endpoint}] "
                           if hivemind_info else "")
                print(
                    f"{peer_tag}step {step_local:7d} | loss {loss_display:.4f} | "
                    f"lr {lr_val:.2e} | grad {grad_norm_display:.3f} | "
                    f"{tok_per_sec / 1e3:.1f}k tok/s | mfu {mfu * 100:.2f}%"
                )

                if use_wandb:
                    log_wandb(
                        {
                            "train/loss": loss_display,
                            "train/lr": lr_val,
                            "train/grad_norm": grad_norm_display,
                            "perf/tokens_per_sec": tok_per_sec,
                            "perf/mfu_pct": mfu * 100,
                            "system/step_local": step_local,
                        },
                        step=step_local,
                    )

                t0 = t1

            # ---- validation (every peer validates independently) ------------
            if step_local % args.val_every == 0 and step_local > start_step + 1:
                val_random, val_seq = validate(
                    model, val_loader, pad_id=0,
                    max_batches=args.eval_steps, ctx=amp_ctx,
                    use_cudagraphs=_use_cudagraphs,
                )
                val_loss = val_seq

                print(
                    f"  [eval] step {step_local:7d} | val_random {val_random:.4f} | "
                    f"val_seq {val_seq:.4f}"
                )
                if use_wandb:
                    log_wandb(
                        {
                            "val/loss": val_loss,
                            "val/loss_random": val_random,
                            "val/loss_seq": val_seq,
                        },
                        step=step_local,
                    )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

            # ---- checkpoint (each peer saves independently) -----------------
            if step_local % args.save_every == 0 and step_local > start_step:
                recipe = _build_recipe(args, config)
                save_checkpoint(
                    checkpoint_dir=args.checkpoint_dir,
                    step=step_local,
                    model=model,
                    optimizers=optimizers,
                    config=config,
                    total_tokens=total_tokens,
                    best_val_loss=best_val_loss,
                    recipe=recipe,
                    train_args=vars(args),
                )
                _prune_checkpoints(args.checkpoint_dir, keep=args.keep_ckpts)

    except KeyboardInterrupt:
        print(f"\n[Training] Interrupted at step {step_local}. Saving checkpoint...")

    # ---- final checkpoint ---------------------------------------------------
    recipe = _build_recipe(args, config)
    save_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        step=step_local,
        model=model,
        optimizers=optimizers,
        config=config,
        total_tokens=total_tokens,
        best_val_loss=best_val_loss,
        recipe=recipe,
        train_args=vars(args),
    )
    print(f"Training complete at step {step_local}. Best val loss: {best_val_loss:.4f}")

    # ---- optionally average checkpoints across peers ------------------------
    if args.hivemind and args.average_checkpoints and hivemind_info is not None:
        print("\n[Hivemind] Averaging final checkpoint across swarm...")
        avg_state = average_checkpoints_via_hivemind(
            model, hivemind_info.peer,
            target_group_size=min(args.target_group_size, 8),
            num_rounds=args.checkpoint_average_rounds,
        )
        avg_path = os.path.join(args.checkpoint_dir, "averaged_final.pt")
        torch.save({"model_state": avg_state, "config": vars(config)}, avg_path)
        print(f"[Hivemind] Averaged checkpoint saved to {avg_path}")


# ---------------------------------------------------------------------------
# Recipe builder
# ---------------------------------------------------------------------------

def _build_recipe(
    args: argparse.Namespace,
    config: ModelConfig,
) -> Dict[str, Any]:
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
            "hivemind": args.hivemind,
        },
        "data": {
            "data_dir": args.data_dir,
        },
    }


# ---------------------------------------------------------------------------
# GPU peak TFLOPS table (same as train_pretrain.py)
# ---------------------------------------------------------------------------

GPU_PEAK_TFLOPS: Dict[str, float] = {
    "NVIDIA A100-SXM4-80GB":      312.0,
    "NVIDIA A100-SXM4-40GB":      312.0,
    "NVIDIA A100-PCIE-40GB":      312.0,
    "NVIDIA H100 SXM5":           989.0,
    "NVIDIA H100 PCIe":           756.0,
    "NVIDIA H200":                989.0,
    "NVIDIA L40S":                362.0,
    "NVIDIA GeForce RTX 4090":    165.0,
    "NVIDIA GeForce RTX 3090":    165.0,
    "NVIDIA GeForce RTX 5090":    260.0,
    "NVIDIA GeForce RTX 4080":    120.0,
    "NVIDIA GeForce RTX 4070 Ti": 82.0,
    "NVIDIA RTX A6000":           155.0,
    "NVIDIA GeForce RTX 3060":    50.0,
    "NVIDIA GeForce RTX 3050":    36.0,
    "NVIDIA GeForce GTX 1080 Ti": 35.0,
    "AMD Instinct MI250X":        383.0,
    "AMD Instinct MI300X":        1307.0,
    "TPU-v4":                     275.0,
    "TPU-v5e":                    197.0,
    "Apple M1":                   5.0,
    "Apple M2":                   7.0,
    "Apple M3":                   12.0,
    "Apple M4":                   18.0,
}

_FALLBACK_TFLOPS: float = 30.0


def _get_peak_tflops(device: torch.device) -> float:
    """Look up the GPU's theoretical peak TFLOPS."""
    if device.type != "cuda":
        return _FALLBACK_TFLOPS
    name = torch.cuda.get_device_name(device)
    for key, val in GPU_PEAK_TFLOPS.items():
        if key.lower() in name.lower() or name.lower() in key.lower():
            return val
    print(f"[MFU] Unknown GPU '{name}', using {_FALLBACK_TFLOPS} TFLOPS fallback.")
    return _FALLBACK_TFLOPS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Hivemind-based decentralized pretraining.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- model (same as train_pretrain.py) ---------------------------------
    p.add_argument("--model-size", default=None,
                   help="Target model size (e.g. '600M', '1.7B', '70B', '1T').")
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--num-kv-heads", type=int, default=None)
    p.add_argument("--intermediate-size", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=None)

    # ---- data ---------------------------------------------------------------
    p.add_argument("--data-dir", default="./packed",
                   help="Directory containing pretrain_tokens*.bin files.")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--val-fraction", type=float, default=0.01)

    # ---- training -----------------------------------------------------------
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--num-steps", type=int, default=100_000)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--no-lr-scale", action="store_true")
    p.add_argument("--z-loss-weight", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])

    # ---- schedule -----------------------------------------------------------
    p.add_argument("--schedule", default="cosine", choices=["cosine", "wsd"])
    p.add_argument("--stable-ratio", type=float, default=0.8)

    # ---- optimizer ----------------------------------------------------------
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])

    # ---- compilation --------------------------------------------------------
    p.add_argument("--jit", action="store_true")
    p.add_argument("--compile-mode", default="default",
                   choices=["default", "reduce-overhead", "max-autotune"])

    # ---- memory -------------------------------------------------------------
    p.add_argument("--gradient-checkpointing", action="store_true")

    # ---- checkpointing ------------------------------------------------------
    p.add_argument("--checkpoint-dir", default="./checkpoints_hivemind")
    p.add_argument("--resume", default=None)
    p.add_argument("--save-every", type=int, default=5_000)
    p.add_argument("--keep-ckpts", type=int, default=3)

    # ---- logging / eval -----------------------------------------------------
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--val-every", "--eval-every", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--seed", type=int, default=42)

    # ---- Hivemind -----------------------------------------------------------
    add_hivemind_args(p)
    p.add_argument("--average-checkpoints", action="store_true", default=False,
                   help="After training, average parameters across the swarm "
                        "to produce a merged evaluation checkpoint.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not _has_data(args.data_dir):
        print("[Data] No packed data found. Please run data/pack_pretrain.py first.")
        print("       Example: python data/pack_pretrain.py --data-dir ./data "
              "--tokenizer ./tokenizer --cache-dir ./packed")
        print("       Or use smoke test mode by creating synthetic data.")
        return

    # ---- Hivemind setup ---------------------------------------------------
    hivemind_info = None
    if args.hivemind:
        check_hivemind_args(args)
        initial_peers = get_initial_peers_from_args(args)
        hivemind_info = setup_hivemind_peer(
            initial_peers=initial_peers,
            host=args.host,
            port=args.port,
            peer_id=args.peer_id,
            verbose=True,
        )
        print(f"[Hivemind] Peer endpoint: {hivemind_info.endpoint}")

    # ---- train -------------------------------------------------------------
    try:
        _train(args, hivemind_info=hivemind_info)
    finally:
        # Shutdown Hivemind peer gracefully
        if hivemind_info is not None:
            try:
                hivemind_info.peer.shutdown()
                print("[Hivemind] Peer shut down.")
            except Exception:
                pass


if __name__ == "__main__":
    main()
