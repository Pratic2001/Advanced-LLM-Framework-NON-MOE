#!/usr/bin/env python3
"""
train_dpo_hivemind.py

DPO preference fine-tuning with Hivemind — distributed across heterogeneous
nodes (laptops, gaming PCs, cloud instances) with async parameter averaging.

Only the **policy model's** optimizer is wrapped in DecentralizedOptimizer.
The reference model stays local and frozen — it is not averaged across peers.

This mirrors ``train_dpo.py`` in every respect except DDP is replaced by
Hivemind's async all-reduce.  Core DPO logic (loss, logprobs, data loading)
is imported directly from ``train_dpo.py`` so fixes propagate automatically.

Usage
-----
  # Bootstrap (first node):
      python hivemind/train_dpo_hivemind.py \\
          --hivemind --initial-peers "" --port 5678 \\
          --checkpoint ./sft_ckpts/latest.pt \\
          --data-dir ./dpo_packed \\
          --tokenizer ./tokenizer \\
          --out-dir ./dpo_ckpts --lora-rank 64

  # Worker nodes:
      python hivemind/train_dpo_hivemind.py \\
          --hivemind --initial-peers "192.168.1.100:5678" \\
          --checkpoint ./sft_ckpts/latest.pt \\
          --data-dir ./dpo_packed \\
          --tokenizer ./tokenizer \\
          --out-dir ./dpo_ckpts_worker1 --lora-rank 64

  # Resume:
      python hivemind/train_dpo_hivemind.py --hivemind ... \\
          --resume ./dpo_ckpts/dpo_step0000050.pt

For the full argument list run: python hivemind/train_dpo_hivemind.py --help

Reference:
    "Direct Preference Optimization" (Rafailov et al., 2023)
    https://arxiv.org/abs/2305.18290
"""

from __future__ import annotations

import argparse
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

# ── Core DPO logic imported from parent ──────────────────────────────────
from train_dpo import (                          # type: ignore[import-unchecked]
    dpo_loss,
    compute_sequence_logprobs,
    build_reference,
    PackedDPODataLoader,
    _raw,
    load_tokenizer,
    get_special_token_id,
    GPU_PEAK_TFLOPS,
    _detect_gpu_peak_tflops,
)

from model import ModelConfig, TransformerForCausalLM, add_architecture_args, apply_architecture_args, count_parameters
from recipe import TrainingRecipe, add_recipe_args, recipe_from_args

# ── Hivemind ──────────────────────────────────────────────────────────────
from hivemind.hivemind_utils import (
    setup_hivemind_peer,
    build_hivemind_optimizer,
    average_checkpoints_via_hivemind,
    get_initial_peers_from_args,
    add_hivemind_args,
    check_hivemind_args,
    _HIVEMIND_AVAILABLE,
)


# ---------------------------------------------------------------------------
# LoRA/DoRA wrapper (mirrors train_sft_hivemind.py)
# ---------------------------------------------------------------------------

def _maybe_wrap_lora(
    model: nn.Module,
    lora_rank: int = 0,
    lora_alpha: float = 128.0,
) -> nn.Module:
    """Apply LoRA adapters if lora_rank > 0."""
    if lora_rank <= 0:
        return model
    try:
        from peft.lora import inject_lora, freeze_base
    except ImportError:
        print("[LoRA] Could not import peft.lora. Make sure it's in your PYTHONPATH.")
        raise

    n_replaced = inject_lora(model, rank=lora_rank, alpha=lora_alpha)
    freeze_base(model)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[LoRA] rank={lora_rank} alpha={lora_alpha} | "
          f"{n_replaced} adapters injected | "
          f"trainable={n_trainable:,}")
    return model


# ---------------------------------------------------------------------------
# Checkpoint helpers (handle Hivemind optimizer unwrapping)
# ---------------------------------------------------------------------------

def save_dpo_hivemind_checkpoint(
    out_dir: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    args_dict: dict,
    is_lora: bool,
    recipe: TrainingRecipe,
) -> str:
    """Save DPO checkpoint, unwrapping Hivemind DecentralizedOptimizer if needed."""
    raw = _raw(model)
    inner_opt = optimizer.opt if hasattr(optimizer, "opt") else optimizer

    ckpt = {
        "step": step,
        "model_state": raw.state_dict(),
        "optimizer_state": inner_opt.state_dict(),
        "config": vars(config),
        "args": args_dict,
        "is_lora": is_lora,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"dpo_step{step:07d}.pt")
    torch.save(ckpt, path)

    latest = os.path.join(out_dir, "latest.pt")
    if os.path.islink(latest) or os.path.exists(latest):
        if os.path.islink(latest):
            os.remove(latest)
        else:
            os.remove(latest)
    try:
        os.symlink(os.path.abspath(path), latest)
    except OSError:
        pass

    recipe.to_json(os.path.join(out_dir, "recipe.json"))

    print(f"[Checkpoint] saved {path}")
    return path


def load_dpo_hivemind_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    is_lora: bool,
) -> int:
    """Load DPO checkpoint, handling Hivemind optimizer wrapper.

    Returns the step number.
    """
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

    if optimizer is not None and "optimizer_state" in ckpt:
        inner_opt = optimizer.opt if hasattr(optimizer, "opt") else optimizer
        try:
            inner_opt.load_state_dict(ckpt["optimizer_state"])
        except Exception as e:
            print(f"[Checkpoint] optimizer state load skipped: {e}")

    step = ckpt.get("step", 0)
    return step


def prune_checkpoints(out_dir: str, keep: int = 3) -> None:
    """Remove old DPO checkpoints, keeping only the ``keep`` most recent."""
    ckpts = sorted(
        Path(out_dir).glob("dpo_step*.pt"),
        key=lambda p: int(p.stem.replace("dpo_step", "")),
    )
    for old in ckpts[:-keep]:
        old.unlink()


# ---------------------------------------------------------------------------
# Validation (local, no DDP)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: TransformerForCausalLM,
    val_dataset: PackedDPODataLoader,
    ref_model: Optional[TransformerForCausalLM],
    beta: float,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Run validation over a subset of preference pairs.

    Returns metrics dict with accuracy, reward margin, etc.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    metrics_accum: Dict[str, List[float]] = {
        "accuracy": [],
        "reward_margin": [],
        "chosen_reward": [],
        "rejected_reward": [],
    }
    n_batches = max(1, min(10, len(val_dataset) // max(1, val_dataset.batch_size)))

    for bidx, (chosen_ids, chosen_mask, rejected_ids, rejected_mask) in enumerate(val_dataset):
        if bidx >= n_batches:
            break

        chosen_ids = chosen_ids.to(device)
        chosen_mask = chosen_mask.to(device)
        rejected_ids = rejected_ids.to(device)
        rejected_mask = rejected_mask.to(device)

        # Policy logprobs
        policy_c_logp, _ = compute_sequence_logprobs(model, chosen_ids)
        policy_r_logp, _ = compute_sequence_logprobs(model, rejected_ids)

        # Reference logprobs
        ref_for_logprob = ref_model if ref_model is not None else model
        ref_c_logp, _ = compute_sequence_logprobs(ref_for_logprob, chosen_ids)
        ref_r_logp, _ = compute_sequence_logprobs(ref_for_logprob, rejected_ids)

        _, metrics = dpo_loss(
            policy_c_logp, policy_r_logp,
            ref_c_logp, ref_r_logp,
            chosen_mask, rejected_mask,
            beta=beta,
        )
        for k, v in metrics.items():
            if k in metrics_accum:
                metrics_accum[k].append(v)

    model.train()
    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics_accum.items()}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train(
    args: argparse.Namespace,
    hivemind_info: Any = None,
) -> None:
    # ── device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if hivemind_info is not None:
        peer_seed = args.seed + hash(hivemind_info.endpoint) % 65536
    else:
        peer_seed = args.seed

    torch.manual_seed(peer_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    peer_tag = f"[{hivemind_info.endpoint}] " if hivemind_info else ""

    # ── recipe ────────────────────────────────────────────────────────────
    recipe = recipe_from_args(args)
    print(f"{peer_tag}[Recipe] mode={recipe.mode}, model_name={recipe.model_name}")

    # ── model (from SFT checkpoint) ──────────────────────────────────────
    if not args.checkpoint:
        raise FileNotFoundError("--checkpoint is required (SFT checkpoint).")

    ckpt_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**{k: v for k, v in ckpt_data["config"].items()
                            if k in ModelConfig.__init__.__code__.co_varnames})
    apply_architecture_args(config, args)

    model = TransformerForCausalLM(config).to(device)
    model.load_state_dict(ckpt_data["model_state"])
    model.tie_weights()

    n_total = count_parameters(model)
    print(f"{peer_tag}Loaded SFT checkpoint: {n_total:,} params ({n_total/1e9:.3f}B)")

    # ── LoRA ──────────────────────────────────────────────────────────────
    is_lora = args.lora
    if is_lora:
        _maybe_wrap_lora(model, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)
    else:
        print(f"{peer_tag}[LoRA] disabled — full fine-tune")

    # ── compile ───────────────────────────────────────────────────────────
    _use_cudagraphs = False
    if args.compile:
        print(f"{peer_tag}[compile] torch.compile(mode='{args.compile_mode}')…")
        model = torch.compile(model, mode=args.compile_mode)
        _use_cudagraphs = (args.compile_mode == "reduce-overhead")

    # ── reference model (local, NOT distributed) ──────────────────────────
    ref_model = build_reference(args.ref_policy, config, args.checkpoint, device)
    ref_for_logprob = ref_model if ref_model is not None else model

    # ── tokenizer ─────────────────────────────────────────────────────────
    tokenizer = load_tokenizer(args.tokenizer)
    eos_id = get_special_token_id(tokenizer, recipe.eos_token)
    pad_id = get_special_token_id(tokenizer, recipe.pad_token)
    print(f"{peer_tag}[Tokenizer] eos_id={eos_id}, pad_id={pad_id}")

    # ── data sharding ─────────────────────────────────────────────────────
    if hivemind_info is not None:
        try:
            visible = len(hivemind_info.peer.get_visible_peers())
            swarm_size = max(visible, args.target_group_size)
        except Exception:
            swarm_size = args.target_group_size
        peer_idx = hash(hivemind_info.endpoint) % swarm_size
    else:
        swarm_size = 1
        peer_idx = 0

    # Training data
    train_ds = PackedDPODataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        rank=peer_idx,
        world_size=swarm_size,
        seed=args.seed,
        split="train",
    )
    print(f"{peer_tag}[Dataset] {len(train_ds):,} preference triples "
          f"(shard {peer_idx}/{swarm_size})")

    # Validation data
    val_ds = None
    if args.val_every > 0:
        try:
            val_ds = PackedDPODataLoader(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                rank=peer_idx,
                world_size=swarm_size,
                seed=args.seed,
                split="val",
            )
        except Exception:
            pass

    # ── auto LR scaling ───────────────────────────────────────────────────
    if not args.no_lr_scale:
        ref_hidden = 2048
        scale = math.sqrt(ref_hidden / config.hidden_size)
        scale = max(0.5, min(scale, 2.0))
        original_lr = args.lr
        args.lr = args.lr * scale
        args.min_lr = args.min_lr * scale
        print(f"{peer_tag}[LR] Auto-scaled from {original_lr:.2e} to "
              f"{args.lr:.2e} (x{scale:.3f}, hidden={config.hidden_size})")

    # ── optimizer ─────────────────────────────────────────────────────────
    from optim.build_optimizer import build_optimizer as _build_optimizer
    local_opt = _build_optimizer(
        model,
        optimizer_type="adamw",
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.hivemind and hivemind_info is not None:
        optimizer = build_hivemind_optimizer(
            model=model,
            base_optimizer=local_opt,
            peer=hivemind_info.peer,
            target_group_size=args.target_group_size,
            averaging_period=args.averaging_period,
            average_parameters=args.average_parameters,
            prefix="dpo_hivemind",
            verbose=True,
        )
    else:
        optimizer = local_opt

    # ── LR schedule ───────────────────────────────────────────────────────
    from optim.lr_schedule import build_scheduler
    scheduler = build_scheduler(
        schedule="cosine",
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        peak_lr=args.lr,
        min_lr=args.min_lr,
    )

    # ── AMP ───────────────────────────────────────────────────────────────
    use_amp = device.type == "cuda" and args.dtype == "bf16"
    ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_amp else nullcontext())

    # ── resume ────────────────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        start_step = load_dpo_hivemind_checkpoint(
            args.resume, model, optimizer, device, is_lora,
        )

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{peer_tag}Effective batch: {args.batch_size * max(1, swarm_size)} triples")
    print(f"{peer_tag}Max steps: {args.max_steps:,}")
    print(f"{peer_tag}DPO beta: {args.beta}")
    print(f"{peer_tag}Reference policy: {args.ref_policy}")

    # ═════════════════════════════════════════════════════════════════ LOOP
    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    loss_window: List[float] = []
    acc_window: List[float] = []

    step = start_step
    data_iter = iter(train_ds)

    try:
        while step < args.max_steps:
            # ── learning rate ─────────────────────────────────────────────
            lr = scheduler(step)
            inner_opt = optimizer.opt if hasattr(optimizer, "opt") else optimizer
            for pg in inner_opt.param_groups:
                pg["lr"] = lr

            # 1. sample a batch of preference triples ──────────────────────
            try:
                chosen_ids, chosen_mask, rejected_ids, rejected_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(train_ds)
                chosen_ids, chosen_mask, rejected_ids, rejected_mask = next(data_iter)

            chosen_ids = chosen_ids.to(device)
            chosen_mask = chosen_mask.to(device)
            rejected_ids = rejected_ids.to(device)
            rejected_mask = rejected_mask.to(device)

            # 2. reference log-probs (no_grad) ─────────────────────────────
            with torch.no_grad():
                ref_c_logp, _ = compute_sequence_logprobs(ref_for_logprob, chosen_ids)
                ref_r_logp, _ = compute_sequence_logprobs(ref_for_logprob, rejected_ids)

            # 3. policy log-probs (with grad) ──────────────────────────────
            with ctx:
                if _use_cudagraphs:
                    torch.compiler.cudagraph_mark_step_begin()
                policy_c_logp, c_mod_aux = compute_sequence_logprobs(model, chosen_ids)
                policy_r_logp, r_mod_aux = compute_sequence_logprobs(model, rejected_ids)
                mod_aux_loss = c_mod_aux + r_mod_aux

            # 4. DPO loss ─────────────────────────────────────────────────
            loss, metrics = dpo_loss(
                policy_c_logp, policy_r_logp,
                ref_c_logp, ref_r_logp,
                chosen_mask, rejected_mask,
                beta=args.beta,
                label_smoothing=args.label_smoothing,
                clip_ratio=args.clip_ratio if args.clip_ratio > 0 else None,
            )
            # MoD auxiliary loss
            loss += mod_aux_loss

            # 5. backward + step ──────────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip,
            ).item()
            optimizer.step()

            # 6. logging ──────────────────────────────────────────────────
            loss_window.append(float(loss.detach().item()))
            acc_window.append(metrics["accuracy"])
            window = max(1, len(loss_window))

            if step % args.log_interval == 0:
                t1 = time.perf_counter()
                sps = args.log_interval / max(t1 - t0, 1e-9)
                t0 = t1
                avg_loss = sum(loss_window) / window
                avg_acc = sum(acc_window) / window
                loss_window.clear()
                acc_window.clear()
                print(
                    f"{peer_tag}step {step:6d} | loss {avg_loss:+.4f} | "
                    f"acc {avg_acc:.2%} | "
                    f"r_margin {metrics['reward_margin']:.4f} | "
                    f"lr {lr:.2e} | g {grad_norm:.2f} | "
                    f"{sps:.2f} step/s"
                )

            # 7. validation ────────────────────────────────────────────────
            if val_ds is not None and step > start_step and step % args.val_every == 0:
                val_metrics = validate(
                    _raw(model), val_ds, ref_model, args.beta, device,
                )
                print(f"{peer_tag}  [val] acc {val_metrics['accuracy']:.2%} | "
                      f"r_margin {val_metrics['reward_margin']:.4f}")

            # 8. checkpoint ────────────────────────────────────────────────
            if step > start_step and step % args.save_every == 0:
                save_dpo_hivemind_checkpoint(
                    args.out_dir, step, model, optimizer,
                    config, vars(args), is_lora, recipe,
                )
                prune_checkpoints(args.out_dir, keep=args.keep_ckpts)

            step += 1

    except KeyboardInterrupt:
        print(f"\n{peer_tag}[DPO] Interrupted at step {step}.")

    # ── final save ────────────────────────────────────────────────────────
    save_dpo_hivemind_checkpoint(
        args.out_dir, step, model, optimizer,
        config, vars(args), is_lora, recipe,
    )
    print(f"{peer_tag}[DPO] complete at step {step}.")

    # ── optional checkpoint averaging across swarm ────────────────────────
    if args.hivemind and args.average_checkpoints and hivemind_info is not None:
        print(f"{peer_tag}[Hivemind] Averaging final checkpoints...")
        avg_state = average_checkpoints_via_hivemind(
            model, hivemind_info.peer,
            target_group_size=min(args.target_group_size, 8),
            num_rounds=args.checkpoint_average_rounds,
        )
        avg_path = os.path.join(args.out_dir, "dpo_averaged_final.pt")
        torch.save({"model_state": avg_state, "config": vars(config)}, avg_path)
        print(f"{peer_tag}[Hivemind] Averaged checkpoint saved to {avg_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DPO preference fine-tuning with Hivemind decentralized training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    p.add_argument("--checkpoint", default=None,
                   help="SFT checkpoint from train_sft.py")
    p.add_argument("--tokenizer", default="./tokenizer",
                   help="Tokenizer directory from train_tokenizer.py")
    p.add_argument("--data-dir", default="./dpo_packed",
                   help="Packed DPO data directory from data/pack_dpo.py")
    p.add_argument("--out-dir", default="./dpo_checkpoints",
                   help="Output directory for checkpoints")

    # LoRA
    p.add_argument("--lora", action="store_true",
                   help="Enable LoRA adapters")
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=float, default=128.0)

    add_architecture_args(p)

    # Reference policy
    p.add_argument("--ref-policy", default="single", choices=["single", "two"],
                   help="'single' reuses trainable model under no_grad; "
                        "'two' keeps a frozen copy in memory.")

    # DPO loss parameters
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO temperature parameter (default: 0.1)")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="DPO label smoothing epsilon (default: 0.0)")
    p.add_argument("--clip-ratio", type=float, default=0.0,
                   help="Optional PPO-style clipping (0 = disabled)")

    # Optim
    p.add_argument("--batch-size", type=int, default=4,
                   help="Preference triples per step")
    p.add_argument("--num-steps", type=int, default=500,
                   help="Total training steps")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Alias for --num-steps")
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
    p.add_argument("--resume", default=None,
                   help="DPO checkpoint to resume from")
    p.add_argument("--save-every", type=int, default=50,
                   help="Checkpoint interval in steps")
    p.add_argument("--keep-ckpts", type=int, default=3)
    p.add_argument("--log-interval", type=int, default=1)

    # Validation
    p.add_argument("--val-every", type=int, default=500,
                   help="Validation interval (0 = disabled)")

    # Recipe
    add_recipe_args(p)

    # Hivemind
    add_hivemind_args(p)
    p.add_argument("--average-checkpoints", action="store_true", default=False,
                   help="After training, average parameters across the swarm.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Backwards compat: --max-steps overrides --num-steps
    if args.max_steps is not None:
        args.num_steps = args.max_steps
    else:
        args.max_steps = args.num_steps

    hivemind_info = None
    if args.hivemind:
        check_hivemind_args(args)
        initial = get_initial_peers_from_args(args)
        hivemind_info = setup_hivemind_peer(
            initial, host=args.host, port=args.port,
        )
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
