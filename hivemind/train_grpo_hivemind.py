#!/usr/bin/env python3
"""
train_grpo_hivemind.py

GRPO RL fine-tuning with Hivemind — distributed across heterogeneous nodes
(laptops, gaming PCs, cloud instances) with async parameter averaging.

Only the **policy model's** optimizer is wrapped in DecentralizedOptimizer.
The reference model stays local and frozen — it is not averaged across peers.

This mirrors ``train_grpo.py`` in every respect except DDP is replaced by
Hivemind's async all-reduce.  Core GRPO logic (reward, rollout, loss) is
imported directly from ``train_grpo`` so fixes propagate automatically.

Usage
-----
  # Bootstrap (first node):
      python hivemind/train_grpo_hivemind.py \\
          --hivemind --initial-peers "" --port 5678 \\
          --checkpoint ./sft_ckpts/latest.pt \\
          --data-dir ./grpo_packed \\
          --tokenizer ./tokenizer \\
          --out-dir ./grpo_ckpts --lora-rank 64

  # Worker nodes:
      python hivemind/train_grpo_hivemind.py \\
          --hivemind --initial-peers "192.168.1.100:5678" \\
          --checkpoint ./sft_ckpts/latest.pt \\
          --data-dir ./grpo_packed \\
          --tokenizer ./tokenizer \\
          --out-dir ./grpo_ckpts_worker1 --lora-rank 64

  # Resume:
      python hivemind/train_grpo_hivemind.py --hivemind ... \\
          --resume ./grpo_ckpts/grpo_step0000050.pt

For the full argument list run: python hivemind/train_grpo_hivemind.py --help
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import sys
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from atomic_io import load_torch_checkpoint

# ── Core GRPO logic imported from parent ──────────────────────────────────
from train_grpo import (                          # type: ignore[import-unchecked]
    reward_fn,
    generate_rollouts,
    compute_logprobs,
    grpo_loss,
    build_reference,
    PackedGRPODataLoader,
    _build_attn_mask,
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

def save_grpo_hivemind_checkpoint(
    out_dir: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    args_dict: dict,
    is_lora: bool,
    recipe: TrainingRecipe,
) -> str:
    """Save GRPO checkpoint, unwrapping Hivemind DecentralizedOptimizer if needed."""
    raw = _raw(model)
    # Unwrap Hivemind optimizer for clean state dict
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
    path = os.path.join(out_dir, f"grpo_step{step:07d}.pt")
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

    # Save recipe alongside checkpoint
    recipe.to_json(os.path.join(out_dir, "recipe.json"))

    print(f"[Checkpoint] saved {path}")
    return path


def load_grpo_hivemind_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    is_lora: bool,
) -> int:
    """Load GRPO checkpoint, handling Hivemind optimizer wrapper.

    Returns the step number.
    """
    ckpt = load_torch_checkpoint(path, map_location=device)
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
    """Remove old GRPO checkpoints, keeping only the ``keep`` most recent."""
    ckpts = sorted(
        Path(out_dir).glob("grpo_step*.pt"),
        key=lambda p: int(p.stem.replace("grpo_step", "")),
    )
    for old in ckpts[:-keep]:
        old.unlink()


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
    rng = random.Random(peer_seed)

    peer_tag = f"[{hivemind_info.endpoint}] " if hivemind_info else ""

    # ── recipe ────────────────────────────────────────────────────────────
    recipe = recipe_from_args(args)
    print(f"{peer_tag}[Recipe] mode={recipe.mode}, model_name={recipe.model_name}")

    # ── model (from SFT checkpoint) ──────────────────────────────────────
    if not args.checkpoint:
        raise FileNotFoundError("--checkpoint is required (SFT checkpoint).")

    ckpt_data = load_torch_checkpoint(args.checkpoint, map_location="cpu")
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
    if args.compile:
        print(f"{peer_tag}[compile] torch.compile(mode='{args.compile_mode}')…")
        model = torch.compile(model, mode=args.compile_mode)

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

    train_ds = PackedGRPODataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        rank=peer_idx,
        world_size=swarm_size,
        seed=args.seed,
    )
    print(f"{peer_tag}[Dataset] {len(train_ds):,} prompts "
          f"(shard {peer_idx}/{swarm_size})")

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
            prefix="grpo_hivemind",
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
        start_step = load_grpo_hivemind_checkpoint(
            args.resume, model, optimizer, device, is_lora,
        )

    os.makedirs(args.out_dir, exist_ok=True)
    eff_prompts = args.batch_size * max(1, swarm_size)
    eff_completions = eff_prompts * args.num_generations
    print(f"{peer_tag}Effective batch: {eff_prompts} prompts "
          f"({eff_completions} completions)")
    print(f"{peer_tag}Group size G: {args.num_generations}")
    print(f"{peer_tag}Max steps: {args.max_steps:,}")
    print(f"{peer_tag}Reference policy: {args.ref_policy}")

    # ═════════════════════════════════════════════════════════════════ LOOP
    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    reward_window: List[float] = []
    step = start_step
    data_iter = iter(train_ds)

    try:
        while step < args.max_steps:
            # ── learning rate ─────────────────────────────────────────────
            lr = scheduler(step)
            inner_opt = optimizer.opt if hasattr(optimizer, "opt") else optimizer
            for pg in inner_opt.param_groups:
                pg["lr"] = lr

            # 1. sample a batch of prompts ─────────────────────────────────
            try:
                batch_prompts, batch_answers, batch_wt = next(data_iter)
            except StopIteration:
                data_iter = iter(train_ds)
                batch_prompts, batch_answers, batch_wt = next(data_iter)

            prompt_list = [p.tolist() for p in batch_prompts]

            # 2. expand to G replicas ──────────────────────────────────────
            expanded_p: List[List[int]] = []
            expanded_a: List[str] = []
            expanded_wt: List[Optional[bool]] = []
            for _g in range(args.num_generations):
                for p, a, wt in zip(prompt_list, batch_answers, batch_wt):
                    expanded_p.append(p)
                    expanded_a.append(a)
                    expanded_wt.append(wt)

            # 3. rollout ───────────────────────────────────────────────────
            rollout_model = _raw(model)
            rollout_model.eval()
            with ctx, torch.no_grad():
                full_ids, gen_mask, _sampled_lp, prompt_pad_mask = \
                    generate_rollouts(
                        rollout_model, expanded_p, recipe,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        eos_id=eos_id,
                        pad_id=pad_id,
                        rng=rng,
                    )
            rollout_model.train()

            # 4. decode + reward ───────────────────────────────────────────
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
            for completion, answer, wt in zip(
                completions_text, expanded_a, expanded_wt
            ):
                r, _info = reward_fn(
                    completion, answer, wt, recipe,
                    max_new_tokens=args.max_new_tokens,
                    correct_weight=args.reward_correct,
                    format_weight=args.reward_format,
                )
                rewards_list.append(r)
            rewards = torch.tensor(rewards_list, dtype=torch.float, device=device)

            # 5. reference log-probs (no_grad) ─────────────────────────────
            with ctx, torch.no_grad():
                ref_logp = compute_logprobs(
                    ref_for_logprob, full_ids, gen_mask, prompt_pad_mask,
                )

            # 6. policy log-probs (with grad) ──────────────────────────────
            policy_attn_mask = _build_attn_mask(
                prompt_pad_mask, full_ids.shape[1], 0,
                next(model.parameters()).dtype,
            )
            T = gen_mask.shape[1]
            with ctx:
                out = model(
                    full_ids, attention_mask=policy_attn_mask,
                    use_cache=False, num_logits_to_keep=T,
                )
                policy_logits = out["logits"].float()
                mod_aux_loss = out.get("mod_aux_loss", 0.0)
            targets = full_ids[:, -T:]
            policy_logp = policy_logits.log_softmax(dim=-1).gather(
                -1, targets.unsqueeze(-1),
            ).squeeze(-1)
            policy_logp = policy_logp * gen_mask

            # 7. loss + step ───────────────────────────────────────────────
            loss, metrics = grpo_loss(
                policy_logp, ref_logp, rewards, gen_mask,
                group_size=args.num_generations,
                kl_coef=args.kl_coef,
                clip_ratio=args.clip_ratio,
                entropy_coef=args.entropy_coeff,
            )
            # MoD auxiliary loss
            loss += mod_aux_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip,
            ).item()
            optimizer.step()

            # 8. logging ───────────────────────────────────────────────────
            reward_window.extend(rewards_list)
            window = max(1, len(reward_window))

            if step % args.log_interval == 0:
                t1 = time.perf_counter()
                sps = args.log_interval / max(t1 - t0, 1e-9)
                t0 = t1
                r_mean = sum(reward_window) / window
                reward_window.clear()
                print(
                    f"{peer_tag}step {step:6d} | loss {loss.item():+.4f} | "
                    f"pg {metrics['pg']:+.4f} | kl {metrics['kl']:+.5f} | "
                    f"r̄ {r_mean:.2f} | lr {lr:.2e} | g {grad_norm:.2f} | "
                    f"{sps:.2f} step/s"
                )

            # 9. checkpoint ────────────────────────────────────────────────
            if step > start_step and step % args.save_every == 0:
                save_grpo_hivemind_checkpoint(
                    args.out_dir, step, model, optimizer,
                    config, vars(args), is_lora, recipe,
                )
                prune_checkpoints(args.out_dir, keep=args.keep_ckpts)

            step += 1

    except KeyboardInterrupt:
        print(f"\n{peer_tag}[GRPO] Interrupted at step {step}.")

    # ── final save ────────────────────────────────────────────────────────
    save_grpo_hivemind_checkpoint(
        args.out_dir, step, model, optimizer,
        config, vars(args), is_lora, recipe,
    )
    print(f"{peer_tag}[GRPO] complete at step {step}.")

    # ── optional checkpoint averaging across swarm ────────────────────────
    if args.hivemind and args.average_checkpoints and hivemind_info is not None:
        print(f"{peer_tag}[Hivemind] Averaging final checkpoints...")
        avg_state = average_checkpoints_via_hivemind(
            model, hivemind_info.peer,
            target_group_size=min(args.target_group_size, 8),
            num_rounds=args.checkpoint_average_rounds,
        )
        avg_path = os.path.join(args.out_dir, "grpo_averaged_final.pt")
        torch.save({"model_state": avg_state, "config": vars(config)}, avg_path)
        print(f"{peer_tag}[Hivemind] Averaged checkpoint saved to {avg_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO RL fine-tuning with Hivemind decentralized training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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

    # Rollouts
    p.add_argument("--num-generations", type=int, default=8,
                   help="G — completions per prompt")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)

    # Reward weights
    p.add_argument("--reward-correct", type=float, default=1.0,
                   help="Reward for correct + well-formatted answer")
    p.add_argument("--reward-format", type=float, default=0.3,
                   help="Reward for wrong-but-well-formed answer")

    # GRPO loss
    p.add_argument("--kl-coef", type=float, default=0.02)
    p.add_argument("--clip-ratio", type=float, default=0.2)
    p.add_argument("--entropy-coeff", type=float, default=0.0,
                   help="Entropy bonus coefficient (0 = disabled)")

    # Optim
    p.add_argument("--batch-size", type=int, default=4,
                   help="Prompts per step")
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
                   help="GRPO checkpoint to resume from")
    p.add_argument("--save-every", type=int, default=50,
                   help="Checkpoint interval in steps")
    p.add_argument("--keep-ckpts", type=int, default=3)
    p.add_argument("--log-interval", type=int, default=1)

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
