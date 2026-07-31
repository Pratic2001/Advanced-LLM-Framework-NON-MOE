#!/usr/bin/env python3
"""
train_dpo_deepspeed.py

DeepSpeed-powered DPO (Direct Preference Optimization) for the dense
transformer model from `model.py`. Mirror of `train_dpo.py` that replaces
DDP + AdamW with the DeepSpeed engine so the same training loop scales
across multi-GPU / multi-node hardware with auto-selected ZeRO stages and
optional CPU offload.

The training algorithm is identical to `train_dpo.py`:

    1. Sample a batch of (prompt, chosen, rejected) triples.
    2. Compute reference log-probs for chosen and rejected completions
       (no_grad).
    3. Compute policy log-probs for chosen and rejected completions (with
       grad).
    4. Apply the DPO loss:
           L = -log σ(β * (logπ(chosen) - logπ_ref(chosen)
                          - (logπ(rejected) - logπ_ref(rejected))))
    5. Backward + step.

This script reuses the DPO primitives (dataset, logprob computation, DPO loss)
verbatim from `train_dpo.py` and only swaps the distribution/optimizer layer
for DeepSpeed.

Launch:
    # Single node, 1 GPU
    deepspeed --num_gpus 1 train_dpo_deepspeed.py \\
        --checkpoint ./sft_checkpoints/latest.pt \\
        --data-dir ./dpo_packed --tokenizer ./tokenizer

    # Single node, 4 GPUs
    deepspeed --num_gpus 4 train_dpo_deepspeed.py \\
        --checkpoint ./sft.pt --data-dir ./dpo_packed

    # Multi-node
    deepspeed --hostfile hostfile train_dpo_deepspeed.py \\
        --checkpoint ./sft.pt --data-dir ./dpo_packed

    # Force a specific ZeRO stage
    deepspeed train_dpo_deepspeed.py \\
        --checkpoint ./sft.pt --data-dir ./dpo_packed \\
        --zero-stage 3 --cpu-offload-optimizer

    # LoRA DPO
    deepspeed train_dpo_deepspeed.py \\
        --checkpoint ./sft.pt --data-dir ./dpo_packed \\
        --lora --lora-rank 64 --lora-alpha 128

    # Smoke test (uses DDP variant's smoke_test)
    python train_dpo_deepspeed.py --smoke-test

Reference:
    "Direct Preference Optimization" (Rafailov et al., 2023)
    https://arxiv.org/abs/2305.18290
"""

import argparse
import atexit
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import ModelConfig, TransformerForCausalLM, add_architecture_args, apply_architecture_args, count_parameters

from atomic_io import atomic_symlink, atomic_torch_save, atomic_write_json, load_torch_checkpoint
from shutdown import install_signal_handlers, should_stop
from logging_utils import setup_logging, get_logger, log_event

# Re-use the DPO machinery so this script stays focused on the engine swap.
from train_dpo import (
    PackedDPODataLoader,
    dpo_loss,
    compute_sequence_logprobs,
    build_reference,
    load_dpo_checkpoint,
    smoke_test,
)

# Re-use the LoRA machinery from peft.
from peft.lora import inject_lora, freeze_base, lora_state_dict

# Re-use recipe for all template decisions.
from recipe import (
    TrainingRecipe,
    add_recipe_args,
    recipe_from_args,
)

# ---------------------------------------------------------------------------
# GPU FLOP/s table (for TFLOPS resolution)
# ---------------------------------------------------------------------------

GPU_PEAK_TFLOPS = {
    # NVIDIA consumer
    "NVIDIA GeForce RTX 4090":      165.2,
    "NVIDIA GeForce RTX 4080 SUPER": 105.0,
    "NVIDIA GeForce RTX 4080":       97.5,
    "NVIDIA GeForce RTX 4070 Ti SUPER": 88.0,
    "NVIDIA GeForce RTX 4070 Ti":    80.8,
    "NVIDIA GeForce RTX 4070 SUPER": 70.9,
    "NVIDIA GeForce RTX 4070":       59.8,
    "NVIDIA GeForce RTX 4060 Ti":    44.0,
    "NVIDIA GeForce RTX 4060":       30.0,
    "NVIDIA GeForce RTX 3090 Ti":    80.0,
    "NVIDIA GeForce RTX 3090":       71.0,
    "NVIDIA GeForce RTX 3080 Ti":    81.1,
    "NVIDIA GeForce RTX 3080":       59.4,
    "NVIDIA GeForce RTX 3070 Ti":    43.1,
    "NVIDIA GeForce RTX 3070":       40.4,
    "NVIDIA GeForce RTX 3060 Ti":    32.0,
    "NVIDIA GeForce RTX 3060":       25.0,
    # NVIDIA data-centre
    "NVIDIA A100-SXM4-80GB":        312.0,
    "NVIDIA A100-SXM4-40GB":        312.0,
    "NVIDIA A100-PCIE-80GB":        312.0,
    "NVIDIA A100-PCIE-40GB":        312.0,
    "NVIDIA H100 SXM5":             989.5,
    "NVIDIA H100 PCIe":             756.0,
    "NVIDIA H200":                  989.5,
    "NVIDIA L40S":                  362.1,
    "NVIDIA L40":                   181.0,
    "NVIDIA L4":                    121.0,
    "NVIDIA A10G":                  125.0,
    "NVIDIA A10":                   125.0,
    "NVIDIA A30":                   165.0,
    "NVIDIA A40":                   149.7,
    "NVIDIA V100-SXM2-32GB":         28.0,
    "NVIDIA V100-SXM2-16GB":         28.0,
    "NVIDIA V100-PCIE-16GB":         14.0,
    # AMD Instinct
    "AMD Instinct MI300X":          1307.4,
    "AMD Instinct MI300A":           383.0,
    "AMD Instinct MI250X":           383.0,
    "AMD Instinct MI210":            181.0,
    "AMD Instinct MI100":             46.1,
}


# ---------------------------------------------------------------------------
# Hardware audit
# ---------------------------------------------------------------------------


def _run(cmd: str) -> str:
    """Run a shell command, return stdout or '' on error."""
    try:
        return subprocess.check_output(
            cmd, shell=True, stderr=subprocess.DEVNULL, timeout=10
        ).decode().strip()
    except Exception:
        return ""


def _tflops_from_cuda_props(props) -> Tuple[float, str]:
    """Estimate bf16 TFLOPS from CUDA device properties."""
    cc_major = props.major
    cc_minor = props.minor
    n_sm = props.multi_processor_count
    clock_hz = props.clock_rate * 1000

    cores_per_sm = {
        (9, 0): 128, (8, 9): 128, (8, 6): 128, (8, 0): 64,
        (7, 5): 64,  (7, 0): 64,  (6, 1): 128, (6, 0): 64,
    }.get((cc_major, cc_minor), 128 if cc_major >= 8 else 64)

    fp32_tflops = (2 * n_sm * cores_per_sm * clock_hz) / 1e12
    if cc_major >= 9:
        bf16_mult = 2.0
    elif cc_major == 8:
        bf16_mult = 2.0
    elif cc_major == 7:
        bf16_mult = 8.0
    else:
        bf16_mult = 1.0

    est = fp32_tflops * bf16_mult
    method = (f"estimated from {n_sm} SMs x {cores_per_sm} cores/SM "
              f"@ {clock_hz/1e9:.2f} GHz x {bf16_mult}x TC")
    return round(est, 1), method


def resolve_gpu_peak_tflops(name: str, gpu_index: int, props) -> Tuple[float, str]:
    """Four-tier TFLOPS resolution for a GPU."""
    name_lo = name.lower()
    for key, val in GPU_PEAK_TFLOPS.items():
        if key.lower() == name_lo:
            return val, "spec-sheet (exact match)"

    def _tokens(s):
        return set(re.split(r'[\s\-_]+', s.lower()))

    name_tokens = _tokens(name)
    best_score, best_key, best_val = 0, None, None
    for key, val in GPU_PEAK_TFLOPS.items():
        key_tokens = _tokens(key)
        score = len(key_tokens & name_tokens)
        if score >= 3 and score > best_score:
            best_score, best_key, best_val = score, key, val

    if best_key is not None:
        return best_val, f"spec-sheet (token match on '{best_key}', {best_score} tokens)"

    est, method = _tflops_from_cuda_props(props)
    return est, f"computed ({method})"


def audit_hardware() -> dict:
    """Collect per-GPU and CPU information for the current node."""
    info: dict = {
        "node": platform.node(),
        "gpus": [],
        "cpu": {},
        "interconnect": {},
    }

    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        name = props.name
        vram_gb = props.total_memory / 1024 ** 3
        cc_major = props.major
        cc_minor = props.minor

        peak_tflops, tflops_source = resolve_gpu_peak_tflops(name, i, props)

        nvlink_str = _run(
            f"nvidia-smi nvlink -s -i {i} 2>/dev/null | grep 'Speed' | head -1"
        )
        has_nvlink = bool(nvlink_str)

        info["gpus"].append({
            "index":         i,
            "name":          name,
            "vram_gb":       round(vram_gb, 2),
            "cc":            f"{cc_major}.{cc_minor}",
            "bf16":          cc_major >= 8,
            "peak_tflops":   peak_tflops,
            "tflops_source": tflops_source,
            "has_nvlink":    has_nvlink,
        })

    try:
        import psutil
        cpu_ram_gb = psutil.virtual_memory().total / 1024 ** 3
        cpu_cores = psutil.cpu_count(logical=False) or 1
    except ImportError:
        cpu_ram_gb = 0.0
        cpu_cores = os.cpu_count() or 1

    info["cpu"] = {
        "ram_gb": round(cpu_ram_gb, 1),
        "cores":  cpu_cores,
        "model":  platform.processor(),
    }

    ib_str = _run("ibstat 2>/dev/null | grep 'State: Active' | wc -l")
    try:
        info["interconnect"]["infiniband_ports"] = int(ib_str)
    except ValueError:
        info["interconnect"]["infiniband_ports"] = 0

    info["interconnect"]["nvlink"] = any(g["has_nvlink"] for g in info["gpus"])

    return info


def print_audit(info: dict, n_params: int):
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  HARDWARE AUDIT  --  node: {info['node']}")
    print(sep)
    print(f"  GPUs: {len(info['gpus'])}")
    for g in info["gpus"]:
        bf16_tag = "bf16 check" if g["bf16"] else "fp16-only"
        nvlink = " NVLink check" if g["has_nvlink"] else ""
        src = g.get("tflops_source", "unknown")
        acc_tag = "" if "spec-sheet" in src else f"  (TFLOPS estimated: {src})"
        print(f"    [{g['index']}] {g['name']}  "
              f"{g['vram_gb']:.1f} GB VRAM  "
              f"{g['peak_tflops']:.0f} TFLOP/s  "
              f"CC{g['cc']}  {bf16_tag}{nvlink}{acc_tag}")
    cpu = info["cpu"]
    print(f"  CPU: {cpu.get('model','?')[:50]}  "
          f"{cpu['cores']} cores  {cpu['ram_gb']:.0f} GB RAM")
    ib = info["interconnect"]["infiniband_ports"]
    nv = "NVLink check" if info["interconnect"]["nvlink"] else "PCIe"
    print(f"  Interconnect: {nv}  "
          f"{'InfiniBand (' + str(ib) + ' ports)' if ib else 'Ethernet'}")

    model_bf16_gb = n_params * 2 / 1024 ** 3
    total_vram_gb = sum(g["vram_gb"] for g in info["gpus"])
    print(f"\n  Model:       {n_params/1e9:.3f}B params  "
          f"({model_bf16_gb:.1f} GB bf16 weights)")
    print(f"  Total VRAM:  {total_vram_gb:.1f} GB across {len(info['gpus'])} GPU(s)")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# ZeRO stage + offload auto-selection
# ---------------------------------------------------------------------------

def select_zero_stage_and_offload(
    info: dict,
    n_params: int,
    world_size: int,
    force_stage: Optional[int],
    force_cpu_offload_optimizer: bool,
    force_cpu_offload_param: bool,
) -> Tuple[int, bool, bool]:
    """
    Return (zero_stage, cpu_offload_optimizer, cpu_offload_param).

    DPO memory profiles:
      - LoRA: only adapter parameters + their optimizer states are trainable;
              base weights are frozen and live in bf16
      - full: every parameter is trainable

    Budget is sized off n_params because base weights must be resident
    regardless of mode. Adam state is sized off n_params (full) or
    n_lora_params (LoRA) — but we use n_params as a conservative estimate.
    """
    if not info["gpus"]:
        return 1, False, False

    min_vram = min(g["vram_gb"] for g in info["gpus"]) * 0.85
    n_gpus = len(info["gpus"])

    full_gb  = n_params * (2 + 2 + 8) / 1024 ** 3 / max(n_gpus, 1)
    zero2_gb = n_params * (2 + 2)     / 1024 ** 3 / max(n_gpus, 1)
    zero3_gb = n_params * 2           / 1024 ** 3 / max(n_gpus, 1)

    if force_stage is not None:
        stage = force_stage
    elif min_vram >= full_gb:
        stage = 1
    elif min_vram >= zero2_gb:
        stage = 2
    else:
        stage = 3

    cpu_offload_opt   = force_cpu_offload_optimizer
    cpu_offload_param = force_cpu_offload_param

    if not cpu_offload_opt and not cpu_offload_param:
        if stage == 3 and min_vram < zero3_gb:
            cpu_offload_opt = True
            print(f"[AutoConfig] ZeRO-3 params still exceed VRAM "
                  f"({zero3_gb:.1f} GB needed, {min_vram:.1f} GB available). "
                  f"Enabling CPU optimizer offload.")
        if stage == 3 and min_vram < (zero3_gb * 0.6):
            cpu_offload_param = True
            print(f"[AutoConfig] VRAM very tight -- also enabling CPU parameter offload.")

    cpu_ram = info["cpu"].get("ram_gb", 0)
    if (cpu_offload_opt or cpu_offload_param) and cpu_ram > 0:
        needed_gb = n_params * 8 / 1024 ** 3
        if needed_gb > cpu_ram * 0.6:
            print(f"[AutoConfig] WARNING: optimizer offload needs ~{needed_gb:.1f} GB "
                  f"CPU RAM but only {cpu_ram:.0f} GB available.")

    return stage, cpu_offload_opt, cpu_offload_param


# ---------------------------------------------------------------------------
# DeepSpeed config builder (DPO-specific)
# ---------------------------------------------------------------------------

def build_ds_config(
    args,
    zero_stage: int,
    cpu_offload_optimizer: bool,
    cpu_offload_param: bool,
    gpu_info: List[dict],
    is_lora: bool,
) -> dict:
    """
    Construct a deepspeed config dict for DPO.

    gradient_accumulation_steps=1 because the DPO training loop drives one
    step per batch.
    """

    optimizer_cfg = {
        "type": "AdamW",
        "params": {
            "lr":           args.lr,
            "betas":        [0.9, 0.95],
            "eps":          1e-8,
            "weight_decay": args.weight_decay,
        },
    }

    # We disable DeepSpeed's native bf16 path and use torch.autocast(bf16)
    # via the forward wrapper. This avoids the degenerate-init issue where
    # DS bf16 master-weights produce a model stuck at uniform-loss on small shapes.
    bf16_cfg = {"enabled": False}
    fp16_cfg = {"enabled": False}
    use_pytorch_bf16 = (args.dtype == "bf16")
    torch_autocast_cfg = (
        {"enabled": True, "dtype": "bfloat16"} if use_pytorch_bf16
        else {"enabled": False}
    )

    zero_cfg: dict = {
        "stage":                       zero_stage,
        "reduce_bucket_size":          5e8,
        "allgather_bucket_size":       5e8,
        "overlap_comm":                True,
        "contiguous_gradients":        True,
        "sub_group_size":              1e9,
        "stage3_max_live_parameters":  1e9,
        "stage3_max_reuse_distance":   1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    }

    if cpu_offload_optimizer:
        zero_cfg["offload_optimizer"] = {
            "device":     "cpu",
            "pin_memory": True,
            "ratio":      1.0,
        }

    if cpu_offload_param and zero_stage == 3:
        zero_cfg["offload_param"] = {
            "device":       "cpu",
            "pin_memory":   True,
            "buffer_count": 5,
            "buffer_size":  1e8,
        }

    has_nvlink = any(g.get("has_nvlink") for g in gpu_info)
    zero_cfg["reduce_scatter"]      = True
    zero_cfg["allgather_partitions"] = True
    if not has_nvlink:
        zero_cfg["reduce_bucket_size"]    = 2e8
        zero_cfg["allgather_bucket_size"] = 2e8

    cfg = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps":    1,
        "gradient_clipping":              args.grad_clip,
        "steps_per_print":                args.log_interval,
        "wall_clock_breakdown":           False,
        "optimizer":                      optimizer_cfg,
        "bf16":                           bf16_cfg,
        "fp16":                           fp16_cfg,
        "zero_optimization":              zero_cfg,
    }

    if use_pytorch_bf16:
        cfg["torch_autocast"] = torch_autocast_cfg

    return cfg


def print_ds_config_summary(cfg: dict, zero_stage: int,
                            cpu_opt: bool, cpu_param: bool):
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  DEEPSPEED CONFIG SUMMARY (DPO)")
    print(sep)
    print(f"  ZeRO Stage            : {zero_stage}")
    print(f"  CPU offload optimizer : {cpu_opt}")
    print(f"  CPU offload params    : {cpu_param}")
    print(f"  BF16 autocast         : {cfg.get('torch_autocast', {}).get('enabled', False)}")
    print(f"  Micro batch / GPU     : {cfg['train_micro_batch_size_per_gpu']}")
    print(f"  Grad clip             : {cfg['gradient_clipping']}")
    z = cfg["zero_optimization"]
    print(f"  Reduce bucket         : {z['reduce_bucket_size']/1e6:.0f} MB")
    print(f"  Overlap comm          : {z['overlap_comm']}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Checkpoint helpers (DeepSpeed-native)
# ---------------------------------------------------------------------------


def save_ds_dpo_checkpoint(
    engine,
    out_dir: str,
    step: int,
    config: ModelConfig,
    args_dict: dict,
    recipe: TrainingRecipe,
    is_lora: bool,
):
    """
    Save a DeepSpeed DPO checkpoint (full model) at `step`.

    For full-FT: writes a DeepSpeed directory with ZeRO-sharded weights.
    For LoRA: writes a .pt checkpoint via train_dpo.save_dpo_checkpoint logic.
    """
    if is_lora:
        # LoRA: save as .pt (small, compatible with merge_lora path). This is
        # a plain file write — NOT a collective — so only rank 0 writes it.
        if getattr(engine, "global_rank", 0) != 0:
            return None
        tag = f"dpo_step{step:07d}.pt"
        path = os.path.join(out_dir, tag)

        # Get the underlying model state dict (only trainable params)
        raw = engine.module
        inner = raw._orig_mod if hasattr(raw, "_orig_mod") else raw
        lora_sd = lora_state_dict(inner)

        ckpt = {
            "step": step,
            "model_state": lora_sd,
            "config": vars(config),
            "args": {
                **args_dict,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_type": args.lora_type,
                "lora_target_modules": args.lora_target_modules,
            },
            "is_lora": True,
        }
        os.makedirs(out_dir, exist_ok=True)
        atomic_torch_save(ckpt, path)

        latest = os.path.join(out_dir, "latest.pt")
        atomic_symlink(path, latest)

        if getattr(engine, "global_rank", 0) == 0:
            print(f"[Checkpoint] saved LoRA {path}")
            log_event(get_logger(), "checkpoint_saved", step=step, path=path,
                      is_lora=True)
        return path
    else:
        # Full-FT: DeepSpeed native checkpoint directory.
        # engine.save_checkpoint is a COLLECTIVE under ZeRO — every rank must
        # call it, or non-master ranks hang. Sidecar writes are rank-0 only.
        tag = f"step_{step:07d}"
        path = os.path.join(out_dir, tag)
        engine.save_checkpoint(out_dir, tag=tag)
        if getattr(engine, "global_rank", 0) != 0:
            return None

        meta = {
            "step":   step,
            "config": vars(config),
            "args":   args_dict,
            "ds_tag": tag,
            "is_lora": False,
        }
        atomic_write_json(meta, os.path.join(path, "meta.json"))

        recipe.to_json(os.path.join(path, "recipe.json"))

        latest = os.path.join(out_dir, "latest_ds")
        atomic_symlink(path, latest)
        if getattr(engine, "global_rank", 0) == 0:
            print(f"[Checkpoint] saved {path}")
            log_event(get_logger(), "checkpoint_saved", step=step, path=path,
                      is_lora=False)
        return path


def load_ds_dpo_checkpoint(
    engine,
    resume_path: str,
    recipe: TrainingRecipe,
) -> Tuple[int, TrainingRecipe]:
    """
    Load a DeepSpeed DPO checkpoint and return (step, recipe).

    Handles both .pt (LoRA) and DeepSpeed directory (full-FT) formats.
    """
    if resume_path.endswith(".pt"):
        # LoRA .pt checkpoint — load via train_dpo helper
        step = load_dpo_checkpoint(resume_path, engine.module, None,
                                    next(engine.module.parameters()).device,
                                    is_lora=True)
        return step, recipe

    # DeepSpeed directory
    meta_path = os.path.join(resume_path, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No meta.json in {resume_path} — not a DeepSpeed DPO checkpoint?"
        )
    with open(meta_path) as f:
        meta = json.load(f)
    tag = meta["ds_tag"]
    engine.load_checkpoint(os.path.dirname(resume_path), tag=tag)
    step = meta.get("step", 0)

    # Load recipe if saved alongside
    recipe_path = os.path.join(resume_path, "recipe.json")
    if os.path.isfile(recipe_path):
        recipe = TrainingRecipe.from_json(recipe_path)

    if getattr(engine, "global_rank", 0) == 0:
        print(f"[Checkpoint] resumed from {resume_path} at step {step}")
    return step, recipe


# ---------------------------------------------------------------------------
# Cosine LR (matches train_dpo.py)
# ---------------------------------------------------------------------------


def _cosine_lr(step, warmup, max_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    t = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# Checkpoint pruning
# ---------------------------------------------------------------------------


def prune_checkpoints_ds(out_dir: str, keep: int = 3, is_lora: bool = False):
    """Prune old checkpoints, handling both .pt (LoRA) and step_* dirs."""
    if is_lora:
        files = sorted(
            [f for f in Path(out_dir).iterdir()
             if f.is_file() and f.name.startswith("dpo_step") and f.suffix == ".pt"],
            key=lambda f: int(f.name.replace("dpo_step", "").replace(".pt", "")),
        )
        for old in files[:-keep]:
            old.unlink()
            print(f"[Checkpoint] pruned {old.name}")
    else:
        dirs = sorted(
            [d for d in Path(out_dir).iterdir()
             if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.replace("step_", "")),
        )
        for old in dirs[:-keep]:
            import shutil
            shutil.rmtree(old)
            print(f"[Checkpoint] pruned {old.name}")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def load_tokenizer(tokenizer_dir: str) -> Tokenizer:
    """Load a HuggingFace `tokenizers` Tokenizer from a directory."""
    path = os.path.join(tokenizer_dir, "tokenizer.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"tokenizer.json not found in {tokenizer_dir}")
    return Tokenizer.from_file(path)


def get_special_token_id_safe(tokenizer: Tokenizer, token: str) -> int:
    tid = tokenizer.token_to_id(token)
    if tid is not None and tid >= 0:
        return tid
    added = tokenizer.get_added_vocabulary()
    if token in added:
        return added[token]
    return tokenizer.get_vocab_size() - 1


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(args):
    # DeepSpeed initialises its own process group.
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK",       0))
    world_size  = int(os.environ.get("WORLD_SIZE",  1))
    master      = global_rank == 0

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Ensure the process group is torn down even if an exception unwinds train().
    atexit.register(lambda: dist.destroy_process_group() if dist.is_initialized() else None)

    # Graceful shutdown on SIGINT/SIGTERM (scheduler preemption, Ctrl+C).
    install_signal_handlers()
    log = setup_logging()
    if master:
        log_event(log, "dpo_start",
                  num_steps=args.max_steps, beta=args.beta,
                  ref_policy=args.ref_policy, resume=args.resume)

    torch.manual_seed(args.seed + global_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32        = True
    rng = random.Random(args.seed + global_rank)

    # ----------------------------------------------------------------- recipe
    recipe = recipe_from_args(args)
    if master:
        print(f"[Recipe] mode={recipe.mode}  model_name={recipe.model_name}")

    # ----------------------------------------------------------------- ckpt
    if not args.checkpoint:
        raise FileNotFoundError(
            "--checkpoint is required. Point it at the SFT checkpoint "
            "produced by train_sft.py."
        )
    ckpt_data = load_torch_checkpoint(args.checkpoint, map_location="cpu")
    config    = ModelConfig(**ckpt_data["config"])

    # ---- architecture variant passthrough (only flags the user explicitly
    # ---- set, so the checkpoint's arch like jamba / MTP heads is preserved)
    apply_architecture_args(config, args, defaults=args.arch_defaults)

    # ---------------------------------------------------------------- auto LR scaling
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

    # ---------------------------------------------------------------- hardware audit
    hw = audit_hardware()

    # ---------------------------------------------------------------- model
    # Build on CPU first; the DeepSpeed engine moves shards to GPU.
    model    = TransformerForCausalLM(config)
    n_params = count_parameters(model)
    if master:
        print_audit(hw, n_params)
        print(f"Loaded SFT checkpoint: {n_params:,} params ({n_params / 1e9:.3f}B)")

    # ----------------------------------------------------------------- LoRA
    is_lora = args.lora
    if is_lora:
        target_modules = tuple(args.lora_target_modules)
        lora_type = args.lora_type
        n_replaced = inject_lora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_type=lora_type,
        )
        freeze_base(model)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if master:
            print(f"[LoRA] injected {n_replaced} adapters | "
                  f"trainable={n_trainable:,} / total={count_parameters(model):,}  "
                  f"type={lora_type}")
    else:
        n_params = count_parameters(model)
        if master:
            print("[LoRA] disabled -- full fine-tune")

    # ----------------------------------------------------------------- ref model
    # Build the reference model on CPU (two-model variant only). For
    # --ref_policy single, we reuse the trainable model under no_grad inside
    # the loop, so no second model is allocated here.
    if args.ref_policy == "two":
        ref_model = build_reference("two", config, args.checkpoint, device)
    else:
        ref_model = None
    ref_for_logprob = ref_model if ref_model is not None else model

    # ----------------------------------------------------------------- ZeRO
    zero_stage, cpu_offload_opt, cpu_offload_param = select_zero_stage_and_offload(
        hw, n_params, world_size,
        force_stage=args.zero_stage,
        force_cpu_offload_optimizer=args.cpu_offload_optimizer,
        force_cpu_offload_param=args.cpu_offload_param,
    )
    if master:
        print(f"[AutoConfig] Selected ZeRO-{zero_stage}  "
              f"cpu_offload_opt={cpu_offload_opt}  "
              f"cpu_offload_param={cpu_offload_param}")

    # ----------------------------------------------------------------- DS config
    ds_cfg = build_ds_config(
        args, zero_stage, cpu_offload_opt, cpu_offload_param,
        gpu_info=hw["gpus"], is_lora=is_lora,
    )
    if master:
        os.makedirs(args.out_dir, exist_ok=True)
        cfg_path = os.path.join(args.out_dir, "ds_config.json")
        with open(cfg_path, "w") as f:
            json.dump(ds_cfg, f, indent=2)
        print_ds_config_summary(ds_cfg, zero_stage, cpu_offload_opt, cpu_offload_param)
        print(f"[DeepSpeed] config written to {cfg_path}")

    # ----------------------------------------------------------------- param groups
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in name or "embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {"params": decay,    "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    # ----------------------------------------------------------------- DeepSpeed init
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=param_groups,
        config=ds_cfg,
    )

    # Bind the reference model: when --ref_policy single, reuse the
    # trainable DeepSpeed engine's underlying model.
    if ref_model is None:
        ref_for_logprob = engine.module

    # --------------------------------------------------------------- tokenizer
    tokenizer = load_tokenizer(args.tokenizer)
    eos_id = get_special_token_id_safe(tokenizer, recipe.eos_token)
    pad_id = get_special_token_id_safe(tokenizer, recipe.pad_token)
    if master:
        print(f"[Tokenizer] eos_id={eos_id}, pad_id={pad_id}, "
              f"vocab={tokenizer.get_vocab_size()}")

    # --------------------------------------------------------------- dataset
    train_ds = PackedDPODataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        rank=global_rank,
        world_size=world_size,
        seed=args.seed,
        split="train",
    )
    if master:
        print(f"[Dataset] {len(train_ds):,} preference triples ({world_size} rank(s))")

    # --------------------------------------------------------------- resume
    start_step = 0
    if args.resume:
        start_step, recipe = load_ds_dpo_checkpoint(engine, args.resume, recipe)

    if master:
        eff_batch = args.batch_size * world_size
        print(f"\nEffective batch     : {eff_batch} preference triples")
        print(f"Max steps           : {args.max_steps:,}")
        print(f"DPO beta            : {args.beta}")
        print(f"Label smoothing     : {args.label_smoothing}")
        print(f"Reference policy    : {args.ref_policy}")
        print(f"Checkpoint every    : {args.save_every:,} steps\n")

    # ================================================================= LOOP
    engine.train()
    t0 = time.perf_counter()
    loss_window: List[float] = []
    acc_window: List[float] = []
    data_iter = iter(train_ds)

    interrupted = False
    for step in range(start_step, args.max_steps):
        # --- graceful shutdown (collective-safe: all ranks agree to stop)
        if should_stop(device, world_size):
            interrupted = True
            if master:
                print(f"\n[Shutdown] requested at step {step} — "
                      f"saving final checkpoint …")
                log_event(log, "shutdown_requested", step=step)
            break

        # LR
        lr = _cosine_lr(step, args.warmup_steps, args.max_steps,
                         args.lr, args.min_lr)
        for pg in engine.optimizer.param_groups:
            pg["lr"] = lr

        # 1. sample a batch of preference triples
        try:
            chosen_ids, chosen_mask, rejected_ids, rejected_mask = next(data_iter)
        except StopIteration:
            data_iter = iter(train_ds)
            chosen_ids, chosen_mask, rejected_ids, rejected_mask = next(data_iter)

        chosen_ids = chosen_ids.to(device)
        chosen_mask = chosen_mask.to(device)
        rejected_ids = rejected_ids.to(device)
        rejected_mask = rejected_mask.to(device)

        # 2. reference log-probs (no_grad)
        with torch.no_grad():
            ref_c_logp, _ = compute_sequence_logprobs(ref_for_logprob, chosen_ids)
            ref_r_logp, _ = compute_sequence_logprobs(ref_for_logprob, rejected_ids)

        # 3. policy log-probs (with grad, through DeepSpeed engine)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=args.dtype == "bf16"):
            policy_c_logp, c_mod_aux = compute_sequence_logprobs(engine, chosen_ids)
            policy_r_logp, r_mod_aux = compute_sequence_logprobs(engine, rejected_ids)
            mod_aux_loss = c_mod_aux + r_mod_aux

        # 4. DPO loss
        loss, metrics = dpo_loss(
            policy_c_logp, policy_r_logp,
            ref_c_logp, ref_r_logp,
            chosen_mask, rejected_mask,
            beta=args.beta,
            label_smoothing=args.label_smoothing,
            clip_ratio=args.clip_ratio if hasattr(args, 'clip_ratio') and args.clip_ratio > 0 else None,
        )
        # ---- MoD auxiliary loss
        loss = loss + mod_aux_loss

        # 5. DeepSpeed step (handles backward + ZeRO all-reduce + optimizer)
        engine.backward(loss)
        engine.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        # 6. logging
        if master:
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
                grad_norm = engine.get_global_grad_norm() or 0.0
                print(
                    f"step {step:6d} | loss {avg_loss:+.4f} | "
                    f"acc {avg_acc:.2%} | "
                    f"r_margin {metrics['reward_margin']:.4f} | "
                    f"lr {lr:.2e} | g {grad_norm:.2f} | "
                    f"{sps:.2f} step/s"
                )

        # 7. checkpoint
        if step > start_step and step % args.save_every == 0:
            # save_ds_dpo_checkpoint is collective for full-FT (all ranks
            # join engine.save_checkpoint); it self-gates to rank 0 for LoRA.
            save_ds_dpo_checkpoint(
                engine, args.out_dir, step, config,
                vars(args), recipe, is_lora,
            )
            if master:
                prune_checkpoints_ds(args.out_dir, keep=args.keep_ckpts, is_lora=is_lora)

    # ---- final
    # On graceful shutdown `step` is exactly the next optimizer step to run
    # (we broke at the top of iteration `step`), so saving it makes resume
    # continue where we stopped. save_ds_dpo_checkpoint is collective for
    # full-FT (all ranks join), so every rank must call it either way.
    final_step = step if interrupted else args.max_steps
    save_ds_dpo_checkpoint(
        engine, args.out_dir, final_step, config,
        vars(args), recipe, is_lora,
    )
    if master:
        if interrupted:
            print(f"\n[Shutdown] checkpoint saved at step {final_step}. "
                  f"Resume with --resume {os.path.join(args.out_dir, 'latest_ds')}")
            log_event(log, "shutdown_checkpoint_saved", step=final_step)
        else:
            print(f"\nDPO complete. Final loss: {loss.detach().item():.4f}")
            log_event(log, "training_complete", step=final_step,
                      final_loss=loss.detach().item())
            if is_lora:
                print(f"\nTo merge LoRA into base weights for deployment:")
                print(f"  python train_dpo.py --merge_lora "
                      f"--checkpoint {args.out_dir}/latest.pt --out_dir ./dpo_merged")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="DeepSpeed DPO preference fine-tuning for dense LLMs."
    )

    # Paths
    p.add_argument("--checkpoint", default=None,
                   help="SFT checkpoint from train_sft.py")
    p.add_argument("--tokenizer", default="./tokenizer",
                   help="Tokenizer directory from tokenizer_train.py")
    p.add_argument("--data-dir", default="./dpo_packed",
                   help="Packed DPO data directory from data/pack_dpo.py")
    p.add_argument("--out-dir", default="./dpo_checkpoints",
                   help="Output directory for checkpoints")
    p.add_argument("--resume", default=None,
                   help="DPO checkpoint to resume from (.pt or DS directory)")

    # LoRA
    p.add_argument("--lora", action="store_true",
                   help="Enable LoRA adapters")
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=float, default=128.0)
    p.add_argument("--lora-target-modules", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
                   help="Target module names for LoRA/DoRA injection")
    p.add_argument("--lora-type", default="lora",
                   choices=["lora", "dora"],
                   help="Type of low-rank adaptation: LoRA or DoRA "
                        "(Weight-Decomposed Low-Rank Adaptation)")

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
    p.add_argument("--max-steps", type=int, default=500,
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
    p.add_argument("--seed", type=int, default=42)

    add_architecture_args(p)

    # Checkpointing
    p.add_argument("--save-every", type=int, default=50,
                   help="Checkpoint interval in steps")
    p.add_argument("--keep-ckpts", type=int, default=3)
    p.add_argument("--log-interval", type=int, default=1)

    # ZeRO / offload (auto-selected if not specified)
    p.add_argument("--zero-stage", type=int, default=None, choices=[1, 2, 3],
                   help="Force ZeRO stage. Default: auto-selected from hardware audit.")
    p.add_argument("--cpu-offload-optimizer", action="store_true",
                   help="Force CPU offload of optimizer states")
    p.add_argument("--cpu-offload-param", action="store_true",
                   help="Force CPU offload of model parameters (ZeRO-3 only)")

    # DeepSpeed launcher
    p.add_argument("--local_rank", type=int, default=-1,
                   help="Set by DeepSpeed launcher; do not set manually.")

    # Recipe
    add_recipe_args(p)

    # Smoke test
    p.add_argument("--smoke-test", action="store_true",
                   help="Run the DDP variant's smoke test and exit")

    args = p.parse_args()
    # Namespace of pure argparse defaults, used to detect which architecture
    # flags the user explicitly set (see apply_architecture_args).
    args.arch_defaults = p.parse_args([])
    return args


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Parse args first to check for smoke-test before requiring deepspeed/GPU
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run self-contained smoke test (no GPU/DeepSpeed needed)")
    parser.add_argument("--checkpoint", type=str, default=None)
    known_args, _ = parser.parse_known_args()

    if known_args.smoke_test:
        smoke_test()
        sys.exit(0)

    if not torch.cuda.is_available():
        print("ERROR: train_dpo_deepspeed.py requires at least one CUDA GPU.")
        sys.exit(1)

    try:
        import deepspeed
    except ImportError:
        print("ERROR: DeepSpeed not installed.  Run:")
        print("  pip install deepspeed")
        sys.exit(1)

    try:
        import psutil
    except ImportError:
        print("[warn] psutil not installed -- CPU RAM reporting will be incomplete.")
        print("       pip install psutil")

    args = parse_args()

    if args.smoke_test:
        smoke_test()
    elif args.checkpoint is None:
        print("No --checkpoint given. Run with --smoke-test for a quick test, "
              "or provide --checkpoint for training.")
        smoke_test()
    else:
        train(args)
