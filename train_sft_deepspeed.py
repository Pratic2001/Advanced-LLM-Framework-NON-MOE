#!/usr/bin/env python3
"""
train_sft_deepspeed.py

DeepSpeed-powered supervised fine-tuning for the dense LLM framework
(Stage 1 of reasoning post-training).

This is the DeepSpeed variant of the SFT training pipeline:

  - full hardware audit (VRAM, NVLink, InfiniBand, CPU RAM) before any
    training logic runs
  - automatic selection of ZeRO stage and CPU offload configuration
  - generated ds_config.json is written to --out-dir and printed
  - MFU estimation via the 4-tier GPU peak-FLOP/s resolver
  - DeepSpeed engine is the optimizer / scheduler / gradient owner
  - checkpoints are written in DeepSpeed's native directory format so
    ZeRO-3 sharded weights round-trip correctly
  - LoRA/DoRA fine-tuning with optional NEFTune noise injection
  - merge-and-save mode for deployment (CPU-only, no DeepSpeed needed)

SFT-specific behaviour:

  - loads a pretrained checkpoint (raw .pt from train.py or a consolidated .pt)
  - reads packed memmap .bin files written by data/pack_sft.py and
    concatenates the worker shards via mmap
  - applies a position-level loss mask: only assistant tokens
    (thinking + answer) contribute to the loss
  - supports LoRA/DoRA on q/k/v/o/gate/up/down projections for single-GPU
    fine-tuning of large models; adapter state is checkpointed separately
  - merge-and-save mode is available without launching the engine

Launch:
    # Single node, 1 GPU
    deepspeed train_sft_deepspeed.py --checkpoint-dir ./checkpoints/latest.pt \\
        --data-dir ./packed --out-dir ./sft_checkpoints_ds

    # Single node, 4 GPUs
    deepspeed --num_gpus 4 train_sft_deepspeed.py --checkpoint-dir ... \\
        --data-dir ./packed --out-dir ./sft_checkpoints_ds

    # Multi-node (2 nodes x 8 GPUs)
    deepspeed --hostfile hostfile.txt train_sft_deepspeed.py \\
        --checkpoint-dir ./checkpoints/latest.pt \\
        --data-dir ./packed --out-dir ./sft_checkpoints_ds

    # LoRA (recommended for >=1B on a single 4090)
    deepspeed train_sft_deepspeed.py --checkpoint-dir ... \\
        --lora-rank 64 --lora-alpha 128 --data-dir ./packed

    # DoRA (weight-decomposed low-rank adaptation)
    deepspeed train_sft_deepspeed.py --checkpoint-dir ... \\
        --lora-rank 32 --lora-type dora --data-dir ./packed

    # Force a specific ZeRO stage (skip auto-selection)
    deepspeed train_sft_deepspeed.py --checkpoint-dir ... --zero-stage 3 \\
        --offload-optimizer --data-dir ./packed

    # Resume from a DeepSpeed SFT checkpoint
    deepspeed train_sft_deepspeed.py --checkpoint-dir ./checkpoints/latest.pt \\
        --data-dir ./packed \\
        --resume ./sft_checkpoints_ds/latest_ds --out-dir ./sft_checkpoints_ds

    # Merge LoRA back into base weights for deployment (no DeepSpeed needed)
    python train_sft_deepspeed.py --merge-and-save \\
        --checkpoint-dir ./sft_checkpoints_ds/latest.pt \\
        --out-dir ./sft_merged

    # NEFTune (embedding noise for smoother loss landscape)
    deepspeed train_sft_deepspeed.py --checkpoint-dir ... --neftune-alpha 5 \\
        --data-dir ./packed
"""

import argparse
import glob
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import deepspeed

from model import ModelConfig, TransformerForCausalLM, count_parameters
from optim.lr_schedule import build_scheduler
from recipe import TrainingRecipe, get_recipe, add_recipe_args, recipe_from_args
from peft.lora import (
    inject_lora,
    merge_lora,
    lora_state_dict,
    freeze_base,
    register_neftune_hook,
)


# ---------------------------------------------------------------------------
# GPU FLOP/s table  (bf16 Tensor Core, per card)
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
    # NVIDIA Jetson / embedded
    "NVIDIA Orin":                     1.3,
    "NVIDIA Xavier":                   1.0,
}


# ---------------------------------------------------------------------------
# Multi-tier TFLOPS resolution
# ---------------------------------------------------------------------------

def _run(cmd: str) -> str:
    """Run a shell command, return stdout or '' on error."""
    try:
        return subprocess.check_output(
            cmd, shell=True, stderr=subprocess.DEVNULL, timeout=10
        ).decode().strip()
    except Exception:
        return ""


def _tflops_from_smi(gpu_index: int) -> Optional[float]:
    """Hook for future nvidia-smi TFLOPS support; currently always None."""
    return None


def _tflops_from_cuda_props(props) -> Tuple[float, str]:
    """Estimate bf16 TFLOPS from CUDA device properties (+-15% accuracy)."""
    cc_major = props.major
    cc_minor = props.minor
    n_sm     = props.multi_processor_count
    clock_hz = props.clock_rate * 1000   # kHz -> Hz

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
    """
    Four-tier TFLOPS resolution:
      Tier 1 - exact table match
      Tier 2 - partial token match
      Tier 3 - nvidia-smi
      Tier 4 - derived from CUDA properties
    """
    name_lo = name.lower()
    for key, val in GPU_PEAK_TFLOPS.items():
        if key.lower() == name_lo:
            return val, "spec-sheet (exact match)"

    import re
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

    smi_val = _tflops_from_smi(gpu_index)
    if smi_val is not None:
        return smi_val, "nvidia-smi query"

    est, method = _tflops_from_cuda_props(props)
    return est, f"computed ({method})"


# ---------------------------------------------------------------------------
# Hardware audit
# ---------------------------------------------------------------------------

def audit_hardware() -> dict:
    """Per-GPU and CPU information for the current node (drives ZeRO choice)."""
    info: dict = {
        "node": platform.node(),
        "gpus": [],
        "cpu": {},
        "interconnect": {},
    }

    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        props    = torch.cuda.get_device_properties(i)
        name     = props.name
        vram_gb  = props.total_memory / 1024**3
        cc_major = props.major
        cc_minor = props.minor

        peak_tflops, tflops_source = resolve_gpu_peak_tflops(name, i, props)

        bw_str = _run(
            f"nvidia-smi --query-gpu=memory.bandwidth --format=csv,noheader,nounits "
            f"-i {i} 2>/dev/null"
        )
        try:
            bw_gb_s = float(bw_str) / 1000
        except ValueError:
            bw_gb_s = {
                "4090": 1008, "3090": 936, "A100": 2000,
                "H100": 3350, "V100": 900,
            }.get(next((k for k in ["4090", "3090", "A100", "H100", "V100"]
                        if k in name), ""), 800)

        nvlink_str = _run(f"nvidia-smi nvlink -s -i {i} 2>/dev/null | grep 'Speed' | head -1")
        has_nvlink = bool(nvlink_str)

        info["gpus"].append({
            "index":         i,
            "name":          name,
            "vram_gb":       round(vram_gb, 2),
            "cc":            f"{cc_major}.{cc_minor}",
            "bf16":          cc_major >= 8,
            "peak_tflops":   peak_tflops,
            "tflops_source": tflops_source,
            "bw_gb_s":       bw_gb_s,
            "has_nvlink":    has_nvlink,
        })

    try:
        import psutil
        cpu_ram_gb = psutil.virtual_memory().total / 1024**3
        cpu_cores  = psutil.cpu_count(logical=False) or 1
    except ImportError:
        cpu_ram_gb = 0.0
        cpu_cores  = os.cpu_count() or 1

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


def print_audit(info: dict, n_trainable: int, n_total: int, lora: bool):
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  HARDWARE AUDIT  —  node: {info['node']}")
    print(sep)
    print(f"  GPUs: {len(info['gpus'])}")
    for g in info["gpus"]:
        bf16_tag = "bf16✓" if g["bf16"] else "fp16-only"
        nvlink   = " NVLink✓" if g["has_nvlink"] else ""
        src      = g.get("tflops_source", "unknown")
        acc_tag  = "" if "spec-sheet" in src else f"  ⚠ TFLOPS estimated ({src})"
        print(f"    [{g['index']}] {g['name']}  "
              f"{g['vram_gb']:.1f} GB VRAM  "
              f"{g['peak_tflops']:.0f} TFLOP/s [{src}]  "
              f"CC{g['cc']}  {bf16_tag}{nvlink}{acc_tag}")
    cpu = info["cpu"]
    print(f"  CPU: {cpu.get('model','?')[:50]}  "
          f"{cpu['cores']} cores  {cpu['ram_gb']:.0f} GB RAM")
    ib = info["interconnect"]["infiniband_ports"]
    nv = "NVLink✓" if info["interconnect"]["nvlink"] else "PCIe"
    print(f"  Interconnect: {nv}  "
          f"{'InfiniBand (' + str(ib) + ' ports)' if ib else 'Ethernet'}")

    mode = "LoRA/DoRA" if lora else "full fine-tune"
    trainable_pct = 100.0 * n_trainable / max(n_total, 1)
    print(f"\n  Model:       {n_total/1e9:.3f}B params total  "
          f"(~{n_total*2/1024**3:.1f} GB bf16 weights)")
    print(f"  Trainable:   {n_trainable/1e6:.2f}M params  "
          f"({trainable_pct:.2f}%)  [{mode}]")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# ZeRO stage auto-selection
# ---------------------------------------------------------------------------

def select_zero_stage_and_offload(
    info: dict,
    n_trainable: int,
    n_total: int,
    world_size: int,
    force_stage: Optional[int],
    force_cpu_offload_optimizer: bool,
    force_cpu_offload_param: bool,
) -> Tuple[int, bool, bool]:
    """
    SFT memory profiles:
      - LoRA/DoRA : only adapter parameters + their optimizer states are
                    trainable; base weights are frozen and live in bf16
      - full      : every parameter is trainable

    Budget is sized off `n_total` because base weights must be resident
    regardless of mode. Adam state is sized off `n_trainable` since only
    those parameters carry optimizer state.
    """
    if not info["gpus"]:
        return 1, False, False

    min_vram  = min(g["vram_gb"] for g in info["gpus"]) * 0.85
    n_gpus    = max(len(info["gpus"]), 1)

    # Static components per GPU (GB)
    full_gb   = n_total * (2 + 2 + 8) / 1024**3 / n_gpus
    zero2_gb  = n_total * (2 + 2)     / 1024**3 / n_gpus
    zero3_gb  = n_total * 2           / 1024**3 / n_gpus

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
            print(f"[AutoConfig] VRAM very tight — also enabling CPU parameter offload.")

    cpu_ram = info["cpu"].get("ram_gb", 0)
    if (cpu_offload_opt or cpu_offload_param) and cpu_ram > 0:
        needed_gb = n_trainable * 8 / 1024**3
        if needed_gb > cpu_ram * 0.6:
            print(f"[AutoConfig] WARNING: optimizer offload needs ~{needed_gb:.1f} GB "
                  f"CPU RAM but only {cpu_ram:.0f} GB available.")

    return stage, cpu_offload_opt, cpu_offload_param


# ---------------------------------------------------------------------------
# DS config builder
# ---------------------------------------------------------------------------

def build_ds_config(
    args,
    zero_stage: int,
    cpu_offload_optimizer: bool,
    cpu_offload_param: bool,
    gpu_info: List[dict],
) -> dict:
    """
    Construct a deepspeed config dict.
    gradient_accumulation_steps=1 because the training loop drives the
    accumulation manually.
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

    bf16_cfg = {"enabled": args.dtype == "bf16"}
    fp16_cfg = {"enabled": False}

    zero_cfg: dict = {
        "stage": zero_stage,
        "reduce_bucket_size":    5e8,
        "allgather_bucket_size": 5e8,
        "overlap_comm":          True,
        "contiguous_gradients":  True,
        "sub_group_size":        1e9,
        "stage3_max_live_parameters":   1e9,
        "stage3_max_reuse_distance":    1e9,
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
    zero_cfg["reduce_scatter"]       = True
    zero_cfg["allgather_partitions"] = True
    if not has_nvlink:
        zero_cfg["reduce_bucket_size"]    = 2e8
        zero_cfg["allgather_bucket_size"] = 2e8

    act_ckpt: dict = {}
    if args.gradient_checkpointing:
        act_ckpt = {
            "partition_activations":          False,
            "cpu_checkpointing":              False,
            "contiguous_memory_optimization": False,
            "synchronize_checkpoint_boundary": False,
        }

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
    if act_ckpt:
        cfg["activation_checkpointing"] = act_ckpt

    return cfg


def print_ds_config_summary(cfg: dict, zero_stage: int,
                             cpu_opt: bool, cpu_param: bool):
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  DEEPSPEED CONFIG SUMMARY")
    print(sep)
    print(f"  ZeRO Stage            : {zero_stage}")
    print(f"  CPU offload optimizer : {cpu_opt}")
    print(f"  CPU offload params    : {cpu_param}")
    print(f"  BF16                  : {cfg['bf16']['enabled']}")
    print(f"  Grad accum steps      : {cfg['gradient_accumulation_steps']}")
    print(f"  Micro batch / GPU     : {cfg['train_micro_batch_size_per_gpu']}")
    print(f"  Grad clip             : {cfg['gradient_clipping']}")
    z = cfg["zero_optimization"]
    print(f"  Reduce bucket         : {z['reduce_bucket_size']/1e6:.0f} MB")
    print(f"  Overlap comm          : {z['overlap_comm']}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Pretrained checkpoint loader
# ---------------------------------------------------------------------------

def load_pretrained_checkpoint(path: str) -> Tuple[ModelConfig, dict]:
    """
    Accept either:
      - a raw .pt with keys {'model_state', 'config'}
      - a DeepSpeed checkpoint directory (points user to consolidator)
    """
    if os.path.isdir(path):
        meta_path = os.path.join(path, "meta.json")
        if os.path.exists(meta_path):
            raise RuntimeError(
                f"{path} looks like a DeepSpeed checkpoint directory.\n"
                f"Run deepspeed_shard_consolidator.py first to produce a "
                f"single .pt, then point --checkpoint-dir at the consolidated file."
            )
        raise RuntimeError(
            f"{path} is a directory but does not contain meta.json; not a "
            f"recognised SFT input. Pass the path to a consolidated .pt."
        )

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "config" not in blob or "model_state" not in blob:
        raise RuntimeError(
            f"{path} does not contain the expected keys "
            f"'config' and 'model_state'. Was it produced by train.py or "
            f"deepspeed_shard_consolidator.py?"
        )
    return ModelConfig(**blob["config"]), blob["model_state"]


# ---------------------------------------------------------------------------
# MFU
# ---------------------------------------------------------------------------

def estimate_mfu(model, tokens_per_sec: float, gpu_info: List[dict]) -> float:
    """
    MFU reported against non-embedding total parameter count.
    Forward+backward still multiply through frozen base-model weights in
    LoRA mode, so counting only adapter params would undercount real FLOPs.
    """
    raw = model.module if hasattr(model, "module") else model
    inner = raw._orig_mod if hasattr(raw, "_orig_mod") else raw
    n = sum(p.numel() for pname, p in inner.named_parameters()
            if "embed_tokens" not in pname)

    flops = 6 * n * tokens_per_sec
    if not gpu_info:
        return 0.0
    peak = gpu_info[0]["peak_tflops"] * 1e12 * len(gpu_info)
    return flops / peak if peak > 0 else 0.0


# ---------------------------------------------------------------------------
# Data: PackedSFTDataLoader (memmap-based, worker-shard aware)
# ---------------------------------------------------------------------------

class PackedSFTDataLoader:
    """
    Reads packed memmap files produced by data/pack_sft.py:

        <data_dir>/sft_<split>_tokens[.w<i>-of-<n>].bin   - uint16 token ids
        <data_dir>/sft_<split>_mask[.w<i>-of-<n>].bin     - uint8 loss mask
        <data_dir>/sft_<split>_manifest[.w<i>-of-<n>].json - metadata

    All worker shards found for the requested split are discovered,
    sorted, and concatenated (still mmap-backed, no RAM copy). The
    result is sharded across DDP ranks by token count.

    get_batch() yields (x, y, loss_mask, num_valid) where num_valid
    is the number of tokens per sample that contribute to the loss.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
        split: str = "train",
    ):
        self.seq_len    = seq_len
        self.rank       = rank
        self.world_size = world_size
        self.split      = split

        manifests = self._discover_manifests(data_dir, split)
        if not manifests:
            raise FileNotFoundError(
                f"No packed manifests found for split={split!r} in {data_dir}. "
                f"Run data/pack_sft.py first."
            )

        dtype_t = np.dtype(manifests[0].get("dtype_tokens", "uint16"))
        dtype_m = np.dtype(manifests[0].get("dtype_mask", "uint8"))

        token_arrays = []
        mask_arrays  = []
        total_records = 0
        for m in manifests:
            tok_path  = os.path.join(data_dir, m["token_file"])
            mask_path = os.path.join(data_dir, m["mask_file"])
            token_arrays.append(np.memmap(tok_path,  dtype=dtype_t, mode="r"))
            mask_arrays.append(np.memmap(mask_path, dtype=dtype_m, mode="r"))
            total_records += m.get("num_records", 0)

        self.tokens = _ConcatMemmap(token_arrays)
        self.mask   = _ConcatMemmap(mask_arrays)

        if rank == 0:
            print(f"[PackedSFTDataLoader] {split}: discovered "
                  f"{len(manifests)} worker shard(s) in {data_dir} "
                  f"({total_records:,} records total)")

        # Shard across DDP ranks by token count
        total = len(self.tokens)
        shard_size = total // world_size
        start = rank * shard_size
        end   = start + shard_size if rank < world_size - 1 else total
        self.tokens = self.tokens[start:end]
        self.mask   = self.mask[start:end]

        n_windows = max(0, (len(self.tokens) - 1) // seq_len)
        print(f"[PackedSFTDataLoader rank {rank}] {split}: "
              f"{len(self.tokens):,} tokens -> {n_windows:,} windows of {seq_len}")

    @staticmethod
    def _discover_manifests(data_dir: str, split: str):
        """
        Find every sft_<split>_manifest*.json in data_dir, sorted by
        worker index for deterministic concatenation.
        """
        pattern = os.path.join(data_dir, f"sft_{split}_manifest*.json")
        manifest_paths = sorted(glob.glob(pattern))
        manifests = []
        for p in manifest_paths:
            with open(p, "r") as f:
                m = json.load(f)
            manifests.append(m)
        # Some manifests may not have a "worker" key; sort by filename instead
        manifests.sort(key=lambda m: m.get("worker", 0))
        return manifests

    def __len__(self) -> int:
        return max(0, (len(self.tokens) - 1) // self.seq_len)

    def get_batch(self, batch_size: int, device: torch.device):
        """
        Sample `batch_size` random windows.

        Returns:
            (x, y, loss_mask, num_valid)
            x: (B, T) input token ids
            y: (B, T) target token ids (shifted by 1)
            loss_mask: (B, T) float, 1=compute loss, 0=ignore
            num_valid: (B,) int, number of tokens per sample contributing to loss
        """
        n = len(self)
        if n == 0:
            raise RuntimeError(
                "PackedSFTDataLoader has no complete windows. "
                "Try a smaller --seq-len, or re-run data/pack_sft.py with "
                "a larger dataset."
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

        num_valid = m.sum(dim=1).long()  # (B,)
        return x, y, m, num_valid


class _ConcatMemmap:
    """
    Read-only view making several np.memmap arrays look like one
    contiguous array, without copying any into RAM.
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
# Masked cross-entropy (only assistant tokens contribute)
# ---------------------------------------------------------------------------

def masked_cross_entropy(
    logits: torch.Tensor,    # (B, T, V)
    targets: torch.Tensor,   # (B, T)
    mask: torch.Tensor,      # (B, T) float, 1=compute loss, 0=ignore
) -> torch.Tensor:
    """Cross-entropy masked to active positions only (assistant tokens)."""
    B, T, V = logits.shape
    logits_flat  = logits.reshape(B * T, V)
    targets_flat = targets.reshape(B * T)
    mask_flat    = mask.reshape(B * T)

    nll = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    denom = mask_flat.sum().clamp(min=1.0)
    return (nll * mask_flat).sum() / denom


# ---------------------------------------------------------------------------
# Checkpoint save / load  (DeepSpeed-native with LoRA sidecar + recipe)
# ---------------------------------------------------------------------------

def save_ds_sft_checkpoint(
    engine,
    step: int,
    out_dir: str,
    config: ModelConfig,
    recipe: TrainingRecipe,
    args_dict: dict,
    best_val_loss: float,
    is_lora: bool,
):
    """
    DeepSpeed-native save. Writes a directory `step_<n>/` containing
    ZeRO-sharded model/optimizer states, plus a sidecar meta.json that
    holds our config, recipe, CLI args, best val loss, and the LoRA adapter
    (if applicable).
    """
    tag  = f"step_{step:07d}"
    path = os.path.join(out_dir, tag)
    engine.save_checkpoint(out_dir, tag=tag)

    sidecar: dict = {
        "step":          step,
        "config":        vars(config),
        "recipe":        recipe.to_dict(),
        "args":          args_dict,
        "best_val_loss": best_val_loss,
        "ds_tag":        tag,
        "is_lora":       is_lora,
    }

    if is_lora:
        raw = engine.module
        inner = raw._orig_mod if hasattr(raw, "_orig_mod") else raw
        sidecar["lora_state"] = lora_state_dict(inner)

    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(sidecar, f, indent=2)

    # Also write recipe.json alongside the checkpoint directory
    recipe.to_json(os.path.join(path, "recipe.json"))

    latest = os.path.join(out_dir, "latest_ds")
    if os.path.islink(latest):
        os.remove(latest)
    os.symlink(os.path.abspath(path), latest)
    print(f"[Checkpoint] saved {path}  (lora={is_lora})")


def load_ds_sft_checkpoint(
    engine,
    resume_path: str,
    is_lora: bool,
) -> Tuple[int, float, Optional[TrainingRecipe]]:
    """
    Reload a DeepSpeed SFT checkpoint.
    LoRA mode recovers the adapter state from the sidecar meta.json
    after engine.load_checkpoint has restored the ZeRO-sharded base weights.
    """
    meta_path = os.path.join(resume_path, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No meta.json in {resume_path} — not a DeepSpeed SFT checkpoint?"
        )
    with open(meta_path) as f:
        meta = json.load(f)
    tag = meta["ds_tag"]
    engine.load_checkpoint(os.path.dirname(resume_path), tag=tag)

    recipe = None
    if "recipe" in meta:
        recipe = TrainingRecipe.from_dict(meta["recipe"])
    elif os.path.exists(os.path.join(resume_path, "recipe.json")):
        recipe = TrainingRecipe.from_json(os.path.join(resume_path, "recipe.json"))

    if is_lora and "lora_state" in meta:
        raw = engine.module
        inner = raw._orig_mod if hasattr(raw, "_orig_mod") else raw
        missing, unexpected = inner.load_state_dict(meta["lora_state"], strict=False)
        if unexpected:
            print(f"[Checkpoint] WARNING: {len(unexpected)} unexpected LoRA "
                  f"keys when resuming; ignoring")
        print(f"[Checkpoint] restored {len(meta['lora_state'])} LoRA tensors "
              f"from sidecar")

    step          = meta.get("step", 0)
    best_val_loss = meta.get("best_val_loss", float("inf"))
    print(f"[Checkpoint] resumed from {resume_path} at step {step}")
    return step, best_val_loss, recipe


def prune_checkpoints(out_dir: str, keep: int = 3):
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
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model,
    val_loader: PackedSFTDataLoader,
    pad_id: int,
    eval_steps: int = 200,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    world_size: int = 1,
) -> float:
    """
    Evaluate the model on the validation set.
    Returns the mean masked cross-entropy loss across eval_steps.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    losses: List[float] = []
    for _ in range(eval_steps):
        x, y, m, _nv = val_loader.get_batch(batch_size, device=device)
        out  = model(x)
        loss = masked_cross_entropy(out["logits"], y, m)
        losses.append(loss.item())
    model.train()
    mean_loss = float(np.mean(losses)) if losses else float("inf")
    if world_size > 1:
        t = torch.tensor(mean_loss, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        mean_loss = t.item()
    return mean_loss


# ---------------------------------------------------------------------------
# NEFTune helper
# ---------------------------------------------------------------------------

def _apply_neftune(model, alpha: float):
    """Register NEFTune noise hook on the embedding layer if alpha > 0."""
    if alpha > 0:
        register_neftune_hook(model, alpha)
        print(f"[NEFTune] enabled with alpha={alpha}")


# ---------------------------------------------------------------------------
# Merge-only mode
# ---------------------------------------------------------------------------

def merge_and_save(args):
    """
    CPU-only path. Reads a (raw or DeepSpeed SFT) LoRA checkpoint and writes
    a consolidated .pt with the adapter folded into the base weights.
    """
    device = torch.device("cpu")
    print(f"[Merge] loading base checkpoint {args.checkpoint_dir} …")
    config, base_state = load_pretrained_checkpoint(args.checkpoint_dir)
    model = TransformerForCausalLM(config)
    model.load_state_dict(base_state)
    model.tie_weights()

    # The SFT sidecar holds the LoRA tensors in raw form
    ckpt_blob = torch.load(args.checkpoint_dir, map_location=device, weights_only=False)
    lora_sd   = ckpt_blob.get("lora_state")
    if lora_sd is None:
        raise RuntimeError(
            f"{args.checkpoint_dir} has no 'lora_state' field — was it saved by "
            f"train_sft_deepspeed.py in LoRA/DoRA mode?"
        )

    rank  = ckpt_blob.get("args", {}).get("lora_rank", 64)
    alpha = ckpt_blob.get("args", {}).get("lora_alpha", 128.0)
    lora_type = ckpt_blob.get("args", {}).get("lora_type", "lora")
    target_modules = ckpt_blob.get("args", {}).get(
        "lora_target_modules",
        ("q_proj", "k_proj", "v_proj", "o_proj",
         "gate_proj", "up_proj", "down_proj"),
    )

    n_lora = inject_lora(
        model, rank=rank, alpha=alpha,
        target_modules=target_modules,
        use_dora=(lora_type == "dora"),
    )
    print(f"[Merge] injected {n_lora} adapters "
          f"(rank={rank}, alpha={alpha}, type={lora_type})")

    missing, unexpected = model.load_state_dict(lora_sd, strict=False)
    if unexpected:
        print(f"[Merge] WARNING: {len(unexpected)} unexpected keys when "
              f"loading LoRA state; ignoring")
    if missing:
        print(f"[Merge] WARNING: {len(missing)} missing keys when loading "
              f"LoRA state (should be empty)")

    model = merge_lora(model)
    model.tie_weights()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "merged_model.pt")
    torch.save({"model_state": model.state_dict(),
                "config":      vars(config)}, out_path)
    print(f"[Merge] saved merged model to {out_path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    # DeepSpeed initialises its own process group
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK",       0))
    world_size  = int(os.environ.get("WORLD_SIZE",  1))
    master      = global_rank == 0

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(args.seed + global_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32        = True

    # ---------------------------------------------------------------- model
    config, base_state = load_pretrained_checkpoint(args.checkpoint_dir)
    model = TransformerForCausalLM(config)
    model.load_state_dict(base_state)
    model.tie_weights()

    is_lora = args.lora_rank > 0
    lora_type = getattr(args, "lora_type", "lora")
    target_modules = getattr(args, "lora_target_modules",
                             ("q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"))

    if is_lora:
        n_replaced = inject_lora(
            model, rank=args.lora_rank, alpha=args.lora_alpha,
            target_modules=target_modules,
            use_dora=(lora_type == "dora"),
        )
        # Freeze all non-LoRA parameters
        freeze_base(model)
        n_trainable = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
        if master:
            print(f"[LoRA] injected {n_replaced} adapters  "
                  f"target={target_modules}  "
                  f"rank={args.lora_rank}  alpha={args.lora_alpha}  "
                  f"type={lora_type}")
    else:
        n_trainable = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)

    n_total = count_parameters(model)
    if master:
        print(f"Pretrained model: {n_total/1e9:.3f}B params  "
              f"({n_total:,} total)")

    # -------------------------------------------------------- NEFTune
    neftune_alpha = getattr(args, "neftune_alpha", 0.0)
    if neftune_alpha > 0:
        _apply_neftune(model, neftune_alpha)

    # -------------------------------------------------------- gradient ckpt
    if args.gradient_checkpointing:
        model.model.enable_gradient_checkpointing()

    # ---------------------------------------------------------------- audit
    hw = audit_hardware()
    if master:
        print_audit(hw, n_trainable, n_total, lora=is_lora)

    # -------------------------------------------------------- ZeRO selection
    zero_stage, cpu_offload_opt, cpu_offload_param = select_zero_stage_and_offload(
        hw, n_trainable, n_total, world_size,
        force_stage=args.zero_stage,
        force_cpu_offload_optimizer=args.offload_optimizer or args.cpu_offload,
        force_cpu_offload_param=args.offload_params or args.cpu_offload,
    )
    if master:
        print(f"[AutoConfig] Selected ZeRO-{zero_stage}  "
              f"cpu_offload_opt={cpu_offload_opt}  "
              f"cpu_offload_param={cpu_offload_param}")

    # ---------------------------------------------------------------- DS cfg
    ds_cfg = build_ds_config(
        args, zero_stage, cpu_offload_opt, cpu_offload_param,
        gpu_info=hw["gpus"],
    )
    if master:
        os.makedirs(args.out_dir, exist_ok=True)
        cfg_path = os.path.join(args.out_dir, "ds_config.json")
        with open(cfg_path, "w") as f:
            json.dump(ds_cfg, f, indent=2)
        print_ds_config_summary(ds_cfg, zero_stage,
                                cpu_offload_opt, cpu_offload_param)
        print(f"[DeepSpeed] config written to {cfg_path}")

    # --------------------------------------------------------- param groups
    # Exclude norms/embeddings/lora_B from weight decay
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in name or "embed" in name or "lora_B" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {"params": decay,    "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if master:
        print(f"[Optimizer] decay={sum(p.numel() for p in decay):,}  "
              f"no_decay={sum(p.numel() for p in no_decay):,}")

    # ---------------------------------------------------------------- compile
    _use_cudagraphs = False
    if args.compile:
        if master:
            print(f"[compile] torch.compile(mode='{args.compile_mode}')...")
        model = torch.compile(model, mode=args.compile_mode)
        _use_cudagraphs = (args.compile_mode == "reduce-overhead")

    # --------------------------------------------------------- DeepSpeed init
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=param_groups,
        config=ds_cfg,
    )

    # ---------------------------------------------------------------- recipe
    recipe = recipe_from_args(args)

    # ---------------------------------------------------------------- data
    if master:
        print(f"\nReading packed SFT data from {args.data_dir} ...")
    train_loader = PackedSFTDataLoader(
        data_dir=args.data_dir, seq_len=args.seq_len,
        rank=global_rank, world_size=world_size, split="train",
    )
    val_loader = PackedSFTDataLoader(
        data_dir=args.data_dir, seq_len=args.seq_len,
        rank=global_rank, world_size=world_size, split="val",
    ) if args.val_fraction > 0 else None

    if len(train_loader) == 0:
        raise RuntimeError(
            "Training dataset has no complete windows. Try a smaller "
            "--seq-len, or re-pack with a larger dataset."
        )

    # ---------------------------------------------------------------- resume
    start_step    = 0
    best_val_loss = float("inf")
    if args.resume:
        start_step, best_val_loss, recipe = load_ds_sft_checkpoint(
            engine, args.resume, is_lora
        )

    # ---------------------------------------------------------------- LR scheduler
    lr_scheduler = build_scheduler(
        schedule="cosine",
        warmup_steps=args.warmup_steps,
        max_steps=args.num_steps,
        peak_lr=args.lr,
        min_lr=args.min_lr,
    )

    # ---------------------------------------------------------------- W&B
    use_wandb = False
    if master and args.wandb_project:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"sft-{args.model_size}-z{zero_stage}",
                config={
                    **vars(config), "n_params": n_total,
                    "n_trainable": n_trainable,
                    "lora": is_lora,
                    "lora_rank": args.lora_rank if is_lora else None,
                    "lora_type": lora_type if is_lora else None,
                    "zero_stage": zero_stage,
                    "recipe_mode": recipe.mode,
                    **vars(args),
                },
            )
            use_wandb = True
        except Exception as e:
            print(f"[W&B] disabled: {e}")

    # --------------------------------------------------------- accounting
    tokens_per_step = (
        args.batch_size * args.seq_len * args.grad_accum_steps * world_size
    )
    if master:
        print(f"\nTokens / optimizer step : {tokens_per_step:,}")
        print(f"Effective batch size    : "
              f"{tokens_per_step // args.seq_len:,} samples")
        print(f"Max steps               : {args.num_steps:,}")
        print(f"Checkpoint every        : {args.ckpt_interval:,} steps\n")

    # ================================================================ LOOP
    engine.train()
    t0         = time.perf_counter()
    loss_accum = 0.0

    pad_id = 0  # padding token id (default 0)

    for step in range(start_step, args.num_steps):
        # Manual LR override via scheduler
        lr = lr_scheduler(step)
        for pg in engine.optimizer.param_groups:
            pg["lr"] = lr

        # One outer iteration == one optimizer step. We drive the
        # grad-accumulation loop by hand and only call engine.step()
        # on the last micro-batch.
        for micro in range(args.grad_accum_steps):
            x, y, m, _nv = train_loader.get_batch(args.batch_size, device)
            if _use_cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()

            # Forward under bfloat16 autocast
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out  = engine(x)
                loss = masked_cross_entropy(
                    out["logits"], y, m
                ) / args.grad_accum_steps

            engine.backward(loss)
            loss_accum += loss.item()

        engine.step()

        # ---- logging
        if master and step % args.log_interval == 0:
            t1          = time.perf_counter()
            tok_per_sec = tokens_per_step * args.log_interval / max(t1 - t0, 1e-9)
            mfu         = estimate_mfu(engine, tok_per_sec, hw["gpus"])
            loss_display = loss_accum / args.log_interval
            loss_accum   = 0.0
            grad_norm    = engine.get_global_grad_norm() or 0.0

            print(
                f"step {step:7d} | loss {loss_display:.4f} | lr {lr:.2e} | "
                f"grad {grad_norm:.3f} | {tok_per_sec/1e3:.1f}k tok/s | "
                f"mfu {mfu*100:.2f}%"
            )
            if use_wandb:
                import wandb
                wandb.log({
                    "train/loss":          loss_display,
                    "train/lr":            lr,
                    "train/grad_norm":     grad_norm,
                    "perf/tokens_per_sec": tok_per_sec,
                    "perf/mfu_pct":        mfu * 100,
                }, step=step)
            t0 = t1

        # ---- validation
        if (step % args.eval_interval == 0 and step > start_step
                and val_loader is not None):
            val_loss = validate(
                engine, val_loader, pad_id,
                eval_steps=args.eval_steps,
                batch_size=args.batch_size,
                device=device,
                world_size=world_size,
            )
            if master:
                improved = " ✓ best" if val_loss < best_val_loss else ""
                print(f"  [eval] step {step:7d} | val_loss {val_loss:.4f}"
                      f"{improved}")
                if use_wandb:
                    import wandb
                    wandb.log({"val/loss": val_loss}, step=step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

        # ---- checkpoint
        if step % args.ckpt_interval == 0 and step > start_step:
            if master:
                save_ds_sft_checkpoint(
                    engine, step, args.out_dir,
                    config, recipe, vars(args),
                    best_val_loss, is_lora,
                )
                prune_checkpoints(args.out_dir, keep=args.keep_ckpts)

    # ---- final checkpoint
    if master:
        save_ds_sft_checkpoint(
            engine, args.num_steps, args.out_dir,
            config, recipe, vars(args),
            best_val_loss, is_lora,
        )
        print(f"\nSFT complete. Best val loss: {best_val_loss:.4f}")
        if is_lora:
            print(f"\nTo merge LoRA into base weights for deployment:")
            print(f"  python train_sft_deepspeed.py --merge-and-save \\")
            print(f"      --checkpoint-dir <base .pt> \\")
            print(f"      --out-dir ./sft_merged")
        if use_wandb:
            import wandb
            wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="DeepSpeed SFT for dense LLM framework "
                    "(Stage 1 of reasoning post-training)."
    )

    # mode
    p.add_argument("--merge-and-save", action="store_true",
                   help="Merge LoRA/DoRA into base weights and save; "
                        "skip training. Runs on CPU, DeepSpeed not required.")

    # paths
    p.add_argument("--checkpoint-dir", default=None,
                   help="Pretrained checkpoint (raw .pt or consolidated DS .pt). "
                        "Required unless --merge-and-save is used.")
    p.add_argument("--data-dir", default="./packed",
                   help="Packed memmap files from data/pack_sft.py")
    p.add_argument("--out-dir", default="./sft_checkpoints_ds")
    p.add_argument("--resume", default=None,
                   help="Path to a DeepSpeed SFT checkpoint directory "
                        "to resume from")

    # LoRA / DoRA
    p.add_argument("--lora-rank", type=int, default=0,
                   help="LoRA/DoRA rank. Set >0 to enable low-rank adaptation. "
                        "Default 0 = full fine-tune.")
    p.add_argument("--lora-alpha", type=float, default=128.0)
    p.add_argument("--lora-target-modules", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
                   help="Target module names for LoRA/DoRA injection")
    p.add_argument("--lora-type", default="lora",
                   choices=["lora", "dora"],
                   help="Type of low-rank adaptation: LoRA or DoRA "
                        "(Weight-Decomposed Low-Rank Adaptation)")

    # NEFTune
    p.add_argument("--neftune-alpha", type=float, default=0.0,
                   help="NEFTune noise alpha. >0 enables embedding noise "
                        "injection for smoother loss landscape "
                        "(typical values: 5-15)")

    # training
    p.add_argument("--model-size", default="sft",
                   help="Label used for logging / W&B run name; the actual "
                        "architecture is read from --checkpoint-dir's config")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Micro-batch size PER GPU (before grad accum)")
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=100000,
                   help="Total training steps")
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-5,
                   help="Peak LR (typically 1e-5 to 5e-5 for SFT)")
    p.add_argument("--min-lr", type=float, default=2e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--compile", action="store_true",
                   help="Run torch.compile for kernel fusion (+25-40% throughput)")
    p.add_argument("--compile-mode", default="default",
                   choices=["default", "reduce-overhead", "max-autotune"],
                   help=("torch.compile mode. "
                         "'default' — safe, good speedup, no CUDAGraphs. "
                         "'reduce-overhead' — uses CUDAGraphs for lower "
                         "kernel-launch overhead. "
                         "'max-autotune' — exhaustive kernel search, "
                         "very slow to compile."))
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Recompute activations on backward (~35% less VRAM)")
    p.add_argument("--seed", type=int, default=42)

    # ZeRO / offload
    p.add_argument("--zero-stage", type=int, default=None,
                   choices=[1, 2, 3],
                   help="Force ZeRO stage. Default: auto-selected "
                        "from hardware audit.")
    p.add_argument("--offload-optimizer", action="store_true",
                   help="Force CPU offload of optimizer states")
    p.add_argument("--offload-params", action="store_true",
                   help="Force CPU offload of model parameters (ZeRO-3 only)")
    p.add_argument("--cpu-offload", action="store_true",
                   help="Convenience: enable BOTH optimizer and parameter "
                        "CPU offload")

    # data split
    p.add_argument("--val-fraction", type=float, default=0.0,
                   help="Validation fraction (must match value used in "
                        "data/pack_sft.py). Set >0 to enable evaluation loop.")

    # checkpointing / logging
    p.add_argument("--ckpt-interval", type=int, default=1000)
    p.add_argument("--keep-ckpts", type=int, default=3)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)

    # DeepSpeed passes its own args
    p.add_argument("--local_rank", type=int, default=-1,
                   help="Set by DeepSpeed launcher; do not set manually.")

    # Add recipe args
    add_recipe_args(p)

    return p.parse_args()


def _has_sft_manifests(data_dir: str) -> bool:
    """True if at least one sft_*_manifest*.json is present."""
    if not os.path.isdir(data_dir):
        return False
    pattern = os.path.join(data_dir, "sft_*_manifest*.json")
    return len(glob.glob(pattern)) > 0


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # merge-and-save runs on CPU without DeepSpeed
    if args.merge_and_save:
        if not args.checkpoint_dir:
            raise ValueError("--checkpoint-dir is required for --merge-and-save")
        merge_and_save(args)
        sys.exit(0)

    if not args.checkpoint_dir:
        raise ValueError(
            "--checkpoint-dir is required (path to a pretrained .pt produced "
            "by train.py or deepspeed_shard_consolidator.py)."
        )

    if not torch.cuda.is_available():
        print("ERROR: train_sft_deepspeed.py requires at least one CUDA GPU.")
        sys.exit(1)

    try:
        import deepspeed  # noqa: F401
    except ImportError:
        print("ERROR: DeepSpeed not installed.  Run:")
        print("  pip install deepspeed")
        sys.exit(1)

    try:
        import psutil  # noqa: F401
    except ImportError:
        print("[warn] psutil not installed — CPU RAM reporting "
              "will be incomplete.")
        print("       pip install psutil")

    if not _has_sft_manifests(args.data_dir):
        print(f"ERROR: no sft_*_manifest*.json found in {args.data_dir}.")
        print("Run data/pack_sft.py first to produce packed SFT data.")
        sys.exit(1)

    train(args)
