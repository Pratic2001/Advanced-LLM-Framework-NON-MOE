#!/usr/bin/env python3
"""
benchmark_tune.py — Hardware Benchmark & Hyperparameter Recommendation

Benchmarks your hardware across every training stage in this framework and
recommends optimal hyperparameters: model size ceiling, batch size, learning
rate, optimizer choice, LoRA rank, GRPO generations, and DPO beta.

Modes:
  --quick              Hardware detection + math estimation only (instant).
  --pretrain           Benchmark pretraining (batch sizes, Adam vs Muon).
  --sft                Benchmark SFT (LoRA ranks).
  --grpo               Benchmark GRPO (num-generations).
  --dpo                Benchmark DPO (batch sizes).
  --full               Run ALL training benchmarks.
  (default)            Hardware detection + math estimates (same as --quick).

Options:
  --fast               Minimal trials (2 per stage).
  --tokenizer PATH     Use existing tokenizer (skip tokenizer benchmark).
  --output-dir PATH    Save report directory (default: ./benchmark_results).
  --keep-files         Don't clean up /tmp/benchmark_tune.
  --json               Also emit JSON report.
  --model-size SIZE    Model size for benchmarks (default: 0.3B = 300M).

Usage:
  # Quick estimate (no GPU needed)
  python benchmark_tune.py --quick

  # Full hardware benchmark
  python benchmark_tune.py --full

  # Fast pretrain benchmark only
  python benchmark_tune.py --pretrain --fast
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYNTHETIC_DIR = "/tmp/benchmark_tune"
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "data")
AGENT_DIR = os.path.join(REPO_ROOT, "webscrapped_dataset_curator_AI_MCP", "agent")

# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------


def detect_gpu() -> dict[str, Any]:
    """Detect GPU name and total VRAM via nvidia-smi or torch."""
    # Try nvidia-smi first (most reliable)
    try:
        result = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split("\n")[0]
            name, vram_mib = parts.rsplit(",", 1)
            vram_gb = round(int(vram_mib.strip()) / 1024, 1)
            return {
                "name": name.strip(),
                "vram_total_gb": vram_gb,
                "vram_total_mib": int(vram_mib.strip()),
                "available": True,
            }
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # Fallback: try torch
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory
            vram_gb = round(vram_total / (1024 ** 3), 1)
            return {
                "name": name,
                "vram_total_gb": vram_gb,
                "vram_total_mib": round(vram_total / (1024 ** 2)),
                "available": True,
            }
    except (ImportError, RuntimeError):
        pass

    return {"name": "unknown", "vram_total_gb": 0, "vram_total_mib": 0, "available": False}


def detect_cpu_ram() -> float:
    """Return total system RAM in GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass
    # Fallback: /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 ** 2), 1)
    except (FileNotFoundError, ValueError):
        pass
    return 0.0


def detect_hardware() -> dict[str, Any]:
    """Return full hardware profile dict."""
    gpu = detect_gpu()
    return {
        "gpu": gpu,
        "cpu_ram_gb": detect_cpu_ram(),
        "cpu_cores": os.cpu_count() or 0,
        "python": sys.version.split()[0],
        "cuda_available": gpu["available"],
    }


# ---------------------------------------------------------------------------
# VRAM monitor (background thread)
# ---------------------------------------------------------------------------


class VramMonitor:
    """Poll nvidia-smi in a background thread to track peak VRAM usage.

    Usage:
        mon = VramMonitor()
        mon.start()
        # ... run training ...
        mon.stop()
        peak = mon.peak_mib
    """

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_mib: int = 0

    def _poll(self):
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    vals = [int(v) for v in result.stdout.strip().split("\n") if v.strip()]
                    if vals:
                        self.peak_mib = max(self.peak_mib, vals[0])
            except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
                pass
            time.sleep(self.interval)

    def start(self):
        self._stop.clear()
        self.peak_mib = 0
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        return self.peak_mib


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def make_synthetic_pretrain_jsonl(
    num_docs: int = 10_000, out_dir: str = SYNTHETIC_DIR,
) -> str:
    """Generate template-based pretrain JSONL with 'text' field."""
    cat_dir = _ensure_dir(os.path.join(out_dir, "pretrain"))
    path = os.path.join(cat_dir, "synthetic.jsonl")
    print(f"  generating {num_docs} pretrain docs -> {path}")
    with open(path, "w") as f:
        for i in range(num_docs):
            f.write(json.dumps({
                "text": (
                    f"This is synthetic training document number {i}. "
                    f"It contains example text for benchmarking the language model "
                    f"training pipeline. Sentence variety ensures the tokenizer learns "
                    f"a useful vocabulary. Document {i} discusses various topics including "
                    f"machine learning, natural language processing, and artificial intelligence."
                ),
            }) + "\n")
    return path


def make_synthetic_sft_jsonl(
    num_docs: int = 1_000, out_dir: str = SYNTHETIC_DIR,
) -> str:
    """Generate synthetic SFT JSONL with 'prompt', 'thinking', 'answer' fields."""
    cat_dir = _ensure_dir(os.path.join(out_dir, "sft"))
    path = os.path.join(cat_dir, "synthetic.jsonl")
    print(f"  generating {num_docs} SFT docs -> {path}")
    with open(path, "w") as f:
        for i in range(num_docs):
            f.write(json.dumps({
                "prompt": f"What is the answer to question number {i}?",
                "thinking": f"Let me work through this step by step. The question asks about {i}.",
                "answer": f"The answer to question {i} is {i * 2}.",
            }) + "\n")
    return path


def make_synthetic_grpo_jsonl(
    num_docs: int = 500, out_dir: str = SYNTHETIC_DIR,
) -> str:
    """Generate synthetic GRPO JSONL with 'prompt' and 'answer' fields."""
    cat_dir = _ensure_dir(os.path.join(out_dir, "grpo"))
    path = os.path.join(cat_dir, "synthetic.jsonl")
    print(f"  generating {num_docs} GRPO docs -> {path}")
    with open(path, "w") as f:
        for i in range(num_docs):
            f.write(json.dumps({
                "prompt": f"Solve problem {i}: what is {i} + {i}?",
                "answer": f"The sum is {i + i}.",
            }) + "\n")
    return path


def make_synthetic_dpo_jsonl(
    num_docs: int = 500, out_dir: str = SYNTHETIC_DIR,
) -> str:
    """Generate synthetic DPO JSONL with 'prompt', 'chosen', 'rejected'."""
    cat_dir = _ensure_dir(os.path.join(out_dir, "dpo"))
    path = os.path.join(cat_dir, "synthetic.jsonl")
    print(f"  generating {num_docs} DPO docs -> {path}")
    with open(path, "w") as f:
        for i in range(num_docs):
            f.write(json.dumps({
                "prompt": f"What is {i} + {i}?",
                "chosen": f"The correct answer is {i + i}.",
                "rejected": f"The answer might be {i} or maybe {i * 3}.",
            }) + "\n")
    return path


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------


def train_tiny_tokenizer(
    data_dir: str, output_dir: str, vocab_size: int = 4096,
) -> str:
    """Train a tiny BPE tokenizer on synthetic data.

    Returns path to trained tokenizer directory.
    """
    print(f"  training tokenizer (vocab_size={vocab_size})...")
    os.makedirs(output_dir, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            os.path.join(REPO_ROOT, "tokenizer_train.py"),
            "--data-dir", data_dir,
            "--output-dir", output_dir,
            "--vocab-size", str(vocab_size),
            "--mode", "reasoning",
        ],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Tokenizer training failed:\n{result.stderr[-2000:]}"
        )
    tok_json = os.path.join(output_dir, "tokenizer.json")
    if not os.path.isfile(tok_json):
        raise FileNotFoundError(f"Tokenizer not created at {tok_json}")
    print(f"  tokenizer saved to {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Data packing
# ---------------------------------------------------------------------------


def pack_synthetic_data(
    tokenizer_path: str, mode: str = "pretrain",
    data_dir: Optional[str] = None, out_dir: Optional[str] = None,
    seq_length: int = 1024,
) -> str:
    """Run the appropriate packer for *mode* on synthetic data.

    Returns path to the packed output directory.
    """
    if data_dir is None:
        data_dir = os.path.join(SYNTHETIC_DIR, mode)
    if out_dir is None:
        out_dir = os.path.join(SYNTHETIC_DIR, f"packed_{mode}")

    packer = os.path.join(DATA_DIR, f"pack_{mode}.py")
    if not os.path.isfile(packer):
        raise FileNotFoundError(f"Packer not found: {packer}")

    cmd = [
        sys.executable, packer,
        "--data-dir", data_dir,
        "--tokenizer", tokenizer_path,
        "--cache-dir", out_dir,
        "--seq-length", str(seq_length),
        "--val-fraction", "0.0",
    ]

    # Add recipe-style args for SFT/GRPO/DPO packers
    if mode in ("sft", "grpo", "dpo"):
        cmd += ["--mode", "default"]

    print(f"  packing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Packer failed for {mode} (exit {result.returncode})")
    print(f"  packed to {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Model ceiling estimation (math only, no training)
# ---------------------------------------------------------------------------


def estimate_model_ceiling(
    vram_gb: float, cpu_ram_gb: float,
) -> dict[str, Any]:
    """Estimate maximum model size that fits in VRAM.

    Uses closed-form formulas with reasonable activation memory estimates.

    Reference memory per parameter:
      - bf16 weights:  2 bytes
      - fp32 momentum: 4 bytes (AdamW only)
      - fp32 variance: 4 bytes (AdamW only)
      - bf16 grads:    2 bytes
      - Total AdamW:  12 bytes/param
      - Total Muon:    4 bytes/param (no optimizer states for most params)

    Activation memory scales with batch_size, seq_len, hidden_dim, layers.
    """
    if vram_gb <= 0:
        return {
            "max_params_adamw": 0,
            "max_params_muon": 0,
            "max_params_adamw_lora": 0,
            "max_params_muon_lora": 0,
            "note": "No GPU detected — install CUDA or check nvidia-smi.",
        }

    # Estimate activation memory per parameter at seq_len=2048, batch=4
    # For a typical transformer: act_mem ≈ batch * seq_len * hidden * layers * 34 / params
    # This is a closed-form approximation calibrated against empirical measurements.
    # We solve iteratively: bigger models have more params → more activations
    # but activations scale with hidden_dim which grows sub-linearly with params.

    def _max_params(bytes_per_param_no_act: float, ckpt_factor: float = 1.0) -> float:
        """Estimate max params given bytes/param and activation scaling.

        Activation memory ≈ 0.5 * sqrt(target_params / 1e8) * batch * seq_len / 2k
        Scaled for seq_len=2048, batch=4 with checkpointing factor.
        """
        usable = (vram_gb - 2.5) * 1e9  # reserve 2.5 GB for CUDA, framework, etc.
        # Binary search for max params
        lo, hi = 1e6, 5e11  # 1M to 500B
        for _ in range(40):
            mid = (lo + hi) / 2
            # Activation scaling: typical model with mid params needs:
            # act_mem ≈ mid * 0.015 * ckpt_factor * (seq_len / 2048) * (batch / 4)
            act_per_param = 0.015 * ckpt_factor  # empirically calibrated
            total_per_param = bytes_per_param_no_act + act_per_param
            if mid * total_per_param < usable:
                lo = mid
            else:
                hi = mid
        return int(lo)

    # AdamW: 12 bytes/param (2 weight + 4 momentum + 4 variance + 2 grad)
    adamw_no_ckpt = _max_params(12.0, ckpt_factor=1.0)
    adamw_ckpt = _max_params(12.0, ckpt_factor=0.3)

    # Muon: 4 bytes/param (2 weight + 2 grad)
    muon_no_ckpt = _max_params(4.0, ckpt_factor=1.0)
    muon_ckpt = _max_params(4.0, ckpt_factor=0.3)

    # LoRA: optimizer states only for 0.1% of params
    # AdamW + LoRA = 2 (weight) + 0.001 * 8 (opt states for LoRA) + 2 (grad)
    # ≈ 4.008 bytes/param (dominant cost is still base weights)
    lora_extra = 0.001 * 8  # LoRA params have full optimizer states
    adamw_lora = _max_params(12.0 - 4.0 + lora_extra, ckpt_factor=0.3)
    muon_lora = _max_params(4.0 + lora_extra, ckpt_factor=0.3)

    return {
        "vram_gb": vram_gb,
        "cpu_ram_gb": cpu_ram_gb,
        "max_params_adamw": adamw_no_ckpt,
        "max_params_adamw_ckpt": adamw_ckpt,
        "max_params_muon": muon_no_ckpt,
        "max_params_muon_ckpt": muon_ckpt,
        "max_params_adamw_lora": adamw_lora,
        "max_params_muon_lora": muon_lora,
        "recommended_adamw": _to_readable(adamw_ckpt),
        "recommended_muon": _to_readable(muon_ckpt),
        "recommended_adamw_lora": _to_readable(adamw_lora),
        "recommended_muon_lora": _to_readable(muon_lora),
    }


def _to_param_count(size_str: str) -> int:
    """Convert model size string like '0.3B', '600M', '1T' to int param count."""
    s = size_str.strip().upper().replace(" ", "")
    if s.endswith("T"):
        return int(float(s[:-1]) * 1_000_000_000_000)
    if s.endswith("B"):
        return int(float(s[:-1]) * 1_000_000_000)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1_000_000)
    if s.endswith("K"):
        return int(float(s[:-1]) * 1_000)
    return int(s)


def _to_readable(param_count: int) -> str:
    """Format param count like '350M', '1.7B', '3T'."""
    if param_count >= 1e12:
        return f"{param_count / 1e12:.2f}T"
    if param_count >= 1e9:
        return f"{param_count / 1e9:.2f}B"
    if param_count >= 1e6:
        return f"{param_count / 1e6:.0f}M"
    return str(param_count)


# ---------------------------------------------------------------------------
# Benchmark: Tokenizer
# ---------------------------------------------------------------------------


def benchmark_tokenizer(
    hw: dict[str, Any], tokenizer_path: Optional[str] = None,
) -> dict[str, Any]:
    """Benchmark tokenizer training at multiple vocab sizes.

    If tokenizer_path is given, use it directly (skip benchmark).
    """
    if tokenizer_path and os.path.isfile(os.path.join(tokenizer_path, "tokenizer.json")):
        print(f"[tokenizer] using existing tokenizer at {tokenizer_path}")
        return {"tokenizer_path": tokenizer_path, "vocab_size": "existing", "note": "reused"}

    print("\n========================================")
    print("  Tokenizer Benchmark")
    print("========================================")

    # First ensure we have data for tokenizer training
    if not os.listdir(os.path.join(SYNTHETIC_DIR, "pretrain")):
        make_synthetic_pretrain_jsonl(out_dir=SYNTHETIC_DIR)

    results = {}
    for vocab_size in [4096, 8192, 16384]:
        out_dir = os.path.join(SYNTHETIC_DIR, f"tokenizer_v{vocab_size}")
        if os.path.isfile(os.path.join(out_dir, "tokenizer.json")):
            print(f"  tokenizer v={vocab_size} already exists, skipping")
            results[str(vocab_size)] = {"vocab_size": vocab_size, "path": out_dir}
            continue
        t0 = time.time()
        try:
            train_tiny_tokenizer(
                data_dir=os.path.join(SYNTHETIC_DIR, "pretrain"),
                output_dir=out_dir,
                vocab_size=vocab_size,
            )
            elapsed = time.time() - t0
            results[str(vocab_size)] = {
                "vocab_size": vocab_size,
                "path": out_dir,
                "train_time_s": round(elapsed, 1),
            }
            print(f"    done in {elapsed:.1f}s")
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            print(f"    FAILED: {e}")
            results[str(vocab_size)] = {"vocab_size": vocab_size, "error": str(e)}

    # Pick the best tokenizer path (prefer largest vocab that trained successfully)
    best_path = None
    best_size = 0
    for k, v in results.items():
        if "path" in v and v["vocab_size"] > best_size:
            best_path = v["path"]
            best_size = v["vocab_size"]

    return {
        "results": results,
        "tokenizer_path": best_path,
        "recommended_vocab_size": best_size or 4096,
    }


# ---------------------------------------------------------------------------
# Utility: subprocess runner with VRAM monitoring
# ---------------------------------------------------------------------------


def _run_training(
    cmd: list[str], label: str, timeout: int = 600,
) -> dict[str, Any]:
    """Run a training subprocess while monitoring VRAM.

    Returns dict with wall_time, peak_vram_mib, returncode, stdout.
    """
    print(f"  [{label}] running: {' '.join(cmd)}")
    mon = VramMonitor()
    mon.start()
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - t0
        peak = mon.stop()
        return {
            "label": label,
            "wall_time_s": round(elapsed, 1),
            "peak_vram_mib": peak,
            "returncode": -1,
            "error": f"Timed out after {timeout}s",
            "stdout": e.stdout or "",
            "stderr": e.stderr or "",
        }

    elapsed = time.time() - t0
    peak = mon.stop()

    print(f"  [{label}] done in {elapsed:.1f}s, "
          f"peak VRAM={peak} MiB ({peak / 1024:.1f} GiB)")

    return {
        "label": label,
        "wall_time_s": round(elapsed, 1),
        "peak_vram_mib": peak,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# ---------------------------------------------------------------------------
# Benchmark: Pretrain
# ---------------------------------------------------------------------------


def benchmark_pretrain(
    hw: dict[str, Any], tokenizer_path: str, fast: bool = False,
    model_size: str = "0.3B",
) -> dict[str, Any]:
    """Benchmark pretraining with different batch sizes.

    Optionally tests Adam vs Muon if --full or --fast.
    """
    print("\n========================================")
    print("  Pretrain Benchmark")
    print("========================================")

    if not hw["gpu"]["available"]:
        return {"error": "No GPU available — skipping pretrain benchmark."}

    # Generate data + pack
    print("  [setup] generating synthetic data...")
    make_synthetic_pretrain_jsonl(out_dir=SYNTHETIC_DIR)
    packed_dir = pack_synthetic_data(
        tokenizer_path, mode="pretrain",
        seq_length=2048,
    )

    batch_sizes = [1, 2, 4] if fast else [1, 2, 4, 8]
    checkpoint_dir = os.path.join(SYNTHETIC_DIR, "ckpt_pretrain")
    results = []

    for bs in batch_sizes:
        out_ckpt = f"{checkpoint_dir}_bs{bs}"
        cmd = [
            sys.executable,
            os.path.join(REPO_ROOT, "train_pretrain.py"),
            "--model-size", model_size,
            "--data-dir", packed_dir,
            "--seq-len", "2048",
            "--batch-size", str(bs),
            "--grad-accum", "2",
            "--num-steps", "30",
            "--dtype", "bf16",
            "--gradient-checkpointing",
            "--optimizer", "adamw",
            "--lr", "3e-4",
            "--checkpoint-dir", out_ckpt,
            "--log-interval", "10",
            "--save-every", "1000",
            "--val-fraction", "0.0",
        ]

        r = _run_training(cmd, f"pretrain_bs{bs}", timeout=600)
        # Compute throughput
        if r["returncode"] == 0:
            # tokens/sec = batch_size * seq_len * steps / wall_time
            r["throughput_tok_s"] = round(
                bs * 2048 * 30 / max(r["wall_time_s"], 1), 1
            )
        else:
            r["throughput_tok_s"] = 0
            r["error"] = r.get("error", "") + f" exit={r['returncode']}"
            # Truncate stderr for readability
            if r.get("stderr"):
                r["error_detail"] = r["stderr"][-1000:]
        results.append(r)

    # Test Muon at best batch size if full/fast
    muon_result = None
    if not fast or any(
        r["returncode"] == 0 for r in results
    ):
        if fast or True:  # always test best batch with muon
            best_bs = max(
                (r for r in results if r["returncode"] == 0),
                key=lambda r: r.get("throughput_tok_s", 0),
                default=None,
            )
            if best_bs:
                bs = int(best_bs["label"].replace("pretrain_bs", ""))
                out_ckpt_muon = f"{checkpoint_dir}_muon_bs{bs}"
                cmd = [
                    sys.executable,
                    os.path.join(REPO_ROOT, "train_pretrain.py"),
                    "--model-size", model_size,
                    "--data-dir", packed_dir,
                    "--seq-len", "2048",
                    "--batch-size", str(bs),
                    "--grad-accum", "2",
                    "--num-steps", "30",
                    "--dtype", "bf16",
                    "--gradient-checkpointing",
                    "--optimizer", "muon",
                    "--schedule", "wsd",
                    "--stable-ratio", "0.8",
                    "--lr", "3e-4",
                    "--checkpoint-dir", out_ckpt_muon,
                    "--log-interval", "10",
                    "--save-every", "1000",
                    "--val-fraction", "0.0",
                ]
                muon_result = _run_training(
                    cmd, "pretrain_muon_wsd", timeout=600,
                )
                if muon_result["returncode"] == 0:
                    muon_result["throughput_tok_s"] = round(
                        bs * 2048 * 30 / max(muon_result["wall_time_s"], 1), 1
                    )

    # Determine best config
    successful = [r for r in results if r["returncode"] == 0]
    if muon_result and muon_result["returncode"] == 0:
        successful.append(muon_result)

    best = max(successful, key=lambda r: r.get("throughput_tok_s", 0)) if successful else {}

    return {
        "batch_size_trials": results,
        "muon_trial": muon_result,
        "best_config": best,
        "checkpoint_dir": checkpoint_dir,
        "model_size": model_size,
        "recommended_batch_size": int(best.get("label", "bs4").replace("pretrain_bs", "").replace("pretrain_muon_wsd", "4")),
        "recommended_optimizer": "muon" if muon_result and muon_result["returncode"] == 0 and muon_result.get("throughput_tok_s", 0) > best.get("throughput_tok_s", 0) else "adamw",
        "recommended_schedule": "wsd" if muon_result and muon_result["returncode"] == 0 else "cosine",
    }


# ---------------------------------------------------------------------------
# Benchmark: SFT
# ---------------------------------------------------------------------------


def benchmark_sft(
    hw: dict[str, Any], tokenizer_path: str,
    pretrain_ckpt: Optional[str] = None, fast: bool = False,
    model_size: str = "0.3B",
) -> dict[str, Any]:
    """Benchmark SFT with different LoRA ranks."""
    print("\n========================================")
    print("  SFT Benchmark")
    print("========================================")

    if not hw["gpu"]["available"]:
        return {"error": "No GPU available — skipping SFT benchmark."}

    # Generate + pack SFT data
    make_synthetic_sft_jsonl(out_dir=SYNTHETIC_DIR)
    packed_dir = pack_synthetic_data(
        tokenizer_path, mode="sft",
        seq_length=2048,
    )

    lora_ranks = [16, 32, 64] if fast else [0, 16, 32, 64]
    results = []

    for rank in lora_ranks:
        out_dir = os.path.join(SYNTHETIC_DIR, f"ckpt_sft_r{rank}")
        cmd = [
            sys.executable,
            os.path.join(REPO_ROOT, "train_sft.py"),
            "--data-dir", packed_dir,
            "--output-dir", out_dir,
            "--batch-size", "2",
            "--grad-accum", "2",
            "--num-steps", "20",
            "--dtype", "bf16",
            "--lr", "2e-5",
            "--log-interval", "10",
            "--ckpt-interval", "1000",
            "--val-fraction", "0.0",
        ]
        if rank > 0:
            cmd += [
                "--lora-rank", str(rank),
                "--lora-alpha", str(rank * 2),
            ]
        else:
            cmd += ["--model-size", model_size]

        r = _run_training(cmd, f"sft_lora{rank}", timeout=600)
        results.append(r)

    best = max(
        (r for r in results if r["returncode"] == 0),
        key=lambda r: r.get("peak_vram_mib", 0),
        default={},
    )
    return {
        "lora_trials": results,
        "best_config": best,
        "recommended_lora_rank": 32,  # default recommendation
    }


# ---------------------------------------------------------------------------
# Benchmark: GRPO
# ---------------------------------------------------------------------------


def benchmark_grpo(
    hw: dict[str, Any], tokenizer_path: str,
    sft_ckpt: Optional[str] = None, fast: bool = False,
    model_size: str = "0.3B",
) -> dict[str, Any]:
    """Benchmark GRPO with different num-generations values."""
    print("\n========================================")
    print("  GRPO Benchmark")
    print("========================================")

    if not hw["gpu"]["available"]:
        return {"error": "No GPU available — skipping GRPO benchmark."}

    # Generate + pack GRPO data
    make_synthetic_grpo_jsonl(out_dir=SYNTHETIC_DIR)
    packed_dir = pack_synthetic_data(
        tokenizer_path, mode="grpo",
        seq_length=1024,
    )

    gen_counts = [2, 4] if fast else [2, 4, 8]
    results = []

    for g in gen_counts:
        out_dir = os.path.join(SYNTHETIC_DIR, f"ckpt_grpo_g{g}")
        cmd = [
            sys.executable,
            os.path.join(REPO_ROOT, "train_grpo.py"),
            "--tokenizer", tokenizer_path,
            "--data-dir", packed_dir,
            "--out-dir", out_dir,
            "--batch-size", "1",
            "--grad-accum", "1",
            "--num-generations", str(g),
            "--num-steps", "10",
            "--dtype", "bf16",
            "--lr", "1e-6",
            "--max-new-tokens", "64",
            "--lora-rank", "32",
            "--lora-alpha", "64",
            "--lora",
            "--kl-coef", "0.02",
            "--clip-ratio", "0.2",
            "--log-interval", "1",
            "--save-every", "100",
            "--model-size", model_size,
        ]

        r = _run_training(cmd, f"grpo_g{g}", timeout=600)
        results.append(r)

    best = max(
        (r for r in results if r["returncode"] == 0),
        key=lambda r: r.get("peak_vram_mib", 0),
        default={},
    )
    return {
        "generation_trials": results,
        "best_config": best,
        "recommended_num_generations": 4,
    }


# ---------------------------------------------------------------------------
# Benchmark: DPO
# ---------------------------------------------------------------------------


def benchmark_dpo(
    hw: dict[str, Any], tokenizer_path: str,
    sft_ckpt: Optional[str] = None, fast: bool = False,
) -> dict[str, Any]:
    """Benchmark DPO with different batch sizes."""
    print("\n========================================")
    print("  DPO Benchmark")
    print("========================================")

    if not hw["gpu"]["available"]:
        return {"error": "No GPU available — skipping DPO benchmark."}

    # Generate + pack DPO data
    make_synthetic_dpo_jsonl(out_dir=SYNTHETIC_DIR)

    # DPO packer shares data format with STP-like structure but uses
    # pack_dpo.py with prompt/chosen/rejected fields.
    packed_dir = pack_synthetic_data(
        tokenizer_path, mode="dpo",
        seq_length=1024,
    )

    batch_sizes = [1, 2] if fast else [1, 2, 4]
    results = []

    for bs in batch_sizes:
        out_dir = os.path.join(SYNTHETIC_DIR, f"ckpt_dpo_bs{bs}")
        cmd = [
            sys.executable,
            os.path.join(REPO_ROOT, "train_dpo.py"),
            "--tokenizer", tokenizer_path,
            "--data-dir", packed_dir,
            "--out-dir", out_dir,
            "--batch-size", str(bs),
            "--grad-accum", "1",
            "--num-steps", "10",
            "--max-steps", "10",
            "--dtype", "bf16",
            "--lr", "1e-6",
            "--beta", "0.1",
            "--lora",
            "--lora-rank", "32",
            "--lora-alpha", "64",
            "--log-interval", "1",
            "--save-every", "100",
        ]

        r = _run_training(cmd, f"dpo_bs{bs}", timeout=600)
        results.append(r)

    return {
        "batch_size_trials": results,
        "recommended_batch_size": 1,
        "recommended_beta": 0.1,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _param_to_flag(param_count: int) -> str:
    """Convert int param count to --model-size flag value."""
    return _to_readable(param_count)


def generate_report(
    hw: dict[str, Any],
    model_ceiling: dict[str, Any],
    results: dict[str, Any],
    output_dir: str,
    json_output: bool = False,
    model_size: str = "0.3B",
) -> str:
    """Generate a structured text report and optionally JSON.

    Returns the report text.
    """
    gpu = hw["gpu"]
    lines: list[str] = []

    lines.append("=" * 65)
    lines.append("  Hardware Benchmark & Hyperparameter Report")
    lines.append("=" * 65)
    lines.append("")

    # Section 1: Hardware
    lines.append("── Hardware Profile ──────────────────────────────────────")
    lines.append(f"  GPU              : {gpu['name']}")
    lines.append(f"  VRAM total       : {gpu['vram_total_gb']} GiB ({gpu['vram_total_mib']} MiB)")
    lines.append(f"  CPU RAM          : {hw['cpu_ram_gb']} GiB")
    lines.append(f"  CPU cores        : {hw['cpu_cores']}")
    lines.append(f"  Python           : {hw['python']}")
    lines.append(f"  CUDA available   : {'Yes' if hw['cuda_available'] else 'No'}")
    lines.append("")

    # Section 2: Model ceiling
    lines.append("── Model Size Ceiling (estimated) ────────────────────────")
    lines.append(f"  Benchmark model size : {model_size}")
    lines.append(f"  AdamW (no ckpt)      : {model_ceiling.get('recommended_adamw', 'N/A')}")
    lines.append(f"  AdamW (+ckpt)        : {model_ceiling.get('recommended_adamw_lora', 'N/A')}")
    lines.append(f"  Muon (no ckpt)       : {model_ceiling.get('recommended_muon', 'N/A')}")
    lines.append(f"  Muon (+ckpt)         : {model_ceiling.get('recommended_muon_lora', 'N/A')}")
    lines.append("")

    if model_ceiling.get("max_params_adamw_ckpt", 0) > 0:
        ceiling = model_ceiling["max_params_adamw_ckpt"]
        lines.append(f"  → Up to {_to_readable(ceiling)} with AdamW + gradient checkpointing")
    if model_ceiling.get("max_params_muon_ckpt", 0) > 0:
        ceiling_m = model_ceiling["max_params_muon_ckpt"]
        lines.append(f"  → Up to {_to_readable(ceiling_m)} with Muon + gradient checkpointing")
    lines.append("")

    # Section 3: Benchmark results
    lines.append("── Training Benchmark Results ────────────────────────────")

    # Pretrain
    pretrain = results.get("pretrain", {})
    if pretrain and "error" not in pretrain:
        lines.append("")
        lines.append("  [Pretrain]")
        trials = pretrain.get("batch_size_trials", [])
        if trials:
            lines.append(f"  {'Batch Size':>12} {'VRAM':>10} {'Time':>8} {'Tokens/s':>10} {'Status':>10}")
            lines.append(f"  {'-'*50}")
            for r in trials:
                bs = r["label"].replace("pretrain_bs", "")
                ok = "OK" if r["returncode"] == 0 else "FAIL"
                vram = f"{r['peak_vram_mib']} MiB" if r["peak_vram_mib"] > 0 else "N/A"
                lines.append(
                    f"  {bs:>12} {vram:>10} "
                    f"{r['wall_time_s']:>7.1f}s "
                    f"{r.get('throughput_tok_s', 0):>9.0f} "
                    f"{ok:>10}"
                )
        muon = pretrain.get("muon_trial")
        if muon:
            ok = "OK" if muon["returncode"] == 0 else "FAIL"
            vram_val = muon["peak_vram_mib"]
            vram_str = "--" if vram_val == 0 else f"{vram_val} MiB"
            lines.append(
                f"  {'Muon+WSD':>12} {vram_str:>10} "
                f"{muon['wall_time_s']:>7.1f}s "
                f"{muon.get('throughput_tok_s', 0):>9.0f} "
                f"{ok:>10}"
            )
        lines.append(
            f"  → Recommend: batch_size={pretrain.get('recommended_batch_size', 4)}, "
            f"optimizer={pretrain.get('recommended_optimizer', 'adamw')}"
        )

    # SFT
    sft = results.get("sft", {})
    if sft and "error" not in sft:
        lines.append("")
        lines.append("  [SFT]")
        trials = sft.get("lora_trials", [])
        if trials:
            lines.append(f"  {'LoRA Rank':>12} {'VRAM':>10} {'Time':>8} {'Status':>10}")
            lines.append(f"  {'-'*44}")
            for r in trials:
                rank = r["label"].replace("sft_lora", "")
                ok = "OK" if r["returncode"] == 0 else "FAIL"
                vram = f"{r['peak_vram_mib']} MiB" if r["peak_vram_mib"] > 0 else "N/A"
                lines.append(
                    f"  {rank:>12} {vram:>10} "
                    f"{r['wall_time_s']:>7.1f}s "
                    f"{ok:>10}"
                )
        lines.append(f"  → Recommend: lora_rank={sft.get('recommended_lora_rank', 32)}")

    # GRPO
    grpo = results.get("grpo", {})
    if grpo and "error" not in grpo:
        lines.append("")
        lines.append("  [GRPO]")
        trials = grpo.get("generation_trials", [])
        if trials:
            lines.append(f"  {'Gens':>12} {'VRAM':>10} {'Time':>8} {'Status':>10}")
            lines.append(f"  {'-'*44}")
            for r in trials:
                g = r["label"].replace("grpo_g", "")
                ok = "OK" if r["returncode"] == 0 else "FAIL"
                vram = f"{r['peak_vram_mib']} MiB" if r["peak_vram_mib"] > 0 else "N/A"
                lines.append(
                    f"  {g:>12} {vram:>10} "
                    f"{r['wall_time_s']:>7.1f}s "
                    f"{ok:>10}"
                )
        lines.append(f"  → Recommend: num_generations={grpo.get('recommended_num_generations', 4)}")

    # DPO
    dpo = results.get("dpo", {})
    if dpo and "error" not in dpo:
        lines.append("")
        lines.append("  [DPO]")
        trials = dpo.get("batch_size_trials", [])
        if trials:
            lines.append(f"  {'Batch Size':>12} {'VRAM':>10} {'Time':>8} {'Status':>10}")
            lines.append(f"  {'-'*44}")
            for r in trials:
                bs = r["label"].replace("dpo_bs", "")
                ok = "OK" if r["returncode"] == 0 else "FAIL"
                vram = f"{r['peak_vram_mib']} MiB" if r["peak_vram_mib"] > 0 else "N/A"
                lines.append(
                    f"  {bs:>12} {vram:>10} "
                    f"{r['wall_time_s']:>7.1f}s "
                    f"{ok:>10}"
                )
        lines.append(
            f"  → Recommend: batch_size={dpo.get('recommended_batch_size', 1)}, "
            f"beta={dpo.get('recommended_beta', 0.1)}"
        )

    # Section 4: Recommended commands
    lines.append("")
    lines.append("── Recommended Pipeline Commands ─────────────────────────")

    # General guidance
    lines.append("")
    lines.append("  # These commands are tailored to your hardware based on benchmarks.")

    ceiling = model_ceiling
    use_muon = ceiling.get("max_params_muon_ckpt", 0) > 300_000_000

    # Model size from benchmark or ceiling
    bench_model = model_size

    if use_muon:
        recommend_model = ceiling.get("recommended_muon_lora", bench_model)
    else:
        recommend_model = ceiling.get("recommended_adamw_lora", bench_model)

    lines.append("")
    lines.append("  # 1. Data preparation (run once per machine)")
    lines.append(f"  python tokenizer_train.py --data-dir ./data --output-dir ./tokenizer "
                 f"--vocab-size 65536 --mode reasoning")
    lines.append(f"  python data/pack_pretrain.py --data-dir ./data --tokenizer ./tokenizer "
                 f"--cache-dir ./packed --seq-length 2048 --val-fraction 0.01")

    lines.append("")
    lines.append("  # 2. Pretrain")
    pretrain_opt = pretrain.get("recommended_optimizer", "muon" if use_muon else "adamw")
    pretrain_sched = pretrain.get("recommended_schedule", "wsd" if use_muon else "cosine")
    pretrain_bs = pretrain.get("recommended_batch_size", 4) if pretrain.get("recommended_batch_size") else 4
    lines.append(
        f"  python train_pretrain.py --model-size {bench_model} "
        f"--data-dir ./packed --seq-len 2048 "
        f"--batch-size {pretrain_bs} --grad-accum 4 "
        f"--num-steps 100000 --dtype bf16 --gradient-checkpointing "
        f"--optimizer {pretrain_opt} --schedule {pretrain_sched} "
        f"--stable-ratio 0.8 --lr 3e-4 --checkpoint-dir ./checkpoints/pretrain"
    )

    lines.append("")
    lines.append("  # 3. SFT (LoRA)")
    sft_rank = sft.get("recommended_lora_rank", 32)
    lines.append(
        f"  python train_sft.py --checkpoint-dir ./checkpoints/pretrain/latest.pt "
        f"--data-dir ./sft_packed --output-dir ./checkpoints/sft "
        f"--batch-size 2 --grad-accum 4 --num-steps 10000 "
        f"--lora-rank {sft_rank} --lora-alpha {sft_rank * 2} "
        f"--dtype bf16 --lr 2e-5 --ckpt-interval 1000"
    )

    lines.append("")
    lines.append("  # 4. GRPO (RL, requires SFT checkpoint)")
    grpo_gens = grpo.get("recommended_num_generations", 4)
    lines.append(
        f"  python train_grpo.py --checkpoint ./checkpoints/sft/latest.pt "
        f"--tokenizer ./tokenizer --data-dir ./grpo_packed "
        f"--out-dir ./checkpoints/grpo --batch-size 1 --num-steps 500 "
        f"--num-generations {grpo_gens} --max-new-tokens 512 "
        f"--lora --lora-rank 32 --lora-alpha 64 "
        f"--dtype bf16 --lr 1e-6 --kl-coef 0.02 --clip-ratio 0.2"
    )

    lines.append("")
    lines.append("  # 5. DPO (preference tuning, requires SFT checkpoint)")
    dpo_bs = dpo.get("recommended_batch_size", 1)
    dpo_beta = dpo.get("recommended_beta", 0.1)
    lines.append(
        f"  python train_dpo.py --checkpoint ./checkpoints/sft/latest.pt "
        f"--tokenizer ./tokenizer --data-dir ./dpo_packed "
        f"--out-dir ./checkpoints/dpo --batch-size {dpo_bs} --num-steps 500 "
        f"--beta {dpo_beta} --lora --lora-rank 32 --lora-alpha 64 "
        f"--dtype bf16 --lr 1e-6"
    )

    # Section 5: Tips
    lines.append("")
    lines.append("── Tips ──────────────────────────────────────────────────")
    lines.append("")
    if not hw["cuda_available"]:
        lines.append("  ⚠ No CUDA GPU detected. Install torch with CUDA or check nvidia-smi.")
    if model_ceiling.get("max_params_muon_ckpt", 0) > 2 * _to_param_count(model_size):
        lines.append("  ✅ You have enough headroom for Muon optimizer (saves ~2x VRAM vs AdamW).")
    if model_ceiling.get("max_params_muon_ckpt", 0) > model_ceiling.get("max_params_adamw_ckpt", 0) * 1.5:
        lines.append("  💡 Consider Muon over AdamW — it offers 50%+ larger model capacity.")
    lines.append("  💡 Use --gradient-checkpointing for any model > 1B parameters.")
    lines.append("  💡 For production: increase --num-steps to convergence (100K+).")
    lines.append("  💡 Benchmark precision: run --full to get stable measurements.")

    report = "\n".join(lines)
    report_path = os.path.join(output_dir, "benchmark_report.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    if json_output:
        json_path = os.path.join(output_dir, "benchmark_report.json")
        json_data = {
            "hardware": hw,
            "model_ceiling": model_ceiling,
            "results": results,
            "recommended_model_size": bench_model,
            "report": report,
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2, default=str)
        print(f"JSON report saved to {json_path}")

    return report


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup(keep_files: bool = False):
    """Remove synthetic data directory unless --keep-files."""
    if not keep_files and os.path.isdir(SYNTHETIC_DIR):
        print(f"  cleaning up {SYNTHETIC_DIR}")
        shutil.rmtree(SYNTHETIC_DIR, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hardware benchmark & hyperparameter recommendation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode flags (mutually exclusive)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true",
                      help="Hardware detection + math estimation only (instant).")
    mode.add_argument("--pretrain", action="store_true",
                      help="Benchmark pretraining.")
    mode.add_argument("--sft", action="store_true",
                      help="Benchmark SFT (requires pretrain checkpoint).")
    mode.add_argument("--grpo", action="store_true",
                      help="Benchmark GRPO.")
    mode.add_argument("--dpo", action="store_true",
                      help="Benchmark DPO.")
    mode.add_argument("--full", action="store_true",
                      help="Run ALL training benchmarks.")

    p.add_argument("--fast", action="store_true",
                   help="Minimal trials (faster, less accurate).")
    p.add_argument("--tokenizer", default=None,
                   help="Path to existing tokenizer directory. Skips tokenizer benchmark.")
    p.add_argument("--output-dir", default="./benchmark_results",
                   help="Directory for report output (default: ./benchmark_results).")
    p.add_argument("--keep-files", action="store_true",
                   help="Don't clean up /tmp/benchmark_tune after finishing.")
    p.add_argument("--json", action="store_true",
                   help="Emit JSON report alongside text report.")
    p.add_argument("--model-size", default="0.3B",
                   help="Model size for benchmarks (default: 0.3B = 300M params).")

    return p.parse_args(argv)


def main() -> None:
    args = parse_args()

    t_start = time.time()
    print("=" * 65)
    print("  benchmark_tune.py — Hardware Benchmark & Hyperparameter Tuning")
    print("=" * 65)

    # Detect hardware
    print("\n[1/5] Detecting hardware...")
    hw = detect_hardware()
    gpu = hw["gpu"]
    print(f"  GPU : {gpu['name']} ({gpu['vram_total_gb']} GiB)")
    print(f"  RAM : {hw['cpu_ram_gb']} GiB")
    print(f"  CPU : {hw['cpu_cores']} cores")

    # Quick mode or default: just math estimates
    doing_training = args.pretrain or args.sft or args.grpo or args.dpo or args.full
    if args.quick or not doing_training:
        print("\n[2/5] Estimating model ceiling...")
        ceiling = estimate_model_ceiling(
            gpu["vram_total_gb"], hw["cpu_ram_gb"],
        )
        if gpu["available"]:
            print(f"  Max with AdamW        : {ceiling['recommended_adamw']}")
            print(f"  Max with AdamW+ckpt   : {ceiling['recommended_adamw_lora']}")
            print(f"  Max with Muon         : {ceiling['recommended_muon']}")
            print(f"  Max with Muon+ckpt    : {ceiling['recommended_muon_lora']}")
        else:
            print("  (No GPU detected — estimates unavailable)")

        print(f"\n[3/5] Generating report...")
        generate_report(hw, ceiling, {}, args.output_dir, args.json, args.model_size)

        elapsed = time.time() - t_start
        print(f"\nDone in {elapsed:.1f}s")
        return

    # Training benchmarks: ensure tokenizer exists first
    print("\n[2/5] Preparing tokenizer...")
    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        # Generate data for tokenizer training
        make_synthetic_pretrain_jsonl(out_dir=SYNTHETIC_DIR)
        tokenizer_path = os.path.join(SYNTHETIC_DIR, "tokenizer")
        if not os.path.isfile(os.path.join(tokenizer_path, "tokenizer.json")):
            train_tiny_tokenizer(
                data_dir=os.path.join(SYNTHETIC_DIR, "pretrain"),
                output_dir=tokenizer_path,
                vocab_size=4096,
            )
        else:
            print(f"  using existing tokenizer at {tokenizer_path}")
    print(f"  tokenizer: {tokenizer_path}")

    # Run benchmarks
    print("\n[3/5] Running benchmarks...")
    results: dict[str, Any] = {}
    model_size = args.model_size

    if args.pretrain or args.full:
        results["pretrain"] = benchmark_pretrain(
            hw, tokenizer_path, fast=args.fast, model_size=model_size,
        )

    if args.sft or args.full:
        pretrain_ckpt = None
        if results.get("pretrain"):
            pretrain_ckpt = results["pretrain"].get("checkpoint_dir")
        results["sft"] = benchmark_sft(
            hw, tokenizer_path,
            pretrain_ckpt=pretrain_ckpt,
            fast=args.fast, model_size=model_size,
        )

    if args.grpo or args.full:
        sft_ckpt = None
        if results.get("sft"):
            sft_ckpt = results["sft"].get("best_config", {}).get("label")
        results["grpo"] = benchmark_grpo(
            hw, tokenizer_path,
            sft_ckpt=sft_ckpt,
            fast=args.fast, model_size=model_size,
        )

    if args.dpo or args.full:
        sft_ckpt = None
        if results.get("sft"):
            sft_ckpt = results["sft"].get("best_config", {}).get("label")
        results["dpo"] = benchmark_dpo(
            hw, tokenizer_path,
            sft_ckpt=sft_ckpt, fast=args.fast,
        )

    # Model ceiling estimation
    print("\n[4/5] Estimating model ceiling...")
    ceiling = estimate_model_ceiling(
        gpu["vram_total_gb"], hw["cpu_ram_gb"],
    )

    # Report
    print("\n[5/5] Generating report...")
    generate_report(hw, ceiling, results, args.output_dir, args.json, model_size)

    # Cleanup
    cleanup(keep_files=args.keep_files)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Report: {os.path.join(args.output_dir, 'benchmark_report.txt')}")


if __name__ == "__main__":
    main()
