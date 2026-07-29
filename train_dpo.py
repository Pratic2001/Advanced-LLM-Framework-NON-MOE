#!/usr/bin/env python3
"""
train_dpo.py

Direct Preference Optimization (DPO) for the dense LLM framework.
Stage 2 of post-training (alternative to GRPO): consumes an SFT checkpoint
+ packed preference pairs and optimises the policy via the DPO objective.

Instead of a learned reward model, DPO directly uses preference pairs
(chosen vs rejected completions) with a closed-form loss:

    L = -E[log σ(β (log π_θ(y_w|x) - log π_ref(y_w|x)
                   - (log π_θ(y_l|x) - log π_ref(y_l|x))))]

Key features:
  - Standard DPO loss with β (beta) temperature parameter
  - Reference-model KL anchoring (log-ratio subtraction)
  - Optional PPO-style clipping on the implicit reward
  - Optional label smoothing
  - Single-model / two-model reference policy (mirrors GRPO)
  - LoRA support (reuses peft.lora module)
  - PackedDPODataLoader for preference-pair packed data
  - TrainingRecipe integration for all template decisions

Usage:
    python train_dpo.py \\
        --data-dir ./dpo_packed \\
        --checkpoint ./sft_checkpoints/latest.pt \\
        --tokenizer ./tokenizer

    python train_dpo.py --smoke-test

Reference:
    "Direct Preference Optimization" (Rafailov et al., 2023)
    https://arxiv.org/abs/2305.18290
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
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

# ---------------------------------------------------------------------------
# Hardware TFLOPS table (for throughput estimation)
# ---------------------------------------------------------------------------

GPU_PEAK_TFLOPS = {
    "A100-80G": 312, "A100-40G": 312,
    "H100": 989, "H100-PCIe": 756, "H200": 989,
    "RTX-4090": 165, "RTX-3090": 165,
    "RTX-5090": 260, "RTX-5090D": 260,
    "RTX-A6000": 155, "RTX-A6000-48gb": 155,
    "RTX-4080": 120, "RTX-4070Ti": 82,
    "L40S": 362,
    "MI250X": 383, "MI300X": 1307,
    "TPU-v4": 275, "TPU-v5e": 197,
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
    added = tokenizer.get_added_vocabulary()
    if token in added:
        return added[token]
    return tokenizer.get_vocab_size() - 1


# ---------------------------------------------------------------------------
# PackedDPODataLoader — reads dpo_prompts.bin / dpo_chosen.bin / dpo_rejected.bin
# ---------------------------------------------------------------------------
#
# Data layout:
#   dpo_prompts.bin     — flat uint16 array, length-prefixed records
#   dpo_chosen.bin      — flat uint16 array, (prompt + chosen completion) length-prefixed
#   dpo_rejected.bin    — flat uint16 array, (prompt + rejected completion) length-prefixed
#   dpo_prompt_lens.json — JSON array of prompt token lengths (for masking)
#
# Each .bin record has a 4-byte uint32 length prefix followed by that many
# uint16 token IDs. Records align 1:1 across the three .bin files.


class PackedDPODataLoader:
    """
    Streams (chosen_ids, chosen_mask, rejected_ids, rejected_mask) from
    DPO-packed data.

    chosen_ids    = prompt + chosen_completion (concatenated)
    chosen_mask   = 1 for chosen_completion positions, 0 for prompt positions
    rejected_ids  = prompt + rejected_completion (concatenated)
    rejected_mask = 1 for rejected_completion positions, 0 for prompt positions
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        split: str = "train",
    ):
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.rng = random.Random(seed + rank)

        # Discover data files
        suffix = f"_{split}"
        bin_patterns = {
            "prompt": os.path.join(data_dir, f"dpo_prompts{suffix}*.bin"),
            "chosen": os.path.join(data_dir, f"dpo_chosen{suffix}*.bin"),
            "rejected": os.path.join(data_dir, f"dpo_rejected{suffix}*.bin"),
        }
        json_pattern = os.path.join(data_dir, f"dpo_prompt_lens{suffix}*.json")

        prompt_bins = sorted(glob.glob(bin_patterns["prompt"]))
        chosen_bins = sorted(glob.glob(bin_patterns["chosen"]))
        rejected_bins = sorted(glob.glob(bin_patterns["rejected"]))
        prompt_lens_json = sorted(glob.glob(json_pattern))

        if not prompt_bins:
            raise FileNotFoundError(
                f"No dpo_prompts{suffix}*.bin files found in {data_dir}. "
                f"Run data/pack_dpo.py first."
            )

        # Load all records
        self._prompt_ids: List[torch.Tensor] = []
        self._chosen_ids: List[torch.Tensor] = []
        self._rejected_ids: List[torch.Tensor] = []
        self._prompt_lens: List[int] = []

        for pb, cb, rb in zip(prompt_bins, chosen_bins, rejected_bins):
            prompts = self._load_bin(pb)
            chosen = self._load_bin(cb)
            rejected = self._load_bin(rb)

            # Load prompt lens matching this shard
            # (prompt_lens.json files align 1:1 with bin shards)
            plens = self._load_lens_json(
                pb.replace("_prompts_", "_prompt_lens_")
                   .replace(".bin", ".json")
            )

            n = min(len(prompts), len(chosen), len(rejected), len(plens))
            for i in range(n):
                self._prompt_ids.append(torch.tensor(prompts[i], dtype=torch.long))
                self._chosen_ids.append(torch.tensor(chosen[i], dtype=torch.long))
                self._rejected_ids.append(torch.tensor(rejected[i], dtype=torch.long))
                self._prompt_lens.append(plens[i])

        # Rank shard
        total = len(self._prompt_ids)
        shard_size = total // world_size
        start = rank * shard_size
        end = start + shard_size if rank < world_size - 1 else total
        self._prompt_ids = self._prompt_ids[start:end]
        self._chosen_ids = self._chosen_ids[start:end]
        self._rejected_ids = self._rejected_ids[start:end]
        self._prompt_lens = self._prompt_lens[start:end]

        if rank == 0:
            print(f"[PackedDPODataLoader] {total:,} total triples, "
                  f"{end - start:,} per rank ({split})")

    # ------------------------------------------------------------------
    @staticmethod
    def _load_bin(path: str) -> List[List[int]]:
        """Load length-prefixed uint16 records from a .bin file."""
        with open(path, "rb") as f:
            data = f.read()
        records: List[List[int]] = []
        offset = 0
        while offset < len(data):
            n_tokens = int.from_bytes(data[offset: offset + 4], "little")
            offset += 4
            tokens = []
            for i in range(n_tokens):
                tok = int.from_bytes(
                    data[offset + i * 2: offset + (i + 1) * 2], "little"
                )
                tokens.append(tok)
            offset += n_tokens * 2
            records.append(tokens)
        return records

    @staticmethod
    def _load_lens_json(path: str) -> List[int]:
        """Load prompt lens from a JSON file."""
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._prompt_ids)

    def __iter__(self):
        """
        Yield (chosen_ids, chosen_mask, rejected_ids, rejected_mask) batches.

        chosen_ids:   (B, T_chosen)  token ids, prompt + chosen completion
        chosen_mask:  (B, T_chosen)  1 for completion positions
        rejected_ids: (B, T_rejected) token ids, prompt + rejected completion
        rejected_mask: (B, T_rejected) 1 for completion positions
        """
        indices = list(range(len(self._prompt_ids)))
        self.rng.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            chosen_ids_list = []
            chosen_mask_list = []
            rejected_ids_list = []
            rejected_mask_list = []

            max_c_len = 0
            max_r_len = 0

            # First pass: find max lengths for padding
            for j in batch_idx:
                pids = self._prompt_ids[j]
                cids = self._chosen_ids[j]
                rids = self._rejected_ids[j]
                max_c_len = max(max_c_len, len(pids) + len(cids))
                max_r_len = max(max_r_len, len(pids) + len(rids))

            for j in batch_idx:
                pids = self._prompt_ids[j]
                cids = self._chosen_ids[j]
                rids = self._rejected_ids[j]
                plen = self._prompt_lens[j]

                # Chosen: concat prompt + completion, mask prompt positions
                full_c = torch.cat([pids, cids])
                mask_c = torch.cat([
                    torch.zeros(plen, dtype=torch.float),
                    torch.ones(len(cids), dtype=torch.float),
                ])
                # Pad to max_c_len
                pad_c_len = max_c_len - len(full_c)
                if pad_c_len > 0:
                    full_c = F.pad(full_c, (0, pad_c_len), value=0)
                    mask_c = F.pad(mask_c, (0, pad_c_len), value=0.0)

                # Rejected: concat prompt + completion, mask prompt positions
                full_r = torch.cat([pids, rids])
                mask_r = torch.cat([
                    torch.zeros(plen, dtype=torch.float),
                    torch.ones(len(rids), dtype=torch.float),
                ])
                pad_r_len = max_r_len - len(full_r)
                if pad_r_len > 0:
                    full_r = F.pad(full_r, (0, pad_r_len), value=0)
                    mask_r = F.pad(mask_r, (0, pad_r_len), value=0.0)

                chosen_ids_list.append(full_c)
                chosen_mask_list.append(mask_c)
                rejected_ids_list.append(full_r)
                rejected_mask_list.append(mask_r)

            chosen_ids = torch.stack(chosen_ids_list)
            chosen_mask = torch.stack(chosen_mask_list)
            rejected_ids = torch.stack(rejected_ids_list)
            rejected_mask = torch.stack(rejected_mask_list)

            yield chosen_ids, chosen_mask, rejected_ids, rejected_mask


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------


def dpo_loss(
    policy_chosen_logp: torch.Tensor,   # (B, T) per-token logprobs of chosen tokens
    policy_rejected_logp: torch.Tensor,  # (B, T) per-token logprobs of rejected tokens
    ref_chosen_logp: torch.Tensor,       # (B, T) reference logprobs for chosen
    ref_rejected_logp: torch.Tensor,     # (B, T) reference logprobs for rejected
    chosen_mask: torch.Tensor,           # (B, T) 1 for chosen completion positions
    rejected_mask: torch.Tensor,         # (B, T) 1 for rejected completion positions
    beta: float = 0.1,                   # DPO temperature parameter
    label_smoothing: float = 0.0,        # DPO label smoothing
    clip_ratio: Optional[float] = None,  # Optional PPO-style clipping
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Direct Preference Optimization (DPO) loss.

    The loss encourages the policy to increase the log-probability of chosen
    completions and decrease that of rejected completions, relative to a
    frozen reference model.

    Args:
        policy_chosen_logp:   Log-probs of tokens from the policy for chosen sequences.
        policy_rejected_logp: Log-probs of tokens from the policy for rejected sequences.
        ref_chosen_logp:      Log-probs from the reference model for chosen sequences.
        ref_rejected_logp:    Log-probs from the reference model for rejected sequences.
        chosen_mask:          1D mask for completion tokens in chosen sequences.
        rejected_mask:        1D mask for completion tokens in rejected sequences.
        beta:                 DPO temperature (higher = more emphasis on preferring chosen).
        label_smoothing:      Smooths the preference label (epsilon in DPO formula).
        clip_ratio:           If set, clips the implicit reward ratio to [1-clip, 1+clip].

    Returns:
        (loss, metrics_dict)
    """
    B, T_c = policy_chosen_logp.shape
    T_r = policy_rejected_logp.shape[1]

    # Per-token log ratios (policy - reference)
    chosen_log_ratio = (policy_chosen_logp - ref_chosen_logp) * chosen_mask
    rejected_log_ratio = (policy_rejected_logp - ref_rejected_logp) * rejected_mask

    # Sum over completion tokens to get log π(y|x)
    # Mask ensures only completion tokens contribute
    chosen_log_sum = chosen_log_ratio.sum(dim=1)   # (B,)
    rejected_log_sum = rejected_log_ratio.sum(dim=1)  # (B,)

    # DPO implicit reward
    logits = beta * (chosen_log_sum - rejected_log_sum)

    # Optional clipping on the implicit reward ratio
    if clip_ratio is not None:
        # Clip the exponentiated log ratios
        ratio_c = (chosen_log_sum - rejected_log_sum).exp().clamp(
            1.0 / (1.0 + clip_ratio), 1.0 + clip_ratio,
        )
        logits = beta * ratio_c.log()

    # Label smoothing: instead of preferring chosen over rejected with
    # probability 1, soften the target to (1 - label_smoothing)
    if label_smoothing > 0:
        # DPO with label smoothing:
        # L = -[(1-ε) * log σ(β * Δ) + ε * log σ(-β * Δ)]
        # where Δ = chosen_log_sum - rejected_log_sum
        loss = -(
            (1.0 - label_smoothing) * F.logsigmoid(logits)
            + label_smoothing * F.logsigmoid(-logits)
        )
    else:
        loss = -F.logsigmoid(logits)

    loss = loss.mean()

    # Metrics
    with torch.no_grad():
        chosen_rewards = beta * chosen_log_sum.detach()
        rejected_rewards = beta * rejected_log_sum.detach()
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

    metrics = {
        "dpo_loss": float(loss.detach().item()),
        "chosen_reward": float(chosen_rewards.mean().item()),
        "rejected_reward": float(rejected_rewards.mean().item()),
        "reward_margin": float((chosen_rewards - rejected_rewards).mean().item()),
        "accuracy": accuracy,
        "logits_mean": float(logits.detach().mean().item()),
    }

    return loss, metrics


# ---------------------------------------------------------------------------
# Log-prob computation
# ---------------------------------------------------------------------------


def compute_sequence_logprobs(
    model: TransformerForCausalLM,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token log-probabilities for a sequence.

    Args:
        model: The transformer model.
        input_ids: (B, T) token ids.

    Returns:
        logprobs: (B, T) per-token log-probabilities.
                  Position t has the logprob of token at position t+1
                  (shifted internally). The last position is 0.
    """
    out = model(input_ids, use_cache=False)
    logits = out["logits"][:, :-1, :].float()  # (B, T-1, V)
    targets = input_ids[:, 1:]  # (B, T-1)
    logp = logits.log_softmax(dim=-1)
    tok_lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    # Pad back to (B, T) with 0 for the last position
    tok_lp = F.pad(tok_lp, (0, 1), value=0.0)
    return tok_lp


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
    Build a frozen reference model for KL anchoring.

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
        print(f"[RefPolicy] two-model: frozen reference ({n_ref / 1e9:.3f}B)")
        return ref_model

    raise ValueError(f"Unknown --ref_policy {ref_policy!r}; expected 'single' or 'two'")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def prune_checkpoints(out_dir: str, keep: int = 3):
    ckpts = sorted(
        Path(out_dir).glob("dpo_step*.pt"),
        key=lambda p: int(p.stem.replace("dpo_step", "")),
    )
    for old in ckpts[:-keep]:
        old.unlink()


def save_dpo_checkpoint(
    out_dir: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    args_dict: dict,
    is_lora: bool,
    recipe: TrainingRecipe,
):
    """Save DPO checkpoint with recipe.json sidecar."""
    raw = _raw(model)
    ckpt = {
        "step": step,
        "model_state": raw.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": vars(config),
        "args": args_dict,
        "is_lora": is_lora,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"dpo_step{step:07d}.pt")
    torch.save(ckpt, path)
    latest = os.path.join(out_dir, "latest.pt")
    if os.path.islink(latest):
        os.remove(latest)
    try:
        os.symlink(os.path.abspath(path), latest)
    except OSError:
        pass

    # Save recipe alongside checkpoint
    recipe.to_json(os.path.join(out_dir, "recipe.json"))

    print(f"[Checkpoint] saved {path}")
    return path


def load_dpo_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    is_lora: bool,
) -> int:
    """Load DPO checkpoint and return step."""
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
    val_dataset: PackedDPODataLoader,
    ref_model: Optional[TransformerForCausalLM],
    beta: float,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Run validation over a subset of preference pairs.

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
        policy_c_logp = compute_sequence_logprobs(model, chosen_ids)
        policy_r_logp = compute_sequence_logprobs(model, rejected_ids)

        # Reference logprobs
        ref_for_logprob = ref_model if ref_model is not None else model
        ref_c_logp = compute_sequence_logprobs(ref_for_logprob, chosen_ids)
        ref_r_logp = compute_sequence_logprobs(ref_for_logprob, rejected_ids)

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
# Smoke test
# ---------------------------------------------------------------------------


def smoke_test():
    """
    Self-contained smoke test using tiny model and synthetic preference data.
    Tests the full DPO pipeline: data loading, logprob computation, loss, backward.
    """
    print("\n=== DPO smoke test ===")
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp()
    data_dir = os.path.join(tmp, "dpo_data")
    ckpt_dir = os.path.join(tmp, "ckpts")
    tok_dir = os.path.join(tmp, "tokenizer")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tok_dir, exist_ok=True)

    # ---- minimal tokenizer ----
    from tokenizers import Tokenizer as _Tok
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers import pre_tokenizers, decoders

    SPECIAL = [
        "<|endoftext|>", "<|pad|>", "<|im_start|>", "<|im_end|>",
        "<think>", "</think>",
    ]
    tok = _Tok(BPE(unk_token=None, byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=512, special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    corpus = [
        "<|im_start|>user\nSolve 2+2<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n2+2=4\n</think>\n4<|im_end|>\n",
    ] * 30
    tok.train_from_iterator(corpus, trainer=trainer)
    tok.save(os.path.join(tok_dir, "tokenizer.json"))

    # ---- create DPO preference data ----
    from recipe import TrainingRecipe
    recipe = TrainingRecipe(mode="reasoning")
    eos_id = tok.token_to_id("<|endoftext|>") or tok.get_vocab_size() - 1

    records = [
        {
            "prompt": "Solve: 2+2",
            "chosen": "4",
            "rejected": "5",
            "chosen_thinking": "2 plus 2 equals 4",
            "rejected_thinking": "Maybe 5?",
        },
        {
            "prompt": "What is 10-3?",
            "chosen": "7",
            "rejected": "8",
            "chosen_thinking": "10 minus 3 is 7",
            "rejected_thinking": "I think 8",
        },
    ] * 25

    # Write packed data using the pack_dpo format
    prompt_ids_list: List[List[int]] = []
    chosen_ids_list: List[List[int]] = []
    rejected_ids_list: List[List[int]] = []
    prompt_lens_list: List[int] = []

    for rec in records:
        prompt = rec["prompt"]
        chosen = rec["chosen"]
        rejected = rec["rejected"]
        chosen_thinking = rec.get("chosen_thinking", "")
        rejected_thinking = rec.get("rejected_thinking", "")

        # Tokenize (same logic as pack_dpo)
        prompt_text = (
            recipe.format_user_turn(prompt) + recipe.turn_prefix_assistant
        )
        p_ids = tok.encode(prompt_text).ids
        p_ids = [min(tid, 65535) for tid in p_ids]

        chosen_body = recipe.format_assistant_turn(
            thinking=chosen_thinking, answer=chosen, want_thinking=True,
        )
        chosen_text = prompt_text + chosen_body
        c_ids = tok.encode(chosen_text).ids
        c_ids = [min(tid, 65535) for tid in c_ids]
        c_ids.append(eos_id)

        rejected_body = recipe.format_assistant_turn(
            thinking=rejected_thinking, answer=rejected, want_thinking=True,
        )
        rejected_text = prompt_text + rejected_body
        r_ids = tok.encode(rejected_text).ids
        r_ids = [min(tid, 65535) for tid in r_ids]
        r_ids.append(eos_id)

        prompt_ids_list.append(p_ids)
        chosen_ids_list.append(c_ids)
        rejected_ids_list.append(r_ids)
        prompt_lens_list.append(len(p_ids))

    # Write bin files (length-prefixed uint16)
    def _write_bin(path, records_list):
        with open(path, "wb") as f:
            for ids in records_list:
                f.write(len(ids).to_bytes(4, "little"))
                for tid in ids:
                    f.write(min(tid, 65535).to_bytes(2, "little"))

    _write_bin(os.path.join(data_dir, "dpo_prompts_train.bin"), prompt_ids_list)
    _write_bin(os.path.join(data_dir, "dpo_chosen_train.bin"), chosen_ids_list)
    _write_bin(os.path.join(data_dir, "dpo_rejected_train.bin"), rejected_ids_list)
    with open(os.path.join(data_dir, "dpo_prompt_lens_train.json"), "w") as f:
        json.dump(prompt_lens_list, f)

    # ---- tiny model ----
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

    # ---- inject LoRA for lower memory ----
    from peft.lora import inject_lora
    n_replaced = inject_lora(model, rank=4, alpha=8.0)
    print(f"[smoke] LoRA: {n_replaced} adapters injected")

    # ---- dataset ----
    ds = PackedDPODataLoader(
        data_dir=data_dir, batch_size=2,
        rank=0, world_size=1,
    )
    print(f"[smoke] dataset size: {len(ds)}")

    # ---- reference ----
    ref = build_reference("single", config, sft_ckpt_path, device)

    # ---- compute loss ----
    for chosen_ids, chosen_mask, rejected_ids, rejected_mask in ds:
        chosen_ids = chosen_ids.to(device)
        chosen_mask = chosen_mask.to(device)
        rejected_ids = rejected_ids.to(device)
        rejected_mask = rejected_mask.to(device)

        # Reference logprobs
        with torch.no_grad():
            ref_c_logp = compute_sequence_logprobs(ref or model, chosen_ids)
            ref_r_logp = compute_sequence_logprobs(ref or model, rejected_ids)

        # Policy logprobs
        policy_c_logp = compute_sequence_logprobs(model, chosen_ids)
        policy_r_logp = compute_sequence_logprobs(model, rejected_ids)

        # DPO loss
        loss, metrics = dpo_loss(
            policy_c_logp, policy_r_logp,
            ref_c_logp, ref_r_logp,
            chosen_mask, rejected_mask,
            beta=0.1,
        )

        print(
            f"[smoke] loss {loss.item():+.4f} | "
            f"reward_margin {metrics['reward_margin']:.4f} | "
            f"accuracy {metrics['accuracy']:.2f}"
        )

        # Backward
        loss.backward()
        print(f"[smoke] backward: OK")

        # Step
        optim = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-5, weight_decay=0.0,
        )
        optim.step()
        optim.zero_grad()
        print(f"[smoke] optimizer step: OK")

        break  # one batch is enough

    # ---- checkpoint round-trip ----
    out_dir = os.path.join(ckpt_dir, "dpo")
    os.makedirs(out_dir, exist_ok=True)
    save_dpo_checkpoint(
        out_dir, 1, model, optim, config,
        vars(argparse.Namespace(test=True)), False, recipe,
    )

    # Load back
    model2 = TransformerForCausalLM(config).to(device)
    inject_lora(model2, rank=4, alpha=8.0)
    load_step = load_dpo_checkpoint(
        os.path.join(out_dir, "dpo_step0000001.pt"),
        model2, None, device, is_lora=False,
    )
    print(f"[smoke] checkpoint round-trip: step {load_step}")

    shutil.rmtree(tmp)
    print("\n=== DPO smoke test passed ===\n")


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
        print(f"Loaded SFT checkpoint: {n_total:,} params ({n_total / 1e9:.3f}B)")

    # ----------------------------------------------------------------- LoRA
    is_lora = args.lora
    if is_lora:
        from peft.lora import inject_lora, freeze_base
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
    train_ds = PackedDPODataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        split="train",
    )
    if master:
        print(f"[Dataset] {len(train_ds):,} preference triples ({world_size} rank(s))")

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
        start_step = load_dpo_checkpoint(
            args.resume, model, optimizer, device, is_lora,
        )

    if master:
        os.makedirs(args.out_dir, exist_ok=True)
        eff_batch = args.batch_size * world_size
        print(f"\nEffective batch     : {eff_batch} preference triples")
        print(f"Max steps           : {args.max_steps:,}")
        print(f"DPO beta            : {args.beta}")
        print(f"Label smoothing     : {args.label_smoothing}")
        print(f"Reference policy    : {args.ref_policy}")
        print(f"Checkpoint every    : {args.save_every:,} steps\n")

    # ================================================================= LOOP
    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    loss_window: List[float] = []
    acc_window: List[float] = []

    step = start_step
    data_iter = iter(train_ds)

    while step < args.max_steps:
        lr = scheduler(step)
        for pg in optimizer.param_groups:
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
            ref_c_logp = compute_sequence_logprobs(ref_for_logprob, chosen_ids)
            ref_r_logp = compute_sequence_logprobs(ref_for_logprob, rejected_ids)

        # 3. policy log-probs (with grad)
        with ctx:
            if _use_cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()
            policy_c_logp = compute_sequence_logprobs(model, chosen_ids)
            policy_r_logp = compute_sequence_logprobs(model, rejected_ids)

        # 4. DPO loss
        loss, metrics = dpo_loss(
            policy_c_logp, policy_r_logp,
            ref_c_logp, ref_r_logp,
            chosen_mask, rejected_mask,
            beta=args.beta,
            label_smoothing=args.label_smoothing,
            clip_ratio=args.clip_ratio if args.clip_ratio > 0 else None,
        )

        # 5. backward + step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        optimizer.step()

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
                print(
                    f"step {step:6d} | loss {avg_loss:+.4f} | "
                    f"acc {avg_acc:.2%} | "
                    f"r_margin {metrics['reward_margin']:.4f} | "
                    f"lr {lr:.2e} | g {grad_norm:.2f} | "
                    f"{sps:.2f} step/s"
                )

        # 7. checkpoint
        if master and step > start_step and step % args.save_every == 0:
            save_dpo_checkpoint(
                args.out_dir, step, model, optimizer, config,
                vars(args), is_lora, recipe,
            )
            prune_checkpoints(args.out_dir, keep=args.keep_ckpts)

        step += 1

    # final
    if master:
        save_dpo_checkpoint(
            args.out_dir, step, model, optimizer, config,
            vars(args), is_lora, recipe,
        )
        print(f"\nDPO complete. Final loss: {loss.detach().item():.4f}")

    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="DPO preference fine-tuning for reasoning / chat models.",
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
    p.add_argument("--resume", default=None,
                   help="DPO checkpoint to resume from")

    # LoRA
    p.add_argument("--lora", action="store_true",
                   help="Enable LoRA adapters")
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=float, default=128.0)

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
    p.add_argument("--kl-coef", type=float, default=0.0,
                   help="KL penalty coefficient (not typically used in standard DPO)")

    # Optim
    p.add_argument("--batch-size", type=int, default=4,
                   help="Preference triples per step")
    p.add_argument("--num-steps", type=int, default=500,
                   help="Total training steps, alias for --max-steps")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Total training steps (overrides --num-steps)")
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

    # Recipe
    add_recipe_args(p)

    # Smoke test
    p.add_argument("--smoke-test", action="store_true",
                   help="Run a self-contained smoke test and exit")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Backwards compat: --num-steps is the alias, --max-steps overrides
    if args.max_steps is not None:
        args.num_steps = args.max_steps
    elif args.num_steps is not None:
        args.max_steps = args.num_steps
    else:
        args.max_steps = args.num_steps = 500

    if args.smoke_test:
        smoke_test()
    elif args.checkpoint is None:
        print("No --checkpoint given. Run with --smoke-test for a quick test, "
              "or provide --checkpoint for training.")
        smoke_test()
    else:
        train(args)
