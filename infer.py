#!/usr/bin/env python3
"""
infer.py

Production-grade inference for the Advanced LLM Framework (dense, non-MoE)
model from `model.py`.  Loads a checkpoint produced by any of the training
scripts (pretrain, SFT, GRPO, merge-lora, deepspeed-shard-consolidator),
and runs text generation with the same KV-cache machinery the trainer uses.

All template strings, special-token lists, and turn markers are read from
a `TrainingRecipe` object (`recipe.py`) -- nothing is hardcoded here.

Highlights
----------
* Reads raw `.pt` checkpoints (with {"model_state", "config"}) and
  transparently handles LoRA checkpoints ("lora_state_dict" key present)
  by injecting adapters, loading weights, and merging in one step.
* Auto-loads the `TrainingRecipe` from `recipe.json` next to the checkpoint
  (or via `--recipe`), so template tokens, turn prefixes, and reasoning
  tags are always consistent with training.
* bf16 by default.  Optional 4-bit / 8-bit quantization through
  bitsandbytes.  Both are *soft* dependencies.
* Full generation hyperparameter set: temperature, top-k, top-p,
  repetition penalty, min / max new tokens, batched generation with
  left-padded prompts, EOS / stop-token handling, seeded RNG, streaming.
* `--chat-template` flag for explicit override; auto-detects from recipe
  when omitted.

Usage
-----
  # one-shot generation
  python infer.py --checkpoint ./checkpoints/latest.pt \\
      --prompt "Solve 2+2"

  # interactive REPL with streaming output
  python infer.py --checkpoint ./checkpoints/latest.pt \\
      --interactive

  # explicit recipe
  python infer.py --checkpoint ./checkpoints/latest.pt \\
      --recipe ./recipe.json --prompt "Hello"

  # batched evaluation
  python infer.py --checkpoint ./checkpoints/latest.pt \\
      --prompts-file ./eval.jsonl --batch-size 8 --output ./out.jsonl

  # quantized
  python infer.py --checkpoint ./checkpoints/latest.pt \\
      --quantize 4bit --prompt "..."

  # smoke test (no checkpoint needed)
  python infer.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tokenizers import Tokenizer

from model import ModelConfig, TransformerForCausalLM, count_parameters
from recipe import TrainingRecipe, get_recipe
from atomic_io import atomic_torch_save


# ---------------------------------------------------------------------------
# Optional dependencies (kept soft so the script runs without them)
# ---------------------------------------------------------------------------
# accelerate is the device-map / offload engine.  bitsandbytes is the
# int4 / int8 quantization engine.  Both are imported lazily.

def _try_import_accelerate():
    try:
        from accelerate import dispatch_model, infer_auto_device_map
        return {
            "dispatch_model": dispatch_model,
            "infer_auto_device_map": infer_auto_device_map,
        }
    except ImportError:
        return None


def _try_import_bitsandbytes():
    try:
        import bitsandbytes as bnb
        return bnb
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Chat template detection
# ---------------------------------------------------------------------------

def detect_chat_template(
    messages: List[Dict[str, str]],
    recipe: TrainingRecipe,
) -> str:
    """
    Auto-detect the chat template to use based on the recipe configuration.

    The recipe is the single source of truth -- no hardcoded marker lists
    are consulted.  Returns one of: "chatml", "raw", "custom".
    """
    template = recipe.chat_template
    if template not in ("chatml", "raw", "custom"):
        # Unknown template in recipe -- fall back to chatml.
        print(f"[infer] WARNING: unknown chat_template={template!r} in recipe; "
              f"falling back to 'chatml'")
        template = "chatml"
    return template


# ---------------------------------------------------------------------------
# Prompt formatting (recipe-driven)
# ---------------------------------------------------------------------------

def format_prompt(
    messages: List[Dict[str, str]],
    recipe: TrainingRecipe,
    template: Optional[str] = None,
    enable_thinking: bool = False,
) -> str:
    """
    Format a list of message dicts into a single prompt string using the
    recipe's chat template.

    Each message is a dict with "role" and "content" keys.  Roles are
    one of: "system", "user", "assistant".

    The formatted output ends with the assistant turn prefix (and
    optionally ``<think>``) so the model can continue generating.
    """
    if template is None:
        template = detect_chat_template(messages, recipe)

    if template == "raw":
        # Raw mode: concatenate all user messages, ignore roles.
        parts = []
        for msg in messages:
            if msg["role"] == "user":
                parts.append(msg["content"])
        return "".join(parts)

    # chatml or custom: use recipe turn prefixes/suffixes.
    parts: List[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            # Use the recipe's explicit system-turn formatter — the role tag
            # is rewritten safely without falling back to string substitution
            # of "user" → "system" (which would corrupt any content that
            # happened to contain the substring "user").
            parts.append(recipe.format_system_turn(content))
        elif role == "user":
            parts.append(recipe.format_user_turn(content))
        elif role == "assistant":
            parts.append(
                f"{recipe.turn_prefix_assistant}"
                f"{content}"
                f"{recipe.turn_suffix_assistant}"
            )

    # End with assistant turn prefix so the model generates the response.
    parts.append(recipe.turn_prefix_assistant)
    if enable_thinking and recipe.mode in ("reasoning", "hybrid"):
        parts.append(f"{recipe.think_open}\n")

    return "".join(parts)



# ---------------------------------------------------------------------------
# Tokenizer loader
# ---------------------------------------------------------------------------

def load_tokenizer(checkpoint_dir: str) -> Tokenizer:
    """
    Read `tokenizer.json` from a directory.  Searches the checkpoint
    directory itself, then its parent, for a `tokenizer.json` file.
    """
    # Try the checkpoint directory (if it's a directory) or its parent.
    candidates = []
    if os.path.isdir(checkpoint_dir):
        candidates.append(os.path.join(checkpoint_dir, "tokenizer.json"))
        candidates.append(checkpoint_dir)  # might BE the tokenizer dir
    parent = os.path.dirname(os.path.abspath(checkpoint_dir))
    candidates.append(os.path.join(parent, "tokenizer.json"))
    # Also check a "tokenizer" subdirectory next to the checkpoint.
    candidates.append(os.path.join(parent, "tokenizer", "tokenizer.json"))

    for cand in candidates:
        if cand.endswith("tokenizer.json") and os.path.isfile(cand):
            return Tokenizer.from_file(cand)

    raise FileNotFoundError(
        f"tokenizer.json not found near {checkpoint_dir!r}.  "
        f"Searched: {[c for c in candidates if c.endswith('.json')]}  "
        f"Run train_tokenizer.py first."
    )


def find_eos_token_id(tokenizer: Tokenizer, recipe: Optional[TrainingRecipe] = None) -> Optional[int]:
    """Pick a reasonable EOS token id from the tokenizer's vocabulary."""
    if recipe is not None:
        eos_str = recipe.eos_token
        vocab = tokenizer.get_vocab()
        if eos_str in vocab:
            return vocab[eos_str]
    vocab = tokenizer.get_vocab()
    for candidate in ("<|endoftext|>", "<|im_end|>", "</s>"):
        if candidate in vocab:
            return vocab[candidate]
    return None


# ---------------------------------------------------------------------------
# Model + tokenizer + recipe loader
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    checkpoint_path: str,
    device: str = "auto",
    quantize: Optional[str] = None,
    max_seq_len: Optional[int] = None,
) -> Tuple[TransformerForCausalLM, Tokenizer, TrainingRecipe]:
    """
    Load a model, tokenizer, and training recipe from a checkpoint.

    The checkpoint can be:
      * A `.pt` file with {"model_state", "config"}  (plain model)
      * A `.pt` file with {"model_state", "config", "lora_state_dict"}  (LoRA)
      * A DeepSpeed checkpoint directory (rejected with a clear message)

    If ``lora_state_dict`` is present, the LoRA adapters are injected,
    loaded, and merged transparently, yielding a plain model ready for
    inference.

    The tokenizer is discovered automatically from the checkpoint's
    directory (or a ``tokenizer/`` subdirectory).

    The recipe is loaded from ``recipe.json`` next to the checkpoint
    (or via ``get_recipe()`` resolution).
    """
    # ---- recipe first (needed for token detection elsewhere, though
    #      the recipe object itself is independent of the checkpoint blob)
    recipe = get_recipe(checkpoint_path)

    # ---- checkpoint validation
    if os.path.isdir(checkpoint_path):
        # A genuine DeepSpeed checkpoint directory contains a `latest_checkpoint`
        # sentinel file or per-partition `zero_pp_*` shards. Anything else is a
        # directory of weights — look for model.safetensors / pytorch_model.bin
        # inside it.
        looks_like_ds = (
            os.path.exists(os.path.join(checkpoint_path, "latest_checkpoint"))
            or any(
                name.startswith("zero_pp_") or name.startswith("zero_dp_")
                for name in os.listdir(checkpoint_path)
            )
        )
        if looks_like_ds:
            raise RuntimeError(
                f"{checkpoint_path} looks like a DeepSpeed checkpoint directory.\n"
                f"Run deepspeed_shard_consolidator.py first to produce a "
                f"single .pt, then point --checkpoint at the consolidated file."
            )
        # If no recognised model artefact is present, surface a clear error.
        if not any(
            os.path.exists(os.path.join(checkpoint_path, n))
            for n in ("model.safetensors", "pytorch_model.bin", "consolidated.pt")
        ):
            raise FileNotFoundError(
                f"{checkpoint_path!r} is a directory but contains no recognisable "
                f"inference artefact (model.safetensors / pytorch_model.bin / "
                f"consolidated.pt) and no DeepSpeed `latest_checkpoint` marker."
            )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path!r}")

    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or "model_state" not in blob or "config" not in blob:
        raise RuntimeError(
            f"{checkpoint_path!r} does not contain the expected keys "
            f"'model_state' and 'config'. Was it produced by train.py, "
            f"train_sft.py, or deepspeed_shard_consolidator.py?"
        )

    # ---- config
    config = ModelConfig(**blob["config"])
    if max_seq_len is not None:
        config.max_position_embeddings = max_seq_len

    # ---- build model and load base weights
    model = TransformerForCausalLM(config)
    missing, unexpected = model.load_state_dict(blob["model_state"], strict=False)
    if unexpected:
        print(f"[infer] WARNING: {len(unexpected)} unexpected keys in "
              f"checkpoint; ignoring. First few: {unexpected[:3]}")
    if missing:
        non_tied = [k for k in missing if "lm_head.weight" not in k]
        if non_tied:
            print(f"[infer] WARNING: {len(non_tied)} non-tied keys missing "
                  f"from checkpoint; first few: {non_tied[:3]}")

    # ---- LoRA detection and loading
    if "lora_state_dict" in blob:
        print(f"[infer] LoRA state dict detected — injecting adapters and merging.")
        from peft.lora import inject_lora, merge_lora
        inject_lora(model)
        # Load LoRA adapter weights (keys like lora_A.weight, lora_B.weight)
        lora_sd = blob["lora_state_dict"]
        lora_missing, lora_unexpected = model.load_state_dict(lora_sd, strict=True)
        if lora_unexpected:
            print(f"[infer] WARNING: {len(lora_unexpected)} unexpected LoRA keys; "
                  f"ignoring. First few: {lora_unexpected[:3]}")
        # Merge LoRA into base weights for inference
        merge_lora(model)
        print(f"[infer] LoRA adapters merged into base model.")

    # ---- re-tie weights after state dict load
    model.tie_weights()
    model.eval()

    # ---- tokenizer
    cp_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    tokenizer = load_tokenizer(cp_dir)

    return model, tokenizer, recipe


# ---------------------------------------------------------------------------
# Quantization (optional, via bitsandbytes)
# ---------------------------------------------------------------------------

def maybe_quantize(
    model: TransformerForCausalLM,
    quantize: Optional[str] = None,
) -> TransformerForCausalLM:
    """
    Apply bitsandbytes int4 / int8 quantization in place if requested.
    ``quantize`` is one of "4bit", "8bit", or None.
    """
    if quantize is None or quantize == "none":
        return model

    bnb = _try_import_bitsandbytes()
    if bnb is None:
        raise RuntimeError(
            f"{quantize} quantization was requested but bitsandbytes is "
            f"not installed.\n"
            f"  pip install bitsandbytes\n"
            f"  (then rerun with --quantize {quantize})"
        )

    is_4bit = quantize == "4bit"
    quant_cls = bnb.nn.Linear4bit if is_4bit else bnb.nn.Linear8bitLt
    quant_kwargs = (
        {
            "compute_dtype": torch.bfloat16,
            "quant_type":    "nf4",
            "use_double_quant": True,
        } if is_4bit
        else {
            "threshold":     6.0,
            "has_fp16_weights": False,
        }
    )

    n_replaced = 0
    for module_path, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            parent_path, attr = module_path.rsplit(".", 1) if "." in module_path else ("", module_path)
            parent = model
            for part in parent_path.split("."):
                if part:
                    parent = getattr(parent, part)
            new_mod = quant_cls(
                module.in_features, module.out_features,
                bias=module.bias is not None,
                **quant_kwargs,
            )
            with torch.no_grad():
                new_mod.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    new_mod.bias.data.copy_(module.bias.data)
            setattr(parent, attr, new_mod)
            n_replaced += 1

    print(f"[infer] quantized {n_replaced} nn.Linear layers with "
          f"bitsandbytes {'4-bit nf4' if is_4bit else '8-bit'}")
    return model



# ---------------------------------------------------------------------------
# Device-map resolution
# ---------------------------------------------------------------------------

def _parse_max_memory(spec: str) -> Dict[Union[int, str], str]:
    """
    Parse "0:18GiB,cpu:30GiB,disk:200GiB" into
    {0: "18GiB", "cpu": "30GiB", "disk": "200GiB"}.
    """
    out: Dict[Union[int, str], str] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"--max-memory chunk {chunk!r} has no ':' separator")
        k, v = chunk.split(":", 1)
        k, v = k.strip(), v.strip()
        if k.isdigit():
            out[int(k)] = v
        else:
            out[k] = v
    return out


def resolve_device_map(
    model: TransformerForCausalLM,
    device: str = "auto",
    max_memory: Optional[str] = None,
) -> Tuple[TransformerForCausalLM, str]:
    """
    Apply device placement per the --device / --max-memory flags.
    Returns (model, mode_label) where mode_label is one of
    {"single", "accelerate", "cpu"} for logging.
    """
    # Mode 1: explicit single device.
    if device != "auto":
        model.to(device)
        n = sum(p.numel() for p in model.parameters())
        print(f"[infer] model placed on {device}  ({n/1e9:.3f}B params)")
        return model, "single"

    # Mode 2: auto with accelerate.
    accel = _try_import_accelerate()
    if accel is not None:
        mem = _parse_max_memory(max_memory) if max_memory else None
        from model import DecoderLayer
        device_map = accel["infer_auto_device_map"](
            model,
            max_memory=mem,
            no_split_module_classes=[DecoderLayer],
        )
        model = accel["dispatch_model"](model, device_map=device_map)
        per_dev: Dict[str, int] = {}
        for _, dev_str in device_map.items():
            per_dev[dev_str] = per_dev.get(dev_str, 0) + 1
        print(f"[infer] accelerate device_map: " +
              ", ".join(f"{d}={n} submodules" for d, n in per_dev.items()))
        return model, "accelerate"

    # Mode 3: auto without accelerate -- fall back.
    target = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(target)
    print(f"[infer] accelerate not installed; falling back to {target} "
          f"(install with `pip install 'accelerate>=0.27'` for auto-shard "
          f"and CPU/disk offload).")
    return model, "cpu"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _apply_repetition_penalty(
    logits: torch.Tensor,
    generated: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Vectorised: divide logits of previously-generated tokens by `penalty`
    (positive logits) or multiply (negative logits). Modifies logits in place.
    """
    if penalty == 1.0:
        return logits
    bsz, vocab = logits.shape
    # (B, T) -> (B, V) one-hot via scatter; out-of-range positions are dropped
    # by masking the T dim down to whatever length `generated` actually has.
    prev = generated
    one_hot = torch.zeros((bsz, vocab), dtype=torch.bool, device=logits.device)
    # only count tokens that exist in the vocab (defensive against -1 sentinels)
    valid = (prev >= 0) & (prev < vocab)
    one_hot.scatter_(
        1,
        prev.clamp(min=0).masked_fill(~valid, 0),
        valid,
    )
    sub = logits[one_hot]                    # (num_masked,)
    adjusted = torch.where(sub > 0, sub / penalty, sub * penalty)
    logits[one_hot] = adjusted
    return logits


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k is None or k <= 0 or k >= logits.size(-1):
        return logits
    v, _ = torch.topk(logits, k)
    logits[logits < v[:, [-1]]] = -float("inf")
    return logits


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p is None or p <= 0.0 or p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_mask = cum > p
    sorted_mask[..., 0] = False
    sorted_logits[sorted_mask] = -float("inf")
    out = torch.full_like(logits, -float("inf"))
    out.scatter_(-1, sorted_idx, sorted_logits)
    return out


def sample_next(
    logits: torch.Tensor,
    generated: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    repetition_penalty: float,
) -> torch.Tensor:
    """Return a (B, 1) tensor of next-token ids."""
    if repetition_penalty != 1.0:
        logits = _apply_repetition_penalty(logits, generated, repetition_penalty)
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0; got {temperature}")
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    logits = _top_k_filter(logits, top_k)
    logits = _top_p_filter(logits, top_p)
    # Guard against degenerate filtered distributions (e.g. all -inf or
    # all-NaN after penalty + filter) so multinomial never samples NaN.
    finite = torch.isfinite(logits).any(dim=-1, keepdim=True)
    safe = torch.where(finite, logits, torch.full_like(logits, -1e9))
    probs = torch.softmax(safe, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _prepare_inputs(
    tokenizer: Tokenizer,
    prompts: List[str],
    device: torch.device,
    recipe: Optional[TrainingRecipe] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
    """
    Tokenize ``prompts``, left-pad to the longest, and return
    (input_ids, pad_mask, prompt_len).

    Left-padding is the right choice for causal LMs: the right edge
    stays in the same position across the batch, so positional
    encodings and KV-cache queries are consistent.

    ``pad_mask`` is a (B, max_len) additive float mask (0.0 real token,
    -inf padding) describing ONLY padding.  Returns None if no padding
    is needed so ``_step`` can use the fast ``is_causal=True`` SDPA path.
    """
    encoded = [tokenizer.encode(p) for p in prompts]
    ids_list = [e.ids for e in encoded]

    # Prefer recipe pad_token; fall back to <|pad|>, then </s>, then 0.
    pad_id = None
    if recipe is not None:
        vocab = tokenizer.get_vocab()
        pad_id = vocab.get(recipe.pad_token, None)
    if pad_id is None:
        # Explicit precedence: each candidate must be checked individually
        # so that `token_to_id("</s>")` returning 0 isn't treated as falsy.
        for candidate in ("<|pad|>", "</s>"):
            tid = tokenizer.token_to_id(candidate)
            if tid is not None:
                pad_id = tid
                break
    if pad_id is None:
        pad_id = 0  # last resort; assume an unused low-id slot

    lens = [len(x) for x in ids_list]
    max_len = max(lens)
    input_ids = torch.full((len(prompts), max_len), pad_id, dtype=torch.long)
    needs_padding = min(lens) != max_len
    pad_mask = (
        torch.zeros((len(prompts), max_len), dtype=torch.float)
        if needs_padding else None
    )
    for i, ids in enumerate(ids_list):
        offset = max_len - len(ids)
        input_ids[i, offset:] = torch.tensor(ids, dtype=torch.long)
        if needs_padding:
            pad_mask[i, :offset] = float("-inf")  # mask left-pad

    input_ids = input_ids.to(device)
    if pad_mask is not None:
        pad_mask = pad_mask.to(device)
    return input_ids, pad_mask, max_len


def _build_combined_mask(
    pad_mask_so_far: Optional[torch.Tensor],
    seq_len: int,
    past_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Build a (B, 1, seq_len, past_len+seq_len) additive mask combining
    causal restriction and key padding.  Also returns the extended padding
    mask for threading into the next step.
    Returns (None, None) when there is no padding at all.
    """
    if pad_mask_so_far is None:
        return None, None

    bsz = pad_mask_so_far.size(0)
    # pad_mask_so_far accumulates across decode steps — its width is
    # past_len + prev_seq_len, but we only want the past_len cache-aligned
    # columns to broadcast against `causal` (which is `past_len + seq_len`).
    # Slice the rightmost past_len columns, defensive left-pad if shorter
    # (can happen on the first decode step if the prefill width was already
    # padded to past_len during prefill).
    cur = pad_mask_so_far.size(1)
    if cur > past_len:
        cache_pad = pad_mask_so_far[:, cur - past_len:]
    elif cur < past_len:
        pad_left = torch.zeros(
            (bsz, past_len - cur), dtype=pad_mask_so_far.dtype, device=device,
        )
        cache_pad = torch.cat([pad_left, pad_mask_so_far], dim=1)
    else:
        cache_pad = pad_mask_so_far
    new_cols = torch.zeros(
        (bsz, seq_len), dtype=pad_mask_so_far.dtype, device=device,
    )
    full_pad = torch.cat([cache_pad, new_cols], dim=1)

    total_len = past_len + seq_len
    q_pos = torch.arange(past_len, total_len, device=device).unsqueeze(1)
    k_pos = torch.arange(0, total_len, device=device).unsqueeze(0)
    causal = torch.zeros((seq_len, total_len), device=device)
    causal.masked_fill_(k_pos > q_pos, float("-inf"))

    mask = causal.unsqueeze(0) + full_pad.unsqueeze(1)  # (B, seq_len, total_len)
    mask = mask.unsqueeze(1).to(dtype)                   # (B, 1, seq_len, total_len)
    return mask, full_pad


@torch.inference_mode()
def _step(
    model: TransformerForCausalLM,
    input_ids: torch.Tensor,
    pad_mask: Optional[torch.Tensor],
    past_key_values: Optional[List],
) -> Tuple[torch.Tensor, Optional[List], Optional[torch.Tensor]]:
    """
    Run one forward pass.  ``input_ids`` is either the full prompt (when
    ``past_key_values`` is None, i.e. the prefill step) or just the last
    token (decode step).

    Returns (next_token_logits, updated_past_key_values, updated_pad_mask).
    """
    bsz, seq_len = input_ids.shape
    model_param = next(model.parameters())
    # past_key_values may mix attention (Tensor) and mamba (None / state
    # tuple) entries — Jamba layers emit None for past_key_values[i][0]
    # (the conv state) and a tuple for [i][1] (the ssm state). Pick the
    # first non-None layer to read past_len. See model.py:1435-1438.
    past_len = 0
    if past_key_values is not None:
        for layer in past_key_values:
            for entry in layer:
                if isinstance(entry, torch.Tensor):
                    past_len = entry.shape[2]
                    break
            if past_len:
                break

    combined_mask, updated_pad_mask = _build_combined_mask(
        pad_mask, seq_len, past_len,
        device=input_ids.device, dtype=model_param.dtype,
    )

    out = model(
        input_ids=input_ids,
        attention_mask=combined_mask,
        past_key_values=past_key_values,
        use_cache=True,
    )
    next_logits = out["logits"][:, -1, :]
    return next_logits, out["past_key_values"], updated_pad_mask


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate(
    model: TransformerForCausalLM,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 512,
    min_new_tokens: int = 0,
    temperature: float = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    stop_on_think_close: bool = False,
    think_close_id: Optional[int] = None,
    pad_mask: Optional[torch.Tensor] = None,
    recipe: Optional[TrainingRecipe] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Autoregressive generation with KV-cache for a batch of prompts.

    ``input_ids`` is (B, T) -- pre-tokenised, left-padded prompts.
    Returns ``(output_ids, stats_dict)`` where output_ids is
    (B, T + new_tokens) and stats_dict contains timing / token counts.
    """
    device = input_ids.device
    bsz = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    past_kv: Optional[List] = None
    logits, past_kv, pad_mask = _step(model, input_ids, pad_mask, past_kv)
    generated = input_ids.clone()
    finished = torch.zeros(bsz, dtype=torch.bool, device=device)

    # Build stop-token set
    stop_ids: set = set()
    if eos_token_id is not None:
        stop_ids.add(eos_token_id)
    if stop_on_think_close and think_close_id is not None:
        stop_ids.add(think_close_id)

    t0 = time.perf_counter()

    for step in range(max_new_tokens):
        if finished.all():
            break

        next_id = sample_next(
            logits, generated,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Don't allow already-finished sequences to terminate again — they
        # should keep emitting whatever the caller asked for (typically
        # EOS) so the right-hand side of `generated` stays aligned. But
        # we never *stop* on them either, so the only effect is the shape.
        # Force-emit EOS for finished sequences so the batch is rectangular.
        if eos_token_id is not None:
            forced = torch.full_like(next_id, eos_token_id)
        else:
            forced = next_id
        next_id = torch.where(finished.unsqueeze(-1), forced, next_id)
        generated = torch.cat([generated, next_id], dim=1)

        # Detect stop tokens (only on active sequences). Sequences that
        # are still below --min_new_tokens are not eligible to finish
        # even if they would have sampled a stop token.
        tok_ids = next_id.squeeze(-1)  # (B,)
        hit_stop = torch.zeros_like(finished)
        for sid in stop_ids:
            hit_stop = hit_stop | (tok_ids == sid)
        below_min = (
            torch.tensor(step, device=device) < min_new_tokens
            if isinstance(min_new_tokens, int) else False
        )
        # `step` is 0-indexed; min_new_tokens is the minimum count of
        # *generated* tokens. After we appended this token we have
        # (step+1) generated tokens.
        below_min_mask = torch.tensor(
            (step + 1) < min_new_tokens, device=device
        ).expand_as(finished)
        newly_finished = hit_stop & ~finished & ~below_min_mask
        finished = finished | newly_finished

        # Decode step: feed only the new token. Skip _step for already-
        # finished sequences to save the no-op forward; their logits are
        # never consulted again.
        active_mask = ~finished
        if active_mask.any():
            # Feed only the active tokens to the model; finished ones we
            # ignore entirely by skipping this step. We still need the
            # full next_id tensor for shape consistency on the next iter,
            # so pass all of it but it doesn't matter — finished rows
            # will keep emitting EOS by force.
            logits, past_kv, pad_mask = _step(model, next_id, pad_mask, past_kv)
        else:
            break

    elapsed = time.perf_counter() - t0
    new_tokens = generated.shape[1] - prompt_len
    tps = new_tokens / elapsed if elapsed > 0 else 0.0

    # Per-row stop reason. The BATCH stop reason is the dominant reason
    # across rows (most common).
    full_ids_all = generated[:, prompt_len:].tolist()  # (B, T)
    per_row_reason: List[str] = ["length"] * bsz
    for i, gen_ids in enumerate(full_ids_all):
        # Find earliest stop-id occurrence (or length if none)
        first_stop_pos = None
        first_stop_id = None
        for sid in stop_ids:
            if sid in gen_ids:
                pos = gen_ids.index(sid)
                if first_stop_pos is None or pos < first_stop_pos:
                    first_stop_pos = pos
                    first_stop_id = sid
        if first_stop_pos is not None:
            if stop_on_think_close and first_stop_id == think_close_id:
                per_row_reason[i] = "think_close"
            else:
                per_row_reason[i] = "eos"
    # Aggregate: dominant reason = mode of per_row_reason
    from collections import Counter
    dominant = Counter(per_row_reason).most_common(1)[0][0]

    stats = {
        "prompt_tokens": prompt_len,
        "generated_tokens": new_tokens,
        "total_tokens": generated.shape[1],
        "elapsed_seconds": round(elapsed, 4),
        "tokens_per_second": round(tps, 2),
        "stop_reason": dominant,
        "per_row_stop_reason": per_row_reason,
        "batch_size": bsz,
    }
    return generated, stats



def generate_batch(
    model: TransformerForCausalLM,
    tokenizer: Tokenizer,
    prompts: List[str],
    *,
    recipe: Optional[TrainingRecipe] = None,
    max_new_tokens: int = 512,
    min_new_tokens: int = 0,
    temperature: float = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    seed: Optional[int] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> List[Dict]:
    """
    Generate completions for a batch of prompts.  Returns a list of
    dicts ``{"prompt", "completion", "stop_reason", "stats"}`` in input order.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if eos_token_id is None:
        eos_token_id = find_eos_token_id(tokenizer, recipe)

    # Determine think_close_id for recipe-aware stopping.
    think_close_id = None
    stop_on_think_close = False
    if recipe is not None and recipe.mode in ("reasoning", "hybrid"):
        vocab = tokenizer.get_vocab()
        tc_str = recipe.think_close
        if tc_str in vocab:
            think_close_id = vocab[tc_str]
            stop_on_think_close = True

    # Prefill
    input_ids, pad_mask, prompt_len = _prepare_inputs(
        tokenizer, prompts, device, recipe,
    )

    generated, stats = generate(
        model, input_ids,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_token_id,
        stop_on_think_close=stop_on_think_close,
        think_close_id=think_close_id,
        pad_mask=pad_mask,
        recipe=recipe,
    )

    # Decode output
    completions: List[Dict] = []
    for i, p in enumerate(prompts):
        full_ids = generated[i].tolist()
        completion_ids = full_ids[prompt_len:]
        stop_reason = "length"
        if eos_token_id is not None and eos_token_id in completion_ids:
            cut = completion_ids.index(eos_token_id)
            completion_ids = completion_ids[:cut]
            stop_reason = "eos"
        if stop_on_think_close and think_close_id is not None:
            if think_close_id in completion_ids:
                eos_pos = (completion_ids.index(eos_token_id)
                           if eos_token_id in completion_ids
                           else len(completion_ids))
                think_pos = completion_ids.index(think_close_id)
                if think_pos < eos_pos:
                    completion_ids = completion_ids[:think_pos]
                    stop_reason = "think_close"
        if stop_reason == "eos" and len(completion_ids) < min_new_tokens:
            stop_reason = "min_new_tokens"
        completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
        completions.append({
            "prompt":      p,
            "completion":  completion,
            "stop_reason": stop_reason,
            "stats":       stats,
        })
    return completions


def generate_stream(
    model: TransformerForCausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    recipe: Optional[TrainingRecipe] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    seed: Optional[int] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Yield decoded chunks one token at a time.  Used by --interactive.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if eos_token_id is None:
        eos_token_id = find_eos_token_id(tokenizer, recipe)

    # Determine think_close_id.
    think_close_id = None
    stop_on_think_close = False
    if recipe is not None and recipe.mode in ("reasoning", "hybrid"):
        vocab = tokenizer.get_vocab()
        tc_str = recipe.think_close
        if tc_str in vocab:
            think_close_id = vocab[tc_str]
            stop_on_think_close = True

    input_ids, pad_mask, _prompt_len = _prepare_inputs(
        tokenizer, [prompt], device, recipe,
    )
    past_kv: Optional[List] = None
    logits, past_kv, pad_mask = _step(model, input_ids, pad_mask, past_kv)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        next_id = sample_next(
            logits, generated,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        tok = int(next_id.item())
        generated = torch.cat([generated, next_id], dim=1)

        # Stop on EOS.
        if eos_token_id is not None and tok == eos_token_id:
            return

        # Stop on think_close in reasoning mode.
        if stop_on_think_close and think_close_id is not None and tok == think_close_id:
            return

        chunk = tokenizer.decode([tok], skip_special_tokens=False)
        yield chunk

        logits, past_kv, pad_mask = _step(model, next_id, pad_mask, past_kv)



# ---------------------------------------------------------------------------
# High-level chat interface
# ---------------------------------------------------------------------------

def chat(
    model: TransformerForCausalLM,
    tokenizer: Tokenizer,
    recipe: TrainingRecipe,
    messages: List[Dict[str, str]],
    *,
    enable_thinking: bool = False,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Full chat interface: takes a list of message dicts and returns the
    assistant's response string plus generation stats.

    Each message is ``{"role": "user"|"assistant"|"system", "content": "..."}``.
    The last message should be a user turn; all preceding turns provide
    conversation context.

    Returns ``(response_text, stats_dict)``.
    """
    if device is None:
        device = _resolve_device(model)

    template = detect_chat_template(messages, recipe)
    prompt = format_prompt(
        messages, recipe,
        template=template,
        enable_thinking=enable_thinking,
    )

    input_ids, pad_mask, prompt_len = _prepare_inputs(
        tokenizer, [prompt], device, recipe,
    )

    # Resolve stop tokens from recipe.
    eos_token_id = find_eos_token_id(tokenizer, recipe)
    think_close_id = None
    stop_on_think_close = False
    vocab = tokenizer.get_vocab()
    if recipe.mode in ("reasoning", "hybrid"):
        tc_str = recipe.think_close
        if tc_str in vocab:
            think_close_id = vocab[tc_str]
            stop_on_think_close = True

    output_ids, stats = generate(
        model, input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_token_id,
        stop_on_think_close=stop_on_think_close,
        think_close_id=think_close_id,
        pad_mask=pad_mask,
        recipe=recipe,
    )

    # Decode only the new tokens.
    new_ids = output_ids[0, prompt_len:].tolist()

    # Trim at stop tokens.
    if eos_token_id is not None and eos_token_id in new_ids:
        new_ids = new_ids[:new_ids.index(eos_token_id)]
    if (stop_on_think_close and think_close_id is not None
            and think_close_id in new_ids):
        new_ids = new_ids[:new_ids.index(think_close_id)]

    response = tokenizer.decode(new_ids, skip_special_tokens=False)
    return response, stats



# ---------------------------------------------------------------------------
# CLI runner helpers
# ---------------------------------------------------------------------------

def _resolve_device(model) -> torch.device:
    """Pick a reasonable device for batched input prep."""
    return next(model.parameters()).device


def run_prompts(
    model: TransformerForCausalLM,
    tokenizer: Tokenizer,
    recipe: TrainingRecipe,
    prompts: List[str],
    args,
) -> List[Dict]:
    """Run --prompt (one or more) through batched generation."""
    # Wrap each prompt in a messages list and format.
    formatted = []
    for p in prompts:
        msgs = []
        if args.system:
            msgs.append({"role": "system", "content": args.system})
        msgs.append({"role": "user", "content": p})
        formatted.append(format_prompt(
            msgs, recipe,
            template=args.chat_template,
            enable_thinking=args.enable_thinking,
        ))

    device = _resolve_device(model)
    completions: List[Dict] = []
    for batch_start in range(0, len(formatted), args.batch_size):
        batch = formatted[batch_start: batch_start + args.batch_size]
        results = generate_batch(
            model, tokenizer, batch,
            recipe=recipe,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=args.eos_token_id,
            seed=args.seed,
            device=device,
        )
        completions.extend(results)
    return completions


def run_prompts_file(
    model: TransformerForCausalLM,
    tokenizer: Tokenizer,
    recipe: TrainingRecipe,
    path: str,
    args,
) -> List[Dict]:
    """Stream-read a .jsonl, run each prompt, write the output .jsonl."""
    in_path = Path(path)
    out_path = Path(args.output) if args.output else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    completions = run_prompts(
        model, tokenizer, recipe, [r["prompt"] for r in records], args,
    )

    out_records: List[str] = []
    for rec, comp in zip(records, completions):
        out_records.append(json.dumps({
            "id":           rec.get("id"),
            "prompt":       comp["prompt"],
            "completion":   comp["completion"],
            "stop_reason":  comp["stop_reason"],
            "stats":        comp["stats"],
        }, ensure_ascii=False))

    if out_path is not None:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_records) + "\n")
        print(f"[infer] wrote {len(out_records)} completions to {out_path}")
    else:
        for line in out_records:
            print(line)
    return completions


def run_interactive(
    model: TransformerForCausalLM,
    tokenizer: Tokenizer,
    recipe: TrainingRecipe,
    args,
) -> None:
    """REPL: read lines, stream completions back."""
    print(f"[infer] interactive mode  (recipe: {recipe.model_name}, "
          f"mode: {recipe.mode}, template: {recipe.chat_template})")
    print("[infer] Commands: /quit, /reset, /system <text>, /thinking [on|off]")
    system = args.system
    enable_thinking = args.enable_thinking
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/reset":
            system = args.system
            enable_thinking = args.enable_thinking
            print("[infer] session reset.")
            continue
        if line.startswith("/system "):
            system = line[len("/system "):].strip()
            print(f"[infer] system message set to: {system!r}")
            continue
        if line.startswith("/thinking"):
            parts = line.split()
            if len(parts) > 1 and parts[1].lower() in ("on", "true", "1"):
                enable_thinking = True
                print("[infer] thinking enabled.")
            elif len(parts) > 1 and parts[1].lower() in ("off", "false", "0"):
                enable_thinking = False
                print("[infer] thinking disabled.")
            else:
                enable_thinking = not enable_thinking
                print(f"[infer] thinking {'enabled' if enable_thinking else 'disabled'}.")
            continue

        # Build messages and format.
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": line})

        device = _resolve_device(model)
        prompt = format_prompt(
            msgs, recipe,
            template=args.chat_template,
            enable_thinking=enable_thinking,
        )

        print("", flush=True)
        try:
            for chunk in generate_stream(
                model, tokenizer, prompt,
                recipe=recipe,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=args.eos_token_id,
                seed=args.seed,
                device=device,
            ):
                print(chunk, end="", flush=True)
        except Exception as e:
            print(f"\n[infer] generation error: {e}", file=sys.stderr)
        print(flush=True)



# ---------------------------------------------------------------------------
# Smoke test (no real checkpoint needed)
# ---------------------------------------------------------------------------

def smoke_test(checkpoint_path: Optional[str] = None) -> int:
    """
    Build a tiny model, save it, run 5-token generation on CPU.
    Returns 0 on success.  Used by ``--smoke-test``.
    """
    import tempfile
    import shutil

    print("[smoke] building tiny model...")
    tmp = tempfile.mkdtemp()
    try:
        # Build a tiny recipe.
        recipe = TrainingRecipe(mode="reasoning")

        # Build a tiny model config (same structure as ModelConfig).
        cfg = ModelConfig(
            vocab_size=512, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16,
            max_position_embeddings=128, tie_word_embeddings=True,
        )
        m = TransformerForCausalLM(cfg)
        ckpt = os.path.join(tmp, "tiny.pt")
        atomic_torch_save({"model_state": m.state_dict(), "config": vars(cfg)}, ckpt)

        # Minimal tokenizer.
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers import pre_tokenizers, decoders
        tok = Tokenizer(BPE(unk_token=None, byte_fallback=True))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        special_toks = list(recipe.special_tokens)
        trainer = BpeTrainer(
            vocab_size=512,
            special_tokens=special_toks,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tok.train_from_iterator(["hello world"], trainer=trainer)
        tok.save(os.path.join(tmp, "tokenizer.json"))

        # Load via the public surface.
        model, tokenizer, loaded_recipe = load_model_and_tokenizer(ckpt)
        model.to("cpu")

        # Run generation.
        eos_id = find_eos_token_id(tokenizer, loaded_recipe)
        results = generate_batch(
            model, tokenizer, ["hello"],
            recipe=loaded_recipe,
            max_new_tokens=5,
            temperature=0.0,
            top_k=None,
            top_p=None,
            eos_token_id=eos_id,
            device=torch.device("cpu"),
        )
        for r in results:
            print(f"[smoke] prompt={r['prompt']!r}  "
                  f"completion={r['completion']!r}  stop={r['stop_reason']}")
        print("[smoke] OK")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run inference with the Advanced LLM Framework (dense, non-MoE).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- I/O
    # Not argparse-required: --smoke-test builds its own tiny checkpoint and
    # must be runnable with no arguments.  Enforced manually in main() when
    # not in smoke-test mode.
    p.add_argument("--checkpoint", required=False,
                   help="Path to a .pt produced by training scripts, "
                        "deepspeed_shard_consolidator.py, or merge-lora.")
    p.add_argument("--recipe", default=None,
                   help="Path to a recipe.json.  If omitted, auto-detected "
                        "from the checkpoint directory.")

    # ---- Device
    p.add_argument("--device", default="auto",
                   help="'auto' (use accelerate if installed), 'cpu', or 'cuda:N'.")
    p.add_argument("--max-memory", default=None,
                   help="Per-device memory budget, e.g. '0:18GiB,cpu:30GiB'.")
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="Override max_position_embeddings in the config.")

    # ---- Quantization
    p.add_argument("--quantize", choices=["4bit", "8bit", "none"], default="none",
                   help="Quantize linear layers via bitsandbytes.")

    # ---- Input (mutually exclusive modes)
    gi = p.add_mutually_exclusive_group(required=True)
    gi.add_argument("--prompt", action="append", default=None,
                    help="One prompt.  Repeat for multiple prompts in a batch.")
    gi.add_argument("--prompts-file", default=None,
                    help="Path to a .jsonl with {'id', 'prompt'} per line.")
    gi.add_argument("--interactive", action="store_true",
                    help="REPL mode.  Commands: /quit, /reset, /system, /thinking.")
    gi.add_argument("--smoke-test", action="store_true",
                    help=argparse.SUPPRESS)

    # ---- Chat formatting
    p.add_argument("--chat-template",
                   choices=["auto", "chatml", "raw", "custom"],
                   default="auto",
                   help="How to wrap prompts.  'auto' reads from the recipe.")
    p.add_argument("--enable-thinking", action="store_true",
                    help="Open a <think> block in the assistant turn (matching "
                         "reasoning-mode recipes).")
    p.add_argument("--system", default=None,
                    help="Optional system message for every prompt.")

    # ---- Generation
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--min-new-tokens", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--eos-token-id", type=int, default=None,
                   help="Override EOS token id (default: auto-detect from recipe).")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Micro-batch size for --prompt / --prompts-file.")

    # ---- Output
    p.add_argument("--output", default=None,
                   help="Write completions to this .jsonl (with --prompts-file).")
    p.add_argument("--stream", action="store_true", default=True,
                   help="(default) Stream tokens to stdout.")
    p.add_argument("--no-stream", dest="stream", action="store_false",
                   help="Buffer full completion before printing.")

    return p



# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.smoke_test:
        return smoke_test()

    if not args.checkpoint:
        raise SystemExit(
            "infer.py: error: the following arguments are required: --checkpoint\n"
            "  (or run with --smoke-test for a self-contained test)"
        )

    # ---- Recipe
    if args.recipe:
        recipe = get_recipe(args.recipe)
    else:
        recipe = get_recipe(args.checkpoint)
    print(f"[infer] recipe: model={recipe.model_name!r}  "
          f"mode={recipe.mode!r}  template={recipe.chat_template!r}")

    # ---- Model
    print(f"[infer] loading checkpoint: {args.checkpoint}")
    t0 = time.perf_counter()
    model, tokenizer, recipe = load_model_and_tokenizer(
        args.checkpoint,
        device=args.device,
        quantize=args.quantize if args.quantize != "none" else None,
        max_seq_len=args.max_seq_len,
    )
    print(f"[infer]   model loaded in {time.perf_counter()-t0:.1f}s  "
          f"({count_parameters(model)/1e9:.3f}B params, "
          f"hidden={model.config.hidden_size}, "
          f"layers={model.config.num_hidden_layers})")

    # ---- Tokenizer
    print(f"[infer]   tokenizer: vocab={tokenizer.get_vocab_size()}  "
          f"template={recipe.chat_template}")

    # ---- Precision (bf16 default, before device placement)
    if args.quantize == "none":
        model.to(torch.bfloat16)
        print(f"[infer]   dtype=bfloat16 (default)")

    # ---- Quantization
    if args.quantize != "none":
        model = maybe_quantize(model, args.quantize)

    # ---- Device placement
    model, mode = resolve_device_map(
        model, args.device, args.max_memory,
    )

    # ---- Dispatch
    if args.interactive:
        run_interactive(model, tokenizer, recipe, args)
    elif args.prompts_file is not None:
        run_prompts_file(model, tokenizer, recipe, args.prompts_file, args)
    else:
        # default: --prompt (one or more)
        completions = run_prompts(model, tokenizer, recipe, args.prompt, args)
        for c in completions:
            print(f"[prompt]   {c['prompt']}")
            print(f"[answer]   {c['completion']}")
            print(f"[stop]     {c['stop_reason']}")
            s = c['stats']
            print(f"[stats]    tokens={s['generated_tokens']}  "
                  f"time={s['elapsed_seconds']}s  "
                  f"speed={s['tokens_per_second']} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
