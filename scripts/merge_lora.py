"""Standalone LoRA merge utility.

Loads a pretrained base checkpoint, injects LoRA with the same hyper-parameters
used during training, applies the trained LoRA deltas on top, then strips the
LoRA wrappers and saves a single merged state_dict.

This is the same logic `train_sft.py --merge-and-save` runs internally; we keep
a standalone copy here so users can re-merge a `*_lora.pt` snapshot after
training without rerunning the trainer.

Usage:
    python scripts/merge_lora.py \
        --base path/to/pretrain/latest.pt \
        --lora-pt path/to/checkpoints/sft_step0000100_lora.pt \
        --config path/to/pretrain/config.json \
        --out merged_model.pt \
        [--lora-rank 64] [--lora-alpha 128.0] \
        [--target-modules q_proj,k_proj,v_proj,o_proj]
"""
import argparse
import json
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import ModelConfig, TransformerForCausalLM  # noqa: E402
from atomic_io import load_torch_checkpoint            # noqa: E402
from peft.lora import inject_lora, merge_lora           # noqa: E402

_LORA_TARGET_SUFFIXES = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight",
    "o_proj.weight", "gate_proj.weight", "up_proj.weight",
    "down_proj.weight",
)


def _remap_for_lora(base_sd, model_sd):
    """Route plain `q_proj.weight` → `q_proj.base.weight` for LoRA-wrapped
    modules; pass everything else through unchanged."""
    out = {}
    for k, v in base_sd.items():
        if k.endswith(_LORA_TARGET_SUFFIXES):
            target_k = k[: -len("weight")] + "base.weight"
            if target_k in model_sd:
                out[target_k] = v
                continue
        if k in model_sd:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True,
                    help="Pretrained checkpoint (.pt) containing 'model_state'")
    ap.add_argument("--lora-pt", required=True,
                    help="LoRA adapter checkpoint produced during SFT/DPO/GRPO")
    ap.add_argument("--config", required=True,
                    help="Path to a config.json (or any ModelConfig JSON)")
    ap.add_argument("--out", required=True,
                    help="Output path for the merged model .pt")
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-alpha", type=float, default=128.0)
    ap.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--lora-type", default="lora", choices=["lora", "dora"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.config) as f:
        cfg_dict = json.load(f)
    config = ModelConfig(**cfg_dict)
    print(f"[merge_lora] config: layers={config.num_hidden_layers} "
          f"hidden={config.hidden_size} vocab={config.vocab_size}")

    model = TransformerForCausalLM(config).to(device)

    inject_lora(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        target_modules=tuple(args.target_modules.split(",")),
        lora_type=args.lora_type,
    )
    print("[merge_lora] LoRA wrappers injected")

    base_ck = load_torch_checkpoint(args.base, map_location=device,
                                    allow_unsafe=True)
    base_sd = base_ck["model_state"]
    print(f"[merge_lora] base state_dict: {len(base_sd)} keys")

    remapped = _remap_for_lora(base_sd, model.state_dict())
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    print(f"[merge_lora] base load: missing={len(missing)} "
          f"unexpected={len(unexpected)}")

    lora_ck = load_torch_checkpoint(args.lora_pt, map_location=device,
                                    allow_unsafe=True)
    lora_sd = lora_ck.get("model_state", lora_ck) if isinstance(lora_ck, dict) \
        else lora_ck
    lora_only = {k: v for k, v in lora_sd.items() if k in model.state_dict()}
    missing, unexpected = model.load_state_dict(lora_only, strict=False)
    print(f"[merge_lora] lora load: missing={len(missing)} "
          f"unexpected={len(unexpected)}")

    _ = merge_lora(model)
    if hasattr(model, "tie_weights"):
        model.tie_weights()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg_dict}, args.out)
    print(f"[merge_lora] saved merged model -> {args.out}")


if __name__ == "__main__":
    main()