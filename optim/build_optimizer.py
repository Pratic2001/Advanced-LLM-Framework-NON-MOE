#!/usr/bin/env python3
"""
optim/build_optimizer.py

Optimizer construction for the dense LLM framework.

Supported optimizers:
    adamw — baseline AdamW (default, for all parameter types)
    muon  — Muon for 2D matmul weights + AdamW for 1D/embeddings/norms
            (A/Phase 3: highest-profile pretraining-speed result)

Usage:
    from optim.build_optimizer import build_optimizer
    optimizer = build_optimizer(model, optimizer_type="muon", lr=5e-4)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Param-group classification helpers
# ---------------------------------------------------------------------------


def classify_params(
    model: nn.Module,
    weight_decay: float = 0.1,
) -> Dict[str, List]:
    """
    Classify parameters into groups for AdamW / Muon.

    Groups:
        decay:    2D matmul weights (≥2D, not norm/embed/lora_B)
        no_decay: 1D params, norms, embeddings, lora_B
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name or "embed" in name or "lora_B" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return {"decay": decay, "no_decay": no_decay}


def classify_for_muon(model: nn.Module) -> Dict[str, List]:
    """
    Classify parameters for Muon.

    Muon groups:
        muon_group: 2D matmul weights (attention projections + MLP) —
                    square(ish) matrices that benefit from orthogonalisation
        adamw_group: everything else (embeddings, norms, biases, 1D tensors,
                    lora_B) — stay with AdamW
    """
    muon_group = []
    adamw_group = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Muon targets: 2D weight matrices that are NOT embeddings/norms/lora_B
        is_2d_matmul = (
            param.ndim == 2
            and "norm" not in name
            and "embed" not in name
            and "lora_B" not in name
        )
        if is_2d_matmul:
            muon_group.append(param)
        else:
            adamw_group.append(param)
    return {"muon": muon_group, "adamw": adamw_group}


# ---------------------------------------------------------------------------
# AdamW builder (baseline)
# ---------------------------------------------------------------------------


def build_adamw(
    model: nn.Module,
    lr: float = 5e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """
    Build AdamW with decay/no-decay param groups.

    This matches the reference Qwen3 repo's build_optimizer:
        - norms, embeddings, biases, lora_B → weight_decay=0
        - everything else → weight_decay as given
    """
    groups = classify_params(model, weight_decay)
    param_groups = [
        {"params": groups["decay"],    "weight_decay": weight_decay},
        {"params": groups["no_decay"], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        fused=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Muon optimizer
# ---------------------------------------------------------------------------

try:
    import Muon  # type: ignore
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False


def build_muon(
    model: nn.Module,
    lr: float = 5e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    muon_lr: Optional[float] = None,
    muon_momentum: float = 0.95,
    muon_nesterov: bool = True,
    muon_ns_steps: int = 5,
) -> Tuple[torch.optim.Optimizer, torch.optim.AdamW]:
    """
    Build Muon for 2D matmul weights + AdamW for everything else.

    Muon applies orthogonal updates (Newton-Schulz iterations) to 2D weight
    matrices, which provably accelerates convergence for linear layers.
    AdamW handles embeddings, norms, and 1D params that Muon doesn't target.

    Args:
        model: The transformer model
        lr: Base learning rate (used for AdamW groups)
        weight_decay: Weight decay for AdamW
        muon_lr: Separate LR for Muon (default: same as lr)
        muon_momentum: Momentum coefficient for Muon
        muon_nesterov: Use Nesterov momentum
        muon_ns_steps: Number of Newton-Schulz iterations

    Returns:
        (muon_optimizer, adamw_optimizer) — optimizers that must be stepped
        together in the training loop.
    """
    if not MUON_AVAILABLE:
        raise ImportError(
            "Muon optimizer not installed. Install with:\n"
            "  pip install Muon\n"
            "Falling back to AdamW for all parameters."
        )

    groups = classify_for_muon(model)
    effective_muon_lr = muon_lr if muon_lr is not None else lr

    # Muon for 2D matmul weights
    muon_optim = Muon.Muon(
        groups["muon"],
        lr=effective_muon_lr,
        momentum=muon_momentum,
        nesterov=muon_nesterov,
        ns_steps=muon_ns_steps,
    )

    # AdamW for embeddings, norms, biases, etc.
    adamw_optim = torch.optim.AdamW(
        groups["adamw"],
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        fused=torch.cuda.is_available(),
    )

    return muon_optim, adamw_optim


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 5e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    **kwargs,
) -> Any:
    """
    Build an optimizer for training.

    Args:
        model: The transformer model
        optimizer_type: "adamw" or "muon"
        lr: Peak learning rate
        weight_decay: Weight decay
        betas: AdamW beta parameters
        eps: Adam epsilon

    Returns:
        For "adamw": a single torch.optim.AdamW
        For "muon": a tuple (muon_optimizer, adamw_optimizer)
    """
    if optimizer_type == "muon":
        return build_muon(
            model, lr=lr, weight_decay=weight_decay,
            betas=betas, eps=eps, **kwargs,
        )
    elif optimizer_type == "adamw":
        return build_adamw(model, lr=lr, weight_decay=weight_decay,
                           betas=betas, eps=eps)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type!r}")
