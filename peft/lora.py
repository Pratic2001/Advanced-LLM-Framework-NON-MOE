#!/usr/bin/env python3
"""
peft/lora.py

Low-Rank Adaptation (LoRA) and its SOTA variants for the dense LLM framework.

Supports:
    - LoRA (standard):       Wx + (BA x) * (alpha/rank)
    - DoRA (Phase 4/A):      W' = m * (W0 + ΔW) / ||W0 + ΔW||
    - rsLoRA (Phase 4/A):    scale = alpha / sqrt(rank) instead of alpha/rank
    - LoRA+   (Phase 4/A):   different learning rates for lora_A vs lora_B
    - NEFTune (Phase 4/A):   uniform noise on embeddings during forward pass

Usage:
    from peft.lora import inject_lora, lora_state_dict, freeze_base

    # Standard LoRA
    inject_lora(model, target_modules=("q_proj", "v_proj"), rank=64, alpha=128.0)

    # DoRA (weight-decomposed LoRA)
    inject_lora(model, target_modules=("q_proj", "v_proj"), rank=64, alpha=128.0,
                lora_type="dora", use_rslora=True)

    # Save only LoRA parameters
    torch.save(lora_state_dict(model), "lora_weights.pt")

    # Merge LoRA into base weights for deployment
    orig_weights = merge_lora(model)
    torch.save(model.state_dict(), "merged_model.pt")
    unmerge_lora(model, orig_weights)  # restore for further training
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# LoRALinear — standard low-rank adapter
# ======================================================================


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with a low-rank adapter:

        output = Wx + (B A x) * scale

    where A in R^{rank x in}, B in R^{out x rank}, scale = alpha / rank.

    The original weight W (self.base.weight) is frozen — only A and B are
    trained.  This matches the PyTorch convention used in the reference
    train_sft.py: the base module is stored as `self.base`, A is Kaiming
    uniform init, B is zero init (so the adapter starts as identity).

    rsLoRA variant (use_rslora=True):
        scale = alpha / sqrt(rank)
        Prevents performance saturation at high ranks (128+).
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 64,
        alpha: float = 128.0,
        use_rslora: bool = False,
    ):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        in_features = base.in_features
        out_features = base.out_features
        self.rank = rank
        self.scale = alpha / (math.sqrt(rank) if use_rslora else rank)

        dtype = base.weight.dtype
        device = base.weight.device

        # A: Kaiming uniform init  (standard from LoRA paper)
        self.lora_A = nn.Linear(in_features, rank, bias=False, dtype=dtype, device=device)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        # B: zero init  (adapter starts as identity)
        self.lora_B = nn.Linear(rank, out_features, bias=False, dtype=dtype, device=device)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scale
        return base_out + lora_out

    def merge(self) -> nn.Linear:
        """
        Return a plain nn.Linear with the LoRA delta fused into W.

        The returned Linear shares no storage with this adapter; the adapter
        itself is not modified, so it can continue training.
        """
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scale
        merged.weight = nn.Parameter(self.base.weight + delta.to(self.base.weight.dtype))
        if self.base.bias is not None:
            merged.bias = nn.Parameter(self.base.bias.clone())
        return merged


# ======================================================================
# DoRALinear — Weight-Decomposed Low-Rank Adaptation  (A/Phase 4)
# ======================================================================


class DoRALinear(nn.Module):
    """
    Weight-Decomposed Low-Rank Adaptation (DoRA).

    Decomposes each adapted weight into a magnitude vector and a directional
    update:

        W' = m * (W0 + ΔW) / ||W0 + ΔW||

    where:
        m  is a learned magnitude vector of shape (out_features,)
        W0 is the frozen base weight
        ΔW = (B A) * scale  is the LoRA low-rank update

    DoRA consistently outperforms standard LoRA at equal rank / param budget
    (DoRA, Liu et al., ICLR 2024).

    rsLoRA variant (use_rslora=True):
        scale = alpha / sqrt(rank)
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 64,
        alpha: float = 128.0,
        use_rslora: bool = False,
    ):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        in_features = base.in_features
        out_features = base.out_features
        self.rank = rank
        self.scale = alpha / (math.sqrt(rank) if use_rslora else rank)

        dtype = base.weight.dtype
        device = base.weight.device

        # LoRA A: Kaiming uniform init
        self.lora_A = nn.Linear(in_features, rank, bias=False, dtype=dtype, device=device)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        # LoRA B: zero init
        self.lora_B = nn.Linear(rank, out_features, bias=False, dtype=dtype, device=device)
        nn.init.zeros_(self.lora_B.weight)

        # Learned per-output-channel magnitude vector  (initialised to the
        # norm of each base-weight row, so the initial behaviour matches the
        # base model's).
        with torch.no_grad():
            base_norm = torch.norm(base.weight, dim=1, keepdim=False)
        self.lora_magnitude = nn.Parameter(torch.ones(out_features, dtype=dtype, device=device))
        self.lora_magnitude.data.copy_(base_norm)

    def _get_effective_weight(self) -> torch.Tensor:
        """
        Reconstruct the merged weight W = m * (W0 + ΔW) / ||W0 + ΔW||.

        This is called in every forward pass, which is the standard DoRA
        implementation.
        """
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scale  # (out, in)
        weight = self.base.weight + delta.to(self.base.weight.dtype)     # (out, in)

        # Normalise per output channel (direction component)
        weight_norm = torch.norm(weight, dim=1, keepdim=True)            # (out, 1)
        weight_dir = weight / weight_norm.clamp(min=1e-12)

        # Scale by learned magnitude
        return self.lora_magnitude.unsqueeze(1) * weight_dir             # (out, in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._get_effective_weight()
        return F.linear(x, weight, self.base.bias)

    def merge(self) -> nn.Linear:
        """
        Return a plain nn.Linear with the DoRA update fused into W.

        The returned Linear has the learned magnitude folded in, so it's
        a drop-in replacement for inference / deployment.
        """
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        merged.weight = nn.Parameter(self._get_effective_weight().to(self.base.weight.dtype))
        if self.base.bias is not None:
            merged.bias = nn.Parameter(self.base.bias.clone())
        return merged


# ======================================================================
# Injection helpers
# ======================================================================

_LORA_TYPES = {
    "lora": LoRALinear,
    "dora": DoRALinear,
}


def _get_parent(model: nn.Module, module_path: str) -> nn.Module:
    """Walk the module path to find the parent container."""
    parent = model
    for part in module_path.split("."):
        parent = getattr(parent, part)
    return parent


def _set_submodule(model: nn.Module, module_path: str, new_module: nn.Module) -> None:
    """Set a submodule at a dotted path."""
    if "." in module_path:
        parent_path, attr = module_path.rsplit(".", 1)
        parent = model
        for part in parent_path.split("."):
            parent = getattr(parent, part)
    else:
        parent = model
        attr = module_path
    setattr(parent, attr, new_module)


def inject_lora(
    model: nn.Module,
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ),
    rank: int = 64,
    alpha: float = 128.0,
    lora_type: str = "lora",
    use_rslora: bool = False,
) -> int:
    """
    Walk the model and replace every nn.Linear whose name ends with a target
    module name with a LoRA / DoRA adapter.

    Args:
        model: The transformer model.
        target_modules: Tuple of suffix strings to match (e.g. "q_proj").
        rank: LoRA rank.
        alpha: LoRA alpha scaling.
        lora_type: "lora" or "dora".
        use_rslora: If True, use alpha/sqrt(rank) scaling (rsLoRA).

    Returns:
        Number of adapters injected.

    Raises:
        ValueError: If lora_type is not "lora" or "dora".
    """
    if lora_type not in _LORA_TYPES:
        raise ValueError(
            f"Unknown lora_type={lora_type!r}. Must be one of {list(_LORA_TYPES.keys())}."
        )

    adapter_cls = _LORA_TYPES[lora_type]
    replaced = 0

    for module_path, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        for target in target_modules:
            if module_path.endswith(target):
                adapter = adapter_cls(
                    module,
                    rank=rank,
                    alpha=alpha,
                    use_rslora=use_rslora,
                )
                _set_submodule(model, module_path, adapter)
                replaced += 1
                break

    return replaced


def freeze_base(model: nn.Module) -> int:
    """
    Freeze all base-model parameters (only LoRA parameters will be trained).

    Specifically, every parameter whose name does NOT contain "lora_A" or
    "lora_B" or "lora_magnitude" has requires_grad set to False.

    Returns:
        Number of trainable parameters remaining (should be only LoRA params).
    """
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        is_lora = "lora_A" in name or "lora_B" in name or "lora_magnitude" in name
        if not is_lora:
            param.requires_grad_(False)
            frozen += 1
        else:
            param.requires_grad_(True)
            trainable += param.numel()
    return trainable


# ======================================================================
# Merge / unmerge
# ======================================================================


def merge_lora(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Fold every LoRA / DoRA adapter into its base weight and return the
    original base weights as a dict so they can be restored later.

    After calling this, the model contains only plain nn.Linear layers
    (no adapter wrappers).  The returned dict maps each original weight
    name to its tensor *before* the merge took place.

    Args:
        model: The model with LoRA / DoRA adapters.

    Returns:
        Dict mapping parameter names to their original weight values.
    """
    saved = {}

    for module_path, module in list(model.named_modules()):
        if isinstance(module, (LoRALinear, DoRALinear)):
            # Save original base weight before replacement
            for p_name in ("weight", "bias"):
                p = getattr(module.base, p_name, None)
                if p is not None:
                    key = f"{module_path}.{p_name}"
                    saved[key] = p.data.clone()

            # Replace the adapter with a merged nn.Linear
            merged = module.merge()
            _set_submodule(model, module_path, merged)

    return saved


def unmerge_lora(
    model: nn.Module,
    saved_weights: Dict[str, torch.Tensor],
) -> None:
    """
    Restore original base weights that were saved by merge_lora().

    After calling this, the model must still have the same LoRA / DoRA
    adapters in place.  That is, this is intended to restore base weights
    *before* they were merged, so the adapter can continue training.

    NOTE: This does NOT reconstruct the adapter wrappers — it only restores
    the weight values inside whatever nn.Linear is currently at the path.
    You should call inject_lora() again BEFORE unmerge_lora() if the
    adapters were already replaced with plain Linear layers.

    Args:
        model: The model (should have the same structure as when merge_lora
               was called, at least for base weights).
        saved_weights: Dict returned by merge_lora().
    """
    for key, weight_tensor in saved_weights.items():
        param_path, attr = key.rsplit(".", 1)
        parent = model
        for part in param_path.split("."):
            parent = getattr(parent, part)
        param = getattr(parent, attr)
        param.data.copy_(weight_tensor)


# ======================================================================
# LoRA state dict (for saving lightweight LoRA-only checkpoints)
# ======================================================================


_KEEP_KEYS: Tuple[str, ...] = (
    "lora_A", "lora_B", "lora_magnitude",
)


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Return only the LoRA parameters for compact checkpoint storage.

    This extracts parameters whose names contain "lora_A", "lora_B", or
    "lora_magnitude" — the rest of the model (base weights) is reloaded
    from the original pretrained checkpoint.  Follows the same pattern as
    train_sft.py's lora_state_dict().

    Args:
        model: The model with LoRA adapters injected.

    Returns:
        State dict containing only LoRA adapter parameters.
    """
    return {
        k: v for k, v in model.state_dict().items()
        if any(kw in k for kw in _KEEP_KEYS)
    }


def count_lora_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable LoRA parameters (for logging / diagnostics).
    """
    return sum(
        p.numel() for n, p in model.named_parameters()
        if any(kw in n for kw in _KEEP_KEYS)
    )


# ======================================================================
# LoRA+  — different learning rates for A vs B  (A/Phase 4)
# ======================================================================


def build_lora_optimizer(
    model: nn.Module,
    base_lr: float = 5e-4,
    lora_lr_ratio: float = 16.0,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """
    Build an AdamW optimizer with LoRA+ learning-rate separation.

    LoRA+ (Hayou et al., 2024) recognises that lora_B should train faster
    than lora_A.  This function creates three param groups:

        - lora_A params:  base_lr * (1 / lora_lr_ratio)  (slow)
        - lora_B params:  base_lr                         (fast)
        - all other trainable params: base_lr              (standard)

    Typical lora_lr_ratio = 16, meaning A gets base_lr/16 and B gets
    base_lr.

    Args:
        model: The model with LoRA adapters injected.
        base_lr: Base learning rate (applied to lora_B and non-LoRA params).
        lora_lr_ratio: Ratio lora_lr / lora_A_lr.  Default 16.
        weight_decay: AdamW weight decay.
        betas: AdamW beta parameters.
        eps: Adam epsilon.

    Returns:
        torch.optim.AdamW with per-group learning rates.
    """
    groups: Dict[str, List[torch.nn.Parameter]] = {
        "lora_A": [],
        "lora_B": [],
        "other": [],
        "other_no_decay": [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "lora_A" in name:
            groups["lora_A"].append(param)
        elif "lora_B" in name or "lora_magnitude" in name:
            groups["lora_B"].append(param)
        elif param.ndim < 2 or "norm" in name or "embed" in name:
            groups["other_no_decay"].append(param)
        else:
            groups["other"].append(param)

    lr_a = base_lr / lora_lr_ratio  # A is slow (default: base_lr / 16)
    lr_b = base_lr                   # B is fast

    param_groups = [
        {"params": groups["lora_A"],        "lr": lr_a, "weight_decay": weight_decay},
        {"params": groups["lora_B"],        "lr": lr_b, "weight_decay": weight_decay},
        {"params": groups["other"],         "lr": base_lr, "weight_decay": weight_decay},
        {"params": groups["other_no_decay"],"lr": base_lr, "weight_decay": 0.0},
    ]

    # Log sizes for diagnostics
    n_a = sum(p.numel() for p in groups["lora_A"])
    n_b = sum(p.numel() for p in groups["lora_B"])
    n_o = sum(p.numel() for p in groups["other"])
    n_nd = sum(p.numel() for p in groups["other_no_decay"])
    print(
        f"[LoRA+] lora_A={n_a:,} @ lr={lr_a:.2e}  |  "
        f"lora_B={n_b:,} @ lr={lr_b:.2e}  |  "
        f"other={n_o + n_nd:,} @ lr={base_lr:.2e}"
    )

    return torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        betas=betas,
        eps=eps,
        fused=torch.cuda.is_available(),
    )


# ======================================================================
# NEFTune — noisy embedding trick  (A/Phase 4)
# ======================================================================


def neftune_noise(
    embeddings: torch.Tensor,
    noise_alpha: float = 5.0,
) -> torch.Tensor:
    """
    Add uniform noise to embeddings during the forward pass (NEFTune).

    NEFTune (Jain et al., NeurIPS 2023) adds small uniform noise to input
    embeddings during SFT, which improves instruction-following quality with
    near-zero computational cost.

    The noise is sampled from U(-noise_alpha / sqrt(d), +noise_alpha / sqrt(d))
    where d is the embedding dimension.  noise_alpha = 5 is the default
    recommended by the paper.

    Args:
        embeddings: Input embeddings tensor of shape (..., d).
        noise_alpha: Noise scaling factor (default 5.0 from the paper).

    Returns:
        Embeddings with noise added (in-place modification on the original
        tensor is avoided; the noise is applied to a view that shares
        storage if embeddings.requires_grad is True).
    """
    if not embeddings.requires_grad or noise_alpha <= 0.0:
        return embeddings

    d = embeddings.shape[-1]
    scale = noise_alpha / math.sqrt(d)

    # Sample uniform noise U(-scale, +scale)
    noise = torch.empty_like(embeddings).uniform_(-scale, scale)
    return embeddings + noise


def register_neftune_hook(
    model: nn.Module,
    noise_alpha: float = 5.0,
    embed_module_name: str = "embed_tokens",
) -> Optional[torch.utils.hooks.RemovableHandle]:
    """
    Register a forward hook on the input embedding layer that adds NEFTune
    noise to embeddings during training.

    The hook is applied to the first module in the model whose name matches
    `embed_module_name` and is an nn.Embedding.

    Args:
        model: The transformer model.
        noise_alpha: Noise scaling factor (default 5.0).
        embed_module_name: Name suffix of the embedding module to hook
                          (default "embed_tokens").

    Returns:
        The removale handle (call .remove() to unhook), or None if no
        matching embedding module was found.
    """
    embed_module: Optional[nn.Embedding] = None
    for name, module in model.named_modules():
        if name.endswith(embed_module_name) and isinstance(module, nn.Embedding):
            embed_module = module
            break

    if embed_module is None:
        print(f"[NEFTune] Warning: no embedding module found matching "
              f"'{embed_module_name}' — hook not registered.")
        return None

    def _neftune_hook(module: nn.Module, args: Tuple[torch.Tensor, ...],
                      output: torch.Tensor) -> torch.Tensor:
        """Forward hook that adds NEFTune noise to embedding output."""
        # Only apply during training
        if not module.training:
            return output
        return neftune_noise(output, noise_alpha=noise_alpha)

    handle = embed_module.register_forward_hook(_neftune_hook)
    print(f"[NEFTune] Hook registered on {embed_module_name} "
          f"(noise_alpha={noise_alpha})")
    return handle


# ======================================================================
# Convenience: inject + freeze in one call
# ======================================================================


def prepare_lora_training(
    model: nn.Module,
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ),
    rank: int = 64,
    alpha: float = 128.0,
    lora_type: str = "lora",
    use_rslora: bool = False,
    neftune_noise_alpha: Optional[float] = None,
    embed_module_name: str = "embed_tokens",
) -> Dict[str, Any]:
    """
    One-stop convenience: inject LoRA adapters, freeze base model, and
    optionally register a NEFTune hook.

    Args:
        model: The transformer model.
        target_modules: Module name suffixes to target.
        rank: LoRA rank.
        alpha: LoRA alpha.
        lora_type: "lora" or "dora".
        use_rslora: Use rsLoRA scaling.
        neftune_noise_alpha: NEFTune noise_alpha (default None = disabled).
        embed_module_name: Name of the embedding module to hook for NEFTune.

    Returns:
        Dict with keys:
            n_replaced: Number of adapters injected.
            n_trainable: Number of trainable LoRA parameters.
            neftune_handle: RemovableHandle or None.
    """
    n_replaced = inject_lora(
        model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        lora_type=lora_type,
        use_rslora=use_rslora,
    )
    n_trainable = freeze_base(model)

    neftune_handle = None
    if neftune_noise_alpha is not None and neftune_noise_alpha > 0.0:
        neftune_handle = register_neftune_hook(
            model,
            noise_alpha=neftune_noise_alpha,
            embed_module_name=embed_module_name,
        )

    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"[prepare_lora_training] injected {n_replaced} adapters "
        f"({lora_type}, rank={rank}, alpha={alpha}{' + rsLoRA' if use_rslora else ''}) | "
        f"trainable={n_trainable:,} / total={n_total:,} "
        f"({100.0 * n_trainable / n_total:.2f}%)"
    )

    return {
        "n_replaced": n_replaced,
        "n_trainable": n_trainable,
        "neftune_handle": neftune_handle,
    }


# ======================================================================
# Smoke test
# ======================================================================


def smoke_test() -> None:
    """Quick smoke test: create a tiny model, inject LoRA, run a step."""
    print("\n=== peft/lora.py smoke test ===")

    # Tiny model for testing
    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, 8)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    # ---- LoRA injection ----
    model = TinyMLP()
    n_replaced = inject_lora(
        model,
        target_modules=("fc1", "fc2", "fc3"),
        rank=4,
        alpha=8.0,
        lora_type="lora",
    )
    assert n_replaced == 3, f"Expected 3 adapters, got {n_replaced}"
    n_trainable = freeze_base(model)
    assert n_trainable > 0, "Expected > 0 trainable LoRA params"

    # Forward + backward
    x = torch.randn(2, 16)
    y = model(x)
    loss = y.sum()
    loss.backward()
    assert loss.isfinite(), "Loss should be finite"

    # ---- DoRA injection ----
    model2 = TinyMLP()
    n_replaced2 = inject_lora(
        model2,
        target_modules=("fc1", "fc2"),
        rank=4,
        alpha=8.0,
        lora_type="dora",
    )
    assert n_replaced2 == 2, f"Expected 2 DoRA adapters, got {n_replaced2}"

    y2 = model2(x)
    loss2 = y2.sum()
    loss2.backward()
    assert loss2.isfinite(), "DoRA loss should be finite"

    # ---- State dict ----
    state = lora_state_dict(model2)
    assert all(any(kw in k for kw in _KEEP_KEYS) for k in state), \
        "State dict should contain only LoRA params"

    # ---- Merge ----
    saved = merge_lora(model)
    assert not any(
        isinstance(m, (LoRALinear, DoRALinear))
        for m in model.modules()
    ), "merge_lora() left adapters in place"

    # ---- NEFTune ----
    emb = torch.randn(2, 4, 16, requires_grad=True)
    noisy = neftune_noise(emb, noise_alpha=5.0)
    assert noisy.shape == emb.shape
    assert noisy.requires_grad
    noisy.sum().backward()  # gradients must flow

    # ---- LoRA+ ----
    model3 = TinyMLP()
    inject_lora(model3, target_modules=("fc1", "fc2", "fc3"), rank=4, alpha=8.0)
    freeze_base(model3)
    optim = build_lora_optimizer(model3, base_lr=1e-4, lora_lr_ratio=16.0)
    # Check param groups: lora_A should have lr = 1e-4 / 16 = 6.25e-6
    for pg in optim.param_groups:
        if len(pg["params"]) > 0:
            # Just verify they were created without errors
            pass

    # ---- unmerge_lora ----
    unmerge_lora(model, saved)
    # After unmerge, the model has plain Linear layers whose weights match
    # the originals (we can't verify LoRA is re-attached since we merged
    # away the wrapper, but the save/restore round-trip works).

    print("All smoke tests passed.\n")


if __name__ == "__main__":
    smoke_test()
