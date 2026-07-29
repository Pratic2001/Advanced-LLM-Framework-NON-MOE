#!/usr/bin/env python3
"""
model.py

Generic dense transformer architecture supporting:
    - RMSNorm pre-norm with embed scaling (1/sqrt(hidden_size))
    - Grouped-Query Attention (GQA) + optional QK-Norm (RMSNorm on Q/K before RoPE)
    - Rotary Position Embeddings (RoPE)
    - SwiGLU or GELU MLP
    - FlashAttention-2/3 through F.scaled_dot_product_attention
    - Fully configurable via ModelConfig.from_target_size() auto-sizing search
    - Component-choice fields: norm_type, mlp_type, use_qk_norm, attn_type

Usage:
    cfg = ModelConfig.from_target_size(target_params=1_700_000_000)   # 1.7B
    cfg = ModelConfig.from_target_size(target_params=70_000_000_000)  # 70B
    cfg = ModelConfig.from_target_size(target_params=1_000_000_000_000)  # 1T
    model = TransformerForCausalLM(cfg)
"""

from __future__ import annotations

import math
import re
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Configuration
# ======================================================================


class ModelConfig:
    """
    Configuration for the dense transformer architecture.

    Component-choice fields (B4) let users pick non-default variants:
        norm_type:  "rmsnorm" (default) or "layernorm"
        mlp_type:   "swiglu"  (default) or "gelu"
        use_qk_norm: bool (default True) — RMSNorm on Q/K before RoPE
        attn_type:  "gqa"    (default) or "mha"

    Extra architectural fields (copy from reference Qwen3):
        scale_emb: bool — whether to scale embeddings by 1/sqrt(hidden_size)
        tie_word_embeddings: bool — tie lm_head to embed_tokens weights
    """

    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        rope_scaling: Optional[Dict] = None,
        tie_word_embeddings: bool = True,
        scale_emb: bool = True,
        # Component-choice fields
        norm_type: str = "rmsnorm",
        mlp_type: str = "swiglu",
        use_qk_norm: bool = True,
        attn_type: str = "gqa",
        # Optional dropout
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        # Weight init
        init_std: float = 0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.tie_word_embeddings = tie_word_embeddings
        self.scale_emb = scale_emb
        self.norm_type = norm_type
        self.mlp_type = mlp_type
        self.use_qk_norm = use_qk_norm
        self.attn_type = attn_type
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.init_std = init_std

        # Derived
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        # ----------------------------------------------------------
        # Variant architecture fields (all with defaults for backward compat)
        # ----------------------------------------------------------
        self.arch_type: str = "dense"              # "dense" | "jamba"
        self.layer_type: str = "sequential"        # "sequential" | "parallel"
        self.sliding_window_size: int = 0          # 0=disabled, >0=window
        self.use_mla: bool = False
        self.kv_lora_rank: Optional[int] = None    # latent dim for MLA
        self.num_mtp_heads: int = 0                # multi-token prediction heads
        self.mtp_discount: float = 0.5             # discount per future token
        self.mod_alpha: float = 0.0                # mixture-of-depth threshold (0=off)
        self.mod_loss_weight: float = 0.01         # MoD aux loss weight
        self.jamba_hybrid_layer_interval: int = 4  # Mamba every N layers
        # rope_scaling is already set above (line 90)

        # Absorb any extra kwargs (forward compat)
        for k, v in kwargs.items():
            setattr(self, k, v)

    # ------------------------------------------------------------------
    # Auto-sizing search
    # ------------------------------------------------------------------

    @staticmethod
    def _param_count(
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
    ) -> int:
        """Closed-form non-embedding parameter count."""
        # Embedding + lm_head are excluded (tied)
        # Attention:  Q, K, V, O projections
        attn_per_layer = (
            hidden_size * num_attention_heads * head_dim       # Q
            + hidden_size * num_key_value_heads * head_dim     # K
            + hidden_size * num_key_value_heads * head_dim     # V
            + num_attention_heads * head_dim * hidden_size     # O
        )
        # MLP: gate + up + down projections (SwiGLU: 3 matrices)
        if intermediate_size > 0:
            mlp_per_layer = 3 * hidden_size * intermediate_size
        else:
            mlp_per_layer = 0
        # RMSNorm: 2 per layer (pre-attn, pre-mlp)
        norm_per_layer = 2 * hidden_size
        # Final RMSNorm
        final_norm = hidden_size

        total = (
            num_hidden_layers * (attn_per_layer + mlp_per_layer + norm_per_layer)
            + final_norm
        )
        return total

    @staticmethod
    def _quality_score(
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        target_params: int,
    ) -> float:
        """
        Score a candidate architecture: lower is better.
        Uses asymmetric param error (overshoot penalised 1.5×),
        MLP aspect-ratio check, KV-head divisibility, and
        a mild depth preference (deeper = better at same param count).

        ``ref_layers`` scales with hidden_size so that large models
        (100B+) aren't penalised for having proportionally more layers.
        """
        n_params = ModelConfig._param_count(
            vocab_size, hidden_size, intermediate_size,
            num_hidden_layers, num_attention_heads,
            num_key_value_heads, head_dim,
        )
        # Asymmetric param error: overshooting costs 50 % more than undershooting
        ratio = n_params / max(1, target_params)
        if ratio > 1.0:
            param_err = (ratio - 1.0) * 1.5
        else:
            param_err = (1.0 - ratio) * 1.0
        # MLP ratio penalty — target 2.75× for SwiGLU (standard practice)
        mlp_ratio = intermediate_size / max(1, hidden_size)
        mlp_penalty = (mlp_ratio - 2.75) ** 2
        # KV-head divisibility — must divide attention heads evenly
        kv_penalty = 0.0
        if num_attention_heads % num_key_value_heads != 0:
            kv_penalty = 0.3
        # Depth preference: deeper models learn better per-param (at equal param count)
        # Scale ref_layers with hidden_size: calibrated at L=28 for H=4096
        ref_layers = max(4, min(200, hidden_size // 146))
        if num_hidden_layers >= ref_layers:
            depth_score = -math.log(num_hidden_layers / ref_layers) * 0.03  # reward deeper
        else:
            depth_score = math.log(ref_layers / num_hidden_layers) * 0.05   # penalty shallower
        # Total — param error dominates
        return param_err + 0.1 * mlp_penalty + 0.5 * kv_penalty + depth_score

    @classmethod
    def from_target_size(
        cls,
        target_params: int,
        vocab_size: int = 65536,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 8192,
        depth_mult: float = 1.0,
    ) -> "ModelConfig":
        """
        Search for an architecture close to ``target_params`` non-embedding
        parameters. Supports the full 10M → trillions range.

        Scaling rules:
          - **hidden_size**: cube-root law — params ∝ H³ (for fixed depth/width
            ratio), so H = 4096 · (target / 5B)^(1/3).
          - **head_dim**: auto-tiered — 64 (< 200M), 128 (200M–10B),
            192 (10B–200B), 256 (200B+).
          - **layers**: L ≈ H / 146, calibrated so L=28 at H=4096 (≈5B model).
          - **KV-head options**: widen with model size so very large models
            can use more KV heads.
          - Search grid adapts: 6 hidden multipliers × 5 MLP ratios ×
            6–13 layer depths × 3 KV options ≈ 540–1170 candidates.
        """
        # ------------------------------------------------------------------
        # Auto-select head_dim by model tier
        # ------------------------------------------------------------------
        if head_dim is None:
            if target_params < 200_000_000:               # 10M  – 200M
                head_dim = 64
            elif target_params < 10_000_000_000:          # 200M – 10B
                head_dim = 128
            elif target_params < 200_000_000_000:         # 10B  – 200B
                head_dim = 192
            else:                                          # 200B – trillions
                head_dim = 256

        # ------------------------------------------------------------------
        # KV-head options scale with model tier
        # ------------------------------------------------------------------
        if target_params < 200_000_000:
            kv_options = [1, 2, 4]
        elif target_params < 10_000_000_000:
            kv_options = [2, 4, 8]
        elif target_params < 200_000_000_000:
            kv_options = [4, 8, 16]
        else:
            kv_options = [8, 16, 32]

        # ------------------------------------------------------------------
        # Base hidden-size estimate — cube-root law
        # Calibrated: H=4096, L=28 → ~5B params (this framework's defaults)
        # params ∝ L·H², and L ∝ H, so params ∝ H³
        # ------------------------------------------------------------------
        base = int(4096 * (target_params / 5_000_000_000) ** (1 / 3))
        hidden = ((base + 127) // 128) * 128  # round to multiple of 128

        best: Optional[ModelConfig] = None
        best_score = float("inf")

        # Search grid — constant across tiers for simplicity
        for h_mult in [0.75, 0.875, 1.0, 1.125, 1.25, 1.375]:
            h = max(128, int(hidden * h_mult))
            h = ((h + 63) // 64) * 64

            # Attention heads = hidden_size / head_dim
            H = max(2, h // head_dim)
            if target_params >= 1_000_000_000:
                H = ((H + 3) // 4) * 4  # round to multiple of 4 for GPU tensor cores
            if H < 2:
                continue

            for kv in kv_options:
                if H % kv != 0:
                    continue

                # Layers proportional to hidden_size
                # Calibrated: H=4096 → L≈28, so L = H / 146
                L_base = max(4, int(h / 146 * depth_mult))

                # Search window widens for larger models
                if h < 4096:
                    layer_window = 4
                elif h < 8192:
                    layer_window = 6
                else:
                    layer_window = 8
                L_start = max(1, L_base - layer_window)
                L_end = L_base + 2 * layer_window + 1  # inclusive

                for L in range(L_start, L_end, 2):
                    for mlp_ratio in [2.5, 2.75, 3.0, 3.25, 3.5]:
                        inter = int(h * mlp_ratio)
                        inter = ((inter + 63) // 64) * 64

                        score = cls._quality_score(
                            vocab_size, h, inter, L, H, kv,
                            head_dim, target_params,
                        )
                        if score < best_score:
                            best_score = score
                            best = cls(
                                vocab_size=vocab_size,
                                hidden_size=h,
                                intermediate_size=inter,
                                num_hidden_layers=L,
                                num_attention_heads=H,
                                num_key_value_heads=kv,
                                head_dim=head_dim,
                                max_position_embeddings=max_position_embeddings,
                            )

        if best is None:
            # Fallback: tiny model
            best = cls(
                vocab_size=vocab_size,
                hidden_size=256,
                intermediate_size=768,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=head_dim or 64,
                max_position_embeddings=max_position_embeddings,
            )

        return best

    @staticmethod
    def parse_param_count(value: str) -> int:
        """Parse '0.6B', '1.7B', '600M', '1T', '3T' → integer parameter count."""
        s = value.strip().upper().replace(" ", "")
        if s.endswith("T"):
            return int(float(s[:-1]) * 1_000_000_000_000)
        if s.endswith("B"):
            return int(float(s[:-1]) * 1_000_000_000)
        if s.endswith("M"):
            return int(float(s[:-1]) * 1_000_000)
        if s.endswith("K"):
            return int(float(s[:-1]) * 1_000)
        return int(s)


# ======================================================================
# Numerical primitives
# ======================================================================


class RMSNorm(nn.Module):
    """RMSNormalisation with learnable scale parameter ``weight``."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., hidden_size) → same shape."""
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            variance = x32.pow(2).mean(-1, keepdim=True)
            x32 = x32 * torch.rsqrt(variance + self.eps)
            out = self.weight.float() * x32
        return out.to(dtype)


class LayerNorm(nn.Module):
    """Standard LayerNorm with learnable scale and shift."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.weight.size(0),), self.weight, self.bias, self.eps)


def _build_norm(hidden_size: int, norm_type: str = "rmsnorm", eps: float = 1e-6) -> nn.Module:
    if norm_type == "layernorm":
        return LayerNorm(hidden_size, eps=eps)
    return RMSNorm(hidden_size, eps=eps)


# ======================================================================
# Rotary Position Embeddings
# ======================================================================


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    Applies rotation to query and key tensors based on position indices.

    Supports optional YaRN / NTK-aware frequency scaling via ``rope_scaling``:
        ``{"type": "yarn", "factor": 8.0}``
        ``{"type": "ntk", "factor": 8.0}``
    """

    def __init__(self, head_dim: int, theta: float = 1000000.0,
                 rope_scaling: Optional[Dict] = None):
        super().__init__()
        self.head_dim = head_dim
        self.rope_scaling = rope_scaling

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # Apply frequency scaling for YaRN / NTK
        if rope_scaling is not None:
            scaling_type = rope_scaling.get("type", "")
            factor = rope_scaling.get("factor", 1.0)
            if factor <= 0:
                factor = 1.0
            if scaling_type == "ntk":
                # NTK-aware: scale each frequency differently
                # Higher frequencies are scaled less → better high-frequency preservation
                dim = torch.arange(0, head_dim, 2, dtype=torch.float32)
                inv_freq = 1.0 / (
                    (theta * (factor ** (dim / (head_dim - 2)))) ** (dim / head_dim)
                )
            elif scaling_type == "yarn":
                # YaRN: simple frequency scaling + attention logit scaling
                inv_freq = inv_freq / factor
                self.yarn_scale = max(1.0, 0.1 * math.log(factor) + 1.0)
            else:
                # Linear / default scaling
                inv_freq = inv_freq / factor

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_cos_sin(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float()  # (1, D/2, 1)
        pos = position_ids[:, None, :].float()            # (B, 1, T)
        angles = pos * inv_freq                           # (B, D/2, T)
        angles = angles.transpose(1, 2)                   # (B, T, D/2)
        cos = angles.cos()
        sin = angles.sin()
        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to q and k.

        Handles both single-head (B, T, D) and multi-head (B, T, H*D)
        inputs by reshaping internally when the last dimension does not
        match ``head_dim``.
        """
        cos, sin = self._compute_cos_sin(position_ids)  # (B, T, D/2)
        # Interleave
        cos = torch.cat([cos, cos], dim=-1)  # (B, T, D_head)
        sin = torch.cat([sin, sin], dim=-1)

        d = cos.shape[-1]  # head_dim

        # --- Q: handle multi-head by reshaping to (B, T, n_heads, d) ---
        q_flat = q.shape[-1]
        if q_flat != d:
            q = q.view(*q.shape[:-1], q_flat // d, d)
            q_embed = (q * cos.unsqueeze(-2)) + (
                RotaryEmbedding._rotate_half(q) * sin.unsqueeze(-2)
            )
            q_embed = q_embed.view(*q.shape[:-2], q_flat)
        else:
            q_embed = (q * cos) + (RotaryEmbedding._rotate_half(q) * sin)

        # --- K: same logic ---
        k_flat = k.shape[-1]
        if k_flat != d:
            k = k.view(*k.shape[:-1], k_flat // d, d)
            k_embed = (k * cos.unsqueeze(-2)) + (
                RotaryEmbedding._rotate_half(k) * sin.unsqueeze(-2)
            )
            k_embed = k_embed.view(*k.shape[:-2], k_flat)
        else:
            k_embed = (k * cos) + (RotaryEmbedding._rotate_half(k) * sin)

        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)


# ======================================================================
# Sliding Window helper
# ======================================================================


_SW_MASK_CACHE: Dict[Tuple[int, int, str], torch.Tensor] = {}

def _build_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build a causal + sliding-window attention mask.

    Caches masks keyed by ``(seq_len, window_size, device)`` so repeated calls
    with the same shape avoid redundant ``torch.arange`` + ``torch.where``.

    Returns ``(1, 1, T, T)`` float additive mask:
        mask[i, j] = 0   if i >= j and i - j < window_size
        mask[i, j] = -inf  otherwise
    """
    key = (seq_len, window_size, str(device))
    if key in _SW_MASK_CACHE:
        cached = _SW_MASK_CACHE[key]
        if cached.dtype == dtype and cached.device == device:
            return cached
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = torch.where((i >= j) & (i - j < window_size), 0.0, float("-inf"))
    mask = mask.to(dtype=dtype).view(1, 1, seq_len, seq_len)
    _SW_MASK_CACHE[key] = mask
    return mask


# ======================================================================
# Attention (GQA + QK-Norm + SDPA)
# ======================================================================


class Attention(nn.Module):
    """
    Grouped-Query Attention with optional QK-Norm, routed through
    F.scaled_dot_product_attention (FlashAttention-2/3 backend when available).

    Supports both GQA (num_kv_heads < num_heads) and MHA (num_kv_heads == num_heads).
    """

    def __init__(self, config: ModelConfig, sliding_window_size: int = 0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_key_value_groups
        self.use_qk_norm = config.use_qk_norm
        self.attn_type = config.attn_type
        self.attention_dropout = config.attention_dropout
        self.sliding_window_size = sliding_window_size

        # Projections
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

        # QK-Norm (per-head RMSNorm on Q and K before RoPE — stabilises training)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _shape_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reshape q, k, v to (B, H, T, D_head) for SDPA."""
        B, T, _ = q.shape
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # QK-Norm: applied per-head before RoPE
        if self.use_qk_norm:
            q = q.view(B, T, self.num_heads, self.head_dim)
            k = k.view(B, T, self.num_kv_heads, self.head_dim)
            # RMSNorm over head_dim — each head normalised independently
            q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(B, T, self.num_heads, self.head_dim)
            k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(B, T, self.num_kv_heads, self.head_dim)
            q = q.reshape(B, T, self.num_heads * self.head_dim)
            k = k.reshape(B, T, self.num_kv_heads * self.head_dim)

        # RoPE
        if rotary_emb is not None and position_ids is not None:
            q, k = rotary_emb(q, k, position_ids)

        # Reshape for attention
        q, k, v = self._shape_qkv(q, k, v)

        # KV-cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        past_key_value_out = (k, v) if use_cache else None

        # Expand KV heads for GQA (grouped query attention)
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Sync q/k/v dtypes before SDPA (autocast can promote them differently)
        if q.dtype != v.dtype:
            q = q.to(v.dtype)
        if k.dtype != v.dtype:
            k = k.to(v.dtype)

        # Scale factor for SDPA (YaRN applies an additional temperature scaling)
        scale = 1.0 / math.sqrt(self.head_dim)
        if (self.config.rope_scaling is not None
                and self.config.rope_scaling.get("type") == "yarn"):
            yarn_factor = self.config.rope_scaling.get("factor", 1.0)
            if yarn_factor > 1.0:
                yarn_temp = 0.1 * math.log(yarn_factor) + 1.0
                scale /= yarn_temp

        # Determine if we can use the is_causal fast path
        # Sliding window forces an explicit mask (no is_causal shortcut)
        if self.sliding_window_size > 0:
            current_len = k.shape[2]  # total length after KV-cache
            sw_mask = _build_sliding_window_mask(
                current_len, self.sliding_window_size, q.device, q.dtype
            )
            # Select the last T query positions from the full mask
            attn_mask = sw_mask[:, :, -T:, :]
            if attention_mask is not None:
                # Combine padding mask (B, 1, 1, T) with sliding window mask (1, 1, T, T)
                attn_mask = attention_mask + attn_mask
            is_causal = False
        elif attention_mask is not None:
            attn_mask = attention_mask
            is_causal = False
        else:
            attn_mask = None
            is_causal = (past_key_value is None and T > 1)

        # Explicitly pin the SDPA backend to avoid a silent fallback to the
        # naive "math" kernel that materialises the full attention matrix.
        # Explicitly pin the SDPA backend to avoid a silent fallback to the
        # naive "math" kernel that materialises the full attention matrix.
        # On CPU we keep math as a last-resort fallback.
        backend_ctx = nullcontext()
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            if is_causal:
                backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            else:
                backends = [SDPBackend.EFFICIENT_ATTENTION]
            backends.append(SDPBackend.MATH)  # always available as fallback
            backend_ctx = sdpa_kernel(backends)
        except ImportError:
            # Older torch: fallback via the legacy context manager.
            # Keep math=True for CPU compatibility.
            backend_ctx = torch.backends.cuda.sdp_kernel(
                enable_flash=is_causal,
                enable_math=True,
                enable_mem_efficient=True,
            )

        with backend_ctx:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
                scale=scale,
            )

        # (B, H, T, D_head) → (B, T, H*D_head)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value_out


# ======================================================================
# MLP (SwiGLU or GELU)
# ======================================================================


class MLP(nn.Module):
    """
    MLP with configurable activation.

    SwiGLU (default):  SiLU(gate(x)) * up(x) → down(x)
        Uses gate_proj, up_proj, down_proj (3 weight matrices).

    GELU:  GELU(up(x)) → down(x)
        Uses up_proj, down_proj (2 weight matrices).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.mlp_type = config.mlp_type
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        if self.mlp_type == "swiglu":
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        else:
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mlp_type == "swiglu":
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        else:
            return self.down_proj(F.gelu(self.up_proj(x)))


def _build_attention(config: ModelConfig, sliding_window_size: int = 0) -> nn.Module:
    """Dispatch to MLA or standard Attention based on config."""
    if config.use_mla:
        return MLAAttention(config, sliding_window_size=sliding_window_size)
    return Attention(config, sliding_window_size=sliding_window_size)


# ======================================================================
# Multi-head Latent Attention (MLA) — DeepSeek-V2/V3 style
# ======================================================================


class MLAAttention(nn.Module):
    """
    Multi-head Latent Attention with low-rank KV joint compression.

    Instead of projecting K and V independently, MLA compresses them into a
    low-dimensional latent ``kv_c`` and expands on-the-fly. RoPE is applied
    only to a decoupled ``rope_dim`` subset of dimensions (the "rope" part);
    the remaining dimensions carry no position information ("nope").

    KV-cache stores ``(kv_c, k_rope)`` instead of ``(k, v)`` — roughly 4×
    smaller for typical latent_rank = hidden_size/4 and rope_dim = head_dim/2.
    """

    def __init__(self, config: ModelConfig, sliding_window_size: int = 0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_key_value_groups
        self.attention_dropout = config.attention_dropout
        self.sliding_window_size = sliding_window_size

        # Latent rank for KV compression
        self.latent_rank = config.kv_lora_rank or (config.hidden_size // 4)

        # RoPE is only applied to a subset of each head
        self.rope_dim = config.head_dim // 2
        self.nope_dim = config.head_dim - self.rope_dim

        # --- Query projections ---
        # Content (nope) part:  num_heads * nope_dim
        self.q_nope = nn.Linear(
            config.hidden_size, self.num_heads * self.nope_dim, bias=False
        )
        # Decoupled RoPE part:  num_heads * rope_dim
        self.q_rope = nn.Linear(
            config.hidden_size, self.num_heads * self.rope_dim, bias=False
        )

        # --- KV joint compression ---
        # Compress: hidden_size → latent_rank
        self.kv_c = nn.Linear(config.hidden_size, self.latent_rank, bias=False)

        # Expand latent → K content (nope), V content, K RoPE
        self.k_nope = nn.Linear(
            self.latent_rank, self.num_kv_heads * self.nope_dim, bias=False
        )
        self.v = nn.Linear(
            self.latent_rank, self.num_kv_heads * self.head_dim, bias=False
        )
        self.k_rope = nn.Linear(
            self.latent_rank, self.num_kv_heads * self.rope_dim, bias=False
        )

        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = hidden_states.shape

        # --- Q: split into nope + rope ---
        q_nope = self.q_nope(hidden_states)   # (B, T, H * nope_dim)
        q_rope = self.q_rope(hidden_states)    # (B, T, H * rope_dim)

        # --- KV joint compression (current tokens) ---
        kv_c = self.kv_c(hidden_states)        # (B, T, latent_rank)
        # Extract pre-RoPE k_rope from current kv_c
        k_rope_current = self.k_rope(kv_c)      # (B, T, KV * rope_dim)

        # --- Apply RoPE to rope parts (current tokens only) ---
        if rotary_emb is not None and position_ids is not None:
            q_rope_out, k_rope_embed_current = rotary_emb(q_rope, k_rope_current, position_ids)
        else:
            q_rope_out = q_rope
            k_rope_embed_current = k_rope_current

        # --- KV-cache (stores compressed kv_c + post-RoPE k_rope) ---
        # The compressed kv_c is re-expanded on every forward so we never
        # cache the expanded k_nope or v — only the compact latent.
        if past_key_value is not None:
            cached_kv_c, cached_k_rope = past_key_value
            kv_c = torch.cat([cached_kv_c, kv_c], dim=1)        # (B, P+T, latent_rank)
            k_rope_embed = torch.cat([cached_k_rope, k_rope_embed_current], dim=1)
        else:
            k_rope_embed = k_rope_embed_current

        past_key_value_out = (kv_c, k_rope_embed) if use_cache else None

        # --- Expand compressed kv_c to k_nope and v (full seq: past + present) ---
        k_nope_full = self.k_nope(kv_c)          # (B, P+T, KV * nope_dim)
        v_full = self.v(kv_c)                     # (B, P+T, KV * head_dim)

        # --- Assemble K = [k_nope || k_rope_embed] and V ---
        total_len = kv_c.shape[1]
        k = torch.cat([k_nope_full, k_rope_embed], dim=-1)  # (B, P+T, KV * head_dim)
        k = k.view(B, total_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v_full.view(B, total_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # --- Assemble Q = [q_nope || q_rope_out] (current tokens only) ---
        q = torch.cat([q_nope, q_rope_out], dim=-1)  # (B, T, H * head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # --- GQA: expand KV heads ---
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Sync dtypes
        if q.dtype != v.dtype:
            q = q.to(v.dtype)
        if k.dtype != v.dtype:
            k = k.to(v.dtype)

        # --- SDPA ---
        scale = 1.0 / math.sqrt(self.head_dim)

        # Sliding window mask
        if self.sliding_window_size > 0:
            current_len = k.shape[2]
            sw_mask = _build_sliding_window_mask(
                current_len, self.sliding_window_size, q.device, q.dtype
            )
            attn_mask = sw_mask[:, :, -T:, :]
            if attention_mask is not None:
                attn_mask = attention_mask + attn_mask
            is_causal = False
        elif attention_mask is not None:
            attn_mask = attention_mask
            is_causal = False
        else:
            attn_mask = None
            is_causal = (past_key_value is None and T > 1)

        backend_ctx = nullcontext()
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            if is_causal:
                backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            else:
                backends = [SDPBackend.EFFICIENT_ATTENTION]
            backends.append(SDPBackend.MATH)
            backend_ctx = sdpa_kernel(backends)
        except ImportError:
            backend_ctx = torch.backends.cuda.sdp_kernel(
                enable_flash=is_causal,
                enable_math=True,
                enable_mem_efficient=True,
            )

        with backend_ctx:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
                scale=scale,
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, past_key_value_out


# ======================================================================
# Mixture of Depth — router + MoD decoder layer
# ======================================================================


class MixtureOfDepthRouter(nn.Module):
    """
    Per-token router for Mixture-of-Depth (MoD).

    A lightweight linear layer with sigmoid activation decides whether
    each token passes through the FFN (``score > alpha``) or takes an
    identity shortcut (``score <= alpha``). A straight-through estimator
    keeps gradients flowing through the hard binary mask.

    An auxiliary load-balancing loss encourages roughly ``alpha`` fraction
    of tokens to be routed through the FFN.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.router = nn.Linear(config.hidden_size, 1, bias=False)
        self.alpha = config.mod_alpha
        self.last_aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, T, D)`` — post-normalised hidden states.

        Returns:
            ``(B, T, 1)`` binary mask (with straight-through gradients).
            ``self.last_aux_loss`` is set as a side-effect.
        """
        g = torch.sigmoid(self.router(x))  # (B, T, 1)
        # Straight-through estimator: hard mask in forward, soft gradient in backward
        mask = (g > self.alpha).float().detach() + g - g.detach()
        # Auxiliary loss: encourage fraction_selected ≈ alpha
        frac = g.mean()
        self.last_aux_loss = (frac - self.alpha) ** 2
        return mask


class MixtureOfDepthDecoderLayer(nn.Module):
    """
    Decoder layer with Mixture-of-Depth routing on the MLP sub-layer.

    Attention always runs on all tokens. The MLP only processes tokens
    whose router score exceeds ``config.mod_alpha``; skipped tokens
    take an identity shortcut (residual-only). This saves compute while
    keeping the attention field global.

    Composes with parallel mode: when ``config.layer_type == "parallel"``,
    attention and MLP are computed from the *same* normalised input.
    """

    def __init__(self, config: ModelConfig, sliding_window_size: int = 0):
        super().__init__()
        self.self_attn = _build_attention(config, sliding_window_size=sliding_window_size)
        self.mlp = MLP(config)
        self.input_layernorm = _build_norm(config.hidden_size, config.norm_type, config.rms_norm_eps)
        self.post_attention_layernorm = _build_norm(config.hidden_size, config.norm_type, config.rms_norm_eps)
        self.router = MixtureOfDepthRouter(config)
        self.parallel = config.layer_type == "parallel"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)

        if self.parallel:
            # Delegate to shared helper so the parallel forward logic is
            # identical across :class:`ParallelDecoderLayer` and this layer.
            return _parallel_forward(
                self, hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                position_ids=position_ids,
                rotary_emb=rotary_emb,
                router=self.router,
            )

        # Sequential mode: attn → residual → MLP (with MoD routing)
        attn_output, present_kv = self.self_attn(
            normed,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            position_ids=position_ids,
            rotary_emb=rotary_emb,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        mlp_input = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(mlp_input)
        mask = self.router(mlp_input)
        mlp_out = mlp_out * mask
        hidden_states = residual + mlp_out
        return hidden_states, present_kv


# ======================================================================
# Shared helper: parallel-layer forward
# ======================================================================


def _parallel_forward(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    position_ids: Optional[torch.Tensor] = None,
    rotary_emb: Optional[RotaryEmbedding] = None,
    router: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Parallel attention + MLP forward shared by :class:`ParallelDecoderLayer`
    and :class:`MixtureOfDepthDecoderLayer`.

    The ``layer`` must have ``self_attn``, ``mlp``, ``input_layernorm``, and
    ``post_attention_layernorm`` attributes.  When ``router`` is provided, the
    MLP output is gated by the router's binary mask (Mixture-of-Depth).
    """
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    attn_output, present_kv = layer.self_attn(
        hidden_states,
        attention_mask=attention_mask,
        past_key_value=past_key_value,
        use_cache=use_cache,
        position_ids=position_ids,
        rotary_emb=rotary_emb,
    )
    mlp_output = layer.mlp(layer.post_attention_layernorm(hidden_states))
    if router is not None:
        mlp_output = mlp_output * router(hidden_states)
    hidden_states = residual + attn_output + mlp_output
    return hidden_states, present_kv


# ======================================================================
# Decoder layers
# ======================================================================


class DecoderLayer(nn.Module):
    """
    Pre-norm decoder layer: attention → residual → MLP → residual.
    Uses the config's norm_type for both pre-attention and pre-MLP norms.
    """

    def __init__(self, config: ModelConfig, sliding_window_size: int = 0):
        super().__init__()
        self.self_attn = _build_attention(config, sliding_window_size=sliding_window_size)
        self.mlp = MLP(config)
        self.input_layernorm = _build_norm(config.hidden_size, config.norm_type, config.rms_norm_eps)
        self.post_attention_layernorm = _build_norm(config.hidden_size, config.norm_type, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_kv = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            position_ids=position_ids,
            rotary_emb=rotary_emb,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states, present_kv


class ParallelDecoderLayer(nn.Module):
    """
    Pre-norm decoder layer with **parallel** attention + MLP computation.

    Instead of sequential (attn → residual → mlp → residual), this computes:
        normed = input_layernorm(x)
        attn_out = attention(normed)
        mlp_out = mlp(post_attention_layernorm(normed))
        output = x + attn_out + mlp_out

    Used in PaLM. ~15% faster than sequential at the same quality.
    """

    def __init__(self, config: ModelConfig, sliding_window_size: int = 0):
        super().__init__()
        self.self_attn = _build_attention(config, sliding_window_size=sliding_window_size)
        self.mlp = MLP(config)
        self.input_layernorm = _build_norm(config.hidden_size, config.norm_type, config.rms_norm_eps)
        self.post_attention_layernorm = _build_norm(config.hidden_size, config.norm_type, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        return _parallel_forward(
            self, hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            position_ids=position_ids,
            rotary_emb=rotary_emb,
        )


# ======================================================================
# Mamba SSM block (pure PyTorch with optional mamba_ssm CUDA kernel)
# ======================================================================

# Optional fast kernel — fall back to pure PyTorch if unavailable
try:
    from mamba_ssm import Mamba as MambaSSMKernel
    _HAS_MAMBA_SSM = True
except ImportError:
    _HAS_MAMBA_SSM = False


def _selective_scan_py(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """
    Pure-PyTorch sequential selective scan (Mamba-1 style).

    Args:
        u: ``(B, L, D)`` — input
        delta: ``(B, L, D)`` — step sizes
        A: ``(D, N)`` — state-transition matrix (neglected exponentiated)
        B: ``(B, L, N)`` — input matrix
        C: ``(B, L, N)`` — output matrix
        D: ``(D,)`` — skip connection

    Returns:
        ``(B, L, D)`` — SSM output
    """
    batch, seq_len, dim = u.shape
    n = A.shape[1]
    dtype = u.dtype

    # Discretise: A_bar = exp(Δ·A), B_bar = Δ·B
    delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, N)
    delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)                          # (B, L, D, N)

    h = torch.zeros(batch, dim, n, device=u.device, dtype=dtype)
    outs: List[torch.Tensor] = []
    for t in range(seq_len):
        # h ← A_bar[:,t]·h + B_bar[:,t]·u[:,t]
        h = delta_A[:, t] * h + delta_B[:, t] * u[:, t].unsqueeze(-1)
        # y_t = C[:,t]·h + D·u[:,t]
        y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1) + D * u[:, t]  # (B, D)
        outs.append(y_t)

    return torch.stack(outs, dim=1)


class MambaBlock(nn.Module):
    """
    Mamba state-space model block (Mamba-1).

    Uses ``mamba_ssm`` CUDA kernel when available; falls back to a pure
    PyTorch sequential scan (correct but slower).

    Architecture:
        in_proj → conv1d → SiLU → selective_scan → (× SiLU(z)) → out_proj
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.hidden_size
        self.d_state: int = 16
        self.d_conv: int = 4
        self.expand: int = 2
        self.d_inner = self.expand * self.d_model

        # Input projection: x → (x_h, z)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # Depth-wise 1D convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=False,
        )

        # Selective SSM projections
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + self.d_inner, bias=False)
        # Δ projection with explicit bias and softplus
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        # A (log-space) and D (skip)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, L, D)`` — hidden states.

        Returns:
            ``(B, L, D)`` — Mamba-processed output.
        """
        batch, seq_len, _ = x.shape

        # Input projection → split into (x_h, z)
        xz = self.in_proj(x)
        x_h, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # 1D convolution (permute to channels-first)
        x_h = x_h.permute(0, 2, 1)  # (B, d_inner, L)
        conv = self.conv1d(x_h)[..., :seq_len]  # remove right-padding
        conv = F.silu(conv)                     # activation

        # Selective SSM
        u = conv.permute(0, 2, 1)  # (B, L, d_inner)

        # Project to Δ, B, C
        proj = self.x_proj(u)  # (B, L, d_state*2 + d_inner)
        B = proj[..., :self.d_state]                          # (B, L, N)
        C = proj[..., self.d_state:self.d_state * 2]          # (B, L, N)
        delta = F.softplus(self.dt_proj(proj[..., -self.d_inner:]))  # (B, L, D)

        # Use CUDA kernel when available, pure-PyTorch fallback otherwise
        if _HAS_MAMBA_SSM:
            # MambaSSMKernel expects different shapes; delegate to its forward
            y = self._mamba_ssm_fwd(u, delta, B, C)
        else:
            A = -torch.exp(self.A_log)  # ensure A is negative-definite
            y = _selective_scan_py(u, delta, A, B, C, self.D)

        # Gate by z and project out
        y = y * F.silu(z)
        return self.out_proj(y)

    def _mamba_ssm_fwd(
        self, u: torch.Tensor, delta: torch.Tensor,
        B: torch.Tensor, C: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate to ``mamba_ssm`` package (fast CUDA kernel)."""
        from mamba_ssm import selective_scan_fn
        # selective_scan_fn(u, delta, A, B, C, D, z=None, ...)
        A = -torch.exp(self.A_log)
        y = selective_scan_fn(
            u.permute(0, 2, 1).contiguous(),  # (B, D, L)
            delta.permute(0, 2, 1).contiguous(),
            A.contiguous(),
            B.permute(0, 2, 1).contiguous(),  # (B, N, L)
            C.permute(0, 2, 1).contiguous(),
            self.D.contiguous(),
            None,  # z
            None,  # delta_bias
            True,  # delta_softplus
        )
        return y.permute(0, 2, 1)  # (B, L, D)


# ======================================================================
# Jamba — hybrid SSM + Attention layers
# ======================================================================


class JambaLayer(nn.Module):
    """
    A single Jamba layer that can be either an Attention layer or a Mamba
    (SSM) layer, followed by an MLP.

    Attention layers appear every ``jamba_hybrid_layer_interval`` layers
    (e.g. every 4th). All other layers are Mamba blocks. This hybrid design
    gives the model global context through attention while keeping most
    layers as efficient SSMs.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.use_attention = (layer_idx % config.jamba_hybrid_layer_interval == 0)

        if self.use_attention:
            self.sublayer = _build_attention(config)
        else:
            self.sublayer = MambaBlock(config)

        self.mlp = MLP(config)
        self.input_norm = _build_norm(config.hidden_size, config.norm_type, config.rms_norm_eps)
        self.post_norm = _build_norm(config.hidden_size, config.norm_type, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)

        if self.use_attention:
            hidden_states, kv = self.sublayer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                position_ids=position_ids,
                rotary_emb=rotary_emb,
            )
        else:
            # Mamba — no positions, no KV-cache
            hidden_states = self.sublayer(hidden_states)
            kv = None

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states, kv


class JambaModel(nn.Module):
    """
    Jamba hybrid model: embeddings → JambaLayers → final norm.

    Attention layers receive RotaryEmbedding; pure-Mamba layers don't
    need positional information.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.scale_emb = config.scale_emb
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.layers = nn.ModuleList([
            JambaLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = _build_norm(config.hidden_size, config.norm_type, config.rms_norm_eps)
        self.embed_scale = 1.0 / math.sqrt(config.hidden_size) if config.scale_emb else 1.0
        self.gradient_checkpointing = False

        # Create RoPE if at least one layer uses attention
        if any(i % config.jamba_hybrid_layer_interval == 0 for i in range(config.num_hidden_layers)):
            self.rotary_emb = RotaryEmbedding(
                head_dim=config.head_dim,
                theta=config.rope_theta,
                rope_scaling=config.rope_scaling,
            )
        else:
            self.rotary_emb = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        hidden_states = self.embed_tokens(input_ids) * self.embed_scale

        if position_ids is None and self.rotary_emb is not None:
            if past_key_values is not None and past_key_values[0] is not None:
                past_len = past_key_values[0][0].shape[2]
            else:
                past_len = 0
            position_ids = torch.arange(
                past_len, past_len + seq_len,
                device=device, dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1)

        present_kv: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
                position_ids=position_ids,
                rotary_emb=self.rotary_emb,
            )
            present_kv.append(kv)

        hidden_states = self.norm(hidden_states)
        outputs: Dict[str, Any] = {"last_hidden_state": hidden_states}
        if use_cache:
            outputs["past_key_values"] = present_kv
        return outputs

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        print("[GradCkpt][Jamba] gradient checkpointing enabled")


# ======================================================================
# Transformer Model (base, no LM head)
# ======================================================================


class TransformerModel(nn.Module):
    """
    Base transformer: embeddings → N decoder layers → final norm.
    Optionally scales embeddings by 1/sqrt(hidden_size) as in Qwen3.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.scale_emb = config.scale_emb
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        # Dispatch layer type
        use_mod = config.mod_alpha > 0.0
        is_parallel = config.layer_type == "parallel"
        if use_mod:
            layer_cls = MixtureOfDepthDecoderLayer  # handles parallel internally
        elif is_parallel:
            layer_cls = ParallelDecoderLayer
        else:
            layer_cls = DecoderLayer
        # Build layers: when sliding_window_size > 0 alternate global/sw
        sw_size = config.sliding_window_size
        self.layers = nn.ModuleList([
            layer_cls(config, sliding_window_size=(sw_size if i % 2 == 1 else 0))
            for i in range(config.num_hidden_layers)
        ])
        self.norm = _build_norm(config.hidden_size, config.norm_type, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        # Embedding scale
        self.embed_scale = 1.0 / math.sqrt(config.hidden_size) if config.scale_emb else 1.0
        self.gradient_checkpointing = False

    def _ckpt_layer(
        self,
        layer: DecoderLayer,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        """
        Wrapper for torch.utils.checkpoint on a single decoder layer.
        KV-cache and attention mask are omitted during gradient checkpointing
        since it only runs in training (no cache) and causality is built into
        SDPA's is_causal flag (no explicit mask needed).
        """
        dummy_attention_mask: Optional[torch.Tensor] = None
        dummy_past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        return torch.utils.checkpoint.checkpoint(
            layer, hidden_states,
            dummy_attention_mask, dummy_past_kv, False, position_ids,
            self.rotary_emb,
            use_reentrant=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
            dict with keys "last_hidden_state" and optionally "past_key_values".
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        hidden_states = self.embed_tokens(input_ids) * self.embed_scale

        # Position IDs
        if position_ids is None:
            if past_key_values is not None and past_key_values[0] is not None:
                past_len = past_key_values[0][0].shape[2]
            else:
                past_len = 0
            position_ids = torch.arange(
                past_len, past_len + seq_len,
                device=device, dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1)

        # Decode through layers with optional gradient checkpointing
        present_kv: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        moe_aux_loss: torch.Tensor = torch.tensor(0.0, device=device)
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                hidden_states, kv = self._ckpt_layer(
                    layer, hidden_states, position_ids,
                )
            else:
                hidden_states, kv = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                    position_ids=position_ids,
                    rotary_emb=self.rotary_emb,
                )
            present_kv.append(kv)
            # Collect MoD auxiliary loss if layer has a router
            if hasattr(layer, 'router') and hasattr(layer.router, 'last_aux_loss'):
                moe_aux_loss = moe_aux_loss + layer.router.last_aux_loss

        hidden_states = self.norm(hidden_states)

        outputs = {"last_hidden_state": hidden_states}
        if use_cache:
            outputs["past_key_values"] = present_kv
        # MoD auxiliary loss (scaled by config weight)
        if self.config.mod_alpha > 0.0:
            outputs["mod_aux_loss"] = moe_aux_loss * self.config.mod_loss_weight
        return outputs

    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing — recompute decoder layer activations
        during backward instead of storing them. Saves ~30-35% VRAM at the
        cost of ~30% slower per step.
        """
        self.gradient_checkpointing = True
        print("[GradCkpt] gradient checkpointing enabled — activations will be "
              "recomputed on backward (saves VRAM, ~30% slower per step)")


# ======================================================================
# Multi-Token Prediction heads
# ======================================================================


class MTPHeads(nn.Module):
    """
    Multi-Token Prediction (MTP) heads — predict future tokens from
    the final hidden states using additional lightweight LM heads.

    Each head predicts ``k+1`` tokens ahead with a shared RMSNorm
    and an independent linear projection. The loss is discounted by
    ``gamma^k`` so the main next-token prediction dominates.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_mtp_heads
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        # Shared norm across all MTP heads
        self.norm = _build_norm(config.hidden_size, config.norm_type, config.rms_norm_eps)
        self.heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.num_mtp_heads)
        ])

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            hidden_states: ``(B, T, D)`` — final-layer hidden states.

        Returns:
            List of ``(B, T, V)`` logit tensors, one per MTP head.
        """
        h = self.norm(hidden_states)
        return [head(h) for head in self.heads]


# ======================================================================
# Shared MTP loss helper
# ======================================================================


def compute_mtp_loss(
    mtp_logits: List[torch.Tensor],
    labels: torch.Tensor,
    discount: float = 0.5,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute Multi-Token Prediction (MTP) auxiliary loss.

    Each MTP head :math:`k` predicts the token :math:`k+1` positions ahead.
    Loss is discounted by :math:`\\gamma^{k+1}`.

    Args:
        mtp_logits: List of ``(B, T, V)`` tensors, one per MTP head.
        labels: ``(B, T)`` target token ids.
        discount: Discount factor :math:`\\gamma` (default 0.5).
        ignore_index: Label index to ignore in cross-entropy (default -100).

    Returns:
        Scalar MTP loss (zero if no heads or sequence too short).
    """
    if not mtp_logits:
        return torch.tensor(0.0, device=labels.device)
    B, T = labels.shape
    V = mtp_logits[0].size(-1)
    total = torch.tensor(0.0, device=labels.device)
    for k, mtp_k in enumerate(mtp_logits):
        if T - 1 - k < 1:  # need at least one valid prediction position
            continue
        # mtp_k[:, t, :] predicts label at [t + 1 + k]
        mtp_k_shift = mtp_k[:, :T - 1 - k, :].contiguous().view(-1, V)
        mtp_labels_k = labels[:, 1 + k:].contiguous().view(-1)
        total += F.cross_entropy(mtp_k_shift, mtp_labels_k, ignore_index=ignore_index) * (discount ** (k + 1))
    return total


# ======================================================================
# Shared helpers: architecture variant CLI args
# ======================================================================


def add_architecture_args(parser: object) -> None:
    """
    Add architecture-variant arguments to an argparse.ArgumentParser.

    Call from every training script to keep the option set in one place.
    """
    p = parser  # type: ignore[attr-defined]
    p.add_argument("--arch", default="dense", choices=["dense", "jamba"],
                   help="Architecture type: dense (default) or jamba.")
    p.add_argument("--layer-type", default="sequential",
                   choices=["sequential", "parallel"],
                   help="Layer computation order: sequential (default) or parallel.")
    p.add_argument("--sliding-window-size", type=int, default=0,
                   help="Sliding window attention size (0 = disabled).")
    p.add_argument("--num-mtp-heads", type=int, default=0,
                   help="Multi-token prediction heads (0 = disabled).")
    p.add_argument("--mtp-discount", type=float, default=0.5,
                   help="Discount factor for MTP loss (default 0.5).")
    p.add_argument("--mod-alpha", type=float, default=0.0,
                   help="Mixture-of-Depth threshold (0 = disabled).")
    p.add_argument("--mod-loss-weight", type=float, default=0.01,
                   help="MoD auxiliary loss weight (default 0.01).")
    p.add_argument("--use-mla", action="store_true",
                   help="Enable Multi-head Latent Attention.")
    p.add_argument("--kv-lora-rank", type=int, default=None,
                   help="KV compression rank for MLA (default: hidden_size//4).")
    p.add_argument("--jamba-interval", type=int, default=4,
                   help="Place attention layer every N layers in Jamba (default 4).")


def apply_architecture_args(config: ModelConfig, args: object) -> ModelConfig:
    """
    Copy architecture-variant fields from parsed args onto a ModelConfig.

    Returns the config for chaining.
    """
    a = args  # type: ignore[attr-defined]
    config.arch_type = a.arch
    config.layer_type = a.layer_type
    config.sliding_window_size = a.sliding_window_size
    config.num_mtp_heads = a.num_mtp_heads
    config.mtp_discount = a.mtp_discount
    config.mod_alpha = a.mod_alpha
    config.mod_loss_weight = getattr(a, "mod_loss_weight", 0.01)
    config.use_mla = a.use_mla
    if getattr(a, "kv_lora_rank", None) is not None:
        config.kv_lora_rank = a.kv_lora_rank
    config.jamba_hybrid_layer_interval = a.jamba_interval
    return config


# ======================================================================
# Transformer with LM head (for causal LM training)
# ======================================================================


class TransformerForCausalLM(nn.Module):
    """
    Transformer with a tied/un-tied language modelling head.
    Supports KV-cache generation, weight tying, and gradient checkpointing.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # Dispatch backbone: JambaModel for hybrid SSM+Attention, TransformerModel otherwise
        if config.arch_type == "jamba":
            self.model = JambaModel(config)
        else:
            self.model = TransformerModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.tie_word_embeddings = config.tie_word_embeddings

        # Weight tying
        if self.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        # Multi-Token Prediction heads
        if config.num_mtp_heads > 0:
            self.mtp_heads = MTPHeads(config)

        # Initialise weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Weight initialisation with depth-appropriate scaling.

        Base std = ``config.init_std`` (default 0.02).
        Output projections (o_proj, down_proj) use a smaller std
        proportional to ``1 / sqrt(2 * num_layers)`` so that residual
        variance does not grow with depth (DeepNet / LLaMA recipe).
        """
        base_std = self.config.init_std
        if isinstance(module, nn.Linear):
            # Check if this is a residual output projection.
            # All such projections have shape (hidden_size, >=hidden_size).
            is_output_proj = (
                module.weight.shape[0] == self.config.hidden_size
                and module.weight.shape[1] >= self.config.hidden_size
            )
            std = base_std / math.sqrt(2 * self.config.num_hidden_layers) if is_output_proj else base_std
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=base_std)

    def tie_weights(self):
        """Re-tie lm_head to embed_tokens (needed after load_state_dict)."""
        if self.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Dict[str, Any]:
        """
        Returns dict with keys:
            "logits": (B, T, V) if labels is None, else (B, T-1, V) shifted.
            "loss": cross-entropy loss if labels provided.
            "past_key_values": optional KV-cache.

        ``num_logits_to_keep``: when > 0, only project the last N *target*
        positions through lm_head (the positions are auto-shifted left by one
        so ``logits[t]`` predicts ``input_ids[t+1]``). This avoids materialising
        the full (B, L, vocab_size) logit tensor — critical for GRPO where we
        only need log-probs for generated (non-prompt) tokens and vocab_size
        can be 100k+. Mutually exclusive with ``labels``.
        """
        if num_logits_to_keep > 0:
            assert labels is None, (
                "num_logits_to_keep and labels are mutually exclusive: "
                "labels needs the full-sequence shift, num_logits_to_keep "
                "is for the pre-sliced last-N-targets case."
            )
            # Only pass the *minimum* hidden states needed through lm_head:
            # one extra position (the last +1) because we want logits[t] for
            # predicting target t+1, and we restrict to the final N targets.
            hidden_subset = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
        )
        hidden_states = outputs["last_hidden_state"]

        # Slice hidden states BEFORE lm_head when num_logits_to_keep is set
        if num_logits_to_keep > 0:
            # +1 then drop the last row = standard "shift left by one"
            # restricted to the last N target positions.
            hidden_for_logits = hidden_states[:, -(num_logits_to_keep + 1):-1, :]
        else:
            hidden_for_logits = hidden_states

        logits = self.lm_head(hidden_for_logits).float()

        result = {"logits": logits}

        # MTP logits (multi-token prediction heads)
        if hasattr(self, "mtp_heads") and self.config.num_mtp_heads > 0:
            mtp_logits_list = self.mtp_heads(hidden_for_logits)
            # Cast to float to match lm_head precision
            result["mtp_logits"] = [mtp.float() for mtp in mtp_logits_list]

        if use_cache:
            result["past_key_values"] = outputs["past_key_values"]

        if labels is not None:
            # Shift such that token t predicts token t+1
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            # MTP loss — each head predicts (k+1)-ahead with discount gamma^(k+1)
            if hasattr(self, "mtp_heads") and self.config.num_mtp_heads > 0:
                mtp_loss = compute_mtp_loss(
                    result["mtp_logits"], labels,
                    discount=self.config.mtp_discount, ignore_index=-100,
                )
                loss += mtp_loss
            # MoD auxiliary loss
            loss += outputs.get("mod_aux_loss", 0.0)
            result["loss"] = loss

        return result

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Simple KV-cached generation. Returns (B, T+max_new_tokens) tokens.
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        past_key_values: Optional[List] = None
        generated = input_ids.clone()

        if pad_token_id is None:
            pad_token_id = 0

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            if finished.all():
                break

            if past_key_values is None:
                inp = generated
            else:
                inp = generated[:, -1:]

            out = self.forward(
                inp,
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = out["logits"][:, -1, :]
            past_key_values = out["past_key_values"]

            # Temperature
            if temperature > 0:
                logits = logits / temperature
            else:
                next_id = logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_id], dim=1)
                if eos_token_id is not None:
                    finished = finished | (next_id.squeeze(-1) == eos_token_id)
                continue

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    prev = generated[b].unique()
                    sub = logits[b, prev]
                    logits[b, prev] = torch.where(
                        sub > 0, sub / repetition_penalty, sub * repetition_penalty,
                    )

            # Top-k
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            # Top-p
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                mask = cum > top_p
                mask[:, 0] = False
                sorted_logits[mask] = float("-inf")
                out = torch.full_like(logits, float("-inf"))
                out.scatter_(-1, sorted_idx, sorted_logits)
                logits = out

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            # Force EOS for finished sequences
            if eos_token_id is not None:
                forced = torch.full_like(next_id, eos_token_id)
                next_id = torch.where(finished.unsqueeze(-1), forced, next_id)

            generated = torch.cat([generated, next_id], dim=1)

            if eos_token_id is not None:
                finished = finished | (next_id.squeeze(-1) == eos_token_id)

        return generated


# ======================================================================
# Utility
# ======================================================================


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_non_embedding_parameters(model: TransformerForCausalLM) -> int:
    """Count parameters excluding embed_tokens and lm_head (which are tied)."""
    total = 0
    for name, param in model.named_parameters():
        if "embed_tokens" not in name and "lm_head" not in name:
            total += param.numel()
    return total


# ======================================================================
# Smoke test
# ======================================================================


def smoke_test_variants():
    """Run forward + backward pass for every architecture variant.

    Returns True if all tests pass, False otherwise.
    """
    torch.manual_seed(42)
    BASE = dict(vocab_size=1024, hidden_size=128, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                max_position_embeddings=64)
    tests = [
        ("dense", {}),
        ("parallel", {"layer_type": "parallel"}),
        ("sliding_window", {"sliding_window_size": 16}),
        ("mod", {"mod_alpha": 0.125}),
        ("mtp", {"num_mtp_heads": 2}),
        ("mla", {"use_mla": True, "kv_lora_rank": 16}),
        ("jamba", {"arch_type": "jamba", "jamba_hybrid_layer_interval": 2}),
        ("jamba+mla", {"arch_type": "jamba", "jamba_hybrid_layer_interval": 2,
                       "use_mla": True, "kv_lora_rank": 16}),
        ("parallel+mod", {"layer_type": "parallel", "mod_alpha": 0.125}),
        ("sw+mla", {"sliding_window_size": 16, "use_mla": True, "kv_lora_rank": 16}),
    ]
    x = torch.randint(0, 1024, (2, 16))
    all_ok = True
    for name, kwargs in tests:
        try:
            cfg = ModelConfig(**BASE, **kwargs)
            m = TransformerForCausalLM(cfg)
            # Forward
            out = m(x)
            assert out["logits"].shape == (2, 16, 1024), f"logits shape mismatch"
            # Forward with loss
            out = m(x, labels=x)
            assert "loss" in out, f"no loss in output"
            loss = out["loss"]
            # Backward
            loss.backward()
            grad_norm = sum(p.grad.norm().item()
                           for p in m.parameters() if p.grad is not None)
            # Verify specific outputs
            if "mtp_logits" in out:
                assert len(out["mtp_logits"]) == kwargs.get("num_mtp_heads", 0)
            print(f"  [{name:>14s}] loss={loss.item():.4f}  "
                  f"grad_norm={grad_norm:.3f}  OK")
        except Exception as e:
            print(f"  [{name:>14s}] FAILED: {e}")
            all_ok = False
    # Generation test (dense + jamba)
    try:
        cfg = ModelConfig(**BASE)
        m = TransformerForCausalLM(cfg)
        gen = m.generate(torch.randint(0, 1024, (1, 8)), max_new_tokens=5)
        assert gen.shape == (1, 13), f"gen shape {gen.shape}"
        print(f"  [       generate] dense OK")
    except Exception as e:
        print(f"  [       generate] FAILED: {e}")
        all_ok = False
    try:
        cfg = ModelConfig(**BASE, arch_type="jamba", jamba_hybrid_layer_interval=2)
        m = TransformerForCausalLM(cfg)
        gen = m.generate(torch.randint(0, 1024, (1, 8)), max_new_tokens=5)
        assert gen.shape == (1, 13), f"jamba gen shape {gen.shape}"
        print(f"  [       generate] jamba OK")
    except Exception as e:
        print(f"  [       generate] jamba FAILED: {e}")
        all_ok = False
    return all_ok


def smoke_test():
    """Legacy single-config smoke test (calls smoke_test_variants)."""
    return smoke_test_variants()


if __name__ == "__main__":
    import sys

    # Run architecture search across 10M → 3T
    print("=== Auto-sizing search ===")
    test_cases = [
        ("10M", 10_000_000),
        ("100M", 100_000_000),
        ("300M", 300_000_000),
        ("1.7B", 1_700_000_000),
        ("8B", 8_000_000_000),
        ("70B", 70_000_000_000),
        ("300B", 300_000_000_000),
        ("1T", 1_000_000_000_000),
        ("3T", 3_000_000_000_000),
    ]
    for label, target in test_cases:
        try:
            cfg = ModelConfig.from_target_size(target)
            n = ModelConfig._param_count(
                cfg.vocab_size, cfg.hidden_size, cfg.intermediate_size,
                cfg.num_hidden_layers, cfg.num_attention_heads,
                cfg.num_key_value_heads, cfg.head_dim,
            )
            err = abs(n - target) / target * 100
            print(f"  {label:>5s} ({target:>13,}) → "
                  f"H={cfg.hidden_size:>6} L={cfg.num_hidden_layers:>3} "
                  f"heads={cfg.num_attention_heads:>3} kv={cfg.num_key_value_heads:>2} "
                  f"D={cfg.head_dim:>3} I={cfg.intermediate_size:>8} → "
                  f"N={n:>13,} (err={err:.1f}%)")
        except Exception as e:
            print(f"  {label:>5s} ({target:>13,}) → ERROR: {e}")

    # Run variant smoke tests
    print("\n=== Variant smoke tests ===")
    ok = smoke_test_variants()
    print(f"\n{'All tests passed!' if ok else 'SOME TESTS FAILED!'}")

