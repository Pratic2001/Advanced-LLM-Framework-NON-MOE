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
    cfg = ModelConfig.from_target_size(target_params=1_700_000_000)
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
        ref_layers = 28
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
        parameters.

        Improvements over the baseline version:
          - ``head_dim`` auto-selected per model tier (64 for < 1 B, 128 otherwise)
            so small models don't waste parameters on oversized heads.
          - Wider search grid (6 hidden multipliers, 5 MLP ratios,
            +/- 4 layers in steps of 2) for better accuracy.
          - KV-head options include 1 for sub-1 B models.
          - The quality score uses asymmetric overshoot penalty and
            a mild depth preference (deeper = better).
        """
        # Auto-select head_dim based on target size
        if head_dim is None:
            head_dim = 64 if target_params < 1_000_000_000 else 128

        # KV-head options — small models can use 1 KV head (MQA)
        kv_options = [1, 2, 4] if target_params < 1_000_000_000 else [2, 4, 8]

        # Base hidden-size estimate  (Chinchilla scaling)
        base = int(4096 * math.sqrt(target_params / 8_000_000_000))
        hidden = ((base + 127) // 128) * 128

        best: Optional[ModelConfig] = None
        best_score = float("inf")

        # More granular search than the baseline
        for h_mult in [0.75, 0.875, 1.0, 1.125, 1.25, 1.375]:
            h = max(128, int(hidden * h_mult))
            h = ((h + 63) // 64) * 64

            # Attention heads
            H = max(2, h // head_dim)
            if target_params >= 1_000_000_000:
                H = ((H + 3) // 4) * 4  # round to multiple of 4 for GPU tensor cores
            if H < 2:
                continue

            for kv in kv_options:
                if H % kv != 0:
                    continue

                # Layers proportional to hidden_size  (more granular: +/-4 in steps of 2)
                L_base = max(6, int(28 * (h / 2048) * depth_mult))

                for L in range(max(4, L_base - 4), L_base + 8, 2):
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
        """Parse '0.6B', '1.7B', '600M' → integer parameter count."""
        s = value.strip().upper().replace(" ", "")
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
    """

    def __init__(self, head_dim: int, theta: float = 1000000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
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
# Attention (GQA + QK-Norm + SDPA)
# ======================================================================


class Attention(nn.Module):
    """
    Grouped-Query Attention with optional QK-Norm, routed through
    F.scaled_dot_product_attention (FlashAttention-2/3 backend when available).

    Supports both GQA (num_kv_heads < num_heads) and MHA (num_kv_heads == num_heads).
    """

    def __init__(self, config: ModelConfig):
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

        # Scale factor for SDPA
        scale = 1.0 / math.sqrt(self.head_dim)

        # Determine if we can use is_causal fast path
        is_causal = (
            attention_mask is None
            and past_key_value is None
            and T > 1
        )

        # Explicitly pin the SDPA backend instead of letting PyTorch choose.
        # Once we pass a custom float attn_mask (needed for left-padding),
        # the flash-attention kernel is unavailable — fine, we want the
        # memory-efficient (xFormers-style) kernel in that case, which is
        # still ~linear in seq_len. What we must NOT allow is a silent
        # fallback to the naive "math" kernel: that materializes the full
        # (batch, heads, seq_len, seq_len) attention-probability matrix and
        # keeps it around for backward on every layer, turning what should
        # be linear memory into quadratic.
        backend_ctx = nullcontext()
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            backends = ([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
                        if is_causal else [SDPBackend.EFFICIENT_ATTENTION])
            backend_ctx = sdpa_kernel(backends)
        except ImportError:
            # Older torch: same intent via the legacy context manager.
            backend_ctx = torch.backends.cuda.sdp_kernel(
                enable_flash=is_causal,
                enable_math=False,
                enable_mem_efficient=True,
            )

        with backend_ctx:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
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


# ======================================================================
# Decoder Layer
# ======================================================================


class DecoderLayer(nn.Module):
    """
    Pre-norm decoder layer: attention → residual → MLP → residual.
    Uses the config's norm_type for both pre-attention and pre-MLP norms.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = Attention(config)
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
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = _build_norm(config.hidden_size, config.norm_type, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            theta=config.rope_theta,
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

        hidden_states = self.norm(hidden_states)

        outputs = {"last_hidden_state": hidden_states}
        if use_cache:
            outputs["past_key_values"] = present_kv
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
        self.model = TransformerModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.tie_word_embeddings = config.tie_word_embeddings

        # Weight tying
        if self.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

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


def smoke_test():
    """Verify forward/backward pass works on random data."""
    config = ModelConfig(
        vocab_size=4096,
        hidden_size=128,
        intermediate_size=384,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=256,
    )
    model = TransformerForCausalLM(config)
    x = torch.randint(0, 4096, (2, 64))
    out = model(x, labels=x)
    loss = out["loss"]
    loss.backward()
    n = count_parameters(model)
    print(f"[smoke] Model params: {n:,} ({n/1e6:.2f}M)")
    print(f"[smoke] Loss: {loss.item():.4f}")
    print(f"[smoke] Forward + backward: OK")
    return True


if __name__ == "__main__":
    # Run architecture search
    for target in [300_000_000, 600_000_000, 1_700_000_000, 4_000_000_000, 8_000_000_000]:
        cfg = ModelConfig.from_target_size(target)
        n = ModelConfig._param_count(
            cfg.vocab_size, cfg.hidden_size, cfg.intermediate_size,
            cfg.num_hidden_layers, cfg.num_attention_heads,
            cfg.num_key_value_heads, cfg.head_dim,
        )
        print(f"  target={cfg.parse_param_count(str(target)):,} → "
              f"H={cfg.hidden_size} L={cfg.num_hidden_layers} "
              f"heads={cfg.num_attention_heads} kv={cfg.num_key_value_heads} "
              f"I={cfg.intermediate_size} → N={n:,} (err={abs(n-target)/target*100:.1f}%)")

    # Smoke test
    smoke_test()
