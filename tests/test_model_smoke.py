#!/usr/bin/env python3
"""
CPU-only smoke tests for model.py — build a tiny transformer, run one forward
pass, and verify the loss is finite and trainable.

Run:  pytest tests/test_model_smoke.py
"""

import pytest
import torch

from model import ModelConfig, TransformerForCausalLM, count_parameters


def _tiny_config(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
    )
    base.update(overrides)
    return ModelConfig(**base)


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = TransformerForCausalLM(cfg)
    model.eval()
    return model


def test_forward_logits_shape(tiny):
    x = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        out = tiny(x)
    assert "logits" in out
    assert out["logits"].shape == (2, 16, 128)


def test_forward_with_labels_loss_finite(tiny):
    x = torch.randint(0, 128, (2, 16))
    y = torch.randint(0, 128, (2, 16))
    tiny.train()
    out = tiny(x, labels=y)
    assert "loss" in out
    assert torch.isfinite(out["loss"])


def test_loss_decreases_one_step(tiny):
    cfg = tiny.config
    x = torch.randint(0, 128, (2, 16))
    y = torch.randint(0, 128, (2, 16))
    opt = torch.optim.AdamW(tiny.parameters(), lr=1e-3)
    tiny.train()
    opt.zero_grad()
    loss1 = tiny(x, labels=y)["loss"]
    loss1.backward()
    opt.step()
    loss2 = tiny(x, labels=y)["loss"]
    assert loss2.item() < loss1.item()


def test_count_parameters_positive(tiny):
    assert count_parameters(tiny) > 0


def test_tied_embeddings(tiny):
    # With tie_word_embeddings=True the lm_head weight is shared with embed
    assert tiny.tie_word_embeddings
    if tiny.tie_word_embeddings:
        embed = tiny.model.embed_tokens.weight
        head = tiny.lm_head.weight
        assert embed is head or torch.equal(embed, head)


def test_gqa_attention_forward(tiny):
    # GQA should run with num_kv_heads < num_q_heads
    assert tiny.config.attn_type == "gqa"
    x = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        out = tiny(x)
    assert torch.isfinite(out["logits"]).all()
