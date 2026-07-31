#!/usr/bin/env python3
"""
Unit tests for recipe.py — the single source of truth for ChatML templates,
special tokens and training mode.

Run:  pytest tests/test_recipe.py
"""

import pytest

from recipe import TrainingRecipe


@pytest.fixture
def recipe() -> TrainingRecipe:
    return TrainingRecipe(
        mode="reasoning",
        model_name="CustomModel",
        chat_template="chatml",
    )


def test_special_tokens_include_base_and_thinking(recipe):
    toks = recipe.special_tokens
    assert "<|pad|>" in toks
    assert "<|endoftext|>" in toks
    assert "<think>" in toks  # reasoning mode adds think tags


def test_non_reasoning_excludes_think_tags():
    r = TrainingRecipe(mode="non_reasoning")
    assert "<think>" not in r.special_tokens


def test_hybrid_includes_hybrid_tokens():
    r = TrainingRecipe(mode="hybrid")
    toks = r.special_tokens
    assert "<|think_on|>" in toks
    assert "<|think_off|>" in toks


def test_pad_and_eos_properties(recipe):
    assert recipe.pad_token == "<|pad|>"
    assert recipe.eos_token == "<|endoftext|>"


def test_to_dict_roundtrip(recipe):
    r2 = TrainingRecipe.from_dict(recipe.to_dict())
    assert r2.mode == recipe.mode
    assert r2.model_name == recipe.model_name
    assert r2.chat_template == recipe.chat_template


def test_json_roundtrip(recipe, tmp_path):
    path = str(tmp_path / "recipe.json")
    recipe.to_json(path)
    r2 = TrainingRecipe.from_json(path)
    assert r2.mode == recipe.mode
    assert r2.model_name == recipe.model_name


def test_format_assistant_turn_reasoning():
    r = TrainingRecipe(mode="reasoning")
    out = r.format_assistant_turn(thinking="let's think", answer="42")
    assert "<think>" in out and "</think>" in out
    assert "42" in out


def test_format_assistant_turn_non_reasoning_forces_no_think():
    r = TrainingRecipe(mode="non_reasoning")
    out = r.format_assistant_turn(thinking="let's think", answer="42")
    assert "<think>" not in out
    assert "42" in out
