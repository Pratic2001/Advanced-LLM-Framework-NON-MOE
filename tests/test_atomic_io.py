#!/usr/bin/env python3
"""
Unit tests for atomic_io.py — crash-safe checkpoint I/O.

Run:  pytest tests/test_atomic_io.py
"""

import io
import json
import os

import pytest
import torch
import torch.nn as nn

from atomic_io import (
    atomic_symlink,
    atomic_torch_save,
    atomic_write_bytes,
    atomic_write_json,
    load_torch_checkpoint,
)


def _lin():
    m = nn.Linear(4, 4)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, fused=False)
    for _ in range(3):
        opt.zero_grad()
        m.weight.grad = torch.randn_like(m.weight)
        opt.step()
    return m, opt


def test_torch_save_roundtrip(tmp_path):
    m, opt = _lin()
    ckpt = {
        "step": 5,
        "model_state": m.state_dict(),
        "optimizer_state": opt.state_dict(),
        "config": {"hidden": 4},
    }
    path = str(tmp_path / "step_00005.pt")
    atomic_torch_save(ckpt, path)
    loaded = torch.load(path, weights_only=True)
    assert loaded["step"] == 5
    assert "optimizer_state" in loaded


def test_torch_save_leaves_no_tmp(tmp_path):
    path = str(tmp_path / "ckpt.pt")
    atomic_torch_save({"step": 1}, path)
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == []


def test_json_roundtrip(tmp_path):
    meta = {"step": 7, "total_tokens": 1234, "best_val_loss": 2.5}
    path = str(tmp_path / "meta.json")
    atomic_write_json(meta, path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded == meta


def test_json_handles_non_serializable(tmp_path):
    # default=str should tolerate objects json can't natively hold
    meta = {"dtype": torch.float32}
    path = str(tmp_path / "meta.json")
    atomic_write_json(meta, path)  # must not raise
    with open(path) as f:
        assert "dtype" in json.load(f)


def test_write_bytes(tmp_path):
    path = str(tmp_path / "blob.bin")
    atomic_write_bytes(b"\x00\x01\x02", path)
    with open(path, "rb") as f:
        assert f.read() == b"\x00\x01\x02"


def test_symlink_replace(tmp_path):
    link = str(tmp_path / "latest.pt")
    atomic_symlink("a.pt", link)
    assert os.readlink(link) == os.path.abspath("a.pt")
    # replace with a regular file in place
    target = str(tmp_path / "b.pt")
    with open(target, "w") as f:
        f.write("x")
    atomic_symlink(target, link)
    assert os.readlink(link) == target


def test_symlink_replaces_regular_file(tmp_path):
    link = str(tmp_path / "latest")
    with open(link, "w") as f:
        f.write("old")
    atomic_symlink("new.pt", link)
    assert os.path.islink(link)
    assert os.readlink(link) == os.path.abspath("new.pt")


def test_load_checkpoint_safe_rejects_custom_class(tmp_path):
    m, _ = _lin()

    class _Evil:
        def __reduce__(self):
            return (os.system, ("true",))

    path = str(tmp_path / "bad.pt")
    torch.save({"model_state": m.state_dict(), "payload": _Evil()}, path)
    with pytest.raises(RuntimeError, match="weights_only=True"):
        load_torch_checkpoint(path, map_location="cpu")


def test_load_checkpoint_safe_allow_unsafe(tmp_path):
    m, _ = _lin()
    path = str(tmp_path / "good.pt")
    torch.save({"step": 3, "model_state": m.state_dict()}, path)
    loaded = load_torch_checkpoint(path, map_location="cpu")
    assert loaded["step"] == 3


def test_load_checkpoint_roundtrip_optimizer(tmp_path):
    m, opt = _lin()
    ckpt = {"step": 9, "model_state": m.state_dict(),
            "optimizer_state": [opt.state_dict(), opt.state_dict()]}
    path = str(tmp_path / "ckpt.pt")
    atomic_torch_save(ckpt, path)
    loaded = load_torch_checkpoint(path, map_location="cpu")
    assert isinstance(loaded["optimizer_state"], list)
    assert len(loaded["optimizer_state"]) == 2
