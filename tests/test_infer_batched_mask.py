"""Regression tests for the infer.py `_build_combined_mask` cache-alignment fix.

The bug: `pad_mask_so_far` carried `past_len + prev_seq_len` columns into the
next decode step, but the causal mask width was only `past_len + seq_len`.
After the first decode step the two widths diverged, crashing batched
generation (`--batch-size > 1`) with a shape-mismatch RuntimeError.

These tests pin the post-fix behaviour: width must always be `past_len + seq_len`
across the first decode step, the equal-width edge case, and the defensive
left-pad edge case.
"""
import pytest
import torch

torch = pytest.importorskip("torch")

from infer import _build_combined_mask  # noqa: E402


def _shape_ok(mask, bsz, past_len, seq_len):
    return mask is not None and mask.shape == (bsz, 1, seq_len, past_len + seq_len)


def test_mask_width_matches_past_plus_seq():
    """Direct unit test: post-decode pad width must align with past_len."""
    bsz, past_len, seq_len = 2, 5, 3
    # pad_mask_so_far has the prefill width (past_len + prev_seq_len).
    prev_seq_len = 7
    pad_mask_so_far = torch.zeros(bsz, past_len + prev_seq_len)
    mask, updated = _build_combined_mask(
        pad_mask_so_far, seq_len, past_len, torch.device("cpu"), torch.float32,
    )
    assert _shape_ok(mask, bsz, past_len, seq_len)
    assert updated.shape == (bsz, past_len + seq_len)


def test_mask_handles_past_len_equal_to_pad_width():
    """Edge case: pad_mask_so_far width already == past_len (no-op slice)."""
    bsz, past_len, seq_len = 3, 4, 2
    pad_mask_so_far = torch.zeros(bsz, past_len)
    mask, updated = _build_combined_mask(
        pad_mask_so_far, seq_len, past_len, torch.device("cpu"), torch.float32,
    )
    assert _shape_ok(mask, bsz, past_len, seq_len)
    assert updated.shape == (bsz, past_len + seq_len)


def test_mask_handles_pad_shorter_than_past_len():
    """Edge case: pad_mask_so_far is narrower than past_len (left-pad)."""
    bsz, past_len, seq_len = 2, 6, 2
    pad_mask_so_far = torch.zeros(bsz, 3)  # shorter than past_len
    mask, updated = _build_combined_mask(
        pad_mask_so_far, seq_len, past_len, torch.device("cpu"), torch.float32,
    )
    assert _shape_ok(mask, bsz, past_len, seq_len)
    assert updated.shape == (bsz, past_len + seq_len)


def test_mask_handles_no_pad():
    """If pad_mask_so_far is None, the function returns (None, None)."""
    out = _build_combined_mask(
        None, seq_len=2, past_len=3, device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert out == (None, None)


def test_mask_includes_causal_upper_right():
    """Causal restriction must mark the upper-right triangle as -inf."""
    bsz, past_len, seq_len = 1, 3, 2
    pad_mask_so_far = torch.zeros(bsz, past_len + 1)
    mask, _ = _build_combined_mask(
        pad_mask_so_far, seq_len, past_len, torch.device("cpu"), torch.float32,
    )
    # mask shape: (B, 1, seq_len, past_len + seq_len)
    # For seq_len=2, past_len=3: total=5. Causal: row 0 can see cols 0..3,
    # row 1 can see cols 0..4. So mask[0,0,0,4] must be -inf.
    upper_right = mask[0, 0, 0, -1].item()
    assert upper_right == float("-inf") or upper_right < -1e9


def test_mask_does_not_crash_on_batch():
    """A common failure mode was RuntimeError on bsz>1 — must not regress."""
    bsz, past_len, seq_len = 4, 8, 4
    pad_mask_so_far = torch.zeros(bsz, past_len + 5)
    mask, updated = _build_combined_mask(
        pad_mask_so_far, seq_len, past_len, torch.device("cpu"), torch.float32,
    )
    assert mask.shape == (bsz, 1, seq_len, past_len + seq_len)
    assert updated.shape == (bsz, past_len + seq_len)