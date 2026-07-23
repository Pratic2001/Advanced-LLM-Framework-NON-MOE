#!/usr/bin/env python3
"""
optim/lr_schedule.py

Learning rate schedulers for the dense LLM framework.

Supported schedules:
    cosine — warmup → cosine decay (reference baseline)
    wsd    — warmup → stable plateau → short decay (MiniCPM/Qwen2 recipe)

Usage:
    from optim.lr_schedule import build_scheduler
    scheduler = build_scheduler("cosine", warmup_steps=100, max_steps=10000,
                                peak_lr=5e-4, min_lr=5e-5)
    lr = scheduler.step(step)
"""

from __future__ import annotations

import math
from typing import Optional


# ---------------------------------------------------------------------------
# Cosine LR schedule (warmup → cosine decay)
# ---------------------------------------------------------------------------


def cosine_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """
    Cosine LR schedule: linear warmup then cosine decay.

    This is the standard schedule used in the reference Qwen3 repo
    (train.py, train_sft.py, train_grpo.py).
    """
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    t = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (peak_lr - min_lr)


# ---------------------------------------------------------------------------
# WSD (Warmup-Stable-Decay) LR schedule
# ---------------------------------------------------------------------------


def wsd_lr(
    step: int,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """
    WSD (Warmup-Stable-Decay) LR schedule.

    Phases:
        1. Warmup:    0 → warmup_steps  (linear ramp to peak_lr)
        2. Stable:    warmup_steps → warmup_steps + stable_steps (constant peak_lr)
        3. Decay:     stable_end → stable_end + decay_steps (cosine decay to min_lr)

    This is the MiniCPM / Qwen2 recipe that enables checkpoint flexibility —
    you can stop at any point during the stable phase and resume with a
    different decay length, or extend training without re-planning.
    """
    total_warmup = warmup_steps
    total_stable = warmup_steps + stable_steps
    total_decay = total_warmup + stable_steps + decay_steps

    if step < total_warmup:
        # Warmup phase
        return peak_lr * (step + 1) / total_warmup
    elif step < total_stable:
        # Stable phase — constant LR
        return peak_lr
    elif step < total_decay:
        # Decay phase — cosine decay
        t = (step - total_stable) / max(1, decay_steps)
        return min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (peak_lr - min_lr)
    else:
        return min_lr


# ---------------------------------------------------------------------------
# Scheduler wrapper class
# ---------------------------------------------------------------------------


class LRScheduler:
    """
    Callable LR scheduler that returns the LR for a given step.

    Usage:
        scheduler = LRScheduler("cosine", warmup_steps=100, max_steps=10000,
                                peak_lr=5e-4, min_lr=5e-5)
        for step in range(10000):
            lr = scheduler(step)
            # apply lr to optimizer
    """

    def __init__(
        self,
        schedule: str = "cosine",
        warmup_steps: int = 100,
        max_steps: int = 10000,
        peak_lr: float = 5e-4,
        min_lr: float = 5e-5,
        stable_ratio: float = 0.8,
    ):
        """
        Args:
            schedule: "cosine" or "wsd"
            warmup_steps: Linear warmup length
            max_steps: Total training steps (for cosine; for WSD, warmup+stable+decay)
            peak_lr: Peak learning rate after warmup
            min_lr: Floor learning rate
            stable_ratio: (WSD only) Fraction of steps in the stable phase
        """
        self.schedule = schedule
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr

        if schedule == "wsd":
            # Split max_steps into warmup / stable / decay
            remaining = max_steps - warmup_steps
            self.stable_steps = int(remaining * stable_ratio)
            self.decay_steps = remaining - self.stable_steps
        else:
            self.stable_steps = 0
            self.decay_steps = 0

    def __call__(self, step: int) -> float:
        if self.schedule == "cosine":
            return cosine_lr(step, self.warmup_steps, self.max_steps,
                             self.peak_lr, self.min_lr)
        elif self.schedule == "wsd":
            return wsd_lr(step, self.warmup_steps, self.stable_steps,
                          self.decay_steps, self.peak_lr, self.min_lr)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule!r}")

    def __repr__(self) -> str:
        return (f"LRScheduler(schedule={self.schedule!r}, "
                f"warmup={self.warmup_steps}, max_steps={self.max_steps}, "
                f"peak_lr={self.peak_lr:.2e}, min_lr={self.min_lr:.2e})")


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def build_scheduler(
    schedule: str = "cosine",
    warmup_steps: int = 100,
    max_steps: int = 10000,
    peak_lr: float = 5e-4,
    min_lr: float = 5e-5,
    stable_ratio: float = 0.8,
) -> LRScheduler:
    """
    Build an LR scheduler from configuration.

    Returns:
        A callable LRScheduler where scheduler(step) returns the LR at that step.
    """
    return LRScheduler(
        schedule=schedule,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        peak_lr=peak_lr,
        min_lr=min_lr,
        stable_ratio=stable_ratio,
    )
