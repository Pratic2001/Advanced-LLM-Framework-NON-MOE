#!/usr/bin/env python3
"""
shutdown.py

Graceful SIGINT/SIGTERM handling for the training scripts.

Problem being solved: by default a SIGTERM (scheduler preemption, OOM-killer
reaction, ``kill``) or SIGINT (Ctrl+C) terminates the process immediately —
potentially mid-optimizer-step. The run dies without saving a final
checkpoint, and (worse) mid-collective kills can leave DDP/DeepSpeed peers
hanging forever waiting for a rank that will never arrive.

This module installs handlers that *record* the signal instead of dying.
The training loop calls :func:`should_stop` once per optimizer step (a
collective-safe point) and, when any rank reports a signal, every rank breaks
out of the loop, saves a final checkpoint, and tears the process group down
cleanly. A *second* signal forces an immediate ``os._exit`` so a hung
graceful shutdown can always be killed.

Why the all-reduce: in a distributed job only the ranks that actually receive
the signal would set their local flag. If rank 0 broke out of the step loop
while ranks 1..N-1 kept training, rank 0 would stop participating in the
per-step gradient all-reduce and the whole job would deadlock. Propagating the
flag through a tiny ``all_reduce(MAX)`` ensures every rank stops together, so
no rank is ever left waiting on a collective that will never come.

Usage:
    from shutdown import install_signal_handlers, should_stop

    install_signal_handlers()          # once, on every rank, near startup
    ...
    for step in range(start_step, args.num_steps):
        if should_stop(device, world_size):   # collective-safe, every rank
            interrupted = True
            break
        ...
"""

from __future__ import annotations

import logging
import os
import signal
from typing import List

import torch

log = logging.getLogger(__name__)

# Signals that have been requested so far. A set, not a count, so repeated
# SIGTERMs don't un-request an earlier SIGINT.
_requested: set = set()

# Whether handlers are installed — signal.signal() is main-thread-only, so
# repeated calls from subprocess workers must be safe no-ops.
_installed = False


def _handler(signum: int, _frame) -> None:
    """Record the signal; on the second signal, force-exit immediately."""
    global _requested
    _requested.add(signum)
    n = len(_requested)
    if n >= 2:
        # First signal started a graceful shutdown but it's not completing
        # (e.g. stuck in a collective). Don't allow the job to hang forever.
        code = 128 + signum
        log.critical("Second signal (%s) received — forcing exit %d",
                     signal.Signals(signum).name, code)
        os._exit(code)  # noqa: PLC0415 — skip further cleanup; a hung run is worse
    if n == 1:
        log.warning("Shutdown signal %s received — finishing current step, "
                    "then saving checkpoint and exiting. Send again to force exit.",
                    signal.Signals(signum).name)


def install_signal_handlers() -> None:
    """Install SIGINT/SIGTERM handlers that request a graceful shutdown.

    Idempotent and safe to call from every rank of a torchrun / DeepSpeed
    launch — only the first call in a process installs anything.
    """
    global _installed
    if _installed:
        return
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            # Not on the main thread, or the platform doesn't support this
            # signal (e.g. SIGTERM on Windows). Skip it.
            continue
    _installed = True
    log.info("Installed graceful shutdown handlers for SIGINT/SIGTERM")


def shutdown_requested() -> bool:
    """True if a SIGINT/SIGTERM has been received by *this* process."""
    return bool(_requested)


def shutdown_signals() -> List[int]:
    """The signals received so far (sorted, for logging)."""
    return sorted(_requested)


def should_stop(device: torch.device, world_size: int) -> bool:
    """Collective-safe per-step shutdown check.

    Returns True when *any* rank has received a shutdown signal. Must be
    called on every rank, once per optimizer step, and before any other
    collective in the step body (it is itself a collective).

    When ``world_size == 1`` (or no process group) this is a pure local
    check with no synchronisation.
    """
    local = 1 if _requested else 0
    if world_size > 1 and torch.distributed.is_initialized():
        flag = torch.tensor([local], dtype=torch.long, device=device)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        return bool(flag[0].item())
    return bool(local)
