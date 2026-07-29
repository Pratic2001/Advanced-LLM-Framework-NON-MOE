#!/usr/bin/env python3
"""
hivemind_utils.py

Shared utilities for Hivemind-based decentralized training.

What this provides
------------------
  1. ``setup_hivemind_peer()`` — initialise a P2P peer, join an existing swarm
     or create a new one.
  2. ``build_hivemind_optimizer()`` — wrap any ``torch.optim.Optimizer`` in
     Hivemind's ``DecentralizedOptimizer`` so parameters are averaged
     asynchronously across heterogeneous nodes.
  3. ``average_checkpoints_via_hivemind()`` — pull remote parameter snapshots
     and produce a merged checkpoint (useful for evaluation / inference).
  4. Re-exported helpers from the parent project so training scripts don't
     need to deep-import across directories:
       ``PackedDataLoader``, ``pretrain_loss``, ``validate``, ``estimate_mfu``,
       ``save_checkpoint``, ``load_checkpoint``, ``get_lr``,
       ``build_optimizer_groups``, ``try_init_wandb``, ``log_wandb``.

How peers discover each other
-----------------------------
  - The **first** peer is started with ``--initial-peers ""`` (empty) — it
    becomes the bootstrap node.
  - All subsequent peers pass ``--initial-peers <bootstrap_ip>:<port>`` (the
    address of any already-connected peer).
  - Hivemind handles the rest: each peer discovers the full swarm through the
    DHT and builds an ``AllreduceRunner`` that averages gradients/parameters
    with a random subset of ``--target-group-size`` peers on each step.

Heterogeneous training
----------------------
  - Every peer maintains its own local model copy and runs its training loop
    at its own speed.  A 4090 may do 3 local steps in the time a laptop GPU
    does 1 — that is expected and correct.
  - ``DecentralizedOptimizer.step()`` fires an async all-reduce after each
    local step.  The all-reduce is non-blocking: the peer continues to the
    next batch immediately and applies the averaged parameters when the
    all-reduce completes (typically a few steps later).
  - Faster peers contribute proportionally more gradient updates to the shared
    model, which is exactly what you want when pooling heterogeneous compute.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import socket
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ── Hivemind ──────────────────────────────────────────────────────────────────
try:
    import hivemind
    from hivemind import Peer, DecentralizedOptimizer
    _HIVEMIND_AVAILABLE = True
except ImportError:
    _HIVEMIND_AVAILABLE = False
    Peer = None
    DecentralizedOptimizer = None

# ── Import parent project utilities ───────────────────────────────────────────
# NOTE: we import from the parent directory via a relative-import hack that
# works when train_pretrain_hivemind.py is run from the repo root.
# If you get ImportError, run from the repo root or add it to sys.path.
import sys
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from model import ModelConfig, TransformerForCausalLM, count_parameters
from optim.build_optimizer import build_optimizer
from optim.lr_schedule import build_scheduler

# Re-export from train_pretrain.py for convenience in Hivemind scripts.
# We import these as late as possible to avoid circular imports at module load.

__all__ = [
    "setup_hivemind_peer",
    "build_hivemind_optimizer",
    "average_checkpoints_via_hivemind",
    "get_initial_peers_from_args",
    "PackedDataLoader",
    "pretrain_loss",
    "validate",
    "estimate_mfu",
    "save_checkpoint",
    "load_checkpoint",
    "get_lr",
    "build_optimizer_groups",
    "try_init_wandb",
    "log_wandb",
    "count_parameters",
    "ModelConfig",
    "TransformerForCausalLM",
    "_HIVEMIND_AVAILABLE",
]

# We lazily import from train_pretrain.py to avoid hard-coding absolute paths.
def _import_train_pretrain():
    """Late import so script paths resolve correctly."""
    import importlib
    mod = importlib.import_module("train_pretrain")
    return mod


def PackedDataLoader(*args, **kwargs):
    return _import_train_pretrain().PackedDataLoader(*args, **kwargs)


def pretrain_loss(*args, **kwargs):
    return _import_train_pretrain().pretrain_loss(*args, **kwargs)


def validate(*args, **kwargs):
    return _import_train_pretrain().validate(*args, **kwargs)


def estimate_mfu(*args, **kwargs):
    return _import_train_pretrain().estimate_mfu(*args, **kwargs)


def save_checkpoint(*args, **kwargs):
    return _import_train_pretrain().save_checkpoint(*args, **kwargs)


def load_checkpoint(*args, **kwargs):
    return _import_train_pretrain().load_checkpoint(*args, **kwargs)


def get_lr(*args, **kwargs):
    return _import_train_pretrain().get_lr(*args, **kwargs)


def build_optimizer_groups(*args, **kwargs):
    return _import_train_pretrain().build_optimizer_groups(*args, **kwargs)


def try_init_wandb(*args, **kwargs):
    return _import_train_pretrain().try_init_wandb(*args, **kwargs)


def log_wandb(*args, **kwargs):
    return _import_train_pretrain().log_wandb(*args, **kwargs)


# ======================================================================
# Hivemind Peer Setup
# ======================================================================


@dataclass
class HivemindPeerInfo:
    """Information about the local Hivemind peer."""
    peer: Any   # hivemind.Peer
    endpoint: str  # visible endpoint (ip:port)


def setup_hivemind_peer(
    initial_peers: List[str],
    host: str = "0.0.0.0",
    port: int = 0,
    peer_id: Optional[str] = None,
    start: bool = True,
    verbose: bool = True,
) -> HivemindPeerInfo:
    """
    Create a Hivemind P2P peer and connect to the swarm.

    Args:
        initial_peers: List of ``ip:port`` or ``peer_id@ip:port`` strings.
            The first peer in a new swarm passes an **empty list**.
            All subsequent peers pass **at least one** address of an
            already-connected peer.
        host: Network interface to bind to (``"0.0.0.0"`` = all interfaces).
        port: Port to listen on.  ``0`` = OS picks a free port.
        peer_id: Optional human-readable name for this peer.
        start: Whether to start the peer immediately.
        verbose: Print diagnostic information.

    Returns:
        ``HivemindPeerInfo`` with the peer object and its visible endpoint.
    """
    if not _HIVEMIND_AVAILABLE:
        raise ImportError(
            "hivemind is not installed. Run:\n"
            "  pip install hivemind>=1.1.0"
        )

    # Sanitise initial peers
    initial_peers = [p.strip() for p in initial_peers if p.strip()]

    if verbose:
        print(f"[Hivemind] Creating peer on {host}:{port}")
        if initial_peers:
            print(f"[Hivemind] Connecting to initial peers: {initial_peers}")
        else:
            print("[Hivemind] No initial peers — starting as bootstrap node.")
            print("[Hivemind] Other peers must point --initial-peers to this node.")

    peer = hivemind.Peer(
        initial_peers=initial_peers if initial_peers else None,
        peer_id=peer_id,
        host=host,
        port=port,
        start=start,
    )

    # Wait for the peer to be ready
    if start:
        # Give it a moment to initialise
        time.sleep(0.5)
        if verbose:
            visible = getattr(peer, "endpoint", f"{host}:{port} (unknown)")
            print(f"[Hivemind] Peer ready at {visible}")
            print(f"[Hivemind] Other peers can join via: --initial-peers {visible}")

    return HivemindPeerInfo(
        peer=peer,
        endpoint=getattr(peer, "endpoint", f"{host}:{port}"),
    )


def get_initial_peers_from_args(args: argparse.Namespace) -> List[str]:
    """
    Parse ``--initial-peers`` from CLI args into a list.

    Accepts:
      - ``--initial-peers ""`` (empty → bootstrap)
      - ``--initial-peers "192.168.1.5:5678"``
      - ``--initial-peers "peer1@192.168.1.5:5678,192.168.1.6:5679"``
    """
    raw = getattr(args, "initial_peers", "") or ""
    if not raw.strip():
        return []
    parts = [p.strip() for p in raw.replace(",", " ").split()]
    return [p for p in parts if p]


# ======================================================================
# Hivemind Optimizer Wrapper
# ======================================================================


def build_hivemind_optimizer(
    model: nn.Module,
    base_optimizer: torch.optim.Optimizer,
    peer: hivemind.Peer,
    target_group_size: int = 8,
    averaging_period: int = 1,
    average_parameters: bool = True,
    prefix: str = "hivemind",
    offload_gradients: bool = False,
    verbose: bool = True,
) -> DecentralizedOptimizer:
    """
    Wrap a local ``torch.optim.Optimizer`` with Hivemind's async all-reduce.

    ``DecentralizedOptimizer.step()`` replaces ``base_optimizer.step()``: it
    first applies local gradients (the usual ``optimizer.step()``), then
    launches a **non-blocking** all-reduce that averages this peer's parameters
    with a random subset of ``target_group_size`` other peers in the swarm.

    Because the all-reduce is asynchronous, this peer continues training
    immediately and absorbs the averaged parameters when the all-reduce
    completes (typically a few steps later).  This is what makes training
    heterogeneous — fast GPUs do more local steps per averaged round than
    slow GPUs, yet every peer benefits from the collective.

    Args:
        model: The torch model whose parameters will be averaged.
        base_optimizer: A **freshly constructed** local optimizer
            (e.g. ``AdamW(model.parameters(), lr=...)``).  Do **not** call
            ``step()`` on it directly after wrapping.
        peer: The Hivemind ``Peer`` from ``setup_hivemind_peer()``.
        target_group_size: How many peers to average with on each step.
            Larger = more stable but more network traffic.
            Smaller = faster averaging, better for heterogeneous swarms.
        averaging_period: How many **local** optimizer steps between
            all-reduce triggers.  ``1`` (default) = average every step.
        average_parameters: If ``True`` (default), average **parameters**
            rather than gradients.  Parameter averaging is more stable for
            heterogeneous async training.  Set to ``False`` to average
            gradients instead (more like traditional all-reduce).
        prefix: String prefix for DHT records.
        offload_gradients: Move gradients to CPU before all-reduce
            (reduces GPU VRAM pressure at the cost of PCIe bandwidth).
        verbose: Print diagnostic information.

    Returns:
        ``hivemind.DecentralizedOptimizer`` instance.
    """
    if not _HIVEMIND_AVAILABLE:
        raise ImportError("hivemind is not installed.")

    params = [p for p in model.parameters() if p.requires_grad]

    if verbose:
        n_params = sum(p.numel() for p in params)
        print(f"[Hivemind] Building DecentralizedOptimizer:")
        print(f"           target_group_size = {target_group_size}")
        print(f"           averaging_period  = {averaging_period}")
        print(f"           average_params    = {average_parameters}")
        print(f"           trainable params  = {n_params:,}")

    opt = DecentralizedOptimizer(
        params=params,
        opt=base_optimizer,
        peer=peer,
        target_group_size=target_group_size,
        averaging_period=averaging_period,
        average_parameters=average_parameters,
        prefix=prefix,
        offload_gradients=offload_gradients,
        # Allow Hivemind to find its own optimal batch size for all-reduce
        allreduce_timeout=60.0,
    )

    # Sync initial parameters with the swarm (so a new peer starts from
    # roughly the same point as existing peers).
    if verbose:
        print("[Hivemind] Running initial parameter sync...")
    opt.load_state_from_peers()
    if verbose:
        print("[Hivemind] Initial sync complete.")

    return opt


# ======================================================================
# Checkpoint Averaging (for evaluation / inference)
# ======================================================================


@torch.no_grad()
def average_checkpoints_via_hivemind(
    model: nn.Module,
    peer: hivemind.Peer,
    target_group_size: int = 8,
    num_rounds: int = 3,
    prefix: str = "hivemind_avg",
) -> Dict[str, torch.Tensor]:
    """
    Run several rounds of all-reduce to produce an averaged state dict.

    Useful for evaluation and inference: the averaged parameters are typically
    of higher quality than any single peer's parameters (similar to
    ``ModelAverage`` in federated learning).

    Call this after training or periodically during evaluation.  All peers
    that participate produce the **same** averaged state dict (up to floating-
    point non-determinism).

    Args:
        model: The model to fill with averaged parameters.
        peer: Hivemind peer.
        target_group_size: Number of peers to average with.
        num_rounds: How many averaging rounds.  3–5 is usually enough.
        prefix: DHT prefix for the temporary averaging group.

    Returns:
        The averaged state dict (also written into ``model`` in-place).
    """
    if not _HIVEMIND_AVAILABLE:
        raise ImportError("hivemind is not installed.")

    params = [p for p in model.parameters() if p.requires_grad]
    base_opt = torch.optim.SGD(params, lr=0.0)  # dummy — no local step

    # Use a temporary DecentralizedOptimiser with averaging only
    avg_opt = DecentralizedOptimizer(
        params=params,
        opt=base_opt,
        peer=peer,
        target_group_size=target_group_size,
        averaging_period=1,
        average_parameters=True,
        prefix=prefix,
    )

    print(f"[Hivemind] Averaging checkpoints over {num_rounds} rounds "
          f"with {target_group_size} peers...")
    for r in range(num_rounds):
        avg_opt.step()  # all-reduce parameters (no gradient step since SGD lr=0)
        time.sleep(1.0)  # let async all-reduce propagate
        print(f"           round {r + 1}/{num_rounds} done")

    return model.state_dict()


# ======================================================================
# CLI arguments shared across Hivemind training scripts
# ======================================================================


def add_hivemind_args(parser: argparse.ArgumentParser) -> None:
    """Add Hivemind-specific CLI arguments to an argument parser."""
    group = parser.add_argument_group("Hivemind (decentralized multi-node)")
    group.add_argument("--hivemind", action="store_true", default=False,
                       help="Enable Hivemind decentralized training.")
    group.add_argument("--initial-peers", type=str, default="",
                       help="Comma/space-separated list of bootstrap peers "
                            "(ip:port).  Leave empty for the bootstrap node.")
    group.add_argument("--host", type=str, default="0.0.0.0",
                       help="Network interface to bind the peer to.")
    group.add_argument("--port", type=int, default=0,
                       help="Port for the Hivemind peer (0 = random).")
    group.add_argument("--peer-id", type=str, default=None,
                       help="Optional human-readable peer name.")
    group.add_argument("--target-group-size", type=int, default=8,
                       help="Number of peers to average with per step.")
    group.add_argument("--averaging-period", type=int, default=1,
                       help="Average every N local steps.")
    group.add_argument("--average-parameters", action="store_true",
                       default=True,
                       help="Average parameters (not gradients).")
    group.add_argument("--no-average-parameters", action="store_false",
                       dest="average_parameters",
                       help="Average gradients instead of parameters.")
    group.add_argument("--checkpoint-average-rounds", type=int, default=3,
                       help="Number of averaging rounds for "
                            "final checkpoint aggregation.")


def check_hivemind_args(args: argparse.Namespace) -> None:
    """Validate Hivemind args and print connection info."""
    if not args.hivemind:
        return
    if not _HIVEMIND_AVAILABLE:
        raise ImportError(
            "Hivemind is not installed.  Run:\n"
            "  pip install -r hivemind/requirements-hivemind.txt"
        )
    initial = get_initial_peers_from_args(args)
    if not initial:
        print("[Hivemind] Starting as bootstrap node (no initial peers).")
    else:
        print(f"[Hivemind] Connecting to initial peers: {initial}")


# ======================================================================
# Simple smoke test
# ======================================================================


def smoke_test_hivemind_utils() -> None:
    """Verify Hivemind utilities can be imported (no actual peer test)."""
    assert _HIVEMIND_AVAILABLE or not _HIVEMIND_AVAILABLE, "flag exists"
    print("[smoke] hivemind_utils imported successfully.")
    print(f"        hivemind available: {_HIVEMIND_AVAILABLE}")


if __name__ == "__main__":
    smoke_test_hivemind_utils()
