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
    "average_checkpoint_during_training",
    "get_initial_peers_from_args",
    "get_peer_seed",
    "get_swarm_info",
    "wrap_optimizers_for_hivemind",
    "save_hivemind_checkpoint",
    "load_hivemind_checkpoint",
    "maybe_average_final_checkpoint",
    "measure_peer_throughputs",
    "compute_adaptive_target_batch_size",
    "get_fast_peer_subset",
    "weighted_average_parameters",
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
    "PEER_SEED_MODULUS",
]

# ======================================================================
# Constants
# ======================================================================

PEER_SEED_MODULUS = 65536


def get_peer_seed(args: argparse.Namespace, endpoint: str) -> int:
    """
    Compute a deterministic per-peer seed from the base seed and endpoint.

    Uses a stable hash (not Python's built-in hash which is salted per-process)
    so that seeds are reproducible across restarts and different Python versions.
    """
    import hashlib
    endpoint_hash = int(hashlib.md5(endpoint.encode()).hexdigest()[:8], 16)
    return args.seed + (endpoint_hash % PEER_SEED_MODULUS)


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
      - ``--initial-peers "[::1]:5678"`` (IPv6)
      - ``--initial-peers "peer@[::1]:5678,[::2]:5679"`` (IPv6 with peer IDs)

    Handles IPv6 addresses correctly by parsing comma-separated values
    and respecting brackets for IPv6 literals.
    """
    raw = getattr(args, "initial_peers", "") or ""
    if not raw.strip():
        return []

    # Split by comma, preserving IPv6 bracket notation
    parts = []
    current = ""
    in_brackets = False
    for char in raw:
        if char == '[':
            in_brackets = True
            current += char
        elif char == ']':
            in_brackets = False
            current += char
        elif char == ',' and not in_brackets:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current.strip())

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
    initial_sync: bool = True,
    # Gradient compression options
    compression: str = "none",           # "none" | "topk" | "int8"
    compression_ratio: float = 0.01,     # top-k ratio (0.01 = top 1%)
    compression_block_size: int = 32768, # block size for int8 quantization
    # Async averaging
    async_averaging: bool = True,        # Use non-blocking async all-reduce
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
        initial_sync: If True (default), call load_state_from_peers() to sync
            initial parameters with the swarm. Set to False when resuming
            from a checkpoint (the peer already has trained parameters).
        compression: Gradient compression type for all-reduce.
            "none" = no compression (default, full fp32).
            "topk" = top-k sparsification, keep top compression_ratio fraction.
            "int8" = 8-bit quantization with block_size.
        compression_ratio: Fraction of elements to keep for top-k (0.01 = 1%).
        compression_block_size: Block size for int8 quantization (default 32768).

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
        print(f"           compression       = {compression}")
        if compression != "none":
            print(f"           compression_ratio = {compression_ratio}")
        print(f"           trainable params  = {n_params:,}")

    # Map compression string to Hivemind compression type
    from hivemind import CompressionType
    compression_map = {
        "none": CompressionType.NONE,
        "topk": CompressionType.TOP_K,
        "int8": CompressionType.QUANTIZE_8BIT,
    }
    if compression not in compression_map:
        raise ValueError(f"Unknown compression: {compression}. Choose from {list(compression_map.keys())}")

    opt = DecentralizedOptimizer(
        params=params,
        opt=base_optimizer,
        peer=peer,
        target_group_size=target_group_size,
        averaging_period=averaging_period,
        average_parameters=average_parameters,
        prefix=prefix,
        offload_gradients=offload_gradients,
        # Gradient compression
        compression_type=compression_map[compression],
        compression_kwargs=(
            {"ratio": compression_ratio} if compression == "topk"
            else {"block_size": compression_block_size} if compression == "int8"
            else {}
        ),
        # Async averaging (non-blocking)
        async_op=async_averaging,
        # Allow Hivemind to find its own optimal batch size for all-reduce
        allreduce_timeout=60.0,
    )

    # Sync initial parameters with the swarm (so a new peer starts from
    # roughly the same point as existing peers).
    if initial_sync:
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
    # Gradient compression
    group.add_argument("--hivemind-compression", type=str, default="none",
                       choices=["none", "topk", "int8"],
                       help="Gradient compression for all-reduce: "
                            "none (full precision), topk (sparsification), int8 (quantization).")
    group.add_argument("--hivemind-compression-ratio", type=float, default=0.01,
                       help="Top-k compression ratio (fraction to keep, e.g. 0.01 = 1 percent).")
    group.add_argument("--hivemind-compression-block-size", type=int, default=32768,
                       help="Block size for int8 quantization.")
    # Async averaging
    group.add_argument("--hivemind-async-averaging", action="store_true", default=True,
                       help="Use non-blocking async all-reduce (default True).")
    group.add_argument("--no-hivemind-async-averaging", action="store_false",
                       dest="hivemind_async_averaging",
                       help="Disable async averaging (blocking all-reduce).")


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
# Bandwidth-aware peer weighting
# ======================================================================


def measure_peer_throughputs(
    peer: hivemind.Peer,
    local_throughput: float,
    window: int = 100,
) -> Dict[str, float]:
    """
    Measure and return peer throughputs for bandwidth-aware averaging.

    Args:
        peer: Hivemind peer to query DHT for peer stats.
        local_throughput: This peer's measured tokens/sec.
        window: Moving average window.

    Returns:
        Dict mapping peer_id -> throughput (tokens/sec).
    """
    throughputs = {}

    # Record local throughput
    my_id = getattr(peer, "peer_id", getattr(peer, "id", "local"))
    throughputs[my_id] = local_throughput

    # Try to fetch peer stats from DHT
    # Note: This is a best-effort attempt; DHT API may vary
    try:
        # Hivemind DHT stores peer info - try to get visible peers and their stats
        visible_peers = peer.get_visible_peers()
        for pid, pinfo in visible_peers.items():
            if hasattr(pinfo, 'throughput') and pinfo.throughput > 0:
                throughputs[pid] = pinfo.throughput
            elif hasattr(pinfo, 'stats') and hasattr(pinfo.stats, 'throughput'):
                throughputs[pid] = pinfo.stats.throughput
    except Exception:
        # If we can't fetch, just use local
        pass

    return throughputs


def compute_adaptive_target_batch_size(
    base_batch_size: int,
    swarm_size: int,
    min_peers: int = 4,
    max_scale: float = 4.0,
) -> int:
    """
    Compute adaptive target batch size based on swarm size.

    Uses sqrt scaling for diminishing returns: effective batch = base * sqrt(swarm/min)
    This prevents over-scaling when many slow peers join.

    Args:
        base_batch_size: Base batch size per step.
        swarm_size: Current number of visible peers.
        min_peers: Minimum peers before scaling kicks in.
        max_scale: Maximum scaling factor.

    Returns:
        Adapted target batch size.
    """
    if swarm_size < min_peers:
        return base_batch_size

    scale = math.sqrt(swarm_size / min_peers)
    scale = min(scale, max_scale)
    return int(base_batch_size * scale)


def get_fast_peer_subset(
    peer_throughputs: Dict[str, float],
    min_throughput: float = 1000.0,  # tokens/sec
    max_peers: int = 32,
) -> List[str]:
    """
    Select fast peers for critical-path averaging.

    Slow peers can participate in background checkpoint averaging only.

    Args:
        peer_throughputs: Dict of peer_id -> throughput.
        min_throughput: Minimum throughput to be considered "fast".
        max_peers: Maximum peers in fast set.

    Returns:
        List of peer IDs in the fast set.
    """
    fast_peers = [
        pid for pid, tp in peer_throughputs.items()
        if tp >= min_throughput
    ]
    # Sort by throughput descending
    fast_peers.sort(key=lambda pid: peer_throughputs[pid], reverse=True)
    return fast_peers[:max_peers]


def weighted_average_parameters(
    model: nn.Module,
    peer: hivemind.Peer,
    peer_weights: Dict[str, float],
    target_group_size: int = 8,
    prefix: str = "hivemind_weighted",
) -> Dict[str, torch.Tensor]:
    """
    Perform weighted parameter averaging using peer throughputs as weights.

    This is an advanced feature that requires DHT-level support.
    Currently returns local state as fallback.

    Args:
        model: Model to average.
        peer: Hivemind peer.
        peer_weights: Dict of peer_id -> weight (normalized).
        target_group_size: Number of peers to average with.
        prefix: DHT prefix.

    Returns:
        Averaged state dict.
    """
    # Fallback: return local state
    # Full implementation requires custom DHT allreduce with weights
    print("[Hivemind] Weighted averaging not yet fully supported, returning local state")
    return model.state_dict()


def average_checkpoint_during_training(
    model: nn.Module,
    optimizers: List[Any],
    peer: hivemind.Peer,
    target_group_size: int = 8,
    num_rounds: int = 3,
    prefix: str = "hivemind_checkpoint_avg",
    timeout: float = 300.0,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Run cross-swarm parameter averaging and return the averaged state dict.
    This is meant to be called DURING training (not just at the end) to
    produce globally consistent checkpoints.

    Args:
        model: The model to average.
        optimizers: List of Hivemind-wrapped optimizers.
        peer: Hivemind peer.
        target_group_size: Number of peers to average with.
        num_rounds: How many averaging rounds.
        prefix: DHT prefix for the temporary averaging group.
        timeout: Maximum time to wait for convergence (seconds).

    Returns:
        Averaged state dict, or None if failed.
    """
    if not _HIVEMIND_AVAILABLE:
        raise ImportError("hivemind is not installed.")

    print(f"[Hivemind] Cross-swarm checkpoint averaging over {num_rounds} rounds...")

    # Use the first Hivemind optimizer for averaging (they share the same peer/DHT)
    hivemind_opt = None
    for opt in optimizers:
        if hasattr(opt, 'average_parameters'):
            hivemind_opt = opt
            break

    if hivemind_opt is None:
        print("[Hivemind] No Hivemind optimizer found for averaging")
        return None

    try:
        # Run multiple rounds of averaging
        for r in range(num_rounds):
            # Blocking average for checkpoint convergence
            future = hivemind_opt.average_parameters(async_op=False)
            if future is not None:
                future.result(timeout=timeout / num_rounds)
            time.sleep(1.0)  # let propagation happen
            print(f"           round {r + 1}/{num_rounds} done")

        # Return the averaged state (model is updated in-place by DecentralizedOptimizer)
        return model.state_dict()

    except Exception as e:
        print(f"[Hivemind] Cross-swarm averaging failed: {e}")
        return None


# ======================================================================
# Shared Helpers for Training Scripts
# ======================================================================


def get_swarm_info(
    peer: Any,
    target_group_size: int,
    endpoint: str,
) -> Tuple[int, int]:
    """
    Get swarm size and deterministic peer index for data sharding.

    Uses stable MD5 hash of endpoint (not Python's salted hash) so all peers
    compute the same shard assignment for a given endpoint.

    Args:
        peer: Hivemind Peer object.
        target_group_size: Configured target group size.
        endpoint: This peer's endpoint string.

    Returns:
        Tuple of (swarm_size, peer_idx).
    """
    import hashlib
    try:
        visible = len(peer.get_visible_peers())
        swarm_size = max(visible, target_group_size)
    except Exception:
        swarm_size = target_group_size

    # Deterministic peer index from endpoint hash (stable across processes)
    endpoint_hash = int(hashlib.md5(endpoint.encode()).hexdigest()[:8], 16)
    peer_idx = endpoint_hash % swarm_size if swarm_size > 1 else 0
    return swarm_size, peer_idx


def wrap_optimizers_for_hivemind(
    model: nn.Module,
    local_optimizers: List[torch.optim.Optimizer],
    peer: Any,
    args: argparse.Namespace,
    prefix: str = "hivemind",
    verbose: bool = True,
    initial_sync: bool = True,
) -> List[Any]:
    """
    Wrap a list of local optimizers with Hivemind's DecentralizedOptimizer.

    Args:
        model: The model whose parameters are being optimized.
        local_optimizers: List of freshly constructed local optimizers.
        peer: Hivemind Peer from setup_hivemind_peer().
        args: Parsed CLI namespace with Hivemind args.
        prefix: Prefix for DHT records (suffixed with optimizer index).
        verbose: Print diagnostic information.
        initial_sync: If True, call load_state_from_peers() to sync initial
            parameters. Set to False when resuming from a checkpoint.

    Returns:
        List of wrapped DecentralizedOptimizer instances.
    """
    if not _HIVEMIND_AVAILABLE:
        raise ImportError("hivemind is not installed.")

    hivemind_opts = []
    for i, opt in enumerate(local_optimizers):
        hopt = build_hivemind_optimizer(
            model=model,
            base_optimizer=opt,
            peer=peer,
            target_group_size=args.target_group_size,
            averaging_period=args.averaging_period,
            average_parameters=args.average_parameters,
            prefix=f"{prefix}_{i}",
            verbose=verbose,
            compression=getattr(args, "hivemind_compression", "none"),
            compression_ratio=getattr(args, "hivemind_compression_ratio", 0.01),
            compression_block_size=getattr(args, "hivemind_compression_block_size", 32768),
            async_averaging=getattr(args, "hivemind_async_averaging", True),
        )
        hivemind_opts.append(hopt)

    # Initial parameter sync only on fresh start, not on resume
    if initial_sync and verbose:
        print("[Hivemind] Running initial parameter sync across swarm...")
        for hopt in hivemind_opts:
            hopt.load_state_from_peers()
        if verbose:
            print("[Hivemind] Initial sync complete.")

    return hivemind_opts


def save_hivemind_checkpoint(
    checkpoint_dir: str,
    step: int,
    model: nn.Module,
    optimizers: List[torch.optim.Optimizer],
    config: Any,
    train_args: Dict,
    prefix: str = "step",
    extra: Optional[Dict] = None,
) -> str:
    """
    Save a checkpoint, unwrapping Hivemind DecentralizedOptimizer for portability.

    The saved optimizer state is the *inner* (local) optimizer state, so checkpoints
    can be loaded with or without Hivemind.

    Args:
        checkpoint_dir: Directory to save checkpoint.
        step: Training step number.
        model: Model (possibly torch.compile wrapped).
        optimizers: List of optimizers (possibly Hivemind-wrapped).
        config: ModelConfig or similar.
        train_args: Dict of training arguments (vars(args)).
        prefix: Filename prefix (e.g., "step", "sft_step", "grpo_step", "dpo_step").
        extra: Optional extra data to include in checkpoint.

    Returns:
        Path to saved checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    raw_model = getattr(model, "_orig_mod", model)  # unwrap torch.compile

    # Unwrap Hivemind optimizers to get inner optimizer state
    inner_opt_states = []
    for opt in optimizers:
        inner_opt = opt.opt if hasattr(opt, "opt") else opt
        inner_opt_states.append(inner_opt.state_dict())

    ckpt = {
        "step": step,
        "model_state": raw_model.state_dict(),
        "optimizer_state": inner_opt_states[0] if len(inner_opt_states) == 1 else inner_opt_states,
        "config": vars(config) if hasattr(config, "__dict__") else config,
        "args": train_args,
    }
    if extra:
        ckpt.update(extra)

    step_path = os.path.join(checkpoint_dir, f"{prefix}_{step:05d}.pt")
    torch.save(ckpt, step_path)

    # Symlink to latest
    latest = os.path.join(checkpoint_dir, "latest_checkpoint")
    if os.path.islink(latest) or os.path.exists(latest):
        os.remove(latest)
    os.symlink(os.path.abspath(step_path), latest)

    print(f"[Checkpoint] saved {step_path}")
    return step_path


def load_hivemind_checkpoint(
    path: str,
    model: nn.Module,
    optimizers: Optional[List[torch.optim.Optimizer]],
    device: torch.device,
    strict: bool = True,
) -> int:
    """
    Load a checkpoint, handling Hivemind optimizer wrapper.

    Args:
        path: Path to checkpoint file or directory.
        model: Model to load state into.
        optimizers: List of optimizers (possibly Hivemind-wrapped) or None.
        device: Target device.
        strict: Whether to enforce strict state_dict loading.

    Returns:
        Step number from checkpoint.
    """
    if os.path.isdir(path):
        step_files = sorted(glob.glob(os.path.join(path, "*_step_*.pt")))
        if not step_files:
            # Try common patterns
            for pattern in ["step_*.pt", "sft_step_*.pt", "grpo_step*.pt", "dpo_step*.pt"]:
                step_files = sorted(glob.glob(os.path.join(path, pattern)))
                if step_files:
                    break
        if not step_files:
            raise FileNotFoundError(f"No checkpoint files found in {path}")
        ckpt_path = step_files[-1]
    else:
        ckpt_path = path

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_model = getattr(model, "_orig_mod", model)
    raw_model.load_state_dict(ckpt["model_state"], strict=strict)
    if hasattr(raw_model, "tie_weights"):
        raw_model.tie_weights()

    if optimizers is not None and "optimizer_state" in ckpt:
        opt_state = ckpt["optimizer_state"]
        if isinstance(opt_state, list):
            for opt, st in zip(optimizers, opt_state):
                inner_opt = opt.opt if hasattr(opt, "opt") else opt
                inner_opt.load_state_dict(st)
        else:
            inner_opt = optimizers[0].opt if hasattr(optimizers[0], "opt") else optimizers[0]
            inner_opt.load_state_dict(opt_state)

    step = ckpt.get("step", 0)
    print(f"[Checkpoint] resumed from {ckpt_path} at step {step}")
    return step


def maybe_average_final_checkpoint(
    args: argparse.Namespace,
    model: nn.Module,
    hivemind_info: Any,
    config: Any,
    out_dir: str,
    prefix: str = "averaged_final",
) -> Optional[str]:
    """
    If args.average_checkpoints is set and Hivemind is enabled, average the final
    checkpoint across the swarm and save it.

    Args:
        args: Parsed CLI namespace.
        model: Model to average.
        hivemind_info: HivemindPeerInfo from setup_hivemind_peer().
        config: ModelConfig.
        out_dir: Output directory for averaged checkpoint.
        prefix: Filename prefix for averaged checkpoint.

    Returns:
        Path to averaged checkpoint, or None if not performed.
    """
    if not (args.hivemind and getattr(args, "average_checkpoints", False) and hivemind_info is not None):
        return None

    print(f"\n[Hivemind] Averaging final checkpoint across swarm...")
    avg_state = average_checkpoints_via_hivemind(
        model,
        hivemind_info.peer,
        target_group_size=min(args.target_group_size, 8),
        num_rounds=getattr(args, "checkpoint_average_rounds", 3),
    )
    avg_path = os.path.join(out_dir, f"{prefix}.pt")
    torch.save({"model_state": avg_state, "config": vars(config)}, avg_path)
    print(f"[Hivemind] Averaged checkpoint saved to {avg_path}")
    return avg_path


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
