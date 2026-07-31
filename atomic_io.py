#!/usr/bin/env python3
"""
atomic_io.py

Shared crash-safe file I/O helpers used by every training script's checkpoint
save path.

Problem being solved: a naive save writes the destination file directly
(``torch.save(ckpt, path)``, ``open(path, "w")``) and then updates a
``latest`` symlink. If the process is killed mid-write — OOM killer, SIGTERM,
power loss — the destination holds a torn/partial file while ``latest``
already points at it. On resume, ``torch.load`` fails with a cryptic pickle
error or ``meta.json`` fails to parse, and the run is stuck with a corrupt
checkpoint.

Guarantee: every write goes to a sibling ``*.tmp`` file in the *same*
directory, is fsynced, and is then ``os.replace``-d onto the final name.
``os.replace`` is atomic on POSIX: a reader (``torch.load``, ``json.load``)
observes either the old complete file or the new complete file — never a torn
write. The directory is fsynced afterwards so the rename itself is durable
across power loss. A concurrent process on the same directory (e.g. two
workers racing to update ``latest``) also cannot observe a half-written
symlink.

Usage:
    from atomic_io import atomic_torch_save, atomic_write_json, atomic_symlink

    atomic_torch_save(ckpt, "step_00050.pt")
    atomic_write_json(meta, "meta.json")
    atomic_symlink("step_00050.pt", "latest_checkpoint")
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Callable

import torch


def _fsync_dir(dirpath: str) -> None:
    """Best-effort fsync of a directory so the rename is durable.

    Fails silently — some filesystems / sandboxes disallow opening a
    directory for fsync, but atomicity of the rename itself is already
    guaranteed by os.replace regardless.
    """
    try:
        fd = os.open(dirpath, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic(path: str, writer: Callable[[str], None]) -> None:
    """Write to a sibling tmp file, fsync, then atomically rename onto ``path``."""
    dirpath = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(dirpath, exist_ok=True)
    tmp = os.path.join(
        dirpath, f".{os.path.basename(path)}.tmp.{os.getpid()}"
    )
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        # Clean up the tmp file on failure (e.g. writer raised, or the
        # rename failed because the target was a non-empty directory).
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    _fsync_dir(dirpath)


def atomic_torch_save(obj: Any, path: str) -> None:
    """Crash-safe ``torch.save(obj, path)``."""

    def _write(tmp: str) -> None:
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())

    _atomic(path, _write)


def atomic_write_json(obj: Any, path: str) -> None:
    """Crash-safe ``json.dump(obj, open(path, "w"))``.

    ``default=str`` is used to tolerate non-serializable fields (torch dtypes,
    enums, objects) the same way the checkpoint code already does.
    """

    def _write(tmp: str) -> None:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

    _atomic(path, _write)


def load_torch_checkpoint(
    path: str,
    map_location: Any = None,
    *,
    allow_unsafe: bool = False,
) -> Any:
    """Load a checkpoint with ``weights_only=True`` by default.

    Every checkpoint written by this framework (model state dicts, AdamW /
    Muon optimizer state, config/args dicts) contains only tensors and
    primitives, so it loads under PyTorch's restricted pickle allowlist.
    Loading a checkpoint with ``weights_only=False`` executes arbitrary
    pickle bytecode — an RCE risk when the ``.pt`` came from a public hub
    or another untrusted source.

    ``allow_unsafe=True`` restores the old ``weights_only=False`` behavior.
    Use it ONLY for checkpoints you fully trust (e.g. hand-authored model
    dumps embedding custom classes); never for downloaded weights.

    Raises:
        RuntimeError: if the checkpoint cannot be loaded under the safe
            allowlist and ``allow_unsafe`` is False.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        if allow_unsafe:
            return torch.load(path, map_location=map_location, weights_only=False)
        raise RuntimeError(
            f"Failed to load {path} with weights_only=True: {exc}\n"
            "This framework only ever saves tensors + primitives, so its own "
            "checkpoints load safely. If this file embeds custom classes, it "
            "was not produced by this framework — pass allow_unsafe=True "
            "explicitly to load it anyway (only if you trust the source)."
        ) from exc


def atomic_write_bytes(data: bytes, path: str) -> None:
    """Crash-safe raw byte write (e.g. copied tokenizer files)."""

    def _write(tmp: str) -> None:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

    _atomic(path, _write)


def atomic_symlink(target: str, link_path: str) -> None:
    """Atomically create or replace ``link_path`` -> ``target``.

    Unlike the ``os.remove(link) + os.symlink(...)`` two-step, there is no
    window where ``link_path`` is missing entirely. Replaces a regular file
    or existing symlink in place (os.replace on a symlink replaces the link
    itself, not its target).
    """
    dirpath = os.path.dirname(os.path.abspath(link_path)) or "."
    os.makedirs(dirpath, exist_ok=True)
    tmp = os.path.join(
        dirpath, f".{os.path.basename(link_path)}.tmp.{os.getpid()}"
    )
    try:
        # os.replace on the tmp path: replace the tmp symlink itself.
        os.symlink(os.path.abspath(target), tmp)
        os.replace(tmp, link_path)
    finally:
        if os.path.lexists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    _fsync_dir(dirpath)
