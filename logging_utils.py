#!/usr/bin/env python3
"""
logging_utils.py

Structured logging for training scripts.

The training scripts already print human-readable progress lines (``[SFT]``,
``[Checkpoint]``, ``[Shutdown]`` ...). This module adds an opt-in **structured
event stream** on top of those prints: each lifecycle event is also emitted as
a single JSON object per line, so production deployments can ship, parse and
query training telemetry (Loki, CloudWatch, ELK, ``jq``) without losing the
interactive output.

Configuration is read from environment variables once at ``setup_logging()``:

  LOG_LEVEL     debug|info|warning|error     (default: info)
  LOG_FORMAT    json|plain                   (default: plain)
  LOG_FILE      path to append events to     (default: none)
  LOG_EVENTS    on|off                       (default: on)

``setup_logging()`` also installs a ``sys.excepthook`` that turns any uncaught
exception into a structured ``uncaught_exception`` event before the process
dies — without the hook, a crash mid-training leaves no machine-readable trace
of *what* failed.

Usage:
    from logging_utils import setup_logging, get_logger, log_event

    setup_logging()                       # once, near the top of main()/train()
    log = get_logger("train_sft")
    log_event(log, "checkpoint_saved", step=50, path="step_00050.pt")

Plain output::

    2026-07-31 11:00:00 INFO train_sft checkpoint_saved step=50 path=step_00050.pt

JSON output::

    {"ts": 1782841200.123, "level": "INFO", "logger": "train_sft",
     "event": "checkpoint_saved", "step": 50, "path": "step_00050.pt"}
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Configuration (read once at setup_logging)
# ---------------------------------------------------------------------------

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().lower()


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------


def _serialize(value: Any) -> Any:
    """Convert values that JSON can't natively hold (paths, tensors, enums)."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Paths and other os.PathLike
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    # torch tensors / dtypes / etc. — stringified so the log line never throws
    try:
        if hasattr(value, "item") and hasattr(value, "ndim") and value.ndim == 0:
            return value.item()
    except Exception:
        pass
    try:
        return str(value)
    except Exception:
        return repr(value)


def log_event(
    logger: logging.Logger,
    event: str,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit a structured event with ``event`` as the type and ``fields`` as data.

    Rendered as JSON (one object per line) when ``LOG_FORMAT=json``, otherwise
    as ``key=value`` on a single line. Always carries a timestamp and level.
    """
    if _env("LOG_EVENTS", "on") != "on":
        return
    data = {k: _serialize(v) for k, v in fields.items()}
    logger.log(level, "%(event)s %(data)s", {"event": event, "data": data})


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


class _StructuredFormatter(logging.Formatter):
    """Render one event as either a single JSON object or a key=value line."""

    def __init__(self, fmt: str) -> None:
        super().__init__()
        self._json = fmt == "json"

    def format(self, record: logging.LogRecord) -> str:
        # The event payload arrives via record.args: {"event": ..., "data": {...}}
        event = ""
        data: Dict[str, Any] = {}
        if isinstance(record.args, dict):
            event = record.args.get("event", "")
            data = record.args.get("data", {})
        ts = time.time()

        if self._json:
            obj: Dict[str, Any] = {
                "ts": round(ts, 3),
                "level": record.levelname,
                "logger": record.name,
            }
            if event:
                obj["event"] = event
            obj.update(data)
            return json.dumps(obj, default=str)

        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        parts = [f"{k}={v}" for k, v in sorted(data.items())]
        suffix = f" {event}" if event else ""
        kv = (" " + " ".join(parts)) if parts else ""
        return f"{when} {record.levelname:7s} {record.name}{suffix}{kv}"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def setup_logging() -> logging.Logger:
    """Configure the root logger and return the ``"train"`` child logger.

    Safe to call multiple times (idempotent). Also installs a
    ``sys.excepthook`` so uncaught exceptions emit a structured event.
    """
    fmt = _env("LOG_FORMAT", "plain")
    if fmt not in ("json", "plain"):
        fmt = "plain"
    level = _LEVELS.get(_env("LOG_LEVEL", "info"), logging.INFO)

    root = logging.getLogger()
    # Reset any handlers configured by a previous setup_logging() so re-entry
    # (e.g. importing a module that also calls it) doesn't duplicate output.
    if getattr(root, "_structured_configured", False):
        return logging.getLogger("train")

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_StructuredFormatter(fmt))
    root.addHandler(handler)
    root.setLevel(level)

    log_file = os.environ.get("LOG_FILE", "").strip()
    if log_file:
        try:
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setFormatter(_StructuredFormatter("json"))
            root.addHandler(fh)
        except OSError:
            # Best-effort: a bad LOG_FILE path must not take down training.
            pass

    # Never emit our own events to stdout (training prints are stdout).
    logging.getLogger("train").propagate = True

    root._structured_configured = True  # type: ignore[attr-defined]
    _install_excepthook()
    return logging.getLogger("train")


def _install_excepthook() -> None:
    """Emit a structured ``uncaught_exception`` event when the process dies.

    Replaces sys.excepthook once; nested handlers are preserved so tests /
    embedding code that sets their own hook keeps working.
    """
    previous = getattr(sys, "excepthook", None)
    if getattr(_install_excepthook, "_installed", False):
        return

    def _hook(etype, value, tb):
        try:
            log = logging.getLogger("train")
            log_event(
                log,
                "uncaught_exception",
                level=logging.ERROR,
                exception=f"{etype.__name__}: {value}",
                traceback="".join(traceback.format_exception(etype, value, tb)),
            )
        except Exception:
            pass  # never mask the real failure
        if previous is not None and previous is not sys.__excepthook__:
            try:
                previous(etype, value, tb)
            except Exception:
                pass
        else:
            sys.__excepthook__(etype, value, tb)

    _install_excepthook._installed = True  # type: ignore[attr-defined]
    sys.excepthook = _hook


def get_logger(name: str = "train") -> logging.Logger:
    """Return a child logger (no setup required if setup_logging ran)."""
    return logging.getLogger(name)
