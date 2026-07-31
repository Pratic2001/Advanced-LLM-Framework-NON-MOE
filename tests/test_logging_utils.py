#!/usr/bin/env python3
"""
Unit tests for logging_utils.py — structured (JSON) event logging.

Run:  pytest tests/test_logging_utils.py
"""

import io
import json
import logging
import os
import sys
from contextlib import redirect_stderr

import pytest

import logging_utils


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Isolate each test: fresh env vars + fresh root logger handlers."""
    for var in ("LOG_FORMAT", "LOG_LEVEL", "LOG_FILE", "LOG_EVENTS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    monkeypatch.setattr(logging_utils._install_excepthook, "_installed", False,
                        raising=False)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    if hasattr(root, "_structured_configured"):
        del root._structured_configured
    yield


def test_plain_format_default(monkeypatch):
    buf = io.StringIO()
    with redirect_stderr(buf):
        log = logging_utils.setup_logging()
        logging_utils.log_event(log, "pretrain_start", model_size="0.6B", num_steps=100)
    line = buf.getvalue().strip()
    assert "pretrain_start" in line
    assert "model_size=0.6B" in line
    assert "num_steps=100" in line


def test_json_format(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "json")
    buf = io.StringIO()
    with redirect_stderr(buf):
        log = logging_utils.setup_logging()
        logging_utils.log_event(log, "checkpoint_saved", step=7, path="/tmp/x.pt")
    obj = json.loads(buf.getvalue().strip())
    assert obj["event"] == "checkpoint_saved"
    assert obj["step"] == 7
    assert obj["path"] == "/tmp/x.pt"
    for k in ("ts", "level", "logger", "event"):
        assert k in obj


def test_serialize_pathlike(monkeypatch):
    import pathlib
    monkeypatch.setenv("LOG_FORMAT", "json")
    buf = io.StringIO()
    with redirect_stderr(buf):
        log = logging_utils.setup_logging()
        logging_utils.log_event(log, "event", p=pathlib.Path("/a/b.pt"))
    obj = json.loads(buf.getvalue().strip())
    assert obj["p"] == "/a/b.pt"


def test_log_events_off(monkeypatch):
    monkeypatch.setenv("LOG_EVENTS", "off")
    buf = io.StringIO()
    with redirect_stderr(buf):
        log = logging_utils.setup_logging()
        logging_utils.log_event(log, "should_not_appear", x=1)
    assert "should_not_appear" not in buf.getvalue()


def test_excepthook_emits_uncaught_exception(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "json")
    buf = io.StringIO()
    with redirect_stderr(buf):
        logging_utils.setup_logging()
        try:
            raise ValueError("boom")
        except ValueError:
            sys.excepthook(*sys.exc_info())
    out = buf.getvalue().strip()
    assert "uncaught_exception" in out
    assert "ValueError: boom" in out


def test_excepthook_preserves_previous(monkeypatch):
    calls = []

    def prev(etype, value, tb):
        calls.append((etype, value))

    monkeypatch.setattr(sys, "excepthook", prev)
    logging_utils.setup_logging()
    try:
        raise RuntimeError("x")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    assert calls and calls[0][1].args == ("x",)


def test_get_logger_without_setup():
    log = logging_utils.get_logger("sometrainer")
    assert log.name == "sometrainer"
