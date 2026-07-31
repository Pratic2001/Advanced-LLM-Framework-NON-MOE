#!/usr/bin/env python3
"""
Unit tests for shutdown.py — graceful SIGINT/SIGTERM handling.

Run:  pytest tests/test_shutdown.py
"""

import os
import signal

import pytest
import torch

import shutdown


@pytest.fixture(autouse=True)
def _reset_state():
    """Fresh module state before and after each test."""
    shutdown._requested = set()
    shutdown._installed = False
    yield
    shutdown._requested = set()
    shutdown._installed = False


def test_shutdown_requested_starts_false():
    assert shutdown.shutdown_requested() is False


def test_handler_records_first_signal():
    shutdown._handler(signal.SIGTERM, None)
    assert shutdown.shutdown_requested() is True
    assert shutdown.shutdown_signals() == [signal.SIGTERM]


def test_second_signal_forces_exit(monkeypatch):
    # Second signal must os._exit(128+signum) rather than hang forever.
    exited = {}

    def fake_exit(code):
        exited["code"] = code

    monkeypatch.setattr(shutdown.os, "_exit", fake_exit)
    shutdown._handler(signal.SIGINT, None)
    shutdown._handler(signal.SIGTERM, None)
    assert exited["code"] == 128 + signal.SIGTERM


def test_install_is_idempotent(monkeypatch):
    calls = []

    def fake_signal(sig, handler):
        calls.append(sig)

    monkeypatch.setattr(shutdown.signal, "signal", fake_signal)
    shutdown.install_signal_handlers()
    shutdown.install_signal_handlers()
    # Only the first call installs; second is a no-op.
    assert len(calls) == 2  # SIGINT + SIGTERM


def test_should_stop_single_rank(monkeypatch):
    # world_size == 1 → pure local check, no distributed call
    assert shutdown.should_stop(torch.device("cpu"), 1) is False
    shutdown._handler(signal.SIGTERM, None)
    assert shutdown.should_stop(torch.device("cpu"), 1) is True
