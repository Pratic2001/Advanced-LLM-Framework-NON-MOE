#!/bin/bash
# =============================================================================
# Trainer container entrypoint — starts sshd and then hands off to whatever
# command the user passed (`docker run ... python train_pretrain.py ...`).
#
# Why ENTRYPOINT and not CMD: this lets `docker run --rm trainer:latest`
# default to the pretrain smoke test (CMD) while still allowing the script
# to keep sshd alive in the background during interactive runs.
# =============================================================================

set -euo pipefail

# sshd logs to stderr by default; redirect to stdout so docker compose logs
# picks them up.
sudo /usr/sbin/sshd -D -e &
SSHD_PID=$!

# If we get SIGTERM, shut sshd down cleanly so docker stop doesn't wait
# 10s for the kernel to SIGKILL it.
trap "kill ${SSHD_PID} 2>/dev/null || true" SIGTERM SIGINT

# Hand off to the requested command (or the smoke test from CMD).
exec "$@"