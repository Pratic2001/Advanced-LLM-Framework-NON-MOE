# =============================================================================
# Advanced LLM Framework — training/inference image (GPU)
#
# After commit 010e9a8 (custom venv support) the image became the "trainer"
# in a three-service compose stack:
#
#   ui-router  ─► :3002   (landing page)
#   ui         ─► :3000, :3001  (Heavy + Lite)
#   trainer    ─► :22     (THIS image — PyTorch + framework code + sshd)
#
# The UI container reaches this one over SSH on the Tailscale MagicDNS
# hostname "trainer" (configured by the ts-trainer sidecar). The matching
# private key is baked into the UI container's /secrets/ui.env
# (SSH_PRIVATE_KEY), and the public key is baked into THIS image at build
# time via `--build-arg UI_PUBKEY=...`.
#
# Build:
#   docker build -t pratic2001/llmforge-trainer:latest \
#     --build-arg UI_PUBKEY="$(cat .secrets/ui-pubkey.pub)" .
#
# Run interactive (debug):
#   docker run --gpus all --rm -it pratic2001/llmforge-trainer:latest \
#     /bin/bash
#
# Run pretrain smoke (no GPU/data needed):
#   docker run --rm -e CUDA_VISIBLE_DEVICES="" pratic2001/llmforge-trainer:latest
# =============================================================================

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

# System deps: git (HF datasets), build tools (DeepSpeed/apex jit kernels),
# openssh-server (so the UI container can SSH in and run training scripts).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        openssh-server \
        sudo \
        && rm -rf /var/lib/apt/lists/*

# ── App dependencies first (cache layer; only busted on requirements.txt) ──
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /workspace/requirements.txt

# ── Framework code ─────────────────────────────────────────────────────────
COPY . /workspace

# ── Non-root `trainer` user that the UI container SSHes in as ─────────────
# uid 1000 matches the bind-mounted /workspace so writes from SSHed-in
# commands don't go root-owned on the host.
ARG UI_PUBKEY=""
RUN useradd --create-home --uid 1000 --shell /bin/bash trainer \
    && echo "trainer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir -p /home/trainer/.ssh \
    && echo "${UI_PUBKEY}" > /home/trainer/.ssh/authorized_keys \
    && chmod 700 /home/trainer/.ssh \
    && chmod 600 /home/trainer/.ssh/authorized_keys \
    && chown -R trainer:trainer /home/trainer/.ssh \
    && mkdir -p /workspace /data \
    && chown -R trainer:trainer /workspace /data \
    && mkdir -p /run/sshd \
    && chmod 0755 /run/sshd

# sshd config — only key auth, no password, no root login. The UI container
# authenticates as `trainer` with the private key shipped in its env file.
COPY sshd_config /etc/ssh/sshd_config

# Entrypoint starts sshd and then blocks. `docker run ... <script>` still
# works (and overrides CMD) because ENTRYPOINT execs the script with the
# user's command appended as args.
COPY trainer-entrypoint.sh /usr/local/bin/trainer-entrypoint.sh
RUN chmod +x /usr/local/bin/trainer-entrypoint.sh

USER trainer
EXPOSE 22

ENTRYPOINT ["/usr/local/bin/trainer-entrypoint.sh"]
# Default to the pretrain smoke test so a bare `docker run` stays useful.
CMD ["python", "-c", "import train_pretrain; train_pretrain.smoke_test()"]