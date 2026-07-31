# =============================================================================
# Advanced LLM Framework — training / inference image
#
# Build:
#   docker build -t advanced-llm-framework .
#
# Run pretrain (single GPU):
#   docker run --gpus all --shm-size 16g -it --rm \
#     -v $PWD/checkpoints:/workspace/checkpoints \
#     -v $PWD/packed:/workspace/packed \
#     advanced-llm-framework \
#     train_pretrain.py --model-size 0.3B --data-dir /workspace/packed \
#       --checkpoint-dir /workspace/checkpoints --seq-len 2048 \
#       --batch-size 32 --grad-accum 4
#
# Run SFT / GRPO / DPO the same way (swap the entrypoint command).
#
# torch comes preinstalled in the base image. To pin a different CUDA build
# (e.g. cu128), override the base tag or install torch yourself first:
#   pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# =============================================================================

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# DeepSpeed builds native ops on first use; make sure it can.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

# System deps: git (HF datasets), build tools (DeepSpeed/apex jit kernels).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# --- App dependencies first (cache layer; only busted on requirements.txt) ---
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /workspace/requirements.txt

# --- Framework code ---
COPY . /workspace

# --- Non-root user: training writes checkpoints, never run as root ---
RUN useradd --create-home --uid 1000 trainer \
    && chown -R trainer:trainer /workspace \
    && mkdir -p /data && chown trainer:trainer /data
USER trainer

# Interactive containers land in a shell; `docker run ... <script> <args>`
# runs the framework's own launcher directly.  Default: self-contained
# pretrain smoke test (no data or GPU needed).
ENTRYPOINT ["/usr/bin/env", "python"]
CMD ["-c", "import train_pretrain; train_pretrain.smoke_test()"]
