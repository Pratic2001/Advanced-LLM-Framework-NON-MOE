# ⚡ Dense LLM Framework

> **A complete, production-grade pipeline for training dense (non-MoE) language models from scratch — tokenizer → data curation → pretrain → SFT → GRPO/DPO → inference.**

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/torch-2.4%2B-orange" alt="PyTorch 2.4+"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"/>
  <img src="https://img.shields.io/badge/model%20size-10M%20to%201T%2B-purple" alt="10M to 1T+"/>
  <img src="https://img.shields.io/badge/deepspeed-ready-brightgreen" alt="DeepSpeed Ready"/>
  <img src="https://img.shields.io/badge/LoRA%2FDoRA-supported-yellow" alt="LoRA/DoRA"/>
  <img src="https://img.shields.io/badge/GRPO%2FDPO-ready-red" alt="GRPO/DPO"/>
</p>

---

## 📋 Table of Contents

- [What is this?](#-what-is-this)
- [Pipeline at a Glance](#-pipeline-at-a-glance)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Module Overview](#-module-overview)
- [Repository Structure](#-repository-structure)
- [Requirements](#-requirements)
- [Containerized Deployment](#-containerized-deployment)
  - [Fresh-PC Setup](#fresh-pc-setup)
  - [Multi-PC Deployment](#multi-pc-deployment)
- [Release Pipeline](#-release-pipeline)
- [Full Documentation](#-full-documentation)
- [License & Citation](#-license--citation)

---

## 🧠 What is this?

This framework lets you **pre-train a dense transformer from scratch**, then fine-tune it through **SFT**, **GRPO**, and **DPO** — all on commodity GPUs. It includes an integrated **data curation agent** that scrapes the web, discovers HuggingFace/Kaggle datasets, uses Ollama-based LLM quality judges, and packs everything into the binary format each training stage needs.

**Why dense (non-MoE)?** Dense models are simpler to train, more predictable in memory usage, and easier to deploy on consumer hardware. If you don't need the parameter-count-to-ROI ratio of Mixture-of-Experts, this framework gives you a fast, debuggable dense alternative.

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   🏗️ Pretrain →  🔧 SFT  →  🎯 GRPO / DPO  →  🚀 Infer             │
│                                                                      │
│   ┌──────────┐   ┌──────────┐   ┌────────────┐   ┌──────────────┐   │
│   │  Self-   │   │ Super-   │   │  Group     │   │  Streaming   │   │
│   │  Super-  │──▶│ vised    │──▶│  Relative  │──▶│  Inference   │   │
│   │  vised   │   │ Fine-    │   │  Policy    │   │  + Quant     │   │
│   │          │   │ Tuning   │   │  Optim.    │   │              │   │
│   └──────────┘   └──────────┘   └────────────┘   └──────────────┘   │
│         │              │               │               │             │
│         ▼              ▼               ▼               ▼             │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │                 Integrated Data Curation                      │   │
│   │  Web Scrape · HF Datasets · Kaggle · LLM Quality Judge       │   │
│   │  Codegen Pipeline · Dedup · Shard Writer                     │   │
│   └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Pipeline at a Glance

```
                    ┌─────────────────────┐
                    │   Train Tokenizer    │
                    │  tokenizer_train.py  │
                    └──────────┬──────────┘
                               │ tokenizer.json
                               ▼
┌────────────────────────────────────────────────────────────────┐
│                     DATA CURATION                               │
│                                                                │
│  ┌──────────────────┐   ┌──────────────────┐                   │
│  │  dataset_agent.py │   │ codegen_pipeline │                   │
│  │  (LLM-planned)    │   │ (simpler, no LLM │                   │
│  │  + MCP scraper    │   │  per row)        │                   │
│  └────────┬─────────┘   └────────┬─────────┘                   │
│           │                      │                              │
│           └──────────┬───────────┘                              │
│                      ▼                                          │
│  ┌──────────────────────────────────────────────────┐          │
│  │   Public Dataset Sources                          │          │
│  │   HuggingFace Hub · Kaggle · Auto-discovery       │          │
│  └──────────────────┬───────────────────────────────┘          │
│                     │ JSONL shards                              │
│                     ▼                                           │
│  ┌──────────────────────────────────────────────────┐          │
│  │   hf_to_packed.py  ── single dataset → pack     │          │
│  └──────────────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────────────┘
                               │ JSONL shards
                               ▼
          ┌─────────────────────────────────────┐
          │         PACK DATA (──>.bin)          │
          │                                      │
          │  pack_pretrain.py                    │
          │  pack_sft.py     pack_grpo.py        │
          │  pack_dpo.py                         │
          └──────────┬──────────────────────────┘
                     │ memmap .bin files
                     ▼
┌────────────────────────────────────────────────────────────────┐
│                       TRAINING                                  │
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   PRETRAIN   │  │     SFT      │  │   GRPO / DPO         │  │
│  │              │  │              │  │                      │  │
│  │  train_pre-  │  │ train_sft.py │  │ train_grpo.py        │  │
│  │  train.py    │  │ (DDP)        │  │ train_dpo.py (DDP)   │  │
│  │  (DDP)       │  │              │  │                      │  │
│  │              │  │ train_sft_   │  │ train_grpo_deep-     │  │
│  │  train_pre-  │  │ deepspeed.py │  │ speed.py             │  │
│  │  train_deep- │  │              │  │                      │  │
│  │  speed.py    │  │ LoRA · DoRA  │  │ train_dpo_deep-      │  │
│  │              │  │              │  │ speed.py             │  │
│  │  Muon / WSD  │  │ rsLoRA       │  │                      │  │
│  │  Schedules   │  │              │  │ Reward tiers         │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
│         └─────────────────┴──────────────────────┘              │
│                           │ checkpoints                         │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    INFERENCE                              │  │
│  │  infer.py — REPL · one-shot · batch · quant (4/8 bit)    │  │
│  │  Streaming · Top-K · Top-P · Temperature · Repetition    │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🔬 Model Architecture (`model.py`)
| Feature | Support |
|---------|---------|
| **Architecture** | Dense decoder-only (non-MoE), **Jamba hybrid (SSM + Attention)**, **MoD (Mixture-of-Depth)** |
| **Layer computation** | Sequential (default) or **Parallel** (PaLM-style, ~15% faster) |
| **Attention** | GQA (Grouped-Query Attention) with configurable KV heads |
| **MLA** | **Multi-head Latent Attention** — DeepSeek-style low-rank KV joint compression (~4× smaller cache) |
| **Mamba SSM** | **Mamba state-space model block** — pure PyTorch + optional `mamba_ssm` CUDA kernel |
| **Sliding Window** | **Local sliding-window attention** with alternating global/sw layers |
| **Position** | RoPE (Rotary Position Embeddings) with **YaRN / NTK-aware** frequency scaling |
| **Multi-Token Prediction** | **MTP** — predict K auxiliary future tokens per position (discounted loss) |
| **Normalization** | Pre-RMSNorm (or LayerNorm) |
| **Activation** | SwiGLU or GELU |
| **QK-Norm** | Optional per-head QK RMSNorm before RoPE |
| **SDPA Backend** | Explicit FLASH / EFFICIENT pinning with **MATH fallback on CPU** |
| **Sizes** | Auto-sized from 10M → 1T+ via `ModelConfig.from_target_size()` |
| **Init** | Carefully tuned std scaling per layer type |

### 🏋️ Training
| Feature | Support |
|---------|---------|
| **Pretrain** | Torch DDP + DeepSpeed (ZeRO 1/2/3 auto-select) |
| **SFT** | Full fine-tune, LoRA, DoRA, rsLoRA |
| **GRPO** | Group Relative Policy Optimization with reward tiers |
| **DPO** | Direct Preference Optimization with preference pairs |
| **Optimizers** | AdamW, FusedAdam, **Muon** (2-3× faster convergence) |
| **Schedules** | Cosine + WSD (Warmup-Stable-Decay) |
| **Architecture Variants** | All scripts support `--arch` (dense/jamba), `--layer-type` (sequential/parallel), `--sliding-window-size`, `--mod-alpha`, `--num-mtp-heads`, `--use-mla`, `--jamba-interval` |
| **Gradient** | Checkpointing, gradient accumulation, gradient clipping |
| **Logging** | W&B integration, loss curves, VRAM estimates |
| **Precision** | FP32 / BF16 mixed precision |
| **Smoke tests** | Every training script has `--smoke-test` |
| **Resume** | Full checkpoint save/load/continue |
| **Decentralized** | Hivemind — heterogeneous multi-node async training (see `hivemind/`) |

### 🗂️ Data Curation
| Feature | Support |
|---------|---------|
| **Web scraping** | MCP server with DuckDuckGo search, multi-format extraction (HTML, PDF, DOCX, PPTX, XLSX, images, video/audio transcripts) |
| **HF Datasets** | Auto-discovery, streaming, column mapping via Ollama |
| **Kaggle** | Dataset search and download |
| **LLM Quality Judge** | Ollama-powered per-row quality filtering |
| **Codegen Pipeline** | Simpler alternative — Ollama writes standalone extractor scripts |
| **Reasoning Codegen** | LangGraph workflow with step-by-step schema analysis (`--reasoning-model`) |
| **hf_to_packed.py** | Single-dataset: sample → codegen → validate → run → pack (auto-fix loop) |
| **Dedup** | ExactDedup + NearDedup (MinHash LSH) |
| **Shard Writer** | Auto-sharded JSONL output with byte budget tracking |

### 🔧 Infrastructure
| Feature | Support |
|---------|---------|
| **Recipe System** | `recipe.py` — single source of truth for templates, tokens, modes |
| **Chat Templates** | ChatML, Llama, Mistral, Gemma, DeepSeek |
| **Thinking Modes** | `reasoning` / `non_reasoning` / `hybrid` — per-example think-tag control |
| **Multi-GPU** | `torchrun` DDP + DeepSpeed ZeRO |
| **Multi-Node** | DeepSpeed with hostfile |
| **Inference** | REPL, one-shot, batched, **4-bit/8-bit quantized** |

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+ and CUDA-capable GPU
# Install PyTorch first (matching your CUDA version):
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install framework dependencies:
pip install -r requirements.txt

# For data curation, start Ollama:
ollama serve
ollama pull llama3.1
```

### End-to-End: 10 Minutes to a Trained Model

```bash
# ── Step 1: Train a tokenizer ──────────────────────────────────
python tokenizer_train.py --data-dir ./data --output-dir ./tokenizer

# ── Step 2: Curate data ────────────────────────────────────────
#    Option A: Codegen pipeline (simpler)
python webscrapped_dataset_curator_AI_MCP/agent/codegen_pipeline.py \
    --target-size 200MB --categories web,knowledge \
    --out-dir ./data --mode pretrain --public-only

#    Option B: Single HF dataset (fastest)
python webscrapped_dataset_curator_AI_MCP/agent/hf_to_packed.py \
    --dataset c4 --config en --mode pretrain \
    --tokenizer ./tokenizer --seq-length 1024 --out-dir ./packed

# ── Step 3: Pack data into memmap ──────────────────────────────
python data/pack_pretrain.py \
    --data-dir ./data --tokenizer ./tokenizer --cache-dir ./packed

# ── Step 4: Pretrain (0.3B, single GPU) ────────────────────────
python train_pretrain.py \
    --model-size 0.3B --data-dir ./packed \
    --checkpoint-dir ./checkpoints --seq-len 2048 \
    --batch-size 32 --grad-accum 4 --jit

# ── Step 5: SFT ────────────────────────────────────────────────
python train_sft.py \
    --checkpoint-dir ./checkpoints --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints --lora-rank 64

# ── Step 6: GRPO (Reinforcement Learning) ──────────────────────
python train_grpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./grpo_packed --tokenizer ./tokenizer \
    --out-dir ./grpo_checkpoints --num-steps 500

# ── Step 7: Inference ──────────────────────────────────────────
python infer.py \
    --checkpoint ./grpo_checkpoints/latest.pt \
    --prompt "Hello, world!" --interactive
```

### 🧪 Smoke Tests (No Data Needed)

```bash
# Verify every component works end-to-end:
python -c "import train_pretrain; train_pretrain.smoke_test()"
python train_sft.py --smoke-test
python train_grpo.py --smoke-test
python train_dpo.py --smoke-test
python infer.py --smoke-test
```

### 🐳 Docker

A `Dockerfile` builds a CUDA image with the framework and all training
dependencies. Checkpoints and data are **not** baked in — mount them at
runtime:

```bash
# Build
docker build -t advanced-llm-framework .

# Smoke-test the container (runs the pretrain smoke test by default)
docker run --gpus all --rm advanced-llm-framework

# Pretrain with data mounted in
docker run --gpus all --shm-size 16g -it --rm \
  -v $PWD/checkpoints:/workspace/checkpoints \
  -v $PWD/packed:/workspace/packed \
  advanced-llm-framework \
  train_pretrain.py --model-size 0.3B --data-dir /workspace/packed \
    --checkpoint-dir /workspace/checkpoints --seq-len 2048 \
    --batch-size 32 --grad-accum 4

# SFT / GRPO / DPO work the same way — swap the script and flags.
```

The container runs as a non-root user (`trainer`, UID 1000); write
directories you mount must be writable by it (or use `-u 0` at your own
risk). For a different CUDA build of torch, override the base tag or install
torch first as shown in the `Dockerfile`.

### 🔄 CI

`.github/workflows/ci.yml` runs on every push/PR:

- **Syntax + lint** — every `.py` file compiles and has no bare `except:`
  (py 3.11 / 3.12 / 3.13)
- **Unit tests** — CPU `pytest` suite (model forward, atomic I/O, shutdown,
  structured logging)
- **Smoke tests** — the five `--smoke-test` entry points above, on CPU
- **Docker build** — image builds and runs the pretrain smoke test inside it

### 🚢 Release

`.github/workflows/release.yml` runs on `v*` tag push (and on
`workflow_dispatch` for manual reruns). Four jobs:

1. **build-and-push** — matrix over `[ui-router, ui, trainer]`,
   builds + pushes to `pratic2001/llmforge-*:<tag>` and `:latest`
2. **generate-env-bundle** — assembles `secrets/{router,ui,ui-lite,trainer}.env`
   from GH Secrets, uploads as the `env-bundle` artifact
3. **smoke-test** — `docker compose up -d` on the freshly-built images,
   waits for `:3000`, `:3001`, `:3002` to return 200 on `/dashboard`,
   then tears down
4. **notify-release** — generates `release-summary-<tag>.pdf` (image
   digests + "What's New" commit log + deployment steps) and emails it
   if SMTP secrets are set

See **[RELEASE.md](./RELEASE.md)** for the full pipeline reference,
required secrets, and PDF/email configuration.

---

## 📦 Module Overview

| Module | Path | Description |
|--------|------|-------------|
| **Model** | `model.py` | `ModelConfig` + `TransformerForCausalLM` — dense decoder-only transformer with GQA, RoPE (YaRN/NTK), SwiGLU, QK-Norm. **New in v2:** `MLAAttention` (DeepSeek-style low-rank KV compression), `MambaBlock` (SSM), `JambaModel` (hybrid SSM+Attn), `MixtureOfDepthDecoderLayer` (selective FFN routing), `ParallelDecoderLayer` (PaLM-style), `MTPHeads` (multi-token prediction), sliding-window attention |
| **Recipe** | `recipe.py` | `TrainingRecipe` — templates, tokens, think modes, formatting |
| **Tokenizer Trainer** | `tokenizer_train.py` | Byte-level BPE tokenizer from JSONL |
| **Pretrain (DDP)** | `train_pretrain.py` | Self-supervised next-token prediction, DDP |
| **Pretrain (DS)** | `train_pretrain_deepspeed.py` | Same + DeepSpeed ZeRO |
| **SFT (DDP)** | `train_sft.py` | Supervised fine-tuning + LoRA/DoRA |
| **SFT (DS)** | `train_sft_deepspeed.py` | Same + DeepSpeed |
| **GRPO (DDP)** | `train_grpo.py` | Group Relative Policy Optimization |
| **GRPO (DS)** | `train_grpo_deepspeed.py` | Same + DeepSpeed |
| **DPO (DDP)** | `train_dpo.py` | Direct Preference Optimization |
| **DPO (DS)** | `train_dpo_deepspeed.py` | Same + DeepSpeed |
| **Benchmark** | `benchmark_tune.py` | Hardware benchmark & hyperparameter recommendation for all stages |
| **Inference** | `infer.py` | Streaming, quantized, REPL, batched |
| **Ollama Judge** | `ollama_judge.py` | Remote Ollama judge for DPO pair generation |
| **Optimizer** | `optim/build_optimizer.py` | AdamW, FusedAdam, Muon |
| **Schedule** | `optim/lr_schedule.py` | Cosine, WSD schedules |
| **LoRA** | `peft/lora.py` | LoRA, DoRA, rsLoRA adapters |
| **Data Agent** | `webscrapped_dataset_curator_AI_MCP/agent/dataset_agent.py` | Async LLM-planned data curator |
| **Codegen Pipeline** | `webscrapped_dataset_curator_AI_MCP/agent/codegen_pipeline.py` | Simpler: Ollama writes extractor scripts |
| **Codegen Graph** | `webscrapped_dataset_curator_AI_MCP/agent/codegen_graph.py` | LangGraph reasoning codegen |
| **HF → Packed** | `webscrapped_dataset_curator_AI_MCP/agent/hf_to_packed.py` | Single HF dataset → packed .bin |
| **Quality** | `webscrapped_dataset_curator_AI_MCP/agent/quality.py` | Filters, dedup, shard writer |
| **Sources** | `webscrapped_dataset_curator_AI_MCP/agent/public_sources.py` | HF Hub / Kaggle connectors |
| **Topics** | `webscrapped_dataset_curator_AI_MCP/agent/topics.py` | Topic seeds per category |
| **MCP Server** | `webscrapped_dataset_curator_AI_MCP/web_scraper_mcp/server.py` | FastMCP server with search+extract tools |
| **Pretrain Packer** | `data/pack_pretrain.py` | Text JSONL → uint16 .bin memmap |
| **SFT Packer** | `data/pack_sft.py` | SFT JSONL → tokens + mask .bin |
| **GRPO Packer** | `data/pack_grpo.py` | GRPO JSONL → length-prefixed .bin + answers |
| **DPO Packer** | `data/pack_dpo.py` | Preference triples → .bin pairs |

---

## 🏗 Repository Structure

```
.
├── model.py                       # Dense transformer + MLA / Mamba / Jamba / MoD / MTP / parallel layers / SWA
├── recipe.py                      # TrainingRecipe (templates, tokens)
├── recipe.json                    # Sample recipe
├── tokenizer_train.py             # BBPE tokenizer trainer
├── benchmark_tune.py              # Hardware benchmark & hyperparameter recommendation
│
├── train_pretrain.py              # Pretrain — torch DDP
├── train_pretrain_deepspeed.py    # Pretrain — DeepSpeed
├── train_sft.py                   # SFT — DDP + LoRA/DoRA
├── train_sft_deepspeed.py         # SFT — DeepSpeed
├── train_grpo.py                  # GRPO RL — DDP
├── train_grpo_deepspeed.py        # GRPO RL — DeepSpeed
├── train_dpo.py                   # DPO — DDP
├── train_dpo_deepspeed.py         # DPO — DeepSpeed
├── infer.py                       # Inference (quant, streaming, REPL)
├── ollama_judge.py                # Remote Ollama judge for DPO
│
├── optim/
│   ├── build_optimizer.py         # AdamW, FusedAdam, Muon
│   └── lr_schedule.py             # Cosine, WSD schedules
│
├── peft/
│   └── lora.py                    # LoRA / DoRA / rsLoRA
│
├── data/
│   ├── pack_pretrain.py           # Pretrain JSONL → .bin
│   ├── pack_sft.py                # SFT JSONL → .bin + mask
│   ├── pack_grpo.py               # GRPO prompts → .bin + answers
│   ├── pack_dpo.py                # DPO preference → .bin
│   └── pack_dataset.py            # Generic tokenizer/packer
│
├── configs/                       # Training config files
├── hivemind/                       # Decentralized multi-node training (see hivemind/README.md)
│   ├── hivemind_utils.py           # Peer setup, DecentralizedOptimizer, checkpoint averaging
│   ├── train_pretrain_hivemind.py  # Decentralized pretraining
│   ├── train_sft_hivemind.py       # Decentralized SFT
│   ├── train_grpo_hivemind.py      # Decentralized GRPO
│   ├── train_dpo_hivemind.py       # Decentralized DPO
│   ├── run.sh                      # Convenience launcher
│   └── requirements-hivemind.txt   # Hivemind dependencies
├── tests/                         # Smoke tests (train_*.py --smoke-test)
├── requirements.txt               # Dependencies
├── MANUAL.md                      # Full user manual
├── README.md                      # This file
│
└── webscrapped_dataset_curator_AI_MCP/
    ├── agent/
    │   ├── dataset_agent.py       # Self-directed curation agent
    │   ├── codegen_pipeline.py    # Simpler: discover → codegen → run
    │   ├── codegen_graph.py       # LangGraph reasoning codegen
    │   ├── hf_to_packed.py        # Single HF dataset → packed .bin
    │   ├── public_sources.py      # HF Hub / Kaggle connectors
    │   ├── quality.py             # Filters, dedup, shard writer
    │   └── topics.py              # Topic seeds per category
    └── web_scraper_mcp/
        ├── server.py              # FastMCP server
        ├── extractors.py          # Multi-format extractors
        ├── crawl4ai_backend.py    # Headless browser backend
        └── net_utils.py           # Retry, proxy, user-agent pool
```

---

## 🏗 Architecture Variants

The framework's `model.py` goes beyond a plain dense decoder — every training script accepts flags to choose alternative architectures. All variants share the same auto-sizing, checkpoint format, and training pipeline.

### Supported Variants

| Variant | Flag | Description |
|---------|------|-------------|
| **Dense (default)** | `--arch dense --layer-type sequential` | Standard pre-norm decoder-only transformer with GQA, RoPE, SwiGLU |
| **Parallel** | `--arch dense --layer-type parallel` | PaLM-style: attention + MLP computed from the same normalised input (~15% faster) |
| **Jamba** | `--arch jamba --jamba-interval 4` | Hybrid SSM + Attention: Mamba blocks every N−1 layers, attention every Nth layer |
| **MoD** | `--mod-alpha 0.25` | Mixture-of-Depth: per-token router skips FFN for low-scoring tokens (saves compute) |
| **MLA** | `--use-mla` | Multi-head Latent Attention: low-rank KV joint compression (~4× smaller KV cache) |
| **MTP** | `--num-mtp-heads 3` | Multi-Token Prediction: predict K future tokens per position (discounted auxiliary loss) |
| **Sliding Window** | `--sliding-window-size 4096` | Alternating global + local sliding-window attention layers |

### Composing Variants

Variants compose naturally. For example, a **Jamba model with MLA, MTP, and sliding window**:

```bash
python train_pretrain.py --arch jamba --use-mla --num-mtp-heads 3 \
  --sliding-window-size 4096 --model-size 1B --data-dir ./packed
```

Or a **dense parallel model with Mixture-of-Depth routing**:

```bash
python train_pretrain.py --layer-type parallel --mod-alpha 0.25 \
  --model-size 300M --data-dir ./packed
```

### RoPE Scaling

All architectures support YaRN and NTK-aware RoPE scaling for extended contexts:

```bash
# YaRN: 8× context extension over pretrained length
python train_pretrain.py --rope-scaling '{"type": "yarn", "factor": 8.0}' ...

# NTK-aware: better high-frequency preservation
python train_pretrain.py --rope-scaling '{"type": "ntk", "factor": 8.0}' ...
```

### Architecture Decision Guide

| Goal | Recommended Config |
|------|-------------------|
| Maximum throughput | `--layer-type parallel --mod-alpha 0.25` |
| Long context, efficient KV | `--use-mla` (or `--use-mla --sliding-window-size 8192`) |
| Best quality-to-compute ratio | `--arch jamba --jamba-interval 4` |
| Faster convergence | `--num-mtp-heads 3` (auxiliary MTP loss) |
| Standard baseline | Default (dense sequential) |

All variant flags are also available in the **hivemind** decentralized training scripts — see [`hivemind/README.md`](hivemind/README.md).

---

## 🐳 Containerized Deployment

The whole stack — UI_RenderRouter, UI, UI_lite, and the GPU trainer — runs as
three Docker images on a single host and is exposed to your tailnet via a
Tailscale sidecar per service. Pushing a git tag publishes the images to
Docker Hub and produces an `env-bundle` artifact with the runtime env files.

### Architecture

```
Browser
   │
   │  https://<your-host>.ts.net/         (Tailscale Funnel via ts-ui-router)
   │  https://<your-host>.ts.net/heavy    (via ts-ui → ui container :3000)
   │  https://<your-host>.ts.net/lite     (via ts-ui → ui container :3001)
   ▼
┌────────────────┐    ┌──────────────────────┐    ┌────────────────────┐
│ ui-router      │    │ ui  (Heavy + Lite)   │    │ trainer  (GPU+SSH) │
│ :3002          │    │ :3000, :3001         │    │ :22                │
│                │    │ Postgres via host    │    │ PyTorch + framework│
│                │    │  .docker.internal    │    │ code, sshd as      │
│                │    │ trainer via          │    │  user `trainer`    │
│                │    │  Tailscale hostname  │    │                    │
└────────────────┘    └──────────────────────┘    └────────────────────┘
```

### One-time setup

1. **Generate a Tailscale reusable auth key** at
   <https://login.tailscale.com/admin/settings/keys>. Scope it to your
   tailnet and tag it `tag:container` (or similar). You'll inject this into
   `docker compose` at boot time.

2. **Generate an SSH keypair** the UI container will use to reach the trainer:
   ```bash
   ssh-keygen -t ed25519 -N '' -f ./secrets/ui-sshkey
   ```
   The public key gets baked into the trainer image; the private key gets
   shipped to the UI container via the env-bundle.

3. **Populate the GitHub Actions secrets** at
   <https://github.com/Pratic2001/Advanced-LLM-Framework-NON-MOE/settings/secrets/actions>:

   | Secret | Description |
   |---|---|
   | `DOCKERHUB_USERNAME` | Docker Hub login (e.g. `pratic2001`) |
   | `DOCKERHUB_TOKEN` | Docker Hub Personal Access Token |
   | `TS_AUTHKEY` | Tailscale reusable auth key from step 1 |
   | `PUBLIC_HOSTNAME` | Your tailnet hostname, e.g. `pratic-battleaxb450mkm2.tail5e5151.ts.net` |
   | `NEXTAUTH_SECRET` | 32+ char secret, `openssl rand -base64 32` |
   | `AUTH_SECRET` | Same value as `NEXTAUTH_SECRET` |
   | `SSH_KEY_ENCRYPTION_KEY` | 32 hex chars for AES-256-GCM |
   | `DATABASE_URL` | `postgresql://...` for the host's Postgres |
   | `SSH_PUBLIC_KEY` | Contents of `./secrets/ui-sshkey.pub` |
   | `SSH_PRIVATE_KEY` | Contents of `./secrets/ui-sshkey` |

### Releasing

```bash
# Tag a release — pushes images + generates env-bundle artifact
git tag v0.2.0 && git push --tags
```

Then in GitHub Actions:
1. Wait for the **Release** workflow to finish.
2. Download the **env-bundle** artifact.
3. Unpack it next to your clone:
   ```bash
   tar -xf env-bundle.tar.gz
   mv bundle/secrets .
   ```
4. (Optional) Download the **release-summary-`<tag>`** artifact — a one-page
   PDF with the digest list, what was deployed, what changed (commit log),
   and deployment instructions.

### Optional: email the release PDF

The Release workflow also emails the PDF to you if you set the SMTP
secrets listed in [Release Pipeline → Receiving the PDF by email](#-release-pipeline).
See **[RELEASE.md](./RELEASE.md)** for the full pipeline reference,
the env-bundle artifact structure, the cold-start math, and a write-up
of why the smoke-test probes `/dashboard` instead of `/api/auth/providers`.

### Running the stack

```bash
# Make sure the Tailscale auth key is in your shell
export TS_AUTHKEY=tskey-xxxxxxxxxxxxxxxxxxxxxxxx

docker compose pull
docker compose up -d

# Verify
docker compose ps                        # all 6 containers healthy
docker exec trainer nvidia-smi           # GPU visible
```

Open <https://your-host.ts.net/> in a browser on your tailnet and you should
land on the RenderRouter.

### Registering the trainer as a Node in the UI

After `docker compose up -d`:

1. Log in to the Heavy UI.
2. **Settings → Nodes → Register Node**.
3. Hostname: `trainer` (the Tailscale MagicDNS name of the trainer container).
4. Port: `22`.
5. Username: `trainer`.
6. Paste the contents of `./secrets/ui-sshkey` into the **Private Key** field.
7. Click **Test Connection** — you should see `NVIDIA GeForce RTX 3050` or
   whatever GPU is in the trainer container.

### Going back to local dev

The `Dockerfile.ui-router` and `Dockerfile.ui` only matter for the release
pipeline. Local dev against the dev servers (`:3210`, `:3211`, `:3212`)
still uses `npm run dev` in each subdirectory — see `UI_RenderRouter/.env.example`
and `UI/.env.example` for the local overrides.

### Fresh-PC Setup

Run `install.sh` on any new machine. It checks prerequisites, prompts for
the two values you can't auto-generate (Tailscale hostname + auth key),
generates SSH keys + secrets, pulls the images, and starts the stack.

```bash
# Clone the repo anywhere — install.sh doesn't need source code, only
# docker + the images it pulls from Docker Hub.
git clone https://github.com/Pratic2001/Advanced-LLM-Framework-NON-MOE.git
cd Advanced-LLM-Framework-NON-MOE

# Default: interactive. Skips prompts for things you already set via env.
./install.sh

# Non-interactive (CI, scripted deploys):
./install.sh --non-interactive \
  --public-hostname=pratic-battleaxb450mkm2.tail5e5151.ts.net \
  --ts-authkey=tskey-xxxxxxxxxxxxxxxxxxxxxxxx \
  --tag=v0.2.0
```

**What `install.sh` does, step by step:**

1. Verifies `docker`, `docker compose` plugin, and `openssl` are installed.
2. Prompts for `PUBLIC_HOSTNAME` (your tailnet hostname) and `TS_AUTHKEY`
   (Tailscale reusable auth key from <https://login.tailscale.com/admin/settings/keys>).
   All other values are auto-generated: NEXTAUTH_SECRET, AUTH_SECRET,
   SSH_KEY_ENCRYPTION_KEY, plus a fresh ed25519 SSH keypair in `secrets/`.
3. Writes `secrets/{router,ui,ui-lite,trainer}.env` from those values.
4. Runs `docker compose pull` and `docker compose up -d`.
5. Verifies the UI containers came up healthy and (if present) the
   trainer can see its GPU via `nvidia-smi`.

**Prerequisites a fresh box needs before `install.sh` will work:**

```bash
# Docker + compose plugin (Ubuntu/Debian)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"    # log out + back in for this to take effect

# Tailscale (optional — only needed if you want to skip the bundled sidecars
# and let the host's existing tailscaled do the routing instead)
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# NVIDIA Container Toolkit (GPU host only)
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
distribution=$(. /etc/os-release;echo "$ID$VERSION_ID")
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -sSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Updating an existing install:**

```bash
TAG=v0.3.0 ./install.sh       # pulls new tag, restarts in place
# Secrets are preserved — install.sh reuses ./secrets/ui-sshkey if present
# and only prompts for missing values.
```

---

### Multi-PC Deployment

The same Docker images cover three topologies. Pick the one that fits your
hardware.

#### Topology A — Split the GPU out (1 UI PC + 1 GPU PC)

Common when the GPU box is in a different room / closet / cloud VM and
your everyday workstation only needs to drive the UI.

```
┌──────────────────────────────────────┐    ┌──────────────────────────────────────┐
│ UI PC  (no GPU, no NVIDIA toolkit)   │    │ GPU PC  (RTX 3050/4050/4090 etc.)   │
│                                      │    │                                      │
│  ui-router   (:3002)                 │    │  trainer   (:22, GPU)                │
│  ui          (:3000 + :3001)         │    │                                      │
│  ts-ui-router, ts-ui (sidecars)      │    │  ts-trainer (sidecar)                │
│                                      │    │                                      │
│  Postgres on host (or shared LAN)    │    │                                      │
└──────────────────┬───────────────────┘    └──────────────────┬───────────────────┘
                   │                                            │
                   └────── Tailscale tailnet ───────────────────┘
                            (MagicDNS name 'trainer'
                             reaches the GPU host)
```

**On the GPU PC:**

```bash
git clone https://github.com/Pratic2001/Advanced-LLM-Framework-NON-MOE.git
cd Advanced-LLM-Framework-NON-MOE
./install.sh --non-interactive \
  --public-hostname=pratic-battleaxb450mkm2.tail5e5151.ts.net \
  --ts-authkey=tskey-xxx
# install.sh generates secrets/ui-sshkey — copy the PUBLIC half to the UI PC.
scp secrets/ui-sshkey.pub ui-pc:~/Advanced-LLM-Framework-NON-MOE/secrets/ui-sshkey.pub
```

**On the UI PC:**

```bash
git clone https://github.com/Pratic2001/Advanced-LLM-Framework-NON-MOE.git
cd Advanced-LLM-Framework-NON-MOE

# Drop the trainer service + ts-trainer sidecar — the GPU lives elsewhere.
docker compose -f docker-compose.yml -f docker-compose.ui-only.yml up -d

# Register the remote trainer as a Node:
#   Settings → Nodes → Register Node:
#     hostname = trainer            # the GPU PC's Tailscale MagicDNS name
#     port     = 22
#     username = trainer
#     key      = contents of secrets/ui-sshkey (private half, on UI PC)
```

The UI container reaches the trainer over SSH on the Tailscale hostname
`trainer`. No special firewall rules — Tailscale's encrypted tunnel handles
NAT traversal.

#### Topology B — Mirror the UI to multiple PCs (centralized trainer)

Useful when you want low-latency access from a laptop at home AND a desktop
at the office, but training still happens on one GPU machine.

```
┌────────────┐    ┌────────────┐    ┌─────────────────┐
│ Home PC    │    │ Office PC  │    │ GPU Server      │
│ ui + ts-ui │    │ ui + ts-ui │    │ trainer + ts-tr │
│ Postgres ↘ │    │ Postgres ↘ │    │        ↑        │
│            └────┴────────────┴────┘                 │
│            shared Postgres (one host, or RDS etc.) │
└────────────┴────────────────────────────────────────┘
```

Each UI PC runs `./install.sh` separately but uses the same
`PUBLIC_HOSTNAME`. The trick is the Tailscale sidecars on each UI host
publish the same path mappings (`/heavy`, `/lite`) under the same MagicDNS
hostname — only one of them can win the funnel race at a time, so use
**Tailscale Serve** (in-process) on the "primary" host and **Tailscale
client routing** (the default `--accept-routes` flag in our compose) on
the others.

In practice: pick one host as the funnel entry point, and on the others
drop the `ts-ui-router` and `ts-ui` sidecars:

```yaml
# docker-compose.ui-internal.yml — Topology B secondary host
# Inherits from docker-compose.yml and -ui-only.yml, but also removes
# the ts-* sidecars because this host's UI is reached via the primary.
include:
  - docker-compose.yml
  - docker-compose.ui-only.yml
```

Override the `ts-ui-router` and `ts-ui` services to no-op, or just edit
`docker-compose.yml` to comment them out before running on secondary hosts.

Postgres must be reachable from every UI host. The simplest setup is to
point `DATABASE_URL` in each `secrets/ui.env` at the same central Postgres
(e.g. `postgresql://...@postgres-host.tailnet:5432/llm_training_ui`).

#### Topology C — Distributed training across multiple GPUs

For multi-GPU / multi-machine training, use the framework's
[hivemind decentralized training scripts](./hivemind/README.md). The
Docker images don't need to know about each other — each trainer container
runs an independent peer and they discover each other via the
`--initial-peers` flag.

```bash
# On GPU PC #1 (bootstrap peer):
docker exec -it trainer bash
python hivemind/train_pretrain_hivemind.py \
  --hivemind --initial-peers "" --port 5678 \
  --model-size 300M --data-dir /workspace/packed \
  --checkpoint-dir /workspace/checkpoints/bootstrap

# On GPU PC #2 (joins the swarm):
docker exec -it trainer bash
python hivemind/train_pretrain_hivemind.py \
  --hivemind --initial-peers "gpu-pc-1.tailnet:5678" --port 5678 \
  --model-size 300M --data-dir /workspace/packed \
  --checkpoint-dir /workspace/checkpoints/worker1
```

Each peer trains locally and asynchronously averages parameters. See
[`hivemind/README.md`](./hivemind/README.md) for the full set of options
(`--target-group-size`, `--average-checkpoints`, etc.) and the cross-swarm
averaging utilities documented in
[hivemind-improvements](../hivemind-improvements) memory.

The UI container doesn't drive hivemind jobs — those are launched
directly inside each trainer. If you want the UI to *coordinate* distributed
training, use the `nodeIds[]` field on `POST /api/jobs` to fan out a job
across multiple Nodes; the existing ssh-manager does the round-trip.

---

## 🚢 Release Pipeline

Pushing a `v*` tag runs `.github/workflows/release.yml`, which does
**four** things in order:

1. **`build-and-push`** — matrix build of the three container images
   (`ui-router`, `ui`, `trainer`), each pushed to
   `pratic2001/llmforge-<name>:<tag>` *and* `:latest` on Docker Hub.
2. **`generate-env-bundle`** — assembles
   `secrets/{router,ui,ui-lite,trainer}.env` from GH Secrets and uploads
   it as the `env-bundle` workflow artifact.
3. **`smoke-test`** — `docker compose up -d` on the freshly-built
   images, waits up to 3 minutes for Next to be serving real HTML on
   `/dashboard` for `:3000`, `:3001`, and `:3002`, then tears down.
4. **`notify-release`** — generates `release-summary-<tag>.pdf`
   (image digests pulled live from Docker Hub + "What's New" commit
   log + deployment instructions) and uploads it as an artifact.
   If you set SMTP secrets (see below), it also **emails** the PDF
   to your address.

### Cutting a release

```bash
git tag v0.2.0
git push --tags
# → Actions tab, wait for "Release" to finish (≈ 6 minutes)
# → Download env-bundle and release-summary-v0.2.0 artifacts
```

### Receiving the PDF by email

The workflow emails the PDF to you if you set these GitHub Secrets
(`Settings → Secrets and variables → Actions`):

| Secret | Example |
|---|---|
| `SMTP_HOST` | `smtp.gmail.com` |
| `SMTP_PORT` | `465` (default) |
| `SMTP_USERNAME` | `cpratic8@gmail.com` |
| `SMTP_PASSWORD` | Gmail **app password** from <https://myaccount.google.com/apppasswords> (requires 2FA; your normal Gmail password will not work) |
| `NOTIFY_EMAIL` | Recipient address (defaults to `SMTP_USERNAME` if unset) |

Without these, the PDF is still uploaded as an artifact — that's the
canonical source of truth. Email is a convenience.

**Full pipeline reference:** see **[RELEASE.md](./RELEASE.md)** for the
required secrets, the env-bundle artifact structure, the cold-start
budget for the smoke-test, and a write-up of why the smoke-test probes
`/dashboard` instead of `/api/auth/providers`.

---

## 📖 Full Documentation

> **👉 [MANUAL.md](./MANUAL.md)** — Complete user manual with detailed instructions for every component.

The manual covers:
- Recipe system (thinking modes, chat templates)
- Training a tokenizer
- Data pipeline (dataset agent + codegen pipeline)
- Packing data for training (pretrain, SFT, GRPO, DPO)
- Training with DDP and DeepSpeed
- Architecture variants (MLA, Mamba, Jamba, MoD, MTP, parallel, sliding window)
- GRPO reward design and hyperparameters
- DPO loss configuration and end-to-end pipeline
- Inference (REPL, batched, quantized)
- Model architecture and auto-sizing
- Troubleshooting & FAQ

```bash
# Quick reference — every script has --help:
python train_pretrain.py --help
python train_sft.py --help
python train_grpo.py --help
python train_dpo.py --help
python benchmark_tune.py --help
python infer.py --help
python webscrapped_dataset_curator_AI_MCP/agent/codegen_pipeline.py --help
python webscrapped_dataset_curator_AI_MCP/agent/hf_to_packed.py --help
```

---

## 📄 License & Citation

This project is released under the MIT License.

```bibtex
@misc{dense-llm-framework,
  author = {Pratic Chakraborty},
  title  = {Advanced LLM Framework — Dense Non-MoE Training Pipeline},
  year   = {2026},
  url    = {https://github.com/Pratic2001/Advanced-LLM-Framework-NON-MOE}
}
```

---

<p align="center">
  <sub>Built with ❤️ and PyTorch</sub>
  <br>
  <sub>Pretrain → SFT → GRPO → 🚀</sub>
</p>
