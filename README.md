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
python train_pretrain.py --smoke-test
python train_sft.py --smoke-test
python train_grpo.py --smoke-test
python train_dpo.py --smoke-test
python infer.py --smoke-test
```

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
