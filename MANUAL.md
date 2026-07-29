# 📖 Advanced LLM Framework — Complete User Manual

> **Command examples for every feature** in the dense (non-MoE) LLM training pipeline.
>
> Tokenizer → Data Curation → Packing → Pretrain → SFT → GRPO/DPO → Inference

---

## Table of Contents

1. [Quick Reference: Smoke Tests](#1-quick-reference-smoke-tests)
2. [Tokenizer Training](#2-tokenizer-training)
3. [Data Curation](#3-data-curation)
4. [Data Packing](#4-data-packing)
5. [Pretraining](#5-pretraining)
6. [Supervised Fine-Tuning (SFT)](#6-supervised-fine-tuning-sft)
7. [GRPO (Reinforcement Learning)](#7-grpo-reinforcement-learning)
8. [DPO (Preference Optimization)](#8-dpo-preference-optimization)
9. [Inference](#9-inference)
10. [Architecture Variants](#10-architecture-variants)
11. [Recipe System](#11-recipe-system)
12. [Optimizer & LR Schedules](#12-optimizer--lr-schedules)
13. [PEFT: LoRA / DoRA / rsLoRA](#13-peft-lora--dora--rslora)
14. [RoPE Scaling (YaRN / NTK)](#14-rope-scaling-yarn--ntk)
15. [Ollama DPO Judge](#15-ollama-dpo-judge)
16. [Distributed Multi-GPU](#16-distributed-multi-gpu)
17. [Decentralized Hivemind Training](#17-decentralized-hivemind-training)
18. [Checkpoint Save / Resume](#18-checkpoint-save--resume)
19. [Model Architecture Reference](#19-model-architecture-reference)
20. [End-to-End Pipeline: Train on an RTX 4090](#20-end-to-end-pipeline-train-a-capable-350m-model-on-an-rtx-4090)
21. [Appendix A: Full CLI Reference](#appendix-a-full-cli-reference)

---

## 1. Quick Reference: Smoke Tests

Every script has a `--smoke-test` mode that runs a tiny end-to-end test with synthetic data — no real checkpoints or data needed:

```bash
# Pretrain smoke test
python train_pretrain.py --smoke-test

# Pretrain + DeepSpeed smoke test
python train_pretrain_deepspeed.py --smoke-test

# SFT smoke test (creates a small model + synthetic data)
python train_sft.py --smoke-test

# SFT + DeepSpeed smoke test
python train_sft_deepspeed.py --smoke-test

# GRPO smoke test
python train_grpo.py --smoke-test

# GRPO + DeepSpeed smoke test
python train_grpo_deepspeed.py --smoke-test

# DPO smoke test
python train_dpo.py --smoke-test

# DPO + DeepSpeed smoke test
python train_dpo_deepspeed.py --smoke-test

# Inference smoke test (no checkpoint needed)
python infer.py --smoke-test
```

---

## 2. Tokenizer Training

Train a Byte-level BPE tokenizer from JSONL text files.

```bash
# Basic: train a 65K vocabulary tokenizer
python tokenizer_train.py \
    --data-dir ./raw_data \
    --output-dir ./tokenizer

# With explicit vocabulary size and minimum frequency
python tokenizer_train.py \
    --data-dir ./raw_data \
    --output-dir ./tokenizer \
    --vocab-size 131072 \
    --min-frequency 3

# With a recipe (defines special tokens like <think>, </think>, etc.)
python tokenizer_train.py \
    --data-dir ./raw_data \
    --output-dir ./tokenizer \
    --recipe ./recipe.json

# With only mode (no recipe.json — uses default special tokens for that mode)
python tokenizer_train.py \
    --data-dir ./raw_data \
    --output-dir ./tokenizer \
    --mode reasoning

# Supported modes: reasoning, non_reasoning, hybrid
python tokenizer_train.py \
    --data-dir ./raw_data \
    --output-dir ./tokenizer \
    --mode hybrid

python tokenizer_train.py \
    --data-dir ./raw_data \
    --output-dir ./tokenizer \
    --mode non_reasoning
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `./data` | Directory of `.jsonl` files with `text` or `prompt` fields |
| `--output-dir` | `./tokenizer` | Where to save `tokenizer.json` |
| `--vocab-size` | `65536` | Target vocabulary size |
| `--min-frequency` | `2` | Minimum token frequency to include |
| `--recipe` | `None` | Path to `recipe.json` (defines special tokens and mode) |
| `--mode` | `None` | Training mode when `--recipe` is not given: `reasoning`, `non_reasoning`, `hybrid` |

---

## 3. Data Curation

Three curation pipelines, each progressively more automated:

All pipelines support a **two-tier data quality system**:

- **Tier 1 — Legacy basic filters**: `passes_prose_quality_filter` checks minimum length, junk markers, and exact URL/email patterns.
- **Tier 2 — Extended quality filters** (enabled by default): Adds zlib compression ratio detection (boilerplate/logs are highly compressible), duplicate/adjacent-line repetition detection (templates/nav bars), vocabulary diversity checks (spam/keyword-stuffing), short-line ratio (cookie banners/navigation), flagged n-gram patterns (copyright/ads/cookie-consent), and optional fasttext-based language detection. Each filter can be tuned independently via CLI flags.

The extended filters are implemented in `webscrapped_dataset_curator_AI_MCP/agent/quality.py` and are applied either inline (dataset_agent) or as a programmatic post-filter pass (codegen_pipeline, hf_to_packed). Language detection is optional — all extended filters degrade gracefully when fasttext is not installed.

### 3A. Codegen Pipeline (simpler, recommended)

Discovers public datasets, has Ollama write extraction scripts, runs them, and packs the output.

```bash
# Public datasets only (HF + Kaggle)
python webscrapped_dataset_curator_AI_MCP/agent/codegen_pipeline.py \
    --target-size 2GB \
    --out-dir ./data \
    --mode pretrain \
    --public-only

# With web crawling (no --public-only → live crawl + public datasets)
python webscrapped_dataset_curator_AI_MCP/agent/codegen_pipeline.py \
    --target-size 500MB \
    --out-dir ./data \
    --categories web,knowledge,code,math \
    --mode pretrain \
    --min-doc-chars 500

# With LangGraph reasoning codegen
python webscrapped_dataset_curator_AI_MCP/agent/codegen_pipeline.py \
    --target-size 1GB \
    --out-dir ./data \
    --public-only \
    --reasoning-model deepseek-r1:7b \
    --mode pretrain

# Custom category mix
python webscrapped_dataset_curator_AI_MCP/agent/codegen_pipeline.py \
    --target-size 1GB \
    --out-dir ./data \
    --public-only \
    --mode sft \
    --mix "web=0.3,knowledge=0.3,code=0.2,math=0.2"

# Multi-language dataset discovery
python webscrapped_dataset_curator_AI_MCP/agent/codegen_pipeline.py \
    --target-size 500MB \
    --out-dir ./data \
    --public-only \
    --language "en,zh,de,fr"

# With extended quality post-filter (stricter filtering)
python webscrapped_dataset_curator_AI_MCP/agent/codegen_pipeline.py \
    --target-size 1GB \
    --out-dir ./data \
    --public-only \
    --mode pretrain \
    --max-compression-ratio 0.30 \
    --min-vocab-diversity 0.20 \
    --max-flagged-ngram-ratio 0.05 \
    --target-langs en

# Disable extended quality post-filter (use only LLM-generated script filters)
python webscrapped_dataset_curator_AI_MCP/agent/codegen_pipeline.py \
    --target-size 500MB \
    --out-dir ./data \
    --no-extended-quality
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--target-size` | **(required)** | Target dataset size, e.g. `500MB`, `2GB` |
| `--out-dir` | `./data` | Output directory for JSONL shards |
| `--categories` | `web,knowledge,reasoning,code,math,science` | Comma-separated category list |
| `--mode` | `pretrain` | `pretrain`, `sft`, or `grpo` |
| `--min-doc-chars` | `500` | Minimum document length in characters |
| `--public-only` | `False` | Use only public HF/Kaggle datasets (skip live crawl) |
| `--mix` | `None` | Budget mix, e.g. `web=0.5,math=0.5` |
| `--discover-limit` | `5` | HF Hub search results per keyword |
| `--max-candidates-to-try` | `3` | Datasets that must contribute per category |
| `--max-total-considered` | `40` | Safety ceiling per category |
| `--reasoning-model` | `None` | LangGraph reasoning model (e.g. `deepseek-r1:7b`) |
| `--language` | `en` | Comma-separated language codes for discovery |
| `--no-extended-quality` | `False` | Disable programmatic extended-quality post-filter pass |
| `--max-compression-ratio` | `0.35` | Max zlib compression ratio for post-filter |
| `--max-line-repetition` | `0.15` | Max fraction of duplicate lines |
| `--max-adjacent-repetition` | `0.15` | Max fraction of adjacent near-identical lines |
| `--min-vocab-diversity` | `0.15` | Min unique/total word ratio |
| `--max-short-line-ratio` | `0.50` | Max fraction of short/navigation lines |
| `--max-flagged-ngram-ratio` | `0.10` | Max fraction of lines with flagged patterns |
| `--target-langs` | `None` | Comma-separated target languages, e.g. `en,de` (requires fasttext) |

### 3B. Single Dataset: hf_to_packed (fastest for one dataset)

Takes one HuggingFace dataset and goes end-to-end: sample → codegen → pack.

```bash
# Raw text for pretraining
python webscrapped_dataset_curator_AI_MCP/agent/hf_to_packed.py \
    --dataset c4 --config en \
    --mode pretrain \
    --tokenizer ./tokenizer \
    --out-dir ./packed \
    --seq-length 2048

# Instruction data for SFT
python webscrapped_dataset_curator_AI_MCP/agent/hf_to_packed.py \
    --dataset databricks/databricks-dolly-15k \
    --mode sft \
    --tokenizer ./tokenizer \
    --out-dir ./sft_packed

# Math problems for GRPO
python webscrapped_dataset_curator_AI_MCP/agent/hf_to_packed.py \
    --dataset gsm8k --config main \
    --mode grpo \
    --tokenizer ./tokenizer \
    --out-dir ./grpo_packed \
    --target-size 200MB

# Keep generated scripts for debugging
python webscrapped_dataset_curator_AI_MCP/agent/hf_to_packed.py \
    --dataset c4 --config en \
    --mode pretrain \
    --tokenizer ./tokenizer \
    --out-dir ./packed \
    --keep-scripts

# Extended quality post-filter (stricter filtering, filter out boilerplate)
python webscrapped_dataset_curator_AI_MCP/agent/hf_to_packed.py \
    --dataset c4 --config en \
    --mode pretrain \
    --tokenizer ./tokenizer \
    --out-dir ./packed \
    --max-compression-ratio 0.30 \
    --min-vocab-diversity 0.20 \
    --max-flagged-ngram-ratio 0.05 \
    --target-langs en

# Disable extended quality post-filter
python webscrapped_dataset_curator_AI_MCP/agent/hf_to_packed.py \
    --dataset gsm8k --config main \
    --mode grpo \
    --tokenizer ./tokenizer \
    --out-dir ./grpo_packed \
    --no-extended-quality
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | **(required)** | HuggingFace dataset ID (e.g. `c4`, `gsm8k`) |
| `--config` | `None` | Dataset config/subset (e.g. `en` for c4) |
| `--mode` | **(required)** | `pretrain`, `sft`, or `grpo` |
| `--tokenizer` | `./tokenizer` | Path to tokenizer directory |
| `--out-dir` | **(required)** | Output directory for packed memmap files |
| `--seq-length` | `None` | Truncate to this many tokens |
| `--target-size` | `None (stream all)` | Stop after this much data, e.g. `500MB` |
| `--language` | `en` | Language code for metadata |
| `--category` | `None (uses mode)` | Category label for output records |
| `--val-fraction` | `0.005` | Fraction of tokens for validation |
| `--min-doc-chars` | `500` | Minimum character count per record |
| `--reasoning-model` | `None` | LangGraph reasoning model |
| `--keep-scripts` | `False` | Keep generated scripts for debugging |
| `--no-extended-quality` | `False` | Disable programmatic extended-quality post-filter |
| `--max-compression-ratio` | `0.35` | Max zlib compression ratio for post-filter |
| `--max-line-repetition` | `0.15` | Max fraction of duplicate lines |
| `--max-adjacent-repetition` | `0.15` | Max fraction of adjacent near-identical lines |
| `--min-vocab-diversity` | `0.15` | Min unique/total word ratio |
| `--max-short-line-ratio` | `0.50` | Max fraction of short/navigation lines |
| `--max-flagged-ngram-ratio` | `0.10` | Max fraction of lines with flagged patterns |
| `--target-langs` | `None` | Comma-separated target languages (requires fasttext) |

### 3C. Full Dataset Agent (LLM-planned, most automated)

Self-directed agent that plans and executes data collection using MCP web scraping.

```bash
# Basic: web crawl + LLM quality judge
python webscrapped_dataset_curator_AI_MCP/agent/dataset_agent.py \
    --target-size 1GB \
    --out-dir ./data

# Skip LLM quality judge for speed
python webscrapped_dataset_curator_AI_MCP/agent/dataset_agent.py \
    --target-size 500MB \
    --out-dir ./data \
    --no-llm-judge

# Public datasets only (huggingface + kaggle)
python webscrapped_dataset_curator_AI_MCP/agent/dataset_agent.py \
    --target-size 2GB \
    --out-dir ./data \
    --public-only

# With specific public sources
python webscrapped_dataset_curator_AI_MCP/agent/dataset_agent.py \
    --target-size 1GB \
    --out-dir ./data \
    --public-sources "huggingface,kaggle" \
    --mode sft

# Custom category budget mix
python webscrapped_dataset_curator_AI_MCP/agent/dataset_agent.py \
    --target-size 2GB \
    --out-dir ./data \
    --mix "web=0.3,knowledge=0.4,code=0.3"

# Log to file
python webscrapped_dataset_curator_AI_MCP/agent/dataset_agent.py \
    --target-size 1GB \
    --out-dir ./data \
    --log-file ./curation.log

# Pre-specified HF datasets per category (semicolon-separated cat=dataset pairs)
python webscrapped_dataset_curator_AI_MCP/agent/dataset_agent.py \
    --target-size 500MB \
    --out-dir ./data \
    --hf-datasets "code=bigcode/the-stack-dedup;knowledge=wiki_text"

# With extended quality inline filtering (stricter quality gates)
python webscrapped_dataset_curator_AI_MCP/agent/dataset_agent.py \
    --target-size 1GB \
    --out-dir ./data \
    --mode pretrain \
    --max-compression-ratio 0.30 \
    --min-vocab-diversity 0.20 \
    --max-flagged-ngram-ratio 0.05 \
    --target-langs en

# Disable extended quality filters (use only legacy basic filters)
python webscrapped_dataset_curator_AI_MCP/agent/dataset_agent.py \
    --target-size 500MB \
    --out-dir ./data \
    --no-extended-quality
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--target-size` | **(required)** | Target dataset size, e.g. `500MB`, `2GB` |
| `--out-dir` | `./data` | Output directory for JSONL shards |
| `--categories` | `web,knowledge,reasoning,code,math,science` | Comma-separated category list |
| `--mode` | `pretrain` | `pretrain`, `sft`, or `grpo` |
| `--concurrency` | `5` | Max concurrent URL extractions per category |
| `--min-doc-chars` | `500` | Minimum document length in characters |
| `--no-llm-judge` | `False` | Skip Ollama quality judge pass |
| `--mix` | `None` | Budget mix, e.g. `web=0.5,math=0.5` |
| `--public-sources` | `""` | Comma-separated: `huggingface,kaggle` |
| `--public-only` | `False` | Skip live web scraping entirely |
| `--hf-datasets` | `""` | Semicolon-separated `cat=dataset` pairs |
| `--kaggle-datasets` | `""` | Same format as `--hf-datasets` |
| `--category-concurrency` | `2` | Max categories running simultaneously |
| `--log-file` | `None` | Write log to file in addition to stdout |
| `--no-extended-quality` | `False` | Disable extended quality inline filters |
| `--max-compression-ratio` | `0.35` | Max zlib compression ratio |
| `--max-line-repetition` | `0.15` | Max fraction of duplicate lines |
| `--max-adjacent-repetition` | `0.15` | Max fraction of adjacent near-identical lines |
| `--min-vocab-diversity` | `0.15` | Min unique/total word ratio |
| `--max-short-line-ratio` | `0.50` | Max fraction of short/navigation lines |
| `--max-flagged-ngram-ratio` | `0.10` | Max fraction of lines with flagged patterns |
| `--target-langs` | `None` | Comma-separated target languages (requires fasttext) |

> **Note:** The agent requires a running Ollama server (`ollama serve`) with a model like `llama3.1` pulled for codegen and quality judging.

---

## 4. Data Packing

Raw JSONL shards must be packed into memmap `.bin` files before training.

### 4A. Pretrain Packing

```bash
# Basic
python data/pack_pretrain.py \
    --data-dir ./data \
    --tokenizer ./tokenizer \
    --cache-dir ./packed

# With validation split
python data/pack_pretrain.py \
    --data-dir ./data \
    --tokenizer ./tokenizer \
    --cache-dir ./packed \
    --val-fraction 0.01

# Multi-worker parallel packing (4 workers)
python data/pack_pretrain.py --worker 0 --num-workers 4 --data-dir ./data \
    --tokenizer ./tokenizer --cache-dir ./packed &
python data/pack_pretrain.py --worker 1 --num-workers 4 --data-dir ./data \
    --tokenizer ./tokenizer --cache-dir ./packed &
python data/pack_pretrain.py --worker 2 --num-workers 4 --data-dir ./data \
    --tokenizer ./tokenizer --cache-dir ./packed &
python data/pack_pretrain.py --worker 3 --num-workers 4 --data-dir ./data \
    --tokenizer ./tokenizer --cache-dir ./packed

# Short-context training (e.g., 512 tokens per window)
python data/pack_pretrain.py \
    --data-dir ./data \
    --tokenizer ./tokenizer \
    --cache-dir ./packed \
    --seq-length 512

# Longer max tokens per document before EOS truncation
python data/pack_pretrain.py \
    --data-dir ./data \
    --tokenizer ./tokenizer \
    --cache-dir ./packed \
    --max-seq-len-pretrain 8192
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | **(required)** | Directory tree of `.jsonl` with `text` field |
| `--tokenizer` | **(required)** | Path to tokenizer directory or `tokenizer.json` |
| `--cache-dir` | `./packed` | Output directory |
| `--val-fraction` | `0.0` | Fraction for validation (0 = no split) |
| `--worker` | `0` | Worker index for multi-process |
| `--num-workers` | `1` | Total workers |
| `--max-seq-len-pretrain` | `4096` | Max tokens per document before EOS truncation |
| `--seq-length` | `None` | Shorthand for `--max-seq-len-pretrain` |

### 4B. SFT Packing

Packs `{prompt, thinking, answer}` JSONL into tokens + loss-mask memmap with recipe-aware formatting.

```bash
# Basic
python data/pack_sft.py \
    --data-dir ./sft_data \
    --tokenizer ./tokenizer \
    --cache-dir ./sft_packed

# With recipe/mode (adds <think> tags for reasoning mode)
python data/pack_sft.py \
    --data-dir ./sft_data \
    --tokenizer ./tokenizer \
    --cache-dir ./sft_packed \
    --mode reasoning

python data/pack_sft.py \
    --data-dir ./sft_data \
    --tokenizer ./tokenizer \
    --cache-dir ./sft_packed \
    --mode non_reasoning

# Multi-worker
python data/pack_sft.py \
    --data-dir ./sft_data \
    --tokenizer ./tokenizer \
    --cache-dir ./sft_packed \
    --worker 0 --num-workers 4

# Set max tokens per example
python data/pack_sft.py \
    --data-dir ./sft_data \
    --tokenizer ./tokenizer \
    --cache-dir ./sft_packed \
    --max-len-per-example 1024
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | **(required)** | Directory tree with `prompt`, `thinking`, `answer` JSONL |
| `--tokenizer` | **(required)** | Path to tokenizer |
| `--cache-dir` | `./packed` | Output directory |
| `--val-fraction` | `0.01` | Validation fraction |
| `--worker` / `--num-workers` | `0` / `1` | Multi-process packing |
| `--max-len-per-example` | `2048` | Max tokens per example before truncation |
| `--seq-length` | `None` | Shorthand for `--max-len-per-example` |
| `--recipe` / `--mode` | `None` | Recipe or mode |

### 4C. GRPO Packing

Packs `{prompt, answer}` JSONL into memmap + answer strings for reward computation.

```bash
# Basic
python data/pack_grpo.py \
    --data-dir ./grpo_data \
    --tokenizer ./tokenizer \
    --cache-dir ./grpo_packed

# With reasoning mode
python data/pack_grpo.py \
    --data-dir ./grpo_data \
    --tokenizer ./tokenizer \
    --cache-dir ./grpo_packed \
    --mode reasoning

# Multi-worker
python data/pack_grpo.py \
    --data-dir ./grpo_data \
    --tokenizer ./tokenizer \
    --cache-dir ./grpo_packed \
    --worker 0 --num-workers 4

# Validation split
python data/pack_grpo.py \
    --data-dir ./grpo_data \
    --tokenizer ./tokenizer \
    --cache-dir ./grpo_packed \
    --val-fraction 0.005
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | **(required)** | Directory tree with `prompt`, `answer` JSONL |
| `--tokenizer` | **(required)** | Path to tokenizer |
| `--cache-dir` | `./packed` | Output directory |
| `--val-fraction` | `0.0` | Validation fraction |
| `--worker` / `--num-workers` | `0` / `1` | Multi-process packing |
| `--max-len-per-example` | `2048` | Max tokens per example before truncation |

### 4D. DPO Packing

Packs `{prompt, chosen, rejected}` preference triples into three memmap files.

```bash
# Basic
python data/pack_dpo.py \
    --data-dir ./dpo_data \
    --tokenizer ./tokenizer \
    --cache-dir ./dpo_packed

# With reasoning mode (adds <think> tags)
python data/pack_dpo.py \
    --data-dir ./dpo_data \
    --tokenizer ./tokenizer \
    --cache-dir ./dpo_packed \
    --mode reasoning

# Multi-worker
python data/pack_dpo.py \
    --data-dir ./dpo_data \
    --tokenizer ./tokenizer \
    --cache-dir ./dpo_packed \
    --worker 0 --num-workers 4

# Validation split
python data/pack_dpo.py \
    --data-dir ./dpo_data \
    --tokenizer ./tokenizer \
    --cache-dir ./dpo_packed \
    --val-fraction 0.01
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | **(required)** | Directory tree with `prompt`, `chosen`, `rejected` JSONL |
| `--tokenizer` | **(required)** | Path to tokenizer |
| `--cache-dir` | `./dpo_packed` | Output directory |
| `--val-fraction` | `0.0` | Validation fraction |
| `--worker` / `--num-workers` | `0` / `1` | Multi-process packing |
| `--recipe` / `--mode` | `None` | Recipe or mode |

---

## 5. Pretraining

### 5A. DDP (Single/Multi-GPU)

```bash
# Single GPU, 300M model (fits RTX 4090)
python train_pretrain.py \
    --model-size 0.3B \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints \
    --seq-len 2048 \
    --batch-size 32 \
    --grad-accum 4 \
    --jit

# Multi-GPU (4 GPUs via torchrun)
torchrun --nproc_per_node=4 train_pretrain.py \
    --model-size 1.7B \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints \
    --batch-size 16 \
    --grad-accum 4 \
    --jit

# Large model with gradient checkpointing (saves ~35% VRAM)
torchrun --nproc_per_node=8 train_pretrain.py \
    --model-size 13B \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints \
    --batch-size 4 \
    --grad-accum 8 \
    --gradient-checkpointing \
    --jit

# WSD schedule with Muon optimizer (2-3× faster convergence)
python train_pretrain.py \
    --model-size 0.6B \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints \
    --batch-size 32 \
    --grad-accum 4 \
    --schedule wsd \
    --stable-ratio 0.8 \
    --optimizer muon \
    --jit

# Override architecture manually (no auto-size)
python train_pretrain.py \
    --hidden-size 2048 \
    --num-layers 24 \
    --num-heads 16 \
    --num-kv-heads 4 \
    --intermediate-size 8192 \
    --head-dim 128 \
    --max-seq-len 8192 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints

# With W&B logging
python train_pretrain.py \
    --model-size 0.3B \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints \
    --wandb-project my-llm-pretrain \
    --wandb-run-name run-001

# Longer context (extends max_position_embeddings)
python train_pretrain.py \
    --model-size 0.3B \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints \
    --seq-len 4096 \
    --max-seq-len 16384 \
    --batch-size 16 \
    --grad-accum 4

# Z-loss for training stability (penalises large logit magnitudes)
python train_pretrain.py \
    --model-size 0.3B \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints \
    --z-loss-weight 1e-4

# FP32 precision (instead of BF16)
python train_pretrain.py \
    --model-size 0.3B \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints \
    --dtype fp32

# Custom LR with auto-scaling disabled
python train_pretrain.py \
    --model-size 0.3B \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints \
    --lr 1e-3 \
    --no-lr-scale

# torch.compile with reduce-overhead mode (CUDAGraphs)
python train_pretrain.py \
    --model-size 0.6B \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints \
    --jit \
    --compile-mode reduce-overhead

# max-autotune mode (slowest compile, fastest run)
python train_pretrain.py \
    --model-size 0.3B \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints \
    --jit \
    --compile-mode max-autotune

# Resume from a specific checkpoint step
python train_pretrain.py \
    --model-size 0.3B \
    --data-dir ./packed \
    --resume ./checkpoints/step_00050000.pt
```

**Pretrain Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model-size` | `None` | Target size, e.g. `0.3B`, `1.7B`, `70B`, `1T` |
| `--vocab-size` | `None` | Override vocab size (reads from `meta.json`) |
| `--hidden-size` | `None` | Manual override (no auto-size) |
| `--num-layers` | `None` | Manual override |
| `--num-heads` | `None` | Manual override |
| `--num-kv-heads` | `None` | Manual override for GQA |
| `--intermediate-size` | `None` | MLP intermediate size |
| `--head-dim` | `128` | Head dimension |
| `--max-seq-len` | `None` | Max position embeddings (default 8192) |
| `--data-dir` | `./packed` | Packed data directory |
| `--seq-len` | `2048` | Training sequence length |
| `--val-fraction` | `0.01` | Validation fraction |
| `--batch-size` | `8` | Per-GPU batch size |
| `--grad-accum` | `4` | Gradient accumulation steps |
| `--num-steps` | `100000` | Total training steps |
| `--warmup-steps` | `2000` | LR warmup steps |
| `--lr` | `3e-4` | Peak learning rate |
| `--min-lr` | `3e-5` | Minimum LR |
| `--no-lr-scale` | `False` | Disable auto LR scaling |
| `--z-loss-weight` | `1e-4` | Z-loss coefficient (0 = disabled) |
| `--weight-decay` | `0.1` | AdamW weight decay |
| `--grad-clip` | `1.0` | Gradient clipping max norm |
| `--dtype` | `bf16` | `bf16` or `fp32` |
| `--schedule` | `cosine` | `cosine` or `wsd` |
| `--stable-ratio` | `0.8` | WSD stable phase fraction |
| `--optimizer` | `adamw` | `adamw` or `muon` |
| `--jit` | `False` | Enable `torch.compile` |
| `--compile-mode` | `default` | `default`, `reduce-overhead`, `max-autotune` |
| `--num-workers` | `2` | Data loader workers |
| `--gradient-checkpointing` | `False` | Save VRAM (~35%) at compute cost (~30%) |
| `--checkpoint-dir` | `./checkpoints` | Checkpoint output directory |
| `--resume` | `None` | Resume from path (file or dir) |
| `--save-every` | `5000` | Save checkpoint every N steps |
| `--keep-ckpts` | `3` | Recent checkpoints to keep |
| `--log-interval` | `10` | Log every N steps |
| `--val-every` / `--eval-every` | `500` | Validate every N steps |
| `--eval-steps` | `50` | Validation batches |
| `--wandb-project` | `None` | W&B project name |
| `--wandb-run-name` | `None` | W&B run name |
| `--seed` | `42` | Random seed |

### 5B. DeepSpeed Pretrain

```bash
# ZeRO-2, single node
deepspeed train_pretrain_deepspeed.py \
    --model-size 1.7B \
    --data-dir ./packed \
    --out-dir ./checkpoints_ds \
    --batch-size 8 \
    --grad-accum-steps 8 \
    --zero-stage 2 \
    --compile

# ZeRO-3 with CPU offload (for large models on constrained GPUs)
deepspeed train_pretrain_deepspeed.py \
    --model-size 7B \
    --data-dir ./packed \
    --out-dir ./checkpoints_ds \
    --batch-size 4 \
    --grad-accum-steps 16 \
    --zero-stage 3 \
    --cpu-offload-optimizer \
    --cpu-offload-param \
    --compile

# Automatic ZeRO stage selection (hardware audit)
deepspeed train_pretrain_deepspeed.py \
    --model-size 0.6B \
    --data-dir ./packed \
    --out-dir ./checkpoints_ds

# WSD schedule
deepspeed train_pretrain_deepspeed.py \
    --model-size 0.6B \
    --data-dir ./packed \
    --out-dir ./checkpoints_ds \
    --schedule wsd \
    --stable-ratio 0.8 \
    --compile

# TensorBoard logging
deepspeed train_pretrain_deepspeed.py \
    --model-size 0.3B \
    --data-dir ./packed \
    --out-dir ./checkpoints_ds \
    --tensorboard

# With recipe and mode
deepspeed train_pretrain_deepspeed.py \
    --model-size 0.3B \
    --data-dir ./packed \
    --out-dir ./checkpoints_ds \
    --recipe ./recipe.json

# Multi-node (requires hostfile)
deepspeed --hostfile ./hostfile train_pretrain_deepspeed.py \
    --model-size 70B \
    --data-dir ./packed \
    --out-dir ./checkpoints_ds \
    --batch-size 2 \
    --grad-accum-steps 32 \
    --zero-stage 3
```

**DeepSpeed Pretrain Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--zero-stage` | `None` | `1`, `2`, or `3` (auto-selected by hardware audit when `None`) |
| `--cpu-offload-optimizer` | `False` | Offload optimizer states to CPU |
| `--cpu-offload-param` | `False` | Offload model parameters to CPU (ZeRO-3 only) |
| `--out-dir` | `./checkpoints_ds` | Output directory |
| `--ckpt-interval` | `100` | Save checkpoint every N steps |
| `--tensorboard` | `False` | Enable TensorBoard logging |
| `--recipe` | `None` | Path to `recipe.json` |
| `--mode` | `None` | Training mode |

*(All pretrain DDP arguments also apply.)*

---

## 6. Supervised Fine-Tuning (SFT)

### 6A. DDP SFT

```bash
# Full fine-tune
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --batch-size 4 \
    --num-steps 10000

# LoRA fine-tune (recommended for 1B+ models on single GPU)
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --lora-rank 64 \
    --lora-alpha 128 \
    --batch-size 8

# DoRA (weight-decomposed LoRA)
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --lora-rank 32 \
    --lora-alpha 64 \
    --lora-type dora \
    --batch-size 8

# rsLoRA (rank-stabilized)
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --lora-rank 128 \
    --lora-alpha 256 \
    --use-rslora \
    --batch-size 8

# Custom LoRA target modules (all linear projections)
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --lora-rank 64 \
    --lora-target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# Merge LoRA weights into base model and save
python train_sft.py \
    --output-dir ./sft_checkpoints \
    --merge-and-save

# With NEFTune noise (for chat datasets — improves diversity)
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --neftune-alpha 5.0 \
    --batch-size 4

# With torch.compile
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --compile \
    --compile-mode max-autotune \
    --batch-size 8

# With W&B
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --wandb-project my-sft \
    --batch-size 4

# Extended context (reuse pretrained model at longer sequence)
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --seq-len 4096 \
    --max-seq-len 16384 \
    --batch-size 2 \
    --grad-accum 8

# With recipe/mode
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --mode reasoning

# Higher LoRA LR ratio (base params learn slower than adapters)
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --lora-rank 64 \
    --lora-lr-ratio 2.0

# Z-loss during SFT (for stability with high learning rates)
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --z-loss-weight 1e-4
```

**SFT Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint-dir` | `None` | Pretrained checkpoint directory |
| `--data-dir` | `./sft_packed` | Packed SFT data |
| `--output-dir` | `./sft_checkpoints` | Output directory |
| `--resume` | `None` | Resume from checkpoint |
| `--merge-and-save` | `False` | Merge LoRA → base and save |
| `--model-size` | `None` | When no checkpoint, create from scratch |
| `--lora-rank` | `64` | LoRA rank (0 = full fine-tune) |
| `--lora-alpha` | `128.0` | LoRA alpha scaling |
| `--lora-target-modules` | `(attention defaults)` | Comma-separated module names |
| `--lora-type` | `lora` | `lora` or `dora` |
| `--use-rslora` | `False` | Enable rank-stabilized LoRA |
| `--lora-lr-ratio` | `1.0` | LoRA params vs base LR ratio |
| `--neftune-alpha` | `0.0` | NEFTune noise magnitude (0 = disabled) |
| `--seq-len` | `2048` | Training sequence length |
| `--max-seq-len` | `None` | Override model max_position_embeddings |
| `--batch-size` | `4` | Per-GPU batch size |
| `--num-steps` | `10000` | Total training steps |
| `--grad-accum` | `4` | Gradient accumulation |
| `--lr` | `2e-5` | Peak learning rate |
| `--min-lr` | `2e-6` | Minimum LR |
| `--z-loss-weight` | `1e-4` | Z-loss coefficient |
| `--weight-decay` | `0.01` | Weight decay |
| `--warmup-steps` | `200` | LR warmup steps |
| `--grad-clip` | `1.0` | Gradient clipping |
| `--compile` | `False` | Enable `torch.compile` |
| `--compile-mode` | `default` | Compilation mode |
| `--ckpt-interval` | `1000` | Checkpoint interval |
| `--eval-interval` | `200` | Evaluation interval |
| `--eval-steps` | `20` | Validation batches |
| `--val-fraction` | `0.05` | Validation split |

### 6B. DeepSpeed SFT

```bash
# ZeRO-2 LoRA fine-tune
deepspeed train_sft_deepspeed.py \
    --checkpoint-dir ./checkpoints \
    --cache-dir ./sft_packed \
    --out-dir ./sft_checkpoints_ds \
    --lora-rank 64 \
    --batch-size 8 \
    --zero-stage 2

# ZeRO-3 full fine-tune
deepspeed train_sft_deepspeed.py \
    --checkpoint-dir ./checkpoints \
    --cache-dir ./sft_packed \
    --out-dir ./sft_checkpoints_ds \
    --batch-size 4 \
    --grad-accum-steps 8 \
    --zero-stage 3 \
    --compile

# With CPU offload for large model
deepspeed train_sft_deepspeed.py \
    --checkpoint-dir ./checkpoints \
    --cache-dir ./sft_packed \
    --out-dir ./sft_checkpoints_ds \
    --batch-size 4 \
    --zero-stage 3 \
    --cpu-offload-optimizer

# Convenience: --cpu-offload enables both optimizer + params
deepspeed train_sft_deepspeed.py \
    --checkpoint-dir ./checkpoints \
    --cache-dir ./sft_packed \
    --out-dir ./sft_checkpoints_ds \
    --batch-size 4 \
    --zero-stage 3 \
    --cpu-offload

# Custom architecture + SFT
deepspeed train_sft_deepspeed.py \
    --checkpoint-dir ./checkpoints \
    --cache-dir ./sft_packed \
    --out-dir ./sft_checkpoints_ds \
    --arch jamba \
    --jamba-interval 4
```

---

## 7. GRPO (Reinforcement Learning)

### 7A. DDP GRPO

```bash
# Basic GRPO
python train_grpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./grpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./grpo_checkpoints \
    --num-steps 500

# With LoRA
python train_grpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./grpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./grpo_checkpoints \
    --lora --lora-rank 64 \
    --num-steps 500

# Custom reward tiers
python train_grpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./grpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./grpo_checkpoints \
    --reward-correct 2.0 \
    --reward-format 0.5 \
    --num-generations 16

# KL penalty and clipping
python train_grpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./grpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./grpo_checkpoints \
    --kl-coef 0.05 \
    --clip-range 0.3 \
    --num-steps 1000

# With entropy bonus (encourages exploration)
python train_grpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./grpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./grpo_checkpoints \
    --entropy-coeff 0.01 \
    --num-steps 500

# Two-model reference policy (more stable, 2× memory)
python train_grpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./grpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./grpo_checkpoints \
    --ref-policy two

# Custom generation parameters
python train_grpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./grpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./grpo_checkpoints \
    --num-generations 8 \
    --max-new-tokens 1024 \
    --temperature 1.2 \
    --top-p 0.9

# With torch.compile
python train_grpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./grpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./grpo_checkpoints \
    --compile \
    --num-steps 500

# Custom model size (for fine-tuning a different architecture)
python train_grpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./grpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./grpo_checkpoints \
    --model-size 0.6B \
    --num-steps 500
```

**GRPO Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | `None` | Path to pretrained checkpoint |
| `--tokenizer` | `./tokenizer` | Tokenizer directory |
| `--data-dir` | `./grpo_packed` | Packed GRPO data |
| `--out-dir` | `./grpo_checkpoints` | Output directory |
| `--resume` | `None` | Resume from checkpoint |
| `--lora` | `False` | Enable LoRA |
| `--lora-rank` | `64` | LoRA rank |
| `--lora-alpha` | `128.0` | LoRA alpha |
| `--ref-policy` | `single` | `single` or `two` (two-model reference) |
| `--num-generations` | `8` | G — completions per prompt |
| `--max-new-tokens` | `512` | Max generation tokens |
| `--temperature` | `1.0` | Sampling temperature |
| `--top-p` | `0.95` | Top-p sampling |
| `--reward-correct` | `1.0` | Reward for correct answer |
| `--reward-format` | `0.3` | Reward for correct `<think>` format |
| `--kl-coef` | `0.02` | KL penalty coefficient |
| `--clip-range` | `0.2` | PPO clipping range |
| `--entropy-coeff` | `0.0` | Entropy bonus coefficient |
| `--batch-size` | `4` | Per-GPU batch size |
| `--num-steps` | `500` | Total training steps |
| `--warmup-steps` | `20` | LR warmup steps |
| `--lr` | `1e-6` | Peak learning rate |
| `--min-lr` | `1e-7` | Minimum LR |
| `--no-lr-scale` | `False` | Disable auto LR scaling |
| `--weight-decay` | `0.0` | Weight decay |
| `--grad-clip` | `1.0` | Gradient clipping |
| `--compile` | `False` | Enable `torch.compile` |
| `--compile-mode` | `default` | Compilation mode |
| `--save-every` | `50` | Save interval |
| `--log-interval` | `1` | Log interval |

### 7B. DeepSpeed GRPO

```bash
# ZeRO-2 GRPO
deepspeed train_grpo_deepspeed.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --cache_dir ./grpo_packed \
    --out_dir ./grpo_checkpoints_ds \
    --tokenizer ./tokenizer \
    --batch_size 8 \
    --zero-stage 2

# ZeRO-3 with LoRA
deepspeed train_grpo_deepspeed.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --cache_dir ./grpo_packed \
    --out_dir ./grpo_checkpoints_ds \
    --tokenizer ./tokenizer \
    --lora --lora_rank 64 \
    --zero-stage 3 \
    --batch_size 4

# Custom reward tiers
deepspeed train_grpo_deepspeed.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --cache_dir ./grpo_packed \
    --out_dir ./grpo_checkpoints_ds \
    --tokenizer ./tokenizer \
    --reward_correct 2.0 \
    --reward_format 0.5

# Merge LoRA after training
deepspeed train_grpo_deepspeed.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --cache_dir ./grpo_packed \
    --out_dir ./grpo_checkpoints_ds \
    --tokenizer ./tokenizer \
    --merge_lora
```

---

## 8. DPO (Preference Optimization)

### 8A. DDP DPO

```bash
# Basic DPO
python train_dpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./dpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./dpo_checkpoints \
    --num-steps 500

# With LoRA
python train_dpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./dpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./dpo_checkpoints \
    --lora --lora-rank 64 \
    --num-steps 500

# Two-model reference (separate frozen reference network)
python train_dpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./dpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./dpo_checkpoints \
    --ref-policy two \
    --num-steps 500

# With label smoothing (epsilon in DPO formula)
python train_dpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./dpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./dpo_checkpoints \
    --beta 0.2 \
    --label-smoothing 0.1 \
    --num-steps 500

# With KL anchoring to reference (extra stability)
python train_dpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./dpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./dpo_checkpoints \
    --kl-coef 0.01 \
    --beta 0.1 \
    --num-steps 500

# Adaptive clipping (PPO-style clipping of implicit reward)
python train_dpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./dpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./dpo_checkpoints \
    --clip-ratio 0.5 \
    --num-steps 500

# With torch.compile
python train_dpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./dpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./dpo_checkpoints \
    --compile \
    --num-steps 500

# Multi-GPU (torchrun)
torchrun --nproc_per_node=4 train_dpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./dpo_packed \
    --tokenizer ./tokenizer \
    --batch-size 4
```

**DPO Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | `None` | Path to pretrained checkpoint |
| `--tokenizer` | `./tokenizer` | Tokenizer directory |
| `--data-dir` | `./dpo_packed` | Packed DPO data |
| `--out-dir` | `./dpo_checkpoints` | Output directory |
| `--resume` | `None` | Resume from checkpoint |
| `--lora` | `False` | Enable LoRA |
| `--lora-rank` | `64` | LoRA rank |
| `--lora-alpha` | `128.0` | LoRA alpha |
| `--ref-policy` | `single` | `single` or `two` |
| `--beta` | `0.1` | DPO beta (inverse temperature) |
| `--label-smoothing` | `0.0` | DPO label smoothing epsilon |
| `--clip-ratio` | `0.0` | Adaptive clipping (0 = disabled) |
| `--kl-coef` | `0.0` | KL penalty (0 = disabled) |
| `--batch-size` | `4` | Per-GPU batch size |
| `--num-steps` | `500` | Total steps |
| `--max-steps` | `None` | Alternative to `--num-steps` |
| `--warmup-steps` | `20` | LR warmup |
| `--lr` | `1e-6` | Peak LR |
| `--min-lr` | `1e-7` | Minimum LR |
| `--weight-decay` | `0.0` | Weight decay |
| `--grad-clip` | `1.0` | Gradient clipping |
| `--save-every` | `50` | Save interval |
| `--log-interval` | `1` | Log interval |

### 8B. DeepSpeed DPO

```bash
# ZeRO-2 DPO
deepspeed train_dpo_deepspeed.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./dpo_packed \
    --out-dir ./dpo_checkpoints_ds \
    --tokenizer ./tokenizer \
    --batch-size 8 \
    --zero-stage 2

# ZeRO-3 with LoRA
deepspeed train_dpo_deepspeed.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./dpo_packed \
    --out-dir ./dpo_checkpoints_ds \
    --tokenizer ./tokenizer \
    --lora --lora-rank 64 \
    --zero-stage 3

# With CPU offload
deepspeed train_dpo_deepspeed.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./dpo_packed \
    --out-dir ./dpo_checkpoints_ds \
    --tokenizer ./tokenizer \
    --zero-stage 3 \
    --cpu-offload-optimizer \
    --batch-size 4
```

---

## 9. Inference

### 9A. Interactive REPL

```bash
# Interactive mode with streaming
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --interactive

# Interactive with thinking enabled
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --interactive \
    --enable-thinking

# Interactive with custom system prompt
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --interactive \
    --system "You are a helpful coding assistant."

# Interactive without streaming
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --interactive \
    --no-stream
```

### 9B. One-Shot Generation

```bash
# Single prompt
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --prompt "What is the derivative of ln(x)?"

# Multiple prompts (batched)
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --prompt "What is 2+2?" \
    --prompt "Explain quantum computing." \
    --batch-size 2

# With explicit recipe
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --recipe ./recipe.json \
    --prompt "Hello, world!"

# Override chat template
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --prompt "Hello" \
    --chat-template raw

# Explicit EOS token override
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --prompt "Complete this: 2+2=" \
    --eos-token-id 50256
```

### 9C. Batched from File

```bash
# Batch from JSONL
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --prompts-file ./eval.jsonl \
    --batch-size 8 \
    --output ./completions.jsonl

# Without streaming (faster for batch)
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --prompts-file ./eval.jsonl \
    --batch-size 16 \
    --output ./completions.jsonl \
    --no-stream
```

### 9D. Quantized Inference

```bash
# 8-bit quantization
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --quantize 8bit \
    --prompt "Hello, world!"

# 4-bit quantization (smallest memory footprint, ~4× reduction)
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --quantize 4bit \
    --interactive

# Quantized + batch
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --quantize 4bit \
    --prompts-file ./eval.jsonl \
    --batch-size 4 \
    --output ./out.jsonl
```

### 9E. Advanced Generation Settings

```bash
# Low temperature (more deterministic)
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --prompt "Complete: The capital of France is" \
    --temperature 0.2 \
    --top-k 10

# Creative generation
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --prompt "Write a poem about AI" \
    --temperature 1.2 \
    --top-p 0.95 \
    --top-k 100

# With repetition penalty (to avoid loops)
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --prompt "Tell me a story" \
    --repetition-penalty 1.2 \
    --max-new-tokens 1000

# Min + max length constraints
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --prompt "Explain gravity" \
    --min-new-tokens 50 \
    --max-new-tokens 200

# Seeded for reproducibility
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --prompt "1 + 1 =" \
    --seed 42
```

### 9F. Device & Memory Management

```bash
# Run on specific GPU
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --device cuda:0 \
    --prompt "Hello"

# Run on CPU
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --device cpu \
    --prompt "Hello"

# With device memory budget (accelerate-style)
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --max-memory "0:18GiB,cpu:30GiB" \
    --prompt "Hello"

# Override max sequence length (for long-context inference)
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --max-seq-len 16384 \
    --prompt "Long context prompt..."
```

### 9G. LoRA Checkpoint Inference

```bash
# Inference with a LoRA checkpoint (auto-detects and merges)
python infer.py \
    --checkpoint ./sft_checkpoints/sft_step_010000.pt \
    --prompt "Hello, world!"
```

The inference script auto-detects whether the checkpoint contains a `lora_state_dict`, creates LoRA adapters, loads weights, and merges them into the base model on-the-fly.

**Inference Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | **(required)** | Path to `.pt` checkpoint |
| `--recipe` | `None` | Path to `recipe.json` (auto-detected from checkpoint dir) |
| `--device` | `auto` | `auto`, `cpu`, or `cuda:N` |
| `--max-memory` | `None` | Per-device memory budget, e.g. `'0:18GiB,cpu:30GiB'` |
| `--max-seq-len` | `None` | Override `max_position_embeddings` |
| `--quantize` | `none` | `4bit`, `8bit`, or `none` |
| `--prompt` | `None` | One or more prompts (repeatable) |
| `--prompts-file` | `None` | JSONL file with prompts |
| `--interactive` | `False` | REPL mode |
| `--smoke-test` | `False` | Smoke test mode |
| `--chat-template` | `auto` | `auto`, `chatml`, `raw`, `custom` |
| `--enable-thinking` | `False` | Open `<think>` block in assistant turn |
| `--system` | `None` | System message for every prompt |
| `--max-new-tokens` | `512` | Max generation length |
| `--min-new-tokens` | `0` | Min generation length |
| `--temperature` | `0.7` | Sampling temperature |
| `--top-k` | `50` | Top-k sampling |
| `--top-p` | `0.9` | Top-p (nucleus) sampling |
| `--repetition-penalty` | `1.0` | Repetition penalty |
| `--seed` | `None` | RNG seed |
| `--eos-token-id` | `None` | Override EOS token |
| `--batch-size` | `1` | Micro-batch size |
| `--output` | `None` | Output JSONL path |
| `--stream` / `--no-stream` | `True` | Token streaming |

---

## 10. Architecture Variants

All training scripts (DDP, DeepSpeed, and Hivemind) accept architecture variant flags. These are defined centrally in `model.py` via `add_architecture_args()` and `apply_architecture_args()`.

### 10A. Quick Reference

| Variant | Flags | Description |
|---------|-------|-------------|
| **Dense (default)** | *(none needed)* | Standard pre-norm decoder-only transformer |
| **Parallel** | `--layer-type parallel` | PaLM-style: attn + MLP computed in parallel from same input |
| **Jamba** | `--arch jamba --jamba-interval 4` | Hybrid SSM + Attention: every Nth layer is attention, rest are Mamba SSM |
| **MoD** | `--mod-alpha 0.25` | Mixture-of-Depth: router skips FFN for low-scoring tokens |
| **MLA** | `--use-mla` | Multi-head Latent Attention (DeepSeek-style, low-rank KV compression) |
| **MTP** | `--num-mtp-heads 3` | Multi-Token Prediction with auxiliary loss (faster convergence) |
| **Sliding Window** | `--sliding-window-size 4096` | Alternating global + local window attention (efficient long context) |

### 10B. Architecture Variant Commands

**Parallel Layer (PaLM-style)**

```bash
python train_pretrain.py \
    --model-size 0.3B \
    --layer-type parallel \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints
```

**Jamba Hybrid (SSM + Attention)**

```bash
# Attention every 4th layer, rest are Mamba SSM
python train_pretrain.py \
    --model-size 0.6B \
    --arch jamba \
    --jamba-interval 4 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints

# Attention every 6th layer (more SSM, less attention — faster)
python train_pretrain.py \
    --model-size 1.7B \
    --arch jamba \
    --jamba-interval 6 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints
```

**Mixture-of-Depth (MoD)**

```bash
# Aggressive routing (~50% of tokens skip FFN)
python train_pretrain.py \
    --model-size 0.3B \
    --mod-alpha 0.3 \
    --mod-loss-weight 0.01 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints

# Light routing (~20% of tokens skip FFN)
python train_pretrain.py \
    --model-size 0.3B \
    --mod-alpha 0.15 \
    --mod-loss-weight 0.005 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints

# MoD + Parallel (maximum throughput combination)
python train_pretrain.py \
    --model-size 0.3B \
    --layer-type parallel \
    --mod-alpha 0.25 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints
```

**Multi-head Latent Attention (MLA)**

```bash
# Default KV compression rank (hidden_size // 4)
python train_pretrain.py \
    --model-size 0.3B \
    --use-mla \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints

# Explicit KV LoRA rank
python train_pretrain.py \
    --model-size 0.6B \
    --use-mla \
    --kv-lora-rank 256 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints

# MLA + Sliding Window (efficient KV for long context)
python train_pretrain.py \
    --model-size 0.6B \
    --use-mla \
    --sliding-window-size 8192 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints
```

**Multi-Token Prediction (MTP)**

```bash
# 3 auxiliary heads, default discount 0.5
python train_pretrain.py \
    --model-size 0.3B \
    --num-mtp-heads 3 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints

# 5 heads with higher discount for later tokens
python train_pretrain.py \
    --model-size 0.6B \
    --num-mtp-heads 5 \
    --mtp-discount 0.6 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints

# MTP + Parallel + MoD (all efficiency tricks combined)
python train_pretrain.py \
    --model-size 0.6B \
    --num-mtp-heads 3 \
    --layer-type parallel \
    --mod-alpha 0.2 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints
```

**Sliding Window Attention**

```bash
# 4096-token sliding window (alternating global/sw layers)
python train_pretrain.py \
    --model-size 0.3B \
    --sliding-window-size 4096 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints

# 8192-token sliding window
python train_pretrain.py \
    --model-size 0.6B \
    --sliding-window-size 8192 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints
```

### 10C. Composed Variants (Everything Together)

```bash
# Jamba + MLA + MTP + Sliding Window
python train_pretrain.py \
    --model-size 1B \
    --arch jamba \
    --use-mla \
    --num-mtp-heads 3 \
    --sliding-window-size 4096 \
    --jamba-interval 4 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints

# Dense + Parallel + MoD + MTP (max throughput + faster convergence)
python train_pretrain.py \
    --model-size 0.6B \
    --layer-type parallel \
    --mod-alpha 0.25 \
    --num-mtp-heads 3 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints

# MLA + MoD + Sliding Window (long context, efficient compute)
python train_pretrain.py \
    --model-size 0.6B \
    --use-mla \
    --mod-alpha 0.2 \
    --sliding-window-size 8192 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints

# Jamba + Parallel layer type
python train_pretrain.py \
    --model-size 0.6B \
    --arch jamba \
    --layer-type parallel \
    --jamba-interval 4 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints
```

### 10D. Variants in SFT, GRPO, DPO

All architecture flags work identically in every training script:

```bash
# SFT with MLA
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --use-mla

# GRPO with parallel layers
python train_grpo.py \
    --checkpoint ./checkpoints/latest.pt \
    --data-dir ./grpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./grpo_checkpoints \
    --layer-type parallel

# DPO with MTP heads (auxiliary loss during preference tuning)
python train_dpo.py \
    --checkpoint ./checkpoints/latest.pt \
    --data-dir ./dpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./dpo_checkpoints \
    --num-mtp-heads 3

# DeepSpeed SFT with Jamba + Sliding Window
deepspeed train_sft_deepspeed.py \
    --checkpoint-dir ./checkpoints \
    --cache-dir ./sft_packed \
    --out-dir ./sft_checkpoints_ds \
    --arch jamba \
    --sliding-window-size 4096
```

### 10E. Architecture Variant Arguments

| Flag | Default | Choices | Description |
|------|---------|---------|-------------|
| `--arch` | `dense` | `dense`, `jamba` | Architecture type |
| `--layer-type` | `sequential` | `sequential`, `parallel` | Layer computation order |
| `--sliding-window-size` | `0` | int | Window size (0 = disabled) |
| `--num-mtp-heads` | `0` | int | MTP heads (0 = disabled) |
| `--mtp-discount` | `0.5` | float | Discount factor per future token |
| `--mod-alpha` | `0.0` | float | MoD routing threshold (0 = disabled) |
| `--mod-loss-weight` | `0.01` | float | MoD auxiliary loss weight |
| `--use-mla` | `False` | flag | Enable Multi-head Latent Attention |
| `--kv-lora-rank` | `None` | int | MLA KV compression rank (default: `hidden_size // 4`) |
| `--jamba-interval` | `4` | int | Place attention layer every N layers in Jamba |

### 10F. Architecture Decision Guide

| Goal | Recommended Flags |
|------|-------------------|
| Maximum throughput | `--layer-type parallel --mod-alpha 0.25` |
| Long context, efficient KV | `--use-mla` or `--use-mla --sliding-window-size 8192` |
| Best quality-to-compute ratio | `--arch jamba --jamba-interval 4` |
| Faster convergence | `--num-mtp-heads 3` (auxiliary MTP loss) |
| Standard baseline | *(default: dense sequential)* |
| Long context + compute efficient | `--use-mla --mod-alpha 0.2 --sliding-window-size 8192` |
| Maximum everything | `--arch jamba --use-mla --layer-type parallel --num-mtp-heads 3 --mod-alpha 0.2` |

---

## 11. Recipe System

The `TrainingRecipe` class in `recipe.py` is the single source of truth for training mode, chat templates, special tokens, and model name.

### 11A. Three Training Modes

| Mode | `<think>` tags? | Use Case |
|------|----------------|----------|
| `reasoning` | Required | Math, code, logic — model shows chain-of-thought |
| `non_reasoning` | Not used | General chat, instruction following, creative writing |
| `hybrid` | Per-example `want_thinking` flag | Mix of reasoning + non-reasoning in one training run |

### 11B. Using Recipes

```bash
# Training scripts: pass --recipe or --mode
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --recipe ./recipe.json

# Or just specify mode (uses default special tokens for that mode)
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --mode reasoning

# Inference auto-detects recipe from checkpoint directory
python infer.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --prompt "Hello" \
    --recipe ./recipe.json

# Data packing scripts also accept recipes
python data/pack_sft.py \
    --data-dir ./sft_data \
    --tokenizer ./tokenizer \
    --cache-dir ./sft_packed \
    --mode hybrid

# DeepSpeed training
deepspeed train_pretrain_deepspeed.py \
    --model-size 0.3B \
    --data-dir ./packed \
    --out-dir ./checkpoints_ds \
    --recipe ./recipe.json

# GRPO with recipe
python train_grpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./grpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./grpo_checkpoints \
    --recipe ./recipe.json
```

### 11C. Recipe JSON Format

```json
{
  "mode": "reasoning",
  "chat_template": "chatml",
  "turn_prefix_user": "<|im_start|>user\n",
  "turn_suffix_user": "<|im_end|>\n",
  "turn_prefix_assistant": "<|im_start|>assistant\n",
  "turn_suffix_assistant": "<|im_end|>\n",
  "think_open": "<think>",
  "think_close": "</think>",
  "hybrid_think_token": "<|think_on|>",
  "hybrid_nothink_token": "<|think_off|>",
  "model_name": "DenseLLM"
}
```

### 11D. Special Tokens

| Token | `reasoning` | `non_reasoning` | `hybrid` |
|-------|-------------|-----------------|----------|
| `<\|endoftext\|>` | ✓ | ✓ | ✓ |
| `<\|pad\|>` | ✓ | ✓ | ✓ |
| `<\|im_start\|>` | ✓ | ✓ | ✓ |
| `<\|im_end\|>` | ✓ | ✓ | ✓ |
| `<think>` | ✓ | ✗ | ✓ |
| `</think>` | ✓ | ✗ | ✓ |
| `<\|think_on\|>` | ✗ | ✗ | ✓ |
| `<\|think_off\|>` | ✗ | ✗ | ✓ |

### 11E. Python API

```python
from recipe import TrainingRecipe

# Create default (reasoning mode)
recipe = TrainingRecipe()

# Explicit config
recipe = TrainingRecipe(
    mode="hybrid",
    chat_template="chatml",
    turn_prefix_user="<|im_start|>user\n",
    turn_suffix_user="<|im_end|>\n",
)

# Format turns
user_turn = recipe.format_user_turn("What is 2+2?")
# → "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"

assistant_turn = recipe.format_assistant_turn(
    thinking="2 plus 2 equals 4",
    answer="4"
)
# → "<|im_start|>assistant\n<think>\n2 plus 2 equals 4\n</think>\n4<|im_end|>\n"

# Full conversation (with optional system message)
conv = recipe.format_full_conversation(
    prompt="What is 2+2?",
    thinking="2 plus 2 equals 4",
    answer="4",
    system="You are a math tutor.",
)

# Token access
print(recipe.special_tokens)
print(recipe.eos_token)       # <|endoftext|>
print(recipe.pad_token)       # <|pad|>

# Serialize
recipe.to_json("./recipe.json")
recipe = TrainingRecipe.from_json("./recipe.json")
```

---

## 12. Optimizer & LR Schedules

### 12A. AdamW (Default)

```bash
# AdamW with default LR auto-scaling
python train_pretrain.py \
    --model-size 1.7B \
    --optimizer adamw \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints

# AdamW with explicit LR
python train_pretrain.py \
    --model-size 0.3B \
    --optimizer adamw \
    --lr 3e-4 \
    --weight-decay 0.1 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints
```

### 12B. Muon (Newton-Schulz, 2-3× Faster)

```bash
# Muon optimizer (recommended for 1B+ models)
python train_pretrain.py \
    --model-size 1.7B \
    --optimizer muon \
    --lr 2e-4 \
    --weight-decay 0.1 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints

# Muon + WSD (best combination)
python train_pretrain.py \
    --model-size 1.7B \
    --optimizer muon \
    --schedule wsd \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints
```

Muon uses two optimizers internally: Muon for embedding-free parameters and AdamW for embeddings/norms.

### 12C. Cosine Schedule (Default)

```bash
# Cosine decay with linear warmup
python train_pretrain.py \
    --model-size 0.3B \
    --schedule cosine \
    --num-steps 100000 \
    --warmup-steps 2000 \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints
```

### 12D. WSD Schedule (Warmup-Stable-Decay)

WSD keeps LR at peak for most of training then quickly decays — often yields better final loss than cosine for the same step count.

```bash
# WSD: 80% stable, 20% decay
python train_pretrain.py \
    --model-size 0.6B \
    --schedule wsd \
    --num-steps 100000 \
    --warmup-steps 2000 \
    --stable-ratio 0.8 \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints

# WSD with Muon
python train_pretrain.py \
    --model-size 1.7B \
    --schedule wsd \
    --stable-ratio 0.85 \
    --optimizer muon \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints
```

### 12E. LR Auto-Scaling

The framework automatically scales the learning rate based on model size using scaling laws:

```bash
# Auto-scaled — the LR you pass is adjusted internally
python train_pretrain.py \
    --model-size 13B \
    --lr 3e-4 \
    --data-dir ./packed

# Disable auto-scaling for manual control
python train_pretrain.py \
    --model-size 13B \
    --lr 1.5e-4 \
    --no-lr-scale \
    --data-dir ./packed
```

---

## 13. PEFT: LoRA / DoRA / rsLoRA

### 13A. LoRA (Low-Rank Adaptation)

```bash
# Standard LoRA in SFT
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --lora-rank 64 \
    --lora-alpha 128

# LoRA in GRPO
python train_grpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./grpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./grpo_checkpoints \
    --lora --lora-rank 32 --lora-alpha 64

# LoRA in DPO
python train_dpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./dpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./dpo_checkpoints \
    --lora --lora-rank 64
```

### 13B. DoRA (Weight-Decomposed Low-Rank Adaptation)

```bash
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --lora-rank 32 \
    --lora-type dora
```

### 13C. rsLoRA (Rank-Stabilized LoRA)

```bash
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --lora-rank 128 \
    --lora-alpha 256 \
    --use-rslora
```

### 13D. Custom Target Modules

```bash
# Adapt all linear projections (not just attention)
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --lora-rank 64 \
    --lora-target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# LoRA parameters learn at a different rate than base parameters
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --lora-rank 64 \
    --lora-lr-ratio 2.0
```

### 13E. Merging LoRA Checkpoints

```bash
# After training, merge LoRA weights into base model and save full weights
python train_sft.py --merge-and-save \
    --output-dir ./sft_checkpoints

# Inference auto-merges LoRA (no separate merge step needed)
python infer.py \
    --checkpoint ./sft_checkpoints/sft_step_010000.pt \
    --prompt "Hello"
```

### 13F. PEFT in DeepSpeed

```bash
# LoRA in DeepSpeed SFT
deepspeed train_sft_deepspeed.py \
    --checkpoint-dir ./checkpoints \
    --cache-dir ./sft_packed \
    --out-dir ./sft_checkpoints_ds \
    --lora-rank 64 \
    --zero-stage 2

# LoRA in DeepSpeed GRPO
deepspeed train_grpo_deepspeed.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --cache_dir ./grpo_packed \
    --out_dir ./grpo_checkpoints_ds \
    --tokenizer ./tokenizer \
    --lora --lora_rank 64 \
    --zero-stage 3
```

---

## 14. RoPE Scaling (YaRN / NTK)

All architectures support YaRN and NTK-aware RoPE frequency scaling for extended context beyond the original `max_position_embeddings`.

### 14A. YaRN Scaling

```bash
# 8× context extension (e.g., 8K → 64K)
python train_pretrain.py \
    --model-size 0.3B \
    --rope-scaling '{"type": "yarn", "factor": 8.0}' \
    --max-seq-len 65536 \
    --seq-len 4096 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints

# YaRN with 4× in SFT
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --rope-scaling '{"type": "yarn", "factor": 4.0}' \
    --max-seq-len 32768 \
    --seq-len 8192
```

### 14B. NTK-Aware Scaling

```bash
# NTK-aware 8× extension (better high-frequency preservation)
python train_pretrain.py \
    --model-size 0.3B \
    --rope-scaling '{"type": "ntk", "factor": 8.0}' \
    --max-seq-len 65536 \
    --seq-len 4096 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints

# NTK with 16× for very long context
python train_pretrain.py \
    --model-size 0.6B \
    --rope-scaling '{"type": "ntk", "factor": 16.0}' \
    --max-seq-len 131072 \
    --seq-len 8192 \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints
```

> **Note:** `rope_scaling` is a JSON object parsed by `ModelConfig`. The `factor` is the context extension multiplier (e.g., `8.0` = 8× the original `max_position_embeddings`). YaRN and NTK work identically across all architectures (dense, Jamba, etc.).

---

## 15. Ollama DPO Judge

The `ollama_judge.py` script generates preference pairs using a remote Ollama server. Given prompts, it generates multiple candidates and uses a judge model to pick preferred/rejected.

### 15A. Single Prompt

```bash
python ollama_judge.py \
    --prompt "What is the capital of France?" \
    --url http://localhost:11434 \
    --judge-model llama3.1:8b \
    --num-candidates 4
```

### 15B. Batch Mode from File

```bash
# Process prompts from JSONL file
python ollama_judge.py \
    --input ./prompts.jsonl \
    --output ./preference_pairs.jsonl \
    --url http://localhost:11434 \
    --judge-model qwen2.5:7b-instruct \
    --gen-model qwen2.5:7b \
    --num-candidates 6 \
    --max-pairs 2 \
    --max-prompts 100
```

### 15C. Custom Generation Settings

```bash
python ollama_judge.py \
    --input ./prompts.jsonl \
    --output ./pairs.jsonl \
    --judge-model llama3.1:8b \
    --gen-model llama3.1:8b \
    --num-candidates 4 \
    --temperature 0.9 \
    --max-tokens 1024 \
    --timeout 180 \
    --system-prompt "You are a helpful math tutor."

# Use separate gen and judge models
python ollama_judge.py \
    --input ./math_prompts.jsonl \
    --output ./pairs.jsonl \
    --gen-model qwen2.5:7b \
    --judge-model qwen2.5:7b-instruct
```

### 15D. Connectivity Test

```bash
# Verify connection to Ollama server
python ollama_judge.py \
    --url http://192.168.1.50:11434 \
    --test
```

### 15E. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Remote Ollama server URL |
| `OLLAMA_JUDGE_MODEL` | `qwen2.5:7b-instruct` | Model used for pairwise judging |
| `OLLAMA_GEN_MODEL` | (same as judge) | Model used for candidate generation |

**Ollama Judge Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--url` | `http://localhost:11434` | Ollama server URL |
| `--judge-model` | `llama3.1:8b` | Judge model name |
| `--gen-model` | `""` | Generator model (same as judge if empty) |
| `--timeout` | `120` | HTTP timeout (seconds) |
| `--num-candidates` | `4` | Candidates per prompt |
| `--max-pairs` | `1` | Max preference pairs per prompt |
| `--temperature` | `0.8` | Sampling temperature |
| `--max-tokens` | `512` | Max tokens per candidate |
| `--system-prompt` | `""` | Optional system prompt |
| `--max-prompts` | `0` | Max prompts to process (0 = all) |
| `--test` | `False` | Run connectivity test |
| `--prompt` | `None` | Single prompt (one-shot mode) |
| `--input` | `None` | Input JSONL file (batch mode) |
| `--output` | `./preference_pairs.jsonl` | Output JSONL file |

---

## 16. Distributed Multi-GPU

### 16A. DDP via torchrun

```bash
# Single-node, 4 GPUs
torchrun --nproc_per_node=4 train_pretrain.py \
    --model-size 1.7B \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints \
    --batch-size 8 \
    --grad-accum 8

# Single-node, 8 GPUs
torchrun --nproc_per_node=8 train_pretrain.py \
    --model-size 7B \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints \
    --batch-size 4 \
    --grad-accum 16 \
    --gradient-checkpointing

# Torch DDP SFT on 4 GPUs
torchrun --nproc_per_node=4 train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --lora-rank 64 \
    --batch-size 8

# Torch DDP DPO on 4 GPUs
torchrun --nproc_per_node=4 train_dpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./dpo_packed \
    --tokenizer ./tokenizer \
    --batch-size 4
```

### 16B. DeepSpeed Multi-Node

```bash
# Single node, ZeRO-2
deepspeed --num_gpus=4 train_pretrain_deepspeed.py \
    --model-size 1.7B \
    --data-dir ./packed \
    --out-dir ./checkpoints_ds \
    --zero-stage 2

# Multi-node via hostfile
deepspeed --hostfile ./hostfile train_pretrain_deepspeed.py \
    --model-size 13B \
    --data-dir ./packed \
    --out-dir ./checkpoints_ds \
    --batch-size 4 \
    --grad-accum-steps 8 \
    --zero-stage 3

# DeepSpeed SFT multi-node
deepspeed --hostfile ./hostfile train_sft_deepspeed.py \
    --checkpoint-dir ./checkpoints \
    --cache-dir ./sft_packed \
    --out-dir ./sft_checkpoints_ds \
    --lora-rank 64 \
    --zero-stage 3
```

### 16C. Host File Format

```
# hostfile example for 2 nodes with 4 GPUs each
node1 slots=4
node2 slots=4
```

---

## 17. Decentralized Hivemind Training

Hivemind enables heterogeneous multi-node training across different GPUs, machines, and even CPUs.

### 17A. Pretrain with Hivemind

**Bootstrap node** (first machine, e.g., RTX 4090):

```bash
bash hivemind/run.sh bootstrap \
    --model-size 300M \
    --data-dir ./packed \
    --batch-size 16 \
    --grad-accum 2 \
    --dtype bf16 \
    --checkpoint-dir ./hivemind_ckpts_a
```

**Worker node** (second machine, e.g., RTX 3050):

```bash
bash hivemind/run.sh worker 192.168.1.100:5678 \
    --model-size 300M \
    --data-dir ./packed \
    --batch-size 4 \
    --dtype bf16 \
    --checkpoint-dir ./hivemind_ckpts_b
```

**CPU laptop:**

```bash
bash hivemind/run.sh worker 192.168.1.100:5678 \
    --model-size 300M \
    --data-dir ./packed \
    --batch-size 1 \
    --dtype fp32 \
    --checkpoint-dir ./hivemind_ckpts_c
```

### 17B. SFT with Hivemind

```bash
# Bootstrap
bash hivemind/run.sh sft-bootstrap \
    --model-size 300M \
    --data-dir ./sft_packed \
    --lora-rank 64 \
    --checkpoint-dir ./sft_ckpts

# Worker
bash hivemind/run.sh sft-worker 192.168.1.100:5678 \
    --model-size 300M \
    --data-dir ./sft_packed \
    --lora-rank 32 \
    --checkpoint-dir ./sft_ckpts_b
```

### 17C. GRPO with Hivemind

```bash
# Bootstrap
bash hivemind/run.sh grpo-bootstrap \
    --checkpoint ./sft_ckpts/latest.pt \
    --data-dir ./grpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./grpo_ckpts \
    --batch-size 8 \
    --num-generations 8

# Worker
bash hivemind/run.sh grpo-worker 192.168.1.100:5678 \
    --checkpoint ./sft_ckpts/latest.pt \
    --data-dir ./grpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./grpo_ckpts_b \
    --batch-size 2 \
    --num-generations 4
```

### 17D. DPO with Hivemind

```bash
# Bootstrap
bash hivemind/run.sh dpo-bootstrap \
    --checkpoint ./sft_ckpts/latest.pt \
    --data-dir ./dpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./dpo_ckpts \
    --batch-size 8

# Worker
bash hivemind/run.sh dpo-worker 192.168.1.100:5678 \
    --checkpoint ./sft_ckpts/latest.pt \
    --data-dir ./dpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./dpo_ckpts_b \
    --batch-size 2
```

### 17E. Architecture Variants + Hivemind

```bash
# Jamba with Hivemind
bash hivemind/run.sh bootstrap \
    --model-size 300M \
    --arch jamba \
    --jamba-interval 4 \
    --data-dir ./packed \
    --checkpoint-dir ./ckpts

# MLA + Sliding Window with Hivemind
bash hivemind/run.sh bootstrap \
    --model-size 300M \
    --use-mla \
    --sliding-window-size 4096 \
    --data-dir ./packed \
    --checkpoint-dir ./ckpts
```

### 17F. Checkpoint Averaging (Merged Model)

```bash
# After distributed training, produce a merged evaluation checkpoint
bash hivemind/run.sh average 192.168.1.100:5678 ./averaged_model
```

### 17G. Hivemind CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--hivemind` | `False` | Enable decentralized training |
| `--initial-peers` | `""` | Bootstrap peers (empty = bootstrap node) |
| `--host` | `0.0.0.0` | Network interface to bind |
| `--port` | `0` | P2P port (0 = random) |
| `--peer-id` | `None` | Human-readable peer name |
| `--target-group-size` | `8` | Averaging fan-out |
| `--averaging-period` | `1` | All-reduce every N steps |
| `--average-parameters` | `True` | Average parameters (not gradients) |
| `--no-average-parameters` | `False` | Average gradients instead |
| `--checkpoint-average-rounds` | `3` | Rounds for final checkpoint averaging |
| `--average-checkpoints` | `False` | After training, average params across swarm |

### 17H. Hivemind Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Peer A        │    │   Peer B        │    │   Peer C        │
│   RTX 4090      │    │   RTX 3060      │    │   MacBook M3    │
│   8 GB batch    │    │   4 GB batch    │    │   2 GB batch    │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Hivemind DHT        │
                    │  (decentralized)      │
                    │  Async parameter      │
                    │  averaging            │
                    └───────────────────────┘
```

Each peer: has the full model → reads own data shard → runs local steps at its own pace → fires async all-reduce after each step → continues immediately without blocking → absorbs averaged parameters when the all-reduce completes.

---

## 18. Checkpoint Save / Resume

### 18A. Save Format

Every training script saves checkpoints containing:
- `model_state` — model parameters
- `optimizer_state` — optimizer state (AdamW moments, Muon states)
- `config` — `ModelConfig` serialization
- `step` — current training step
- `loss` — current loss value
- `recipe` — `TrainingRecipe` in JSON form

### 18B. Resuming Training

```bash
# Resume from a specific checkpoint file
python train_pretrain.py \
    --model-size 0.3B \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints \
    --resume ./checkpoints/step_00050000.pt

# Resume from a directory (auto-finds latest step)
python train_pretrain.py \
    --model-size 0.3B \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints \
    --resume ./checkpoints

# Resume SFT
python train_sft.py \
    --checkpoint-dir ./checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --resume ./sft_checkpoints/sft_step_0005000.pt

# Resume GRPO
python train_grpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./grpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./grpo_checkpoints \
    --resume ./grpo_checkpoints/grpo_step_0000200.pt

# Resume DPO
python train_dpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./dpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./dpo_checkpoints \
    --resume ./dpo_checkpoints/dpo_step_0000200.pt

# Resume DeepSpeed (from DeepSpeed checkpoint directory)
deepspeed train_pretrain_deepspeed.py \
    --model-size 0.6B \
    --data-dir ./packed \
    --out-dir ./checkpoints_ds \
    --resume ./checkpoints_ds/global_step00050000

# Resume Hivemind (each peer resumes from its own directory)
bash hivemind/run.sh worker 192.168.1.100:5678 \
    --model-size 300M \
    --data-dir ./packed \
    --checkpoint-dir ./hivemind_ckpts \
    --resume ./hivemind_ckpts/step_00000050.pt
```

### 18C. Checkpoint Management

```bash
# Keep only the 5 most recent checkpoints
python train_pretrain.py \
    --model-size 0.3B \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints \
    --keep-ckpts 5 \
    --save-every 2000

# Save infrequently for long training runs
python train_pretrain.py \
    --model-size 1.7B \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints \
    --save-every 10000 \
    --keep-ckpts 2
```

---

## 19. Model Architecture Reference

### 19A. ModelConfig Fields

```python
config = ModelConfig(
    vocab_size=65536,                # Vocabulary size
    hidden_size=2048,                # Hidden dimension
    intermediate_size=6144,          # MLP intermediate dimension
    num_hidden_layers=28,            # Number of decoder layers
    num_attention_heads=16,          # Number of attention heads
    num_key_value_heads=4,           # KV heads (GQA)
    head_dim=128,                    # Head dimension
    max_position_embeddings=8192,    # Max sequence length
    rms_norm_eps=1e-6,               # RMSNorm epsilon
    rope_theta=1000000.0,            # RoPE base frequency
    rope_scaling=None,               # RoPE scaling (YaRN, NTK)
    tie_word_embeddings=True,        # Tie LM head ↔ embeddings
    scale_emb=True,                  # Scale embeddings by 1/√d
    norm_type="rmsnorm",             # rmsnorm / layernorm
    mlp_type="swiglu",               # swiglu / gelu
    use_qk_norm=True,                # QK RMSNorm before RoPE
    attn_type="gqa",                 # gqa / mha
    attention_dropout=0.0,           # Attention dropout rate
    hidden_dropout=0.0,              # Hidden dropout rate
    init_std=0.02,                   # Weight init standard deviation
    # Architecture variant fields:
    arch_type="dense",               # dense / jamba
    layer_type="sequential",         # sequential / parallel
    sliding_window_size=0,           # 0 = disabled
    use_mla=False,                   # Multi-head Latent Attention
    kv_lora_rank=None,               # MLA KV compression rank
    num_mtp_heads=0,                 # MTP heads
    mtp_discount=0.5,                # MTP discount factor
    mod_alpha=0.0,                   # MoD threshold (0 = disabled)
    mod_loss_weight=0.01,            # MoD auxiliary loss weight
    jamba_hybrid_layer_interval=4,   # Jamba attention interval
)
```

### 19B. Auto-Sizing (`from_target_size`)

The auto-sizer searches over hidden sizes, layers, and heads to find a configuration closest to the target parameter count. It uses cube-root scaling (params ∝ H³), auto-tiered head_dim, and adaptive layer search to support the full 10M → trillions range.

```python
# ~1.7B param model
config = ModelConfig.from_target_size(target_params=1_700_000_000)
```

### 19C. Sample Model Configurations

| Model Size | Layers | Hidden | Heads | KV Heads | Head Dim | Intermediate | Typical GPU |
|-----------|--------|--------|-------|----------|---------|-------------|-------------|
| 10M | 3 | 576 | 9 | 1 | 64 | 1,472 | CPU / laptop |
| 100M | 12 | 896 | 14 | 1 | 64 | 2,496 | CPU / laptop |
| 300M | 16 | 1,280 | 10 | 2 | 128 | 3,840 | RTX 4090 (24 GB) |
| 600M | 18 | 1,792 | 14 | 2 | 128 | 4,928 | RTX 4090 (24 GB) |
| 1.7B | 25 | 2,624 | 20 | 4 | 128 | 6,592 | A100-40GB / 2× RTX 4090 |
| 8B | 41 | 4,288 | 36 | 2 | 128 | 11,840 | H100 / 4× A100-40GB |
| 70B | 68 | 9,984 | 52 | 4 | 192 | 27,456 | 8× H100 |
| 300B | 102 | 16,128 | 64 | 32 | 256 | 44,352 | Multi-node |
| 1T | 162 | 24,064 | 96 | 16 | 256 | 66,176 | Multi-node |

### 19D. Flash Attention

Uses PyTorch's `F.scaled_dot_product_attention` which automatically leverages FlashAttention-2/3 on compatible GPUs (SM80+).

### 19E. Gradient Checkpointing

```bash
# Pass to any training script
python train_pretrain.py \
    --model-size 1.7B \
    --gradient-checkpointing \
    --data-dir ./packed
```

Saves ~35% VRAM at the cost of ~30% additional compute.

### 19F. MFU Estimation

The framework estimates Model FLOPS Utilization (MFU) using a GPU peak TFLOPS lookup table:

- A100 SXM: 312 TFLOPS (BF16)
- RTX 4090: 165 TFLOPS (BF16)
- H100: 494 TFLOPS (BF16)
- H200: 524 TFLOPS (BF16)

At each log interval, the measured FLOPS is compared to the GPU peak to report MFU%.

### 19G. Z-Loss

Z-loss is an auxiliary loss that penalises large logit magnitudes by adding `z_loss_weight * mean(logits^2)` to the total loss. This improves training stability, especially for large models or long training runs:

```bash
# Disable Z-loss
python train_pretrain.py \
    --model-size 0.3B \
    --z-loss-weight 0 \
    --data-dir ./packed

# Strong Z-loss penalty
python train_pretrain.py \
    --model-size 13B \
    --z-loss-weight 1e-3 \
    --data-dir ./packed
```

---

## 20. End-to-End Pipeline: Train a Capable ~350M Model on an RTX 4090

This section walks through the **complete pipeline** — from nothing to a working model after GRPO and DPO — on a single RTX 4090 (24 GB VRAM) with 28 GB CPU RAM. Every setting is chosen to **safely fit in memory** with headroom.

### 20A. Hardware & Design Constraints

| Constraint | Value | How We Stay Within It |
|-----------|-------|----------------------|
| VRAM | 24 GB | ~350M param model (bf16), LoRA for SFT/GRPO/DPO, gradient checkpointing, batch_size=2-4 |
| CPU RAM | 28 GB | Small dataset budgets, `--target-size 256MB`, short sequences (2048 tok) |
| Disk | Any | ~3 GB total for checkpoints, data, and tokenizer |

**Model architecture** (auto-sized by the framework):

```
ModelConfig(
    vocab_size=16384,       # compact vocabulary
    d_model=1024,           # hidden dimension
    n_layers=16,            # transformer layers
    n_heads=16,             # attention heads
    head_dim=64,            # d_model / n_heads
    intermediate_size=2816, # 2.75× d_model (SwiGLU)
    max_seq_len=2048,       # short context = less activation memory
    use_flash=True,         # Flash Attention 2 saves VRAM
)
```

**Total parameter count**: ~350M → ~700 MB in bf16, ~6 GB training footprint including optimizer and activations.

### 20B. Quick Start (a synthetic smoke test, 10 minutes)

If you want to verify the pipeline works end-to-end before committing to real data, use the built-in smoke test — it creates a tiny model + synthetic data:

```bash
# Tokenizer smoke test
python tokenizer_train.py --smoke-test

# Pretrain smoke test (creates a tiny 10M model + synthetic data)
python train_pretrain.py --smoke-test

# SFT smoke test
python train_sft.py --smoke-test

# GRPO smoke test
python train_grpo.py --smoke-test

# DPO smoke test
python train_dpo.py --smoke-test

# Inference smoke test
python infer.py --smoke-test
```

Each smoke test runs in seconds and prints `[PASSED]` on success. Once all pass, proceed with the real pipeline below.

### 20C. Step 1 — Train a Compact Tokenizer

```bash
# Collect some raw text first (500 KB is enough for a small tokenizer)
mkdir -p raw_tokens
python -c "
import json, os, random
# Use the codegen pipeline to fetch a tiny sample, or create synthetic data
with open('raw_tokens/sample.txt', 'w') as f:
    for i in range(2000):
        f.write(f'This is sample sentence number {i} for training the tokenizer vocabulary. It contains enough variety to learn byte-pair encodings.\\n')
"

# Train a 16K-vocabulary tokenizer (tiny vocab = smaller embedding table)
python tokenizer_train.py \
    --data-dir ./raw_tokens \
    --output-dir ./tokenizer_16k \
    --vocab-size 16384 \
    --min-frequency 2

# Verify
ls -lh ./tokenizer_16k/tokenizer.json
```

For a real project, point `--data-dir` at a few MB of actual .jsonl text from your domain.

### 20D. Step 2 — Collect & Pack Pretraining Data

Use `hf_to_packed.py` with a small byte budget so it fits in 28 GB RAM:

```bash
# Fetch a small subset of C4 (English), stop at 256 MB of raw JSONL
python webscrapped_dataset_curator_AI_MCP/agent/hf_to_packed.py \
    --dataset c4 --config en \
    --mode pretrain \
    --tokenizer ./tokenizer_16k \
    --out-dir ./packed_pretrain \
    --target-size 256MB \
    --seq-length 2048 \
    --min-doc-chars 200 \
    --max-compression-ratio 0.35 \
    --max-flagged-ngram-ratio 0.10

# After packing, you should see:
#   ./packed_pretrain/train.bin
#   ./packed_pretrain/val.bin
#   ./packed_pretrain/tokenizer.json
ls -lh ./packed_pretrain/
```

> **Memory note:** The `--target-size 256MB` cap keeps CPU RAM usage well under 28 GB during packing. If you have more RAM, increase to `512MB` or `1GB` for better results.

### 20E. Step 2b (Alternative) — Codegen Pipeline for Multi-Source Data

Instead of a single dataset, use the codegen pipeline to discover and mix sources:

```bash
python webscrapped_dataset_curator_AI_MCP/agent/codegen_pipeline.py \
    --target-size 256MB \
    --out-dir ./data_pretrain \
    --mode pretrain \
    --public-only \
    --min-doc-chars 200 \
    --discover-limit 3 \
    --max-candidates-to-try 2 \
    --language "en" \
    --no-extended-quality

# Then pack the collected JSONL shards
python data/pack_pretrain.py \
    --data-dir ./data_pretrain \
    --tokenizer ./tokenizer_16k \
    --cache-dir ./packed_pretrain \
    --seq-length 2048 \
    --val-fraction 0.01
```

### 20F. Step 3 — Pretrain the ~350M Model

Now train the base model. The settings below use **~11 GB VRAM** total, leaving 13 GB headroom:

```bash
python train_pretrain.py \
    --data-dir ./packed_pretrain \
    --tokenizer ./tokenizer_16k \
    --model-size 0.35B \
    --max-seq-len 2048 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --max-steps 5000 \
    --warmup-steps 200 \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --weight-decay 0.1 \
    --grad-clip 1.0 \
    --gradient-checkpointing \
    --bf16 \
    --save-every 1000 \
    --output-dir ./checkpoints_pretrain
```

**VRAM breakdown during pretrain (~10.5 GB):**

| Component | Memory |
|-----------|--------|
| Model weights (bf16) | ~700 MB |
| Optimizer (Adam, fp32) | ~2.1 GB |
| Gradients (bf16) | ~700 MB |
| Activations (checkpointed) | ~4-5 GB |
| CUDA context, buffers | ~2 GB |
| **Total** | **~10.5 GB** |

> **If you run low on VRAM:** lower `--batch-size` to 2, add `--gradient-accumulation-steps 8`, or set `--gradient-checkpointing` (already on above).

**Expected output after 5K steps** (≈30-60 minutes depending on disk I/O):
- Loss should drop from ~11 → ~4-5
- Checkpoint saved at `./checkpoints_pretrain/step_5000/`
- Perplexity ≈ 50-150 on validation

For a more capable model, train to 20K-50K steps overnight.

### 20G. Step 4 — Collect & Pack SFT Data

```bash
# Download a small instruction dataset using hf_to_packed
python webscrapped_dataset_curator_AI_MCP/agent/hf_to_packed.py \
    --dataset databricks/databricks-dolly-15k \
    --mode sft \
    --tokenizer ./tokenizer_16k \
    --out-dir ./packed_sft \
    --seq-length 2048 \
    --min-doc-chars 50 \
    --target-size 50MB \
    --no-extended-quality
```

### 20H. Step 5 — Supervised Fine-Tuning with LoRA

LoRA drastically cuts VRAM by training only low-rank adapters (~0.1% of parameters):

```bash
python train_sft.py \
    --checkpoint ./checkpoints_pretrain/step_5000/model.pt \
    --tokenizer ./tokenizer_16k \
    --cache-dir ./packed_sft \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --max-steps 1000 \
    --warmup-steps 50 \
    --lr 2e-4 \
    --bf16 \
    --gradient-checkpointing \
    --lora-rank 32 \
    --lora-alpha 64 \
    --lora-dropout 0.05 \
    --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --save-every 500 \
    --output-dir ./checkpoints_sft
```

**VRAM breakdown with LoRA (~7 GB):**
- Base model (frozen, bf16): ~700 MB
- LoRA adapters (trainable, fp32): ~10 MB
- Optimizer (only LoRA params): ~40 MB
- Activations (checkpointed): ~5 GB
- **Total**: ~6-7 GB

### 20I. Step 6 — GRPO with LoRA (Reinforcement Learning)

GRPO improves reasoning capabilities. Use the SFT checkpoint as the starting point:

```bash
python train_grpo.py \
    --checkpoint ./checkpoints_sft/step_1000 \
    --tokenizer ./tokenizer_16k \
    --cache-dir ./packed_sft \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --max-steps 500 \
    --lr 1e-4 \
    --bf16 \
    --gradient-checkpointing \
    --lora-rank 32 \
    --lora-alpha 64 \
    --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --grpo-epsilon 0.2 \
    --grpo-kl-coef 0.01 \
    --save-every 250 \
    --output-dir ./checkpoints_grpo
```

> **⚠️ GRPO on a 4090:** GRPO generates K completions per prompt (default K=4), which multiplies activation memory. With `--batch-size 2` and gradient checkpointing, this uses ~14 GB VRAM. If you see OOM, set `--batch-size 1` or reduce the prompt length with `--max-seq-len 1024`.

### 20J. Step 7 — DPO with LoRA (Preference Optimization)

DPO aligns the model with human preferences. First, generate preference pairs, then train:

```bash
# (Optional) Generate preference pairs with Ollama judge
python ollama_judge.py \
    --prompt-file ./eval_prompts.txt \
    --output-dir ./dpo_data \
    --batch-size 4

# Pack DPO data (if you have preference pairs in the right format)
# Otherwise, DPO can also be trained on the SFT dataset with a reference model

python train_dpo.py \
    --checkpoint ./checkpoints_grpo/step_500 \
    --tokenizer ./tokenizer_16k \
    --cache-dir ./packed_sft \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --max-steps 500 \
    --lr 1e-4 \
    --bf16 \
    --gradient-checkpointing \
    --lora-rank 32 \
    --lora-alpha 64 \
    --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --dpo-beta 0.1 \
    --save-every 250 \
    --output-dir ./checkpoints_dpo

# On a 4090, DPO uses ~12 GB VRAM with these settings.
# The LoRA adapters are tiny (~10 MB) so checkpointing is fast.
```

### 20K. Step 8 — Inference & Evaluation

```bash
# Interactive REPL with the final checkpoint (auto-merges LoRA adapters into the base model)
python infer.py \
    --checkpoint ./checkpoints_dpo/step_500 \
    --tokenizer ./tokenizer_16k \
    --max-tokens 512 \
    --temperature 0.7 \
    --top-p 0.9

# One-shot generation
python infer.py \
    --checkpoint ./checkpoints_dpo/step_500 \
    --tokenizer ./tokenizer_16k \
    --prompt "Explain how attention works in transformer models." \
    --max-tokens 512 \
    --temperature 0.7

# Compare before and after training
python infer.py \
    --checkpoint ./checkpoints_pretrain/step_5000 \
    --tokenizer ./tokenizer_16k \
    --prompt "What is 2+2?" \
    --max-tokens 256

python infer.py \
    --checkpoint ./checkpoints_dpo/step_500 \
    --tokenizer ./tokenizer_16k \
    --prompt "What is 2+2?" \
    --max-tokens 256
```

### 20L. Batch Size & Memory Quick Reference

Use this table to pick safe batch sizes for your GPU:

| Model Size | VRAM | Batch Size | Grad Accum | LoRA? | Grad CKPT? | Est. VRAM Use |
|-----------|------|-----------|------------|-------|-----------|--------------|
| ~350M | 24 GB | 4 | 4 | No | Yes | ~10.5 GB |
| ~350M | 24 GB | 2 | 8 | No | Yes | ~8 GB |
| ~350M | 24 GB | 4 | 4 | Yes (SFT) | Yes | ~7 GB |
| ~350M | 24 GB | 1 | 8 | Yes (GRPO) | Yes | ~12 GB |
| ~350M | 24 GB | 2 | 4 | Yes (DPO) | Yes | ~12 GB |
| ~1B | 24 GB | 2 | 4 | No | Yes | ~18 GB |
| ~1B | 24 GB | 2 | 8 | Yes | Yes | ~14 GB |

> **Rule of thumb:** If you hit OOM, halve `--batch-size` and double `--gradient-accumulation-steps` — the effective batch size stays the same, so training quality is unchanged.

### 20M. Full Pipeline Script

Save and run this to execute the entire pipeline in one shot (adjust paths as needed):

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== Pretrain ==="
python train_pretrain.py \
    --data-dir ./packed_pretrain \
    --tokenizer ./tokenizer_16k \
    --model-size 0.35B \
    --max-seq-len 2048 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --max-steps 5000 \
    --warmup-steps 200 \
    --lr 3e-4 \
    --gradient-checkpointing \
    --bf16 \
    --save-every 1000 \
    --output-dir ./checkpoints_pretrain

echo "=== SFT with LoRA ==="
python train_sft.py \
    --checkpoint ./checkpoints_pretrain/step_5000/model.pt \
    --tokenizer ./tokenizer_16k \
    --cache-dir ./packed_sft \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --max-steps 1000 \
    --lr 2e-4 \
    --bf16 \
    --gradient-checkpointing \
    --lora-rank 32 \
    --output-dir ./checkpoints_sft

echo "=== GRPO with LoRA ==="
python train_grpo.py \
    --checkpoint ./checkpoints_sft/step_1000 \
    --tokenizer ./tokenizer_16k \
    --cache-dir ./packed_sft \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --max-steps 500 \
    --lr 1e-4 \
    --bf16 \
    --gradient-checkpointing \
    --lora-rank 32 \
    --output-dir ./checkpoints_grpo

echo "=== DPO with LoRA ==="
python train_dpo.py \
    --checkpoint ./checkpoints_grpo/step_500 \
    --tokenizer ./tokenizer_16k \
    --cache-dir ./packed_sft \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --max-steps 500 \
    --lr 1e-4 \
    --bf16 \
    --gradient-checkpointing \
    --lora-rank 32 \
    --output-dir ./checkpoints_dpo

echo "=== Inference test ==="
python infer.py \
    --checkpoint ./checkpoints_dpo/step_500 \
    --tokenizer ./tokenizer_16k \
    --prompt "Hello, how are you?" \
    --max-tokens 128

echo "=== Done ==="
```

> **Expected runtime:** ~2-4 hours total on a 4090 (pretrain: 30-60 min, SFT: 15-30 min, GRPO: 30-60 min, DPO: 15-30 min). Let pretrain run overnight for a noticeably more capable model.

---

## Appendix A: Full CLI Reference

Every script has a `--help` flag for the complete list of arguments:

```bash
python model.py --help                      # Model configuration (read source)
python tokenizer_train.py --help             # Tokenizer training
python train_pretrain.py --help              # Pretrain (DDP)
python train_pretrain_deepspeed.py --help    # Pretrain (DeepSpeed)
python train_sft.py --help                   # SFT
python train_sft_deepspeed.py --help         # SFT (DeepSpeed)
python train_grpo.py --help                  # GRPO
python train_grpo_deepspeed.py --help        # GRPO (DeepSpeed)
python train_dpo.py --help                   # DPO
python train_dpo_deepspeed.py --help         # DPO (DeepSpeed)
python infer.py --help                       # Inference
python ollama_judge.py --help                # DPO preference generation
python data/pack_pretrain.py --help          # Pretrain data packing
python data/pack_sft.py --help               # SFT data packing
python data/pack_grpo.py --help              # GRPO data packing
python data/pack_dpo.py --help               # DPO data packing
python webscrapped_dataset_curator_AI_MCP/agent/codegen_pipeline.py --help
python webscrapped_dataset_curator_AI_MCP/agent/hf_to_packed.py --help
python webscrapped_dataset_curator_AI_MCP/agent/dataset_agent.py --help
python hivemind/train_pretrain_hivemind.py --help
python hivemind/train_sft_hivemind.py --help
python hivemind/train_grpo_hivemind.py --help
python hivemind/train_dpo_hivemind.py --help
```

### Shared Architecture Arguments (all training scripts)

```
--arch {dense,jamba}
--layer-type {sequential,parallel}
--sliding-window-size N
--num-mtp-heads N
--mtp-discount F
--mod-alpha F
--mod-loss-weight F
--use-mla
--kv-lora-rank N
--jamba-interval N
```

### Shared Recipe Arguments (packing + training scripts)

```
--recipe PATH
--mode {reasoning,non_reasoning,hybrid}
```

### Shared Hivemind Arguments (hivemind training scripts)

```
--hivemind
--initial-peers "ip:port"
--host 0.0.0.0
--port N
--peer-id NAME
--target-group-size N
--averaging-period N
--average-parameters / --no-average-parameters
--checkpoint-average-rounds N
```

---

> **Pro tip:** Every training script auto-saves its `ModelConfig`, `TrainingRecipe`, and hyperparameters alongside checkpoints. You never lose track of what produced a given `.pt` file — just inspect the checkpoint metadata.

> For architecture-specific help: `python -c "from model import add_architecture_args; help(add_architecture_args)"`
