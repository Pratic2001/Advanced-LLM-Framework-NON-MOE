# Advanced LLM Framework — Complete User Manual

> **Dense, Non-MoE Transformer** with a full three-stage training pipeline
> (pretrain → SFT → GRPO) and an integrated data curation agent.

![Pipeline Overview](https://via.placeholder.com/800x150?text=Pretrain+->+SFT+->+GRPO+Post-Training+Pipeline)

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Architecture Overview](#2-architecture-overview)
3. [Recipe System](#3-recipe-system)
4. [Training a Tokenizer](#4-training-a-tokenizer)
5. [Data Pipeline (Curating Datasets)](#5-data-pipeline)
   - 5.1  Using the Data Curation Agent
   - 5.2  Public Dataset Sources (HF / Kaggle)
   - 5.3  Codegen Pipeline (Simpler Alternative)
   - 5.4  Web Scraping MCP Server
6. [Packing Data for Training](#6-packing-data-for-training)
   - 6.1  Pretrain Packing
   - 6.2  SFT Packing
   - 6.3  GRPO Packing
7. [Training: Pretraining](#7-training-pretraining)
   - 7.1  Torch DDP (train_pretrain.py)
   - 7.2  DeepSpeed (train_pretrain_deepspeed.py)
8. [Training: Supervised Fine-Tuning (SFT)](#8-training-sft)
   - 8.1  Torch DDP (train_sft.py)
   - 8.2  DeepSpeed (train_sft_deepspeed.py)
9. [Training: GRPO Reinforcement Learning](#9-training-grpo)
   - 9.1  Torch DDP (train_grpo.py)
   - 9.2  DeepSpeed (train_grpo_deepspeed.py)
10. [Inference](#10-inference)
11. [Model Architecture](#11-model-architecture)
12. [Troubleshooting & FAQ](#12-troubleshooting--faq)

---

## 1. Quick Start

```bash
# 1. Train a tokenizer
python tokenizer_train.py --data-dir ./data --output-dir ./tokenizer --mode reasoning

# 2. Curate a dataset (200 MB of pretraining data)
python webscrapped_dataset_curator_AI_MCP/agent/dataset_agent.py \
    --target-size 200MB --categories web,knowledge,math \
    --out-dir ./data --mode pretrain --concurrency 5

# 3. Pack it
python data/pack_pretrain.py --data-dir ./data --tokenizer ./tokenizer --cache-dir ./packed

# 4. Pretrain (single GPU, 0.3B model)
python train_pretrain.py --model-size 0.3B --data-dir ./packed --checkpoint-dir ./checkpoints

# 5. Generate text
python infer.py --checkpoint ./checkpoints/latest_checkpoint --prompt "Hello, world!"
```

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      TrainingRecipe                          │
│  (mode, chat template, special tokens, model_name)           │
└─────┬──────────────────────┬──────────────────┬──────────────┘
      │                      │                  │
      ▼                      ▼                  ▼
┌──────────┐        ┌──────────────┐    ┌──────────────┐
│ Tokenizer│───────▶│  Pack Data   │───▶│  Train       │
│ Trainer  │        │ pretrain/sft │    │ DDP / DS     │
│          │        │ /grpo        │    │              │
└──────────┘        └──────────────┘    └──────┬───────┘
                                               │
                                               ▼
┌──────────┐                            ┌──────────────┐
│  Data    │◀───────────────────────────│    Infer     │
│  Agent   │                            │  (Load ckpt) │
└──────────┘                            └──────────────┘
```

### Directory Layout

```
.
├── recipe.py                          # TrainingRecipe — single source of truth
├── model.py                           # Dense transformer (ModelConfig + TransformerForCausalLM)
├── tokenizer_train.py                 # BBPE tokenizer trainer
│
├── train_pretrain.py                  # Pretrain — torch DDP
├── train_pretrain_deepspeed.py        # Pretrain — DeepSpeed
├── train_sft.py                       # SFT — torch DDP (+ LoRA/DoRA)
├── train_sft_deepspeed.py             # SFT — DeepSpeed
├── train_grpo.py                      # GRPO RL — torch DDP
├── train_grpo_deepspeed.py            # GRPO RL — DeepSpeed
├── infer.py                           # Inference (quant, streaming, REPL)
│
├── model.py                           # Model definitions
├── optim/
│   ├── build_optimizer.py             # AdamW, FusedAdam, Muon
│   └── lr_schedule.py                 # Cosine, WSD schedules
├── peft/
│   └── lora.py                        # LoRA / DoRA / rsLoRA
├── data/
│   ├── pack_pretrain.py               # Pack pretrain JSONL → .bin
│   ├── pack_sft.py                    # Pack SFT JSONL → .bin + mask
│   └── pack_grpo.py                   # Pack GRPO prompts → .bin + answers
├── configs/                           # Training config files
├── tests/
│   └── test_model.py                  # Forward/backward smoke tests
│
└── webscrapped_dataset_curator_AI_MCP/
    ├── README.md
    ├── agent/
    │   ├── dataset_agent.py           # Self-directed curation agent
    │   ├── codegen_pipeline.py        # Alternative simpler pipeline
    │   ├── public_sources.py          # HF Hub / Kaggle connectors
    │   ├── quality.py                 # Filters, dedup, shard writer
    │   └── topics.py                  # Topic seeds per category
    └── web_scraper_mcp/
        ├── server.py                  # FastMCP server (tools: search, fetch, extract…)
        ├── extractors.py              # Multi-format extractors
        ├── crawl4ai_backend.py        # Headless browser backend
        └── net_utils.py               # Retry, proxy, user-agent pool
```

---

## 3. Recipe System

> **`recipe.py`** is the single source of truth that every script imports.
> Instead of hardcoding template strings, special tokens, or model names
> in each script, define them once in a `TrainingRecipe`.

### Three Training Modes

Mode | `<think>` tags? | Use case
---|---|---
`reasoning` | Required | Math, code, logic — model must show chain-of-thought
`non_reasoning` | Forbidden | General chat, instruction following, creative writing
`hybrid` | Per-example | Mix of reasoning and non-reasoning examples in same training run

### Creating a Recipe

```python
from recipe import TrainingRecipe

# Default — reasoning mode
recipe = TrainingRecipe()

# Explicit
recipe = TrainingRecipe(
    mode="hybrid",
    turn_prefix_user="<|im_start|>user\n",
    turn_suffix_user="<|im_end|>\n",
)

print(recipe.special_tokens)
# ['<|endoftext|>', '<|pad|>', '<|im_start|>', '<|im_end|>', '<think>', '</think>', '<|think_on|>', '<|think_off|>']
```

### Sample Recipe File

A reference `recipe.json` is provided at the project root:

```json
{
    "comment": "Sample TrainingRecipe for the Advanced LLM Framework",
    "description": "Hybrid mode recipe with ChatML template.",

    "mode": "hybrid",

    "chat_template": "chatml",

    "turn_prefix_user": "<|im_start|>user\n",
    "turn_suffix_user": "<|im_end|>\n",

    "turn_prefix_assistant": "<|im_start|>assistant\n",
    "turn_suffix_assistant": "<|im_end|>\n",

    "think_open": "<think>",
    "think_close": "</think>",

    "hybrid_think_token": "<|think_on|>",
    "hybrid_nothink_token": "<|think_off|>",

    "base_special_tokens": [
        "<|endoftext|>",
        "<|pad|>",
        "<|im_start|>",
        "<|im_end|>"
    ],

    "model_name": "DenseLLM"
}
```

Copy and edit this file to create your own recipe variants for different
training runs.

### Saving / Loading

```bash
python train_sft.py --recipe ./recipe.json ...   # loads saved recipe
python train_sft.py --mode hybrid ...             # OR: infer from --mode
```

Recipes are auto-saved alongside every checkpoint. Inference auto-loads them:

```bash
python infer.py --checkpoint ./sft_checkpoints/latest.pt
# 👆 automatically loads ./sft_checkpoints/recipe.json
```

### Formatting Conversations

```python
# Format a full conversation as a training string
text = recipe.format_full_conversation(
    prompt="What is 2+2?",
    thinking="2 plus 2 equals 4",
    answer="4",
    system="You are a math tutor.",
)
# → "<|im_start|>system\nYou are a math tutor.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n2 plus 2 equals 4\n</think>\n4<|im_end|>\n"
```

---

## 4. Training a Tokenizer

The framework uses a **Byte-level BPE** tokenizer that reads from the same
JSONL files your data agent produces.

```bash
# Basic usage
python tokenizer_train.py --data-dir ./data --output-dir ./tokenizer

# With a recipe (defines special tokens)
python tokenizer_train.py --data-dir ./data --recipe ./recipe.json --vocab-size 131072

# With a mode (uses default special tokens for that mode)
python tokenizer_train.py --data-dir ./data --mode reasoning
```

### Input Format

Reads **all `.jsonl` files** recursively under `--data-dir`. Each line must
have either a `text` field (pretrain) or `prompt` + optionally `answer` (SFT):

```jsonl
{"text": "The quick brown fox jumps over the lazy dog."}
{"prompt": "What is 2+2?", "answer": "4"}
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `./data` | Directory of .jsonl files |
| `--output-dir` | `./tokenizer` | Where to save `tokenizer.json` |
| `--vocab-size` | `65536` | Target vocabulary size |
| `--min-frequency` | `2` | Minimum token frequency |
| `--recipe` | `None` | Path to recipe.json for special tokens |
| `--mode` | `None` | Fallback training mode |

### Loading for Later Use

```python
from tokenizer_train import load_tokenizer
tokenizer = load_tokenizer("./tokenizer")  # loads tokenizer.json
ids = tokenizer.encode("Hello!").ids       # → list of ints
text = tokenizer.decode(ids)               # → "Hello!"
```

---

## 5. Data Pipeline

The data pipeline has **two parallel paths** — you can use either or both:

- **`dataset_agent.py`** — self-directed async agent with LLM planning + judging
- **`codegen_pipeline.py`** — simpler alternative: generates standalone Python scripts

Both produce the **same JSONL shard format** that the packers read.

### 5.1 Using the Data Curation Agent (`dataset_agent.py`)

The agent uses:
1. An **Ollama model** (local) to plan search queries and judge quality
2. An **MCP server** (`server.py`) with DuckDuckGo search + multi-format extraction
3. Optionally **HuggingFace Hub / Kaggle** public datasets

#### 🔧 Setup

```bash
# Start Ollama
ollama serve

# Pull a model for planning + judging
ollama pull llama3.1
# or: ollama pull qwen2.5:7b-instruct

# Run the MCP server (or let dataset_agent.py auto-start it)
cd webscrapped_dataset_curator_AI_MCP
pip install httpx mcp duckduckgo-search
python web_scraper_mcp/server.py
```

#### Basic Usage

```bash
# Curate 500 MB of pretraining data across 3 categories
python agent/dataset_agent.py \
    --target-size 500MB \
    --categories web,knowledge,math \
    --out-dir ./data \
    --mode pretrain \
    --concurrency 8
```

#### SFT Mode

```bash
# Curate 300 MB of SFT data (prompt + thinking + answer triples)
python agent/dataset_agent.py \
    --target-size 300MB \
    --categories math,code,reasoning \
    --out-dir ./sft_data \
    --mode sft
```

#### GRPO Mode

```bash
# Curate 100 MB of GRPO data (prompt + answer pairs)
python agent/dataset_agent.py \
    --target-size 100MB \
    --categories math \
    --out-dir ./grpo_data \
    --mode grpo
```

#### Resuming a Run

```bash
# Just re-run the same command — the agent resumes automatically
python agent/dataset_agent.py --target-size 500MB --categories web --out-dir ./data
```

`RunState` per category persists to `.run_state_<category>.json`. Used queries
and seen URLs are saved so the next iteration generates new queries.

#### Important Flags

| Flag | Default | Description |
|---|---|---|
| `--target-size` | **(required)** | Target dataset size: `500MB`, `2GB`, etc. |
| `--categories` | `web,knowledge,reasoning,code,math,science` | Categories to curate |
| `--out-dir` | `./data` | Output directory for JSONL shards |
| `--mode` | `pretrain` | `pretrain` / `sft` / `grpo` |
| `--concurrency` | `5` | Concurrent URL extractions |
| `--public-sources` | `""` | enable: `huggingface,kaggle` |
| `--public-only` | `false` | Skip web scraping (public datasets only) |
| `--no-llm-judge` | `false` | Disable Ollama quality judge (faster) |
| `--mix` | `equal` | Budget split: `web=0.2,knowledge=0.3,code=0.5` |
| `--category-concurrency` | `2` | Max categories running simultaneously |

### 5.2 Public Dataset Sources (HF / Kaggle)

Topping up from existing public datasets is **faster and more reliable** than
scraping the open web — no robots.txt, no rate limiting, no HTML boilerplate.

#### Auto-Discovery Mode

```bash
# Let the agent find datasets matching each category
python agent/dataset_agent.py \
    --target-size 1GB --mode pretrain \
    --categories web,knowledge,code,math \
    --out-dir ./data \
    --public-sources huggingface,kaggle
```

The agent searches HuggingFace datasets and Kaggle using each category's topic
keywords from `agent/topics.py`. Datasets whose columns don't match known
patterns (text/prompt/answer/code/conversation) are auto-rejected.

#### Named Datasets

```bash
# Pin specific datasets per category
export KAGGLE_USERNAME=you KAGGLE_KEY=xxxx
export HF_TOKEN=hf_your_token  # for gated datasets

python agent/dataset_agent.py \
    --target-size 500MB --mode sft \
    --categories math,code \
    --public-sources huggingface,kaggle \
    --hf-datasets "math=openai/gsm8k;code=codeparrot/apps" \
    --kaggle-datasets "code=owner/some-code-qa-dataset"
```

#### Public-Only Mode

```bash
# Skip scraping entirely — public datasets only
python agent/dataset_agent.py \
    --target-size 2GB --mode pretrain \
    --categories knowledge,science \
    --public-sources huggingface \
    --public-only
```

#### Column Mapping (Auto)

When a dataset is streamed, the agent calls Ollama once to map its columns
to the target schema. For example, a dataset with `input` and `output` columns
gets mapped to `prompt` and `answer`. This happens once per dataset, not per row.

#### Credentials

| Source | Required |
|---|---|
| HuggingFace (public) | None |
| HuggingFace (gated) | `HF_TOKEN` env var |
| Kaggle | `KAGGLE_USERNAME` + `KAGGLE_KEY` or `~/.kaggle/kaggle.json` |

Missing credentials for one backend won't break the run — that backend is
silently skipped with a log warning.

### 5.3 Codegen Pipeline (`codegen_pipeline.py`)

An **alternative** to `dataset_agent.py` when you prefer a simpler
discover→codegen→run approach instead of per-row async orchestration.

**How it works:**

```
  PUBLIC DATASETS               LIVE WEB CRAWL
  ──────────────                ─────────────
  1. discover (HF search)       1. crawl-raw batch via scraper
  2. sample first N rows        2. show Ollama samples
  3. codegen: Ollama writes     → Ollama writes a script that
     a standalone script that      reads raw JSONL, filters,
     streams the full dataset,     dedups, and writes shards
     maps columns, filters,     3. validate + run
     dedups, writes shards
  4. validate + run
```

#### Usage

```bash
# Public datasets only
python agent/codegen_pipeline.py \
    --target-size 500MB --public-only \
    --categories web,knowledge,math \
    --out-dir ./data --mode pretrain

# With budget tuning
python agent/codegen_pipeline.py \
    --target-size 5GB --public-only \
    --categories web,math \
    --discover-limit 20 --max-candidates-to-try 8

# Live web crawl (default)
python agent/codegen_pipeline.py \
    --target-size 200MB \
    --categories web,knowledge --out-dir ./data
```

Each generated script is saved to `./data/_generated_scripts/` and its
full Ollama transcript to `./data/_logs/` for debugging.

### 5.4 Web Scraping MCP Server (`server.py`)

The MCP server provides tools that both `dataset_agent.py` and
`codegen_pipeline.py` use under the hood. You can also use it directly.

#### Starting the Server

```bash
cd webscrapped_dataset_curator_AI_MCP

# Default (auto HTMX/crawl4ai backend)
python web_scraper_mcp/server.py

# Force httpx-only (lighter, no headless browser)
SCRAPER_HTML_BACKEND=httpx python web_scraper_mcp/server.py
```

#### Tools

| Tool | Description |
|---|---|
| `web_search(query, max_results)` | DuckDuckGo search |
| `fetch_page(url)` | Fetch HTML text |
| `fetch_binary(url)` | Fetch raw bytes |
| `extract_content(url)` | Auto-detect & extract (HTML/PDF/DOCX/PPTX/XLSX/image/video/audio) |
| `deep_crawl(seed, max_pages, max_depth)` | BFS crawl with link extraction |
| `transcribe_media(url)` | Video/audio → transcript |
| `healthcheck()` | Check which format backends are available |

#### Supported Formats

```
┌──────────────────────────────────────────────────────────┐
│                    extract_content(url)                   │
│                                                          │
│  url ──► detect_content_kind(url, content_type)          │
│           │                                              │
│           ├── html ───► trafilatura → readability fallback│
│           │            OR crawl4ai (Playwright)          │
│           ├── pdf ────► pdfplumber → OCR fallback        │
│           ├── docx ───► python-docx (+ tables)           │
│           ├── pptx ───► python-pptx (+ speaker notes)    │
│           ├── xlsx ───► openpyxl                          │
│           ├── csv ────► raw decode                       │
│           ├── image ──► pytesseract OCR                  │
│           ├── video ──► yt-dlp captions → whisper ASR    │
│           └── audio ──► faster-whisper transcription     │
│                                                          │
│  Returns: {title, text, author, date, content_type,      │
│            url, error, extra}                            │
└──────────────────────────────────────────────────────────┘
```

#### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SCRAPER_HTML_BACKEND` | `auto` | `auto` / `crawl4ai` / `httpx` |
| `SCRAPER_ALLOWED_DOMAINS` | `""` | Comma-separated allow list |
| `SCRAPER_BLOCKED_DOMAINS` | `""` | Comma-separated block list |
| `SCRAPER_MIN_HOST_INTERVAL` | `2.0` | Seconds between same-host requests |
| `SCRAPER_EXTRACT_WORKERS` | CPUs | Pool size for CPU-bound extraction |
| `SCRAPER_MEDIA_WORKERS` | CPUs/2 | Pool size for ASR transcription |
| `PROXY_LIST` | `""` | Comma-separated proxy URLs for rotation |
| `WHISPER_MODEL` | `base` | Whisper model size (tiny/base/small/medium/large-v3) |

---

## 6. Packing Data for Training

Each training script reads **packed memmap `.bin` files**, not raw JSONL.
Run the appropriate packer first.

### 6.1 Pretrain Packing (`pack_pretrain.py`)

Converts JSONL with `text` fields into uint16 `.bin` files.

```bash
# Basic
python data/pack_pretrain.py \
    --data-dir ./data \
    --tokenizer ./tokenizer \
    --cache-dir ./packed

# With validation split (last 1% of records)
python data/pack_pretrain.py \
    --data-dir ./data \
    --tokenizer ./tokenizer \
    --cache-dir ./packed \
    --val-fraction 0.01

# Multi-worker (4 parallel processes)
python data/pack_pretrain.py --data-dir ./data --tokenizer ./tokenizer --worker 0 --num-workers 4 &
python data/pack_pretrain.py --data-dir ./data --tokenizer ./tokenizer --worker 1 --num-workers 4 &
python data/pack_pretrain.py --data-dir ./data --tokenizer ./tokenizer --worker 2 --num-workers 4 &
python data/pack_pretrain.py --data-dir ./data --tokenizer ./tokenizer --worker 3 --num-workers 4 &
wait
```

**Output in `--cache-dir`:**
- `pretrain_tokens_train.bin` — training tokens
- `pretrain_tokens_val.bin` — validation tokens (if `--val-fraction > 0`)
- `meta_train.json` / `meta_val.json`

**Input JSONL format:**
```jsonl
{"text": "The quick brown fox jumps over the lazy dog."}
{"text": "This is another document that will be tokenized."}
```

### 6.2 SFT Packing (`pack_sft.py`)

Converts JSONL with `prompt` + `answer` (optionally `thinking`) into
token `.bin` + loss-mask `.bin` pairs.

```bash
# Basic
python data/pack_sft.py \
    --data-dir ./sft_data \
    --tokenizer ./tokenizer \
    --cache-dir ./sft_packed

# With recipe mode awareness
python data/pack_sft.py \
    --data-dir ./sft_data \
    --tokenizer ./tokenizer \
    --cache-dir ./sft_packed \
    --mode reasoning

# Validation split
python data/pack_sft.py \
    --data-dir ./sft_data \
    --tokenizer ./tokenizer \
    --cache-dir ./sft_packed \
    --val-fraction 0.05
```

**Output in `--cache-dir`:**
- `sft_train_tokens.bin` — uint16 token ids
- `sft_train_mask.bin` — uint8 loss mask (1 = train, 0 = ignore)
- `sft_train_manifest.json`

**Input JSONL format:**
```jsonl
{"prompt": "What is 2+2?", "thinking": "Let's calculate... 2+2=4", "answer": "4"}
{"prompt": "Explain gravity", "answer": "Gravity is a force...", "want_thinking": false}
```

Each record is formatted using the `TrainingRecipe`:
```
user_turn (mask=0) + assistant_turn (mask=1) + EOS (mask=0)
```

### 6.3 GRPO Packing (`pack_grpo.py`)

Converts JSONL with `prompt` + `answer` into length-prefixed uint32 `.bin`
prompt files + JSON answer sidecar files.

```bash
# Basic
python data/pack_grpo.py \
    --data-dir ./grpo_data \
    --tokenizer ./tokenizer \
    --cache-dir ./grpo_packed

# Reasoning mode (includes <think> tags in recipe awareness)
python data/pack_grpo.py \
    --data-dir ./grpo_data \
    --tokenizer ./tokenizer \
    --cache-dir ./grpo_packed \
    --mode reasoning
```

**Output in `--cache-dir`:**
- `grpo_prompt_tokens.bin` — uint32 length-prefixed prompt tokens
- `grpo_answers.json` — JSON array of answer strings (for reward computation)
- `grpo_manifest.json`

**Input JSONL format:**
```jsonl
{"prompt": "Solve: 2+2", "thinking": "2+2=4", "answer": "4"}
{"prompt": "What is the capital of France?", "answer": "Paris"}
```

The prompt is tokenized as `user_turn + assistant_prefix + EOS`. The model
generates the assistant answer during GRPO training.

---

## 7. Training: Pretraining

Two variants: **torch DDP** (simpler, good for single/multi-GPU) and
**DeepSpeed** (ZeRO auto-selection, CPU offload for larger models).

### 7.1 Torch DDP (`train_pretrain.py`)

#### Single GPU

```bash
# 0.3B model (fits RTX 4090)
python train_pretrain.py \
    --model-size 0.3B \
    --data-dir ./packed \
    --checkpoint-dir ./checkpoints \
    --seq-len 2048 \
    --batch-size 32 \
    --grad-accum 4 \
    --jit

# 1.7B model (fits on 24 GB GPU with gradient checkpointing)
python train_pretrain.py \
    --model-size 1.7B \
    --data-dir ./packed \
    --gradient-checkpointing \
    --batch-size 4 --grad-accum 8 \
    --jit
```

#### Multi-GPU (torchrun)

```bash
# 4 GPUs
torchrun --nproc_per_node=4 train_pretrain.py \
    --model-size 1.7B \
    --data-dir ./packed \
    --batch-size 8 --grad-accum 4 \
    --jit
```

#### Resume from Checkpoint

```bash
# Resume from directory (finds latest step)
python train_pretrain.py \
    --resume ./checkpoints \
    --data-dir ./packed

# Resume from specific step
python train_pretrain.py \
    --resume ./checkpoints/step_10000.pt \
    --data-dir ./packed
```

#### Advanced: Muon Optimizer

```bash
python train_pretrain.py \
    --model-size 0.6B \
    --data-dir ./packed \
    --optimizer muon \
    --lr 1e-4 \
    --jit
```

Muon (from the Moonlight paper) often trains 2-3× faster than AdamW for
the same compute budget. It uses two optimizers internally: Muon for
embedding-free parameters and AdamW for embeddings/norms.

#### Advanced: WSD Schedule

```bash
python train_pretrain.py \
    --model-size 0.3B \
    --data-dir ./packed \
    --schedule wsd \
    --stable-ratio 0.8 \
    --lr 3e-4 \
    --jit
```

WSD (Warmup-Stable-Decay) keeps LR constant during the stable phase, then
decays at the end — often yields better final loss than cosine.

#### Logging with W&B

```bash
python train_pretrain.py \
    --model-size 0.3B \
    --data-dir ./packed \
    --wandb-project my-project \
    --wandb-run-name dense-0.3B-v1
```

#### VRAM Estimation

At startup the script prints a VRAM estimate and suggests adjustments:

```
VRAM         : 24.0 GB total
  static     : ~5.2 GB  (weights + grads + Adam)
  activations: ~3.8 GB  (batch=32, seq=2048)
  headroom   : ~15.0 GB
```

If headroom is low, the script warns and suggests a safe batch size.

#### Pretrain Loss (Double-Shift Fix)

> **⚠️ Important:** The data loader produces **pre-shifted targets**:
> `y[t] = data[i+1+t]`. Calling `model(x, labels=y)` would shift again
> internally and produce wrong gradients. The loss is computed externally
> in `pretrain_loss()` to avoid this double shift.

### 7.2 DeepSpeed (`train_pretrain_deepspeed.py`)

#### Launch

```bash
# Single GPU
deepspeed train_pretrain_deepspeed.py \
    --model-size 0.6B \
    --data-dir ./packed

# Multi-GPU (single node)
deepspeed --num_gpus 4 train_pretrain_deepspeed.py \
    --model-size 1.7B \
    --data-dir ./packed

# Multi-node (requires hostfile)
deepspeed --hostfile hostfile.txt train_pretrain_deepspeed.py \
    --model-size 8B \
    --data-dir ./packed
```

#### Automatic ZeRO Stage Selection

The DeepSpeed variant runs a **hardware audit** on first launch and
auto-selects the optimal ZeRO configuration:

```
   ZeRO-1  → shard optimizer states only
              (≥40 GB VRAM per GPU, abundant headroom)

   ZeRO-2  → shard optimizer states + gradients
              (16–40 GB, default for consumer/data-center GPUs)

   ZeRO-3  → shard optimizer states + gradients + params
              (model params > 2× VRAM per GPU)

   CPU offload → optimizer states (and optionally params) moved to CPU RAM
                 (VRAM severely constrained)
```

#### Override Auto-Selection

```bash
# Force ZeRO-3 with CPU offload
deepspeed train_pretrain_deepspeed.py \
    --model-size 4B \
    --zero-stage 3 \
    --cpu-offload-optimizer \
    --data-dir ./packed
```

#### Mixed Precision

Uses `torch.amp.autocast(bf16)` instead of DeepSpeed's native bf16 handler
for consistency with the DDP variant. DeepSpeed is used for ZeRO sharding,
gradient clipping, and optimizer stepping.

---

## 8. Training: Supervised Fine-Tuning (SFT)

### 8.1 Torch DDP (`train_sft.py`)

Supports **full fine-tune**, **LoRA**, and **DoRA** (Weight-Decomposed
Low-Rank Adaptation).

#### Full Fine-Tune (small model)

```bash
python train_sft.py \
    --checkpoint-dir ./pretrained_checkpoints \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints \
    --lr 2e-5 \
    --num-steps 10000
```

#### LoRA Fine-Tune (recommended for 1B+ models on single GPU)

```bash
python train_sft.py \
    --checkpoint-dir ./pretrained_checkpoints \
    --data-dir ./sft_packed \
    --lora-rank 64 \
    --lora-alpha 128 \
    --output-dir ./sft_checkpoints \
    --lr 2e-4
```

#### DoRA Fine-Tune

```bash
python train_sft.py \
    --checkpoint-dir ./pretrained_checkpoints \
    --data-dir ./sft_packed \
    --lora-rank 64 --lora-alpha 128 --lora-type dora \
    --output-dir ./sft_checkpoints
```

#### Train from Scratch (no pretrained checkpoint)

```bash
python train_sft.py \
    --model-size 1.7B \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints
```

#### Resume

```bash
python train_sft.py \
    --resume ./sft_checkpoints/sft_step0005000.pt \
    --data-dir ./sft_packed \
    --output-dir ./sft_checkpoints
```

#### Merge LoRA → Saved Full Model

After LoRA training, merge adapters back into base weights for inference:

```bash
python train_sft.py --merge-and-save \
    --checkpoint-dir ./sft_checkpoints \
    --output-dir ./sft_merged
```

Then use `./sft_merged/merged_model.pt` with `infer.py`.

#### Key Flags

| Flag | Default | Description |
|---|---|---|
| `--lora-rank` | `64` | LoRA rank (0 = full fine-tune) |
| `--lora-type` | `lora` | `lora` or `dora` |
| `--lora-alpha` | `128.0` | LoRA scaling alpha |
| `--lora-target-modules` | `q_proj,k_proj,...` | Which projections to adapt |
| `--use-rslora` | `false` | Rank-stabilized scaling |
| `--lora-lr-ratio` | `1.0` | LR multiplier for LoRA params |
| `--neftune-alpha` | `0.0` | NEFTune noise (0 = disabled) |
| `--compile` | `false` | Enable torch.compile |
| `--ckpt-interval` | `1000` | Checkpoint frequency in steps |

#### Understanding Loss Masking

SFT uses a **loss mask** — only assistant tokens contribute to the gradient:

```
User: "What is 2+2?"      → mask=0  (no loss computed)
Assistant: "<think>\n...\n</think>\n4"  → mask=1 (model learns this)
<EOS>                     → mask=0  (no loss)
```

This is why `pack_sft.py` writes a separate mask `.bin` alongside the token
`.bin`. The `masked_cross_entropy` function applies the mask before computing
the mean.

### 8.2 DeepSpeed (`train_sft_deepspeed.py`)

```bash
deepspeed --num_gpus 4 train_sft_deepspeed.py \
    --checkpoint-dir ./pretrained_checkpoints \
    --data-dir ./sft_packed \
    --lora-rank 64 \
    --output-dir ./sft_checkpoints
```

Same auto-ZeRO logic as the pretrain DeepSpeed variant.

---

## 9. Training: GRPO Reinforcement Learning

GRPO (Group Relative Policy Optimization) is the **second stage** of
post-training. It takes an SFT checkpoint and applies RL to improve
reasoning and correctness.

### 9.1 Torch DDP (`train_grpo.py`)

```bash
python train_grpo.py \
    --checkpoint ./sft_checkpoints/latest.pt \
    --data-dir ./grpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./grpo_checkpoints \
    --num-steps 500
```

#### Smoke Test

```bash
python train_grpo.py --smoke-test
```

This runs a self-contained end-to-end test with a tiny random model,
synthetic data, one rollout batch, reward computation, and loss,
then does a checkpoint round-trip.

#### How GRPO Works (Illustrated)

```
                         ┌────────────────────────┐
                         │    Prompt: "Solve 2+2"  │
                         │    Answer: "4"          │
                         └────────┬───────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │              │
               ┌────▼────┐  ┌────▼────┐   ┌────▼────┐
               │ Rollout │  │ Rollout │   │ Rollout │  ... (G=8)
               │   1     │  │   2     │   │   3     │
               └────┬────┘  └────┬────┘   └────┬────┘
                    │             │              │
               ┌────▼────┐  ┌────▼────┐   ┌────▼────┐
               │"2+2=4"   │  │"It's 4"  │   │"5"      │  ...
               │<think>.. │  │           │   │         │
               └────┬────┘  └────┬────┘   └────┬────┘
                    │             │              │
               ┌────▼────────────▼──────────────▼─────┐
               │  Reward Function (recipe-aware)       │
               │  - tier 1: correct + think format     │
               │  - tier 2: correct but no think tags  │
               │  - tier 3: format + has answer        │
               │  - tier 4: wrong                      │
               └────────────────┬─────────────────────┘
                                │
               ┌────────────────▼─────────────────────┐
               │  Group-normalized advantages          │
               │  r̄_g = mean(rewards in group)          │
               │  A_i = (r_i - r̄_g) / σ_g              │
               └────────────────┬─────────────────────┘
                                │
               ┌────────────────▼─────────────────────┐
               │  GRPO Loss                            │
               │  L = -E[min(ratio·A, clip(ratio)·A)]  │
               │     + λ_KL · KL(π || π_ref)           │
               └──────────────────────────────────────┘
```

#### Reward Function — Three Tiers

| Condition | Reward | Purpose |
|---|---|---|
| Correct answer + balanced think tags | 1.0 | Perfect |
| Correct but no/malformed think tags | 0.5 | Encourages format |
| Wrong but has think + answer | 0.3 | Encourages trying |
| Wrong or truncated | 0.0 | No reward |

The recipe mode controls whether think-format is checked:

| Recipe Mode | Think Check |
|---|---|
| `reasoning` | **Always** required |
| `non_reasoning` | **Never** checked |
| `hybrid` | Only if `want_thinking=True` |

#### Reward Weights

```bash
# Custom reward weights
python train_grpo.py \
    --checkpoint ./sft.pt \
    --data-dir ./grpo_packed \
    --reward-correct 1.5 \
    --reward-format 0.5
```

Answer extraction supports `\boxed{...}` (LaTeX) and last-numeric-fallback:

```
Extraction priority:
  1. \boxed{answer}   → "\\boxed{4}" → 4.0
  2. Last numeric     → "... equals 4.5" → 4.5
  3. Last token       → "Four" → str match
```

#### Reference Policy

```bash
# Single model (default) — reuse trainable model under no_grad
python train_grpo.py --ref-policy single ...

# Two-model — separate frozen copy (more stable but 2× memory)
python train_grpo.py --ref-policy two --checkpoint ./sft.pt ...
```

Two-model is more stable because the reference never drifts from the initial
SFT policy. Single-model is more memory-efficient.

#### Generation Hyperparameters

```bash
python train_grpo.py \
    --num-generations 8 \      # G completions per prompt
    --max-new-tokens 512 \     # Max tokens per completion
    --temperature 1.0 \        # Sampling temp
    --top-p 0.95               # Nucleus sampling
```

#### GRPO Loss Parameters

```bash
python train_grpo.py \
    --kl-coef 0.02 \            # KL penalty strength
    --clip-range 0.2 \          # PPO clipping
    --entropy-coeff 0.01        # Entropy bonus (exploration)
```

### 9.2 DeepSpeed (`train_grpo_deepspeed.py`)

```bash
deepspeed --num_gpus 4 train_grpo_deepspeed.py \
    --checkpoint ./sft.pt \
    --data-dir ./grpo_packed \
    --tokenizer ./tokenizer \
    --out-dir ./grpo_checkpoints \
    --lora
```

---

## 10. Inference

The inference script (`infer.py`) supports one-shot generation, interactive
REPL, batched evaluation, and quantized modes.

### One-Shot Generation

```bash
python infer.py \
    --checkpoint ./checkpoints/latest_checkpoint \
    --prompt "Solve 2+2"
```

### Interactive REPL

```bash
python infer.py \
    --checkpoint ./checkpoints/latest_checkpoint \
    --interactive
```

Type prompts at the `>>>` prompt. Generation streams token-by-token.

### Explicit Recipe

```bash
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --recipe ./recipe.json \
    --prompt "Hello!"
```

### Batched Evaluation

```bash
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --prompts-file ./eval.jsonl \
    --batch-size 8 \
    --output ./results.jsonl
```

Input JSONL format: `{"prompt": "..."}` or `{"prompt": "...", "answer": "..."}`.

### Quantized Inference

```bash
# 4-bit quantization
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --quantize 4bit \
    --prompt "Hello"

# 8-bit quantization
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --quantize 8bit \
    --prompt "Hello"
```

Requires `bitsandbytes` installed. The model loads compressed and runs with
reduced memory (~4× for 4-bit).

### LoRA Checkpoint Inference

```bash
# Infer.py auto-detects LoRA checkpoints and merges on-the-fly
python infer.py \
    --checkpoint ./sft_checkpoints/sft_step0005000_lora.pt \
    --base-checkpoint ./pretrained_checkpoints/latest.pt \
    --prompt "What is 2+2?"
```

### Generation Parameters

```bash
python infer.py \
    --checkpoint ./checkpoints/latest.pt \
    --prompt "Write a poem" \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.95 \
    --repetition-penalty 1.1 \
    --max-new-tokens 256 \
    --min-new-tokens 10
```

### Smoke Test (No Checkpoint Needed)

```bash
python infer.py --smoke-test
```

Creates a tiny random model and runs one generation to verify dependencies.

---

## 11. Model Architecture

The model (`model.py`) is a dense (non-MoE) transformer with these configurable
components:

```
┌──────────────────────────────────────────────────┐
│              TransformerForCausalLM               │
│                                                   │
│  Token Embeddings (scale_emb = 1/√d)             │
│         │                                         │
│  ┌──────┴──────┐                                 │
│  │ Decoder × N │  (each:                         │
│  │   ┌────────┐│   pre-RMSNorm → Self-Attn       │
│  │   │ Self-  ││   → residual                    │
│  │   │ Attn   ││   → pre-RMSNorm → MLP           │
│  │   └────────┘│   → residual)                   │
│  │   ┌────────┐│                                  │
│  │   │ MLP    ││  GQA (key-value groups)          │
│  │   └────────┘│  RoPE (rotary embeddings)        │
│  └──────┬──────┘  QK-Norm (optional)              │
│         │         SwiGLU / GELU                   │
│         │         RMSNorm / LayerNorm             │
│  ┌──────┴──────┐                                 │
│  │  Final Norm │                                  │
│  └──────┬──────┘                                  │
│         │                                         │
│  ┌──────┴──────┐                                 │
│  │  LM Head    │  (tied with embeddings)          │
│  └─────────────┘                                  │
└──────────────────────────────────────────────────┘
```

### ModelConfig Fields

```python
config = ModelConfig(
    vocab_size=65536,           # Vocabulary size
    hidden_size=2048,           # Hidden dimension
    intermediate_size=6144,     # MLP intermediate dimension
    num_hidden_layers=28,       # Decoder layers
    num_attention_heads=16,     # Attention heads
    num_key_value_heads=4,      # KV heads (GQA)
    head_dim=128,               # Head dimension
    max_position_embeddings=8192,  # Max sequence length
    rms_norm_eps=1e-6,          # RMSNorm epsilon
    rope_theta=1000000.0,       # RoPE base frequency
    rope_scaling=None,          # RoPE scaling (NTK-aware, YaRN, etc.)
    tie_word_embeddings=True,   # Tie LM head ↔ embeddings
    scale_emb=True,             # Scale embeddings by 1/√d
    norm_type="rmsnorm",        # rmsnorm / layernorm
    mlp_type="swiglu",          # swiglu / gelu
    use_qk_norm=True,           # QK RMSNorm before RoPE
    attn_type="gqa",            # gqa / mha
    attention_dropout=0.0,      # Attention dropout rate
    hidden_dropout=0.0,         # Hidden dropout rate
    init_std=0.02,              # Weight init standard deviation
)
```

### Auto-Sizing (`from_target_size`)

```python
# Generates a balanced architecture for ~1.7B params
config = ModelConfig.from_target_size(target_params=1_700_000_000)
```

The auto-sizer searches over hidden sizes, layers, and heads to find a
configuration closest to the target parameter count while respecting
constraints (head_dim=128, num_kv_heads = num_attention_heads / 4).

### Supported Configurations

| Model Size | Layers | Hidden | Heads | KV Heads | Intermediate | Typical GPU |
|---|---|---|---|---|---|---|
| 0.3B | 24 | 1024 | 8 | 2 | 3072 | RTX 4090 (24 GB) |
| 0.6B | 24 | 1536 | 12 | 3 | 4608 | RTX 4090 (24 GB) |
| 1.7B | 28 | 2048 | 16 | 4 | 6144 | A100-40GB / 2× RTX 4090 |
| 4B | 32 | 2560 | 20 | 5 | 7680 | A100-80GB / 2× A100-40GB |
| 8B | 36 | 3584 | 28 | 7 | 10752 | H100 / 4× A100-40GB |
| 14B | 40 | 4096 | 32 | 8 | 12288 | 8× A100-40GB |

### Flash Attention

Uses PyTorch's `F.scaled_dot_product_attention` which automatically
leverages FlashAttention-2/3 on compatible GPUs.

### Gradient Checkpointing

```python
model.model.enable_gradient_checkpointing()  # saves ~35% VRAM, costs ~30% compute
```

Pass `--gradient-checkpointing` to the pretrain script.

---

## 12. Troubleshooting & FAQ

### Common Issues

#### `CUDA out of memory`

Solutions (in order of effectiveness):
1. Reduce `--batch-size` (the biggest lever)
2. Enable `--gradient-checkpointing` (saves ~35% activation VRAM)
3. Reduce `--seq-len` (halving sequence length halves activations)
4. Use LoRA (`--lora-rank 64`) for SFT
5. Use DeepSpeed (`train_*_deepspeed.py`) with auto-ZeRO

#### `FileNotFoundError: No ... files found`

Ensure you've run the data pipeline step in order:

```
dataset_agent.py  →  pack_*.py  →  train_*.py
     ↓                    ↓             ↓
  JSONL shards       memmap .bin     model checkpoint
```

Check that `--data-dir` and `--out-dir`/`--cache-dir` point to the correct
directories from the previous step.

#### `Double-shift` or `NaN` loss in pretraining

The `PackedDataLoader` produces pre-shifted targets, and `pretrain_loss()`
computes the cross-entropy directly against those targets. If you see NaN
loss, make sure you're using `pretrain_loss()` and NOT passing `labels` to
`model(x)`.

#### Ollama connection errors in data agent

```bash
# Ensure Ollama is running
ollama serve

# Check which model is available
ollama list

# The agent needs both:
#   OLLAMA_URL    (default: http://localhost:11434)
#   OLLAMA_MODEL  (default: llama3.1)
```

#### `datasets` package errors

If `--public-sources huggingface` is set but `datasets` or
`huggingface_hub` are not installed:

```bash
pip install datasets huggingface_hub
```

Missing deps produce a logged warning, not a crash — that backend is skipped.

### FAQ

**Q: Can I add a new category for web scraping?**

Edit `agent/topics.py` and add an entry to `TOPIC_SEEDS` and
`HUB_SEARCH_KEYWORDS` for your new category.

**Q: How do I use my own training recipe across multiple machines?**

Save `recipe.json` with `recipe.to_json("./recipe.json")`, then pass
`--recipe ./recipe.json` to every training script and to `infer.py`.

**Q: Can I continue an SFT run after changing LoRA rank?**

No — the LoRA adapter dimensions are baked into the checkpoint. Load
the base weights fresh and re-inject with the new rank.

**Q: What happens if I mix sources in `--public-sources`?**

Each source is tried in order. After all public sources are exhausted for
a category, the remaining budget is filled via live web scraping (unless
`--public-only` is set).

**Q: How do I serve the trained model?**

```bash
python infer.py --checkpoint ./checkpoints/latest.pt --interactive
```

For production serving, the `infer.py` script is designed as a reference
implementation you can wrap in a REST API, gRPC server, or MCP server.

---

> **End of Manual** — For CLI-specific help: `python <script> --help`
