# Infinite Data Agent

Live-scraping companion to your `build_dataset.py` / `download_sft_data.py`
pipeline. When your fixed HF sources (FineWeb, the-stack, OpenR1-Math, etc.)
run dry, this agent keeps producing data by searching and scraping the open
web, using a **local Ollama model as the planner + quality judge** and an
**MCP server as the hands** (search/fetch/extract) — across HTML, PDFs,
Office documents, images, and video/audio, not just web articles.

It writes output in the **exact same JSONL shard + manifest format** your
existing scripts already use, so `pack_dataset.py` / `pack_sft_data.py` /
`pack_grpo_data.py` need zero changes — just point them at the same
`--out-dir` (or merge directories) and they'll pack the agent's shards
alongside the HF-sourced ones.

## What's new: multi-format extraction

The original version only understood HTML articles. `extract_content` now
auto-detects and extracts:

| Format | Method |
|---|---|
| HTML articles | `trafilatura`, falls back to `readability-lxml` |
| PDF | `pdfplumber` text layer; **OCR fallback** (`pdf2image` + `pytesseract`) for scanned/image-only pages |
| Word (`.docx`) | `python-docx` — paragraphs + tables |
| PowerPoint (`.pptx`) | `python-pptx` — slide text, tables, speaker notes |
| Excel (`.xlsx`) / CSV | `openpyxl` — sheet contents as flattened text |
| Images (`.png`/`.jpg`/etc.) | OCR via `pytesseract` |
| Video (YouTube/Vimeo/direct files) | **Captions/subtitles first** (fast, accurate, free) via `yt-dlp`; **local ASR fallback** via `faster-whisper` if none exist |
| Audio (`.mp3`/`.wav`/etc.) | `faster-whisper` transcription |

Every extractor returns the same shape — `{title, text, author, date,
content_type, url, error, extra}` — so the agent's filtering/dedup/writing
logic never branches on format. Every heavy dependency is imported lazily,
so missing one optional package (say, `faster-whisper`) degrades that one
format to a clear error instead of breaking the whole server. Run
`healthcheck()` to see which formats are actually usable in your
environment before kicking off a real run.

## Architecture

```
Ollama (planner + judge)  <-->  dataset_agent.py  <-->  MCP server (server.py)
        local model              orchestrator            web_search
                                  + quality.py             fetch_page / fetch_binary
                                  (same filters as         extract_content  ---> extractors.py
                                   build_dataset.py)              |            (pdf/docx/pptx/xlsx/
                                        |                         |             image/video/audio)
                                        |                         v
                                        |                 live web pages, PDFs,
                                        |                 office docs, media
                                        v
                              agent/public_sources.py  ---> Hugging Face Hub (datasets, streaming)
                              (HF/Kaggle top-up,             Kaggle (dataset search + download)
                               runs BEFORE web scraping)
                                        |
                                        v
                              ./data/<category>/*.jsonl
                              + manifest.json
                              (same shape as build_dataset.py output, regardless of which
                               path -- live scrape or public dataset hub -- produced a row)
```

### Public dataset sources (Hugging Face / Kaggle top-up)

When `--public-sources` is set, each category is topped up from Hugging
Face and/or Kaggle **before** falling back to live web search+scraping --
no robots.txt, no rate limiting, no HTML boilerplate to strip, so it's
faster and more reliable than scraping wherever a matching public dataset
already exists.

- **`agent/public_sources.py`** streams rows (`datasets.load_dataset(...,
  streaming=True)` for HF; download + pandas read for Kaggle CSV/JSON/TSV/
  text files) and normalizes each row into the exact same shape
  `extract_content` returns, so every downstream step -- heuristic quality
  filters, the Ollama quality judge, dedup, shard writing -- runs
  identically no matter which path a row came from.
- **The agent's built-in AI (Ollama) does the shaping work**, same as it
  already does for scraped pages: `judge_quality` gates every row, and in
  `--mode sft`, `extract_sft_pair` turns raw rows into `{prompt, answer}`
  pairs. If a dataset already has its own instruction/response-style
  columns (`prompt`/`question`/`instruction` + `answer`/`response`/
  `output`), those are used directly instead of asking the model to invent
  a question -- the dataset author's own labels are trusted over an LLM
  guess.
- If you don't name specific datasets, `public_sources.py` **auto-discovers**
  a few per category by searching each hub with that category's topic
  seeds.
- Requires the optional deps in `requirements.txt`'s "public dataset
  sources" section, and (for Kaggle) `KAGGLE_USERNAME`/`KAGGLE_KEY` env
  vars or `~/.kaggle/kaggle.json`. Hugging Face gated/private datasets need
  `HF_TOKEN` set; public datasets need no credentials at all.
- Missing a dependency or credential doesn't break the run -- that one
  backend just gets skipped with a logged warning, same graceful-
  degradation pattern as the per-format extractors.

1. **`web_scraper_mcp/server.py`** — MCP server with these tools:
   - `web_search(query, max_results)` — DuckDuckGo search, no API key needed.
   - `fetch_page(url)` / `fetch_binary(url)` — raw text / raw bytes fetch.
     Both respect `robots.txt`, per-host rate limiting, and domain
     allow/deny lists, and retry transient failures with exponential backoff
     + jitter.
   - **`extract_content(url)`** — the primary tool. Detects format and
     dispatches to the right extractor. HTML pages go through **crawl4ai**
     first (real headless-browser fetch + pruned markdown), falling back to
     the original `httpx` + `trafilatura`/`readability` path.
   - **`deep_crawl(seed_url, max_pages, max_depth, keywords, same_domain_only)`**
     — BFS-crawls outward from a seed URL with crawl4ai and extracts every
     page it visits in one call.
   - `transcribe_media(url)` — dedicated video/audio → transcript tool.
   - `extract_article(url)` — HTML-only alias kept for backward
     compatibility.
   - `healthcheck()` — reports which optional per-format dependencies are
     importable.
   - Rotates User-Agent per request and can round-robin across proxies via
     `PROXY_LIST`, to reduce single-fingerprint WAF blocks on long runs.

2. **`agent/dataset_agent.py`** — the orchestrator:
   - Asks Ollama for a batch of specific search queries per category,
     seeded from `agent/topics.py`, explicitly avoiding recently used
     queries so coverage keeps growing.
   - Calls `extract_content` for every hit — works uniformly whether the
     hit turned out to be an article, a PDF, a slide deck, or a YouTube
     video.
   - **Processes each round's URLs concurrently** (`--concurrency`, default
     5) instead of one at a time.
   - Runs the **same heuristic filters** as `build_dataset.py` (alpha
     ratio, repetition ratio, code extension/hygiene checks) plus
     format-aware additions.
   - **Near-dedup uses MinHash + LSH** (`datasketch`) when installed and
     falls back to shingle-overlap comparison otherwise.
   - Optionally runs an **LLM quality judge** pass (Ollama). Disable with
     `--no-llm-judge` for raw throughput.
   - **Resumable state**: each category's used queries and seen URLs
     persist to `.run_state.json` and reload on the next invocation.
   - Writes shards with `ShardWriter`, which **resumes shard numbering**
     from whatever's already in `--out-dir`, so repeated runs append rather
     than overwrite.

3. **`agent/topics.py`** — starter topic lists per category. The agent
   expands past these on its own; edit this file to steer initial direction
   or add new categories.

4. **`agent/codegen_pipeline.py`** — simpler alternative to the per-row
   async pipeline. Uses Ollama to generate standalone Python extraction
   scripts per dataset (one call per dataset, not per row), then runs them
   as subprocesses. Supports both public datasets and live web crawl paths.

## Setup

```bash
cd webscrapped_dataset_curator_AI_MCP

# Core dependencies
pip install httpx mcp ddgs

# Optional per-format dependencies
pip install trafilatura readability-lxml   # HTML extraction
pip install pdfplumber                      # PDF
pip install python-docx python-pptx openpyxl # Office docs
pip install pillow pytesseract              # Image OCR
pip install yt-dlp                          # Video captions
pip install faster-whisper                  # Audio/video ASR
pip install datasketch                      # Near-dedup (MinHash LSH)
pip install crawl4ai                        # Headless browser HTML backend
crawl4ai-setup                              # One-time browser setup

# Public dataset sources
pip install datasets huggingface_hub        # Hugging Face
pip install kaggle pandas                   # Kaggle

# System packages (for OCR/ASR)
# apt install poppler-utils tesseract-ocr ffmpeg

# Pull a small, fast instruct model for planning/judging
ollama pull llama3.1          # or qwen2.5:7b-instruct, mistral, etc.
ollama serve                  # if not already running
```

Sanity-check what's actually usable in your environment:
```bash
python -c "import server; print(server.healthcheck())"
```

## Usage

Pretraining-style output (matches `build_dataset.py`):
```bash
python agent/dataset_agent.py --target-size 500MB \
    --categories web,knowledge,reasoning,code,math \
    --out-dir ./data --mode pretrain --concurrency 8
```

SFT-style output (`{prompt, thinking, answer}`):
```bash
python agent/dataset_agent.py --target-size 300MB \
    --categories math,code,reasoning \
    --out-dir ./sft_data --mode sft
```

Codegen pipeline (simpler alternative, uses generated scripts):
```bash
python agent/codegen_pipeline.py --target-size 500MB --public-only \
    --categories web,knowledge,math --out-dir ./data --mode pretrain
```

Then pack as usual:
```bash
python pack_dataset.py --data-dir ./data ...
python pack_sft_data.py --data-dir ./sft_data ...
```

### Topping up from public dataset hubs

Auto-discover and pull matching datasets per category:
```bash
python agent/dataset_agent.py --target-size 1GB --mode pretrain \
    --categories web,knowledge,code,math \
    --out-dir ./data \
    --public-sources huggingface,kaggle
```

Name specific datasets:
```bash
export KAGGLE_USERNAME=you KAGGLE_KEY=xxxx
python agent/dataset_agent.py --target-size 500MB --mode sft \
    --categories math,code \
    --public-sources huggingface,kaggle \
    --hf-datasets "math=openai/gsm8k;code=codeparrot/apps" \
    --kaggle-datasets "code=owner/some-code-qa-dataset"
```

Skip live scraping entirely (public-only):
```bash
python agent/dataset_agent.py --target-size 2GB --mode pretrain \
    --categories knowledge,science \
    --public-sources huggingface \
    --public-only
```

## Notes, limits, and things worth tuning

- **Categories run concurrently, not one after another** via
  `asyncio.gather` (bounded by `--category-concurrency`, default 0 =
  unbounded). Each category has its own `ShardWriter`/`ExactDedup`/
  `RunState`, so there's no shared mutable state to race on.
- **The MCP server offloads CPU-bound extraction to process pools** via
  `ProcessPoolExecutor`, sized by `SCRAPER_EXTRACT_WORKERS` (default: all
  CPU cores) for fast extractors and `SCRAPER_MEDIA_WORKERS` (default: half
  the cores) for video/audio transcription.
- **Search backend**: DuckDuckGo's free endpoint is rate-limited. For
  serious scale, swap `web_search` to a paid API (Bing Search, Serper,
  Tavily, Brave Search API).
- **crawl4ai** is heavier than the plain httpx path (real headless browser).
  Set `SCRAPER_HTML_BACKEND=httpx` to skip it for static pages.
- **robots.txt / rate limiting / domain rules**: the server checks
  robots.txt per host and caps request rate to 1 per 2s per domain
  (`SCRAPER_MIN_HOST_INTERVAL`). Use `SCRAPER_ALLOWED_DOMAINS` and
  `SCRAPER_BLOCKED_DOMAINS` for explicit allow/deny control.
- **Near-dup filter**: install `datasketch` for real MinHash+LSH dedup at
  scale; without it, falls back to slower O(n) shingle-overlap comparison.
- **Cost/speed knob**: `--no-llm-judge` skips the Ollama quality-judging
  call. `--concurrency N` controls parallel extract+filter tasks.
- **`ExactDedup`** no longer opens/closes its persistence file per
  document — keeps one line-buffered file handle open for the run.
- **MemoryGovernor** provides process-wide back-pressure with RAM/swap
  thresholds, gc.collect()+malloc_trim, and a stall timeout to prevent
  permanent hangs from live referenced state.
