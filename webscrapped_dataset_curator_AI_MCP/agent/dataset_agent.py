#!/usr/bin/env python3
"""
dataset_agent.py

Self-directed data curation agent for the advanced LLM framework.

This agent orchestrates multi-format web scraping, public dataset pulls
(HuggingFace Hub / Kaggle), heuristic quality filtering, LLM-based quality
judging (Ollama), near-dedup, exact dedup, and resumable shard writing.

Architecture:
    - Asyncio event loop driving concurrent URL-level extraction via an MCP
      server (server.py).
    - Per-category RunState (persisted to JSON) tracking used queries and
      seen URLs so runs resume without repeating work.
    - Batched Ollama calls via an OllamaBatcher that coalesces concurrent
      requests to the same model.
    - MemoryGovernor that monitors RSS+swap, triggers gc + malloc_trim under
      pressure, and applies a stall timeout as a last resort against
      permanently-live referenced state.
    - ScraperClient wrapping the MCP Session for web_search + extract_content.

Two execution paths, controlled by --public-sources:
    1.  PUBLIC DATASET TOP-UP (runs first, before any web scraping):
        - Reads HF and/or Kaggle datasets, either auto-discovered or
          explicitly named via --hf-datasets / --kaggle-datasets.
        - Dispatches rows to the same quality filters, LLM judge, dedup, and
          ShardWriter as scraped rows.
    2.  LIVE WEB SCRAPING (runs against the remaining byte budget):
        - Asks Ollama for a batch of specific search queries per category,
          seeded from topics.py, explicitly avoiding recently used queries so
          coverage keeps growing.
        - Calls extract_content for every hit -- works uniformly whether the
          hit turned out to be an article, a PDF, a slide deck, or a YouTube
          video.
        - Runs the same heuristic filters as build_dataset.py (alpha ratio,
          repetition ratio, code extension/hygiene checks) plus format-aware
          additions.
        - Optionally runs an LLM quality judge pass (Ollama). Disable with
          --no-llm-judge for raw throughput.

Usage:
    python dataset_agent.py --target-size 500MB \\
        --categories web,knowledge,reasoning,code,math \\
        --out-dir ./data --mode pretrain --concurrency 8

    python dataset_agent.py --target-size 300MB \\
        --categories math,code,reasoning \\
        --out-dir ./sft_data --mode sft --concurrency 4

    python dataset_agent.py --target-size 1GB --mode pretrain \\
        --categories web,knowledge,code,math \\
        --out-dir ./data \\
        --public-sources huggingface,kaggle

    python dataset_agent.py --target-size 500MB --mode sft \\
        --categories math,code \\
        --public-sources huggingface,kaggle \\
        --hf-datasets "math=openai/gsm8k;code=codeparrot/apps" \\
        --kaggle-datasets "code=owner/some-code-qa-dataset"

    python dataset_agent.py --target-size 2GB --mode pretrain \\
        --categories knowledge,science \\
        --public-sources huggingface \\
        --public-only
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import math
import os
import random
import re
import resource
import signal
import struct
import sys
import time
import traceback
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Optional

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(__file__))
from quality import ExactDedup, NearDedup, RunState, ShardWriter
from quality import passes_prose_quality_filter as _passes_prose
from quality import passes_sft_pair_quality_filter as _passes_sft_pair
from quality import passes_code_quality_filter as _passes_code
from quality import passes_transcript_quality_filter as _passes_transcript
from quality import passes_extended_text_quality as _passes_ext_prose
from quality import passes_extended_sft_quality as _passes_ext_sft
from quality import detect_language
from topics import HUB_SEARCH_KEYWORDS, TOPIC_SEEDS

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
LLM_JUDGE_PROMPT_TEMPLATE = os.environ.get(
    "LLM_JUDGE_PROMPT_TEMPLATE",
    "{category}: rate this data from 0 (useless) to 10 (excellent). "
    "Return ONLY a single integer (no explanation).\n\n{text}",
)

log = logging.getLogger("dataset_agent")


# ---------------------------------------------------------------------------
# Memory back-pressure (MemoryGovernor)
# ---------------------------------------------------------------------------

class MemoryGovernor:
    """Process-wide RAM+swap back-pressure.

    Scans rss vs a soft threshold every N doc completions. When over the
    threshold it triggers gc.collect() + malloc_trim, escalating to a longer
    stall if pressure persists. This avoids OOM kills on large-hash near-dedup
    or bursty concurrent downloads.
    """

    def __init__(self, max_rss_gb: float = 24.0, max_swap_gb: float = 4.0,
                 check_every: int = 10, stall_timeout: float = 2.0):
        self.max_rss = int(max_rss_gb * 1024 * 1024 * 1024)
        self.max_swap = int(max_swap_gb * 1024 * 1024 * 1024)
        self.check_every = check_every
        self.stall_timeout = stall_timeout
        self._counter = 0
        self._in_stall = False
        self._last_check = 0.0

    @staticmethod
    def _page_size() -> int:
        return os.sysconf("SC_PAGE_SIZE")

    def _rss_kb(self) -> int:
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1])
        except Exception:
            pass
        return 0

    def _swap_kb(self) -> int:
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmSwap:"):
                        return int(line.split()[1])
        except Exception:
            pass
        return 0

    def check(self) -> None:
        self._counter += 1
        if self._counter % self.check_every != 0:
            return

        rss_kb = self._rss_kb()
        swap_kb = self._swap_kb()
        over_rss = rss_kb * 1024 > self.max_rss
        over_swap = swap_kb * 1024 > self.max_swap

        if not over_rss and not over_swap:
            return

        log.warning("memory pressure: rss=%d MB swap=%d MB -- collecting garbage",
                     rss_kb // 1024, swap_kb // 1024)
        gc.collect()
        try:
            import ctypes
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except Exception:
                pass
        except Exception:
            pass

        rss_after = self._rss_kb()
        new_rss_mb = rss_after // 1024

        if rss_after < rss_kb * 0.85:
            log.info(f"gc freed {rss_kb - rss_after} KB, rss now {new_rss_mb} MB")
        else:
            if not self._in_stall:
                log.warning(f"gc did not reduce rss ({rss_kb} KB -> {rss_after} KB), "
                            f"stalling {self.stall_timeout}s as last resort")
                self._in_stall = True
                time.sleep(self.stall_timeout)
                gc.collect()
                self._in_stall = False
            else:
                log.warning("still under pressure after stall -- continuing anyway")


# ---------------------------------------------------------------------------
# Per-document LLM judge (batched)
# ---------------------------------------------------------------------------

_OLLAMA_SEM = asyncio.Semaphore(1)


async def _ollama_judge(text: str, category: str, mode: str, client: httpx.AsyncClient) -> Optional[dict]:
    """LLM quality judge. Returns dict like {"score": 7} or {"score": 0}
    on failure."""
    prompt = LLM_JUDGE_PROMPT_TEMPLATE.format(category=category, mode=mode, text=text[:4000])
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "think": False}
    try:
        async with _OLLAMA_SEM:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=httpx.Timeout(30.0),
            )
            resp.raise_for_status()
            data = resp.json()
        raw = data.get("response", "").strip()
        return {"score": _parse_judge_score(raw)}
    except Exception as e:
        log.debug(f"ollama judge failed: {e}")
        return None


def _parse_judge_score(raw: str) -> int:
    raw = re.sub(r"\D", " ", raw)
    nums = raw.split()
    if nums:
        val = int(nums[0])
        return max(0, min(10, val))
    return 0


class _BufferedLineWriter:
    """Line-buffered file handle that flushes every N seconds."""

    def __init__(self, path: str, flush_interval: float = 2.0):
        self.fh = open(path, "a", buffering=1, encoding="utf-8")
        self._flush_interval = flush_interval
        self._last_flush = time.monotonic()

    def write(self, line: str) -> None:
        self.fh.write(line)
        now = time.monotonic()
        if now - self._last_flush >= self._flush_interval:
            self.fh.flush()
            self._last_flush = now

    def close(self) -> None:
        try:
            self.fh.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# OllamaBatcher — coalescing batch layer for concurrent Ollama requests
# ---------------------------------------------------------------------------

class _BatcRequest:
    __slots__ = ("prompt", "future", "created_at")

    def __init__(self, prompt: str):
        self.prompt = prompt
        self.future: asyncio.Future[Optional[dict]] = asyncio.Future()
        self.created_at = time.monotonic()


class OllamaBatcher:
    """Coalesces concurrent Ollama requests into batch calls.

    Instead of N concurrent /api/generate calls for N concurrent documents,
    requests arriving within a short window are sent as separate calls but
    managed together: the batcher fires one batch-window cycle, collects all
    pending prompts, and dispatches them concurrently with a semaphore limit.
    This is NOT a true server-side batch API (Ollama doesn't expose one for
    generate) but it avoids the thundering-herd of N simultaneous calls.
    """

    def __init__(self, model: str = OLLAMA_MODEL, max_concurrent: int = 4,
                 window: float = 0.05, timeout: float = 30.0):
        self.model = model
        self.max_concurrent = max_concurrent
        self.window = window
        self.timeout = timeout
        self._queue: list[_BatcRequest] = []
        self._loop_task: Optional[asyncio.Task] = None
        self._sem = asyncio.Semaphore(max_concurrent)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def submit(self, prompt: str) -> Optional[dict]:
        req = _BatcRequest(prompt)
        self._queue.append(req)
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.create_task(self._drain_loop())
        return await req.future

    async def _drain_loop(self) -> None:
        while self._queue:
            batch = self._gather_batch()
            if not batch:
                await asyncio.sleep(self.window)
                continue
            tasks = []
            for req in batch:
                tasks.append(asyncio.create_task(self._dispatch(req)))
            await asyncio.gather(*tasks)

    def _gather_batch(self) -> list[_BatcRequest]:
        batch, self._queue = self._queue[:], []
        return batch

    async def _dispatch(self, req: _BatcRequest) -> None:
        async with self._sem:
            payload = {
                "model": self.model,
                "prompt": req.prompt,
                "stream": False,
                "think": False,
            }
            try:
                resp = await self._client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                raw = data.get("response", "").strip()
                if not req.future.done():
                    req.future.set_result({"response": raw})
            except Exception as e:
                if not req.future.done():
                    req.future.set_exception(e)

    async def close(self) -> None:
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
        await self._client.aclose()


# ---------------------------------------------------------------------------
# ScraperClient — thin async wrapper around the MCP session
# ---------------------------------------------------------------------------

class ScraperClient:
    """Wraps an MCP ClientSession to expose the scraper tools as async
    methods. Instantiated once per category run."""

    def __init__(self, session):
        self._session = session

    async def search(self, query: str, max_results: int = 8) -> list[dict]:
        result = await self._session.call_tool(
            "web_search",
            {"query": query, "max_results": max_results},
        )
        return _mcp_result_to_list(result)

    async def extract(self, url: str) -> Optional[dict]:
        result = await self._session.call_tool(
            "extract_content",
            {"url": url},
        )
        return _mcp_result_to_dict(result)

    async def healthcheck(self) -> Optional[dict]:
        result = await self._session.call_tool("healthcheck", {})
        return _mcp_result_to_dict(result)


def _mcp_result_to_list(result) -> list:
    try:
        raw = json.loads(result.content[0].text)
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict) and "results" in raw:
            return raw["results"]
        return [raw] if isinstance(raw, dict) else []
    except Exception:
        return []


def _mcp_result_to_dict(result) -> Optional[dict]:
    try:
        raw = json.loads(result.content[0].text)
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# URL normalization + dedup
# ---------------------------------------------------------------------------

def _canonical_url(url: str) -> str:
    """Normalize a URL for dedup: remove tracking params, hash, www prefix,
    trailing slash."""
    try:
        from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
        parsed = urlparse(url)
        query = parse_qs(parsed.query, keep_blank_values=True)
        for noisy in ("utm_source", "utm_medium", "utm_campaign",
                      "utm_term", "utm_content", "fbclid", "gclid",
                      "ref", "source", "mc_cid", "mc_eid"):
            query.pop(noisy, None)
        clean_query = urlencode(sorted(query.items()), doseq=True)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parsed.path.rstrip("/") or "/"
        return urlunparse((parsed.scheme, netloc, path, parsed.params,
                           clean_query, ""))
    except Exception:
        return url


# ---------------------------------------------------------------------------
# Quality filters (heuristic — same names as quality.py)
# ---------------------------------------------------------------------------

def _format_specific_filter(record: dict) -> tuple[bool, str]:
    """Format-specific checks beyond the generic prose/code filters."""
    ct = record.get("content_type", "")
    text = record.get("text", "")
    if ct == "pdf":
        if len(text) < 50:
            return False, "pdf too short"
    if ct == "image":
        if len(text) < 100:
            return False, "image OCR too short"
    if ct in ("video", "audio"):
        result, _ = _passes_transcript(text)
        if not result:
            return False, "transcript quality filter failed"
    return True, ""


def _make_heuristic_filter(
    mode: str,
    min_doc_chars: int,
    record: dict,
    use_extended: bool = True,
    quality_kwargs: Optional[dict] = None,
) -> tuple[bool, str]:
    text = record.get("text", "")
    if len(text) < min_doc_chars:
        return False, f"too short ({len(text)} < {min_doc_chars})"

    if use_extended:
        kwargs = dict(quality_kwargs or {})
        kwargs.setdefault("min_chars", min_doc_chars)
        target_langs = kwargs.pop("target_langs", None)

        if mode == "pretrain":
            return _passes_ext_prose(text, target_langs=target_langs, **kwargs)
        if mode in ("sft", "grpo"):
            prompt = record.get("prompt") or record.get("extra", {}).get("prompt") or ""
            answer = record.get("answer") or record.get("extra", {}).get("answer") or ""
            if not prompt or not answer:
                return False, f"{mode} mode needs both prompt and answer"
            return _passes_ext_sft(prompt, answer, target_langs=target_langs, **kwargs)
        return _passes_ext_prose(text, target_langs=target_langs, **kwargs)
    else:
        # Legacy basic filters
        if mode == "pretrain":
            return _passes_prose(text, min_doc_chars=min_doc_chars)
        if mode in ("sft", "grpo"):
            prompt = record.get("prompt") or record.get("extra", {}).get("prompt") or ""
            answer = record.get("answer") or record.get("extra", {}).get("answer") or ""
            if not prompt or not answer:
                return False, f"{mode} mode needs both prompt and answer"
            return _passes_sft_pair(prompt, answer, min_chars=min_doc_chars)
        return _passes_prose(text, min_doc_chars=min_doc_chars)


# ---------------------------------------------------------------------------
# Query planning via Ollama
# ---------------------------------------------------------------------------

_QUERY_PLAN_PROMPT = """You are a data curator planning web search queries.

Category: {category}
Context: {context}

Previously used queries (avoid these): {used_queries}

Generate {num_queries} NEW, specific, diverse search queries that will find
high-quality long-form content for this category. Queries should target
blog posts, technical articles, tutorials, academic essays.

Return ONLY a JSON array of strings: ["query1", "query2", ...]

Focus on quality over quantity — each query should be precise enough to
avoid generic results."""


async def _plan_queries_from_ollama(
    category: str,
    context: str,
    num_queries: int,
    used_queries: set[str],
    batcher: OllamaBatcher,
) -> list[str]:
    used_str = ", ".join(sorted(used_queries)[-30:])
    prompt = _QUERY_PLAN_PROMPT.format(
        category=category, context=context,
        used_queries=used_str or "(none so far)",
        num_queries=num_queries,
    )
    result = await batcher.submit(prompt)
    if result is None:
        return []
    raw = result.get("response", "")
    try:
        queries = json.loads(raw)
        if isinstance(queries, list):
            return [q.strip() for q in queries if q.strip()]
    except json.JSONDecodeError:
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            try:
                queries = json.loads(match.group(0))
                if isinstance(queries, list):
                    return [q.strip() for q in queries if q.strip()]
            except json.JSONDecodeError:
                pass
    words = raw.split(",")
    if len(words) >= 2:
        return [w.strip().strip('"[]').strip() for w in words if w.strip()]
    return []


# ---------------------------------------------------------------------------
# SFTPair extraction (for sft/grpo modes)
# ---------------------------------------------------------------------------

_SFT_EXTRACT_PROMPT = """You are a data transformation assistant.

Given a text passage, extract a high-quality instruction-following pair
for training an LLM in {mode} mode.

- If the text contains an explicit question + answer/solution, use those.
- If the text is a tutorial or walkthrough, frame it as a how-to question.
- If the text is a discussion or analysis, extract the most instructive
  question being addressed and the key answer.
- If the text is code, frame it as "Write a function/code for X" with the
  code as answer.

Return ONLY valid JSON without explanation:
{{
    "prompt": "the question or instruction",
    "thinking": "the chain-of-thought or reasoning (only if the text has an
                 intermediate reasoning step, else empty string)",
    "answer": "the final and complete answer"
}}

If you can't extract a meaningful pair, return:
{{"prompt": "", "thinking": "", "answer": ""}}

Text:
{text}"""


async def _extract_sft_pair(record: dict, mode: str, batcher: OllamaBatcher) -> Optional[dict]:
    """Try to extract a prompt+answer pair from a record using Ollama."""
    text = record.get("text", "")
    if not text or len(text) < 200:
        return None

    prompt = _SFT_EXTRACT_PROMPT.format(mode=mode, text=text[:5000])
    result = await batcher.submit(prompt)
    if result is None:
        return None

    raw = result.get("response", "")
    try:
        pair = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                pair = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        else:
            return None

    prompt_text = (pair.get("prompt") or "").strip()
    answer_text = (pair.get("answer") or "").strip()
    if not prompt_text or not answer_text:
        return None

    return {
        "prompt": prompt_text,
        "thinking": (pair.get("thinking") or "").strip(),
        "answer": answer_text,
    }


# ---------------------------------------------------------------------------
# Column mapper for public datasets
# ---------------------------------------------------------------------------

_COLUMN_MAP_PROMPT = """You are analyzing a public dataset for LLM training.

Dataset ID: {dataset_id}
Columns: {columns}
Mode: {mode}

Map the dataset's columns to the target schema. Consider:
- If mode is pretrain, look for a "text", "content", or "document" column.
- If mode is sft or grpo, look for "prompt"/"question"/"instruction" +
  "answer"/"response"/"output" columns, or a "conversations"/"messages" array.
- "answer_col" can also be a code column ("code", "solution_code").

First row as a preview: {preview}

Return ONLY valid JSON without explanation:
{{
    "text_col": "name or null",
    "prompt_col": "name or null",
    "answer_col": "name or null",
    "code_col": "name or null",
    "conversation_col": "name or null"
}}"""


async def _column_mapper(
    dataset_id: str,
    config_or_path: str,
    columns: list[str],
    first_row: dict,
    batcher: OllamaBatcher,
    mode: str,
) -> Optional[dict]:
    preview = json.dumps(first_row, ensure_ascii=False, default=str)[:2000]
    prompt = _COLUMN_MAP_PROMPT.format(
        dataset_id=dataset_id,
        columns=columns,
        mode=mode,
        preview=preview,
        config_or_path=config_or_path or "default",
    )
    result = await batcher.submit(prompt)
    if result is None:
        return None
    raw = result.get("response", "")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# MCP server locator
# ---------------------------------------------------------------------------

def _find_server_path() -> str:
    """Locate the MCP server script. Check several likely locations."""
    candidates = [
        "server.py",
        "web_scraper_mcp/server.py",
        os.path.join(os.path.dirname(__file__), "..", "web_scraper_mcp", "server.py"),
        os.path.join(os.getcwd(), "web_scraper_mcp", "server.py"),
        os.path.join(os.path.dirname(__file__), "..", "server.py"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    # Fall back to first candidate path for a useful error message
    return os.path.abspath(candidates[0])


# ---------------------------------------------------------------------------
# Category run
# ---------------------------------------------------------------------------

class CategoryRunner:
    """Holds all state for one category across one invocation (or resumption).

    Each runner gets its OWN ScraperClient (wrapping the same MCP session),
    ExactDedup, RunState, ShardWriter, etc., so concurrent categories don't
    share mutable state.
    """

    def __init__(self, category: str, budget_bytes: int, out_dir: str,
                 mode: str, min_doc_chars: int, concurrency: int,
                 use_llm_judge: bool, batcher: OllamaBatcher,
                 memory_gov: MemoryGovernor, public_only: bool,
                 use_extended_quality: bool = True,
                 quality_thresholds: Optional[dict] = None,
                 target_langs: Optional[set] = None):
        self.category = category
        self.budget_bytes = budget_bytes
        self.out_dir = out_dir
        self.mode = mode
        self.min_doc_chars = min_doc_chars
        self.concurrency = concurrency
        self.use_llm_judge = use_llm_judge
        self.batcher = batcher
        self.memory_gov = memory_gov
        self.public_only = public_only
        self.use_extended_quality = use_extended_quality
        self.quality_thresholds = quality_thresholds or {}
        self.target_langs = target_langs

        # State built during run
        self.total_bytes: int = 0
        self.total_docs: int = 0
        self.total_urls_seen: int = 0
        self.total_urls_failed: int = 0
        self.total_urls_filtered: int = 0
        self.total_llm_rejected: int = 0
        self.total_dup_skipped: int = 0
        self.total_queries_used: int = 0

        # Persistable objects
        self.state = RunState(os.path.join(out_dir, f".run_state_{category}.json"))
        self.dedup = ExactDedup(persist_path=os.path.join(
            out_dir, f".exact_dedup_{category}.bloom"))
        self.near_dedup = NearDedup()
        self.shard_writer = ShardWriter(out_dir, category)

        # For --public-sources public dataset pulling
        self._public_sources: list[str] = []

        # Ollama judge HTTP client (separate from batcher's)
        self._judge_client: Optional[httpx.AsyncClient] = None

    def _init_judge_client(self) -> httpx.AsyncClient:
        if self._judge_client is None:
            self._judge_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_keepalive_connections=4,
                                     max_connections=8),
            )
        return self._judge_client

    async def _close_judge_client(self) -> None:
        if self._judge_client is not None:
            await self._judge_client.aclose()
            self._judge_client = None

    def _record_accepted(self, record: dict) -> None:
        byte_cost = len((record.get("text") or "").encode("utf-8"))
        self.total_bytes += byte_cost
        self.total_docs += 1
        self.shard_writer.write(record)

    def _filter_and_dedup(self, record: dict) -> Optional[dict]:
        # Heuristic filter (extended quality)
        quality_kwargs = dict(self.quality_thresholds)
        if self.target_langs:
            quality_kwargs["target_langs"] = self.target_langs
        ok, reason = _make_heuristic_filter(
            self.mode,
            self.min_doc_chars,
            record,
            use_extended=self.use_extended_quality,
            quality_kwargs=quality_kwargs if self.use_extended_quality else None,
        )
        if not ok:
            self.total_urls_filtered += 1
            return None

        # Format-specific filter
        ok, reason = _format_specific_filter(record)
        if not ok:
            self.total_urls_filtered += 1
            return None

        # Exact dedup
        text = record.get("text", "")
        if self.dedup.is_duplicate(text):
            self.total_dup_skipped += 1
            return None

        # Near dedup
        n_dup = self.near_dedup.is_duplicate(text)
        if n_dup:
            self.total_dup_skipped += 1
            return None

        # Mark as seen in dedup
        self.dedup.mark_seen(text)
        self.near_dedup.mark_seen(text)

        return record

    async def _llm_judge(self, record: dict) -> bool:
        if not self.use_llm_judge:
            return True
        client = self._init_judge_client()
        text = record.get("text", "")
        if self.mode in ("sft", "grpo"):
            prompt = record.get("prompt") or record.get("extra", {}).get("prompt") or ""
            answer = record.get("answer") or record.get("extra", {}).get("answer") or ""
            text = f"Prompt: {prompt}\n\nAnswer: {answer}"
        result = await _ollama_judge(text, self.category, self.mode, client)
        if result is None or result.get("score", 0) < 4:
            self.total_llm_rejected += 1
            return False
        return True

    async def process_one_record(self, record: dict) -> None:
        if self.total_bytes >= self.budget_bytes:
            return

        self.total_urls_seen += 1

        # SFT/GRPO: try to extract prompt+answer pair
        if self.mode in ("sft", "grpo") and not record.get("prompt"):
            pair = await _extract_sft_pair(record, self.mode, self.batcher)
            if pair and pair.get("prompt") and pair.get("answer"):
                record["prompt"] = pair["prompt"]
                record["thinking"] = pair.get("thinking", "")
                record["answer"] = pair["answer"]

        # Filter + dedup
        filtered = self._filter_and_dedup(record)
        if filtered is None:
            return

        # LLM judge
        if not await self._llm_judge(filtered):
            return

        # Accept
        self._record_accepted(filtered)
        self.memory_gov.check()

    async def run_public_sources(self, sources: list[str],
                                  hf_datasets: Optional[dict[str, str]] = None,
                                  kaggle_datasets: Optional[dict[str, str]] = None) -> bool:
        """Top-up public datasets for this category. Returns True if any
        data was produced."""
        if not sources:
            return False

        from public_sources import (
            discover_hf_datasets,
            stream_hf_dataset,
            discover_kaggle_datasets,
            fetch_kaggle_dataset_rows,
        )

        any_data = False

        skipped_no_columns = 0
        skipped_download_error = 0
        skipped_schema = 0
        accepted = 0

        if "huggingface" in sources:
            # Specific datasets named via CLI
            hf_specific = {}
            if hf_datasets:
                for cat, spec in hf_datasets.items():
                    if cat == self.category:
                        spec_list = [s.strip() for s in spec.split(";") if s.strip()]
                        for entry in spec_list:
                            parts = entry.split(":")
                            did = parts[0].strip()
                            cfg = parts[1].strip() if len(parts) > 1 else None
                            hf_specific[did] = (cfg, self.mode)

            # Auto-discovered
            auto_datasets = []
            for kw in HUB_SEARCH_KEYWORDS.get(self.category, [self.category]):
                auto_datasets.extend(discover_hf_datasets(kw, limit=5))

            # Mix: specific first, then auto
            datasets_to_try = list(hf_specific.items()) + [(d, None) for d in auto_datasets if d not in hf_specific]

            for dataset_id, cfg_mode in datasets_to_try:
                if self.total_bytes >= self.budget_bytes:
                    break
                ds_mode = cfg_mode[1] if isinstance(cfg_mode, tuple) else self.mode
                config_arg = cfg_mode[0] if isinstance(cfg_mode, tuple) else cfg_mode
                if config_arg is None:
                    config_arg = None

                max_rows_per_dataset = max(50, int(
                    (self.budget_bytes - self.total_bytes) / max(1, 5000)
                ))
                max_rows_per_dataset = min(max_rows_per_dataset, 500)

                try:
                    records = list(stream_hf_dataset(
                        dataset_id,
                        max_rows=max_rows_per_dataset,
                        split="train",
                        config=config_arg,
                        column_mapper=lambda did, cfg, cols, row, batcher=self.batcher, mode=ds_mode:
                            COLUMN_MAP_CALLBACK(did, cfg, cols, row, batcher, mode),
                    ))
                except Exception as e:
                    log.debug(f"hf streaming failed for {dataset_id}: {e}")
                    skipped_download_error += 1
                    continue

                if not records:
                    skipped_no_columns += 1
                    continue

                for rec in records:
                    if rec.get("error"):
                        skipped_schema += 1
                        continue
                    if self.total_bytes >= self.budget_bytes:
                        break
                    await self.process_one_record(rec)
                    if rec.get("text"):
                        any_data = True
                        accepted += 1

        if "kaggle" in sources:
            kaggle_specific = {}
            if kaggle_datasets:
                for cat, spec in kaggle_datasets.items():
                    if cat == self.category:
                        spec_list = [s.strip() for s in spec.split(";") if s.strip()]
                        kaggle_specific.update({d: None for d in spec_list})

            auto_kaggle = []
            for kw in HUB_SEARCH_KEYWORDS.get(self.category, [self.category]):
                auto_kaggle.extend(discover_kaggle_datasets(kw, limit=3))

            datasets_to_try = list(kaggle_specific.keys()) + [d for d in auto_kaggle if d not in kaggle_specific]

            for ref in datasets_to_try:
                if self.total_bytes >= self.budget_bytes:
                    break
                max_rows = max(50, int(
                    (self.budget_bytes - self.total_bytes) / max(1, 5000)
                ))
                max_rows = min(max_rows, 500)
                try:
                    records = list(fetch_kaggle_dataset_rows(
                        ref,
                        max_rows=max_rows,
                        column_mapper=lambda did, cfg, cols, row, batcher=self.batcher, mode=self.mode:
                            COLUMN_MAP_CALLBACK(did, cfg, cols, row, batcher, mode),
                    ))
                except Exception as e:
                    log.debug(f"kaggle download failed for {ref}: {e}")
                    skipped_download_error += 1
                    continue
                for rec in records:
                    if rec.get("error"):
                        skipped_schema += 1
                        continue
                    if self.total_bytes >= self.budget_bytes:
                        break
                    await self.process_one_record(rec)
                    if rec.get("text"):
                        any_data = True
                        accepted += 1

        log.info(f"[{self.category}] public sources done: {accepted} accepted, "
                 f"{skipped_no_columns} no-columns, {skipped_schema} schema-rejected, "
                 f"{skipped_download_error} download-errors")
        return any_data

    async def run_web(self, session, num_rounds: int = 5) -> None:
        """Run the main web scraping loop for this category."""
        scraper = ScraperClient(session)
        queries_used = self.state.get("used_queries", set())
        seen_urls = self.state.get("seen_urls", set())

        if not queries_used:
            queries_used = set()

        log.info(f"[{self.category}] starting web scrape (budget={self.budget_bytes} bytes, "
                 f"{num_rounds} rounds)")

        sem = asyncio.Semaphore(self.concurrency)

        for round_idx in range(1, num_rounds + 1):
            if self.total_bytes >= self.budget_bytes:
                log.info(f"[{self.category}] budget reached, stopping web rounds")
                break

            # 1. Plan queries
            context = random.choice(TOPIC_SEEDS.get(self.category, [self.category]))
            num_queries = 5 + round_idx  # grows with each round
            new_queries = await _plan_queries_from_ollama(
                self.category, context, num_queries, queries_used, self.batcher,
            )
            if not new_queries:
                log.warning(f"[{self.category}] no new queries from planner")
                continue

            # 2. Search for each query
            all_hits: list[dict] = []
            for query in new_queries:
                if self.total_bytes >= self.budget_bytes:
                    break
                try:
                    hits = await scraper.search(query, max_results=8)
                    all_hits.extend(hits or [])
                except Exception as e:
                    log.debug(f"[{self.category}] search failed for {query!r}: {e}")
                queries_used.add(query)
                self.state.set("used_queries", queries_used)

            if not all_hits:
                continue

            # 3. Dedup URLs at the hit level
            unique_hits = []
            for hit in all_hits:
                url = None
                if isinstance(hit, dict):
                    url = hit.get("url") or hit.get("link") or hit.get("href")
                if not url:
                    continue
                canonical = _canonical_url(url)
                if canonical in seen_urls:
                    continue
                seen_urls.add(canonical)
                self.state.set("seen_urls", seen_urls)
                unique_hits.append((url, hit))

            self.total_queries_used += len(new_queries)

            # 4. Extract content concurrently
            async def _process(url: str, hit: dict) -> None:
                async with sem:
                    try:
                        extracted = await scraper.extract(url)
                    except Exception as e:
                        self.total_urls_failed += 1
                        log.debug(f"[{self.category}] extract failed for {url}: {e}")
                        return
                    if extracted is None:
                        self.total_urls_failed += 1
                        return
                    record = extracted
                    record.setdefault("extra", {})
                    record["extra"]["query"] = hit.get("query", hit.get("title", ""))
                    await self.process_one_record(record)

            tasks = [_process(url, hit) for url, hit in unique_hits]
            await asyncio.gather(*tasks)
            self.total_queries_used += len(new_queries)

            # Save state each round
            self.state.save()

        self.state.save()

    async def close(self) -> None:
        self.shard_writer.close()
        self.state.save()
        self.dedup.close()
        await self._close_judge_client()


def COLUMN_MAP_CALLBACK(dataset_id: str, config_or_path: str,
                         columns: list[str], first_row: dict,
                         batcher: OllamaBatcher, mode: str) -> Optional[dict]:
    """Synchronous callback wrapper for the async column mapper."""
    try:
        loop = asyncio.get_running_loop()
        future = asyncio.run_coroutine_threadsafe(
            _column_mapper(dataset_id, config_or_path, columns, first_row, batcher, mode),
            loop,
        )
        return future.result(timeout=30)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def _run_category(
    category: str,
    budget_bytes: int,
    out_dir: str,
    mode: str,
    min_doc_chars: int,
    concurrency: int,
    use_llm_judge: bool,
    batcher: OllamaBatcher,
    memory_gov: MemoryGovernor,
    session,
    public_sources: list[str],
    public_only: bool,
    hf_datasets: Optional[dict[str, str]] = None,
    kaggle_datasets: Optional[dict[str, str]] = None,
    use_extended_quality: bool = True,
    quality_thresholds: Optional[dict] = None,
    target_langs: Optional[set] = None,
) -> dict:
    runner = CategoryRunner(
        category=category,
        budget_bytes=budget_bytes,
        out_dir=out_dir,
        mode=mode,
        min_doc_chars=min_doc_chars,
        concurrency=concurrency,
        use_llm_judge=use_llm_judge,
        batcher=batcher,
        memory_gov=memory_gov,
        public_only=public_only,
        use_extended_quality=use_extended_quality,
        quality_thresholds=quality_thresholds,
        target_langs=target_langs,
    )
    try:
        # Phase 1: Public sources (if enabled)
        if public_sources:
            log.info(f"[{category}] topping up from public sources: {public_sources}")
            await runner.run_public_sources(
                sources=public_sources,
                hf_datasets=hf_datasets,
                kaggle_datasets=kaggle_datasets,
            )

        # Phase 2: Live web scraping
        if not public_only and runner.total_bytes < budget_bytes:
            num_rounds = max(3, int(
                math.log2((budget_bytes - runner.total_bytes) / max(1, 100 * 1024)) + 2
            ))
            num_rounds = min(num_rounds, 15)
            await runner.run_web(session, num_rounds=num_rounds)

    except Exception as e:
        log.error(f"[{category}] error: {e}")
        traceback.print_exc()
    finally:
        await runner.close()

    return {
        "category": category,
        "target_bytes": budget_bytes,
        "actual_bytes": runner.total_bytes,
        "docs": runner.total_docs,
        "urls_seen": runner.total_urls_seen,
        "urls_failed": runner.total_urls_failed,
        "urls_filtered": runner.total_urls_filtered,
        "llm_rejected": runner.total_llm_rejected,
        "dup_skipped": runner.total_dup_skipped,
        "queries_used": runner.total_queries_used,
    }


async def _run_categories(
    categories: list[dict],
    out_dir: str,
    mode: str,
    min_doc_chars: int,
    concurrency: int,
    use_llm_judge: bool,
    public_sources: list[str],
    public_only: bool,
    hf_datasets: Optional[dict[str, str]] = None,
    kaggle_datasets: Optional[dict[str, str]] = None,
    max_category_concurrency: int = 2,
    use_extended_quality: bool = True,
    quality_thresholds: Optional[dict] = None,
    target_langs: Optional[set] = None,
) -> dict:
    """Run multiple categories, respecting max_category_concurrency."""
    server_path = _find_server_path()

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_path],
        env=os.environ.copy(),
    )

    memory_gov = MemoryGovernor()
    batcher = OllamaBatcher()

    results = {}

    # Worker function that takes an MCP session and runs one category
    async def _run_one(cat_config: dict, session) -> None:
        cat_name = cat_config["category"]
        result = await _run_category(
            category=cat_name,
            budget_bytes=cat_config["budget"],
            out_dir=out_dir,
            mode=mode,
            min_doc_chars=min_doc_chars,
            concurrency=concurrency,
            use_llm_judge=use_llm_judge,
            batcher=batcher,
            memory_gov=memory_gov,
            session=session,
            public_sources=public_sources,
            public_only=public_only,
            hf_datasets=hf_datasets,
            kaggle_datasets=kaggle_datasets,
            use_extended_quality=use_extended_quality,
            quality_thresholds=quality_thresholds,
            target_langs=target_langs,
        )
        results[cat_name] = result

    category_sem = asyncio.Semaphore(max_category_concurrency)
    pending = []

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=30)

            async def _bounded_run(cat_config: dict) -> None:
                async with category_sem:
                    await _run_one(cat_config, session)

            tasks = [_bounded_run(c) for c in categories]
            await asyncio.gather(*tasks)

    await batcher.close()

    total_bytes = sum(r.get("actual_bytes", 0) for r in results.values())
    total_docs = sum(r.get("docs", 0) for r in results.values())

    return {
        "target_bytes": sum(c["budget"] for c in categories),
        "actual_bytes": total_bytes,
        "docs": total_docs,
        "categories": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_size(size_str: str) -> int:
    size_str = size_str.strip().upper()
    for unit, mult in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024), ("B", 1)):
        if size_str.endswith(unit):
            return int(float(size_str[: -len(unit)] if len(unit) else size_str) * mult)
    return int(float(size_str))


def _parse_hf_datasets(raw: str) -> dict[str, str]:
    """Parse --hf-datasets 'math=openai/gsm8k;code=codeparrot/apps'"""
    result = {}
    if not raw:
        return result
    for part in raw.split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        cat, val = part.split("=", 1)
        result[cat.strip()] = val.strip()
    return result


def _parse_kaggle_datasets(raw: str) -> dict[str, str]:
    """Same format as _parse_hf_datasets."""
    return _parse_hf_datasets(raw)


def _setup_logging(log_file: Optional[str] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ] + ([logging.FileHandler(log_file)] if log_file else []),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-directed data curation agent for LLM training.")
    parser.add_argument("--target-size", required=True,
                        help="Target total dataset size, e.g. 500MB, 2GB")
    parser.add_argument("--out-dir", default="./data",
                        help="Output directory for shards")
    parser.add_argument("--categories",
                        default="web,knowledge,reasoning,code,math,science",
                        help="Comma-separated category list")
    parser.add_argument("--mode", choices=("pretrain", "sft", "grpo"),
                        default="pretrain")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max concurrent URL extractions per category")
    parser.add_argument("--min-doc-chars", type=int, default=500,
                        help="Minimum document length in characters")
    parser.add_argument("--no-llm-judge", action="store_true",
                        help="Skip Ollama quality judge pass")
    parser.add_argument("--mix", default=None,
                        help="Budget mix, e.g. web=0.2,knowledge=0.3,code=0.5")
    parser.add_argument("--public-sources", default="",
                        help="Comma-separated: huggingface,kaggle")
    parser.add_argument("--public-only", action="store_true",
                        help="Skip live web scraping entirely")
    parser.add_argument("--hf-datasets", default="",
                        help="Semicolon-separated cat=dataset pairs")
    parser.add_argument("--kaggle-datasets", default="",
                        help="Same format as --hf-datasets")
    parser.add_argument("--category-concurrency", type=int, default=2,
                        help="Max categories running simultaneously (default 2)")
    parser.add_argument("--log-file", default=None,
                        help="Write log to this file in addition to stdout")
    # Extended quality filter arguments
    parser.add_argument("--no-extended-quality", action="store_true",
                        help="Disable extended quality filters (compression ratio, vocab diversity, etc.)")
    parser.add_argument("--max-compression-ratio", type=float, default=0.35,
                        help="Max zlib compression ratio for text quality (default 0.35)")
    parser.add_argument("--max-line-repetition", type=float, default=0.15,
                        help="Max fraction of duplicate lines (default 0.15)")
    parser.add_argument("--max-adjacent-repetition", type=float, default=0.15,
                        help="Max fraction of adjacent near-identical lines (default 0.15)")
    parser.add_argument("--min-vocab-diversity", type=float, default=0.15,
                        help="Min unique/total word ratio (default 0.15)")
    parser.add_argument("--max-short-line-ratio", type=float, default=0.50,
                        help="Max fraction of short/navigation lines (default 0.50)")
    parser.add_argument("--max-flagged-ngram-ratio", type=float, default=0.10,
                        help="Max fraction of lines with flagged patterns (default 0.10)")
    parser.add_argument("--target-langs", default=None,
                        help="Comma-separated target languages for filtering, e.g. en,de (requires fasttext)")
    args = parser.parse_args()

    # Set up logging
    _setup_logging(args.log_file)

    log.info("dataset_agent starting")
    log.info(f"target_size={args.target_size} mode={args.mode} "
             f"categories={args.categories} out_dir={args.out_dir}")

    # Parse args
    target_bytes = _parse_size(args.target_size)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    if args.mix:
        raw_fracs = args.mix.split(",")
        mix = {}
        for raw in raw_fracs:
            if "=" not in raw:
                continue
            k, v = raw.split("=", 1)
            mix[k.strip()] = float(v.strip())
        total_frac = sum(mix.values())
        if total_frac > 0:
            mix = {k: v / total_frac for k, v in mix.items()}
    else:
        mix = {c: 1.0 / len(categories) for c in categories}

    hf_datasets = _parse_hf_datasets(args.hf_datasets)
    kaggle_datasets = _parse_kaggle_datasets(args.kaggle_datasets)
    public_sources = [s.strip() for s in args.public_sources.split(",")
                      if s.strip()] if args.public_sources else []
    use_llm_judge = not args.no_llm_judge

    # Extended quality filter config
    use_extended_quality = not args.no_extended_quality
    quality_thresholds = {
        "max_compression_ratio": args.max_compression_ratio,
        "max_line_repetition": args.max_line_repetition,
        "max_adjacent_repetition": args.max_adjacent_repetition,
        "min_vocab_diversity": args.min_vocab_diversity,
        "max_short_line_ratio": args.max_short_line_ratio,
        "max_flagged_ngram_ratio": args.max_flagged_ngram_ratio,
    } if use_extended_quality else None
    target_langs = set(l.strip() for l in args.target_langs.split(",")
                       if l.strip()) if args.target_langs else None

    # Build category configs
    category_configs = []
    for cat in categories:
        frac = mix.get(cat, 0)
        if frac <= 0:
            continue
        budget = int(target_bytes * frac)
        if budget <= 0:
            continue
        category_configs.append({
            "category": cat,
            "budget": budget,
        })

    if not category_configs:
        log.error("No categories with positive budget -- check --mix and --categories")
        sys.exit(1)

    # Run
    os.makedirs(args.out_dir, exist_ok=True)
    asyncio.run(_run_categories(
        categories=category_configs,
        out_dir=args.out_dir,
        mode=args.mode,
        min_doc_chars=args.min_doc_chars,
        concurrency=args.concurrency,
        use_llm_judge=use_llm_judge,
        public_sources=public_sources,
        public_only=args.public_only,
        hf_datasets=hf_datasets if hf_datasets else None,
        kaggle_datasets=kaggle_datasets if kaggle_datasets else None,
        max_category_concurrency=args.category_concurrency,
        use_extended_quality=use_extended_quality,
        quality_thresholds=quality_thresholds,
        target_langs=target_langs,
    ))


if __name__ == "__main__":
    main()
