#!/usr/bin/env python3
"""
web_scraper_mcp/server.py

MCP server providing tools for turning "I need more data about X" into clean
text -- from HTML pages, PDFs, Office docs, images, and video/audio.

Tools:
    web_search(query, max_results)  -> list of {title, url, snippet}
    fetch_page(url)                 -> raw HTML text (truncated) + status
    fetch_binary(url)               -> base64 bytes + content-type + status
    extract_content(url)            -> format-agnostic: auto-detects format
                                        and returns clean text + metadata.
                                        HTML pages go through crawl4ai first
                                        (real headless-browser fetch +
                                        pruned markdown), falling back to
                                        httpx + trafilatura/readability.
    deep_crawl(seed_url, ...)       -> BFS-crawl outward from a seed URL,
                                        extracting up to max_pages in ONE call
    extract_article(url)            -> deprecated alias for extract_content
    transcribe_media(url)           -> video/audio -> transcript
    healthcheck()                   -> connectivity/config probe

Run with:
    python server.py                # stdio transport, for MCP-aware clients
"""

import asyncio
import atexit
import base64
import concurrent.futures
import os
import sys
import time
import urllib.parse
import urllib.robotparser
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extractors
import net_utils
import crawl4ai_backend

mcp = FastMCP(
    "web-scraper",
    host=os.environ.get("MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("MCP_PORT", "8000")),
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_TEXT_BYTES = int(os.environ.get("SCRAPER_MAX_TEXT_MB", "10")) * 1024 * 1024
MAX_BINARY_BYTES = int(os.environ.get("SCRAPER_MAX_BINARY_MB", "50")) * 1024 * 1024
FETCH_TIMEOUT = float(os.environ.get("SCRAPER_FETCH_TIMEOUT", "20"))
RETRY_ATTEMPTS = int(os.environ.get("SCRAPER_RETRY_ATTEMPTS", "3"))

# "auto" (default): try crawl4ai first for HTML, fall back to httpx.
# "crawl4ai": require crawl4ai, surface its errors.
# "httpx": disable crawl4ai entirely.
HTML_BACKEND = os.environ.get("SCRAPER_HTML_BACKEND", "auto").strip().lower()
if HTML_BACKEND not in ("auto", "crawl4ai", "httpx"):
    print(f"[config] SCRAPER_HTML_BACKEND={HTML_BACKEND!r} not recognized "
          f"(expected auto/crawl4ai/httpx) -- defaulting to 'auto'", file=sys.stderr)
    HTML_BACKEND = "auto"

_allowed_env = os.environ.get("SCRAPER_ALLOWED_DOMAINS", "")
_blocked_env = os.environ.get("SCRAPER_BLOCKED_DOMAINS", "")
ALLOWED_DOMAINS = {d.strip().lower() for d in _allowed_env.split(",") if d.strip()}
BLOCKED_DOMAINS = {d.strip().lower() for d in _blocked_env.split(",") if d.strip()}

# ---------------------------------------------------------------------------
# CPU-bound work offload
# ---------------------------------------------------------------------------

EXTRACT_WORKERS = int(os.environ.get("SCRAPER_EXTRACT_WORKERS", str(os.cpu_count() or 4)))
_extract_pool = concurrent.futures.ProcessPoolExecutor(max_workers=EXTRACT_WORKERS)

MEDIA_WORKERS = int(os.environ.get("SCRAPER_MEDIA_WORKERS",
                    str(max(1, (os.cpu_count() or 4) // 2))))
_media_pool = concurrent.futures.ProcessPoolExecutor(max_workers=MEDIA_WORKERS)

_io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=16, thread_name_prefix="scraper-io")


async def _run_cpu(fn, *args):
    """Run a synchronous, CPU-bound extractor in the process pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_extract_pool, fn, *args)


async def _run_io(fn, *args):
    """Run a synchronous, I/O-bound function in the thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_io_pool, fn, *args)

VIDEO_MEDIA_DOMAINS = {
    "youtube.com", "youtu.be", "vimeo.com", "dailymotion.com", "twitch.tv",
    "tiktok.com", "soundcloud.com", "podcasts.apple.com",
}
UNSUPPORTED_DOMAINS = {
    "instagram.com", "facebook.com", "x.com", "twitter.com",
}


def _host(url: str) -> str:
    h = urllib.parse.urlparse(url).netloc.lower()
    return h[4:] if h.startswith("www.") else h


def _domain_allowed(url: str) -> Optional[str]:
    """Returns None if allowed, else a human-readable reason it's blocked."""
    host = _host(url)
    if ALLOWED_DOMAINS and not any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS):
        return f"{host} not in SCRAPER_ALLOWED_DOMAINS allow-list"
    if any(host == d or host.endswith("." + d) for d in BLOCKED_DOMAINS):
        return f"{host} is in SCRAPER_BLOCKED_DOMAINS"
    if any(host == d or host.endswith("." + d) for d in UNSUPPORTED_DOMAINS):
        return f"{host} is a login-walled/feed-only platform, not supported"
    return None


def _is_video_or_media_url(url: str) -> bool:
    host = _host(url)
    return any(host == d or host.endswith("." + d) for d in VIDEO_MEDIA_DOMAINS)


def _default_headers() -> dict:
    return {
        "User-Agent": net_utils.user_agents.get(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }


def _client_kwargs() -> dict:
    kwargs = {"follow_redirects": True, "timeout": FETCH_TIMEOUT}
    proxy = net_utils.proxies.get()
    if proxy:
        kwargs["proxy"] = proxy
    return kwargs


_robots_cache: dict = {}
_last_request_time: dict = {}
MIN_SECONDS_BETWEEN_REQUESTS_PER_HOST = float(os.environ.get("SCRAPER_MIN_HOST_INTERVAL", "2.0"))

_robots_locks: dict = {}


async def _robots_allowed(url: str) -> bool:
    host = _host(url)
    if host in _robots_cache:
        rp = _robots_cache[host]
        return True if rp is None else rp.can_fetch(net_utils.user_agents.get(), url)

    lock = _robots_locks.setdefault(host, asyncio.Lock())
    async with lock:
        if host not in _robots_cache:
            rp = urllib.robotparser.RobotFileParser()
            robots_url = f"{urllib.parse.urlparse(url).scheme}://{host}/robots.txt"
            try:
                async with httpx.AsyncClient(timeout=5, headers=_default_headers()) as client:
                    resp = await client.get(robots_url)
                rp.parse(resp.text.splitlines())
                _robots_cache[host] = rp
            except Exception:
                _robots_cache[host] = None

    rp = _robots_cache[host]
    if rp is None:
        return True
    return rp.can_fetch(net_utils.user_agents.get(), url)


async def _rate_limit(url: str):
    host = _host(url)
    now = time.monotonic()
    last = _last_request_time.get(host, 0)
    wait = MIN_SECONDS_BETWEEN_REQUESTS_PER_HOST - (now - last)
    if wait > 0:
        await asyncio.sleep(wait)
    _last_request_time[host] = time.monotonic()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _web_search_sync(query: str, max_results: int) -> list[dict]:
    from ddgs import DDGS
    try:
        from ddgs.exceptions import DDGSException
    except ImportError:
        DDGSException = Exception

    backends_env = os.environ.get("DDGS_BACKENDS")
    backends = [b.strip() for b in backends_env.split(",")] if backends_env else \
        ["brave", "yandex", "auto"]

    last_err = None
    for backend in backends:
        for attempt in range(2):
            try:
                results = []
                skipped = 0
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=max_results, backend=backend):
                        url = r.get("href", "")
                        if "bing.com/aclick" in url:
                            continue
                        reason = _domain_allowed(url)
                        if reason:
                            skipped += 1
                            continue
                        results.append({"title": r.get("title", ""), "url": url,
                                         "snippet": r.get("body", "")})
                if results:
                    print(f"[web_search] {query!r} via backend={backend} -> {len(results)} hits"
                          + (f" ({skipped} filtered by domain rules)" if skipped else ""),
                          file=sys.stderr, flush=True)
                    return results
                last_err = f"backend={backend} returned zero results (likely rate-limited/blocked)"
            except DDGSException as e:
                last_err = f"backend={backend}: {type(e).__name__}: {e}"
            except Exception as e:
                last_err = f"backend={backend}: {type(e).__name__}: {e}"
            time.sleep(1)

    print(f"[web_search] {query!r} FAILED all backends -- {last_err}", file=sys.stderr, flush=True)
    return [{"error": last_err or "unknown search failure", "query": query}]


@mcp.tool()
async def web_search(query: str, max_results: int = 10) -> list[dict]:
    """Search the public web and return title/url/snippet hits.

    Use this to discover candidate pages for a topic before fetching them.
    Keep queries short and specific (3-8 words).

    IMPORTANT: never fails silently -- if every backend errors out, returns
    a single-item list with {"error": "...", "query": "..."}.
    """
    return await _run_io(_web_search_sync, query, max_results)


# ---------------------------------------------------------------------------
# Fetching (text + binary)
# ---------------------------------------------------------------------------

async def _do_fetch(url: str, max_bytes: int, want_text: bool) -> dict:
    reason = _domain_allowed(url)
    if reason:
        return {"status": None, "content_type": None, "error": f"blocked: {reason}"}
    if not await _robots_allowed(url):
        return {"status": None, "content_type": None, "error": "disallowed by robots.txt"}

    await _rate_limit(url)

    async def _attempt():
        async with httpx.AsyncClient(headers=_default_headers(), **_client_kwargs()) as client:
            resp = await client.get(url)
        if net_utils.is_retryable_status(resp.status_code):
            resp.raise_for_status()
        return resp

    try:
        def _on_retry(attempt, exc):
            print(f"[fetch] retry {attempt}/{RETRY_ATTEMPTS} for {url}: {exc}",
                  file=sys.stderr, flush=True)
        resp = await net_utils.retry_async(
            _attempt, attempts=RETRY_ATTEMPTS,
            retry_on=(httpx.HTTPStatusError, httpx.TransportError, httpx.TimeoutException),
            on_retry=_on_retry,
        )
    except httpx.HTTPStatusError as e:
        return {"status": e.response.status_code, "content_type": None,
                "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        return {"status": None, "content_type": None, "error": str(e)}

    content_type = resp.headers.get("content-type", "")
    if resp.status_code >= 400:
        return {"status": resp.status_code, "content_type": content_type,
                "error": f"HTTP {resp.status_code}"}

    body = resp.content[:max_bytes]
    truncated = len(resp.content) > max_bytes
    if want_text:
        return {"status": resp.status_code, "content_type": content_type,
                "text": body.decode(resp.encoding or "utf-8", errors="replace"),
                "truncated": truncated, "error": None}
    return {"status": resp.status_code, "content_type": content_type,
            "data": body, "truncated": truncated, "error": None}


@mcp.tool()
async def fetch_page(url: str, max_chars: int = 200_000) -> dict:
    """Fetch the raw text/HTML content at a URL (GET only). Honors
    robots.txt, domain allow/deny lists, and a per-host rate limit; retries
    transient failures with backoff.
    Returns {status, content_type, html, error}."""
    result = await _do_fetch(url, MAX_TEXT_BYTES, want_text=True)
    if result.get("error"):
        return {"status": result.get("status"), "content_type": result.get("content_type"),
                "html": "", "error": result["error"]}
    return {"status": result["status"], "content_type": result["content_type"],
            "html": result["text"][:max_chars], "error": None}


@mcp.tool()
async def fetch_binary(url: str) -> dict:
    """Fetch raw bytes at a URL (GET only) for non-HTML content -- PDFs,
    images, Office docs, etc. Returns base64-encoded content since MCP
    tool results are JSON.
    Returns {status, content_type, data_base64, size_bytes, truncated, error}.
    """
    result = await _do_fetch(url, MAX_BINARY_BYTES, want_text=False)
    if result.get("error"):
        return {"status": result.get("status"), "content_type": result.get("content_type"),
                "data_base64": "", "size_bytes": 0, "truncated": False, "error": result["error"]}
    data = result["data"]
    return {"status": result["status"], "content_type": result["content_type"],
            "data_base64": base64.b64encode(data).decode("ascii"),
            "size_bytes": len(data), "truncated": result["truncated"], "error": None}


# ---------------------------------------------------------------------------
# Format-agnostic extraction
# ---------------------------------------------------------------------------

@mcp.tool()
async def extract_content(url: str) -> dict:
    """Fetch a URL and extract clean text, auto-detecting its format:
    HTML article, PDF (with OCR fallback for scanned pages), DOCX, PPTX,
    XLSX, CSV, image (OCR), or video/audio (captions, else local ASR).

    Returns {title, text, author, date, content_type, url, error, extra}.
    """
    reason = _domain_allowed(url)
    if reason:
        return {"title": None, "text": "", "author": None, "date": None,
                "content_type": "unknown", "url": url, "error": f"blocked: {reason}", "extra": {}}

    if _is_video_or_media_url(url):
        return await transcribe_media(url)

    content_type_hint = None
    try:
        if await _robots_allowed(url):
            async with httpx.AsyncClient(headers=_default_headers(), **_client_kwargs()) as client:
                head = await client.head(url)
                content_type_hint = head.headers.get("content-type")
    except Exception:
        pass

    kind = extractors.detect_content_kind(url, content_type_hint)

    if kind == "html":
        if HTML_BACKEND in ("auto", "crawl4ai"):
            await _rate_limit(url)
            c4_result = await crawl4ai_backend.extract_single(url)
            if not c4_result.get("error"):
                return c4_result
            if HTML_BACKEND == "crawl4ai":
                return c4_result
            print(f"[extract_content] crawl4ai path failed for {url} ({c4_result['error']}) "
                  f"-- falling back to httpx+trafilatura", file=sys.stderr, flush=True)

        page = await fetch_page(url)
        if page["error"] or not page["html"]:
            return {"title": None, "text": "", "author": None, "date": None,
                    "content_type": "html", "url": url,
                    "error": page["error"] or "empty page", "extra": {}}
        return await _run_cpu(extractors.extract_html, page["html"], url)

    if kind in ("video", "audio"):
        return await transcribe_media(url)

    binary = await fetch_binary(url)
    if binary["error"]:
        return {"title": None, "text": "", "author": None, "date": None,
                "content_type": kind, "url": url, "error": binary["error"], "extra": {}}
    data = base64.b64decode(binary["data_base64"])
    return await _run_cpu(extractors.extract_from_bytes, kind, data, url)


@mcp.tool()
async def deep_crawl(seed_url: str, max_pages: int = 20, max_depth: int = 2,
                      keywords: str = "", same_domain_only: bool = True) -> list[dict]:
    """Crawl outward from a seed URL using crawl4ai's JS-aware headless-browser
    crawler, following in-page links up to max_depth hops and extracting up
    to max_pages pages in ONE call.

    keywords: optional comma-separated terms to prioritize which discovered
    links get visited first.
    same_domain_only: if True (default), never follows links off the seed
    URL's own host.

    Returns a list of {title, text, url, ...} dicts -- same shape as
    extract_content -- one per successfully extracted page.
    """
    reason = _domain_allowed(seed_url)
    if reason:
        return [{"error": f"blocked: {reason}", "url": seed_url}]

    kw_list = [k.strip() for k in keywords.split(",") if k.strip()] or None
    await _rate_limit(seed_url)
    pages = await crawl4ai_backend.deep_crawl(
        seed_url, max_pages=max_pages, max_depth=max_depth,
        keywords=kw_list, same_domain_only=same_domain_only,
    )
    return [p for p in pages if not _domain_allowed(p.get("url", seed_url))]


@mcp.tool()
async def extract_article(url: str) -> dict:
    """DEPRECATED alias for extract_content, kept for backward compatibility.
    New code should call extract_content directly.
    Returns {title, text, author, date, url, error}."""
    result = await extract_content(url)
    return {"title": result["title"], "text": result["text"], "author": result["author"],
            "date": result["date"], "url": result["url"], "error": result["error"]}


@mcp.tool()
async def transcribe_media(url: str, asr_fallback: bool = True,
                            asr_model_size: str = "base",
                            max_duration_seconds: int = 3600) -> dict:
    """Get a transcript for a video or audio URL. Prefers existing
    captions/subtitles; falls back to local ASR via faster-whisper.

    Requires: yt-dlp (always); faster-whisper + ffmpeg (only for ASR path).
    """
    reason = _domain_allowed(url)
    if reason:
        return {"title": None, "text": "", "author": None, "date": None,
                "content_type": "video", "url": url, "error": f"blocked: {reason}", "extra": {}}
    kind = extractors.detect_content_kind(url)
    if kind == "audio" and not _is_video_or_media_url(url):
        if not asr_fallback:
            return {"title": None, "text": "", "author": None, "date": None,
                    "content_type": "audio", "url": url,
                    "error": "direct audio file with asr_fallback=False", "extra": {}}
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _media_pool, extractors.extract_audio, url, False, asr_model_size)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _media_pool, extractors.extract_video, url, True, asr_fallback,
        asr_model_size, max_duration_seconds)


@mcp.tool()
async def healthcheck() -> dict:
    """Connectivity/config probe -- call before kicking off a real scrape.
    Reports which optional extraction dependencies are importable and whether
    robots.txt fetching works."""
    import importlib
    report = {
        "ddgs_importable": False,
        "robots_check_ok": False,
        "proxy_pool_enabled": net_utils.proxies.enabled,
        "formats": {},
        "error": None,
    }
    try:
        importlib.import_module("ddgs")
        report["ddgs_importable"] = True
    except Exception as e:
        report["error"] = f"ddgs import failed: {e}"

    optional_deps = {
        "pdf": ["pdfplumber"],
        "pdf_ocr": ["pdf2image", "pytesseract"],
        "docx": ["docx"],
        "pptx": ["pptx"],
        "xlsx": ["openpyxl"],
        "image_ocr": ["PIL", "pytesseract"],
        "video_captions": ["yt_dlp"],
        "video_audio_asr": ["yt_dlp", "faster_whisper"],
        "crawl4ai_html": ["crawl4ai"],
    }
    for label, modules in optional_deps.items():
        ok = True
        for m in modules:
            try:
                importlib.import_module(m)
            except Exception:
                ok = False
                break
        report["formats"][label] = ok

    report["html_backend"] = HTML_BACKEND
    report["extract_workers"] = EXTRACT_WORKERS
    if HTML_BACKEND != "httpx" and not report["formats"]["crawl4ai_html"]:
        note = ("SCRAPER_HTML_BACKEND is "
                f"{HTML_BACKEND!r} but crawl4ai isn't importable -- every HTML "
                "extraction will silently fall back to httpx+trafilatura" +
                (" (auto mode, so this is fine, just lower-fidelity)"
                 if HTML_BACKEND == "auto" else
                 " and error out (backend=crawl4ai requires it)"))
        report["error"] = (report["error"] + "; " if report["error"] else "") + note

    try:
        report["robots_check_ok"] = await _robots_allowed("https://example.com/")
    except Exception as e:
        report["error"] = (report["error"] + "; " if report["error"] else "") + f"robots check failed: {e}"
    return report


def _cleanup_crawl4ai():
    try:
        asyncio.run(crawl4ai_backend.shutdown_browser())
    except Exception:
        pass


def _cleanup_pools():
    for pool in (_extract_pool, _media_pool, _io_pool):
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


atexit.register(_cleanup_crawl4ai)
atexit.register(_cleanup_pools)


if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)
