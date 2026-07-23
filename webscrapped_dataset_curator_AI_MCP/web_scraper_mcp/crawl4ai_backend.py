"""
crawl4ai_backend.py

Optional higher-fidelity HTML backend using crawl4ai (Playwright-based).
Shared browser with periodic recycling. Deep-crawl via BFS.

Crawl4ai is not a hard dependency — the main server falls back to
httpx + trafilatura/readability when it's absent.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Crawl4aiConfig:
    headless: bool = True
    recycle_every: int = 20
    timeout_ms: int = 45000
    max_pages: int = 50
    max_depth: int = 2
    same_domain_only: bool = True
    wait_until: str = "domcontentloaded"
    js_code: list = field(default_factory=list)
    user_agent: Optional[str] = None
    extra_args: list = field(default_factory=list)
    viewport_width: int = 1280
    viewport_height: int = 800


# ---------------------------------------------------------------------------
# Lazy crawl4ai import
# ---------------------------------------------------------------------------

def _crawl4ai_available() -> bool:
    try:
        import crawl4ai  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Browser manager — shared, recyclable
# ---------------------------------------------------------------------------

class _BrowserManager:
    """Manages a shared async browser with periodic recycling."""

    def __init__(self, config: Crawl4aiConfig):
        self.config = config
        self._browser = None
        self._manager = None
        self._page_count = 0
        self._lock = asyncio.Lock()

    async def _ensure_browser(self):
        if self._browser is None or self._page_count >= self.config.recycle_every:
            await self._close()
            await self._open()

    async def _open(self):
        try:
            from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
        except ImportError:
            from playwright.async_api import async_playwright
            self._manager = await async_playwright().__aenter__()
            self._browser = await self._manager.chromium.launch(
                headless=self.config.headless,
                args=self.config.extra_args,
            )
        else:
            strategy = AsyncPlaywrightCrawlerStrategy(
                headless=self.config.headless,
                extra_args=self.config.extra_args,
            )
            self._browser = strategy.playwright_browser
        self._page_count = 0

    async def _close(self):
        try:
            if self._browser is not None:
                await self._browser.close()
        except Exception:
            pass
        self._browser = None
        try:
            if self._manager is not None:
                await self._manager.__aexit__(None, None, None)
        except Exception:
            pass
        self._manager = None

    async def new_page(self):
        async with self._lock:
            await self._ensure_browser()
            self._page_count += 1
            return await self._browser.new_page(
                user_agent=self.config.user_agent,
                viewport={"width": self.config.viewport_width,
                           "height": self.config.viewport_height},
            )

    async def close(self):
        async with self._lock:
            await self._close()


_browser_managers: dict[int, _BrowserManager] = {}


def _get_browser_manager(config: Crawl4aiConfig | None = None) -> _BrowserManager:
    config = config or Crawl4aiConfig()
    key = id(config)
    if key not in _browser_managers:
        _browser_managers[key] = _BrowserManager(config)
    return _browser_managers[key]


# ---------------------------------------------------------------------------
# Single-page extraction
# ---------------------------------------------------------------------------

async def extract_single(
    url: str,
    config: Crawl4aiConfig | None = None,
    *,
    extract_markdown: bool = True,
    js_code: list | None = None,
) -> dict:
    if not _crawl4ai_available():
        return {"url": url, "markdown": "", "error": "crawl4ai not installed"}

    config = config or Crawl4aiConfig()
    mgr = _get_browser_manager(config)
    page = None

    try:
        page = await mgr.new_page()

        extra_js = list(config.js_code) + list(js_code or [])

        try:
            from crawl4ai import AsyncWebCrawler
            async with AsyncWebCrawler(
                headless=config.headless,
                browser_type="chromium",
            ) as crawler:
                result = await crawler.arun(
                    url=url,
                    word_count_threshold=10,
                    bypass_cache=True,
                    delay_before_return_html=2.0,
                    js_code=extra_js or None,
                    css_selector="article, main, .content, .post, .entry-content",
                )
                if result.success:
                    return {
                        "url": url,
                        "markdown": result.markdown.raw_markdown or "",
                        "title": getattr(result, "metadata", {}).get("title", ""),
                        "error": None,
                    }
                return {"url": url, "markdown": "", "error": result.error_message}
        except ImportError:
            pass

        try:
            resp = await page.goto(url, timeout=config.timeout_ms,
                                    wait_until=config.wait_until)
            if resp and resp.status >= 400:
                return {"url": url, "markdown": "",
                         "error": f"HTTP {resp.status}"}
        except Exception as e:
            return {"url": url, "markdown": "", "error": str(e)}

        await asyncio.sleep(2.0)

        for js in extra_js:
            try:
                await page.evaluate(js)
            except Exception:
                pass

        if extract_markdown:
            try:
                markdown = await page.evaluate(
                    "el => el.innerText", "article, main, .content, .post, body"
                )
                if markdown and len(markdown) > 500:
                    return {"url": url, "markdown": markdown.strip(), "error": None}
            except Exception:
                pass

        try:
            html_content = await page.content()
            import re as _re
            text = _re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=_re.S)
            text = _re.sub(r"<style[^>]*>.*?</style>", "", text, flags=_re.S)
            text = _re.sub(r"<[^>]+>", " ", text)
            text = _re.sub(r"\s+", " ", text).strip()
            return {"url": url, "markdown": text[:50000], "error": None}
        except Exception as e:
            return {"url": url, "markdown": "", "error": str(e)}

    finally:
        try:
            if page is not None:
                await page.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Deep crawl (BFS)
# ---------------------------------------------------------------------------

def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    return normalized


def _same_domain(a: str, b: str) -> bool:
    return urlparse(a).netloc == urlparse(b).netloc


async def deep_crawl(
    seed_url: str,
    config: Crawl4aiConfig | None = None,
    on_page=None,
) -> list[dict]:
    config = config or Crawl4aiConfig()
    max_depth = config.max_depth
    max_pages = config.max_pages

    from collections import deque
    queue: deque = deque([(seed_url, 0)])
    visited: set = set()
    results: list = []
    visited.add(_normalize_url(seed_url))

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return [{"url": seed_url, "markdown": "",
                  "error": "playwright not installed"}]

    mgr = _get_browser_manager(config)
    page = None

    try:
        page = await mgr.new_page()

        while queue and len(results) < max_pages:
            url, depth = queue.popleft()
            result = await extract_single(url, config)
            results.append(result)

            if on_page:
                try:
                    on_page(result)
                except Exception:
                    pass

            if depth < max_depth:
                links = await _extract_links(page, url)
                for link in links:
                    norm = _normalize_url(link)
                    if norm in visited:
                        continue
                    if config.same_domain_only and not _same_domain(seed_url, link):
                        continue
                    visited.add(norm)
                    queue.append((link, depth + 1))

    finally:
        pass

    return results


async def _extract_links(page, base_url: str) -> list[str]:
    links: list[str] = []
    try:
        raw_links = await page.evaluate("""
            () => {
                const anchors = document.querySelectorAll('a[href]');
                return Array.from(anchors).map(a => a.href).filter(h =>
                    h.startsWith('http://') || h.startsWith('https://')
                );
            }
        """)
        for href in raw_links:
            full = urljoin(base_url, href)
            full = re.sub(r"#.*$", "", full)
            if full.startswith("http"):
                links.append(full)
    except Exception:
        pass
    return links


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

async def shutdown_browser():
    for mgr in _browser_managers.values():
        await mgr.close()
    _browser_managers.clear()
