#!/usr/bin/env python3
"""
diagnose_search.py

Test web_search backends directly (no MCP, no agent) to find why
search is returning nothing.

Usage:
    python diagnose_search.py "climate change ocean currents"
"""
import sys
import time
import traceback

query = sys.argv[1] if len(sys.argv) > 1 else "python programming tutorial"

try:
    from ddgs import DDGS
except ImportError:
    print("FAIL: `ddgs` isn't installed. Run: pip install ddgs")
    sys.exit(1)

try:
    from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
except ImportError:
    DDGSException = RatelimitException = TimeoutException = Exception

backends_to_try = ["duckduckgo", "bing", "brave", "mojeek", "yandex", "auto"]

print(f"Query: {query!r}\n")

for backend in backends_to_try:
    print(f"--- backend={backend} ---")
    start = time.time()
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5, backend=backend))
        elapsed = time.time() - start
        if results:
            print(f"  OK  ({elapsed:.1f}s) -- {len(results)} results")
            print(f"    first: {results[0].get('title', '')!r} -> {results[0].get('href', '')}")
        else:
            print(f"  EMPTY ({elapsed:.1f}s) -- zero results, no exception raised")
    except RatelimitException as e:
        print(f"  RATE-LIMITED ({time.time()-start:.1f}s): {e}")
    except TimeoutException as e:
        print(f"  TIMEOUT ({time.time()-start:.1f}s): {e}")
    except DDGSException as e:
        print(f"  DDGS ERROR ({time.time()-start:.1f}s): {type(e).__name__}: {e}")
    except Exception as e:
        print(f"  UNEXPECTED ERROR ({time.time()-start:.1f}s): {type(e).__name__}: {e}")
        traceback.print_exc()
    print()
    time.sleep(1)

print(
    "If every backend above shows RATE-LIMITED, TIMEOUT, or EMPTY: your "
    "network (likely a cloud/datacenter IP) is probably blocked by these "
    "engines' anti-bot systems. The fix is a paid search API (Brave Search, "
    "Serper, Tavily) or a residential/rotating proxy.\n"
    "If one backend returns OK: set DDGS_BACKEND to that value before "
    "running server.py."
)
