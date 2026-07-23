#!/usr/bin/env python3
"""
ollama_judge.py

Ollama remote client for generating preference pairs for DPO training.

Instead of human annotators, an Ollama model running on a (local or remote)
server acts as the judge to:
  1. Generate multiple candidate completions for a given prompt.
  2. Rank them via pairwise comparisons or direct scoring.
  3. Return (chosen, rejected) preference pairs ready for DPO packing.

Usage:
    # One-shot: generate preference pairs for a single prompt
    python ollama_judge.py --prompt "What is 2+2?" --url http://remote:11434 --model qwen2.5:7b

    # Batch mode: read prompts from JSONL, write preference pairs
    python ollama_judge.py --input ./data.jsonl --output ./prefs.jsonl --num-candidates 4

    # Connectivity test
    python ollama_judge.py --test

Environment variables:
    OLLAMA_BASE_URL   — default "http://localhost:11434"
    OLLAMA_JUDGE_MODEL — default "qwen2.5:7b-instruct"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="[ollama_judge] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_JUDGE_MODEL = os.environ.get("OLLAMA_JUDGE_MODEL", "qwen2.5:7b-instruct")
DEFAULT_GEN_MODEL = os.environ.get("OLLAMA_GEN_MODEL", "")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class PreferencePair:
    """A single preference training example."""

    prompt: str
    chosen: str
    rejected: str
    chosen_thinking: str = ""
    rejected_thinking: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class Completion:
    """A candidate completion with optional thinking."""

    text: str
    thinking: str = ""


# ---------------------------------------------------------------------------
# Ollama HTTP Client
# ---------------------------------------------------------------------------


class OllamaClient:
    """
    Lightweight client for Ollama's REST API.

    Covers the two endpoints used during DPO data generation:
        - POST /api/generate  (stateless inference)
        - POST /api/chat      (chat-formatted inference, preferred for
                               instruction-tuned judge models)
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: int = 120,
        max_retries: int = 3,
    ):
        if httpx is None:
            raise ImportError(
                "httpx is required for ollama_judge.py. Install it with:\n"
                "  pip install httpx"
            )

        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s),
        )

    # ------------------------------------------------------------------
    # Health / Info
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            resp = self._client.get("/api/tags", timeout=10)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Return list of model names available on the server."""
        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as exc:
            log.warning("Failed to list models: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Generate (stateless, for generating candidate completions)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        model: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Call POST /api/generate with a plain-text prompt.

        Returns the generated text string.
        """
        payload: Dict = {
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system
        if stop:
            payload["options"]["stop"] = stop

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.post(
                    "/api/generate",
                    json=payload,
                )
                resp.raise_for_status()
                # Ollama streams JSON lines; collect the full response
                lines = resp.text.strip().split("\n")
                full_text = ""
                for line in lines:
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        full_text += chunk.get("response", "")
                    except json.JSONDecodeError:
                        pass
                return full_text.strip()
            except Exception as exc:
                log.warning(
                    "Generate attempt %d/%d failed: %s", attempt, self.max_retries, exc
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Ollama generate failed after {self.max_retries} attempts")

    # ------------------------------------------------------------------
    # Chat (for judge — structured role-based prompting)
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 256,
    ) -> str:
        """
        Call POST /api/chat with a list of message dicts.

        Each message: {"role": "user"|"assistant"|"system", "content": "..."}

        Returns the assistant's reply text.
        """
        payload: Dict = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.post("/api/chat", json=payload)
                resp.raise_for_status()
                lines = resp.text.strip().split("\n")
                full_text = ""
                for line in lines:
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        full_text += chunk.get("message", {}).get("content", "")
                    except json.JSONDecodeError:
                        pass
                return full_text.strip()
            except Exception as exc:
                log.warning(
                    "Chat attempt %d/%d failed: %s", attempt, self.max_retries, exc
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Ollama chat failed after {self.max_retries} attempts")

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Preference generation logic
# ---------------------------------------------------------------------------


def generate_candidates(
    client: OllamaClient,
    prompt: str,
    gen_model: str,
    num_candidates: int = 4,
    temperature: float = 0.8,
    max_tokens: int = 512,
    system_prompt: str = "",
) -> List[Completion]:
    """
    Generate ``num_candidates`` diverse completions for a prompt.

    Uses the Ollama generate endpoint. Each completion is independent so calls
    are parallelised via a thread pool.

    Returns:
        List of Completion objects, one per generated candidate.
    """
    if not gen_model:
        raise ValueError("--gen-model (or OLLAMA_GEN_MODEL) is required for candidate generation")

    candidates: List[Optional[Completion]] = [None] * num_candidates
    # Vary temperature slightly for diversity
    temps = [max(0.1, temperature + random.uniform(-0.2, 0.2)) for _ in range(num_candidates)]

    with ThreadPoolExecutor(max_workers=min(num_candidates, 8)) as pool:
        futures = {}
        for i in range(num_candidates):
            fut = pool.submit(
                client.generate,
                prompt=prompt,
                model=gen_model,
                system=system_prompt,
                temperature=temps[i],
                max_tokens=max_tokens,
            )
            futures[fut] = i

        for fut in as_completed(futures):
            i = futures[fut]
            try:
                text = fut.result()
                candidates[i] = Completion(text=text, thinking="")
            except Exception as exc:
                log.warning("Candidate %d failed: %s", i, exc)
                candidates[i] = Completion(text="", thinking="")

    return [c for c in candidates if c and c.text]


# ---------------------------------------------------------------------------
# Judge — pairwise comparison
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an expert AI assistant evaluator. Your job is to judge which of two model responses is better given a user's prompt.

Evaluate the responses on these criteria:
1. **Correctness** — Is the answer factually accurate?
2. **Helpfulness** — Does it directly address the user's query?
3. **Clarity** — Is it well-written and easy to understand?
4. **Safety** — Does it avoid harmful or biased content?

Respond with ONLY a JSON object in exactly this format — no other text:
{"verdict": "A" or "B", "reasoning": "One short sentence explaining your choice."}

- "A" means the first response (under "Response A") is better.
- "B" means the second response (under "Response B") is better.
- If both are equivalent in quality, choose the safer, more helpful one — always pick one."""


def judge_pair(
    client: OllamaClient,
    prompt: str,
    completion_a: str,
    completion_b: str,
    judge_model: str,
) -> Tuple[str, str]:
    """
    Ask the judge model which of two completions is better.

    Args:
        client: Connected OllamaClient.
        prompt: The original user prompt.
        completion_a: First candidate text.
        completion_b: Second candidate text.
        judge_model: Name of the judge model on the Ollama server.

    Returns:
        (verdict, reasoning) where verdict is "A" or "B".
        "A" means completion_a was judged better (chosen).
    """
    user_message = f"""User Prompt: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Which response is better? Respond with the JSON verdict format."""

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    reply = client.chat(messages, model=judge_model, temperature=0.1, max_tokens=256)

    # Attempt to parse JSON from the reply
    verdict = "A"
    reasoning = ""
    try:
        # Find JSON in the response
        start = reply.find("{")
        end = reply.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(reply[start:end])
            v = parsed.get("verdict", "A").strip().upper()
            if v in ("A", "B"):
                verdict = v
            reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback: look for "A" or "B" in first few chars
        reply_upper = reply.strip().upper()
        for token in ["VERDICT: A", "VERDICT: B", '"VERDICT": "A"', '"VERDICT": "B"']:
            if token in reply_upper:
                verdict = token[-1]
                break
        # Last-resort check
        if verdict not in ("A", "B"):
            verdict = "A"

    return verdict, reasoning


# ---------------------------------------------------------------------------
# Full pipeline: generate → rank → preference pairs
# ---------------------------------------------------------------------------


def create_preference_pairs(
    client: OllamaClient,
    prompt: str,
    gen_model: str,
    judge_model: str,
    num_candidates: int = 4,
    max_pairs: int = 2,
    temperature: float = 0.8,
    max_tokens: int = 512,
    system_prompt: str = "",
) -> List[PreferencePair]:
    """
    Generate candidate completions for a prompt, rank via pairwise
    elimination, and return ``max_pairs`` (chosen, rejected) preference pairs.

    Algorithm:
        1. Generate N candidate completions.
        2. Run a mini-tournament of pairwise comparisons.
        3. For each comparison, the loser is "rejected" and the winner
           advances; a higher-rated completion can appear as "chosen"
           in multiple pairs.
        4. Return at most ``max_pairs`` unique (chosen, rejected) pairs.

    Returns:
        List of PreferencePair objects.
    """
    candidates = generate_candidates(
        client, prompt, gen_model,
        num_candidates=num_candidates,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )

    if len(candidates) < 2:
        log.warning(
            "Not enough candidates (%d < 2) for prompt: %.60s",
            len(candidates), prompt,
        )
        return []

    # Round-robin: compare every pair, collect preferences
    pairs: List[PreferencePair] = []
    used_indices = set()

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if len(pairs) >= max_pairs:
                break

            verdict, reasoning = judge_pair(
                client, prompt,
                candidates[i].text, candidates[j].text,
                judge_model,
            )

            if verdict == "A":
                chosen = candidates[i]
                rejected = candidates[j]
            else:
                chosen = candidates[j]
                rejected = candidates[i]

            # Avoid duplicate pairs
            pair_key = (chosen.text[:80], rejected.text[:80])
            if pair_key not in used_indices:
                used_indices.add(pair_key)
                pairs.append(PreferencePair(
                    prompt=prompt,
                    chosen=chosen.text,
                    rejected=rejected.text,
                    chosen_thinking=chosen.thinking,
                    rejected_thinking=rejected.thinking,
                    metadata={"judge_reasoning": reasoning} if reasoning else {},
                ))

        if len(pairs) >= max_pairs:
            break

    return pairs


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def batch_create_preference_pairs(
    input_path: str,
    output_path: str,
    client: OllamaClient,
    gen_model: str,
    judge_model: str,
    num_candidates: int = 4,
    max_pairs_per_prompt: int = 1,
    temperature: float = 0.8,
    max_tokens: int = 512,
    system_prompt: str = "",
    max_prompts: int = 0,
) -> int:
    """
    Read prompts from a JSONL file, generate preference pairs for each,
    and write the resulting preference pairs as JSONL.

    Input JSONL format (one per line):
        {"prompt": "..."}
        or
        {"prompt": "...", "answers": [{"text": "...", "thinking": "..."}, ...]}

    Output JSONL format (one PreferencePair per line):
        {"prompt": "...", "chosen": "...", "rejected": "...",
         "chosen_thinking": "", "rejected_thinking": "",
         "metadata": {...}}

    Returns:
        Number of preference pairs written.
    """
    total_written = 0
    prompts_processed = 0

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Skipping invalid JSON: %s", exc)
                continue

            prompt = record.get("prompt", "").strip()
            if not prompt:
                continue

            pairs = create_preference_pairs(
                client=client,
                prompt=prompt,
                gen_model=gen_model,
                judge_model=judge_model,
                num_candidates=num_candidates,
                max_pairs=max_pairs_per_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )

            for pair in pairs:
                fout.write(json.dumps({
                    "prompt": pair.prompt,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                    "chosen_thinking": pair.chosen_thinking,
                    "rejected_thinking": pair.rejected_thinking,
                    "metadata": pair.metadata,
                }, ensure_ascii=False) + "\n")
                total_written += 1

            prompts_processed += 1
            log.info(
                "Processed prompt %d → %d pairs (total %d so far)",
                prompts_processed, len(pairs), total_written,
            )

            if max_prompts > 0 and prompts_processed >= max_prompts:
                log.info("Reached --max-prompts limit (%d)", max_prompts)
                break

    log.info(
        "Batch complete: %d prompts → %d preference pairs written to %s",
        prompts_processed, total_written, output_path,
    )
    return total_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ollama judge — generate preference pairs for DPO training "
                    "using a remote Ollama server.",
    )

    # Server connection
    p.add_argument("--url", default=DEFAULT_BASE_URL,
                   help=f"Ollama server URL (default: {DEFAULT_BASE_URL})")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                   help=f"Model name for the judge (default: {DEFAULT_JUDGE_MODEL})")
    p.add_argument("--gen-model", default="",
                   help=f"Model for generating candidates (default: same as judge)")
    p.add_argument("--timeout", type=int, default=120,
                   help="HTTP timeout in seconds")

    # Generation
    p.add_argument("--num-candidates", type=int, default=4,
                   help="Number of candidates to generate per prompt (default: 4)")
    p.add_argument("--max-pairs", type=int, default=1,
                   help="Max preference pairs per prompt (default: 1)")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature for candidate generation (default: 0.8)")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="Max tokens per candidate (default: 512)")
    p.add_argument("--system-prompt", default="",
                   help="Optional system prompt for candidate generation")
    p.add_argument("--max-prompts", type=int, default=0,
                   help="Max prompts to process in batch mode (0 = all)")

    # Modes
    p.add_argument("--test", action="store_true",
                   help="Run connectivity test and exit")
    p.add_argument("--prompt", default=None,
                   help="Single prompt to process (one-shot mode)")
    p.add_argument("--input", default=None,
                   help="Input JSONL file (batch mode)")
    p.add_argument("--output", default="./preference_pairs.jsonl",
                   help="Output JSONL file (default: ./preference_pairs.jsonl)")

    return p.parse_args()


def main():
    args = parse_args()

    # Determine gen model: default to judge model if not separately specified
    gen_model = args.gen_model or args.judge_model

    # Create client
    client = OllamaClient(
        base_url=args.url,
        timeout_s=args.timeout,
    )

    # ---- test mode ----
    if args.test:
        if not client.ping():
            print(f"ERROR: Cannot reach Ollama at {args.url}")
            sys.exit(1)
        print(f"✓ Ollama server at {args.url} is reachable")

        models = client.list_models()
        if models:
            print(f"  Available models: {', '.join(models[:10])}")
        else:
            print("  (No models found or failed to list)")

        # Quick judge test
        if args.judge_model:
            print(f"\n  Testing judge with {args.judge_model}...")
            try:
                verdict, reason = judge_pair(
                    client,
                    "What is 2+2?",
                    "4", "5",
                    args.judge_model,
                )
                print(f"  ✓ Judge works! Verdict: {verdict} — {reason}")
            except Exception as exc:
                print(f"  ✗ Judge test failed: {exc}")

        client.close()
        return

    # ---- single prompt mode ----
    if args.prompt:
        print(f"Generating preference pairs for: {args.prompt[:60]}...")
        pairs = create_preference_pairs(
            client=client,
            prompt=args.prompt,
            gen_model=gen_model,
            judge_model=args.judge_model,
            num_candidates=args.num_candidates,
            max_pairs=args.max_pairs,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            system_prompt=args.system_prompt,
        )
        print(f"Generated {len(pairs)} preference pair(s):\n")
        for i, pair in enumerate(pairs):
            print(f"  Pair {i + 1}:")
            print(f"    Chosen:   {pair.chosen[:80]}...")
            print(f"    Rejected: {pair.rejected[:80]}...")
            if pair.metadata.get("judge_reasoning"):
                print(f"    Reason:   {pair.metadata['judge_reasoning']}")
            print()
        client.close()
        return

    # ---- batch mode ----
    if args.input:
        if not client.ping():
            print(f"ERROR: Cannot reach Ollama at {args.url}")
            sys.exit(1)
        print(f"Batch processing: {args.input} → {args.output}")
        n = batch_create_preference_pairs(
            input_path=args.input,
            output_path=args.output,
            client=client,
            gen_model=gen_model,
            judge_model=args.judge_model,
            num_candidates=args.num_candidates,
            max_pairs_per_prompt=args.max_pairs,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            system_prompt=args.system_prompt,
            max_prompts=args.max_prompts,
        )
        print(f"Done: {n} preference pairs written to {args.output}")
        client.close()
        return

    # No mode selected
    print("No action specified. Use --test, --prompt, or --input.")
    print("Run with --help for usage.")
    client.close()


if __name__ == "__main__":
    main()
