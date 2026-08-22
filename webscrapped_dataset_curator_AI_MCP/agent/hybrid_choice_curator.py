#!/usr/bin/env python3
"""
hybrid_choice_curator.py

Build a hybrid non-reasoning + reasoning SFT-style dataset with choice / multiple-
choice questions, synthesised by an Ollama-served LLM via LangChain + LangGraph.

Layout:

    [plan_questions]  -> reasoning model drafts a JSON list of question specs
                       (one per category, prompt, choices, answer_index, rationale)
    [synthesize]      -> for each spec, the same model writes the full
                       {prompt, choices, answer_index, rationale, answer_text,
                        thinking, source, category} record
    [validate]        -> Pydantic schema check + duplicate guard + sequence-length
                       cap; failed records loop back to [synthesize] with the
                       error in the state
    [persist]         -> ShardWriter appends jsonl to out-dir/category/

The CLI exposes --sequence-length (hard cap on the final prompt+thinking+answer
text length, used to drop oversize records). All LangChain / LangGraph imports
are lazy so the module is importable without those packages installed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import lru_cache
from typing import Any, Optional, TypedDict

# Same env-var contract as codegen_graph.py
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Hard cap that overrides any user value: a record whose
# prompt+thinking+answer chars exceed this is dropped at validate().
_HARD_SEQUENCE_CAP = 16384

DEFAULT_CATEGORIES = ["reasoning", "knowledge", "code", "math"]
DEFAULT_QUESTIONS_PER_CATEGORY = 5


# ---------------------------------------------------------------------------
# Pydantic schema for a single choice record (lazy import only at use sites)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class HybridState(TypedDict, total=False):
    # Inputs
    model: str
    categories: list
    questions_per_category: int
    sequence_length: int            # soft cap (user-requested)
    hard_cap: int                   # hard cap (constant, but stored here for clarity)
    reasoning_required: bool        # whether to include a "thinking" field
    seed_topics: list

    # Graph-produced
    plan: Optional[list]            # list of question specs across all categories
    records: list                   # accepted records, ready to persist
    failures: int                   # count of validate-fail -> re-synthesize loops
    attempt_counts: dict            # spec_index -> attempts


# ---------------------------------------------------------------------------
# Lazy LangChain / LangGraph bootstrap (LRU cache so we pay once)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_chat_model(model: str):
    """Return a ChatOllama bound to the named reasoning/chat model."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model,
        base_url=OLLAMA_URL,
        temperature=0.4,
        num_predict=4096,
    )


def _ensure_pydantic():
    from pydantic import BaseModel, Field, conlist

    class ChoiceRecord(BaseModel):
        prompt: str = Field(min_length=5)
        choices: conlist(str, min_length=2, max_length=8)  # type: ignore[valid-type]
        answer_index: int = Field(ge=0)
        answer_text: str = Field(min_length=1)
        category: str
        source: str = "ollama:synthetic"
        thinking: str = ""          # populated when reasoning_required=True
        rationale: str = Field(default="", max_length=2000)

        @property
        def total_chars(self) -> int:
            return len(self.prompt) + len(self.answer_text) + len(self.thinking)

    return ChoiceRecord


def _ensure_langgraph():
    from langgraph.graph import END, StateGraph

    return StateGraph, END


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_code_fence(text: str) -> str:
    """Extract the largest ```json (or ```) fenced block; else return raw."""
    import re

    blocks = re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


def _safe_json_loads(text: str) -> Any:
    """Tolerant JSON load: try direct, then grab first {...} or [...] span."""
    try:
        return json.loads(text)
    except Exception:
        pass
    import re

    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None


def _seed_topic_for(category: str) -> str:
    return {
        "reasoning": "logical deduction puzzle in everyday life",
        "knowledge": "world history or science fact retrieval",
        "code": "short Python comprehension question about a tiny snippet",
        "math": "secondary-school arithmetic word problem",
    }.get(category, "general multiple-choice question")


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def _plan_questions(state: HybridState) -> dict:
    """Ask the model to produce a JSON list of question specs across all
    categories. Each spec = {category, topic, difficulty}."""
    model = _get_chat_model(state["model"])
    categories = state["categories"]
    n = state["questions_per_category"]

    prompt = (
        "You design multiple-choice dataset questions for training.\n"
        "Return a JSON array of objects, one per question. No commentary, "
        "no markdown.\n\n"
        f"Categories allowed: {categories}\n"
        f"Total questions needed: {n * len(categories)} "
        f"(exactly {n} per category).\n\n"
        "Each object schema:\n"
        '  {"category": <one of the allowed categories>,\n'
        '   "topic":    <short concrete sub-topic>,\n'
        '   "difficulty": "easy" | "medium" | "hard"}\n\n'
        "Distribute difficulties roughly evenly. Diversify topics so two "
        "questions never share a sub-topic."
    )
    resp = model.invoke(prompt)
    parsed = _safe_json_loads(_strip_code_fence(resp.content if hasattr(resp, "content") else str(resp)))
    if not isinstance(parsed, list):
        return {"plan": []}
    plan = [
        {
            "category": str(s.get("category", "knowledge")),
            "topic": str(s.get("topic", "general")).strip(),
            "difficulty": str(s.get("difficulty", "medium")).strip(),
        }
        for s in parsed
        if isinstance(s, dict) and s.get("category") in categories
    ]
    # backstop: if the model produced nothing usable, fall back to deterministic seeds
    if not plan:
        plan = [
            {"category": c, "topic": _seed_topic_for(c), "difficulty": "medium"}
            for c in categories
            for _ in range(n)
        ]
    # hard-truncate to the requested total so we don't blow the budget
    plan = plan[: n * len(categories)]
    return {"plan": plan, "attempt_counts": {}}


def _synthesize_one(model, spec: dict, sequence_length: int, reasoning_required: bool) -> dict:
    """Call the LLM once to produce a single ChoiceRecord-compatible dict."""
    sys_prompt = (
        "You write high-quality multiple-choice training data. "
        "Respond ONLY with a single JSON object -- no prose, no markdown."
    )
    user_prompt = (
        f"Category: {spec['category']}\n"
        f"Topic:    {spec['topic']}\n"
        f"Difficulty: {spec['difficulty']}\n\n"
        "Produce a JSON object with this exact schema:\n"
        '  "prompt":       a clear question string (no leading numbering).\n'
        '  "choices":      list of 3 to 5 distinct, plausible answer strings.\n'
        '  "answer_index": integer 0..len(choices)-1 pointing to the correct one.\n'
        '  "answer_text":  the exact string at choices[answer_index].\n'
        '  "rationale":    one short sentence explaining why it is correct.\n'
        + ('  "thinking":     a concise chain-of-thought derivation (3-6 sentences).\n'
           if reasoning_required else '  "thinking":     an empty string.\n')
        + "\nHard constraint: prompt + thinking + answer_text combined MUST be "
        f"<= {sequence_length} characters. Keep it tight."
    )
    resp = model.invoke(sys_prompt + "\n\n" + user_prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)
    parsed = _safe_json_loads(_strip_code_fence(text))
    if not isinstance(parsed, dict):
        return {"error": "model returned non-JSON", "raw": text[:500]}
    parsed.setdefault("category", spec["category"])
    parsed.setdefault("source", "ollama:synthetic")
    parsed.setdefault("thinking", "" if not reasoning_required else "")
    parsed.setdefault("rationale", "")
    return parsed


def _synthesize_node(state: HybridState) -> dict:
    """Materialise one record per planned spec (or repair a failed one)."""
    model = _get_chat_model(state["model"])
    ChoiceRecord = _ensure_pydantic()
    sequence_length = state["sequence_length"]
    hard_cap = state.get("hard_cap") or _HARD_SEQUENCE_CAP
    reasoning_required = state.get("reasoning_required", True)

    plan = state.get("plan") or []
    records = list(state.get("records") or [])
    failures = state.get("failures", 0)
    attempt_counts = dict(state.get("attempt_counts") or {})

    next_records: list = []
    for idx, spec in enumerate(plan):
        # skip specs that were already accepted during a prior pass
        if any(r.get("__spec_idx") == idx for r in records):
            continue
        attempts = attempt_counts.get(idx, 0) + 1
        record = _synthesize_one(model, spec, min(sequence_length, hard_cap), reasoning_required)
        try:
            cr = ChoiceRecord(**record)
        except Exception as e:
            # mark failure, will be retried in a later pass
            attempt_counts[idx] = attempts
            failures += 1
            continue
        # enforce hard sequence-length cap (chars, not tokens -- user requested cap)
        if cr.total_chars > hard_cap:
            attempt_counts[idx] = attempts
            failures += 1
            continue
        record["__spec_idx"] = idx
        next_records.append(record)

    merged = records + next_records
    return {"records": merged, "failures": failures, "attempt_counts": attempt_counts}


def _validate_node(state: HybridState) -> dict:
    """Deduplicate by prompt text; loop back to synth if anything was dropped."""
    ChoiceRecord = _ensure_pydantic()
    hard_cap = state.get("hard_cap") or _HARD_SEQUENCE_CAP
    seen_prompts: set = set()
    kept: list = []
    dropped = 0
    for r in state.get("records") or []:
        try:
            cr = ChoiceRecord(**{k: v for k, v in r.items() if not k.startswith("__")})
        except Exception:
            dropped += 1
            continue
        key = cr.prompt.strip().lower()
        if not key or key in seen_prompts:
            dropped += 1
            continue
        if cr.total_chars > hard_cap:
            dropped += 1
            continue
        seen_prompts.add(key)
        kept.append({k: v for k, v in r.items() if not k.startswith("__")})
    return {"records": kept, "failures": state.get("failures", 0) + dropped}


def _persist_node(state: HybridState) -> dict:
    """Flush accepted records to per-category jsonl shards via ShardWriter."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from quality import ShardWriter  # type: ignore

    out_dir = state["out_dir"]  # type: ignore[index]
    writers: dict = {}
    for r in state.get("records") or []:
        cat = str(r.get("category", "knowledge"))
        w = writers.get(cat)
        if w is None:
            w = ShardWriter(out_dir, cat)
            writers[cat] = w
        clean = {k: v for k, v in r.items() if not k.startswith("__")}
        w.write(clean)
    for w in writers.values():
        w.close()
    return {}


def _build_graph():
    """Compose the StateGraph; returned compiled app."""
    StateGraph, END = _ensure_langgraph()

    g = StateGraph(HybridState)
    g.add_node("plan", _plan_questions)
    g.add_node("synthesize", _synthesize_node)
    g.add_node("validate", _validate_node)
    g.add_node("persist", _persist_node)

    g.set_entry_point("plan")
    g.add_edge("plan", "synthesize")
    g.add_edge("synthesize", "validate")

    # if validate dropped anything AND synthesize has remaining capacity, loop once
    def _needs_retry(state: HybridState) -> str:
        return "synthesize" if state.get("failures", 0) > 0 else "persist"

    g.add_conditional_edges("validate", _needs_retry, {"synthesize": "synthesize", "persist": "persist"})
    g.add_edge("persist", END)
    return g.compile()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="llama3.1:8b", help="Ollama model name")
    p.add_argument(
        "--categories",
        default=",".join(DEFAULT_CATEGORIES),
        help="Comma-separated category list",
    )
    p.add_argument("--questions-per-category", type=int, default=DEFAULT_QUESTIONS_PER_CATEGORY)
    p.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
        help=(
            "Soft cap (chars) for prompt+thinking+answer_text per record. "
            f"Hard-capped internally at {_HARD_SEQUENCE_CAP}; values above are clamped down."
        ),
    )
    p.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Skip the 'thinking' chain-of-thought field (non-reasoning mode).",
    )
    p.add_argument("--out-dir", required=True, help="Output dir; categories become subdirs of jsonl shards")
    p.add_argument("--ollama-url", default=OLLAMA_URL, help="Ollama base URL")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["OLLAMA_URL"] = args.ollama_url

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    if not categories:
        print("error: --categories must be non-empty", file=sys.stderr)
        return 2

    # enforce the hard cap upstream
    soft = max(64, int(args.sequence_length))
    hard_cap = min(_HARD_SEQUENCE_CAP, soft)

    state: HybridState = {
        "model": args.model,
        "categories": categories,
        "questions_per_category": max(1, int(args.questions_per_category)),
        "sequence_length": soft,
        "hard_cap": hard_cap,
        "reasoning_required": not args.no_reasoning,
        "plan": None,
        "records": [],
        "failures": 0,
        "attempt_counts": {},
    }
    # out_dir is read inside persist_node; stash on state via a side-channel kwarg
    state["out_dir"] = args.out_dir  # type: ignore[typeddict-unknown-key]

    app = _build_graph()
    final = app.invoke(state)
    records = final.get("records") or []
    failures = final.get("failures", 0)
    print(f"[hybrid_choice_curator] wrote {len(records)} records, "
          f"{failures} validate-fail / re-synthesize loops", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
