#!/usr/bin/env python3
"""
codegen_graph.py

LangGraph workflow for reasoning-enhanced codegen. Used by codegen_pipeline.py
and hf_to_packed.py when --reasoning-model is set.

The graph replaces the simple retry loop with a structured multi-step process:

    1. analyze_schema   — reasoning model analyses dataset columns / sample
                          rows and produces a structured column→field mapping
    2. generate_script  — writes the extraction script based on the analysis
    3. validate_compile — writes script to disk and py_compile's it
    4. repair_script    — on compile failure, reasoning model fixes the error
                          (loops back to validate_compile)

All LangChain/LangGraph imports are lazy (inside the functions that need
them) so the module is importable without those packages installed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from functools import lru_cache
from typing import Any, Optional, TypedDict

# ---------------------------------------------------------------------------
# Constants (duplicated from codegen_pipeline.py to keep this module
# self-contained with no circular imports).
# ---------------------------------------------------------------------------

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

MODE_SCHEMAS = {
    "pretrain": {
        "record_shape": '{"text": str, "source": str, "category": str}',
        "explanation": (
            "text is the full document body (article text, a Q&A pair joined "
            "into one block, whatever prose the dataset provides). No prompt/"
            "answer split needed."
        ),
    },
    "sft": {
        "record_shape": '{"prompt": str, "thinking": str, "answer": str, '
                        '"source": str, "category": str}',
        "explanation": (
            "prompt is the question/instruction, answer is the final answer/"
            "solution, thinking is the chain-of-thought/derivation if the "
            "dataset has one (else an empty string \"\" -- never fabricate "
            "one). If the dataset has a single combined solution field with "
            "both reasoning and a final answer, put the reasoning portion in "
            "thinking and just the final answer/result in answer."
        ),
    },
    "grpo": {
        "record_shape": '{"prompt": str, "answer": str, "source": str, '
                        '"category": str}',
        "explanation": (
            "prompt is the question, answer is ONLY the canonical ground-truth "
            "answer (numeric, boxed, or a short comparable string) -- no "
            "reasoning trace at all, since GRPO generates its own rollout."
        ),
    },
}

_QUALITY_API_SUMMARY = """\
A module `quality.py` is importable from the same directory (add
`sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))` before
importing it) with these ALREADY-IMPLEMENTED functions/classes -- use them,
do not reimplement filtering or dedup logic yourself:

    passes_prose_quality_filter(text: str, min_doc_chars: int = 500) -> (bool, reason)
        Use for pretrain-mode "text" fields (or any free-form prose).

    passes_sft_pair_quality_filter(prompt: str, answer: str, min_chars: int = 20) -> (bool, reason)
        Use for sft/grpo-mode (prompt, answer) pairs.

    passes_code_quality_filter(text: str, path: str, min_doc_chars: int = 500) -> (bool, reason)
        Use only if the category is "code".

    ExactDedup(persist_path: Optional[str] = None)
        .is_duplicate(text: str) -> bool

    ShardWriter(out_dir: str, category: str)
        .write(record: dict)
        .close()
        .total_bytes / .total_docs
"""

# ---------------------------------------------------------------------------
# State TypedDict
# ---------------------------------------------------------------------------


class CodegenState(TypedDict):
    """State flowing through the LangGraph workflow."""

    # --- Immutable inputs ---
    category: str
    mode: str
    script_path: str
    max_repair: int
    prior_error: Optional[str]
    model: str  # the reasoning model name (e.g. "deepseek-r1:7b")

    # --- HF codegen inputs ---
    dataset_id: Optional[str]
    config: Optional[str]
    split: Optional[str]
    columns: Optional[list]
    rows: Optional[list]

    # --- Web codegen inputs ---
    raw_path: Optional[str]
    sample_rows: Optional[list]

    # --- Mutable state (graph nodes produce these) ---
    schema_analysis: Optional[str]
    generated_code: Optional[str]
    compile_error: Optional[str]
    attempt_count: int
    success: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_code_block(text: str) -> str:
    """Pull the largest ```python fenced block, or assume the whole response
    is code."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return text.strip() + "\n"


def _py_compile_check(script_path: str) -> Optional[str]:
    """Returns None if the script compiles cleanly, else the error text."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", script_path],
        capture_output=True,
        text=True,
    )
    return None if result.returncode == 0 else (result.stderr or result.stdout)


# ---------------------------------------------------------------------------
# LangChain model helper (lazy import, LRU-cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_chat_model(model: str):
    """Create a ChatOllama instance for the given reasoning model."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model,
        base_url=OLLAMA_URL,
        temperature=0.2,
        num_predict=8192,
    )


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_analysis_prompt(state: CodegenState) -> str:
    """Prompt for the analyze_schema node — reasoning model plans column
    mappings without writing any code."""
    schema = MODE_SCHEMAS[state["mode"]]
    is_web = bool(state.get("raw_path"))

    if is_web:
        return _build_web_analysis_prompt(state, schema)
    else:
        return _build_hf_analysis_prompt(state, schema)


def _build_hf_analysis_prompt(state: CodegenState, schema: dict) -> str:
    sample_json = json.dumps(
        (state.get("rows") or [])[:5], ensure_ascii=False, indent=2
    )[:4000]
    columns = state.get("columns") or []
    dataset_id = state.get("dataset_id", "")
    config = state.get("config")
    split = state.get("split", "train")
    category = state["category"]
    mode = state["mode"]

    return f"""You are a data engineer analyzing a Hugging Face dataset.

Dataset: {dataset_id}
Config: {config!r}
Split: {split}
Category: {category}
Mode: {mode}

Target record shape:
{schema['record_shape']}
{schema['explanation']}
Always set "source" to "{dataset_id}", "category" to the --category CLI arg (default "{category}").

Available columns: {columns}

Sample rows (first {min(len(state.get('rows') or []), 5)}):
{sample_json}

Think step by step about how to map the available columns to the target
fields. For each target field, specify:
1. The exact source column name
2. Any transformation needed (strip whitespace, join with newline, etc.)
3. If no direct mapping exists, state what field to use instead

Also note any columns that appear irrelevant and why.

Do NOT write code. Only analyze.

Output your analysis as:

ANALYSIS:
- <target_field> -> <source_column> [transformation notes]
- <target_field> -> <source_column> [transformation notes]
[...]
- Unused columns: <col> (<reason>), <col> (<reason>)
"""


def _build_web_analysis_prompt(state: CodegenState, schema: dict) -> str:
    sample_rows = state.get("sample_rows") or []
    sample_preview = json.dumps(
        [
            {"url": r.get("url"), "text": (r.get("text") or "")[:600]}
            for r in sample_rows[:4]
        ],
        ensure_ascii=False,
        indent=2,
    )[:3000]
    category = state["category"]
    mode = state["mode"]
    raw_path = state.get("raw_path", "")

    return f"""You are a data engineer analyzing raw scraped web pages.

Source file: {raw_path}
Category: {category}
Mode: {mode}

Target record shape:
{schema['record_shape']}
{schema['explanation']}
Set "source" to the page URL, "category" to the --category CLI arg.

Sample raw pages (first {min(len(sample_rows), 4)} rows):
{sample_preview}

Analyze the following aspects -- think step by step, then produce a
structured analysis:

1. MAIN CONTENT: For each target field, identify which part of the scraped
   page text should populate it. Is the text usable as-is, or does it need
   cleaning?
2. BOILERPLATE: What recurring patterns do you see across samples (nav menus,
   cookie notices, "subscribe" prompts, ads, footers)? List specific patterns
   to strip.
3. For non-pretrain modes: Is there a natural Q&A or prompt/answer split
   visible in the text, or should you skip pages that lack one?

Do NOT write code. Only analyze.

Output your analysis as:

ANALYSIS:
- text -> <what to extract and how to clean>
- source -> "url"
- category -> {category}
[additional target field mappings...]

Boilerplate to strip:
- <pattern 1>
- <pattern 2>
[...]
"""


def _build_generation_prompt(state: CodegenState) -> str:
    """Prompt for the generate_script node — includes schema analysis from
    the previous step."""
    schema = MODE_SCHEMAS[state["mode"]]
    is_web = bool(state.get("raw_path"))
    analysis = state.get("schema_analysis") or ""

    if is_web:
        return _build_web_generation_prompt(state, schema, analysis)
    else:
        return _build_hf_generation_prompt(state, schema, analysis)


def _build_hf_generation_prompt(state: CodegenState, schema: dict,
                                 analysis: str) -> str:
    dataset_id = state.get("dataset_id", "")
    config = state.get("config")
    split = state.get("split", "train")
    columns = state.get("columns") or []
    rows = state.get("rows") or []
    category = state["category"]
    mode = state["mode"]
    sample_json = json.dumps(rows[:5], ensure_ascii=False, indent=2)[:4000]

    return f"""You write ONE standalone Python script, nothing else.

SCHEMA ANALYSIS (from previous reasoning step -- follow this plan):
{analysis}

TASK: write a complete, runnable Python script that streams the Hugging Face
dataset "{dataset_id}" (config={config!r}, split="{split}") in FULL via
`datasets.load_dataset(..., streaming=True)`, maps each row to this target
shape for mode="{mode}":
{schema['record_shape']}
{schema['explanation']}
Always set "source" to "{dataset_id}", "category" to --category (default "{category}").

Real columns: {columns}
First few rows:
{sample_json}

{_QUALITY_API_SUMMARY}

Script requirements:
- argparse with: --target-size (parse GB/MB/KB/B suffixes, case-insensitive),
  --out-dir (default "./data"), --category (default "{category}"),
  --min-doc-chars (default 500).
- Loop over the stream, map columns using this dataset's REAL column names.
- Skip rows failing the appropriate quality.py filter or ExactDedup on main
  text/answer content.
- Write via ShardWriter(out_dir, category).write(record).
- Print progress every ~5s. Stop when ShardWriter.total_bytes >= target_bytes.
- At the very end print "RESULT_JSON:" followed by
  {{"actual_bytes": int, "docs": int}} -- must be the last line.
- Wrap in broad try/except that always closes ShardWriter and prints
  RESULT_JSON with whatever was produced so far.

Output ONLY the script in a single ```python fence.
"""


def _build_web_generation_prompt(state: CodegenState, schema: dict,
                                  analysis: str) -> str:
    raw_path = state.get("raw_path", "")
    sample_rows = state.get("sample_rows") or []
    category = state["category"]
    mode = state["mode"]
    sample_preview = json.dumps(
        [
            {"url": r.get("url"), "text": (r.get("text") or "")[:600]}
            for r in sample_rows[:3]
        ],
        ensure_ascii=False,
        indent=2,
    )[:3000]
    filter_instruction = (
        "Use passes_prose_quality_filter for pretrain mode (clean up "
        "boilerplate as identified in the analysis)."
        if mode == "pretrain"
        else "Only keep pages that look like coherent Q&A/tutorial/exercise "
             "-- split into prompt/answer where a clear structure exists, "
             "else skip the page."
    )

    return f"""You write ONE standalone Python script, nothing else.

SCHEMA ANALYSIS (from previous reasoning step -- follow this plan):
{analysis}

TASK: write a complete, runnable Python script that reads a raw JSONL file of
scraped web pages (one {{"url": str, "text": str}} object per line) and
produces cleaned, filtered, deduped JSONL shards in this target shape for
mode="{mode}":
{schema['record_shape']}
{schema['explanation']}
Set "source" to the row's "url" and "category" to "{category}".

A few real sample rows:
{sample_preview}

{_QUALITY_API_SUMMARY}

Script requirements:
- argparse with: --raw-path (default "{raw_path}"),
  --target-size (parse GB/MB/KB/B suffixes), --out-dir (default "./data"),
  --category (default "{category}"), --min-doc-chars (default 500).
- Read the raw JSONL line by line (do NOT load all into memory).
- {filter_instruction}
- Apply ExactDedup on main text before writing. Use ShardWriter to write.
- Print progress every ~5 seconds. Stop once total_bytes >= target_bytes.
- At the very end print "RESULT_JSON:" followed by
  {{"actual_bytes": int, "docs": int}} -- must be the last line.
- Wrap main loop in try/except that always closes ShardWriter and prints
  RESULT_JSON with whatever was produced so far.

Output ONLY the script in a single ```python fence.
"""


def _build_repair_prompt(state: CodegenState) -> str:
    """Prompt for the repair_script node — includes previous code, compile
    error, and schema analysis."""
    analysis = state.get("schema_analysis") or ""
    return f"""Fix the compile error in the script below. Output ONLY the fixed
script in a single ```python fence.

SCHEMA ANALYSIS (from earlier reasoning -- do not change the approach):
{analysis}

PREVIOUS CODE (which failed to compile):
{state.get('generated_code', '')}

COMPILE ERROR:
{state.get('compile_error', '')}

Fix the error while keeping the same column mapping approach. Output ONLY the
fixed script.
"""


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def analyze_schema(state: CodegenState) -> dict:
    """Node: reasoning model analyses columns/samples -> structured mapping."""
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = _build_analysis_prompt(state)
    messages = [
        SystemMessage(
            content="You are a data engineer analyzing dataset schemas. Think "
                    "step by step about column mappings. Do NOT write code."),
        HumanMessage(content=prompt),
    ]
    response = _get_chat_model(state["model"]).invoke(messages)
    return {"schema_analysis": response.content}


def generate_script(state: CodegenState) -> dict:
    """Node: reasoning model writes extraction script based on analysis."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser

    prompt = _build_generation_prompt(state)
    messages = [
        SystemMessage(
            content="You write standalone Python scripts for data extraction. "
                    "Output ONLY the script in a ```python fence."),
        HumanMessage(content=prompt),
    ]
    response = (
        _get_chat_model(state["model"]) | StrOutputParser()
    ).invoke(messages)
    code = _extract_code_block(response)
    return {"generated_code": code}


def validate_compile(state: CodegenState) -> dict:
    """Tool node: write code to disk and py_compile it."""
    code = state.get("generated_code", "")
    script_path = state["script_path"]
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)
    error = _py_compile_check(script_path)
    return {"compile_error": error}


def repair_script(state: CodegenState) -> dict:
    """Node: reasoning model fixes compile errors."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser

    prompt = _build_repair_prompt(state)
    messages = [
        SystemMessage(
            content="You fix Python compile errors. Output ONLY the fixed "
                    "script in a ```python fence, no explanations."),
        HumanMessage(content=prompt),
    ]
    response = (
        _get_chat_model(state["model"]) | StrOutputParser()
    ).invoke(messages)
    code = _extract_code_block(response)
    return {"generated_code": code, "attempt_count": state.get("attempt_count", 0) + 1}


def should_repair(state: CodegenState) -> str:
    """Conditional edge: decide what to do after validate_compile."""
    if state.get("compile_error") is None:
        return "run"  # Success -> END
    if state.get("attempt_count", 0) >= state.get("max_repair", 2):
        return "fail"  # Exhausted -> END
    return "repair"  # Try again -> repair_script


# ---------------------------------------------------------------------------
# Graph builder (module-level cache — compiled once, reused across calls)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _build_codegen_graph():
    """Build and compile the LangGraph StateGraph."""
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(CodegenState)

    workflow.add_node("analyze_schema", analyze_schema)
    workflow.add_node("generate_script", generate_script)
    workflow.add_node("validate_compile", validate_compile)
    workflow.add_node("repair_script", repair_script)

    workflow.set_entry_point("analyze_schema")
    workflow.add_edge("analyze_schema", "generate_script")
    workflow.add_edge("generate_script", "validate_compile")
    workflow.add_conditional_edges(
        "validate_compile",
        should_repair,
        {"repair": "repair_script", "run": "__end__", "fail": "__end__"},
    )
    workflow.add_edge("repair_script", "validate_compile")

    return workflow.compile()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_script_with_reasoning(
    category: str,
    mode: str,
    dataset_id: Optional[str] = None,
    config: Optional[str] = None,
    split: Optional[str] = None,
    columns: Optional[list] = None,
    rows: Optional[list] = None,
    script_path: str = "",
    model: str = "deepseek-r1:7b",
    max_repair: int = 2,
    prior_error: Optional[str] = None,
    **kwargs: Any,
) -> bool:
    """Generate an extraction script using a LangGraph workflow with a
    reasoning LLM.

    For HF dataset codegen: pass *dataset_id*, *config*, *split*,
    *columns*, *rows*.
    For web codegen: pass *raw_path* and *sample_rows* via *kwargs*.

    Returns True if a compilable script now exists at *script_path*.
    """
    # Lazy guard for the optional dependencies
    try:
        import langchain_core  # noqa: F401
        import langgraph  # noqa: F401
    except ImportError:
        print("[error] langchain and langgraph are required for "
              "--reasoning-model. Install with:\n"
              "  pip install langchain langchain-ollama langgraph")
        sys.exit(1)

    initial: CodegenState = {
        "category": category,
        "mode": mode,
        "model": model,
        "dataset_id": dataset_id,
        "config": config,
        "split": split,
        "columns": columns or [],
        "rows": rows or [],
        "raw_path": kwargs.get("raw_path"),
        "sample_rows": kwargs.get("sample_rows"),
        "script_path": script_path,
        "max_repair": max_repair,
        "prior_error": prior_error,
        "schema_analysis": None,
        "generated_code": None,
        "compile_error": None,
        "attempt_count": 0,
        "success": False,
    }

    graph = _build_codegen_graph()
    result = graph.invoke(initial)

    success = result.get("compile_error") is None
    if success:
        print(f"  [codegen] {os.path.basename(script_path)} compiles OK "
              f"({result.get('attempt_count', 0) + 1} attempt(s))")
    else:
        print(f"  [codegen] giving up on {os.path.basename(script_path)} after "
              f"{max_repair + 1} attempt(s) -- skipping this source")

    return success
