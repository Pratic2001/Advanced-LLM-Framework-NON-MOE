#!/usr/bin/env python3
"""
hf_to_packed.py

Give it a HuggingFace dataset ID + mode, and it does the same thing as
codegen_pipeline.py does for public datasets — but for a single, explicit
dataset rather than sweeping a category:

    1. Sample the dataset (first N rows, discover columns)
    2. Ask Ollama to write a standalone extraction script (same prompt
       as codegen_pipeline.py's build_hf_codegen_prompt)
    3. Compile-validate the generated script, repair up to 2 times
    4. Run the script to produce filtered JSONL shards
    5. If the script crashes or produces zero data, capture the error
       log and feed it back to Ollama to fix the script (auto-fix loop,
       up to 5 total attempts). Once data is produced, continue.
    6. Call the mode-appropriate packer (data/pack_pretrain.py,
       data/pack_sft.py, data/pack_grpo.py) to produce the final memmap .bin files.

No hardcoded column mappings — the LLM figures out which columns map to
which target fields from the actual dataset schema, just like
codegen_pipeline.py. If the generated script fails at runtime, the error
log is captured and Ollama rewrites the script automatically.

Usage:

    # Pretrain: text dataset → packed .bin
    python hf_to_packed.py --dataset c4 --config en --mode pretrain \\
        --tokenizer ./tokenizer --seq-length 1024 --out-dir ./packed_c4

    # SFT: instruction dataset → packed memmap + mask
    python hf_to_packed.py --dataset databricks/databricks-dolly-15k \\
        --mode sft --tokenizer ./tokenizer --seq-length 2048 \\
        --out-dir ./packed_dolly

    # GRPO: math dataset → packed prompt memmap (for train_grpo.py)
    python hf_to_packed.py --dataset gsm8k --config main --mode grpo \\
        --tokenizer ./tokenizer --seq-length 2048 --out-dir ./packed_gsm8k

    # With a byte budget (stop after ~500 MB of raw JSONL)
    python hf_to_packed.py --dataset c4 --config en --mode pretrain \\
        --tokenizer ./tokenizer --seq-length 1024 --target-size 500MB \\
        --out-dir ./packed_c4

    # Custom category label
    python hf_to_packed.py --dataset gsm8k --config main --mode grpo \\
        --tokenizer ./tokenizer --out-dir ./packed_gsm8k \\
        --category math
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Optional

# Add the agent directory to sys.path so downstream imports (quality.py
# etc.) resolve the same way they do for codegen_pipeline.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codegen_pipeline import (
    build_hf_codegen_prompt,
    call_ollama,
    discover_hf_configs,
    generate_and_validate_script,
    parse_size,
    post_filter_shards,
    run_generated_script,
    sample_hf_dataset,
)

QUALITY_PY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality.py")

# Optional LangGraph reasoning codegen
try:
    from codegen_graph import generate_script_with_reasoning as _reasoning_codegen
    _HAS_REASONING = True
except ImportError:
    _HAS_REASONING = False


# ---------------------------------------------------------------------------
# Stage quality.py next to generated scripts so imports work
# ---------------------------------------------------------------------------

def _stage_quality_module(scripts_dir: str) -> None:
    """Copy quality.py into scripts_dir so generated scripts can import it
    via their own ``sys.path.insert(0, ...)`` (which points at their own
    directory)."""
    os.makedirs(scripts_dir, exist_ok=True)
    dest = os.path.join(scripts_dir, "quality.py")
    shutil.copy2(QUALITY_PY_PATH, dest)
    print(f"  staged quality.py -> {dest}")


# ---------------------------------------------------------------------------
# Packer invocation
# ---------------------------------------------------------------------------

def run_packer(
    mode: str,
    jsonl_dir: str,
    tokenizer_dir: str,
    out_dir: str,
    seq_length: Optional[int],
    val_fraction: float,
) -> bool:
    """Subprocess-call the appropriate packer for *mode*."""
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    packer_map = {
        "pretrain": os.path.join("data", "pack_pretrain.py"),
        "sft": os.path.join("data", "pack_sft.py"),
        "grpo": os.path.join("data", "pack_grpo.py"),
    }
    packer_rel = packer_map.get(mode)
    if not packer_rel:
        print(f"Error: unknown mode {mode!r}")
        return False

    packer_path = os.path.join(repo_root, packer_rel)
    if not os.path.exists(packer_path):
        packer_path = packer_rel  # fallback to PATH

    cmd = [
        sys.executable, packer_path,
        "--tokenizer", tokenizer_dir,
        "--val-fraction", str(val_fraction),
    ]

    # All current packers use --cache-dir as the output directory
    cmd += ["--data-dir", jsonl_dir, "--cache-dir", out_dir]

    if seq_length is not None:
        cmd += ["--seq-length", str(seq_length)]

    print(f"  running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  packer failed (exit {result.returncode})")
        return False

    print(f"  packer finished in {elapsed:.0f}s")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser(
        description="Download a HuggingFace dataset and pack it into the "
                    "memmap format for training. Uses Ollama to write an "
                    "extraction script (same approach as codegen_pipeline.py).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("--dataset", required=True,
                   help="HuggingFace dataset ID, e.g. 'c4', "
                        "'databricks/databricks-dolly-15k', 'gsm8k'.")
    p.add_argument("--config", default=None,
                   help="Dataset config/subset name, e.g. 'en' for c4, "
                        "'main' for gsm8k. Omit to auto-discover.")
    p.add_argument("--mode", required=True, choices=["pretrain", "sft", "grpo"],
                   help="Target training mode. Controls the target record "
                        "shape the LLM is asked to produce, which quality "
                        "filter to apply, and which packer to call.")
    p.add_argument("--tokenizer", default="./tokenizer",
                   help="Directory containing tokenizer.json "
                        "(from train_tokenizer.py). Default: ./tokenizer")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for the packed memmap files.")
    p.add_argument("--seq-length", type=int, default=None,
                   help="Truncate documents/examples to this many tokens "
                        "(passed through to the packer as --seq-length). "
                        "Default: no truncation (packer's own default).")
    p.add_argument("--target-size", default=None,
                   help="Stop after this much raw JSONL data, e.g. "
                        "'500MB', '1GB'. Default: stream the entire dataset "
                        "(uses 1TB as the internal upper bound).")
    p.add_argument("--language", default="en",
                   help="Language code (e.g. en, zh, de) for metadata. "
                        "Not used for content filtering. Default: en.")
    p.add_argument("--category", default=None,
                   help="Category label for the output records. Defaults "
                        "to the --mode value (e.g. 'pretrain', 'sft').")
    p.add_argument("--val-fraction", type=float, default=0.005,
                   help="Fraction of tokens for validation (default 0.005 = 0.5%%).")
    p.add_argument("--min-doc-chars", type=int, default=500,
                   help="Minimum character count for a record (passed through "
                        "to quality.py filters and the generated script).")
    p.add_argument("--reasoning-model", default=None,
                   help="Reasoning LLM model (e.g. deepseek-r1:7b, qwq:latest). "
                        "When set, uses LangGraph with step-by-step schema "
                        "analysis instead of the direct Ollama codegen loop.")
    p.add_argument("--keep-scripts", action="store_true",
                   help="Don't delete the generated scripts directory after "
                        "packing (useful for debugging). Default: delete.")
    # Extended quality filter CLI args
    p.add_argument("--no-extended-quality", action="store_true",
                   help="Disable programmatic extended-quality post-filter pass.")
    p.add_argument("--max-compression-ratio", type=float, default=0.35,
                   help="Max zlib compression ratio for post-filter (default 0.35)")
    p.add_argument("--max-line-repetition", type=float, default=0.15,
                   help="Max fraction of duplicate lines for post-filter (default 0.15)")
    p.add_argument("--max-adjacent-repetition", type=float, default=0.15,
                   help="Max fraction of adjacent near-identical lines for post-filter (default 0.15)")
    p.add_argument("--min-vocab-diversity", type=float, default=0.15,
                   help="Min unique/total word ratio for post-filter (default 0.15)")
    p.add_argument("--max-short-line-ratio", type=float, default=0.50,
                   help="Max fraction of short/navigation lines for post-filter (default 0.50)")
    p.add_argument("--max-flagged-ngram-ratio", type=float, default=0.10,
                   help="Max fraction of lines with flagged patterns for post-filter (default 0.10)")
    p.add_argument("--target-langs", default=None,
                   help="Comma-separated target languages for post-filter, e.g. en,de (requires fasttext)")

    return p.parse_args(argv)


def main() -> None:
    args = parse_args()

    # Resolve target bytes
    target_bytes_str = args.target_size or "1TB"  # 1TB = stream everything

    print(f"Dataset : {args.dataset}")
    print(f"Config  : {args.config or '(auto-discover)'}")
    print(f"Mode    : {args.mode}")
    print(f"Out dir : {args.out_dir}")
    print(f"Language: {args.language}")

    # Resolve config: if none given, try to discover
    config = args.config
    if config is None:
        configs = discover_hf_configs(args.dataset) or [None]
        if len(configs) > 1:
            print(f"  discovered configs: {configs}")
        config = configs[0]
        print(f"  using config: {config!r}")

    # Resolve category label
    category = args.category or args.mode

    # Extended quality filter config (programmatic post-filter safety net)
    use_ext_quality = not args.no_extended_quality
    quality_thresholds = {
        "max_compression_ratio": args.max_compression_ratio,
        "max_line_repetition": args.max_line_repetition,
        "max_adjacent_repetition": args.max_adjacent_repetition,
        "min_vocab_diversity": args.min_vocab_diversity,
        "max_short_line_ratio": args.max_short_line_ratio,
        "max_flagged_ngram_ratio": args.max_flagged_ngram_ratio,
    } if use_ext_quality else None
    target_langs = set(l.strip() for l in args.target_langs.split(",")
                       if l.strip()) if args.target_langs else None

    # ------------------------------------------------------------------
    # Step 1: Sample the dataset
    # ------------------------------------------------------------------
    print(f"\n--- Step 1: Sampling {args.dataset} (config={config}) ---")
    sample = sample_hf_dataset(args.dataset, config)
    if sample is None:
        print("Dataset unsuitable or not found — exiting")
        sys.exit(1)
    print(f"  split={sample['split']}, columns={sample['columns']}")

    # ------------------------------------------------------------------
    # Steps 2-4: Codegen → validate → run, with auto-fix retry loop.
    # If the script compiles but produces zero data or crashes at
    # runtime, we capture the error log and feed it back to Ollama to
    # fix the script and try again, up to max_attempts total rounds.
    # ------------------------------------------------------------------
    scripts_dir = os.path.join(args.out_dir, "_generated_scripts")
    logs_dir = os.path.join(args.out_dir, "_logs")
    jsonl_dir = os.path.join(args.out_dir, "_jsonl_cache")
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(jsonl_dir, exist_ok=True)
    _stage_quality_module(scripts_dir)

    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", args.dataset.replace("/", "_"))
    script_path = os.path.join(scripts_dir, f"{safe_name}.py")
    log_path = os.path.join(logs_dir, f"{safe_name}.log")

    max_attempts = 5
    prior_runtime_error: Optional[str] = None
    result: dict = {"actual_bytes": 0, "docs": 0}

    for attempt in range(1, max_attempts + 1):
        print(f"\n--- Attempt {attempt}/{max_attempts}: codegen ({script_path}) ---")

        if args.reasoning_model and _HAS_REASONING:
            ok = _reasoning_codegen(
                category=category, mode=args.mode,
                dataset_id=args.dataset, config=config,
                split=sample["split"],
                columns=sample["columns"], rows=sample["rows"],
                script_path=script_path,
                model=args.reasoning_model,
                prior_error=prior_runtime_error,
            )
        else:
            ok = generate_and_validate_script(
                lambda err: build_hf_codegen_prompt(
                    category, args.mode, args.dataset, config,
                    sample["split"], sample["columns"], sample["rows"],
                    prior_error=err,
                ),
                script_path,
                prior_error=prior_runtime_error,
            )
        if not ok:
            print(f"  Codegen failed after compile retries on attempt {attempt}")
            prior_runtime_error = prior_runtime_error or (
                "Codegen could not produce a compilable script "
                "(check Ollama / model availability)"
            )
            if attempt == max_attempts:
                print("  All attempts exhausted — exiting")
                if not args.keep_scripts:
                    shutil.rmtree(scripts_dir, ignore_errors=True)
                shutil.rmtree(jsonl_dir, ignore_errors=True)
                sys.exit(1)
            continue

        # --- Run the generated script ---
        print(f"--- Attempt {attempt}/{max_attempts}: running script ---")
        result = run_generated_script(
            script_path,
            target_bytes_str,
            jsonl_dir,
            category,
            args.min_doc_chars,
            log_path=log_path,
        )

        actual_bytes = result.get("actual_bytes", 0)
        docs = result.get("docs", 0)

        if actual_bytes > 0:
            print(f"  produced: {docs} docs, {actual_bytes:,} bytes of JSONL")
            # Programmatic extended-quality post-filter (safety net)
            if use_ext_quality and docs > 0:
                pf_result = post_filter_shards(
                    jsonl_dir, category, args.mode,
                    min_doc_chars=args.min_doc_chars,
                    quality_thresholds=quality_thresholds,
                    target_langs=target_langs,
                )
                docs = pf_result.get("kept_docs", docs)
                print(f"  after post-filter: {docs} docs kept")
            break  # success — exit the retry loop

        # --- Capture runtime error context for the next attempt ---
        log_tail = ""
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as f:
                log_tail = f.read()[-4000:]
        prior_runtime_error = (
            f"Script produced zero data (attempt {attempt}). "
            f"Exit code: see log. Last output:\n{log_tail[-2500:]}"
        )
        print(f"  No data produced on attempt {attempt}, retrying...")
        print(f"  Error context ({len(log_tail)} chars of log captured)")

    else:
        # Loop exhausted without break — all attempts failed
        print("\n  Failed to extract data after all attempts")
        if not args.keep_scripts:
            shutil.rmtree(scripts_dir, ignore_errors=True)
        shutil.rmtree(jsonl_dir, ignore_errors=True)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 5: Pack
    # ------------------------------------------------------------------
    print(f"\n--- Step 5: Packing → {args.out_dir} ---")
    pack_ok = run_packer(
        mode=args.mode,
        jsonl_dir=jsonl_dir,
        tokenizer_dir=args.tokenizer,
        out_dir=args.out_dir,
        seq_length=args.seq_length,
        val_fraction=args.val_fraction,
    )

    # Cleanup intermediate JSONL and scripts
    shutil.rmtree(jsonl_dir, ignore_errors=True)
    if not args.keep_scripts:
        shutil.rmtree(scripts_dir, ignore_errors=True)

    print(f"\n{'Done' if pack_ok else 'Packing failed'} — "
          f"{result.get('docs', 0)} docs, "
          f"{result.get('actual_bytes', 0):,} raw bytes → {args.out_dir}")

    if pack_ok:
        print(f"\nNext step:")
        if args.mode == "pretrain":
            print(f"  python train_pretrain.py --model-size 0.6B "
                  f"--data-dir {args.out_dir}")
        elif args.mode == "sft":
            print(f"  python train_sft.py --checkpoint ./checkpoints/latest.pt "
                  f"--tokenizer {args.tokenizer} "
                  f"--cache-dir {args.out_dir}")
        else:
            print(f"  python train_grpo.py --checkpoint ./checkpoints/latest.pt "
                  f"--tokenizer {args.tokenizer} "
                  f"--cache-dir {args.out_dir}")


if __name__ == "__main__":
    main()
