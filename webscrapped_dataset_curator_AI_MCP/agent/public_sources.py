"""
public_sources.py

Pulls rows from public dataset hubs (Hugging Face Hub, Kaggle) instead of
live web search + scrape. This is the "faster top-up" path: no robots.txt,
no rate limiting, no HTML noise -- just structured rows that already exist
as public datasets, normalized into the same shape dataset_agent.py
already knows how to quality-filter, LLM-judge, dedup, and shard-write.

Public contract (mirrors extract_content()'s shape):

    {
        "title": str | None,
        "text": str,
        "author": None,
        "date": None,
        "content_type": "dataset_row",
        "url": "hf://<dataset_id>#<row_idx>" | "kaggle://<ref>/<file>#<row_idx>",
        "error": str | None,
        "extra": {"source": "huggingface"|"kaggle", "dataset": str,
                   "columns": [...],
                   "prompt": str | None, "answer": str | None},
    }
"""

from __future__ import annotations

import glob
import logging
import os
import tempfile
from typing import Iterator, Optional

log = logging.getLogger("dataset_agent.public_sources")

# ---------------------------------------------------------------------------
# Column-name heuristics
# ---------------------------------------------------------------------------

_TEXT_COLUMNS = (
    "text", "content", "document", "article", "body", "passage",
    "abstract", "sentence", "description",
)
_PROMPT_COLUMNS = ("prompt", "question", "instruction", "input", "query", "problem", "task")
_ANSWER_COLUMNS = ("answer", "response", "output", "completion", "solution", "answers")
_CODE_COLUMNS = ("code", "solution_code", "func_code", "program")

_CONVERSATION_COLUMNS = ("conversations", "messages", "conversation")
_TURN_ROLE_KEYS = ("from", "role")
_TURN_VALUE_KEYS = ("value", "content", "text")
_HUMAN_ROLE_VALUES = {"human", "user", "prompter"}
_ASSISTANT_ROLE_VALUES = {"gpt", "assistant", "bot", "model"}

_ALL_KNOWN_COLUMNS = (frozenset(c.lower() for c in _TEXT_COLUMNS)
                      | frozenset(c.lower() for c in _PROMPT_COLUMNS)
                      | frozenset(c.lower() for c in _ANSWER_COLUMNS)
                      | frozenset(c.lower() for c in _CODE_COLUMNS)
                      | frozenset(c.lower() for c in _CONVERSATION_COLUMNS))


def schema_is_suitable(columns, column_hint: Optional[dict] = None) -> bool:
    """Gate applied ONCE per dataset/config (on the first row). True means
    row_to_record has at least one real column to draw from."""
    if column_hint and any(column_hint.get(k) for k in
                            ("prompt_col", "answer_col", "conversation_col", "text_col")):
        return True
    cols_lower = {c.lower() for c in columns}
    return bool(cols_lower & _ALL_KNOWN_COLUMNS)


def _first_str(row: dict, candidates) -> Optional[str]:
    for key in candidates:
        for col in row:
            if col.lower() == key and isinstance(row[col], str) and row[col].strip():
                return row[col].strip()
    return None


def _turn_role_and_value(turn: dict) -> tuple:
    role = None
    for k in _TURN_ROLE_KEYS:
        if isinstance(turn.get(k), str):
            role = turn[k].strip().lower()
            break
    value = None
    for k in _TURN_VALUE_KEYS:
        if isinstance(turn.get(k), str) and turn[k].strip():
            value = turn[k].strip()
            break
    return role, value


def _str_from_col(row: dict, col_name: Optional[str]) -> Optional[str]:
    """Case-insensitive lookup of one specific column name."""
    if not col_name:
        return None
    target = col_name.strip().lower()
    for col, val in row.items():
        if col.lower() == target and isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _prompt_answer_from_conversation(row: dict, column_names=_CONVERSATION_COLUMNS) -> tuple:
    """Pull a (prompt, answer) pair from the first human->assistant turn
    of a chat-format column."""
    names = {c.lower() for c in column_names} if column_names else set()
    for col in row:
        if col.lower() not in names:
            continue
        turns = row[col]
        if not isinstance(turns, list) or not turns:
            continue
        prompt, answer = None, None
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role, value = _turn_role_and_value(turn)
            if not value:
                continue
            if prompt is None and role in _HUMAN_ROLE_VALUES:
                prompt = value
            elif prompt is not None and answer is None and role in _ASSISTANT_ROLE_VALUES:
                answer = value
                break
        if prompt and answer:
            return prompt, answer
    return None, None


def row_to_record(row: dict, source_label: str, dataset_id: str, ref: str,
                   column_hint: Optional[dict] = None) -> dict:
    """Normalize one raw dict from a HF/Kaggle dataset into the shared
    extract_content-shaped record."""
    prompt = answer = text = None
    if column_hint:
        prompt = _str_from_col(row, column_hint.get("prompt_col"))
        answer = (_str_from_col(row, column_hint.get("answer_col"))
                  or _str_from_col(row, column_hint.get("code_col")))
        if not (prompt and answer) and column_hint.get("conversation_col"):
            conv_prompt, conv_answer = _prompt_answer_from_conversation(
                row, column_names=[column_hint["conversation_col"]])
            prompt = prompt or conv_prompt
            answer = answer or conv_answer
        text = _str_from_col(row, column_hint.get("text_col"))

    if not (prompt and answer):
        fb_prompt = _first_str(row, _PROMPT_COLUMNS)
        fb_answer = _first_str(row, _ANSWER_COLUMNS) or _first_str(row, _CODE_COLUMNS)
        prompt = prompt or fb_prompt
        answer = answer or fb_answer
    if not (prompt and answer):
        conv_prompt, conv_answer = _prompt_answer_from_conversation(row)
        prompt = prompt or conv_prompt
        answer = answer or conv_answer

    if not text:
        text = _first_str(row, _TEXT_COLUMNS)
    if not text:
        if prompt and answer:
            text = f"{prompt}\n\n{answer}"
        else:
            parts = [v.strip() for v in row.values() if isinstance(v, str) and v.strip()]
            text = "\n\n".join(parts)

    return {
        "title": None,
        "text": text or "",
        "author": None,
        "date": None,
        "content_type": "dataset_row",
        "url": ref,
        "error": None if text else "row had no extractable string content",
        "extra": {
            "source": source_label,
            "dataset": dataset_id,
            "columns": list(row.keys()),
            "prompt": prompt,
            "answer": answer,
        },
    }


# ---------------------------------------------------------------------------
# Hugging Face Hub
# ---------------------------------------------------------------------------

def discover_hf_datasets(query: str, limit: int = 5) -> list:
    """Search the Hugging Face Hub for dataset ids matching a query."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.warning("huggingface_hub not installed -- pip install huggingface_hub datasets")
        return []
    try:
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        results = api.list_datasets(search=query, limit=limit, sort="downloads", direction=-1)
        return [d.id for d in results]
    except Exception as e:
        log.warning(f"[hf] dataset search failed for {query!r}: {e}")
        return []


def discover_hf_configs(dataset_id: str, token: Optional[str] = None) -> list:
    """Returns every config/subset name a HF dataset exposes."""
    try:
        from datasets import get_dataset_config_names
    except ImportError:
        return [None]
    try:
        configs = get_dataset_config_names(dataset_id, token=token)
        return list(configs) if configs else [None]
    except Exception as e:
        log.warning(f"[hf] could not list configs for {dataset_id}: {e} -- falling back to default")
        return [None]


def stream_hf_dataset(dataset_id: str, max_rows: int = 200, split: Optional[str] = None,
                       config: Optional[str] = None, column_mapper=None) -> Iterator[dict]:
    """Yields normalized records from a Hugging Face dataset via streaming.
    Discovers all configs and fair-shares max_rows across them."""
    try:
        from datasets import load_dataset
    except ImportError:
        yield {"title": None, "text": "", "author": None, "date": None,
               "content_type": "dataset_row", "url": f"hf://{dataset_id}",
               "error": "`datasets` package not installed -- pip install datasets",
               "extra": {"source": "huggingface", "dataset": dataset_id}}
        return

    token = os.environ.get("HF_TOKEN")
    tried_splits = [split] if split else ["train", "test", "validation"]
    configs_to_try = [config] if config else discover_hf_configs(dataset_id, token)
    if config is None and configs_to_try != [None]:
        log.info(f"[hf] {dataset_id} has {len(configs_to_try)} config(s): {configs_to_try}")

    per_config_cap = max(1, max_rows // len(configs_to_try))

    count = 0
    any_success = False
    last_err = None
    schema_rejected_configs = []
    for cfg_attempt in configs_to_try:
        if count >= max_rows:
            break
        ds = None
        used_split = None
        for s in tried_splits:
            try:
                ds = load_dataset(dataset_id, cfg_attempt, split=s, streaming=True, token=token)
                used_split = s
                break
            except Exception as e:
                last_err = e
                continue
        if ds is None:
            log.warning(f"[hf] could not load config {cfg_attempt!r} of {dataset_id}: {last_err}")
            continue

        any_success = True
        cfg_count = 0
        cfg_label = f":{cfg_attempt}" if cfg_attempt else ""
        column_hint = None
        hint_resolved = False
        for row in ds:
            if count >= max_rows or cfg_count >= per_config_cap:
                break
            if not isinstance(row, dict):
                continue
            if not hint_resolved:
                if column_mapper is not None:
                    try:
                        column_hint = column_mapper(dataset_id, cfg_attempt, list(row.keys()), row)
                    except Exception as e:
                        log.warning(f"[hf] column_mapper failed for {dataset_id}{cfg_label}: {e}")
                        column_hint = None
                hint_resolved = True
                if not schema_is_suitable(list(row.keys()), column_hint):
                    schema_rejected_configs.append(cfg_attempt)
                    log.warning(f"[hf] REJECT {dataset_id}{cfg_label}: columns "
                                f"{list(row.keys())} don't match any known prompt/answer/"
                                f"text/conversation pattern -- skipping")
                    break
            ref = f"hf://{dataset_id}{cfg_label}#{used_split}:{cfg_count}"
            yield row_to_record(row, "huggingface", dataset_id, ref, column_hint=column_hint)
            count += 1
            cfg_count += 1

    if not any_success:
        yield {"title": None, "text": "", "author": None, "date": None,
               "content_type": "dataset_row", "url": f"hf://{dataset_id}",
               "error": f"could not load any split/config of {dataset_id}: {last_err}",
               "extra": {"source": "huggingface", "dataset": dataset_id}}
    elif count == 0 and schema_rejected_configs:
        yield {"title": None, "text": "", "author": None, "date": None,
               "content_type": "dataset_row", "url": f"hf://{dataset_id}",
               "error": f"rejected: no config of {dataset_id} has matching columns "
                        f"(configs checked: {schema_rejected_configs})",
               "extra": {"source": "huggingface", "dataset": dataset_id}}


# ---------------------------------------------------------------------------
# Kaggle
# ---------------------------------------------------------------------------

def discover_kaggle_datasets(query: str, limit: int = 5) -> list:
    """Search Kaggle for dataset refs matching a query."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        log.warning("kaggle package not installed -- pip install kaggle and set "
                     "KAGGLE_USERNAME/KAGGLE_KEY")
        return []
    try:
        api = KaggleApi()
        api.authenticate()
        results = api.dataset_list(search=query)
        return [d.ref for d in results[:limit]]
    except Exception as e:
        log.warning(f"[kaggle] dataset search failed for {query!r} (check credentials): {e}")
        return []


_TABULAR_EXTS = (".csv", ".tsv", ".json", ".jsonl")


def fetch_kaggle_dataset_rows(dataset_ref: str, max_rows: int = 200, column_mapper=None) -> Iterator[dict]:
    """Downloads a Kaggle dataset into a temp dir, then reads tabular/text
    files yielding normalized records row-by-row up to max_rows."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        yield {"title": None, "text": "", "author": None, "date": None,
               "content_type": "dataset_row", "url": f"kaggle://{dataset_ref}",
               "error": "kaggle package not installed -- pip install kaggle",
               "extra": {"source": "kaggle", "dataset": dataset_ref}}
        return

    try:
        import pandas as pd
    except ImportError:
        yield {"title": None, "text": "", "author": None, "date": None,
               "content_type": "dataset_row", "url": f"kaggle://{dataset_ref}",
               "error": "pandas not installed -- required to read Kaggle CSV/JSON files",
               "extra": {"source": "kaggle", "dataset": dataset_ref}}
        return

    tmp_dir = tempfile.mkdtemp(prefix="kaggle_ds_")
    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_ref, path=tmp_dir, unzip=True, quiet=True)
    except Exception as e:
        yield {"title": None, "text": "", "author": None, "date": None,
               "content_type": "dataset_row", "url": f"kaggle://{dataset_ref}",
               "error": f"download failed: {e}",
               "extra": {"source": "kaggle", "dataset": dataset_ref}}
        return

    files = sorted(glob.glob(os.path.join(tmp_dir, "**", "*"), recursive=True))
    count = 0

    for fpath in files:
        if count >= max_rows or not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fpath)[1].lower()
        rel = os.path.relpath(fpath, tmp_dir)
        try:
            if ext == ".csv":
                df_iter = pd.read_csv(fpath, chunksize=200, on_bad_lines="skip")
            elif ext == ".tsv":
                df_iter = pd.read_csv(fpath, sep="\t", chunksize=200, on_bad_lines="skip")
            elif ext == ".jsonl":
                df_iter = pd.read_json(fpath, lines=True, chunksize=200)
            elif ext == ".json":
                size_mb = os.path.getsize(fpath) / 1024**2
                max_json_mb = float(os.environ.get("KAGGLE_MAX_JSON_FILE_MB", "300"))
                if size_mb > max_json_mb:
                    log.warning(f"[kaggle] skipping {rel}: {size_mb:.0f}MB .json exceeds "
                                f"KAGGLE_MAX_JSON_FILE_MB={max_json_mb:.0f}")
                    continue
                df_iter = [pd.read_json(fpath)]
            elif ext == ".txt":
                with open(fpath, "r", errors="ignore") as f:
                    text = f.read()
                yield row_to_record({"text": text}, "kaggle", dataset_ref,
                                     f"kaggle://{dataset_ref}/{rel}")
                count += 1
                continue
            else:
                continue
        except Exception as e:
            log.warning(f"[kaggle] failed reading {rel} from {dataset_ref}: {e}")
            continue

        file_hint = None
        file_hint_resolved = False
        file_rejected = False
        for chunk in df_iter:
            for i, row in chunk.iterrows():
                if count >= max_rows:
                    break
                row_dict = row.dropna().to_dict()
                if not file_hint_resolved:
                    if column_mapper is not None:
                        try:
                            full_row = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
                            file_hint = column_mapper(dataset_ref, rel, list(chunk.columns), full_row)
                        except Exception as e:
                            log.warning(f"[kaggle] column_mapper failed for {dataset_ref}/{rel}: {e}")
                            file_hint = None
                    file_hint_resolved = True
                    if not schema_is_suitable(list(chunk.columns), file_hint):
                        log.warning(f"[kaggle] REJECT {dataset_ref}/{rel}: columns "
                                    f"{list(chunk.columns)} don't match any pattern")
                        file_rejected = True
                        break
                record = row_to_record(row_dict, "kaggle", dataset_ref,
                                        f"kaggle://{dataset_ref}/{rel}#{i}", column_hint=file_hint)
                yield record
                count += 1
            if count >= max_rows or file_rejected:
                break
