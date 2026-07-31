# End-to-End Pipeline Verification: Dataset Creators → Packers → Training Scripts

## Summary

✅ **All three dataset creators produce JSONL records that are fully compatible with the three packers and training scripts.**

---

## 1. Dataset Creators Output Schema

### A. `dataset_agent.py` (async per-row web scraping + public datasets)
**Output location:** `<out_dir>/<category>/<category>_XXXXX.jsonl` (256MB shards)

**Record fields written by `_record_accepted()` → `ShardWriter.write()`:**
| Field | Type | Description |
|-------|------|-------------|
| `text` | str | Full document body (always present) |
| `prompt` | str | Question/instruction (sft/grpo modes only, extracted via LLM) |
| `thinking` | str | Chain-of-thought reasoning (sft mode only, from LLM extraction) |
| `answer` | str | Final answer/solution (sft/grpo modes only, from LLM extraction) |
| `source` | str | URL or `hf://<dataset_id>#<row>` / `kaggle://...` |
| `category` | str | Category name (e.g., "web", "math", "code") |
| `extra` | dict | Metadata: `prompt`, `answer`, `dataset`, `columns`, etc. |

**Mode behavior:**
- `pretrain`: Uses `text` field only
- `sft`: Uses `prompt`, `thinking`, `answer` (LLM-extracted if missing)
- `grpo`: Uses `prompt`, `answer` (LLM-extracted if missing)

**Quality filters applied:** Extended quality (compression ratio, line repetition, vocab diversity, etc.) + ExactDedup + NearDedup + optional LLM judge

---

### B. `codegen_pipeline.py` (per-dataset LLM codegen for public HF datasets + web crawl)
**Output location:** `<out_dir>/<category>/<category>_XXXXX.jsonl` (256MB shards)

**LLM instructed to produce exact target schema per mode:**

| Mode | Target Schema (from `MODE_SCHEMAS`) |
|------|-------------------------------------|
| `pretrain` | `{"text": str, "source": str, "category": str}` |
| `sft` | `{"prompt": str, "thinking": str, "answer": str, "source": str, "category": str}` |
| `grpo` | `{"prompt": str, "answer": str, "source": str, "category": str}` |

**Quality filters used in generated scripts:** Same `quality.py` functions (ExactDedup, ShardWriter, `passes_prose_quality_filter`, `passes_sft_pair_quality_filter`, `passes_extended_text_quality`, `passes_extended_sft_quality`)

**Post-filter safety net:** `post_filter_shards()` re-applies extended quality filters after generation

---

### C. `hf_to_packed.py` (single HF dataset → packer)
**Output location:** Same category shards + direct packer call

**Uses identical codegen approach as `codegen_pipeline.py`** (reuses `build_hf_codegen_prompt`, `generate_and_validate_script`, `run_generated_script`, `post_filter_shards`)

**Then calls packers directly via subprocess:**
- `data/pack_pretrain.py` for `pretrain`
- `data/pack_sft.py` for `sft`
- `data/pack_grpo.py` for `grpo`

---

## 2. Packers Input Requirements

### `data/pack_pretrain.py`
```python
# Reads: "text" field from JSONL
text = rec.get("text", "")
ids = tokenizer.encode(text).ids + [eos_id]
```
**Requires:** `text` field (string)

---

### `data/pack_sft.py`
```python
# Reads: "prompt", "thinking", "answer" fields
prompt = rec.get("prompt", "").strip()
thinking = rec.get("thinking", "").strip()
answer = rec.get("answer", "").strip()
want_thinking = rec.get("want_thinking", None)

# Tokenizes via TrainingRecipe (ChatML template)
user_text = recipe.format_user_turn(prompt)
assistant_text = recipe.format_assistant_turn(thinking, answer, want_thinking)
```
**Requires:** `prompt` (non-empty), `answer` (non-empty), `thinking` (optional), `want_thinking` (optional)

---

### `data/pack_grpo.py`
```python
# Reads: "prompt", "answer" fields
prompt = rec.get("prompt", "").strip()
answer = rec.get("answer", "").strip()
want_thinking = rec.get("want_thinking", True)  # preserved for reward

# Tokenizes prompt + assistant_prefix via TrainingRecipe
user_text = recipe.format_user_turn(prompt)
assistant_prefix = recipe.turn_prefix_assistant  # "<s>assistant\n"
full_text = user_text + assistant_prefix
token_ids = tokenizer.encode(full_text).ids + [eos_id]

# Writes: length-prefixed uint32 tokens + JSON with {"answer": ..., "want_thinking": ...}
```
**Requires:** `prompt` (non-empty), `answer` (non-empty), `want_thinking` (optional, default True)

---

## 3. Training Scripts Input Requirements

### `train_pretrain.py` — `PackedDataLoader`
```python
# Reads: memmap .bin files (uint16/uint32) + meta.json
# Expects: tokens separated by EOS tokens at document boundaries
# Outputs: (x, y) where y = x shifted by 1 (next-token prediction)
```
**Input:** `pretrain_tokens.*.bin` + `meta.*.json` from `pack_pretrain.py`

---

### `train_sft.py` — `PackedSFTDataLoader`
```python
# Reads: sft_train_tokens.*.bin + sft_train_mask.*.bin + manifest.json
# Expects: tokens + loss_mask (0=user, 1=assistant, 0=EOS)
# Loss: masked_cross_entropy (only on assistant tokens)
```
**Input:** `sft_train_tokens.*.bin`, `sft_train_mask.*.bin`, `sft_manifest.*.json` from `pack_sft.py`

---

### `train_grpo.py` — `PackedGRPODataLoader` + `GRPOPromptDataset`
```python
# Reads: grpo_prompt_tokens.*.bin (length-prefixed uint32) + grpo_answers.*.json
# Also reads packed SFT memmaps for prompt initialization
# Generates: rollouts from prompts, computes rewards against stored answers
```
**Input:** `grpo_prompt_tokens.*.bin`, `grpo_answers.*.json`, `grpo_manifest.*.json` from `pack_grpo.py`

---

## 4. Compatibility Matrix

| Dataset Creator | Mode | Packer | Training Script | Status |
|----------------|------|--------|----------------|--------|
| `dataset_agent.py` | `pretrain` | `pack_pretrain.py` | `train_pretrain.py` | ✅ Compatible |
| `dataset_agent.py` | `sft` | `pack_sft.py` | `train_sft.py` | ✅ Compatible |
| `dataset_agent.py` | `grpo` | `pack_grpo.py` | `train_grpo.py` | ✅ Compatible |
| `codegen_pipeline.py` | `pretrain` | `pack_pretrain.py` | `train_pretrain.py` | ✅ Compatible |
| `codegen_pipeline.py` | `sft` | `pack_sft.py` | `train_sft.py` | ✅ Compatible |
| `codegen_pipeline.py` | `grpo` | `pack_grpo.py` | `train_grpo.py` | ✅ Compatible |
| `hf_to_packed.py` | `pretrain` | `pack_pretrain.py` | `train_pretrain.py` | ✅ Compatible (direct call) |
| `hf_to_packed.py` | `sft` | `pack_sft.py` | `train_sft.py` | ✅ Compatible (direct call) |
| `hf_to_packed.py` | `grpo` | `pack_grpo.py` | `train_grpo.py` | ✅ Compatible (direct call) |

---

## 5. Key Integration Points

### Shared Quality Module (`quality.py`)
All three dataset creators use the **same quality filters**:
- `passes_prose_quality_filter` / `passes_extended_text_quality` (pretrain)
- `passes_sft_pair_quality_filter` / `passes_extended_sft_quality` (sft/grpo)
- `passes_code_quality_filter` (code category)
- `ExactDedup` + `NearDedup` (deduplication)
- `ShardWriter` (sharded JSONL output)

### Shared Recipe System (`recipe.py`)
All packers and training scripts use **`TrainingRecipe`** for:
- ChatML template formatting (`format_user_turn`, `format_assistant_turn`)
- Special tokens (`think_open`, `think_close`, `eos_token`, `pad_token`)
- Mode-aware behavior (`reasoning`, `non_reasoning`, `hybrid`)
- GRPO reward thinking check (`reward_should_check_thinking`)

### Deterministic Train/Val Split
All packers use **identical interleaving split logic**:
```python
def _is_val(record_idx: int, val_fraction: float) -> bool:
    period = max(1, round(1.0 / val_fraction))
    return (record_idx % period) == 0
```

### Multi-Worker Parallelism
All packers support deterministic round-robin file assignment:
```python
[f for i, f in enumerate(files) if i % num_workers == worker]
```

---

## 6. Verified Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATASET CREATORS                                  │
├─────────────────────┬─────────────────────┬─────────────────────────────┤
│ dataset_agent.py    │ codegen_pipeline.py │ hf_to_packed.py             │
│ (web scrape + HF)   │ (LLM codegen + HF)  │ (single HF → packer)        │
└──────────┬──────────┴──────────┬──────────┴─────────────┬──────────────┘
           │                     │                        │
           ▼                     ▼                        ▼
    JSONL shards          JSONL shards              JSONL shards
    (<cat>_XXXXX.jsonl)   (<cat>_XXXXX.jsonl)       (<cat>_XXXXX.jsonl)
           │                     │                        │
           └─────────────────────┼────────────────────────┘
                                 ▼
                    ┌────────────────────────┐
                    │      PACKERS            │
                    ├─────────┬───────────────┤
                    │pack_pre-│pack_sft.py    │pack_grpo.py
                    │train.py │               │
                    └────┬────┴───────┬────────┘
                         ▼           ▼
              ┌─────────────┐ ┌─────────────┐
              │ Memmap .bin │ │ Memmap .bin │
              │ + meta.json │ │ + mask/.json│
              └──────┬──────┘ └──────┬──────┘
                     ▼               ▼
          ┌─────────────────┐ ┌───────────────┐
          │ train_pretrain  │ │ train_sft /   │
          │ .py             │ │ train_grpo.py │
          └─────────────────┘ └───────────────┘
```

---

## 7. Conclusion

**The pipeline is fully seamless.** Every dataset creator outputs JSONL records that:
1. **Match the exact schema** expected by the corresponding packer (`MODE_SCHEMAS`)
2. **Pass through the same quality filters** (`quality.py`) ensuring consistent quality
3. **Use the same shard format** (`ShardWriter` 256MB chunks) for deterministic packing
4. **Produce memmap files** that the training scripts read directly via zero-copy `np.memmap`
5. **Share the same recipe system** (`TrainingRecipe`) for consistent ChatML formatting
6. **Use identical train/val split logic** for reproducibility

No schema translation, format conversion, or adapter code is needed between any stage.