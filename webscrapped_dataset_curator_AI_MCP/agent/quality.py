"""
quality.py

Shared quality-filtering / dedup / shard-writing primitives, so the
live-scraping agent applies the same bar as the HF-streaming pipeline.

Filters (in increasing order of sophistication):
  1. Length / alpha-ratio / repetition / junk markers (original)
  2. Zlib compression ratio  -- catches boilerplate, logs, templates
  3. Line-level repetition   -- catches nav bars, repeated headers/footers
  4. Vocabulary diversity    -- catches spammy / low-info text
  5. Short-line ratio        -- catches navigation-heavy pages
  6. Flagged n-grams         -- known low-quality patterns (TOS, copyright, etc.)
  7. Language detection      -- fasttext-based, optional
"""

import hashlib
import json
import os
import re
import zlib
from collections import deque, Counter
from typing import Optional

SHARD_MAX_BYTES = 256 * 1024 * 1024  # 256MB per shard file

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_LINE_RE = re.compile(r".*$", re.MULTILINE)

_JUNK_MARKERS = (
    "enable javascript", "cookies to continue", "subscribe to continue",
    "404 not found", "access denied", "please verify you are a human",
    "add to cart", "sign in to your account", "captcha",
)

# ---------------------------------------------------------------------------
# Flagged n-grams -- substrings that strongly correlate with low-quality text
# regardless of domain.  These are checked as case-insensitive substring
# matches; even a few hits flags the line as boilerplate.
# ---------------------------------------------------------------------------
_FLAGGED_NGRAMS = [
    # copyright / terms boilerplate
    "all rights reserved", "terms of service", "privacy policy",
    "copyright ©", "all trademarks are", "subject to change without notice",
    # navigation / layout
    "click here", "read more", "learn more", "subscribe now",
    "sign up for", "follow us on", "share this", "leave a comment",
    "comments are closed", "related posts", "you might also like",
    "top of page", "back to top", "scroll to", "table of contents",
    # cookie / consent / GDPR
    "accept cookies", "cookie policy", "we use cookies", "consent to",
    "your privacy", "do not sell", "cookie settings", "accept all",
    # ad / promotional
    "advertisement", "sponsored content", "brought to you by",
    "download now", "limited time offer", "buy now", "add to cart",
    # pagination
    "page 1 of", "next page", "previous page", "first page", "last page",
    # generic low-effort
    "lorem ipsum", "under construction", "coming soon",
    "this is a placeholder", "your message has been sent",
]

# Regex for fast flagged-ngram matching (built once at import time)
_FLAGGED_RE = re.compile(
    "|".join(re.escape(ng) for ng in _FLAGGED_NGRAMS),
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Language detection helper (lazy-loaded)
# ---------------------------------------------------------------------------
_LANG_DETECTOR = None  # populated on first call to detect_language()


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(1 for c in text if c.isalpha())
    return alpha / len(text)


def _alnum_ratio(text: str) -> float:
    if not text:
        return 0.0
    content = sum(1 for c in text if c.isalnum())
    return content / len(text)


def _top_word_repetition_ratio(text: str) -> float:
    words = _WORD_RE.findall(text.lower())
    if len(words) < 10:
        return 0.0
    counts: dict = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return max(counts.values()) / len(words)


def passes_prose_quality_filter(text: str, min_doc_chars: int = 500) -> tuple:
    """Returns (passed: bool, reason: Optional[str])."""
    if len(text) < min_doc_chars:
        return False, "too_short"
    if _alpha_ratio(text) < 0.5:
        return False, "low_alpha_ratio"
    if _top_word_repetition_ratio(text) > 0.40:
        return False, "high_repetition"
    lines = text.split("\n")
    if len(lines) < 3 and len(text) > 2000:
        return False, "single_block_too_long"
    lowered = text.lower()
    for marker in _JUNK_MARKERS:
        if marker in lowered:
            return False, f"junk_marker:{marker}"
    return True, None


def passes_sft_pair_quality_filter(prompt: str, answer: str, min_chars: int = 20) -> tuple:
    """Quality bar for a (prompt, answer) pair that the source already labeled."""
    prompt = (prompt or "").strip()
    answer = (answer or "").strip()
    if not prompt:
        return False, "empty_prompt"
    if not answer:
        return False, "empty_answer"
    combined = f"{prompt}\n\n{answer}"
    if len(combined) < min_chars:
        return False, "too_short"
    if _alnum_ratio(combined) < 0.25:
        return False, "low_content_ratio"
    if answer.strip().lower() == prompt.strip().lower():
        return False, "answer_echoes_prompt"
    lowered = combined.lower()
    for marker in _JUNK_MARKERS:
        if marker in lowered:
            return False, f"junk_marker:{marker}"
    return True, None


def _has_extractable_answer(answer: str) -> tuple:
    """Check if an answer contains content that the GRPO reward function can
    actually score: a \\boxed{...} expression (common in math datasets), a
    number, or a short concrete token. These are the same patterns the reward
    function in train_grpo.py extracts -- see ANSWER_RE and NUM_RE there.

    Returns (passed: bool, reason: Optional[str])."""
    if not answer:
        return False, "empty_answer"
    # \\boxed{...} capture -- single-level braces, matches math datasets
    if re.search(r"\\boxed\{[^}]+\}", answer):
        return True, None
    # Numeric fallback (GSM8K, ScienceQA, etc.)
    if re.search(r"-?\d+(?:\.\d+)?", answer):
        return True, None
    # Short concrete token (single word up to ~40 chars)
    stripped = answer.strip()
    if stripped and len(stripped.split()) <= 3 and len(stripped) <= 60:
        return True, None
    # Code output pattern (e.g. "True", "False", "None", "42", "error")
    if re.search(r"^(?:true|false|none|null|ok|yes|no)$", answer.strip().lower()):
        return True, None
    return False, "no_extractable_answer"


def passes_grpo_pair_quality_filter(prompt: str, answer: str, min_chars: int = 20) -> tuple:
    """Quality bar for a (prompt, answer) pair being prepared for GRPO
    training, where the answer must be machine-extractable by the reward
    function. Reuses all the checks from passes_sft_pair_quality_filter AND
    additionally verifies the answer contains extractable content (numeric,
    boxed, or a short concrete token).

    Returns (passed, reason). Checks:
    - all passes_sft_pair_quality_filter checks apply
    - answer must have extractable content (boxed/numeric/short concrete)"""
    prompt = (prompt or "").strip()
    answer = (answer or "").strip()
    if not prompt:
        return False, "empty_prompt"
    if not answer:
        return False, "empty_answer"
    combined = f"{prompt}\n\n{answer}"
    if len(combined) < min_chars:
        return False, "too_short"
    if _alnum_ratio(combined) < 0.25:
        return False, "low_content_ratio"
    if answer.strip().lower() == prompt.strip().lower():
        return False, "answer_echoes_prompt"
    lowered = combined.lower()
    for marker in _JUNK_MARKERS:
        if marker in lowered:
            return False, f"junk_marker:{marker}"
    extractable_ok, extractable_reason = _has_extractable_answer(answer)
    if not extractable_ok:
        return False, extractable_reason
    return True, None


_TRANSCRIPT_JUNK_MARKERS = ("[music]", "[applause]", "[laughter]", "♪ ♪ ♪")


def passes_transcript_quality_filter(text: str, min_doc_chars: int = 300) -> tuple:
    if len(text) < min_doc_chars:
        return False, "too_short"
    if _alpha_ratio(text) < 0.5:
        return False, "low_alpha_ratio"
    if _top_word_repetition_ratio(text) > 0.35:
        return False, "high_repetition"
    lowered = text.lower()
    stripped = re.sub(r"\[music\]|\[applause\]|\[laughter\]|♪", "", lowered)
    if len(stripped) < min_doc_chars * 0.5:
        return False, "mostly_nonspeech"
    return True, None


CODE_ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp", ".hpp",
    ".cc", ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala",
    ".sh", ".sql", ".r", ".m", ".jl", ".lua", ".ml", ".hs", ".erl", ".ex",
}
CODE_SKIP_PATH_MARKERS = (
    "node_modules/", "vendor/", "third_party/", "dist/", "build/",
    ".min.js", ".min.css", "-lock.json", ".lock", "generated", ".pb.go",
)
CODE_MAX_LINE_LEN = 1000
CODE_MAX_AVG_LINE_LEN = 200


def passes_code_quality_filter(text: str, path: str, min_doc_chars: int = 500) -> tuple:
    if len(text) < min_doc_chars:
        return False, "too_short"
    lower_path = path.lower()
    ext = os.path.splitext(lower_path)[1]
    if ext and ext not in CODE_ALLOWED_EXTENSIONS:
        return False, "disallowed_extension"
    for marker in CODE_SKIP_PATH_MARKERS:
        if marker in lower_path:
            return False, f"skip_path:{marker}"
    lines = text.split("\n")
    if any(len(l) > CODE_MAX_LINE_LEN for l in lines):
        return False, "line_too_long"
    avg_line_len = sum(len(l) for l in lines) / max(1, len(lines))
    if avg_line_len > CODE_MAX_AVG_LINE_LEN:
        return False, "avg_line_too_long"
    return True, None


# ---------------------------------------------------------------------------
# Extended quality filters (v2)
# ---------------------------------------------------------------------------

def compression_ratio(text: str) -> float:
    """Zlib compression ratio: compressed_size / raw_size.

    Lower values = more compressible = more boilerplate / templated / spammy.
    Typical ranges:
        Normal prose English:     0.45 - 0.60
        Boilerplate / navigation: 0.20 - 0.40
        Log files / templates:    0.10 - 0.35
        Code:                     0.35 - 0.55
        Random / high-entropy:    0.60+
    """
    raw = text.encode("utf-8", errors="ignore")
    if len(raw) < 50:
        return 1.0  # too short to measure meaningfully
    compressed = zlib.compress(raw, level=1)
    return len(compressed) / max(1, len(raw))


def line_repetition_score(text: str) -> float:
    """Fraction of lines that appear more than once (exact duplicate).

    A high score means the document is dominated by repeated boilerplate
    lines (navigation, headers, footers, templates).  Normal prose
    typically scores < 0.05.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 5:
        return 0.0
    counts = Counter(lines)
    repeated_lines = sum(c - 1 for c in counts.values() if c > 1)
    return repeated_lines / max(1, len(lines))


def adjacent_line_repetition_score(text: str) -> float:
    """Fraction of adjacent line pairs that are nearly identical.

    Catches content like:
        Section 1: ...
        Section 2: ...
        Section 3: ...
    where each line is unique but structurally repetitive.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 5:
        return 0.0
    similar = 0
    total = 0
    for i in range(len(lines) - 1):
        total += 1
        # Simple similarity: same first 30 chars counts as "adjacent repeat"
        if lines[i][:30] == lines[i + 1][:30]:
            similar += 1
    return similar / max(1, total)


def vocabulary_diversity(text: str) -> float:
    """Fraction of unique words over total words.

    Normal prose:        0.30 - 0.55
    Repetitive / spammy: 0.05 - 0.20
    Code:                0.20 - 0.40
    """
    words = _WORD_RE.findall(text.lower())
    if len(words) < 10:
        return 1.0
    return len(set(words)) / len(words)


def short_line_ratio(text: str, max_len: int = 30) -> float:
    """Fraction of lines shorter than *max_len* characters.

    High ratio indicates navigation-heavy pages, tag clouds, or
    link-list pages with little prose content.
    """
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 5:
        return 0.0
    short = sum(1 for l in lines if len(l.strip()) <= max_len)
    return short / len(lines)


def flagged_ngram_line_ratio(text: str) -> float:
    """Fraction of lines that contain at least one flagged n-gram.

    Uses a compiled regex over _FLAGGED_NGRAMS.  Even a few
    flagged lines indicate boilerplate; > 0.10 is suspicious.
    """
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 3:
        return 0.0
    flagged = sum(1 for l in lines if _FLAGGED_RE.search(l))
    return flagged / len(lines)


# ---------------------------------------------------------------------------
# Language detection (lightweight, optional)
# ---------------------------------------------------------------------------

def detect_language(
    text: str,
    target_langs: Optional[set] = None,
    threshold: float = 0.5,
    lazy_init: bool = True,
) -> tuple:
    """Detect document language using fasttext (if installed).

    Args:
        text:        Document text to classify.
        target_langs: Set of ISO codes to accept (e.g. {"en"}, {"en", "de"}).
                     If None, returns the detected language without filtering.
        threshold:   Minimum probability to accept.  Below this, returns
                     ("unknown", False).
        lazy_init:   If True, the fasttext model is loaded on first call
                     and cached thereafter.

    Returns:
        (lang_code: str, is_target: bool or None)
        - lang_code is the detected ISO-639-1 code (e.g. "en", "de", "fr").
        - is_target is True/False if target_langs is set, else None.
    """
    lang_code = _detect_lang_fasttext(text, lazy_init=lazy_init)
    if lang_code is None:
        return ("unknown", None)
    if target_langs is not None:
        return (lang_code, lang_code in target_langs)
    return (lang_code, None)


def _detect_lang_fasttext(text: str, lazy_init: bool = True) -> Optional[str]:
    """Internal: run fasttext language ID on *text*.  Returns ISO code or None."""
    global _LANG_DETECTOR
    if lazy_init and _LANG_DETECTOR is None:
        _LANG_DETECTOR = _load_fasttext_lang_detector()
    detector = _LANG_DETECTOR
    if detector is None:
        return None
    try:
        sample = text.strip()[:2000]
        if len(sample) < 20:
            return None
        predictions = detector.predict(sample, k=1)
        lang = predictions[0][0].replace("__label__", "")
        return lang
    except Exception:
        return None


def _load_fasttext_lang_detector():
    """Try to load fasttext's lid.176 model.  Returns None if unavailable."""
    try:
        import fasttext
        import fasttext.util
    except ImportError:
        return None
    # Common paths for the language identification model
    candidates = [
        "/usr/local/lib/python3.10/site-packages/fasttext/util/lid.176.ftz",
        "/usr/local/lib/python3.11/site-packages/fasttext/util/lid.176.ftz",
        "/usr/local/lib/python3.12/site-packages/fasttext/util/lid.176.ftz",
        os.path.expanduser("~/.cache/fasttext/lid.176.ftz"),
        os.path.expanduser("~/.fasttext/lid.176.ftz"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return fasttext.load_model(path)
            except Exception:
                continue
    # Try auto-download
    try:
        fasttext.util.download_model("lid.176")  # noqa
        path = fasttext.util.model_path("lid.176")
        if os.path.exists(path):
            return fasttext.load_model(path)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Combined extended quality filters
# ---------------------------------------------------------------------------

# Sensible defaults for extended quality checks
_EXTENDED_DEFAULTS = {
    "min_chars": 500,
    "max_compression_ratio": 0.35,   # zlib ratio; lower = more repetitive
    "max_line_repetition": 0.15,     # fraction of duplicate lines
    "max_adjacent_repetition": 0.15, # fraction of adjacent near-duplicate lines
    "min_vocab_diversity": 0.15,     # unique / total words
    "max_short_line_ratio": 0.50,    # fraction of lines < 30 chars
    "max_flagged_ngram_ratio": 0.10, # fraction of lines with flagged n-grams
}


def passes_extended_text_quality(
    text: str,
    min_chars: int = 500,
    max_compression_ratio: float = 0.35,
    max_line_repetition: float = 0.15,
    max_adjacent_repetition: float = 0.15,
    min_vocab_diversity: float = 0.15,
    max_short_line_ratio: float = 0.50,
    max_flagged_ngram_ratio: float = 0.10,
    target_langs: Optional[set] = None,
) -> tuple:
    """Multi-signal text quality gate for prose documents.

    Returns (passed: bool, reason: str).  Each check returns a specific
    reason so you can tune thresholds per dataset.

    Extends the basic ``passes_prose_quality_filter`` with:
      - Zlib compression ratio (boilerplate / logs / templates)
      - Line-level repetition (nav bars, headers, footers)
      - Adjacent-line repetition (templated vertical lists)
      - Vocabulary diversity (spam / low-info text)
      - Short-line ratio (navigation-heavy pages)
      - Flagged n-gram ratio (copyright, cookie-consent, ad markers)
      - Language detection (fasttext, optional / graceful fallback)
    """
    # --- Basic prose-level checks (original) ---
    passed, reason = passes_prose_quality_filter(text, min_chars)
    if not passed:
        return False, reason

    # --- Compression ratio (boilerplate / logs) ---
    cr = compression_ratio(text)
    if cr < max_compression_ratio:
        return False, f"high_compressibility:{cr:.3f}"

    # --- Line repetition (nav / footer) ---
    lr = line_repetition_score(text)
    if lr > max_line_repetition:
        return False, f"high_line_repetition:{lr:.3f}"

    # --- Adjacent line repetition (templates) ---
    ar = adjacent_line_repetition_score(text)
    if ar > max_adjacent_repetition:
        return False, f"high_adjacent_repetition:{ar:.3f}"

    # --- Vocabulary diversity (spam) ---
    vd = vocabulary_diversity(text)
    if vd < min_vocab_diversity:
        return False, f"low_vocab_diversity:{vd:.3f}"

    # --- Short line ratio (navigation) ---
    sr = short_line_ratio(text)
    if sr > max_short_line_ratio:
        return False, f"high_short_line_ratio:{sr:.3f}"

    # --- Flagged n-grams (boilerplate) ---
    nr = flagged_ngram_line_ratio(text)
    if nr > max_flagged_ngram_ratio:
        return False, f"high_flagged_ngrams:{nr:.3f}"

    # --- Language detection (optional) ---
    if target_langs is not None:
        _, is_target = detect_language(text, target_langs=target_langs)
        if is_target is not None and not is_target:
            return False, "wrong_language"

    return True, None


def passes_extended_sft_quality(
    prompt: str,
    answer: str,
    min_chars: int = 20,
    max_compression_ratio: float = 0.35,
    max_flagged_ngram_ratio: float = 0.15,
    target_langs: Optional[set] = None,
) -> tuple:
    """Multi-signal quality gate for (prompt, answer) pairs.

    Applies the SFT pair filter first, then extended content checks on
    the combined text.
    """
    passed, reason = passes_sft_pair_quality_filter(prompt, answer, min_chars)
    if not passed:
        return False, reason

    combined = f"{prompt}\n\n{answer}"

    # Compression ratio
    cr = compression_ratio(combined)
    if cr < max_compression_ratio:
        return False, f"high_compressibility:{cr:.3f}"

    # Flagged n-grams
    nr = flagged_ngram_line_ratio(combined)
    if nr > max_flagged_ngram_ratio:
        return False, f"high_flagged_ngrams:{nr:.3f}"

    # Language (optional)
    if target_langs is not None:
        _, is_target = detect_language(combined, target_langs=target_langs)
        if is_target is not None and not is_target:
            return False, "wrong_language"

    return True, None


def score_text_quality(
    text: str,
    min_chars: int = 500,
    target_langs: Optional[set] = None,
) -> float:
    """Return a continuous quality score in [0, 1] for ranking / filtering.

    Unlike the pass/fail gates above, this returns a score you can use
    for weighted sampling.  Higher = better quality.

    Heuristic components (each 0-1):
      - Compression score: 1.0 if ratio >= 0.55, 0.0 if <= 0.30, linear ramp
      - Vocabulary score:  1.0 if diversity >= 0.35, 0.0 if <= 0.10, linear ramp
      - Line repetition:   1.0 if rep <= 0.05, 0.0 if >= 0.30, linear ramp
      - Flagged n-gram:    1.0 if ratio <= 0.02, 0.0 if >= 0.15, linear ramp
      - Length bonus:      ln(len) clamped to [0.3, 1.0] -- longer docs slightly preferred
      - Language bonus:    +0.1 if target_langs match, no penalty if unset
    """
    if len(text) < min_chars:
        return 0.0

    scores = []

    # Compression
    cr = compression_ratio(text)
    comp_score = max(0.0, min(1.0, (cr - 0.30) / 0.25))
    scores.append(comp_score)

    # Vocabulary diversity
    vd = vocabulary_diversity(text)
    vocab_score = max(0.0, min(1.0, (vd - 0.10) / 0.25))
    scores.append(vocab_score)

    # Line repetition
    lr = line_repetition_score(text)
    rep_score = max(0.0, min(1.0, (0.30 - lr) / 0.25))
    scores.append(rep_score)

    # Flagged n-grams
    nr = flagged_ngram_line_ratio(text)
    flag_score = max(0.0, min(1.0, (0.15 - nr) / 0.13))
    scores.append(flag_score)

    # Length bonus: log-scale preference for longer docs
    length_bonus = max(0.3, min(1.0, len(text) / 10000.0 * 0.7 + 0.3))

    combined = sum(scores) / len(scores) * length_bonus

    # Language bonus
    if target_langs is not None:
        _, is_target = detect_language(text, target_langs=target_langs)
        if is_target:
            combined = min(1.0, combined + 0.1)

    return max(0.0, min(1.0, combined))


# ---------------------------------------------------------------------------
# End of extended quality filters
# ---------------------------------------------------------------------------


class _BloomFilter:
    """Fixed-size Bloom filter with sha256-based hash slices."""
    def __init__(self, size_bytes: int, num_hashes: int = 4):
        self.num_bits = max(8, size_bytes * 8)
        self.num_hashes = num_hashes
        self.bits = bytearray(size_bytes)

    def _slices(self, item: bytes):
        digest = hashlib.sha256(item).digest()
        for i in range(self.num_hashes):
            chunk = digest[i * 4:i * 4 + 4]
            yield int.from_bytes(chunk, "little") % self.num_bits

    def add(self, item: bytes) -> None:
        for bit in self._slices(item):
            self.bits[bit // 8] |= (1 << (bit % 8))

    def contains(self, item: bytes) -> bool:
        return all(self.bits[bit // 8] & (1 << (bit % 8)) for bit in self._slices(item))


class ExactDedup:
    """Exact-duplicate filter. Uses Bloom filter if max_memory_mb is set."""
    def __init__(self, persist_path: Optional[str] = None, max_memory_mb: Optional[float] = None):
        self.persist_path = persist_path
        self._fh = None
        self._bloom: Optional[_BloomFilter] = None
        self._seen: Optional[set] = None
        if max_memory_mb:
            self._bloom = _BloomFilter(size_bytes=int(max_memory_mb * 1024 * 1024))
        else:
            self._seen = set()
        if persist_path and os.path.exists(persist_path):
            with open(persist_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        h = bytes.fromhex(line)
                        if self._bloom is not None:
                            self._bloom.add(h)
                        else:
                            self._seen.add(h)
        if persist_path:
            self._fh = open(persist_path, "a", buffering=1)

    def is_duplicate(self, text: str) -> bool:
        h = hashlib.sha1(text.encode("utf-8", errors="ignore")).digest()
        if self._bloom is not None:
            if self._bloom.contains(h):
                return True
            self._bloom.add(h)
        else:
            if h in self._seen:
                return True
            self._seen.add(h)
        if self._fh:
            self._fh.write(h.hex() + "\n")
        return False

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


class _ShingleNearDedup:
    """Fallback near-dedup via shingle-overlap comparison."""
    def __init__(self, shingle_size: int = 5, threshold: float = 0.8):
        self.shingle_size = shingle_size
        self.threshold = threshold
        self._fingerprints = []

    def _shingles(self, text: str):
        words = _WORD_RE.findall(text.lower())
        n = self.shingle_size
        return {
            hashlib.md5(" ".join(words[i:i + n]).encode()).hexdigest()
            for i in range(0, max(0, len(words) - n + 1), n)
        }

    def is_near_duplicate(self, text: str) -> bool:
        shingles = self._shingles(text)
        if not shingles:
            return False
        sample = shingles if len(shingles) <= 500 else set(list(shingles)[:500])
        for fp in self._fingerprints:
            inter = len(sample & fp)
            union = len(sample | fp)
            if union and inter / union >= self.threshold:
                return True
        if len(self._fingerprints) > 5000:
            self._fingerprints.pop(0)
        self._fingerprints.append(sample)
        return False


class _MinHashLSHNearDedup:
    """MinHash + LSH near-dedup via datasketch."""
    def __init__(self, shingle_size: int = 5, num_perm: int = 128, threshold: float = 0.8,
                 max_docs: Optional[int] = 200_000):
        from datasketch import MinHash, MinHashLSH
        self._MinHash = MinHash
        self.shingle_size = shingle_size
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._counter = 0
        self.max_docs = max_docs
        self._keys = deque()

    def _minhash(self, text: str):
        words = _WORD_RE.findall(text.lower())
        n = self.shingle_size
        mh = self._MinHash(num_perm=self.num_perm)
        for i in range(0, max(0, len(words) - n + 1), n):
            mh.update(" ".join(words[i:i + n]).encode())
        return mh

    def is_near_duplicate(self, text: str) -> bool:
        mh = self._minhash(text)
        if self.lsh.query(mh):
            return True
        self._counter += 1
        key = f"doc-{self._counter}"
        self.lsh.insert(key, mh)
        self._keys.append(key)
        if self.max_docs and len(self._keys) > self.max_docs:
            oldest = self._keys.popleft()
            try:
                self.lsh.remove(oldest)
            except KeyError:
                pass
        return False


def NearDedup(shingle_size: int = 5, threshold: float = 0.8, max_docs: Optional[int] = 200_000):
    """Factory: returns datasketch-backed LSH if installed, else shingle fallback."""
    try:
        return _MinHashLSHNearDedup(shingle_size=shingle_size, threshold=threshold, max_docs=max_docs)
    except ImportError:
        return _ShingleNearDedup(shingle_size=shingle_size, threshold=threshold)


class RunState:
    """Persists used-queries and seen-URLs per category for resumable runs."""
    def __init__(self, out_dir: str, category: str):
        self.path = os.path.join(out_dir, category, ".run_state.json")
        self.used_queries: list = []
        self.seen_urls: set = set()
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                self.used_queries = data.get("used_queries", [])
                self.seen_urls = set(data.get("seen_urls", []))
            except Exception:
                pass

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "used_queries": self.used_queries[-2000:],
                "seen_urls": list(self.seen_urls)[-50000:],
            }, f)


class ShardWriter:
    """Writes JSONL records to size-capped shard files."""
    def __init__(self, out_dir: str, category: str, max_shard_bytes: int = SHARD_MAX_BYTES):
        self.dir = os.path.join(out_dir, category)
        os.makedirs(self.dir, exist_ok=True)
        self.category = category
        self.max_shard_bytes = max_shard_bytes
        existing = [f for f in os.listdir(self.dir)
                     if f.startswith(f"{category}_") and f.endswith(".jsonl")]
        self.shard_idx = max((int(f[len(category) + 1: -6]) for f in existing), default=-1) + 1
        self.bytes_in_shard = 0
        self.total_bytes = 0
        self.total_docs = 0
        self._fh = self._open_new_shard()

    def _open_new_shard(self):
        path = os.path.join(self.dir, f"{self.category}_{self.shard_idx:05d}.jsonl")
        return open(path, "w", encoding="utf-8")

    def write(self, record: dict):
        line = json.dumps(record, ensure_ascii=False) + "\n"
        line_bytes = len(line.encode("utf-8"))
        if self.bytes_in_shard + line_bytes > self.max_shard_bytes and self.bytes_in_shard > 0:
            self._fh.close()
            self.shard_idx += 1
            self.bytes_in_shard = 0
            self._fh = self._open_new_shard()
        self._fh.write(line)
        self.bytes_in_shard += line_bytes
        self.total_bytes += line_bytes
        self.total_docs += 1

    def close(self):
        self._fh.close()
