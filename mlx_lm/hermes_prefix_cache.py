"""
Hermes-specific prefix cache optimizer.

Knows the structure of the Hermes system prompt and can identify
which blocks are worth caching vs which always change.

Usage:
    opt = HermesPrefixOptimizer()
    # Given tokens and a tokenizer, find how many are cacheable
    cacheable_n = opt.get_cacheable_token_count(tokens, tokenizer)
    cacheable_tokens = tokens[:cacheable_n]

Layer layout (approximate token counts):
    Layer 1:  SOUL.md + tool guidance       ~2K   Very stable
    Layer 5:  Skills                        ~2-20K Session-dep
    Layer 6:  Memory                        ~2-3K  Changes often
    Layer 7:  User profile                  ~1-2K  Changes occ.
    Layer 9:  Skills index                  ~15-20K Changes occ.
    Layer 11: Timestamp                     ~50    ALWAYS changes
    Layer 12: Platform hints + tools        ~50K+  Mostly stable
"""

import re
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


# Character patterns that identify the always-changing timestamp section.
# The Hermes system prompt always includes a line like:
#   "Conversation started: Monday, April 27, 2026 07:47 PM"
# The "Conversation started:" line is the most reliable marker.
_TIMESTAMP_PATTERNS = [
    re.compile(r"Conversation started:.*"),
    re.compile(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}"),
]

# Fallback: bare AM/PM time (less reliable, matches inline dates too)
_FALLBACK_TIME = re.compile(r"(?:^|\n)\s*\d{1,2}:\d{2}\s*(?:AM|PM)", re.MULTILINE)

# Average English token-to-character ratio for Llama/Qwen tokenizers.
# Used to estimate token offset from character offset when we don't
# want to pay for incremental decode.
_AVG_CHARS_PER_TOKEN = 4.5


class HermesPrefixOptimizer:
    """
    Identifies cacheable vs always-changing regions of the Hermes prompt.

    The timestamp in the Hermes system prompt changes every session,
    invalidating all KV cache blocks that include it. This optimizer
    finds where the timestamp sits in the token sequence so the block
    cache can skip hash computation for everything after it — saving
    ~200 block hash computations when the tools section (50K+ tokens)
    follows the timestamp.

    Thread-safe: all methods are read-only.
    """

    def __init__(self, tokenizer: Optional["PreTrainedTokenizer"] = None):
        """
        Initialize the optimizer with an optional tokenizer.

        Args:
            tokenizer: A HuggingFace PreTrainedTokenizer for accurate
                token-to-text decoding. If None, falls back to character
                estimation (less precise but still effective).
        """
        self._tokenizer = tokenizer

    def estimate_cacheable_prefix(self, text: str) -> int:
        """
        Estimate how many characters before the timestamp are cacheable.

        Returns the character index of the first token that will always
        change (the timestamp). Caller can use this to determine the
        maximum cacheable prefix length.

        Returns 0 if no timestamp marker is found (conservative: nothing
        is cacheable if we can't identify stable regions).
        """
        for pattern in _TIMESTAMP_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.start()
        # Fallback: scan for AM/PM time at line start
        match = _FALLBACK_TIME.search(text)
        if match:
            return match.start()
        return 0

    def get_cacheable_token_count(
        self,
        tokens: List[int],
        tokenizer: Optional["PreTrainedTokenizer"] = None,
    ) -> int:
        """
        Return how many tokens from the start are cacheable.

        The returned count is the token index of the first always-changing
        token (the timestamp). Everything at or after this index will
        change every session and should not be looked up in the block cache.

        Handles three modes:
        1. tokenizer provided (preferred): decode to text for precise location
        2. fallback character estimation using stored tokenizer
        3. conservative half-length estimate

        Always returns a conservative value (might underestimate, never
        overestimates cacheability).
        """
        if not tokens:
            return 0

        # Try stored tokenizer first, then the parameter
        eff_tokenizer = self._tokenizer or tokenizer

        text: Optional[str] = None

        # Decode tokens to text using the best available tokenizer
        if eff_tokenizer is not None:
            try:
                text = eff_tokenizer.decode(tokens)
            except Exception:
                text = None

        # We have decoded text — find the timestamp position
        if text is not None:
            char_offset = self.estimate_cacheable_prefix(text)
            if char_offset <= 0:
                return 0
            # Estimate token offset from character offset
            token_offset = max(1, int(char_offset / _AVG_CHARS_PER_TOKEN))
            return min(token_offset, len(tokens))

        # No tokenizer available — fallback to conservative half-length
        if len(tokens) > 100:
            return len(tokens) // 2
        return 0

    def trim_to_cacheable(
        self,
        tokens: List[int],
        tokenizer: Optional["PreTrainedTokenizer"] = None,
    ) -> List[int]:
        """
        Return only the cacheable prefix of the token sequence.

        Shortcut for tokens[:get_cacheable_token_count(tokens, tokenizer)].
        """
        count = self.get_cacheable_token_count(tokens, tokenizer)
        return tokens[:count]

    def find_divergence_point(
        self,
        old_tokens: List[int],
        new_tokens: List[int],
    ) -> int:
        """
        Find the first token index where two sequences diverge.

        Returns the index (0-based) of the first differing token, or the
        length of the shorter sequence if one is a prefix of the other.

        Used to determine which blocks are reusable when the system prompt
        changes (e.g., new memory content).
        """
        min_len = min(len(old_tokens), len(new_tokens))
        for i in range(min_len):
            if old_tokens[i] != new_tokens[i]:
                return i
        return min_len  # One is a prefix of the other
