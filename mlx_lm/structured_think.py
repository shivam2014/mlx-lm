# Copyright © 2025 mlx-lm contributors
# Structured Chain-of-Thought: Grammar-constrained thinking for reasoning models.
# Based on https://andthattoo.dev/blog/structured_cot

"""
Structured CoT: constrain only the think block with a field-based grammar.

Data flow (step by step, what actually happens):
  1. Model generates tokens one at a time via generate_step
  2. At each step, AFTER the model computes logits but BEFORE sampling,
     this processor runs as a logits_processor
  3. The processor checks: "is the output currently inside a think block?"
     (by scanning token history for think_start / think_end token sequences)
  4. If outside think: logits pass through unchanged
  5. If inside think: the processor applies a mask that only allows
     tokens valid at the current grammar position
  6. After each token is sampled, the processor advances its grammar state

The grammar defines the SHAPE of the think block:
  `` markers
  - Between them: N fields (e.g., GOAL, APPROACH, EDGE), each one line
  - After ``: unconstrained code output

Why this works (Feynman-style, grounded):
  - The model still predicts next-token from its full vocabulary
  - But during thinking, we set invalid token logits to -inf
  - So the sampler can only pick from tokens that continue the
    expected think-block structure
  - The model's hidden state still processes all tokens normally
  - We're not changing HOW the model thinks, just WHAT it's allowed
    to say while thinking
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, Optional, Set, Tuple

import mlx.core as mx

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Grammar state machine
# ---------------------------------------------------------------------------

class ThinkState(Enum):
    """States within the think-block grammar."""
    CODE = auto()              # outside think block — no constraint
    WAITING_PREFIX = auto()    # inside think, expecting a field prefix like "GOAL: "
    IN_LINE = auto()           # inside think, accepting line content (non-newline chars)
    WAITING_THINK_END = auto() # inside think, expecting ""


@dataclass
class GrammarConfig:
    """Configuration for the structured CoT grammar.

    Args:
        fields: Ordered tuple of field names. Each field produces a
            "FIELDNAME: <content>\\n" section inside the think block.
            Default: ("GOAL", "APPROACH", "EDGE").
    """
    fields: Tuple[str, ...] = ("GOAL", "APPROACH", "EDGE")

    @staticmethod
    def from_field_names(*names: str) -> "GrammarConfig":
        return GrammarConfig(fields=tuple(n.upper() for n in names))

    @staticmethod
    def rich() -> "GrammarConfig":
        """LiveCodeBench grammar: GOAL/STATE/ALGO/EDGE/VERIFY."""
        return GrammarConfig(fields=("GOAL", "STATE", "ALGO", "EDGE", "VERIFY"))


def _tokenize_prefixes(
    tokenizer, field_prefixes: Tuple[str, ...]
) -> Tuple[Tuple[int, ...], ...]:
    """Tokenize each field prefix string into a tuple of token IDs.

    Uses the tokenizer's encode method to get the exact token sequence
    for each prefix (e.g., "GOAL: " -> (15513, 969, 25, 220)).
    These token IDs are used to force-emit prefixes deterministically
    during generation, bypassing character-by-character WAITING_PREFIX matching.

    Args:
        tokenizer: A tokenizer with an encode() method.
        field_prefixes: Tuple of prefix strings like ("GOAL: ", "APPROACH: ", "EDGE: ").

    Returns:
        Tuple of token ID tuples, one per field prefix.
    """
    result = []
    for prefix in field_prefixes:
        try:
            token_ids = tokenizer.encode(prefix, add_special_tokens=False)
        except Exception:
            token_ids = []
        result.append(tuple(token_ids))
    return tuple(result)


# ---------------------------------------------------------------------------
# Token mask computation
# ---------------------------------------------------------------------------

def _decode_token(tokenizer, token_id: int) -> str:
    """Decode a single token ID to text."""
    try:
        return tokenizer.decode([token_id])
    except Exception:
        return ""


def _build_token_text_cache(tokenizer) -> Dict[int, str]:
    """Build a cache of token_id -> decoded_text for the full vocabulary."""
    vocab_size = tokenizer.vocab_size
    cache = {}
    for tid in range(vocab_size):
        cache[tid] = _decode_token(tokenizer, tid)
    return cache


def _tokens_matching_literal_prefix(
    token_texts: Dict[int, str],
    expected: str,
    chars_already_consumed: int,
) -> Set[int]:
    """
    Find token IDs whose decoded text is a valid continuation of `expected`
    starting at position `chars_already_consumed`.

    A token matches if:
      - Its decoded text equals the next N characters of expected
      - Or it's a prefix-match that consumes part of the remaining literal
    """
    remaining = expected[chars_already_consumed:]
    if not remaining:
        return set()

    allowed = set()
    for tid, text in token_texts.items():
        if not text:
            continue
        if remaining.startswith(text):
            allowed.add(tid)
        elif text.startswith(remaining):
            allowed.add(tid)
    return allowed


def _tokens_for_non_newline_content(token_texts: Dict[int, str]) -> Set[int]:
    """
    Find token IDs allowed while inside a field's content line.

    A token is allowed if it contains at least one non-newline character, OR
    if it is a bare newline (which signals end-of-field to the state machine).

    Bare newlines must be allowed because the model transitions from IN_LINE
    to the next WAITING_PREFIX via _advance_by_token checking endswith('\\n').
    Blocking bare \\n caused infinite loops: the model couldn't emit the
    transition token and repeated content tokens until max_tokens.
    """
    allowed = set()
    for tid, text in token_texts.items():
        if not text:
            continue
        text_bytes = text.encode('utf-8')
        # Check for newlines NOT at the very end of the token.
        # A token ending in newline is also allowed (it transitions state),
        # but only if there's at least one non-newline char before it.
        has_mid_newline = any(
            b == 0x0A for b in text_bytes[:-1]
        ) if len(text_bytes) > 1 else False
        # Allow bare newline tokens (they trigger field transitions in the
        # state machine's _advance_by_token via endswith('\\n')).
        # Previously blocked because [^\\n]+ was interpreted literally at the
        # mask level, but the state machine handles content-length correctly.
        if text.strip('\\n') == '':
            allowed.add(tid)
            continue
        if not has_mid_newline:
            allowed.add(tid)
    return allowed


def _tokens_for_non_newline_content(token_texts: Dict[int, str]) -> Set[int]:
    """
    Find token IDs whose decoded text contains at least one non-newline
    character (to satisfy [^\\n]+) — or whose decoded text ends with a
    newline (which transitions to the next grammar state).

    A bare newline token (e.g., just '\\n') is NOT allowed: it violates
    the [^\\n]+ rule which requires at least one non-newline character.
    """
    allowed = set()
    for tid, text in token_texts.items():
        if not text:
            continue
        text_bytes = text.encode('utf-8')
        # Check for newlines NOT at the very end of the token.
        # A token ending in newline is also allowed (it transitions state),
        # but only if there's at least one non-newline char before it.
        has_mid_newline = any(
            b == 0x0A for b in text_bytes[:-1]
        ) if len(text_bytes) > 1 else False
        # Reject tokens that are ONLY newlines (no non-newline chars)
        if text.strip('\n') == '':
            continue
        if not has_mid_newline:
            allowed.add(tid)
    return allowed


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

def make_structured_think_processor(
    tokenizer,
    grammar_config: Optional[GrammarConfig] = None,
) -> Callable[[mx.array, mx.array], mx.array]:
    """
    Create a logit processor that constrains think-block output with a grammar.

    Args:
        tokenizer: TokenizerWrapper (must have think_start_tokens, think_end_tokens)
        grammar_config: GrammarConfig defining the think-block structure.
            Defaults to GOAL/APPROACH/EDGE.

    Returns:
        Callable[[mx.array, mx.array], mx.array]: logit processor for generate_step
    """
    if grammar_config is None:
        grammar_config = GrammarConfig()

    think_start_tokens = tokenizer.think_start_tokens
    think_end_tokens = tokenizer.think_end_tokens

    if not think_start_tokens or not think_end_tokens:
        raise ValueError(
            "Tokenizer does not have think start/end tokens. "
            "Structured think requires a reasoning model with `` tags."
        )

    fields = grammar_config.fields
    field_prefixes = tuple(f"{f}: " for f in fields)
    think_end_literal = "\n\n"

    # Pre-compute prefix token sequences for forced emission.
    # Each prefix like "GOAL: " becomes a tuple of token IDs, e.g. (15513, 969, 25, 220).
    # During WAITING_PREFIX, logits are masked to allow ONLY the next forced token,
    # eliminating the character-by-character approach that caused the "GO" display bug.
    field_prefix_token_seqs = _tokenize_prefixes(tokenizer, field_prefixes)

    # Build token text cache
    token_texts = _build_token_text_cache(tokenizer)
    vocab_size = tokenizer.vocab_size

    # Internal state
    state = {
        "inside_think": False,
        "grammar_state": ThinkState.CODE,
        "chars_consumed": 0,
        "field_index": 0,  # 0 = first field, len(fields) = waiting for think_end
        "last_token_count": 0,  # for dedup
        "forced_prefix_remaining": [],  # token IDs yet to be force-emitted for current prefix
        "pending_flush": False,  # True when detokenizer should finalize() after prefix
        "_pending_newline": False,  # True after a lone \n token; \n\n triggers field transition
        "_tokens_in_field": 0,  # tokens generated since last field start
        "_max_field_tokens": 300,  # force field transition after this many tokens
    }

    # Pre-compute masks for all (grammar_state, field_index, chars_consumed) combos
    mask_cache: Dict[Tuple[ThinkState, int, int], mx.array] = {}

    def _get_mask_for_state(
        gs: ThinkState,
        field_index: int = 0,
        chars_consumed: int = 0,
    ) -> mx.array:
        """Get the boolean mask for a given grammar state and position."""
        cache_key = (gs, field_index, chars_consumed)
        if cache_key in mask_cache:
            return mask_cache[cache_key]

        if gs == ThinkState.CODE:
            mask = mx.ones((vocab_size,), dtype=mx.bool_)

        elif gs == ThinkState.IN_LINE:
            allowed = _tokens_for_non_newline_content(token_texts)
            mask = mx.zeros((vocab_size,), dtype=mx.bool_)
            if allowed:
                indices = mx.array(sorted(allowed))
                mask[indices] = True

        elif gs == ThinkState.WAITING_PREFIX:
            # Use forced prefix tokens if available (deterministic emission).
            # Falls back to character-by-character matching only if forced
            # tokens aren't available (e.g., empty prefix or tokenizer failure).
            forced = state.get("forced_prefix_remaining", [])
            if forced:
                # Mask allows ONLY the next forced prefix token
                mask = mx.zeros((vocab_size,), dtype=mx.bool_)
                mask[forced[0]] = True
            elif field_index < len(field_prefixes):
                expected = field_prefixes[field_index]
                allowed = _tokens_matching_literal_prefix(
                    token_texts, expected, chars_consumed
                )
                mask = mx.zeros((vocab_size,), dtype=mx.bool_)
                if allowed:
                    indices = mx.array(sorted(allowed))
                    mask[indices] = True
            else:
                # Shouldn't happen, but allow all if it does
                mask = mx.ones((vocab_size,), dtype=mx.bool_)

        elif gs == ThinkState.WAITING_THINK_END:
            allowed = _tokens_matching_literal_prefix(
                token_texts, think_end_literal, chars_consumed
            )
            mask = mx.zeros((vocab_size,), dtype=mx.bool_)
            if allowed:
                indices = mx.array(sorted(allowed))
                mask[indices] = True

        else:
            mask = mx.ones((vocab_size,), dtype=mx.bool_)

        # Don't cache WAITING_PREFIX masks: they depend on the dynamic
        # forced_prefix_remaining state (set at runtime when entering think block),
        # not just (gs, field_index, chars_consumed). Caching would return stale
        # masks computed during pre-warming when forced_prefix_remaining was empty.
        if gs != ThinkState.WAITING_PREFIX:
            mask_cache[cache_key] = mask
        return mask

    # Pre-warm the cache for all expected states (except WAITING_PREFIX
    # which is computed fresh each time due to forced_prefix_remaining dependency).
    for cc in range(len(think_end_literal) + 1):
        _get_mask_for_state(ThinkState.WAITING_THINK_END, 0, cc)
    _get_mask_for_state(ThinkState.IN_LINE, 0, 0)
    _get_mask_for_state(ThinkState.CODE, 0, 0)

    def _detect_think_region(tokens_list: list) -> bool:
        """Check if the latest token sequence is inside a think block."""
        ts = list(think_start_tokens)
        te = list(think_end_tokens)
        n = len(tokens_list)
        last_start = -1
        last_end = -1
        i = 0
        while i <= n - len(ts):
            if all(tokens_list[i + j] == ts[j] for j in range(len(ts))):
                last_start = i
                i += len(ts)
            else:
                i += 1
        i = 0
        while i <= n - len(te):
            if all(tokens_list[i + j] == te[j] for j in range(len(te))):
                last_end = i
                i += len(te)
            else:
                i += 1
        return last_start > last_end

    def _advance_by_token(token_id: int):
        """Advance grammar state after a token is generated."""
        gs = state["grammar_state"]
        cc = state["chars_consumed"]
        fi = state["field_index"]
        text = token_texts.get(token_id, "")

        if gs == ThinkState.CODE:
            return  # no advancement needed in code state

        if gs == ThinkState.IN_LINE:
            # In line-content: check for field delimiter (\\n\\n).
            #
            # Single \n is content (e.g., code line breaks, list items).
            # Double \n signals end-of-field. This avoids premature field
            # transitions when the model writes multi-line content (code,
            # bullet lists, etc.) within a single field.
            #
            # The _pending_newline flag tracks whether we just saw a lone
            # \\n token, so a second consecutive \n forms \\n\\n.
            _last_pending = state.get("_pending_newline", False)
            # Increment field token counter and enforce max field length
            state["_tokens_in_field"] = state.get("_tokens_in_field", 0) + 1
            if state["_tokens_in_field"] >= state.get("_max_field_tokens", 300):
                # Force field transition due to token limit exceeded
                fi += 1
                state["field_index"] = fi
                state["chars_consumed"] = 0
                if fi >= len(fields):
                    state["grammar_state"] = ThinkState.WAITING_THINK_END
                else:
                    state["grammar_state"] = ThinkState.WAITING_PREFIX
                    state["_tokens_in_field"] = 0
                    if field_prefix_token_seqs and fi < len(field_prefix_token_seqs):
                        state["forced_prefix_remaining"] = list(field_prefix_token_seqs[fi])
                    else:
                        state["forced_prefix_remaining"] = []
                return


            # Check if this token CONTAINS \\n\\n (single token like 271 or 4558)
            if '\n\n' in text:
                # Immediate transition
                state["_pending_newline"] = False
                fi += 1
                state["field_index"] = fi
                state["chars_consumed"] = 0
                if fi >= len(fields):
                    state["grammar_state"] = ThinkState.WAITING_THINK_END
                else:
                    state["grammar_state"] = ThinkState.WAITING_PREFIX
                    state["_tokens_in_field"] = 0
                    if field_prefix_token_seqs and fi < len(field_prefix_token_seqs):
                        state["forced_prefix_remaining"] = list(field_prefix_token_seqs[fi])
                    else:
                        state["forced_prefix_remaining"] = []
                return

            # Check if this is a bare \n token (= token entirely \\n)
            if text == '\n':
                if _last_pending:
                    # Two consecutive \\n tokens = \\n\\n → field transition
                    state["_pending_newline"] = False
                    fi += 1
                    state["field_index"] = fi
                    state["chars_consumed"] = 0
                    if fi >= len(fields):
                        state["grammar_state"] = ThinkState.WAITING_THINK_END
                    else:
                        state["grammar_state"] = ThinkState.WAITING_PREFIX
                        state["_tokens_in_field"] = 0
                        if field_prefix_token_seqs and fi < len(field_prefix_token_seqs):
                            state["forced_prefix_remaining"] = list(field_prefix_token_seqs[fi])
                        else:
                            state["forced_prefix_remaining"] = []
                    return
                else:
                    # First bare \n — might be start of \\n\\n, stay in IN_LINE
                    state["_pending_newline"] = True
                    return

            # Token ends with \\n (e.g., "  \\n" indent tokens):
            # set pending flag so next \\n forms the delimiter
            if text.endswith('\n'):
                state["_pending_newline"] = True
            else:
                # Non-newline content resets the pending flag
                state["_pending_newline"] = False
            return

        if gs == ThinkState.WAITING_PREFIX:
            # If we have forced prefix tokens, consume them deterministically
            forced = state.get("forced_prefix_remaining", [])
            if forced:
                # Verify the received token matches the expected forced token.
                # If it doesn't match (e.g., an intermediate prompt token between
                # think_start and the actual generation), skip it without consuming.
                if token_id != forced[0]:
                    return
                forced.pop(0)
                if not forced:
                    # All forced prefix tokens consumed — transition to IN_LINE
                    state["grammar_state"] = ThinkState.IN_LINE
                    state["chars_consumed"] = 0
                    # Signal that detokenizer should finalize to flush prefix text
                    state["pending_flush"] = True
                return

            # Fallback: character-by-character prefix matching (no forced tokens)
            if fi < len(field_prefixes):
                expected = field_prefixes[fi]
            else:
                expected = ""
            remaining = expected[cc:]
            consumed = min(len(text), len(remaining))
            new_cc = cc + consumed

            if new_cc >= len(expected):
                # Prefix fully consumed — transition to line content
                state["grammar_state"] = ThinkState.IN_LINE
                state["chars_consumed"] = 0

                # If token extends past the prefix, handle the overflow
                overflow = text[consumed:]
                if overflow:
                    if '\n' in overflow:
                        # Line ended within this token — skip to next field
                        fi += 1
                        state["field_index"] = fi
                        if fi >= len(fields):
                            state["grammar_state"] = ThinkState.WAITING_THINK_END
                        else:
                            state["grammar_state"] = ThinkState.WAITING_PREFIX
                            state["_tokens_in_field"] = 0
                    else:
                        # Overflow is valid line content, stay in IN_LINE
                        pass
            else:
                state["chars_consumed"] = new_cc
            return

        if gs == ThinkState.WAITING_THINK_END:
            remaining = think_end_literal[cc:]
            consumed = min(len(text), len(remaining))
            new_cc = cc + consumed

            if new_cc >= len(think_end_literal):
                # Think end fully consumed — exit to CODE
                state["grammar_state"] = ThinkState.CODE
                state["chars_consumed"] = 0
                state["field_index"] = 0
                state["inside_think"] = False  # grammar is the authority
            else:
                state["chars_consumed"] = new_cc
            return

    def processor(tokens, logits):
        """
        Logit processor: constrains tokens during think blocks.

        Args:
            tokens: mx.array of token history (shape: (seq_len,))
            logits: mx.array of logit scores (shape: (1, vocab_size) or (vocab_size,))

        Returns:
            logits with non-allowed tokens set to -inf during think blocks
        """
        squeeze = False
        if logits.ndim == 1:
            logits = logits[None, :]
            squeeze = True

        # Convert to list for think-region detection
        if hasattr(tokens, 'tolist'):
            tokens_list = tokens.tolist()
        else:
            tokens_list = list(tokens)

        # Detect think region FIRST — before advancing grammar state
        was_inside = state["inside_think"]
        state["inside_think"] = _detect_think_region(tokens_list)

        if state["inside_think"] and not was_inside:
            # Just entered think — reset grammar to start
            state["grammar_state"] = ThinkState.WAITING_PREFIX
            state["_tokens_in_field"] = 0
            state["chars_consumed"] = 0
            state["field_index"] = 0
            # Queue the first field's prefix tokens for forced emission
            if field_prefix_token_seqs and len(field_prefix_token_seqs[0]) > 0:
                state["forced_prefix_remaining"] = list(field_prefix_token_seqs[0])
            else:
                state["forced_prefix_remaining"] = []
            state["pending_flush"] = False
            # Rewind last_token_count to after the think-start token,
            # so tokens inside the think block get re-processed correctly.
            ts = list(think_start_tokens)
            n = len(tokens_list)
            last_start = -1
            i = 0
            while i <= n - len(ts):
                if all(tokens_list[i + j] == ts[j] for j in range(len(ts))):
                    last_start = i
                    i += len(ts)
                else:
                    i += 1
            if last_start >= 0:
                state["last_token_count"] = last_start + len(ts)

        # Exit detection: grammar state machine is the authority.
        # _detect_think_region may see a </think> token before the grammar
        # has finished consuming "</think>\n\n".  Don't exit until grammar
        # agrees (i.e., grammar_state is CODE).
        if was_inside and not state["inside_think"]:
            if state["grammar_state"] != ThinkState.CODE:
                state["inside_think"] = True  # override: grammar still in think

        if not state["inside_think"]:
            # Outside think — no constraint
            if was_inside:
                state["grammar_state"] = ThinkState.CODE
                state["chars_consumed"] = 0
                state["field_index"] = 0
                state["forced_prefix_remaining"] = []
                state["pending_flush"] = False
            state["last_token_count"] = len(tokens_list)
            if squeeze:
                logits = logits.squeeze(0)
            return logits

        # Advance grammar state on new tokens
        current_count = len(tokens_list)
        while state["last_token_count"] < current_count:
            idx = state["last_token_count"]
            if idx < len(tokens_list):
                _advance_by_token(tokens_list[idx])
            state["last_token_count"] += 1

        # Inside think — apply grammar mask
        gs = state["grammar_state"]
        fi = state["field_index"]
        cc = state["chars_consumed"]
        mask = _get_mask_for_state(gs, fi, cc)

        # Pad mask to match logits vocab size (model may have extra tied-embedding slots)
        logits_vocab = logits.shape[-1]
        if mask.shape[0] < logits_vocab:
            pad_val = bool(gs == ThinkState.CODE)  # True for CODE (allow all), False otherwise
            pad = mx.full((logits_vocab - mask.shape[0],), pad_val, dtype=mx.bool_)
            mask = mx.concatenate([mask, pad])

        logits = mx.where(mask[None, :], logits, NEG_INF)

        if squeeze:
            logits = logits.squeeze(0)
        return logits

    # Expose for testing and for generate.py to check flush state
    processor._state = state
    processor._mask_cache = mask_cache
    processor._config = grammar_config
    processor._field_prefix_token_seqs = field_prefix_token_seqs

    @property
    def pending_flush(self):
        """True if the detokenizer should finalize() to flush buffered prefix text.

        Set after a field prefix has been fully force-emitted. The caller
        (generate.py) should check this after each generation step and call
        finalize() on the detokenizer if True, then reset it to False.

        This fixes the display bug where single-byte prefix tokens like
        'G' and 'O' get buffered by the streaming detokenizer, causing
        the user to see 'AL:' instead of 'GOAL:'.
        """
        result = state.get("pending_flush", False)
        if result:
            state["pending_flush"] = False
        return result

    return processor
