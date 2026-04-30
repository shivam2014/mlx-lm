# Copyright © 2025 mlx-lm contributors
"""Unit tests for structured CoT grammar processor."""

import unittest
from unittest.mock import MagicMock

import mlx.core as mx

from mlx_lm.structured_think import (
    GrammarConfig,
    ThinkState,
    _tokens_for_non_newline_content,
    _tokens_matching_literal_prefix,
    make_structured_think_processor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_tokenizer(vocab, think_start_tokens=(1,), think_end_tokens=(2,)):
    """Create a mock tokenizer with a known vocabulary and think tokens.

    vocab: dict of {token_id: decoded_text}
    """
    t = MagicMock()
    t.vocab_size = max(vocab.keys()) + 1 if vocab else 0

    def mock_decode(ids):
        return "".join(vocab.get(i, "") for i in ids)

    t.decode = mock_decode
    t.think_start_tokens = think_start_tokens
    t.think_end_tokens = think_end_tokens
    return t


def _make_tokens(*token_ids):
    """Make a tokens array for processor input."""
    return mx.array(token_ids)


# ---------------------------------------------------------------------------
# Token mask tests
# ---------------------------------------------------------------------------

class TestTokenMaskHelpers(unittest.TestCase):
    """Test the low-level token mask computation functions."""

    def test_matching_literal_exact(self):
        """Tokens matching the expected literal exactly."""
        token_texts = {
            0: "GO",
            1: "AL",
            2: ":",
            3: " ",
            4: "GOAL: ",
            5: "x",
        }
        # Expected: "GOAL: ", consumed 0 chars
        allowed = _tokens_matching_literal_prefix(token_texts, "GOAL: ", 0)
        # "GO" starts matching "GOAL: ", "AL" doesn't (it's not a prefix of remainder)
        # "GOAL: " starts matching the full thing
        # But: "AL" is NOT a prefix of "GOAL: ", so it shouldn't match
        self.assertIn(0, allowed)  # "GO" is prefix of "GOAL: "
        self.assertIn(4, allowed)  # "GOAL: " matches full
        self.assertNotIn(1, allowed)  # "AL" is not a prefix of "GOAL: "
        self.assertNotIn(5, allowed)  # "x" doesn't match

    def test_matching_literal_partial_consumed(self):
        """Tokens matching after some chars already consumed."""
        token_texts = {0: "AL", 1: ": ", 2: "AL: "}
        # Expected: "GOAL: ", consumed 2 chars -> remaining "AL: "
        allowed = _tokens_matching_literal_prefix(token_texts, "GOAL: ", 2)
        self.assertIn(0, allowed)   # "AL" is prefix of "AL: "
        # ": " does NOT start with "AL: " and "AL: " doesn't start with ": "
        self.assertNotIn(1, allowed)
        self.assertIn(2, allowed)   # "AL: " matches full remaining

    def test_matching_literal_fully_consumed(self):
        """No tokens needed when literal is fully consumed."""
        token_texts = {0: "x"}
        allowed = _tokens_matching_literal_prefix(token_texts, "GOAL: ", 6)
        self.assertEqual(len(allowed), 0)

    def test_matching_literal_overshoot(self):
        """Tokens that are longer than the remaining literal are still allowed."""
        token_texts = {0: "L: extra"}
        # remaining: "L: " -> "L: extra" starts with "L: "
        allowed = _tokens_matching_literal_prefix(token_texts, "GOAL: ", 3)
        self.assertIn(0, allowed)

    def test_non_newline_content_blocks_bare_newline(self):
        """Bare newline tokens should NOT be allowed (violates [^\\n]+)."""
        token_texts = {0: "\n", 1: "hello", 2: "hello\n", 3: "\n\n", 4: "\nworld"}
        allowed = _tokens_for_non_newline_content(token_texts)
        self.assertNotIn(0, allowed)   # bare \n - REJECTED
        self.assertNotIn(3, allowed)   # \n\n - REJECTED (only newlines)
        self.assertNotIn(4, allowed)   # \nworld - REJECTED (newline mid-token)
        self.assertIn(1, allowed)      # "hello" - no newline
        self.assertIn(2, allowed)      # "hello\n" - newline only at end

    def test_non_newline_content_allows_multiline_tokens(self):
        """Tokens with newlines only at the very end are allowed."""
        token_texts = {0: "text\n", 1: "\nmore", 2: "a\nb"}
        allowed = _tokens_for_non_newline_content(token_texts)
        self.assertIn(0, allowed)    # newline at end only
        self.assertNotIn(1, allowed)  # newline at start
        self.assertNotIn(2, allowed)  # newline in middle


# ---------------------------------------------------------------------------
# Processor state machine tests
# ---------------------------------------------------------------------------

class TestProcessorBasics(unittest.TestCase):
    """Test that the processor can be created and has correct shape."""

    def test_create_with_default_config(self):
        """Processor created with default GOAL/APPROACH/EDGE config."""
        vocab = {0: "", 1: "<think>", 2: "</think>", 3: "G", 4: "hello", 5: "\n"}
        t = _mock_tokenizer(vocab)
        proc = make_structured_think_processor(t)
        self.assertIsNotNone(proc)
        self.assertEqual(proc._config.fields, ("GOAL", "APPROACH", "EDGE"))

    def test_create_with_custom_fields(self):
        """Processor with 2-field config (Plan/Check for Gemma)."""
        vocab = {0: "", 1: "<|channel>thought", 2: "<channel|>", 3: "P", 4: "\n"}
        t = _mock_tokenizer(vocab, (1,), (2,))
        config = GrammarConfig(fields=("Plan", "Check"))
        proc = make_structured_think_processor(t, config)
        self.assertEqual(proc._config.fields, ("Plan", "Check"))

    def test_raises_on_missing_think_tokens(self):
        """Raises ValueError when tokenizer lacks think tokens."""
        vocab = {0: ""}
        t = _mock_tokenizer(vocab, think_start_tokens=(), think_end_tokens=())
        with self.assertRaises(ValueError):
            make_structured_think_processor(t)

    def test_processor_returns_logits_unchanged_outside_think(self):
        """Logits unchanged when outside think block."""
        vocab = {0: "", 1: "<think>", 2: "</think>", 3: "a"}
        t = _mock_tokenizer(vocab)
        proc = make_structured_think_processor(t)

        tokens = _make_tokens(3, 3, 3)  # No think tokens at all
        logits = mx.array([0.0, 0.0, 0.0, 0.0])  # 1D shape
        result = proc(tokens, logits)
        mx.eval(result)
        self.assertTrue(mx.all(result == logits))


class TestStateMachine3Fields(unittest.TestCase):
    """Test state transitions through a 3-field grammar."""

    def setUp(self):
        # Minimal vocabulary:
        #   1 = <think>, 2 = </think>
        #   10 = "GOAL: ", 11 = "APPROACH: ", 12 = "EDGE: "
        #   20 = "hello\n" (line content token), 21 = "hello" (no newline)
        #   30 = "</think>\n\n", 31 = "\n\n"
        self.vocab = {
            0: "",
            1: "<think>", 2: "</think>",
            10: "GOAL: ", 11: "APPROACH: ", 12: "EDGE: ",
            20: "hello\n", 21: "hello", 22: "world\n",
            30: "</think>\n\n", 31: "\n\n",
        }
        self.t = _mock_tokenizer(self.vocab)
        self.proc = make_structured_think_processor(self.t)
        self.state = self.proc._state
        self._token_history = []

    def _step(self, *token_ids):
        """Advance processor with given tokens, accumulating history across calls."""
        self._token_history.extend(token_ids)
        tokens = _make_tokens(*self._token_history)
        logits = mx.zeros((self.t.vocab_size,))
        self.proc(tokens, logits)
        mx.eval(logits)

    def test_initial_state_is_code(self):
        self.assertEqual(self.state["grammar_state"], ThinkState.CODE)
        self.assertFalse(self.state["inside_think"])

    def test_enters_think_and_starts_goal(self):
        self._step(1)  # <think>
        self.assertTrue(self.state["inside_think"])
        self.assertEqual(self.state["grammar_state"], ThinkState.WAITING_PREFIX)
        self.assertEqual(self.state["field_index"], 0)

    def test_goal_prefix_consumed(self):
        self._step(1, 10)  # <think>, "GOAL: "
        self.assertTrue(self.state["inside_think"])
        self.assertEqual(self.state["grammar_state"], ThinkState.IN_LINE)
        self.assertEqual(self.state["field_index"], 0)  # Still on first field

    def test_goal_line_completed(self):
        self._step(1, 10, 20)  # <think>, "GOAL: ", "hello\n"
        self.assertTrue(self.state["inside_think"])
        self.assertEqual(self.state["grammar_state"], ThinkState.WAITING_PREFIX)
        self.assertEqual(self.state["field_index"], 1)  # Moved to APPROACH

    def test_full_3_field_sequence(self):
        self._step(
            1,   # <think>
            10,  # GOAL:
            20,  # hello\n
            11,  # APPROACH:
            22,  # world\n
            12,  # EDGE:
            20,  # hello\n
        )
        self.assertEqual(self.state["grammar_state"], ThinkState.WAITING_THINK_END)
        self.assertEqual(self.state["field_index"], 3)  # All fields done

    def test_think_end_and_exit(self):
        self._step(
            1, 10, 20, 11, 22, 12, 20,  # Full 3 fields
            2,   # </think> — think-end boundary marker
            31,  # \n\n — completes grammar literal "</think>\n\n"
        )
        self.assertEqual(self.state["grammar_state"], ThinkState.CODE)
        self.assertFalse(self.state["inside_think"])

    def test_reentry_after_multiple_think_blocks(self):
        """After exiting think and re-entering, state resets correctly."""
        # First think block
        self._step(1, 10, 20, 11, 22, 12, 20, 2, 31)
        self.assertEqual(self.state["grammar_state"], ThinkState.CODE)

        # Some code tokens
        self._step(21, 21)

        # Second think block
        self._step(1)
        self.assertTrue(self.state["inside_think"])
        self.assertEqual(self.state["grammar_state"], ThinkState.WAITING_PREFIX)
        self.assertEqual(self.state["field_index"], 0)

    def test_line_content_without_newline_stays_in_line(self):
        """Tokens without newline keep the state in IN_LINE."""
        self._step(1, 10)  # Enter think, consume GOAL prefix -> IN_LINE
        self._step(21)  # "hello" - no newline
        self.assertEqual(self.state["grammar_state"], ThinkState.IN_LINE)
        self._step(21)  # Another non-newline
        self.assertEqual(self.state["grammar_state"], ThinkState.IN_LINE)


class TestStateMachine2Fields(unittest.TestCase):
    """Test state transitions through a 2-field grammar (Gemma)."""

    def setUp(self):
        self.vocab = {
            0: "",
            1: "<|channel>thought", 2: "<channel|>",
            10: "PLAN: ", 11: "CHECK: ",
            20: "ok\n",
            30: "<channel|>\n\n", 31: "\n\n",
        }
        self.t = _mock_tokenizer(self.vocab, (1,), (2,))
        config = GrammarConfig(fields=("PLAN", "CHECK"))
        self.proc = make_structured_think_processor(self.t, config)
        self.state = self.proc._state
        self._token_history = []

    def _step(self, *token_ids):
        self._token_history.extend(token_ids)
        tokens = _make_tokens(*self._token_history)
        logits = mx.zeros((self.t.vocab_size,))
        self.proc(tokens, logits)
        mx.eval(logits)

    def test_2_field_sequence_to_think_end(self):
        self._step(1, 10, 20, 11, 20)
        self.assertEqual(self.state["grammar_state"], ThinkState.WAITING_THINK_END)
        self.assertEqual(self.state["field_index"], 2)

    def test_2_field_exit(self):
        self._step(1, 10, 20, 11, 20, 2, 31)
        self.assertEqual(self.state["grammar_state"], ThinkState.CODE)
        self.assertFalse(self.state["inside_think"])


class TestStateMachine5Fields(unittest.TestCase):
    """Test state transitions through a 5-field grammar (LiveCodeBench)."""

    def setUp(self):
        self.vocab = {
            0: "",
            1: "<think>", 2: "</think>",
            10: "GOAL: ", 11: "STATE: ", 12: "ALGO: ",
            13: "EDGE: ", 14: "VERIFY: ",
            20: "line\n", 30: "</think>\n\n", 31: "\n\n",
        }
        self.t = _mock_tokenizer(self.vocab)
        config = GrammarConfig(fields=("GOAL", "STATE", "ALGO", "EDGE", "VERIFY"))
        self.proc = make_structured_think_processor(self.t, config)
        self.state = self.proc._state
        self._token_history = []

    def _step(self, *token_ids):
        self._token_history.extend(token_ids)
        tokens = _make_tokens(*self._token_history)
        logits = mx.zeros((self.t.vocab_size,))
        self.proc(tokens, logits)
        mx.eval(logits)

    def test_5_field_sequence(self):
        self._step(
            1,   # <think>
            10,  # GOAL:
            20,  # line
            11,  # STATE:
            20,  # line
            12,  # ALGO:
            20,  # line
            13,  # EDGE:
            20,  # line
            14,  # VERIFY:
            20,  # line
        )
        self.assertEqual(self.state["grammar_state"], ThinkState.WAITING_THINK_END)
        self.assertEqual(self.state["field_index"], 5)

    def test_5_field_exit(self):
        self._step(
            1, 10, 20, 11, 20, 12, 20, 13, 20, 14, 20,  # 5 fields
            2,   # </think> — think-end boundary marker
            31,  # \n\n — completes grammar literal "</think>\n\n"
        )
        self.assertEqual(self.state["grammar_state"], ThinkState.CODE)


# ---------------------------------------------------------------------------
# Mask application tests
# ---------------------------------------------------------------------------

class TestMaskApplication(unittest.TestCase):
    """Test that masks actually constrain logits during think blocks."""

    def setUp(self):
        # Vocabulary:
        #   1 = <think>, 2 = </think>
        #   10 = "GOAL: " (allowed in WAITING_PREFIX for field 0)
        #   11 = "APPROACH: "
        #   12 = "EDGE: "
        #   20 = "hello\n" (allowed in IN_LINE)
        #   21 = "hello" (allowed in IN_LINE)
        #   30 = "</think>\n\n"
        #   99 = "xyz" (never allowed in think block)
        self.vocab = {
            0: "", 1: "<think>", 2: "</think>",
            10: "GOAL: ", 11: "APPROACH: ", 12: "EDGE: ",
            20: "hello\n", 21: "hello",
            30: "</think>\n\n",
            99: "xyz",
        }
        self.t = _mock_tokenizer(self.vocab)
        self.proc = make_structured_think_processor(self.t)

    def _get_result_logits(self, *token_ids):
        """Return logits after processing given tokens. Non-allowed tokens = -inf."""
        tokens = _make_tokens(*token_ids)
        logits = mx.zeros((self.t.vocab_size,))
        result = self.proc(tokens, logits)
        mx.eval(result)
        return result

    def test_outside_think_all_logits_zero(self):
        """Outside think block, logits are unchanged (all zero)."""
        result = self._get_result_logits(99, 99, 99)
        self.assertTrue(mx.all(result == 0.0))

    def test_inside_think_masks_irrelevant_tokens(self):
        """Inside think block, irrelevant tokens are set to -inf."""
        # Step 1: enter think
        result = self._get_result_logits(1)  # <think>
        # Should be waiting for "GOAL: " prefix
        self.assertEqual(result[0].item(), float("-inf"))  # empty token blocked
        self.assertEqual(result[99].item(), float("-inf"))  # "xyz" blocked
        # Token 10 ("GOAL: ") should be allowed
        self.assertEqual(result[10].item(), 0.0)

    def test_code_state_restored_after_exit(self):
        """After exiting think, all logits are unconstrained again."""
        # Enter and exit think
        self._get_result_logits(
            1, 10, 20, 11, 20, 12, 20, 30  # Full 3-field + exit
        )
        # Now outside think
        result = self._get_result_logits(99)  # Just "xyz"
        self.assertTrue(mx.all(result == 0.0))


# ---------------------------------------------------------------------------
# Test tokenizer compatibility (no actual model needed)
# ---------------------------------------------------------------------------

class TestTokenizerDetection(unittest.TestCase):
    """Verify the processor works with different think-token formats."""

    def test_single_token_think(self):
        """Single-token think markers (common for Qwen)."""
        vocab = {0: "", 1: "<think>", 2: "</think>", 3: "a"}
        t = _mock_tokenizer(vocab, think_start_tokens=(1,), think_end_tokens=(2,))
        proc = make_structured_think_processor(t)
        self.assertIsNotNone(proc)

    def test_multi_token_think(self):
        """Multi-token think markers (fallback like longcat)."""
        vocab = {0: "", 1: "<long", 2: "cat_think", 3: ">", 4: "</longcat_think>"}
        t = _mock_tokenizer(
            vocab,
            think_start_tokens=(1, 2, 3),   # "<long" + "cat_think" + ">"
            think_end_tokens=(4,),           # single token for end
        )
        proc = make_structured_think_processor(t)
        self.assertIsNotNone(proc)

    def test_empty_think_tokens_raises(self):
        """Empty think token tuples should raise ValueError."""
        vocab = {0: ""}
        t = _mock_tokenizer(vocab, think_start_tokens=(), think_end_tokens=())
        with self.assertRaises(ValueError):
            make_structured_think_processor(t)


# ---------------------------------------------------------------------------
# GrammarConfig tests
# ---------------------------------------------------------------------------

class TestGrammarConfig(unittest.TestCase):
    """Test GrammarConfig construction and convenience methods."""

    def test_default_config(self):
        cfg = GrammarConfig()
        self.assertEqual(cfg.fields, ("GOAL", "APPROACH", "EDGE"))

    def test_from_field_names(self):
        cfg = GrammarConfig.from_field_names("plan", "check")
        self.assertEqual(cfg.fields, ("PLAN", "CHECK"))

    def test_rich_config(self):
        cfg = GrammarConfig.rich()
        self.assertEqual(cfg.fields, ("GOAL", "STATE", "ALGO", "EDGE", "VERIFY"))


if __name__ == "__main__":
    unittest.main()
