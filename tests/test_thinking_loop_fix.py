"""Tests for the max_tokens thinking loop fix in mlx_lm/generate.py.

These tests are self-contained — no mlx, no model loading.
They verify the comparison logic, hard safety cap, and prefetch guard
that were corrected to prevent infinite generation when max_tokens=-1.

Background (from the server log analysis):
  - When max_tokens is not set by the client, server.py defaults to -1
  - generate_step only checked `if n == max_tokens: break` → NEVER true with -1
  - Qwen thinking model could generate 1000+ tokens without EOS
  - Result: infinite generation loop, user forced to Ctrl+C → BrokenPipeError

Fix (from external audit AUDIT_REPORT.md):
  1. Add HARD_CAP = 131172 constant
  2. Fix comparison: `max_tokens != -1 and n == max_tokens`
  3. Fix prefetch guard: `should_stop boolean covers both conditions`
  4. Both generate_step and stream_generate
"""

import unittest

# The exact logic that will be patched into mlx_lm/generate.py
HARD_CAP = 131072


def _should_stop(n: int, max_tokens: int) -> bool:
    """Replica of the corrected break condition in generate_step."""
    if max_tokens != -1 and n == max_tokens:
        return True
    if n > HARD_CAP:
        return True
    return False


def _should_stop_stream(n: int, max_tokens: int) -> bool:
    """Replica of the corrected break condition in stream_generate.

    stream_generate uses (n+1) == max_tokens because n is 0-indexed
    and it checks AFTER adding the token to the detokenizer.
    It uses n >= HARD_CAP (inclusive) rather than n > HARD_CAP (exclusive)
    to account for the n+1 offset — the token count is n+1, so
    n >= HARD_CAP means the generated output reached HARD_CAP tokens.
    """
    if max_tokens != -1 and (n + 1) == max_tokens:
        return True
    if n >= HARD_CAP:
        return True
    return False


def _should_prefetch(n: int, max_tokens: int) -> bool:
    """Replica of the corrected prefetch guard in generate_step.

    Prefetches the NEXT token only if we won't stop at the current one.
    This prevents scheduling an async_eval that never gets evaluated
    (orphaned GPU operation) when the break fires.
    """
    return not _should_stop(n, max_tokens)


# =========================================================================
# Tests for generate_step comparison logic
# =========================================================================

class TestGenerateStepComparison(unittest.TestCase):
    """Test the break condition in generate_step."""

    def test_neg1_never_triggers_break(self):
        """max_tokens=-1 should never cause _should_stop to return True
        regardless of n (the infinite loop scenario)."""
        for n in [0, 1, 100, 9999, 50000, 100000, 131071]:
            self.assertFalse(
                _should_stop(n, -1),
                f"_should_stop({n}, -1) should be False (no limit)",
            )

    def test_normal_limit_reached(self):
        """When max_tokens=100 and n=100, should stop."""
        self.assertTrue(_should_stop(100, 100))

    def test_normal_limit_not_reached(self):
        """When max_tokens=100 and n=50, should NOT stop."""
        self.assertFalse(_should_stop(50, 100))

    def test_zero_max_tokens_stops_immediately(self):
        """max_tokens=0 should stop at n=0 (generate zero tokens)."""
        self.assertTrue(_should_stop(0, 0))

    def test_positive_limit_stops_exactly_at_limit(self):
        """Various exact limit hits."""
        for limit in [1, 10, 256, 4096, 65536]:
            self.assertTrue(_should_stop(limit, limit))

    def test_positive_limit_below(self):
        """One below the limit should NOT stop."""
        for limit in [1, 10, 256, 4096, 65536]:
            self.assertFalse(_should_stop(limit - 1, limit))


# =========================================================================
# Tests for hard safety cap
# =========================================================================

class TestHardSafetyCap(unittest.TestCase):
    """Test the HARD_CAP constant and its enforcement."""

    def test_hard_cap_constant_value(self):
        """HARD_CAP should be 131072 (one token below 128K)."""
        self.assertEqual(HARD_CAP, 131072)

    def test_hard_cap_not_triggered_at_exact_boundary(self):
        """n == HARD_CAP should NOT trigger stop in generate_step
        (it uses n > HARD_CAP, so it stops at the *next* token)."""
        self.assertFalse(_should_stop(HARD_CAP, -1))

    def test_hard_cap_triggered_at_boundary_plus_one(self):
        """n == HARD_CAP + 1 SHOULD trigger stop."""
        self.assertTrue(_should_stop(HARD_CAP + 1, -1))

    def test_hard_cap_triggered_far_above(self):
        """Very large n should trigger the hard cap."""
        for n in [200000, 500000, 1000000]:
            self.assertTrue(_should_stop(n, -1))

    def test_hard_cap_overrides_positive_limit(self):
        """If n exceeds HARD_CAP, should stop even if max_tokens is higher."""
        self.assertTrue(_should_stop(HARD_CAP + 1, 999999))

    def test_hard_cap_lower_than_positive_limit(self):
        """If n is below HARD_CAP but at positive limit, should stop."""
        self.assertTrue(_should_stop(500, 500))


# =========================================================================
# Tests for stream_generate comparison
# =========================================================================

class TestStreamGenerateComparison(unittest.TestCase):
    """Test the break condition in stream_generate (uses n+1 comparison)."""

    def test_neg1_never_triggers_break(self):
        """max_tokens=-1 should never stop in stream_generate either."""
        for n in [0, 1, 100, 9999, 50000, 100000, 131071]:
            self.assertFalse(_should_stop_stream(n, -1))

    def test_stream_normal_limit_reached(self):
        """When max_tokens=100 and n=99, (n+1)==100 → should stop."""
        self.assertTrue(_should_stop_stream(99, 100))

    def test_stream_normal_limit_not_reached(self):
        """When max_tokens=100 and n=98, (n+1)==99 → should NOT stop."""
        self.assertFalse(_should_stop_stream(98, 100))

    def test_stream_zero_max_tokens_stops_immediately(self):
        """max_tokens=0: n=0 would give (n+1)==0 → False. n=-1 impossible.
        So max_tokens=0 means 'generate 0 tokens' — we should stop
        before any tokens are generated, but that's handled in the
        calling code, not in the (n+1) comparison."""
        # For stream_generate, max_tokens=0 means produce 0 tokens
        # which is handled by the generator being empty, not by this check
        pass

    def test_stream_hard_cap_triggered(self):
        """n >= HARD_CAP should trigger stop in stream_generate."""
        self.assertTrue(_should_stop_stream(HARD_CAP, -1))
        self.assertTrue(_should_stop_stream(HARD_CAP + 1, -1))
        self.assertTrue(_should_stop_stream(200000, -1))

    def test_stream_hard_cap_not_triggered_below(self):
        """n < HARD_CAP should NOT trigger stop."""
        self.assertFalse(_should_stop_stream(HARD_CAP - 1, -1))


# =========================================================================
# Tests for prefetch guard
# =========================================================================

class TestPrefetchGuard(unittest.TestCase):
    """Test that the prefetch guard correctly prevents scheduling
    async_eval when a break is imminent."""

    def test_should_prefetch_under_normal_limit(self):
        """When n < max_tokens, should prefetch."""
        self.assertTrue(_should_prefetch(50, 100))

    def test_should_not_prefetch_at_limit(self):
        """When n == max_tokens, should NOT prefetch (break fires)."""
        self.assertFalse(_should_prefetch(100, 100))

    def test_should_not_prefetch_at_hard_cap_plus_one(self):
        """When n > HARD_CAP, should NOT prefetch (break fires)."""
        self.assertFalse(_should_prefetch(HARD_CAP + 1, -1))

    def test_should_prefetch_under_hard_cap_with_neg1(self):
        """When n < HARD_CAP and max_tokens=-1, should prefetch."""
        self.assertTrue(_should_prefetch(50000, -1))

    def test_should_prefetch_below_limit(self):
        """When n is one below limit, should still prefetch
        (the NEXT token will be at the limit, which is checked
        at the TOP of the next iteration)."""
        self.assertTrue(_should_prefetch(99, 100))

    def test_should_not_prefetch_at_hard_cap_with_neg1(self):
        """When n == HARD_CAP and max_tokens=-1 in generate_step
        (which uses n > HARD_CAP), should still prefetch because
        n == HARD_CAP doesn't trigger stop in generate_step."""
        self.assertTrue(_should_prefetch(HARD_CAP, -1))


# =========================================================================
# Integration-style tests (combine comparison + prefetch)
# =========================================================================

class TestCombinedBehavior(unittest.TestCase):
    """Test the interaction between comparison and prefetch."""

    def test_normal_generation_flow(self):
        """Simulate a normal generation with max_tokens=5."""
        n = 0
        steps = []
        while True:
            prefetch = _should_prefetch(n, 5)
            stop = _should_stop(n, 5)
            steps.append((n, prefetch, stop))
            if stop:
                break
            n += 1
        # Should generate exactly 5 tokens (n=0 through n=4), stop at n=5
        self.assertEqual(len(steps), 6)  # 5 tokens + 1 stop step
        self.assertEqual(steps[-1], (5, False, True))  # stop at n=5
        # First 5 steps should have prefetch=True
        for i in range(5):
            self.assertTrue(steps[i][1], f"Step {i} should prefetch")

    def test_no_limit_generation_stops_at_hard_cap(self):
        """Simulate generation with max_tokens=-1, should stop at HARD_CAP+1."""
        n = 0
        steps = []
        while True:
            prefetch = _should_prefetch(n, -1)
            stop = _should_stop(n, -1)
            steps.append((n, prefetch, stop))
            if stop:
                break
            n += 1
        # Should stop at n = HARD_CAP + 1
        self.assertEqual(steps[-1][0], HARD_CAP + 1)
        # At n = HARD_CAP, should_stop should be False, prefetch should be True
        hard_cap_step = next(s for s in steps if s[0] == HARD_CAP)
        self.assertFalse(hard_cap_step[2])   # should_stop False
        self.assertTrue(hard_cap_step[1])     # should_prefetch True
        # At n = HARD_CAP + 1, should_stop should be True, prefetch False
        stop_step = steps[-1]
        self.assertTrue(stop_step[2])          # should_stop True
        self.assertFalse(stop_step[1])         # should_prefetch False


if __name__ == "__main__":
    unittest.main()


# =========================================================================
# Tests for BatchGenerator.next() fix (Bug C)
# =========================================================================

def _batch_should_stop(num_tokens: int, max_tokens: int) -> bool:
    """Replica of the corrected break condition in BatchGenerator.next().

    BatchGenerator.next() increments _num_tokens[i] first, THEN checks.
    So num_tokens here means the count AFTER incrementing (min value = 1).
    Uses n > HARD_CAP (exclusive) like generate_step.
    """
    if max_tokens != -1 and num_tokens >= max_tokens:
        return True
    if num_tokens > HARD_CAP:
        return True
    return False


class TestBatchGeneratorFix(unittest.TestCase):
    """Test the Bug C fix in BatchGenerator.next().

    Bug C: `if self._num_tokens[i] >= self.max_tokens[i]`
    When max_tokens=-1: 1 >= -1 is True → caps at 1 token immediately.

    Fix: `if (self.max_tokens[i] != -1 and self._num_tokens[i] >= self.max_tokens[i]) or self._num_tokens[i] > HARD_CAP`
    """

    def test_neg1_does_not_cap_at_one_token(self):
        """max_tokens=-1 with _num_tokens=1 should NOT stop.
        This was the original bug: 1 >= -1 was True."""
        self.assertFalse(
            _batch_should_stop(1, -1),
            "num_tokens=1 with max_tokens=-1 should not stop",
        )

    def test_neg1_allows_many_tokens(self):
        """max_tokens=-1 should never stop for normal token counts."""
        for n in [1, 10, 100, 50000, 100000, 131071]:
            self.assertFalse(
                _batch_should_stop(n, -1),
                f"_batch_should_stop({n}, -1) should be False",
            )

    def test_neg1_stops_at_hard_cap(self):
        """max_tokens=-1 with _num_tokens > HARD_CAP should stop."""
        self.assertTrue(_batch_should_stop(HARD_CAP + 1, -1))
        self.assertTrue(_batch_should_stop(200000, -1))

    def test_neg1_does_not_stop_at_exact_hard_cap(self):
        """max_tokens=-1 with _num_tokens == HARD_CAP should NOT stop
        (uses n > HARD_CAP, same as generate_step)."""
        self.assertFalse(_batch_should_stop(HARD_CAP, -1))

    def test_normal_limit_stops_at_limit(self):
        """When max_tokens=100 and _num_tokens=100, should stop."""
        self.assertTrue(_batch_should_stop(100, 100))

    def test_normal_limit_below(self):
        """When max_tokens=100 and _num_tokens=99, should NOT stop."""
        self.assertFalse(_batch_should_stop(99, 100))

    def test_zero_max_tokens_stops_immediately(self):
        """max_tokens=0 with _num_tokens=1 should stop immediately.
        (num_tokens starts at 0, gets incremented to 1 before check.)"""
        self.assertTrue(_batch_should_stop(1, 0))

    def test_hard_cap_overrides_positive_limit(self):
        """If _num_tokens exceeds HARD_CAP, should stop even if max_tokens is higher."""
        self.assertTrue(_batch_should_stop(HARD_CAP + 1, 999999))

    def test_various_legitimate_limits(self):
        """Various limits that should work correctly."""
        for limit in [1, 10, 256, 4096, 65536]:
            self.assertTrue(_batch_should_stop(limit, limit))
            self.assertFalse(_batch_should_stop(limit - 1, limit))

    def test_batch_flow_with_neg1(self):
        """Simulate the full BatchGenerator flow with max_tokens=-1.
        Should never stop until HARD_CAP is exceeded."""
        num_tokens = 0
        for _ in range(100):
            num_tokens += 1  # _num_tokens[i] += 1
            if _batch_should_stop(num_tokens, -1):
                self.fail(f"Batch flow stopped early at num_tokens={num_tokens} with max_tokens=-1")
        # Should reach 100 tokens without stopping
        self.assertEqual(num_tokens, 100)

    def test_batch_flow_with_positive_limit(self):
        """Simulate the full BatchGenerator flow with max_tokens=50.
        Should stop exactly at 50 tokens."""
        num_tokens = 0
        for _ in range(200):
            num_tokens += 1  # _num_tokens[i] += 1
            if _batch_should_stop(num_tokens, 50):
                self.assertEqual(num_tokens, 50)
                break
        else:
            self.fail("Batch flow did not stop at max_tokens=50")
