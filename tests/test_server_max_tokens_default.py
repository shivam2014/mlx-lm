"""Unit tests for max_tokens fallback and _validate whitelist logic.

These tests are self-contained — no mlx dependency. They verify the
parsing logic and validation that were changed in the response-truncation
fix: default max_tokens to -1 when client omits it, and whitelist -1
through _validate() despite min_val=0.
"""

import unittest


def parse_max_tokens(body: dict) -> int:
    """Extract max_tokens from a request body, replicating the server's
    fallback chain: max_completion_tokens > max_tokens > -1."""
    max_tokens = body.get("max_completion_tokens", None)
    if max_tokens is None:
        max_tokens = body.get("max_tokens", None)
        if max_tokens is None:
            max_tokens = -1
    return max_tokens


class TestMaxTokensFallback(unittest.TestCase):
    """Test the fallback chain: when client omits max_tokens, default to -1."""

    def test_no_max_tokens_defaults_to_neg1(self):
        self.assertEqual(parse_max_tokens({"model": "x"}), -1)

    def test_max_completion_tokens_taken(self):
        self.assertEqual(parse_max_tokens({"max_completion_tokens": 2048}), 2048)

    def test_max_tokens_taken_when_no_max_completion(self):
        self.assertEqual(parse_max_tokens({"max_tokens": 1024}), 1024)

    def test_max_completion_overrides_max_tokens(self):
        self.assertEqual(
            parse_max_tokens({"max_completion_tokens": 4096, "max_tokens": 512}),
            4096,
        )

    def test_zero_accepted(self):
        self.assertEqual(parse_max_tokens({"max_tokens": 0}), 0)

    def test_max_completion_zero_accepted(self):
        self.assertEqual(parse_max_tokens({"max_completion_tokens": 0}), 0)

    def test_empty_body_defaults_to_neg1(self):
        self.assertEqual(parse_max_tokens({}), -1)


def _validate(self, name, expected_type, min_val=None, max_val=None,
              optional=False, whitelist=None):
    """Replica of server.py APIHandler._validate for unit testing."""
    value = getattr(self, name)
    if optional and value is None:
        return
    if not isinstance(value, expected_type):
        try:
            allowed = tuple(et.__name__ for et in expected_type)
        except TypeError:
            allowed = expected_type.__name__
        raise ValueError(f"{name} must be of type {allowed}")
    if whitelist is not None and value in whitelist:
        return
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be at least {min_val}")
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be at most {max_val}")


class TestMaxTokensValidation(unittest.TestCase):
    """Test that -1 passes _validate() via whitelist, but other neg values don't."""

    @staticmethod
    def _make_handler(max_tokens):
        return type("MockHandler", (), {"max_tokens": max_tokens})()

    def test_neg1_passes_validation(self):
        h = self._make_handler(-1)
        _validate(h, "max_tokens", int, min_val=0, whitelist=[-1])

    def test_0_passes_validation(self):
        h = self._make_handler(0)
        _validate(h, "max_tokens", int, min_val=0, whitelist=[-1])

    def test_512_passes_validation(self):
        h = self._make_handler(512)
        _validate(h, "max_tokens", int, min_val=0, whitelist=[-1])

    def test_neg5_raises(self):
        h = self._make_handler(-5)
        with self.assertRaises(ValueError):
            _validate(h, "max_tokens", int, min_val=0, whitelist=[-1])

    def test_neg2_raises(self):
        h = self._make_handler(-2)
        with self.assertRaises(ValueError):
            _validate(h, "max_tokens", int, min_val=0, whitelist=[-1])

    def test_string_raises(self):
        h = self._make_handler("unlimited")
        with self.assertRaises(ValueError):
            _validate(h, "max_tokens", int, min_val=0, whitelist=[-1])


if __name__ == "__main__":
    unittest.main()
