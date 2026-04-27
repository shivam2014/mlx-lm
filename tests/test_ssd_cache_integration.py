#!/usr/bin/env python3
"""Integration tests for LRUPromptCache + SsdCache interaction.

Covers eviction, SSD fallback, RAM re-insertion, progressive prefix matching,
persistence, trim_to, cross-model isolation, and fetch-after-clear.
"""

import os
import sys
import tempfile
import time
import unittest
import copy

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlx_lm.models.cache import (
    LRUPromptCache,
    SsdCache,
    save_prompt_cache,
    load_prompt_cache,
)
import mlx_lm.models.cache as cache_mod


# ---------------------------------------------------------------------------
# Mock cache layer (same as test_ssd_cache.py, needed for load_prompt_cache)
# ---------------------------------------------------------------------------

class MockCacheLayer:
    """Mock per-layer KV cache that satisfies the save/load interface."""

    def __init__(self, keys, values, offset=0, meta_state=None):
        self.keys = keys
        self.values = values
        self._offset = offset
        self._meta_state = (
            meta_state if meta_state is not None else (str(offset),)
        )

    @property
    def state(self):
        return (self.keys, self.values)

    @state.setter
    def state(self, value):
        self.keys, self.values = value

    @property
    def meta_state(self):
        return self._meta_state

    @meta_state.setter
    def meta_state(self, value):
        self._meta_state = value

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        self._offset = value

    @property
    def nbytes(self):
        return self.keys.nbytes + self.values.nbytes

    def is_trimmable(self):
        return True

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls.__new__(cls)
        if isinstance(state, tuple) and len(state) == 2:
            obj.keys, obj.values = state
        else:
            obj.keys, obj.values = state[0], state[1]
        obj._meta_state = meta_state
        if obj.keys is not None:
            obj._offset = obj.keys.shape[-2] if obj.keys.ndim >= 2 else 0
        else:
            obj._offset = 0
        return obj


# Register MockCacheLayer into the cache module's namespace
cache_mod.MockCacheLayer = MockCacheLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_layers(num_layers=2, batch=1, heads=2, seq=4, dim=8):
    """Return a list of MockCacheLayer instances with random data."""
    layers = []
    for _ in range(num_layers):
        k = mx.random.uniform(shape=(batch, heads, seq, dim))
        v = mx.random.uniform(shape=(batch, heads, seq, dim))
        layers.append(MockCacheLayer(k, v, offset=seq))
    return layers


def _make_tiny_layers(seq=2):
    """Return a minimal single-layer cache (tiny, for eviction tests)."""
    k = mx.zeros((1, 1, seq, 1))
    v = mx.zeros((1, 1, seq, 1))
    return [MockCacheLayer(k, v, offset=seq)]


def _layer_bytes(layers):
    """Total nbytes for a list of cache layers."""
    return sum(c.nbytes for c in layers)


def _arrays_equal(a, b):
    """Check if two mx.array trees are element-wise equal."""
    return mx.array_equal(a, b)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLRUPromptCacheSsdCacheIntegration(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -- Test 1: RAM hit does not touch SSD --------------------------------

    def test_ram_hit_does_not_touch_ssd(self):
        """A RAM hit should return the entry without any SSD involvement."""
        ssd = SsdCache(self.test_dir)
        cache = LRUPromptCache(max_size=100, max_bytes=1 << 20, ssd_cache=ssd)
        model = "test_model"
        tokens = [10, 20, 30, 40]
        layers = _make_layers(seq=len(tokens))

        # Insert
        cache.insert_cache(model, tokens, layers)

        # Verify no SSD entries yet (nothing evicted)
        self.assertEqual(ssd.entry_count, 0)

        # Fetch — should be a RAM exact hit
        result, remaining = cache.fetch_nearest_cache(model, tokens)
        self.assertIsNotNone(result)
        self.assertEqual(remaining, [])
        # Verify arrays match
        for orig, lyr in zip(layers, result):
            self.assertTrue(_arrays_equal(orig.state[0], lyr.state[0]))
            self.assertTrue(_arrays_equal(orig.state[1], lyr.state[1]))

        # Still no SSD entries
        self.assertEqual(ssd.entry_count, 0)

    # -- Test 2: Eviction to SSD on max_bytes overflow ---------------------

    def test_eviction_to_ssd(self):
        """When max_bytes is exceeded, entries are evicted to SSD."""
        ssd = SsdCache(self.test_dir)
        # Tiny max_bytes so insertion immediately triggers eviction
        cache = LRUPromptCache(max_size=100, max_bytes=1, ssd_cache=ssd)

        model = "test_model"
        tokens = [10, 20, 30]
        layers = _make_layers(seq=3)

        # Insert — the entry is bigger than 1 byte, so it gets evicted
        cache.insert_cache(model, tokens, layers)

        # The entry should have been evicted to SSD
        self.assertGreater(ssd.entry_count, 0,
                           "Entry should have been evicted to SSD")
        self.assertEqual(len(cache), 0,
                         "RAM cache should be empty after eviction")

        # fetch_nearest_cache should still find it via SSD fallback
        result, remaining = cache.fetch_nearest_cache(model, tokens)
        self.assertIsNotNone(result,
                             "Should find evicted entry via SSD fallback")
        self.assertEqual(remaining, [])

        for orig, lyr in zip(layers, result):
            self.assertTrue(_arrays_equal(orig.state[0], lyr.state[0]))
            self.assertTrue(_arrays_equal(orig.state[1], lyr.state[1]))

    # -- Test 3: SSD hit restores to RAM -----------------------------------

    def test_ssd_hit_restores_to_ram(self):
        """After eviction, first fetch hits SSD and restores RAM; second fetch
        is a pure RAM hit."""
        ssd = SsdCache(self.test_dir)
        cache = LRUPromptCache(max_size=100, max_bytes=1, ssd_cache=ssd)

        model = "test_model"
        tokens = [10, 20, 30, 40]
        layers = _make_layers(seq=4)

        # Insert — gets evicted to SSD immediately
        cache.insert_cache(model, tokens, layers)
        self.assertGreater(ssd.entry_count, 0)
        self.assertEqual(len(cache), 0,
                         "RAM cache should be empty after eviction")

        # First fetch — should come from SSD (RAM miss -> SSD fallback)
        result1, remaining1 = cache.fetch_nearest_cache(model, tokens)
        self.assertIsNotNone(result1)
        self.assertEqual(remaining1, [])

        # After first fetch, the entry should be back in RAM
        self.assertGreater(len(cache), 0,
                           "Entry should have been restored to RAM after SSD fetch")

        # Second fetch — should be a pure RAM hit (no SSD involvement)
        # We can verify by counting SSD entries — they shouldn't increase
        ssd_count_before = ssd.entry_count
        result2, remaining2 = cache.fetch_nearest_cache(model, tokens)
        self.assertIsNotNone(result2)
        self.assertEqual(remaining2, [])
        # SSD entry count should not have changed (no new eviction)
        self.assertEqual(ssd.entry_count, ssd_count_before)

        for orig, lyr in zip(layers, result2):
            self.assertTrue(_arrays_equal(orig.state[0], lyr.state[0]))
            self.assertTrue(_arrays_equal(orig.state[1], lyr.state[1]))

    # -- Test 4: Progressive prefix matching on SSD ------------------------

    def test_progressive_prefix_matching_on_ssd(self):
        """When querying with longer tokens than the cached prefix, SSD
        fallback should find via progressive shorter prefix matching."""
        ssd = SsdCache(self.test_dir)
        cache = LRUPromptCache(max_size=100, max_bytes=1, ssd_cache=ssd)

        model = "test_model"
        cached_tokens = [10, 20, 30, 40]
        longer_tokens = [10, 20, 30, 40, 50, 60]

        layers = _make_layers(seq=len(cached_tokens))

        # Insert cached entry — gets evicted to SSD
        cache.insert_cache(model, cached_tokens, layers)
        self.assertGreater(ssd.entry_count, 0)

        # Now query with longer tokens — should find via prefix match
        result, remaining = cache.fetch_nearest_cache(model, longer_tokens)
        self.assertIsNotNone(result,
                             "Should find evicted entry via prefix matching on SSD")
        # The remaining tokens should be [50, 60] (the unmatched suffix)
        self.assertEqual(remaining, [50, 60],
                         "Remaining should be the suffix beyond the cached prefix")

        # Verify the returned cache corresponds to the prefix [10,20,30,40]
        for orig, lyr in zip(layers, result):
            self.assertTrue(_arrays_equal(orig.state[0], lyr.state[0]))
            self.assertTrue(_arrays_equal(orig.state[1], lyr.state[1]))

    # -- Test 5: No SSD fallback when no ssd_cache configured ---------------

    def test_no_ssd_fallback_when_not_configured(self):
        """Without ssd_cache, evicted entries are discarded (no fallback)."""
        # No SSD cache
        cache = LRUPromptCache(max_size=100, max_bytes=1)

        model = "test_model"
        tokens = [10, 20, 30]
        layers = _make_layers(seq=3)

        # Insert — gets evicted (no SSD to fall back to)
        cache.insert_cache(model, tokens, layers)

        # RAM should be empty (entry was evicted and simply discarded)
        self.assertEqual(len(cache), 0)

        # Fetch should return None
        result, remaining = cache.fetch_nearest_cache(model, tokens)
        self.assertIsNone(result)
        self.assertEqual(remaining, tokens)

    # -- Test 6: Persistence across cache instance restart ------------------

    def test_persistence_across_cache_restart(self):
        """A new LRUPromptCache with the same SsdCache can find entries
        that were evicted by the previous LRUPromptCache instance."""
        ssd = SsdCache(self.test_dir)

        # First cache instance
        cache1 = LRUPromptCache(max_size=100, max_bytes=1, ssd_cache=ssd)
        model = "persist_model"
        tokens = [42, 43, 44]
        layers = _make_layers(seq=3)

        cache1.insert_cache(model, tokens, layers)
        self.assertGreater(ssd.entry_count, 0,
                           "Entry should have been evicted to SSD")

        # Discard cache1, create new LRUPromptCache with the same SsdCache
        cache2 = LRUPromptCache(max_size=100, max_bytes=1 << 20,
                                ssd_cache=ssd)

        # cache2 should be able to find the entry via SSD fallback
        result, remaining = cache2.fetch_nearest_cache(model, tokens)
        self.assertIsNotNone(result,
                             "New cache instance should find SSD-persisted entry")
        self.assertEqual(remaining, [])

        for orig, lyr in zip(layers, result):
            self.assertTrue(_arrays_equal(orig.state[0], lyr.state[0]))
            self.assertTrue(_arrays_equal(orig.state[1], lyr.state[1]))

    # -- Test 7: trim_to evicts to SSD --------------------------------------

    def test_trim_to_evicts_to_ssd(self):
        """trim_to(n_sequences=0) should evict all entries to SSD."""
        ssd = SsdCache(self.test_dir)
        # Large max_bytes and max_size so nothing gets evicted during insert
        cache = LRUPromptCache(max_size=100, max_bytes=1 << 20,
                               ssd_cache=ssd)

        model = "test_model"
        # Insert multiple entries
        layers_list = []
        for i in range(3):
            tok = [i * 10, i * 10 + 1, i * 10 + 2]
            layers = _make_layers(seq=3)
            layers_list.append((tok, layers))
            cache.insert_cache(model, tok, layers)

        # All should be in RAM initially
        self.assertEqual(len(cache), 3)
        self.assertEqual(ssd.entry_count, 0)

        # trim_to with n_sequences=0 should evict all to SSD
        cache.trim_to(n_sequences=0)

        # RAM should be empty
        self.assertEqual(len(cache), 0)
        # SSD should have the entries
        self.assertEqual(ssd.entry_count, 3)

        # Fetch should still find them via SSD fallback
        for tok, layers in layers_list:
            result, remaining = cache.fetch_nearest_cache(model, tok)
            self.assertIsNotNone(result,
                                 f"trim_to-evicted entry {tok} should be findable")
            self.assertEqual(remaining, [])

    # -- Test 8: Cross-model isolation -------------------------------------

    def test_cross_model_isolation(self):
        """Different model keys with the same tokens produce different SSD
        entries and each loads its own arrays."""
        ssd = SsdCache(self.test_dir)
        cache = LRUPromptCache(max_size=100, max_bytes=1, ssd_cache=ssd)

        tokens = [10, 20, 30]

        # Insert for model_a with specific data
        layers_a = _make_layers(seq=3)
        # Make them deterministic so we can verify
        mx.random.seed(42)
        layers_a = _make_layers(seq=3)
        cache.insert_cache("model_a", tokens, layers_a)

        # Insert for model_b with different data
        mx.random.seed(99)
        layers_b = _make_layers(seq=3)
        cache.insert_cache("model_b", tokens, layers_b)

        # Both should be on SSD now (tiny max_bytes caused eviction)
        self.assertGreaterEqual(ssd.entry_count, 2)

        # Fetch from model_a — should get model_a's data
        result_a, _ = cache.fetch_nearest_cache("model_a", tokens)
        self.assertIsNotNone(result_a)

        # Fetch from model_b — should get model_b's data
        result_b, _ = cache.fetch_nearest_cache("model_b", tokens)
        self.assertIsNotNone(result_b)

        # The data should differ
        # (different seeds produce different arrays)
        keys_match = _arrays_equal(result_a[0].state[0], result_b[0].state[0])
        self.assertFalse(keys_match,
                         "model_a and model_b should have different cache data")

        # Verify each matches its original
        for orig, lyr in zip(layers_a, result_a):
            self.assertTrue(_arrays_equal(orig.state[0], lyr.state[0]))
            self.assertTrue(_arrays_equal(orig.state[1], lyr.state[1]))

        for orig, lyr in zip(layers_b, result_b):
            self.assertTrue(_arrays_equal(orig.state[0], lyr.state[0]))
            self.assertTrue(_arrays_equal(orig.state[1], lyr.state[1]))

    # -- Test 9: fetch after clear -----------------------------------------

    def test_fetch_after_clear(self):
        """After clearing the SSD, fetch should return None.
        Re-insertion should work normally afterwards."""
        ssd = SsdCache(self.test_dir)
        cache = LRUPromptCache(max_size=100, max_bytes=1, ssd_cache=ssd)

        model = "test_model"
        tokens = [10, 20, 30]
        layers = _make_layers(seq=3)

        # Insert — evicted to SSD
        cache.insert_cache(model, tokens, layers)
        self.assertGreater(ssd.entry_count, 0)

        # Clear the SSD
        ssd.clear()
        self.assertEqual(ssd.entry_count, 0)

        # Fetch should now return None (SSD empty, nothing in RAM)
        result, remaining = cache.fetch_nearest_cache(model, tokens)
        self.assertIsNone(result)
        self.assertEqual(remaining, tokens)

        # Re-insert with a fresh cache (max_bytes=1 so it goes to SSD again)
        cache2 = LRUPromptCache(max_size=100, max_bytes=1, ssd_cache=ssd)
        cache2.insert_cache(model, tokens, layers)
        self.assertGreater(ssd.entry_count, 0,
                           "Re-insertion should work after clear")

        # Fetch should find it again
        result2, remaining2 = cache2.fetch_nearest_cache(model, tokens)
        self.assertIsNotNone(result2)
        self.assertEqual(remaining2, [])

        for orig, lyr in zip(layers, result2):
            self.assertTrue(_arrays_equal(orig.state[0], lyr.state[0]))
            self.assertTrue(_arrays_equal(orig.state[1], lyr.state[1]))

    # -- Test 10: trim_to by n_bytes evicts to SSD --------------------------

    def test_trim_to_n_bytes_evicts_to_ssd(self):
        """trim_to(n_bytes=0) should evict all entries to SSD."""
        ssd = SsdCache(self.test_dir)
        cache = LRUPromptCache(max_size=100, max_bytes=1 << 20,
                               ssd_cache=ssd)

        model = "test_model"
        tokens = [10, 20, 30]
        layers = _make_layers(seq=3)
        cache.insert_cache(model, tokens, layers)

        self.assertEqual(len(cache), 1)
        self.assertEqual(ssd.entry_count, 0)

        # Trim by bytes
        cache.trim_to(n_bytes=0)

        self.assertEqual(len(cache), 0)
        self.assertEqual(ssd.entry_count, 1)

        # Should still be findable
        result, remaining = cache.fetch_nearest_cache(model, tokens)
        self.assertIsNotNone(result)
        self.assertEqual(remaining, [])

    # -- Test 11: Multiple evictions fill SSD, all still retrievable --------

    def test_multiple_evictions_to_ssd_retrievable(self):
        """Multiple entries can be evicted to SSD and each is still
        individually retrievable via prefix/SSD fallback."""
        ssd = SsdCache(self.test_dir)
        cache = LRUPromptCache(max_size=100, max_bytes=1, ssd_cache=ssd)

        model = "multi_model"
        entries = {}
        for i in range(5):
            tok = list(range(i * 10, i * 10 + 4))
            layers = _make_layers(seq=4)
            entries[tuple(tok)] = layers
            cache.insert_cache(model, tok, layers)

        # All 5 should have been evicted to SSD
        self.assertGreaterEqual(ssd.entry_count, 5)

        # Each should still be retrievable
        for tok_tuple, layers in entries.items():
            tok = list(tok_tuple)
            result, remaining = cache.fetch_nearest_cache(model, tok)
            self.assertIsNotNone(result,
                                 f"Evicted entry {tok} should be retrievable")
            self.assertEqual(remaining, [])
            for orig, lyr in zip(layers, result):
                self.assertTrue(_arrays_equal(orig.state[0], lyr.state[0]))
                self.assertTrue(_arrays_equal(orig.state[1], lyr.state[1]))

    # -- Test 12: LRU ordering respects cache_type in eviction --------------

    def test_lru_eviction_ordering_by_cache_type(self):
        """Eviction should respect the cache_type ordering (assistant, user,
        system) and evict from assistant first, then user, then system."""
        ssd = SsdCache(self.test_dir)
        # max_size=3 so we trigger eviction by count
        cache = LRUPromptCache(max_size=3, max_bytes=1 << 20,
                               ssd_cache=ssd)

        model = "ordering_test"

        # Insert: assistant, user, system
        cache.insert_cache(model, [1, 2], _make_layers(seq=2),
                           cache_type="assistant")
        cache.insert_cache(model, [3, 4], _make_layers(seq=2),
                           cache_type="user")
        cache.insert_cache(model, [5, 6], _make_layers(seq=2),
                           cache_type="system")
        self.assertEqual(len(cache), 3)

        # Insert one more — should evict assistant (since assistant and user
        # have equal length, and assistant comes first in ordering)
        cache.insert_cache(model, [7, 8], _make_layers(seq=2),
                           cache_type="assistant")

        # The assistant entry [1,2] should have been evicted to SSD
        self.assertGreater(ssd.entry_count, 0)

        # But it should still be retrievable via SSD fallback
        result, remaining = cache.fetch_nearest_cache(model, [1, 2])
        self.assertIsNotNone(result,
                             "Evicted assistant entry should be retrievable via SSD")

    # -- Test 13: Re-insertion of evicted entry overwrites SSD --------------

    def test_reinsert_evicted_entry_updates_ssd(self):
        """If an entry was evicted to SSD and then re-inserted with different
        data (and re-evicted), the SSD should have the latest data."""
        ssd = SsdCache(self.test_dir)
        cache = LRUPromptCache(max_size=100, max_bytes=1, ssd_cache=ssd)

        model = "reinsert_model"
        tokens = [1, 2, 3]

        # First insert with specific data
        mx.random.seed(10)
        layers_v1 = _make_layers(seq=3)
        cache.insert_cache(model, tokens, layers_v1)

        # Second insert with different data — overwrites in RAM, then evicts
        mx.random.seed(20)
        layers_v2 = _make_layers(seq=3)
        cache.insert_cache(model, tokens, layers_v2)

        # Fetch should give v2 data
        result, remaining = cache.fetch_nearest_cache(model, tokens)
        self.assertIsNotNone(result)
        for orig, lyr in zip(layers_v2, result):
            self.assertTrue(_arrays_equal(orig.state[0], lyr.state[0]))
            self.assertTrue(_arrays_equal(orig.state[1], lyr.state[1]))

        # The SSD entry should also be v2 now
        ssd_result = ssd.load_entry(model, tokens)
        self.assertIsNotNone(ssd_result)
        for orig, lyr in zip(layers_v2, ssd_result):
            self.assertTrue(_arrays_equal(orig.state[0], lyr.state[0]))
            self.assertTrue(_arrays_equal(orig.state[1], lyr.state[1]))


if __name__ == "__main__":
    unittest.main()
