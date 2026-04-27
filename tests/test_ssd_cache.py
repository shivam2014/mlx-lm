#!/usr/bin/env python3
"""Unit tests for the SsdCache class.

Tests save/load cycle, eviction, clear, and persistence across instances.
"""

import os
import sys
import tempfile
import time
import unittest

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlx_lm.models.cache import SsdCache, save_prompt_cache, load_prompt_cache
import mlx_lm.models.cache as cache_mod


# ---------------------------------------------------------------------------
# Mock cache layer compatible with save_prompt_cache / load_prompt_cache
# ---------------------------------------------------------------------------

class MockCacheLayer:
    """Mock per-layer KV cache that satisfies the save/load interface.

    ``meta_state`` must be a string or tuple of strings because
    ``save_prompt_cache`` stores metadata via ``mx.save_safetensors``
    which only accepts string-typed values.

    Each instance has:
      .state      -> (keys, values)      tuple of mx.arrays
      .meta_state -> tuple[str, ...]      strings compatible with safetensors
      .offset     -> int                  sequence position
      .nbytes     -> int                  approximate memory footprint
    """

    def __init__(self, keys, values, offset=0, meta_state=None):
        self.keys = keys
        self.values = values
        self._offset = offset
        # meta_state must be string-typed for safetensors metadata
        self._meta_state = (
            meta_state if meta_state is not None else (str(offset),)
        )

    # --- state property ---

    @property
    def state(self):
        return (self.keys, self.values)

    @state.setter
    def state(self, value):
        self.keys, self.values = value

    # --- meta_state property ---

    @property
    def meta_state(self):
        return self._meta_state

    @meta_state.setter
    def meta_state(self, value):
        self._meta_state = value

    # --- offset ---

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        self._offset = value

    # --- nbytes ---

    @property
    def nbytes(self):
        return self.keys.nbytes + self.values.nbytes

    # --- from_state (called by load_prompt_cache) ---

    @classmethod
    def from_state(cls, state, meta_state):
        """Reconstruct a MockCacheLayer from saved state/metadata."""
        obj = cls.__new__(cls)
        if isinstance(state, tuple) and len(state) == 2:
            obj.keys, obj.values = state
        else:
            obj.keys, obj.values = state[0], state[1]
        obj._meta_state = meta_state
        # Infer offset from the sequence dimension of keys
        if obj.keys is not None:
            obj._offset = obj.keys.shape[-2] if obj.keys.ndim >= 2 else 0
        else:
            obj._offset = 0
        return obj


# Register MockCacheLayer into the cache module's namespace so that
# load_prompt_cache's globals()[class_name] lookup succeeds.
cache_mod.MockCacheLayer = MockCacheLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_layers(num_layers=4, batch=1, heads=4, seq=8, dim=16):
    """Return a list of MockCacheLayer instances with random data."""
    layers = []
    for _ in range(num_layers):
        k = mx.random.uniform(shape=(batch, heads, seq, dim))
        v = mx.random.uniform(shape=(batch, heads, seq, dim))
        layers.append(MockCacheLayer(k, v, offset=seq))
    return layers


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSsdCache(unittest.TestCase):

    # -- lifecycle ----------------------------------------------------------

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -- 1. save_entry ------------------------------------------------------

    def test_save_entry(self):
        """save_entry writes a .safetensors file to disk."""
        cache = SsdCache(self.test_dir)
        layers = _make_layers()
        key = cache.save_entry("test_model", [1, 2, 3], layers)
        self.assertIsNotNone(key)
        # Verify file exists
        expected_path = cache._entry_path(key)
        self.assertTrue(expected_path.exists(),
                        f"Expected safetensors file at {expected_path}")
        self.assertGreater(expected_path.stat().st_size, 0)

    # -- 2. load_entry ------------------------------------------------------

    def test_load_entry_roundtrip(self):
        """load_entry returns arrays that match the originals."""
        cache = SsdCache(self.test_dir)
        layers = _make_layers()
        model_key = "test_model"
        tokens = [10, 20, 30, 40]

        # Save
        key = cache.save_entry(model_key, tokens, layers)
        self.assertIsNotNone(key)

        # Load
        loaded = cache.load_entry(model_key, tokens)
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded), len(layers))

        for orig, lyr in zip(layers, loaded):
            # KV arrays should match
            self.assertTrue(
                mx.array_equal(orig.state[0], lyr.state[0]),
                "Keys mismatch after load",
            )
            self.assertTrue(
                mx.array_equal(orig.state[1], lyr.state[1]),
                "Values mismatch after load",
            )

    # -- 3. save/load multiple entries --------------------------------------

    def test_multiple_entries(self):
        """Multiple entries can be saved and loaded independently."""
        cache = SsdCache(self.test_dir)
        entries = {
            "model_a": ([1, 2], _make_layers(seq=4)),
            "model_a": ([3, 4, 5], _make_layers(seq=6)),
            "model_b": ([99], _make_layers(seq=2)),
        }

        saved_keys = {}
        # (re-key to avoid dict key collision on "model_a")
        dataset = [
            ("model_a", [1, 2], _make_layers(seq=4)),
            ("model_a", [3, 4, 5], _make_layers(seq=6)),
            ("model_b", [99], _make_layers(seq=2)),
        ]
        for model_key, tokens, layers in dataset:
            key = cache.save_entry(model_key, tokens, layers)
            self.assertIsNotNone(key)
            saved_keys[(model_key, tuple(tokens))] = key

        self.assertEqual(cache.entry_count, 3)

        for model_key, tokens, layers in dataset:
            loaded = cache.load_entry(model_key, tokens)
            self.assertIsNotNone(loaded)
            for orig, lyr in zip(layers, loaded):
                self.assertTrue(mx.array_equal(orig.state[0], lyr.state[0]))
                self.assertTrue(mx.array_equal(orig.state[1], lyr.state[1]))

    # -- 4. delete_entry ----------------------------------------------------

    def test_delete_entry(self):
        """delete_entry removes the file and index entry."""
        cache = SsdCache(self.test_dir)
        layers = _make_layers()
        key = cache.save_entry("mdl", [5, 6, 7], layers)
        self.assertIsNotNone(key)
        self.assertEqual(cache.entry_count, 1)

        # Delete
        deleted = cache.delete_entry(key)
        self.assertTrue(deleted)

        # File should be gone
        self.assertFalse(cache._entry_path(key).exists())
        self.assertEqual(cache.entry_count, 0)

        # Loading should now return None
        loaded = cache.load_entry("mdl", [5, 6, 7])
        self.assertIsNone(loaded)

    # -- 5. _enforce_size_limit (eviction) ----------------------------------

    def test_eviction_via_size_limit(self):
        """Entries beyond max_size_gb are evicted (oldest first)."""
        # Set max_size_gb very small (~1 KB)
        cache = SsdCache(self.test_dir, max_size_gb=0.000_001)  # ~1 KB

        # Save several entries; each uses ~2 * (1*4*8*16) * 4 bytes ≈ 4 KB
        # So even 1 entry may exceed the tiny limit — adjust approach:
        # Use very tiny arrays so multiple entries fit within budget.
        def tiny_layers():
            k = mx.zeros((1, 1, 2, 1))
            v = mx.zeros((1, 1, 2, 1))
            return [MockCacheLayer(k, v, offset=2)]

        keys = []
        for i in range(5):
            key = cache.save_entry("mdl", [i], tiny_layers())
            if key is not None:
                keys.append((i, key))

        # With such a tiny max_bytes, the cache should have evicted older
        # entries. At most 1-2 entries should remain.
        self.assertLess(cache.entry_count, 5,
                        "Expected eviction with tiny max_size_gb")
        self.assertGreater(cache.entry_count, 0,
                           "At least the most recent entry should remain")

    # -- 6. clear -----------------------------------------------------------

    def test_clear(self):
        """clear removes all entries from disk and index."""
        cache = SsdCache(self.test_dir)
        for i in range(3):
            cache.save_entry("mdl", [i], _make_layers(seq=2))

        self.assertEqual(cache.entry_count, 3)
        self.assertGreater(cache.total_bytes, 0)

        cache.clear()

        self.assertEqual(cache.entry_count, 0)
        self.assertEqual(cache.total_bytes, 0)
        # Verify no safetensors files remain
        safetensors_files = []
        for root, dirs, files in os.walk(self.test_dir):
            for f in files:
                if f.endswith(".safetensors"):
                    safetensors_files.append(os.path.join(root, f))
        self.assertEqual(len(safetensors_files), 0)

    # -- 7. entry_count and total_bytes properties --------------------------

    def test_entry_count_and_total_bytes(self):
        """entry_count and total_bytes reflect the on-disk state."""
        cache = SsdCache(self.test_dir)
        self.assertEqual(cache.entry_count, 0)
        self.assertEqual(cache.total_bytes, 0)

        # Save two entries
        layers1 = _make_layers(num_layers=2, seq=4)
        layers2 = _make_layers(num_layers=2, seq=8)
        cache.save_entry("mdl", [1, 2], layers1)
        cache.save_entry("mdl", [3, 4], layers2)

        self.assertEqual(cache.entry_count, 2)
        self.assertGreater(cache.total_bytes, 0)

        # Delete one
        key = cache._compute_key("mdl", [1, 2])
        cache.delete_entry(key)
        self.assertEqual(cache.entry_count, 1)
        self.assertGreater(cache.total_bytes, 0)

        # Delete the other
        key2 = cache._compute_key("mdl", [3, 4])
        cache.delete_entry(key2)
        self.assertEqual(cache.entry_count, 0)
        self.assertEqual(cache.total_bytes, 0)

    # -- 8. Persistence across instances ------------------------------------

    def test_persistence_across_instances(self):
        """A new SsdCache pointing to the same dir finds existing entries."""
        cache1 = SsdCache(self.test_dir)
        layers = _make_layers()
        key = cache1.save_entry("persist_mdl", [42, 43], layers)
        self.assertIsNotNone(key)
        entry_count_1 = cache1.entry_count

        # Create a *new* SsdCache instance pointing to the same directory
        cache2 = SsdCache(self.test_dir)

        self.assertEqual(
            cache2.entry_count,
            entry_count_1,
            "New instance should discover existing entries",
        )

        # Load the entry from the new instance
        loaded = cache2.load_entry("persist_mdl", [42, 43])
        self.assertIsNotNone(loaded)
        for orig, lyr in zip(layers, loaded):
            self.assertTrue(mx.array_equal(orig.state[0], lyr.state[0]))
            self.assertTrue(mx.array_equal(orig.state[1], lyr.state[1]))

    # -- 9. Index recovery from directory scan ------------------------------

    def test_index_recovery_from_scan(self):
        """When _index.json is missing, SsdCache scans .safetensors files."""
        cache1 = SsdCache(self.test_dir)
        layers = _make_layers()
        key = cache1.save_entry("scan_mdl", [1, 2], layers)
        self.assertIsNotNone(key)

        # Delete the index file to simulate corruption / fresh start
        index_path = cache1._index_path
        self.assertTrue(index_path.exists())
        index_path.unlink()

        # Create new instance — should fall back to directory scan
        cache2 = SsdCache(self.test_dir)

        self.assertGreaterEqual(
            cache2.entry_count, 1,
            "Should recover at least one entry via directory scan"
        )
        loaded = cache2.load_entry("scan_mdl", [1, 2])
        self.assertIsNotNone(loaded)

    # -- 10. Deterministic keys ---------------------------------------------

    def test_deterministic_keys(self):
        """Same model_key + tokens produce the same cache key."""
        cache = SsdCache(self.test_dir)
        k1 = cache._compute_key("model_x", [1, 2, 3])
        k2 = cache._compute_key("model_x", [1, 2, 3])
        self.assertEqual(k1, k2)

        k3 = cache._compute_key("model_x", [1, 2, 4])
        self.assertNotEqual(k1, k3)

        k4 = cache._compute_key("model_y", [1, 2, 3])
        self.assertNotEqual(k1, k4)

    # -- 11. Save twice (overwrite) -----------------------------------------

    def test_overwrite_entry(self):
        """Saving the same model_key + tokens overwrites the old entry."""
        cache = SsdCache(self.test_dir)
        tokens = [1, 2, 3]

        layers1 = _make_layers(seq=4)
        key1 = cache.save_entry("mdl", tokens, layers1)
        self.assertIsNotNone(key1)

        size1 = cache._entry_path(key1).stat().st_size
        count1 = cache.entry_count

        layers2 = _make_layers(seq=16)  # different data / size
        key2 = cache.save_entry("mdl", tokens, layers2)
        self.assertIsNotNone(key2)
        self.assertEqual(key1, key2, "Same tokens should produce same key")

        # The old file should have been replaced
        size2 = cache._entry_path(key2).stat().st_size
        self.assertNotEqual(size1, size2, "File content should differ")
        # Count should not have increased
        self.assertEqual(cache.entry_count, count1)

        # Load back — should get layers2, not layers1
        loaded = cache.load_entry("mdl", tokens)
        seq_loaded = loaded[0].state[0].shape[-2]
        self.assertEqual(seq_loaded, 16)


if __name__ == "__main__":
    unittest.main()
