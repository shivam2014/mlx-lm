"""Unit test for the __value__ trie desync bug in pop_prefixes.

The bug: pop_prefixes() destructively pops __value__ from every prefix node.
When insert_cache preserves system/user entries, it continues without
re-inserting __value__. Later, eviction hits KeyError('__value__').

This test re-implements PromptTrie locally (pure dicts, no mlx dep)
to verify the fix: self._trie.add() re-inserts the preserved value.
"""

import unittest
from typing import Any, Dict, List, Optional, Tuple


class PromptTrie:
    """Replica of cache.py PromptTrie for standalone testing."""

    def __init__(self):
        self._trie: Dict = {}

    def add(self, model: Any, tokens: List[int], value: Any):
        if model not in self._trie:
            self._trie[model] = {}
        current = self._trie[model]
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]
        prev = current.get("__value__", None)
        current["__value__"] = value
        return prev

    def pop(self, model: Any, tokens: List[int]):
        path = [self._trie[model]]
        for tok in tokens:
            path.append(path[-1][tok])
        value = path[-1].pop("__value__")
        for i in range(len(tokens), 0, -1):
            node = path[i]
            parent = path[i - 1]
            tok = tokens[i - 1]
            if len(node) > 0:
                break
            del parent[tok]
        return value

    def pop_prefixes(self, model: Any, tokens: List[int]):
        values = []
        current = self._trie[model]
        for i, tok in enumerate(tokens):
            if "__value__" in current:
                values.append((i, current.pop("__value__")))
            current = current[tok]
        return values


class SimpleEntry:
    """Lightweight stand-in for LRUPromptCache.CacheEntry."""
    def __init__(self, name: str, cache_type: str = "assistant"):
        self.name = name
        self.cache_type = cache_type
        self.nbytes = 0


class TestPopPrefixesDestructivePop(unittest.TestCase):
    """Verify that pop_prefixes destructively removes __value__."""

    def test_pop_prefixes_removes_value_from_trie(self):
        t = PromptTrie()
        t.add("m", [1, 2, 3], "final")
        t.add("m", [1], "prefix_val")
        # After add, node [1] has __value__ = "prefix_val"
        # pop_prefixes should remove it
        results = t.pop_prefixes("m", [1, 2, 3])
        self.assertEqual(len(results), 1, "should find exactly one prefix")
        self.assertEqual(results[0][1], "prefix_val",
                         "should return prefix value")
        # Node [1] should no longer have __value__
        current = t._trie["m"][1]
        self.assertNotIn("__value__", current,
                         "__value__ should be GONE after pop_prefixes")


class TestTrieDesyncBug(unittest.TestCase):
    """The actual bug: preserved entries lose __value__ in trie."""

    def test_system_entry_reinserted_after_pop_prefixes(self):
        """Simulate the fix: after pop_prefixes removes a system entry's
        __value__, _trie.add() puts it back."""
        t = PromptTrie()
        model = "m"
        # Insert system entry at prefix [1]
        system_entry = SimpleEntry("system", "system")
        t.add(model, [1], system_entry)
        # Insert full path [1, 2, 3] — this triggers pop_prefixes
        full_entry = SimpleEntry("assistant_response")
        t.add(model, [1, 2, 3], full_entry)
        # pop_prefixes walks [1,2,3], finds system __value__ at [1], POPS it
        t.pop_prefixes(model, [1, 2, 3])
        #  Now node [1] is missing __value__. Simulate the fix:
        t.add(model, [1], system_entry)
        # Node [1] should have __value__ back
        current = t._trie[model][1]
        self.assertIn("__value__", current,
                      "after re-insert, __value__ should exist")
        self.assertIs(current["__value__"], system_entry,
                      "re-inserted value should be the system entry")

    def test_system_entry_survives_eviction_after_fix(self):
        """Full scenario: insert system, insert assistant (triggers pop_prefixes),
        re-insert system, then pop() the system entry — should not crash."""
        t = PromptTrie()
        model = "m"
        system_entry = SimpleEntry("system", "system")
        t.add(model, [1], system_entry)
        assistant_entry = SimpleEntry("assistant")
        t.add(model, [1, 2, 3], assistant_entry)
        # pop_prefixes + re-insert (the fix)
        for prefix_len, entry in t.pop_prefixes(model, [1, 2, 3]):
            if entry.cache_type in ("system", "user"):
                t.add(model, [1], entry)  # re-insert
        # Now pop the system entry — should not raise KeyError
        popped = t.pop(model, [1])
        self.assertIs(popped, system_entry,
                      "should pop the system entry successfully")


class TestAssistantPrefixesAreCleaned(unittest.TestCase):
    """Verifiy that non-system/user prefixes (assistant) are NOT re-inserted,
    so they get cleaned up as intended."""

    def test_assistant_prefix_not_reinserted(self):
        t = PromptTrie()
        model = "m"
        # Insert assistant at prefix [1]
        t.add(model, [1], SimpleEntry("old_assistant", "assistant"))
        # Insert longer assistant path
        t.add(model, [1, 2], SimpleEntry("new_assistant", "assistant"))
        # pop_prefixes finds the prefix at [1], pops it
        results = t.pop_prefixes(model, [1, 2])
        self.assertEqual(len(results), 1)
        # Do NOT re-insert (assistant is not preserved)
        # Now node [1] should NOT have __value__
        current = t._trie[model][1]
        self.assertNotIn("__value__", current,
                         "assistant prefix __value__ should stay removed")


if __name__ == "__main__":
    unittest.main()
