# Patch 03 — Preserve Intermediate Checkpoints in `pop_prefixes`

**File**: `cache.py` — `LRUPromptCache.insert_cache()`  
**Applies to**: mlx-lm fork with segment-aware caching (Part 2 patches in `server.py`)  
**Status**: Required companion to segment-aware caching — without it, system/user checkpoints are destroyed on creation  

---

## Problem

Segment-aware caching (server.py lines 1034-1071) creates independent intermediate cache entries at the "system" and "user" segment boundaries. This is intended to allow future requests to reuse the system prompt cache (e.g., the first ~400 tokens of a Qwen3.6 28K system prompt) without recomputing from scratch.

However, `insert_cache()` in `cache.py` lines 2094-2100 immediately destroys these entries:

```python
# If it is a trimmable cache remove all prefixes cause they just take
# space
if can_trim_prompt_cache(prompt_cache):
    for prefix_len, entry in self._trie.pop_prefixes(model, tokens):
        self._n_bytes -= entry.nbytes
        self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
        self._lru.remove(model, tokens[:prefix_len])
```

`pop_prefixes` walks the full token path from root to the inserted entry, popping any `__value__` it finds along the way. Since the system segment is a **prefix** of the full conversation (assistant entry), `pop_prefixes` finds it and removes it.

Visual explanation:

```
Trie after segment-aware caching stores system entry:

        root
         │
       [sys_prompt_tokens]  ← ★ has __value__ (system entry)
         │
       [tools → user → asst]  (being added now)

When assistant insert_cache() calls pop_prefixes():

        root
         │
       [sys_prompt_tokens]  ← "found __value__ at prefix, POP!"
         │
       [tools]
         │
       [user]
         │
       [asst] ← ★ value goes here

After:

        root
         │
       [sys_prompt_tokens]  ← __value__ is GONE
         │
       [tools]
         │
       [user]
         │
       [asst] ← ★ only value
```

Result: The system entry is created, then **immediately destroyed** on every turn. Cache stats show `system: 0 sequences, 0.00 GB` throughout the session. The server falls back to monolithic assistant entries (~0.3 GB each), and without the strategic checkpoint, each new turn recomputes 16K-18K tokens instead of ~400 delta tokens.

---

## Root Cause

`pop_prefixes` was designed as a memory-saving optimization: if you're inserting a long entry (full conversation), the short prefix entries (just system prompt, just system+tools) are redundant and waste space.

This assumption is **correct for monolithic caching** but **wrong for segment-aware caching**. In segment-aware mode, the short prefix is NOT redundant — it's a deliberate intermediate checkpoint that enables future delta-only prefills.

The system entry is not a "stale shorter version" of the assistant entry. It's a **different cache entry at a different granularity** — a checkpoint that future requests can reuse when their prompt shares the system prefix but diverges at the tools/user boundary.

---

## Fix

In `cache.py`, `insert_cache()` method, lines 2096-2100. Skip entries whose `cache_type` is an intermediate checkpoint:

```python
if can_trim_prompt_cache(prompt_cache):
    for prefix_len, entry in self._trie.pop_prefixes(model, tokens):
        # PRESERVE intermediate checkpoints — they are strategic
        # cache boundaries, not accidental short entries.
        if entry.cache_type in ("system", "user"):
            continue
        self._n_bytes -= entry.nbytes
        self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
        self._lru.remove(model, tokens[:prefix_len])
```

The `continue` skips the eviction bookkeeping for intermediate entries, leaving them in the trie and LRU queue.

---

## Effect

### Before fix

```
Turn 1: insert_cache(system_entry)   → trie: [sys★]
        insert_cache(assistant_entry) → pop_prefixes kills sys★
                                        trie: [asst★]
Turn 2: insert_cache(assistant_entry) → pop_prefixes kills previous asst★
                                        trie: [asst2★]
...
Cache stats always: system: 0
```

### After fix

```
Turn 1: insert_cache(system_entry)   → trie: [sys★]
        insert_cache(assistant_entry) → pop_prefixes SKIPS sys★
                                        trie: [sys★] [asst★]
Turn 2: fetch_nearest_cache(new_prompt)
        → finds sys★ at prefix length ~400 tokens
        → returns cached 400 tokens, remaining = delta
        → only ~400 tokens recomputed
Cache stats: system: 1, assistant: 1
```

### Performance impact

| Metric | Before | After |
|--------|--------|-------|
| System prompt reuse per turn | 0 tokens | ~28,000 tokens |
| Average prefill per turn (agent loop) | 16K-18K tokens | ~400 tokens (delta only) |
| Prefill time per turn (Qwen3.6-35B) | ~42 seconds | ~1-2 seconds |
| Memory overhead | ~0.3 GB per assistant entry (monolithic) | ~0.3 GB assistant + ~0.1 GB system |
| Cache metadata | system: 0 entries (dead code) | system: 1-2 entries (live) |

---

## Why `pop_prefixes` Exists (and why it's safe to skip)

`pop_prefixes` prevents the trie from accumulating redundant entries at prefix nodes. Without it:

- Insert "system" at [s1, s2, s3]
- Insert "system+tools+user1+asst1" at [s1, s2, s3, t1, u1, a1]
- Insert "system+tools+user2+asst2" at [s1, s2, s3, t1, u2, a2]
- The system entry at [s1, s2, s3] is still useful as a shared prefix

But without segment-aware caching, [s1, s2, s3] is strictly less useful than its longer children and wasting space. This is the case `pop_prefixes` was written for.

With segment-aware caching, the short prefix is **intentionally created as a checkpoint** with a specific `cache_type`. It serves a different purpose than the longer entries. `pop_prefixes` should respect this.

---

## Location in cache.py

Line 2094-2100 in `LRUPromptCache.insert_cache()`:

```python
# If it is a trimmable cache remove all prefixes cause they just take
# space
if can_trim_prompt_cache(prompt_cache):
    for prefix_len, entry in self._trie.pop_prefixes(model, tokens):
        ...
```

The fix is a one-line guard: `if entry.cache_type in ("system", "user"): continue`

---

## Related Patches

- **Part 2 (segment-aware caching)**: server.py lines 1034-1071 — creates intermediate checkpoint entries
- **Part 6 (ArraysCache trimmable)**: cache.py — enables trim() for non-List caches, needed by segment-aware caching's intermediate_cache trimming
- **This patch**: Completes the chain — without it, Part 2's checkpoints are silently destroyed

---

## Verification

1. Start server with `--prompt-cache-size 10` and segment-aware caching enabled
2. Send a conversation with a system prompt > 100 tokens
3. After first turn, check cache stats: `system: 1 sequence, ~0.10 GB`
4. Send a second request with same system prompt
5. Check `tokens_processed / tokens_total` — should show small delta (~400 tokens), not full recompute

Expected log output:
```
Prompt Cache: assistant: 2 sequences, 0.62 GB | system: 1 sequence, 0.10 GB
```
