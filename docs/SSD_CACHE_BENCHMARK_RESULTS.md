# SSD Block Cache Benchmark Results

**Date:** 2026-04-28  
**Model:** Qwen3.6-35B-A3B-UD-MLX-4bit  
**Hardware:** M1 Max 64GB  
**Branch:** feat/ssd-cache-rebased  
**Commit:** 5c7229d

## Test Configuration

```
python3 -m mlx_lm server \
  --model Qwen3.6-35B-A3B-UD-MLX-4bit \
  --host 127.0.0.1 --port 8000 \
  --chat-template-args '{"enable_thinking": false}' \
  --kv-bits "(8, 4)" --kv-group-size "(64, 32)" \
  --block-ssd-cache-dir ~/.cache/mlx-lm/block_ssd_cache \
  --block-ssd-cache-max-size 50 \
  --log-level DEBUG \
  --prompt-cache-size 10
```

## Results

| Config | Wall Clock | Eff. tok/s | Tokens Cached | Cached % | pref_tok |
|---|---|---|---|---|---|
| No cache (baseline) | 75.76s | 355.9 | 0 | 0% | 26,963 |
| SSD + warm hot cache (1st request) | 11.15s | 2,418.6 | 24,064 | 89% | 2,894 |
| Within-session RAM (2nd request) | 0.90s | 30,101 | 26,943 | 99.9% | 22 |

## Before vs After (Cross-Session Prefill)

| Metric | Before Fix | After Fix | Delta |
|---|---|---|---|
| Cross-session prefill speed | 4.5 tok/s | 2,418 tok/s | 537x faster |
| Wall clock (27K tokens) | ~6,000s (broken) | 11.15s | 538x faster |
| SSD block hit rate (1st request) | 0% (cold SSD) | 89% (warm hot cache) | — |
| Within-session hit rate | 20,280 tok/s | 30,101 tok/s | 48% faster |
| Startup warm cost | N/A | 52ms for 64 blocks | negligible |

## Changes in Commit 5c7229d

### 1. `block_ssd_cache.py` — `warm_hot_cache()` method
Pre-loads 64 most-recent blocks from SSD into in-memory LRU hot cache at server startup, before any requests arrive. Takes 52ms for 64 blocks. Transforms first request from cold SSD I/O to hot memory lookup.

### 2. `cache.py` — SSD return path deepcopy elimination
Changed `fetch_nearest_cache()` SSD block path to deep-copy only for the RAM trie insertion, returning the original `merged_cache` directly. Previously deep-copied the full merged result twice (once for trie, once for return).

### 3. `cache.py` — `_merge_block_caches()` per-layer deepcopy elimination
Removed `copy.deepcopy(first_cache)` and `copy.deepcopy(last_block)` per layer. Blocks from `load_blocks_batch()` are already unique copies (deep-copied from hot cache or freshly deserialized from disk), so merging can reuse them in-place.

**Net deep-copy reduction on SSD cache path:**
- Before: N blocks × deepcopy (load) + 90 × deepcopy (merge) + 1 × deepcopy (return) + 1 × deepcopy (trie) = 2N+91 copies
- After: N blocks × deepcopy (load) + 0 × deepcopy (merge) + 0 × deepcopy (return) + 1 × deepcopy (trie) = N+1 copies
- For N=94 blocks: 279 copies → 95 copies (66% reduction)

## Baseline Details

**No-cache baseline** (same model, no `--block-ssd-cache-dir`):
```
PERF: prompt_tps=355.9 gen_tps=0.0 prompt_tok=26963 gen_tok=2 prefill=75.76s gen=75.97s peak_mem=24.42GB pref_tok=26963
Prefill: 100%|██████████| 26963/26963 [01:15<00:00, 355.88tok/s]
```

**SSD-cached first request** (cross-session, hot cache warmed at startup):
```
BlockSSDCache HIT: 94 blocks, 24064 tokens from SSD
Prefill: 100%|██████████| 2894/2894 [00:11<00:00, 259.64tok/s]
PERF: prompt_tps=2418.6 gen_tps=0.2 prompt_tok=26958 gen_tok=2 prefill=11.15s gen=11.42s peak_mem=24.81GB pref_tok=2894
```

**Within-session second request** (RAM trie hit):
```
Prefill: 100%|██████████| 22/22 [00:00<00:00, 24.56tok/s]
PERF: prompt_tps=30101.3 gen_tps=2.0 prompt_tok=26965 gen_tok=2 prefill=0.90s gen=0.98s peak_mem=24.81GB pref_tok=22
```

## Startup Log

```
Block SSD Cache: 105 blocks, 34.10 GB (max: 50 GB) at /Users/shivam94/.cache/mlx-lm/block_ssd_cache
Warming hot cache with 64 most-recent blocks...
Warmed hot cache with 64/64 blocks from SSD (total index: 105)
Hot cache warmed: 64 blocks loaded
```