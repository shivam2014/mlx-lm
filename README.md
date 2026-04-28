# mlx-lm — KV Cache Fork

Fork of [mlx-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) with two additions:

1. **Asymmetric KV cache quantization** — keys at higher precision, values lower
2. **Block-level SSD prompt cache** — persists KV blocks to disk for cross-session reuse

No model weight changes. No new architectures. Everything operates on the
key-value cache that accumulates during prompt prefill and generation.

---

## Quick Start

```bash
pip install -e .

mlx_lm.server \
  --model ~/.cache/huggingface/hub/Qwen3.6-35B-A3B-UD-MLX-4bit \
  --host 127.0.0.1 --port 8000 \
  --chat-template-args '{"enable_thinking": false}' \
  --kv-bits "(8, 4)" --kv-group-size "(64, 32)" \
  --block-ssd-cache-dir ~/.cache/mlx-lm/block_ssd_cache \
  --block-ssd-cache-max-size 50 \
  --prompt-cache-size 10
```

---

## KV Cache Quantization

### Asymmetric K/V bits

Keys and values are quantized at different precision:

- **Keys** (K8) — participate in attention dot product, need precision
- **Values** (V4) — softmax-weighted sums, tolerate more compression
- **Result** — ~30% KV memory reduction vs uniform 8-bit

```bash
--kv-bits "(8, 4)" --kv-group-size "(64, 32)"
```

Reference: [KIVI (NeurIPS 2024)](https://arxiv.org/abs/2402.02750).
Upstream: [mlx-lm PR #1074](https://github.com/ml-explore/mlx-lm/pull/1074).
This fork adds CLI tuple parsers (upstream only accepts a single int).

### Boundary layer protection

The first and last KV layers are more sensitive to V-bit reduction.
`--kv-boundary-layers N` (default: 2) keeps them at K8+V8.

```bash
--kv-boundary-layers 4 --kv-boundary-bits "(8,8)"   # wider boundary
--kv-boundary-layers 0                                # disable
```

Credit: [TurboQuant](https://github.com/ariG23498/TurboQuant) /
[TurboQuant+](https://github.com/TheTom/turboquant_plus).

---

## Block-Level SSD Prompt Cache

Extends KV cache to disk. A 45K-token conversation fills 6-8GB in RAM —
SSD tiering extends effective capacity to 50+ GB.

```
fetch_nearest_cache(model, tokens)
  +-- RAM tier (PromptTrie + hot cache)     → hit: deepcopy
  +-- SSD tier (BlockSSDCache, 256-tok blocks) → hit: deserialize + promote
```

### How blocks work

Cache is split into 256-token blocks, chained via hash:

```
SHA256(parent_hash || model_key || block_tokens)
```

Each block is a safetensors file on disk. The chain hash means any token
change at position N invalidates all blocks from N onward — see the
[system prompt caching strategy](../hermes_files/hermes-system-prompt-caching-strategy.md)
for how we handle this on the Hermes agent side.

### Startup warm-up

On server start, the 64 most-recent blocks are pre-loaded from SSD into an
in-memory LRU hot cache (~52ms). First request hits RAM, not disk.

---

## Performance Fixes

The initial SSD cache had 4 bugs that caused prefill to regress from
~650 tok/s to 4.5 tok/s on cache hit. All fixed:

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| No hot cache | Every SSD hit read from disk | LRU `_hot_cache` in RAM (commit `4a58354`) |
| Per-block mx.eval | 120 GPU sync points during prefill | Batch eval after all blocks loaded (`4154495`) |
| Deep copy overhead | `copy.deepcopy()` per block per layer | Direct `mx.load()` without intermediary (`9627d30`) |
| Eviction thrashing | `_maybe_evict()` called per block save | Deferred to post-request flush (`6726f33`) |

---

## Observability (Fix B)

Every request logs SSD cache health in a single `PERF` line:

```
PERF: prompt_tps=2330.4 prompt_tok=29624 pref_tok=3256 prefill=12.71s
      block_hit=103 block_write=0 chain_break=0
```

| Field | Meaning |
|-------|---------|
| `block_hit` | Blocks loaded from SSD this request |
| `block_write` | New blocks written to SSD |
| `chain_break` | 1 = block-0 prefix mismatch (prompt drifted) |

Find regressions:
```bash
grep "PERF:" server.log | grep "chain_break=1"
```

---

## Benchmarks

**Hardware:** M1 Max 64GB | **Model:** Qwen3.6-35B-A3B-UD-MLX-4bit

### Cross-session SSD cache (Hermes Agent, 29K-token system prompt)

| | prefill | tok/s | cached |
|---|---|---|---|
| Cold (no SSD) | 85.01s | 348 | 0% |
| SSD hit (Run 2) | 12.71s | 2,330 | 89% |
| Within-session | 0.90s | 30,101 | 99.9% |

**6.7x** cross-session prefill speedup. **538x** within-session.

### Long conversation (8 requests, 29K → 47K tokens)

| prompt_tok | pref_tok | cached | chain_break | prefill |
|---|---|---|---|---|
| 29,629 | 3,261 | 89% | 0 | 12.78s |
| 33,019 | 3,360 | 90% | 0 | 13.22s |
| 39,943 | 3,318 | 92% | 0 | 15.01s |
| 47,165 | 3,672 | 92% | 0 | 17.83s |

Prefill stays flat (~13-18s) despite prompt growing to 47K tokens.
`chain_break=0` on every request — prompt byte-stable.

![SSD Cache Benchmark Results](benchmark_fix_abc.png)

Full data: [SSD_CACHE_BENCHMARK_FIX_ABC.md](SSD_CACHE_BENCHMARK_FIX_ABC.md) · [SSD_CACHE_BENCHMARK_RESULTS.md](SSD_CACHE_BENCHMARK_RESULTS.md)

---

## CLI Flags (Fork-Specific)

| Flag | Default | Description |
|------|---------|-------------|
| `--kv-bits` | `8` | KV bit-width: int or tuple `(key, value)` |
| `--kv-group-size` | `64` | Quantization group size: int or tuple |
| `--kv-boundary-layers` | `2` | Boundary KV layers to protect (0=off) |
| `--kv-boundary-bits` | `(8,8)` | Bit-width for boundary layers |
| `--block-ssd-cache-dir` | — | Directory for SSD block cache |
| `--block-ssd-cache-max-size` | `50` | Max SSD cache size (GB) |
| `--prompt-cache-size` | `10` | RAM cache entries |

---

## Attribution

- [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) — upstream
- [KIVI (Liu et al., NeurIPS 2024)](https://arxiv.org/abs/2402.02750) — asymmetric quantization
- [TurboQuant+](https://github.com/TheTom/turboquant_plus) — boundary-aware compression
- [jundot/omlx](https://github.com/jundot/omlx) — `paged_ssd_cache.py` reference
- [Goose #4610](https://github.com/block/goose/issues/4610) — timestamps preventing LLM caching
- [EPIC (2410.15332)](https://arxiv.org/abs/2410.15332) — position-independent caching
- [Prompt Cache (2311.04934)](https://arxiv.org/abs/2311.04934) — modular attention reuse
- [CacheBlend (2405.16444)](https://arxiv.org/abs/2405.16444) — selective KV recomputation
