mlx-lm — fork with KV cache quantization & SSD-tier prompt cache
===============================================================

Fork of [mlx-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) focused on
KV cache optimization for long-context agent workloads on memory-constrained
Apple Silicon.

All additions target the KV cache layer — no model weight changes, no new
model architectures. Every feature here operates on the key-value cache that
accumulates during prompt prefill and generation.

---

KV Cache Quantization
---------------------

### Asymmetric K/V bits (KVSplit)

KV cache entries are quantized per-tensor: keys at higher precision, values at
lower precision. Keys participate in the attention dot product — quantization
noise there directly shifts the attention distribution. Values are
softmax-weighted sums, which acts as a low-pass filter, so they tolerate more
aggressive compression.

The `--kv-bits` flag accepts a tuple `(key_bits, value_bits)`:

```bash
--kv-bits "(8, 4)" --kv-group-size "(64, 32)"
```

K8 + V4 reduces KV cache memory by ~30% vs uniform 8-bit on Qwen3.6-35B-A3B
(10 KV layers out of 40). Separate group sizes for keys (64) and values (32)
let each side use its optimal quantization granularity.

Reference: [KIVI — A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) (Liu et al., NeurIPS 2024)

The upstream implementation is [mlx-lm PR #1074](https://github.com/ml-explore/mlx-lm/pull/1074)
(from ml-explore, merged). This fork adds CLI tuple parsers for the server
path — upstream `--kv-bits` only accepts a single int.

### Boundary layer protection

The first and last KV cache layers are systematically more sensitive to V-bit
reduction — they handle input embedding projection and final logit
transformation where quantization noise compounds. Middle layers operate on
already-abstracted representations where V precision has less marginal impact.

`--kv-boundary-layers N` (default 2, set to 0 to disable) applies higher V
precision to the first N and last N actual KV cache layers. Default boundary
bits are (8,8) — K8+V8 on boundary layers, whatever `--kv-bits` specifies for
middle layers.

For hybrid architectures (Qwen3.5/3.6, full_attention_interval=4), only 10 of
40 layers have KV caches. The selection logic correctly finds actual KVCache
entries in the prompt_cache list, skipping linear-attention layers
(ArraysCache). Boundary protection costs ~64 MB at 128K context for N=2.

```bash
# Default (on): first 2 + last 2 KV layers at (8,8), rest at (8,4)
# No extra flags needed

# Custom boundary width
--kv-boundary-layers 4 --kv-boundary-bits "(8,8)"

# Disable boundary protection
--kv-boundary-layers 0
```

Credit: [ariG23498/TurboQuant](https://github.com/ariG23498) and
[TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) for the
boundary-aware approach. The TurboQuant+ Layer-Aware V Compression paper
validates that protecting the first 2 and last 2 layers with higher V
precision recovers a meaningful fraction of quality loss from aggressive V
compression, at minimal memory cost.

---

Block-Level SSD Prompt Cache
----------------------------

The prompt KV cache (LRUPromptCache) lives entirely in RAM. On a 64GB M1 Max,
a 35B MoE model uses ~25-30GB, leaving 25-35GB for KV cache. A single
45K-token conversation fills 6-8GB. SSD tiering extends effective cache
capacity to 50+ GB by persisting evicted blocks to disk.

Architecture (two-tier):

```
fetch_nearest_cache(model, tokens)
  |
  +-- RAM tier (PromptTrie + in-memory hot cache)
  |     Hit -> return deepcopy
  |     Miss -> fall through to SSD
  |
  +-- SSD tier (BlockSSDCache, chain-hash addressed)
        Hit -> deserialize -> promote to hot cache -> return
        Miss -> return None (create fresh)
```

### Block addressing

Cache is split into 256-token blocks, chained via hash:
`SHA256(parent_hash || model_key || block_tokens)`. Each block stored as a
safetensors file with two-char hash prefix subdirectory layout to avoid
flat-directory explosion. The chain hash means any token change invalidates
all subsequent blocks — the Hermes system prompt reordering (Phase 1, see
docs) addresses this by placing dynamic sections (memory, timestamp) at the
end.

### Hot cache warm-up

On server startup, the 64 most-recent blocks are pre-loaded from SSD into an
in-memory LRU hot cache in ~52ms. This transforms the first request from cold
SSD I/O (~150 tok/s, 34 sequential reads) to hot memory lookup (~2,400 tok/s).

### Performance bug fixes (4 critical issues)

The initial SSD cache implementation had four performance bugs that caused
prefill to regress from ~650 tok/s (baseline) to 4.5 tok/s on cache hit.
All fixed:

**Fix 1 — Hot/in-memory LRU block cache.** The original `load_block()` hit
disk on every call via `load_prompt_cache()`. A 34-block SSD hit meant 34
sequential disk reads + deserialization during the prefill hot path. Fixed by
adding an LRU `_hot_cache` dict that keeps deserialized blocks in memory and
promotes on SSD load. (Commit `4a58354`)

**Fix 2 — mx.eval() GPU sync per block during save.** `_extract_cache_bytes()`
called `mx.eval()` per block. For 120 blocks, that was 120 GPU
synchronization points stalling the Metal pipeline. Fixed by deferring SSD
saves to post-request, batching all tensors, and running a single `mx.eval()`
before extracting bytes for all blocks. (Commit `4154495`)

**Fix 3 — copy.deepcopy() per layer per block.** `slice_cache_for_block()`
deep-copied every `boundary_only` and `unknown` layer for each of 120 blocks.
For a 90-layer model, that's ~10,800 deep copies. Fixed by eliminating
redundant deep copies: blocks loaded from disk are already unique copies, so
merging can reuse them in-place. (Commit `9627d30`)

**Fix 4 — load_prompt_cache() overhead.** Block load used
`load_prompt_cache()` which includes metadata reconstruction, cache class
instantiation, and compatibility layers. Replaced with direct `mx.load()` +
`_reconstruct_cache_data()` following the omlx pattern. (Commit `4a58354`)

**Net effect:** 66% reduction in deep copies (279 -> 95 copies per block
batch), zero GPU syncs during prefill save path, ~2x faster per-block load.

### Eviction thrashing fix

The original `_save_blocks_to_ssd()` called `_maybe_evict()` eagerly on every
single block save. When the cache hit its size limit during a batch save,
eviction removed blocks that had already been verified as present by
`contains()` earlier in the same loop — creating holes in the block chain and
rendering the cache useless for subsequent requests.

Fixed by deferring eviction to end-of-batch: `save_block()` only indexes,
`shrink()` runs once after all blocks are saved. (Commit `6726f33`)

---

Prefill Cache Instrumentation
-----------------------------

### Actual prefill wall time (was: queue round-trip)

`_prefill_time` was measured from `generate()` call to return, which captured
queue round-trip time (~0.18s for 28K tokens) rather than actual prefill wall
time (~40s). Both the single-request and batch paths put context on the
response queue BEFORE prefill starts.

Fixed by using the `keepalive_callback` closure (already wired for SSE
keepalive) to measure actual prefill: record `perf_counter` when prefill
starts (tqdm bar first appears), compute elapsed when `processed >= total`
(prefill completes). (Commit `580b9a5`)

### Live tqdm prefill bar

Replaced the text-based `Prompt processing progress: {processed}/{total}`
INFO log line with a live tqdm progress bar writing to stderr, borrowing the
pattern from mlx-vlm's `generate.py`. SSE keepalive messages are preserved to
keep the HTTP connection alive during long prefill. (Commit `383e694`)

### Per-request performance metrics

Each request now logs at INFO level:

```
PERF: prompt_tps=2418.6 gen_tps=0.2 prompt_tok=26958 gen_tok=2
      prefill=11.15s gen=11.42s peak_mem=24.81GB pref_tok=2894
```

`pref_tok` shows real uncached tokens (prompt_tok - cached_tokens), so at a
glance you see how many tokens were actually processed by the model vs served
from cache. (Commits `87536ee`, `0489b7c`)

---

Bug Fixes
---------

- **CachePy trie `pop_prefixes`** — Destructively popped `__value__` before
  `cache_type` check, causing silent context corruption in multi-turn
  conversations. Fixed by re-inserting system/user entries via `_trie.add()`.
- **Default `max_tokens`** — Upstream default was 512, which truncated
  responses when clients omitted the parameter. Changed to -1 (unlimited).
- **`ArraysCache.is_trimmable`** — Inverted `hasattr` workaround caused
  incorrect feature detection for multimodal ArraysCache compatibility.
- **Response truncation** — Non-zero truncation count lost final tokens of
  generation.

---

Benchmarks
----------

### SSD Cache: Cross-session prefill speedup

**Hardware:** M1 Max 64GB | **Model:** Qwen3.6-35B-A3B-UD-MLX-4bit
**Context:** ~27K tokens (system prompt + tool definitions)
**Date:** 2026-04-28 | **Branch:** feat/ssd-cache-rebased

| Config | Wall Clock | Eff. tok/s | Cached % | pref_tok |
|--------|-----------|------------|----------|----------|
| No cache (baseline) | 75.76s | 355.9 | 0% | 26,963 |
| SSD + warm hot cache (1st request) | 11.15s | 2,418.6 | 89% | 2,894 |
| Within-session RAM (2nd request) | 0.90s | 30,101 | 99.9% | 22 |

**Before vs After (cross-session prefill):**

| Metric | Before Fix | After Fix | Delta |
|--------|-----------|-----------|-------|
| Cross-session prefill | 4.5 tok/s | 2,418 tok/s | 537x |
| Wall clock (27K tokens) | ~6,000s | 11.15s | 538x |
| Within-session hit rate | 20,280 tok/s | 30,101 tok/s | 48% faster |

The baseline (no SSD cache, no RAM cache) takes 75.76 seconds to prefill 27K
tokens. With the fixed SSD cache and hot cache warmup, the first request drops
to 11.15 seconds (6.8x faster). The second request within the same session
hits the RAM trie: 0.90 seconds (84x vs first request, 51x vs cold start).

### Server startup

```
Block SSD Cache: 105 blocks, 34.10 GB (max: 50 GB)
Warming hot cache with 64 most-recent blocks...
Warmed hot cache with 64/64 blocks from SSD (total index: 105)
Hot cache warmed: 64 blocks loaded
```

Startup warm cost: 52ms for 64 blocks.

Full benchmark results: [SSD_CACHE_BENCHMARK_RESULTS.md](SSD_CACHE_BENCHMARK_RESULTS.md)

---

CLI Flags
---------

| Flag | Default | Description |
|------|---------|-------------|
| `--kv-bits` | `8` | KV cache bit-width: int (uniform) or tuple (key,value) |
| `--kv-group-size` | `64` | Quantization group size: int or tuple (key,value) |
| `--quantized-kv-start` | `0` | Token offset to start quantization |
| `--kv-boundary-layers` | `2` | Number of boundary KV layers to protect (0=disable) |
| `--kv-boundary-bits` | `(8,8)` | Bit-width for boundary layers |
| `--block-ssd-cache-dir` | `None` | Directory for SSD-persisted block cache |
| `--block-ssd-cache-max-size` | `50` | Max SSD cache size in GB |
| `--prompt-cache-size` | `10` | Number of cache entries in RAM |

---

Server Usage
------------

```bash
python3 -m mlx_lm server \
  --model Qwen3.6-35B-A3B-UD-MLX-4bit \
  --host 127.0.0.1 --port 8000 \
  --chat-template-args '{"enable_thinking": false}' \
  --kv-bits "(8, 4)" --kv-group-size "(64, 32)" \
  --block-ssd-cache-dir ~/.cache/mlx-lm/block_ssd_cache \
  --block-ssd-cache-max-size 50 \
  --prompt-cache-size 10
```

Boundary protection is on by default (first 2 + last 2 layers at 8-bit V).
No extra flags needed to enable it.

---

Papers & References
-------------------

| Paper | Topic | Link |
|-------|-------|------|
| **KIVI** — Liu et al., NeurIPS 2024 | Asymmetric K/V cache quantization (keys higher bits, values lower bits) | [2402.02750](https://arxiv.org/abs/2402.02750) |
| **TurboQuant+ Layer-Aware V Compression** — TheTom, 2026 | Boundary layer protection for aggressive V quantization | [turboquant_plus](https://github.com/TheTom/turboquant_plus) |
| **SpecPrefill** — Jin et al., ICML 2025 | TTFT reduction via sparse prefill with draft-model token importance scores | [2502.02789](https://arxiv.org/abs/2502.02789) |
| **Prompt Cache** — Gim et al., 2023 | Modular attention reuse with schema-defined modules | [2311.04934](https://arxiv.org/abs/2311.04934) |
| **EPIC** — Hu et al., 2024 | Position-independent context caching with AttnLink algorithm | [2410.15332](https://arxiv.org/abs/2410.15332) |
| **CacheBlend** — Wang et al., 2024 | Selective KV cache recomputation for non-prefix chunks | [2405.16444](https://arxiv.org/abs/2405.16444) |

---

Attribution
-----------

This fork is based on [mlx-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)
by Apple Inc. (Copyright 2023-2025 Apple Inc.). All upstream code retains its
original copyright.

The live tqdm prefill progress bar pattern is from
[mlx-vlm](https://github.com/ml-explore/mlx-vlm)'s `generate.py`.

The hot cache and batch-load patterns are adapted from
[jundot/omlx](https://github.com/jundot/omlx)'s `paged_ssd_cache.py`.

All new files (`block_ssd_cache.py`, `block_cache_utils.py`,
`hermes_prefix_cache.py`, test files) are original to this fork.

---

*Below this line is the upstream README from
[mlx-explore/mlx-lm](https://github.com/ml-explore/mlx-lm).*

## MLX LM

MLX LM is a Python package for generating text and fine-tuning large language
models on Apple silicon with MLX.

Some key features include:

* Integration with the Hugging Face Hub to easily use thousands of LLMs with a
  single command.
* Support for quantizing and uploading models to the Hugging Face Hub.
* [Low-rank and full model
  fine-tuning](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
  with support for quantized models.
* Distributed inference and fine-tuning with `mx.distributed`

The easiest way to get started is to install the `mlx-lm` package:

**With `pip`**:

```sh
pip install mlx-lm
```

**With `conda`**:

```sh
conda install -c conda-forge mlx-lm
```

### Quick Start

To generate text with an LLM use:

```bash
mlx_lm.generate --prompt "How tall is Mt Everest?"
```

To chat with an LLM use:

```bash
mlx_lm.chat
```

This will give you a chat REPL that you can use to interact with the LLM. The
chat context is preserved during the lifetime of the REPL.

Commands in `mlx-lm` typically take command line options which let you specify
the model, sampling parameters, and more. Use `-h` to see a list of available
options for a command, e.g.:

```bash
mlx_lm.generate -h
```

The default model for generation and chat is
`mlx-community/Llama-3.2-3B-Instruct-4bit`.  You can specify any MLX-compatible
model with the `--model` flag. Thousands are available in the
[MLX Community](https://huggingface.co/mlx-community) Hugging Face
organization.

### Python API

You can use `mlx-lm` as a module:

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

prompt = "Write a story about Einstein"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True,
)

text = generate(model, tokenizer, prompt=prompt, verbose=True)
```

To see a description of all the arguments you can do:

```
>>> help(generate)
```

Check out the [generation
example](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/examples/generate_response.py)
to see how to use the API in more detail. Check out the [batch generation
example](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/examples/batch_generate_response.py)
to see how to efficiently generate continuations for a batch of prompts.

The `mlx-lm` package also comes with functionality to quantize and optionally
upload models to the Hugging Face Hub.

You can convert models using the Python API:

```python
from mlx_lm import convert

repo = "mistralai/Mistral-7B-Instruct-v0.3"
upload_repo = "mlx-community/My-Mistral-7B-Instruct-v0.3-4bit"

convert(repo, quantize=True, upload_repo=upload_repo)
```

This will generate a 4-bit quantized Mistral 7B and upload it to the repo
`mlx-community/My-Mistral-7B-Instruct-v0.3-4bit`. It will also save the
converted model in the path `mlx_model` by default.

To see a description of all the arguments you can do:

```
>>> help(convert)
```

#### Streaming

For streaming generation, use the `stream_generate` function. This yields
a generation response object.

For example,

```python
from mlx_lm import load, stream_generate

repo = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
model, tokenizer = load(repo)

prompt = "Write a story about Einstein"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True,
)

for response in stream_generate(model, tokenizer, prompt, max_tokens=512):
    print(response.text, end="", flush=True)
print()
```

#### Sampling

The `generate` and `stream_generate` functions accept `sampler` and
`logits_processors` keyword arguments. A sampler is any callable which accepts
a possibly batched logits array and returns an array of sampled tokens.  The
`logits_processors` must be a list of callables which take the token history
and current logits as input and return the processed logits. The logits
processors are applied in order.

Some standard sampling functions and logits processors are provided in
`mlx_lm.sample_utils`.

### Command Line

You can also use `mlx-lm` from the command line with:

```
mlx_lm.generate --model mistralai/Mistral-7B-Instruct-v0.3 --prompt "hello"
```

This will download a Mistral 7B model from the Hugging Face Hub and generate
text using the given prompt.

For a full list of options run:

```
mlx_lm.generate --help
```

To quantize a model from the command line run:

```
mlx_lm.convert --model mistralai/Mistral-7B-Instruct-v0.3 -q
```

For more options run:

```
mlx_lm.convert --help
```

You can upload new models to Hugging Face by specifying `--upload-repo` to
`convert`. For example, to upload a quantized Mistral-7B model to the
[MLX Hugging Face community](https://huggingface.co/mlx-community) you can do:

```
mlx_lm.convert \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    -q \
    --upload-repo mlx-community/my-4bit-mistral
```

Models can also be converted and quantized directly in the
[mlx-my-repo](https://huggingface.co/spaces/mlx-community/mlx-my-repo) Hugging
Face Space.

### Long Prompts and Generations

`mlx-lm` has some tools to scale efficiently to long prompts and generations:

- A rotating fixed-size key-value cache.
- Prompt caching

To use the rotating key-value cache pass the argument `--max-kv-size n` where
`n` can be any integer. Smaller values like `512` will use very little RAM but
result in worse quality. Larger values like `4096` or higher will use more RAM
but have better quality.

Caching prompts can substantially speedup reusing the same long context with
different queries. To cache a prompt use `mlx_lm.cache_prompt`. For example:

```bash
cat prompt.txt | mlx_lm.cache_prompt \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --prompt - \
  --prompt-cache-file mistral_prompt.safetensors
```

Then use the cached prompt with `mlx_lm.generate`:

```
mlx_lm.generate \
    --prompt-cache-file mistral_prompt.safetensors \
    --prompt "\nSummarize the above text."
```

The cached prompt is treated as a prefix to the supplied prompt. Also notice
when using a cached prompt, the model to use is read from the cache and need
not be supplied explicitly.

Prompt caching can also be used in the Python API in order to avoid
recomputing the prompt. This is useful in multi-turn dialogues or across
requests that use the same context. See the
[example](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/examples/chat.py)
for more usage details.

### Supported Models

`mlx-lm` supports thousands of LLMs available on the Hugging Face Hub. If the
model you want to run is not supported, file an
[issue](https://github.com/ml-explore/mlx-lm/issues/new) or better yet, submit
a pull request. Many supported models are available in various quantization
formats in the [MLX Community](https://huggingface.co/mlx-community) Hugging
Face organization.

For some models the tokenizer may require you to enable the `trust_remote_code`
option. You can do this by passing `--trust-remote-code` in the command line.
If you don't specify the flag explicitly, you will be prompted to trust remote
code in the terminal when running the model.

Tokenizer options can also be set in the Python API. For example:

```python
model, tokenizer = load(
    "qwen/Qwen-7B",
    tokenizer_config={"eos_token": "<|endoftext|>", "trust_remote_code": True},
)
```

### Large Models

> [!NOTE]
    This requires macOS 15.0 or higher to work.

Models which are large relative to the total RAM available on the machine can
be slow. `mlx-lm` will attempt to make them faster by wiring the memory
occupied by the model and cache. This requires macOS 15 or higher to
work.

If you see the following warning message:

> [WARNING] Generating with a model that requires ...

then the model will likely be slow on the given machine. If the model fits in
RAM then it can often be sped up by increasing the system wired memory limit.
To increase the limit, set the following `sysctl`:

```bash
sudo sysctl iogpu.wired_limit_mb=N
```

The value `N` should be larger than the size of the model in megabytes but
smaller than the memory size of the machine.
