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

### Hot cache warm-up

On server startup, the 64 most-recent blocks are pre-loaded from SSD into an
in-memory LRU hot cache in ~52ms. This transforms the first request from cold
SSD I/O (~150 tok/s, 34 sequential reads) to hot memory lookup (~2,400 tok/s).



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

### Prompt Stability (Fix A + Fix B) — Hermes Agent Live Session

**Date:** 2026-04-28 | **Branch:** fix/ssd-caching-prompt-break
**Client:** Hermes Agent v0.11.0, default profile

This benchmark verifies prompt byte-stability and SSD cache observability
in a real multi-turn Hermes conversation across server restarts.

**Run 1 — Cold cache (empty SSD):**

```
PERF: prompt_tok=29623  pref_tok=29623  prefill=85.01s  block_hit=0   chain_break=1
```

**Run 2 — SSD cache hit (fresh server, same prompt):**

```
PERF: prompt_tok=29624  pref_tok=3256   prefill=12.71s  block_hit=103  chain_break=0
```

| Metric | Run 1 (cold) | Run 2 (SSD hit) | Improvement |
|---|---|---|---|
| pref_tok | 29,623 | 3,256 | 89% cached |
| block_hit | 0 | 103 | 103 blocks from SSD |
| chain_break | 1 | 0 | Prompt byte-stable |
| prefill | 85.01s | 12.71s | 6.7x faster |

**Long conversation (8 requests, 29K → 47K tokens):**

| prompt_tok | pref_tok | cached% | chain_break | prefill |
|---|---|---|---|---|
| 29,629 | 3,261 | 89% | 0 | 12.78s |
| 33,019 | 3,360 | 90% | 0 | 13.22s |
| 36,235 | 2,430 | 93% | 0 | 10.83s |
| 39,943 | 3,318 | 92% | 0 | 15.01s |
| 43,166 | 2,845 | 93% | 0 | 13.61s |
| 47,165 | 3,672 | 92% | 0 | 17.83s |

`chain_break=0` on all requests — the system prompt remained byte-stable
throughout a long conversation. Prefill stays flat at ~13-18s despite the
prompt growing from 29K to 47K tokens (only the delta is recomputed).

Full benchmark data: [SSD_CACHE_BENCHMARK_FIX_ABC.md](SSD_CACHE_BENCHMARK_FIX_ABC.md)

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

| Flag | Recommended | Description |
|------|---------|-------------|
| `--kv-bits` | `(8, 4)` | KV cache bit-width: int (uniform) or tuple (key,value) |
| `--kv-group-size` | `(64, 32)` | Quantization group size: int or tuple (key,value) |
| `--kv-boundary-layers` | `2` | Number of boundary KV layers to protect (0=disable) |
| `--kv-boundary-bits` | `(8,8)` | Bit-width for boundary layers |
| `--block-ssd-cache-dir` | `/path/to/cache/dir` | Directory for SSD-persisted block cache |
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
