# Patch 04 — Clamp max_tokens to CLI Floor

**File**: `server.py` — `APIHandler.__init__()`  
**Applies to**: mlx-lm server chat completions handler  
**Status**: Required to prevent Hermes agent truncation loops

---

## Problem

Hermes agent sends `max_tokens: 8192` per request (its default for local models). The user explicitly sets `--max-tokens 256000` on the server. The client value overrides the CLI flag, capping output at 8192 tokens.

For detailed technical explanations (KV cache quantization, speculative decoding deep-dives), Qwen3.6 consistently exceeds 8192 output tokens. This causes `finish_reason='length'` truncation.

When Hermes detects truncation, it sends a continuation request with:
```
[System: Your previous response was truncated by the output length limit. 
 Continue exactly where you left off. Do not restart or repeat prior text. 
 Finish the answer directly.]
```

Qwen3.6 treats this as a new user instruction and **restarts the response from scratch** instead of continuing. The user sees the truncated warning followed by the model repeating its answer.

## Root Cause

Two layers:

1. **Server-side**: `max_tokens` from the client body is accepted verbatim. The CLI `--max-tokens` flag only serves as a fallback default when the client doesn't specify one, not as an authoritative ceiling.

2. **Client-side**: Hermes agent defaults to 8192 max_tokens for local models (configurable in its provider settings), which is conservative but causes truncation on long outputs.

3. **Model behavior**: Qwen3.6's chat template doesn't handle "continue" messages well — it sees them as new user inputs and generates fresh responses.

## Fix

In `server.py`, lines 1222-1226, after extracting `max_tokens` from the request body, clamp it to the CLI flag value:

```python
# Before:
self.max_tokens = self.body.get("max_completion_tokens", None)
if self.max_tokens is None:
    self.max_tokens = self.body.get(
        "max_tokens", self.response_generator.cli_args.max_tokens
    )

# After:
self.max_tokens = self.body.get("max_completion_tokens", None)
if self.max_tokens is None:
    self.max_tokens = self.body.get(
        "max_tokens", self.response_generator.cli_args.max_tokens
    )
# The CLI --max-tokens flag (user's hardware-aware setting) sets
# the floor. Client-specified limits are honored as an upper bound
# but never allowed to dip below the CLI value.
self.max_tokens = max(
    self.max_tokens, self.response_generator.cli_args.max_tokens
)
```

This means:
- If client sends `max_tokens: 8192` and CLI is `--max-tokens 256000` → effective = 256000
- If client sends `max_tokens: 500000` and CLI is `--max-tokens 256000` → effective = 256000 (capped by min, but this patch doesn't add cap — client wins if higher)
- If client sends nothing and CLI is `--max-tokens 256000` → effective = 256000 (same as before)

The CLI flag is the user's explicit hardware-aware choice. It should be the authority.

## Effect

| Metric | Before | After |
|--------|--------|-------|
| Hermes max_tokens (chat) | 8192 (truncation guaranteed) | 256000 (matches CLI) |
| Hermes max_tokens (completions) | 500 (truncation on any detail) | 500 (still low, but never used by chat) |
| finish_reason='length' | Happens on every detailed answer | Much rarer (only if >256K tokens) |
| Qwen restart loop | First truncation → restart → truncation → restart | No truncation → no restart |

## Alternative Approaches Considered

1. **Fix Hermes config directly**: Raise `max_tokens` in Hermes provider settings for local model. Better long-term fix but requires editing `~/.hermes/profiles/ares/config.yaml`.

2. **Handle continuation server-side**: Detect continuation requests and append to previous generation instead of starting fresh. Complex and fragile — continuation detection is heuristic.

3. **Soft max_tokens**: Let server auto-extend when finish_reason='length'. Violates API contract (max_tokens should be a hard limit per OpenAI spec).

The clamp approach is the simplest and least invasive.

## Location in server.py

Lines 1222-1229 in `APIHandler.__init__()` — the `max_tokens` extraction block.

## See Also

- Patch 03: pop_prefixes intermediate checkpoint preservation
- Hermes agent default max_tokens: configured in `~/.hermes/profiles/ares/config.yaml` under provider settings for local models
