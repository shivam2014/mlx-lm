"""
Utilities for extracting per-block slices from KV caches.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx

from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

logger = logging.getLogger(__name__)

# ── Cache type classification ──────────────────────────────────────────────


def get_cache_type_name(cache_obj: Any) -> str:
    """Return the cache type name for an individual cache object."""
    return type(cache_obj).__name__


def classify_cache_layer(cache_obj: Any) -> str:
    """
    Classify a cache layer as sliceable or not.

    Returns:
        'sliceable': Can extract arbitrary token ranges (KVCache, RotatingKVCache)
        'boundary_only': Can only snapshot at block boundaries (ArraysCache)
        'unknown': Cannot determine (treat as boundary_only for safety)
    """
    type_name = get_cache_type_name(cache_obj)
    if type_name in ("KVCache", "RotatingKVCache"):
        return "sliceable"
    if type_name in ("ArraysCache",):
        return "boundary_only"
    return "unknown"


# ── Tensor slicing ─────────────────────────────────────────────────────────


def _slice_kv_cache(cache_obj, start_tok: int, end_tok: int):
    """
    Extract tokens [start_tok:end_tok] from a KVCache-like object.

    KVCache tensors have shape [B, n_kv_heads, seq_len, head_dim].
    We slice along axis=2 (the sequence dimension).

    This avoids KVCache.trim() which only removes from the END of the
    cache by decrementing offset. Direct tensor slicing is correct for
    extracting arbitrary token ranges.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    state = cache_obj.state
    if state is None:
        return None

    # state is (keys, values), each [B, n_kv_heads, seq, head_dim]
    keys, values = state
    sliced_keys = keys[:, :, start_tok:end_tok, :]
    sliced_values = values[:, :, start_tok:end_tok, :]

    new_kv = KVCache.__new__(KVCache)
    new_kv.keys = sliced_keys
    new_kv.values = sliced_values
    new_kv.offset = end_tok - start_tok
    return new_kv


# ── Slicing ────────────────────────────────────────────────────────────────


def slice_cache_for_block(
    cache_data: List[Any],
    block_start_tokens: int,
    block_tokens: int,
) -> List[Any]:
    """
    Extract one block's worth of KV cache from full cache_data.

    For sliceable layers (KVCache, RotatingKVCache): takes
    keys/values[:, :, block_start:block_start+block_tokens, :].

    For boundary_only layers (ArraysCache): takes a full snapshot.
    Only the LAST block for these layers will have valid state.

    Args:
        cache_data: Full per-layer cache list from prefill.
        block_start_tokens: Global token index where this block starts.
        block_tokens: Number of tokens in this block.

    Returns:
        New per-layer cache list containing only this block's KV data.
    """
    result = []
    for layer_cache in cache_data:
        layer_type = classify_cache_layer(layer_cache)

        if layer_type == "sliceable":
            end_tok = block_start_tokens + block_tokens
            sliced = _slice_kv_cache(layer_cache, block_start_tokens, end_tok)
            result.append(sliced)

        elif layer_type == "boundary_only":
            # Boundary-only layers (e.g. ArraysCache) don't need mutation
            # safety: their state is snapshot-based and won't be mutated
            # elsewhere. Skip deep copy to avoid ~10,800 copies during prefill.
            result.append(layer_cache)

        else:
            # Unknown type — be conservative but avoid deep copy.
            # Shallow copy is sufficient since these layers don't share
            # mutable state that would be mutated elsewhere.
            result.append(copy.copy(layer_cache))

    return result


def get_cache_layer_info(cache_data: List[Any]) -> Dict[str, Any]:
    """
    Extract layer type information from a cache list.

    Returns dict with:
        num_layers: int
        layer_cache_types: list[str]
        has_boundary_only_layers: bool
    """
    return {
        "num_layers": len(cache_data),
        "layer_cache_types": [get_cache_type_name(c) for c in cache_data],
        "has_boundary_only_layers": any(
            classify_cache_layer(c) in ("boundary_only", "unknown")
            for c in cache_data
        ),
    }
