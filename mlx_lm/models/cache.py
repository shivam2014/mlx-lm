# Copyright © 2023-2024 Apple Inc.

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_reduce, tree_unflatten

from .base import create_causal_mask

import dataclasses
import hashlib
import json
import logging
import pickle
import threading
from pathlib import Path

from mlx_lm.models.block_ssd_cache import (
    BLOCK_SIZE,
    BlockSSDCache,
    compute_block_hash,
    compute_content_hash,
    DEFAULT_ROOT_HASH,
)

logger = logging.getLogger(__name__)


def make_prompt_cache(
    model: nn.Module,
    max_kv_size: Optional[int] = None,
) -> List[Any]:
    """
    Construct the model's cache for use in generation.

    This function will defer the cache construction to the model if it has a
    ``make_cache`` method, otherwise it will make a default KV cache.

    Args:
        model (nn.Module): The language model.
        max_kv_size (Optional[int]): If provided and the model does not have a
            ``make_cache`` method, a ``RotatingKVCache`` is used with a maximum
            size of ``max_kv_size``
    """
    if hasattr(model, "make_cache"):
        return model.make_cache()

    num_layers = len(model.layers)
    if max_kv_size is not None:
        return [
            RotatingKVCache(max_size=max_kv_size, keep=4) for _ in range(num_layers)
        ]
    else:
        return [KVCache() for _ in range(num_layers)]


def save_prompt_cache(file_name: str, cache: List[Any], metadata: Dict[str, str] = {}):
    """
    Save a pre-computed prompt cache to a file.

    Args:
        file_name (str): The ``.safetensors`` file name.
        cache (List[Any]): The model state.
        metadata (Dict[str, str]): Optional metadata to save along with model
            state.
    """
    cache_data = [c.state for c in cache]
    cache_info = [c.meta_state for c in cache]
    cache_data = dict(tree_flatten(cache_data))
    cache_classes = [type(c).__name__ for c in cache]
    cache_metadata = [cache_info, metadata, cache_classes]
    cache_metadata = dict(tree_flatten(cache_metadata))
    mx.save_safetensors(file_name, cache_data, cache_metadata)


def load_prompt_cache(file_name, return_metadata=False):
    """
    Load a prompt cache from a file.

    Args:
        file_name (str): The ``.safetensors`` file name.
        return_metadata (bool): Whether or not to return metadata.
            Default: ``False``.

    Returns:
        List[Any] or Tuple[List[Any], Dict[str, str]]: The prompt cache and
            the metadata if requested.
    """
    arrays, cache_metadata = mx.load(file_name, return_metadata=True)
    arrays = tree_unflatten(list(arrays.items()))
    cache_metadata = tree_unflatten(list(cache_metadata.items()))
    info, metadata, classes = cache_metadata
    cache = [
        globals()[c].from_state(state, meta_state)
        for c, state, meta_state in zip(classes, arrays, info)
    ]
    if return_metadata:
        return cache, metadata
    return cache


def can_trim_prompt_cache(cache: List[Any]) -> bool:
    """
    Check if model's cache can be trimmed.
    """
    return all(c.is_trimmable() for c in cache)


def trim_prompt_cache(cache: List[Any], num_tokens: int) -> List[Any]:
    """
    Trim the model's cache by the given number of tokens.

    This function will trim the cache if possible (in-place) and return the
    number of tokens that were trimmed.

    Args:
        cache (List[Any]): The model's cache.
        num_tokens (int): The number of tokens to trim.

    Returns:
        (int): The number of tokens that were trimmed.
    """
    if not can_trim_prompt_cache(cache) or len(cache) == 0:
        return 0
    return [c.trim(num_tokens) for c in cache][0]


def create_attention_mask(
    N: int, offset: int, return_array: bool, window_size: Optional[int]
):
    if window_size is not None:
        return create_causal_mask(N, offset, window_size=window_size)
    elif N == 1:
        return None
    elif return_array:
        return create_causal_mask(N, offset, window_size=window_size)
    else:
        return "causal"


class _BaseCache:
    @property
    def state(self):
        return []

    @state.setter
    def state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no state but a state was set.")

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no meta_state but a meta_state was set.")

    def is_trimmable(self):
        return False

    def size(self):
        """
        Return the size (i.e. sequence length) of the cache.

        Not every cache is required to implement this, in which case the size
        will always be 0 (though the cache may not be empty).
        """
        return 0

    @property
    def nbytes(self):
        """Return the size of this cache in bytes"""
        raise NotImplementedError("Cache sub-class must implement nbytes")

    def empty(self):
        """
        Return if the cache is empty or not.
        """
        raise NotImplementedError("Cache sub-class must implement this.")

    @classmethod
    def from_state(cls, state, meta_state):
        # Create an instance of cls without calling __init__
        obj = cls.__new__(cls)
        obj.state = state
        obj.meta_state = meta_state
        return obj


class ConcatenateKVCache(_BaseCache):
    """ConcatenateKVCache the simplest KV cache implementation.

    Can be used as a mock KV cache or when large blocks are being processed at
    a time in which case KVCache isn't necessarily faster. Consider using the
    KVCache with a larger step size before using this cache.
    """

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=-2)
            self.values = mx.concatenate([self.values, values], axis=-2)
        self.offset = self.keys.shape[-2]

        return self.keys, self.values

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[-2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class QuantizedKVCache(_BaseCache):
    step = 256

    def __init__(self, group_size: int = 64, bits: int = 8):
        self.keys = None
        self.values = None
        self.offset = 0
        # Support KVSplit: bits/group_size may be (key, value) tuples
        if isinstance(bits, (tuple, list)):
            self.key_bits, self.value_bits = bits
        else:
            self.key_bits = self.value_bits = bits
        if isinstance(group_size, (tuple, list)):
            self.key_group_size, self.value_group_size = group_size
        else:
            self.key_group_size = self.value_group_size = group_size

    # Backward-compatible properties
    @property
    def bits(self):
        if self.key_bits == self.value_bits:
            return self.key_bits
        return (self.key_bits, self.value_bits)

    @bits.setter
    def bits(self, v):
        if isinstance(v, (tuple, list)):
            self.key_bits, self.value_bits = v
        else:
            self.key_bits = self.value_bits = v

    @property
    def group_size(self):
        if self.key_group_size == self.value_group_size:
            return self.key_group_size
        return (self.key_group_size, self.value_group_size)

    @group_size.setter
    def group_size(self, v):
        if isinstance(v, (tuple, list)):
            self.key_group_size, self.value_group_size = v
        else:
            self.key_group_size = self.value_group_size = v

    def update_and_fetch(self, keys, values):
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self.offset

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def init_quant(dim, bits, gs):
                el_per_int = 8 * mx.uint32.size // bits
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // gs), dtype=keys.dtype),
                    mx.zeros((*shape, dim // gs), dtype=keys.dtype),
                )

            def expand_quant(x):
                new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=-2)

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys, self.values = tree_map(
                        lambda x: x[..., :prev, :], (self.keys, self.values)
                    )

                self.keys, self.values = tree_map(
                    expand_quant, (self.keys, self.values)
                )
            else:
                self.keys = init_quant(k_head_dim, self.key_bits, self.key_group_size)
                self.values = init_quant(
                    v_head_dim, self.value_bits, self.value_group_size
                )

        self.offset += num_steps

        keys = mx.quantize(
            keys, group_size=self.key_group_size, bits=self.key_bits
        )
        values = mx.quantize(
            values, group_size=self.value_group_size, bits=self.value_bits
        )
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self.offset, :] = keys[i]
            self.values[i][..., prev : self.offset, :] = values[i]

        return tree_map(lambda x: x[..., : self.offset, :], (self.keys, self.values))

    @property
    def state(self):
        if self.offset == self.keys[0].shape[2]:
            return self.keys, self.values
        else:
            return tree_map(
                lambda x: x[..., : self.offset, :], (self.keys, self.values)
            )

    @state.setter
    def state(self, v):
        self.keys, self.values = v

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    self.offset,
                    self.key_group_size,
                    self.key_bits,
                    self.value_group_size,
                    self.value_bits,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        if len(v) == 3:
            # Backward compat: old format (offset, group_size, bits)
            self.offset = int(v[0])
            self.key_group_size = self.value_group_size = int(v[1])
            self.key_bits = self.value_bits = int(v[2])
        else:
            (
                self.offset,
                self.key_group_size,
                self.key_bits,
                self.value_group_size,
                self.value_bits,
            ) = map(int, v)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)


class KVCache(_BaseCache):
    step = 256

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def size(self):
        return self.offset

    @property
    def state(self):
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        else:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def to_quantized(self, group_size=64, bits=4) -> QuantizedKVCache:
        quant_cache = QuantizedKVCache(group_size=group_size, bits=bits)
        quant_cache.offset = self.offset
        if self.keys is not None:
            quant_cache.keys = mx.quantize(
                self.keys,
                group_size=quant_cache.key_group_size,
                bits=quant_cache.key_bits,
            )
            quant_cache.values = mx.quantize(
                self.values,
                group_size=quant_cache.value_group_size,
                bits=quant_cache.value_bits,
            )
        return quant_cache

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    @classmethod
    def merge(_, caches):
        return BatchKVCache.merge(caches)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class RotatingKVCache(_BaseCache):
    step = 256

    def __init__(self, max_size, keep=0):
        self.keep = keep
        self.keys = None
        self.values = None
        self.offset = 0
        self.max_size = max_size
        self._idx = 0

    def _trim(self, trim_size, v, append=None):
        to_cat = []
        if trim_size > 0:
            to_cat = [v[..., : self.keep, :], v[..., trim_size + self.keep :, :]]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def _temporal_order(self, v):
        """
        Rearrange the cache into temporal order, slicing off the end if unused.
        """
        if self._idx == v.shape[2]:
            return v
        elif self._idx < self.offset:
            return mx.concatenate(
                [
                    v[..., : self.keep, :],
                    v[..., self._idx :, :],
                    v[..., self.keep : self._idx, :],
                ],
                axis=2,
            )
        else:
            return v[..., : self._idx, :]

    def _update_concat(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            # Put the keys/values in temporal order to
            # preserve context
            self.keys = self._temporal_order(self.keys)
            self.values = self._temporal_order(self.values)
            self._idx = self.keys.shape[2]

            # The largest size is self.max_size + S - 1 to ensure
            # every token gets at least self.max_size context
            trim_size = self._idx - self.max_size + 1
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)
        self.offset += keys.shape[2]
        self._idx = self.keys.shape[2]
        return self.keys, self.values

    def _update_in_place(self, keys, values):
        # May not have hit the max size yet, so potentially
        # keep growing the cache
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self.offset
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        # Trim if needed
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size

        # Rotate
        if self._idx == self.max_size:
            self._idx = self.keep

        # Assign
        self.keys[..., self._idx : self._idx + S, :] = keys
        self.values[..., self._idx : self._idx + S, :] = values
        self.offset += S
        self._idx += S

        # If the buffer is not full, slice off the end
        if self.offset < self.max_size:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    def update_and_fetch(self, keys, values):
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    def size(self):
        return min(self.offset, self.max_size)

    @property
    def state(self):
        if self.offset < self.keys.shape[2]:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        else:
            return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v

    @property
    def meta_state(self):
        return tuple(map(str, (self.keep, self.max_size, self.offset, self._idx)))

    @meta_state.setter
    def meta_state(self, v):
        self.keep, self.max_size, self.offset, self._idx = map(
            int,
            v,
        )

    def is_trimmable(self):
        return self.offset < self.max_size

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        self._idx -= n
        return n

    def to_quantized(
        self, group_size: int = 64, bits: int = 4
    ) -> "QuantizedRotatingKVCache":
        qcache = QuantizedRotatingKVCache(
            max_size=self.max_size, keep=self.keep, group_size=group_size, bits=bits
        )
        qcache.offset = self.offset
        qcache._idx = self._idx
        if self.keys is not None:
            S = self.keys.shape[2]
            qcache.keys = list(
                mx.quantize(
                    self.keys,
                    group_size=qcache.key_group_size,
                    bits=qcache.key_bits,
                )
            )
            qcache.values = list(
                mx.quantize(
                    self.values,
                    group_size=qcache.value_group_size,
                    bits=qcache.value_bits,
                )
            )
        return qcache

    def make_mask(
        self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        if N > 1:
            window_size = window_size or self.max_size
            offset = min(self.max_size - 1, self.offset)
            if offset + N > window_size or return_array:
                return create_causal_mask(N, offset, window_size=window_size)
            else:
                return "causal"
        else:
            if window_size is None:
                return None
            # May need a mask for when window_size < max_size
            if self.offset >= window_size and self.max_size > window_size:
                idx = self._idx
                if idx >= self.max_size:
                    idx = 0
                if self.offset < self.max_size:
                    mask_size = self.offset + 1
                else:
                    mask_size = self.max_size
                mask = mx.arange(mask_size) >= (mask_size - window_size)
                mask = mx.roll(mask, shift=idx + 1)
                return mask

    @classmethod
    def merge(_, caches):
        return BatchRotatingKVCache.merge(caches)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class QuantizedRotatingKVCache(RotatingKVCache):
    """A rotating KV cache that stores quantized (data, scales, biases) tuples.

    Supports KVSplit: bits/group_size may be (key, value) tuples for
    asymmetric quantization of keys and values.
    """

    step = 256

    def __init__(self, max_size, keep=0, group_size: int = 64, bits: int = 8):
        super().__init__(max_size=max_size, keep=keep)
        # Support KVSplit: bits/group_size may be (key, value) tuples
        if isinstance(bits, (tuple, list)):
            self.key_bits, self.value_bits = bits
        else:
            self.key_bits = self.value_bits = bits
        if isinstance(group_size, (tuple, list)):
            self.key_group_size, self.value_group_size = group_size
        else:
            self.key_group_size = self.value_group_size = group_size

    # Backward-compatible properties
    @property
    def bits(self):
        if self.key_bits == self.value_bits:
            return self.key_bits
        return (self.key_bits, self.value_bits)

    @bits.setter
    def bits(self, v):
        if isinstance(v, (tuple, list)):
            self.key_bits, self.value_bits = v
        else:
            self.key_bits = self.value_bits = v

    @property
    def group_size(self):
        if self.key_group_size == self.value_group_size:
            return self.key_group_size
        return (self.key_group_size, self.value_group_size)

    @group_size.setter
    def group_size(self, v):
        if isinstance(v, (tuple, list)):
            self.key_group_size, self.value_group_size = v
        else:
            self.key_group_size = self.value_group_size = v

    @staticmethod
    def _is_quantized(v):
        """Check if a value is a quantized tuple/list (data, scales, biases)."""
        return isinstance(v, (tuple, list))

    def _seq_len(self):
        """Return the current sequence length in the buffer."""
        if self.keys is None:
            return 0
        if self._is_quantized(self.keys):
            return self.keys[0].shape[2]
        return self.keys.shape[2]

    def _trim(self, trim_size, v, append=None):
        if self._is_quantized(v):
            parts = []
            if trim_size > 0:
                parts = [
                    tree_map(lambda x: x[..., : self.keep, :], v),
                    tree_map(lambda x: x[..., trim_size + self.keep :, :], v),
                ]
            else:
                parts = [v]
            if append is not None:
                parts.append(append)
            if len(parts) == 1:
                return parts[0]
            result = tree_map(
                lambda *arrs: mx.concatenate(arrs, axis=2), *parts
            )
            return result
        # Unquantized fallback (should not happen after first quantize)
        return super()._trim(trim_size, v, append)

    def _temporal_order(self, v):
        """Rearrange the cache into temporal order, slicing off the end if unused."""
        if self._is_quantized(v):
            seq_len = v[0].shape[2]
            if self._idx == seq_len:
                return v
            elif self._idx < self.offset:
                return tree_map(
                    lambda x: mx.concatenate(
                        [
                            x[..., : self.keep, :],
                            x[..., self._idx :, :],
                            x[..., self.keep : self._idx, :],
                        ],
                        axis=2,
                    ),
                    v,
                )
            else:
                return tree_map(lambda x: x[..., : self._idx, :], v)
        return super()._temporal_order(v)

    def _update_concat(self, keys, values):
        # Quantize the incoming keys/values
        q_keys = list(
            mx.quantize(keys, group_size=self.key_group_size, bits=self.key_bits)
        )
        q_values = list(
            mx.quantize(
                values, group_size=self.value_group_size, bits=self.value_bits
            )
        )

        if self.keys is None:
            self.keys = q_keys
            self.values = q_values
        else:
            self.keys = self._temporal_order(self.keys)
            self.values = self._temporal_order(self.values)
            self._idx = self._seq_len()

            trim_size = self._idx - self.max_size + 1
            self.keys = self._trim(trim_size, self.keys, q_keys)
            self.values = self._trim(trim_size, self.values, q_values)

        self.offset += keys.shape[2]
        self._idx = self._seq_len()
        return self.keys, self.values

    def _update_in_place(self, keys, values):
        B, n_kv_heads, S, k_head_dim = keys.shape
        v_head_dim = values.shape[3]
        prev = self.offset

        if self.keys is None or (
            prev >= self._seq_len() and self._seq_len() < self.max_size
        ):
            new_size = min(self.step, self.max_size - prev)
            shape = (B, n_kv_heads, new_size)

            def _init_quant(dim, bits, gs):
                el_per_int = 8 * mx.uint32.size // bits
                return [
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // gs), dtype=keys.dtype),
                    mx.zeros((*shape, dim // gs), dtype=keys.dtype),
                ]

            if self.keys is not None:
                new_k = _init_quant(k_head_dim, self.key_bits, self.key_group_size)
                new_v = _init_quant(
                    v_head_dim, self.value_bits, self.value_group_size
                )
                self.keys = tree_map(
                    lambda a, b: mx.concatenate([a, b], axis=2),
                    self.keys,
                    new_k,
                )
                self.values = tree_map(
                    lambda a, b: mx.concatenate([a, b], axis=2),
                    self.values,
                    new_v,
                )
            else:
                self.keys = _init_quant(
                    k_head_dim, self.key_bits, self.key_group_size
                )
                self.values = _init_quant(
                    v_head_dim, self.value_bits, self.value_group_size
                )
            self._idx = prev

        # Trim if needed
        trim_size = self._seq_len() - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size

        # Rotate
        if self._idx == self.max_size:
            self._idx = self.keep

        # Quantize and assign
        q_keys = mx.quantize(
            keys, group_size=self.key_group_size, bits=self.key_bits
        )
        q_values = mx.quantize(
            values, group_size=self.value_group_size, bits=self.value_bits
        )
        for i in range(len(self.keys)):
            self.keys[i][..., self._idx : self._idx + S, :] = q_keys[i]
            self.values[i][..., self._idx : self._idx + S, :] = q_values[i]

        self.offset += S
        self._idx += S

        # If the buffer is not full, slice off the end
        if self.offset < self.max_size:
            return (
                tree_map(lambda x: x[..., : self.offset, :], self.keys),
                tree_map(lambda x: x[..., : self.offset, :], self.values),
            )
        return self.keys, self.values

    @property
    def state(self):
        if self.keys is None:
            return []
        seq_len = self._seq_len()
        if self.offset < seq_len:
            return (
                tree_map(lambda x: x[..., : self.offset, :], self.keys),
                tree_map(lambda x: x[..., : self.offset, :], self.values),
            )
        return self.keys, self.values

    @state.setter
    def state(self, v):
        if v is not None and v:
            self.keys, self.values = v
        else:
            self.keys = None
            self.values = None

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    self.keep,
                    self.max_size,
                    self.offset,
                    self._idx,
                    self.key_group_size,
                    self.key_bits,
                    self.value_group_size,
                    self.value_bits,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        if len(v) == 4:
            # Backward compat: old RotatingKVCache format
            self.keep, self.max_size, self.offset, self._idx = map(int, v)
        else:
            (
                self.keep,
                self.max_size,
                self.offset,
                self._idx,
                self.key_group_size,
                self.key_bits,
                self.value_group_size,
                self.value_bits,
            ) = map(int, v)

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)

    def to_quantized(
        self, group_size: int = 64, bits: int = 4
    ) -> "QuantizedRotatingKVCache":
        return self


class ArraysCache(_BaseCache):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.left_padding = None
        instance.lengths = None
        return instance

    def __init__(self, size, left_padding: Optional[List[int]] = None):
        self.cache = [None] * size
        if left_padding:
            self.left_padding = mx.array(left_padding)

    @property
    def batch_size(self):
        for c in self.cache:
            if c is not None:
                return c.shape[0]
        if self.left_padding is not None:
            return self.left_padding.size
        elif self.lengths is not None:
            return self.lengths.size
        else:
            return 1

    def __setitem__(self, idx, value):
        self.cache[idx] = value

    def __getitem__(self, idx):
        return self.cache[idx]

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, v):
        self.cache = v

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        self.cache = [c[batch_indices] if c is not None else None for c in self.cache]
        if self.left_padding is not None:
            self.left_padding = self.left_padding[batch_indices]
        if self.lengths is not None:
            self.lengths = self.lengths[batch_indices]

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """

        a_batch = self.batch_size
        b_batch = other.batch_size

        def cat(a, b):
            shape = dtype = None
            if a is not None:
                shape = a.shape
                dtype = a.dtype
            if b is not None:
                shape = b.shape
                dtype = b.dtype

            if shape is None:
                return None

            if a is None:
                a = mx.zeros((a_batch,) + shape[1:], dtype=dtype)
            if b is None:
                b = mx.zeros((b_batch,) + shape[1:], dtype=dtype)

            return mx.concatenate([a, b])

        self.cache = [cat(c, o) for c, o in zip(self.cache, other.cache)]
        self.left_padding = cat(self.left_padding, other.left_padding)
        self.lengths = cat(self.lengths, other.lengths)

    def extract(self, idx):
        cache = ArraysCache(len(self.cache))
        cache.cache = [c[idx : idx + 1] for c in self.cache]
        return cache

    def prepare(self, lengths=None, **kwargs):
        self.lengths = mx.array(lengths)

    def finalize(self):
        self.lengths = None
        self.left_padding = None

    def advance(self, N):
        if self.lengths is not None:
            self.lengths -= N
        if self.left_padding is not None:
            self.left_padding -= N

    def make_mask(self, N: int):
        if self.left_padding is not None:
            pos = mx.arange(N)
            return pos >= self.left_padding[:, None]
        elif self.lengths is not None:
            pos = mx.arange(N)
            return pos < self.lengths[:, None]
        else:
            return None

    @classmethod
    def merge(cls, caches):
        n_state = len(caches[0].cache)
        B = len(caches)
        cache = cls(n_state)

        # All caches are empty so return early
        if all(c.empty() for c in caches):
            cache.left_padding = mx.array([0] * B)
            return cache

        for e in range(n_state):
            c_init = next(iter(c[e] for c in caches if c[e] is not None))
            shape = list(c_init.shape)
            shape[0] = B
            cache[e] = mx.zeros(shape, c_init.dtype)
            for i in range(B):
                if caches[i][e] is None:
                    continue
                cache[e][i : i + 1] = caches[i][e]
        return cache

    def is_trimmable(self):
        return True

    def trim(self, n):
        # ArraysCache stores static per-layer image features with no
        # token-position offset to rewind. Report success without mutation.
        return n

    def empty(self):
        return self.cache[0] is None

    @property
    def nbytes(self):
        return sum(c.nbytes for c in self.cache if c is not None)


class ChunkedKVCache(_BaseCache):
    step = 256

    def __init__(self, chunk_size):
        self.keys = None
        self.values = None
        self.offset = 0
        self.chunk_size = chunk_size
        self.start_position = 0

    def maybe_trim_front(self):
        # Maintain the cache below the chunk size
        if self.keys is not None and self.keys.shape[2] >= self.chunk_size:
            self.start_position += self.keys.shape[2] - self.chunk_size
            self.keys = self.keys[..., -self.chunk_size :, :]
            self.values = self.values[..., -self.chunk_size :, :]

    def update_and_fetch(self, keys, values):
        prev = self.offset - self.start_position
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        end = self.offset - self.start_position
        self.keys[..., prev:end, :] = keys
        self.values[..., prev:end, :] = values
        return self.keys[..., :end, :], self.values[..., :end, :]

    @property
    def state(self):
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        else:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset - self.start_position, n)
        self.offset -= n
        return n

    @property
    def meta_state(self):
        return tuple(map(str, (self.chunk_size, self.start_position)))

    @meta_state.setter
    def meta_state(self, v):
        self.chunk_size, self.start_position = map(int, v)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class CacheList(_BaseCache):
    def __init__(self, *caches):
        self.caches = caches

    def __getitem__(self, idx):
        return self.caches[idx]

    def is_trimmable(self):
        return all(c.is_trimmable() for c in self.caches)

    def trim(self, n):
        for c in self.caches:
            m = c.trim(n)
        return m

    @property
    def state(self):
        return [c.state for c in self.caches]

    @state.setter
    def state(self, v):
        for c, s in zip(self.caches, v):
            c.state = s

    @property
    def meta_state(self):
        return (
            [type(c).__name__ for c in self.caches],
            [c.meta_state for c in self.caches],
        )

    @meta_state.setter
    def meta_state(self, v):
        for c, m in zip(self.caches, v[1]):
            c.meta_state = m

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        for c in self.caches:
            c.filter(batch_indices)

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        for c, o in zip(self.caches, other.caches):
            c.extend(o)

    @classmethod
    def merge(cls, caches):
        cache = cls()
        cache.caches = tuple(
            caches[0].caches[i].merge([c.caches[i] for c in caches])
            for i in range(len(caches[0].caches))
        )
        return cache

    def extract(self, idx):
        return CacheList(*(c.extract(idx) for c in self.caches))

    def prepare(self, **kwargs):
        for c in self.caches:
            c.prepare(**kwargs)

    def finalize(self):
        for c in self.caches:
            c.finalize()

    def size(self):
        return max(c.size() for c in self.caches)

    def empty(self):
        return self.caches[0].empty()

    @property
    def nbytes(self):
        return sum(c.nbytes for c in self.caches)

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls.__new__(cls)
        obj.caches = [
            globals()[c].from_state(s, m) for s, c, m in zip(state, *meta_state)
        ]
        return obj


def dynamic_roll(x, shifts, axis):
    n = x.shape[axis]
    expand_shifts = (...,) + (None,) * (x.ndim - axis)
    expand_indices = expand_shifts[:-1]
    idx = (mx.arange(n)[expand_indices] - shifts[expand_shifts]) % n
    rolled = mx.take_along_axis(x, idx, axis=axis)
    return rolled


class BatchKVCache(_BaseCache):
    step = 256

    def __init__(self, left_padding: List[int]):
        """
        The BatchKV cache expects inputs to be left-padded.

        E.g. the following prompts:

            [1, 3, 5]
            [7]
            [2, 6, 8, 9]

        Should be padded like so:

            [0, 1, 3, 5]
            [0, 0, 0, 7]
            [2, 6, 8, 9]

        And ``left_padding`` specifies the amount of padding for each.
        In this case, ``left_padding = [1, 3, 0]``.
        """
        self.keys = None
        self.values = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])
        self._idx = 0

        self._right_padding = None

    def update_and_fetch(self, keys, values):
        prev = self._idx
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self._idx += keys.shape[2]
        self.keys[..., prev : self._idx, :] = keys
        self.values[..., prev : self._idx, :] = values
        return self.keys[..., : self._idx, :], self.values[..., : self._idx, :]

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None:
            padding = self._right_padding
            self.keys = dynamic_roll(self.keys, padding[:, None], axis=2)
            self.values = dynamic_roll(self.values, padding[:, None], axis=2)
            self.offset -= padding
            self.left_padding += padding
            self._right_padding = None

    @property
    def state(self):
        k, v = self.keys, self.values
        if self._idx < k.shape[2]:
            k = k[..., : self._idx, :]
            v = v[..., : self._idx, :]
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding = v
        self._idx = self.keys.shape[2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        return n

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        if self.keys is not None:
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        # Shift left to reduce padding
        min_left_pad = self.left_padding.min().item()
        if min_left_pad > 0:
            if self.keys is not None:
                self.keys = self.keys[..., min_left_pad:, :]
                self.values = self.values[..., min_left_pad:, :]
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self._idx, other._idx)
        L1 = L2 = 0
        if self.keys is not None:
            B, H, L1, D = self.keys.shape
            M = self.values.shape[3]
        if other.keys is not None:
            B, H, L2, D = other.keys.shape
            M = other.values.shape[3]
        max_size = max(L1, L2)

        # Pad the keys and values so they are right-justified
        # with the index and the same size
        def pad(c):
            k, v = c.keys, c.values
            if k is None:
                Bc = c.offset.shape[0]
                k = mx.array([]).reshape(Bc, H, 0, D)
                v = mx.array([]).reshape(Bc, H, 0, M)
            left = max_idx - c._idx
            right = max_size - k.shape[2] - left
            if right < 0:
                k = k[..., :right, :]
                v = v[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                pad = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = mx.pad(k, pad)
                v = mx.pad(v, pad)
            left_padding = c.left_padding + left
            return k, v, c.offset, left_padding

        self.keys, self.values, self.offset, self.left_padding = map(
            mx.concatenate, zip(*(pad(self), pad(other)))
        )
        self._idx = max_idx

    def extract(self, idx):
        cache = KVCache()
        padding = self.left_padding[idx].item()
        cache.keys = mx.contiguous(self.keys[idx : idx + 1, :, padding : self._idx])
        cache.values = mx.contiguous(self.values[idx : idx + 1, :, padding : self._idx])
        cache.offset = cache.keys.shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        lengths = [c.size() for c in caches]
        max_length = max(lengths)

        # No cache has content so make an empty one
        if max_length == 0:
            return BatchKVCache([0] * len(caches))

        padding = [max_length - l for l in lengths]
        B = len(caches)
        H = max(c.keys.shape[1] for c in caches if c.keys is not None)
        Dk = max(c.keys.shape[3] for c in caches if c.keys is not None)
        Dv = max(c.values.shape[3] for c in caches if c.values is not None)
        dt = next(iter(c.keys.dtype for c in caches if c.keys is not None))

        keys = mx.zeros((B, H, max_length, Dk), dtype=dt)
        values = mx.zeros((B, H, max_length, Dv), dtype=dt)
        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.keys is None:
                continue
            keys[i : i + 1, :, p : p + c.offset] = c.keys[..., : c.offset, :]
            values[i : i + 1, :, p : p + c.offset] = c.values[..., : c.offset, :]

        cache = cls(padding)
        cache.keys = keys
        cache.values = values
        cache.offset += keys.shape[2]
        cache._idx = keys.shape[2]

        return cache

    def size(self):
        return self._idx

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class BatchRotatingKVCache(_BaseCache):
    step = 256

    def __init__(self, max_size, left_padding: List[int]):
        self.keys = None
        self.values = None

        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])

        self.max_size = max_size
        self._idx = 0
        self._offset = 0
        self.rotated = False

        # Lengths for right_padded inputs to make sure that padding tokens do
        # not evict valid tokens.
        self._lengths = None

    def _trim(self, trim_size, v, append=None):
        if trim_size > 0:
            v = v[..., trim_size:, :]
        if append is not None:
            return mx.concatenate([v, append], axis=2)
        return v

    def _temporal_order(self):
        """
        Rearrange the cache into temporal order.
        """
        if self.rotated:
            self.keys = mx.roll(self.keys, -self._idx, axis=2)
            self.values = mx.roll(self.values, -self._idx, axis=2)
            self._idx = self.keys.shape[2]
            self.rotated = False

    def _update_concat(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            # Put the keys/values in temporal order to
            # preserve context
            self._temporal_order()

            # Slice off the end if needed
            if self.keys.shape[2] > self._idx:
                self.keys = self.keys[..., : self._idx, :]
                self.values = self.values[..., : self._idx, :]

            # Roll right sequences that are padded to make sure that we don't
            # trim valid cache entries
            if self._lengths is not None:
                roll = mx.maximum(0, self.offset - self._lengths)
                self.keys = dynamic_roll(self.keys, roll[:, None], axis=2)
                self.values = dynamic_roll(self.values, roll[:, None], axis=2)
                self.left_padding += roll
                self.offset -= roll

            # The largest size is self.max_size + S - 1 to ensure
            # every token gets at least self.max_size context
            trim_size = self._idx - self.max_size + 1
            if trim_size > 0:
                self.left_padding -= trim_size
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)
        self.offset += keys.shape[2]
        self._offset += keys.shape[2]
        self._idx = self.keys.shape[2]

        # Make sure left_padding and offset are evaluated
        self.keys = mx.depends(self.keys, (self.left_padding, self.offset))

        return self.keys, self.values

    def _update_in_place(self, keys, values):
        if self._lengths is not None:
            raise RuntimeError(
                "finalize() should be called before deocoding with BatchRotatingKVCache"
            )

        # May not have hit the max size yet, so potentially
        # keep growing the cache
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self._offset
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        # Trim if needed
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size
            self.left_padding -= trim_size

        # Rotate
        if self._idx == self.max_size:
            self.rotated = True
            self._idx = 0
        if self.rotated:
            self.left_padding -= S

        # Assign
        self.keys[..., self._idx : self._idx + S, :] = keys
        self.values[..., self._idx : self._idx + S, :] = values
        self._offset += S
        self.offset += S
        self._idx += S

        # Make sure left_padding and offset are evaluated
        self.keys = mx.depends(self.keys, (self.left_padding, self.offset))

        # If the buffer is not full, slice off the end
        if self._offset < self.max_size:
            return (
                self.keys[..., : self._offset, :],
                self.values[..., : self._offset, :],
            )
        return self.keys, self.values

    def update_and_fetch(self, keys, values):
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchRotatingKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._lengths = mx.array(lengths) + self.offset

    def finalize(self):
        if self._lengths is not None:
            roll = mx.maximum(0, self.offset - self._lengths)
            self.keys = dynamic_roll(self.keys, roll[:, None], axis=2)
            self.values = dynamic_roll(self.values, roll[:, None], axis=2)
            self.left_padding += roll
            self.offset -= roll
            self._lengths = None

    @property
    def state(self):
        k, v = self.keys, self.values
        if self._offset < k.shape[2]:
            k, v = k[..., : self._offset, :], v[..., : self._offset, :]
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding = v

    @property
    def meta_state(self):
        return tuple(map(str, (self.max_size, self._offset, self._idx, self.rotated)))

    @meta_state.setter
    def meta_state(self, v):
        self.max_size, self._offset, self._idx = map(
            int,
            v[:3],
        )
        self.rotated = bool(v[3])

    def is_trimmable(self):
        return self._offset < self.max_size

    def trim(self, n):
        n = min(self._offset, n)
        self._offset -= n
        self._idx -= n
        self.offset -= n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
        raise NotImplementedError("BatchRotatingKVCache Quantization NYI")

    def make_mask(
        self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        left_padding = self.left_padding
        window_size = window_size or self.max_size
        offset = min(self.max_size - 1, self._offset)
        rinds = mx.arange(offset + N)
        linds = mx.arange(offset, offset + N) if offset else rinds
        linds = linds[:, None]
        rinds = rinds[None]
        mask = linds >= rinds
        mask &= linds < rinds + window_size
        if (trim_size := self._idx - self.max_size + int(N > 1)) > 0:
            left_padding = left_padding - trim_size

        rotated = N == 1 and (self.rotated or self._idx >= self.max_size)
        if rotated:
            left_padding = left_padding - 1

        mask = mask & (rinds >= mx.expand_dims(left_padding, (1, 2, 3)))

        if rotated:
            idx = self._idx
            if idx >= self.max_size:
                idx = 0
            mask = mx.roll(mask, shift=idx + 1, axis=-1)

        return mask

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        if self.keys is not None:
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        if (self.rotated != other.rotated) or self._idx != other._idx:
            self._temporal_order()
            other._temporal_order()

        max_idx = max(self._idx, other._idx)
        L1 = L2 = 0
        if self.keys is not None:
            B, H, L1, D = self.keys.shape
            M = self.values.shape[3]
        if other.keys is not None:
            B, H, L2, D = other.keys.shape
            M = other.values.shape[3]
        max_size = max(L1, L2)

        def pad(c):
            left = max_idx - c._idx
            k, v = c.keys, c.values
            if k is None:
                Bc = c.offset.shape[0]
                k = mx.array([]).reshape(Bc, H, 0, D)
                v = mx.array([]).reshape(Bc, H, 0, M)
            right = max_size - k.shape[2] - left
            if right < 0:
                k = k[..., :right, :]
                v = v[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                pad = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = mx.pad(k, pad)
                v = mx.pad(v, pad)
            left_padding = c.left_padding + left
            return k, v, c.offset, left_padding

        self.keys, self.values, self.offset, self.left_padding = map(
            mx.concatenate, zip(*(pad(self), pad(other)))
        )
        self._idx = max_idx
        self._offset = max(self._offset, other._offset)

    def extract(self, idx):
        mx.eval(self.left_padding, self.offset)
        cache = RotatingKVCache(self.max_size)
        padding = max(0, self.left_padding.tolist()[idx])
        offset = self.offset.tolist()[idx]
        cache.keys = self.keys[idx : idx + 1]
        cache.values = self.values[idx : idx + 1]
        cache._idx = self._idx
        if self.rotated:
            cache.keys = mx.roll(cache.keys, -self._idx, axis=2)
            cache.values = mx.roll(cache.values, -self._idx, axis=2)
            cache._idx = self.max_size
        cache.keys = mx.contiguous(cache.keys[:, :, padding : cache._idx])
        cache.values = mx.contiguous(cache.values[:, :, padding : cache._idx])
        cache.offset = offset
        cache._idx = cache.keys.shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        if not all(c.max_size == caches[0].max_size for c in caches):
            raise ValueError(
                "BatchRotatingKVCache can only merge caches with the same maximum size"
            )

        offsets = [c.offset for c in caches]
        lengths = [c.size() for c in caches]
        max_length = max(lengths)

        # No cache has content so make an empty one
        if max_length == 0:
            return cls(caches[0].max_size, [0] * len(caches))

        padding = [max_length - l for l in lengths]
        B = len(caches)
        H = max(c.keys.shape[1] for c in caches if c.keys is not None)
        Dk = max(c.keys.shape[3] for c in caches if c.keys is not None)
        Dv = max(c.values.shape[3] for c in caches if c.values is not None)
        dt = next(iter(c.keys.dtype for c in caches if c.keys is not None))

        keys = mx.zeros((B, H, max_length, Dk), dtype=dt)
        values = mx.zeros((B, H, max_length, Dv), dtype=dt)
        for i, (p, l, c) in enumerate(zip(padding, lengths, caches)):
            if c.keys is None:
                continue
            keys[i : i + 1, :, p : p + l] = c._temporal_order(c.keys)[..., -l:, :]
            values[i : i + 1, :, p : p + l] = c._temporal_order(c.values)[..., -l:, :]

        cache = cls(caches[0].max_size, padding)
        cache.keys = keys
        cache.values = values
        cache.offset = mx.array(offsets)
        cache._idx = keys.shape[2]
        cache._offset = keys.shape[2]

        return cache

    def size(self):
        return min(self._offset, self.max_size)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class TokenBuffer:
    """A simple token buffer that can be efficiently appended to in a similar
    fashion to the KVCache.

    Perhaps these could share some logic in the future.
    """

    step = 256

    def __init__(self, tokens=[]):
        self._buffer = mx.array(tokens, dtype=mx.int32)
        self._size = len(tokens)

    def update_and_fetch(self, tokens):
        start = self._size
        end = start + len(tokens)

        new_size = ((end + self.step - 1) // self.step) * self.step
        if new_size > self._buffer.size:
            self._buffer = mx.concatenate(
                [self._buffer, mx.zeros(new_size - self._buffer.size, dtype=mx.int32)]
            )
        self._buffer[start:end] = tokens
        self._size = end

        return self._buffer[:end]

    @property
    def state(self):
        return self._buffer

    @property
    def tokens(self):
        return self._buffer[: self._size]


@dataclass
class PromptTrieResult:
    model: Any
    exact: Optional[List[int]]  # Exact match found
    shorter: Optional[List[int]]  # Longest prefix with a value
    longer: Optional[List[int]]  # Shortest value that extends beyond tokens
    common_prefix: int  # Length of common prefix with any path


class PromptTrie:
    def __init__(self):
        self._trie = {}

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

    def get(self, model: Any, tokens: List[int]):
        current = self._trie[model]
        for tok in tokens:
            current = current[tok]
        return current["__value__"]

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

    def search(self, model: Any, tokens: List[int]) -> PromptTrieResult:
        if model not in self._trie:
            return PromptTrieResult(model, None, None, None, 0)

        current = self._trie[model]

        if not tokens and "__value__" in current:
            return PromptTrieResult(model, [], None, None, 0)

        # Walk the tokens as far as we can
        last_index = -1
        index = 0
        while index < len(tokens) and tokens[index] in current:
            current = current[tokens[index]]
            if "__value__" in current:
                last_index = index
            index += 1

        # Got an exact match
        if last_index == len(tokens) - 1 >= 0:
            return PromptTrieResult(model, tokens, None, None, 0)

        # Check if we found a prefix at any point
        shorter = None
        if last_index > 0:
            shorter = tokens[: last_index + 1]

        # Check for sequences that are longer
        longer = None
        common_prefix = index
        if index > 0:
            best = None
            stack = [(current, [])]
            while stack:
                current, extra = stack.pop()
                if "__value__" in current:
                    if best is None or len(extra) < len(best):
                        best = extra
                elif best is None or len(extra) < len(best):
                    for tok in current:
                        stack.append((current[tok], extra + [tok]))
            longer = tokens[:index] + best
        return PromptTrieResult(model, None, shorter, longer, common_prefix)


class LRUPromptCache:
    @dataclass
    class CacheEntry:
        prompt_cache: List[Any]
        nbytes: int
        cache_type: str

    class CacheOrder:
        def __init__(self, ordering: List[str] = ["assistant", "user", "system"]):
            self._ordering = ordering
            self._lrus = {k: deque() for k in ordering}

        def __len__(self):
            return sum(len(lru) for lru in self._lrus.values())

        def push(self, model: Any, tokens: List[Any], cache_type: str = "assistant"):
            self._lrus[cache_type].append((model, tokens))

        def remove(self, model: Any, tokens: List[Any]):
            for cache_type in self._ordering:
                try:
                    self._lrus[cache_type].remove((model, tokens))
                    break
                except ValueError:
                    pass

        def pop(self):
            i = 0
            while i + 1 < len(self._ordering):
                lru_a = self._lrus[self._ordering[i]]
                lru_b = self._lrus[self._ordering[i + 1]]
                if lru_a and len(lru_a) >= len(lru_b):
                    return lru_a.popleft()
                i += 1
            return lru_b.popleft()

    def __init__(
        self,
        max_size: int = 10,
        max_bytes: int = 1 << 63,
        ssd_cache: Optional["SsdCache"] = None,
        block_ssd_cache: Optional["BlockSSDCache"] = None,
        hermes_optimizer: Optional = None,
    ):
        self.max_size = max_size
        self.max_bytes = max_bytes
        self._ssd_cache = ssd_cache
        self._block_ssd_cache = block_ssd_cache
        self._hermes_optimizer = hermes_optimizer
        self._trie = PromptTrie()
        self._lru = LRUPromptCache.CacheOrder()
        self._n_bytes = 0
        self._n_bytes_by_type = {k: 0 for k in self._lru._ordering}
        self._pending_ssd_saves: List[Tuple[str, List[int], List[Any]]] = []
        # Per-request SSD cache observability (Fix B).
        # Reset at the start of each fetch_nearest_cache() call,
        # read by server.py at the end of the request.
        self._last_fetch_stats: Dict[str, Any] = {
            "block_hit_blocks": 0,
            "block_hit_tokens": 0,
            "blocks_written": 0,
            "chain_break": False,
            "trie_hit": False,
        }


    def __len__(self):
        return len(self._lru)

    @property
    def nbytes(self):
        return self._n_bytes

    def set_hermes_optimizer(self, optimizer=None):
        """Set or update the Hermes optimizer after construction.
        
        The optimizer cannot be created at construction time because
        the tokenizer isn't available yet. Call this after model loading
        with a properly initialized optimizer.
        """
        self._hermes_optimizer = optimizer

    def fetch_nearest_cache(self, model: Any, tokens: List[int]):
        # Reset per-request observability counters (Fix B)
        self._last_fetch_stats = {
            'block_hit_blocks': 0,
            'block_hit_tokens': 0,
            'blocks_written': 0,
            'chain_break': False,
            'trie_hit': False,
        }
        result = self._trie.search(model, tokens)
        if result.exact is not None:
            self._last_fetch_stats['trie_hit'] = True
            cache_entry = self._trie.get(result.model, result.exact)
            return copy.deepcopy(cache_entry.prompt_cache), []

        short_length = len(result.shorter) if result.shorter is not None else 0
        if result.longer is not None and result.common_prefix > short_length:
            cache_entry = self._trie.get(result.model, result.longer)
            if can_trim_prompt_cache(cache_entry.prompt_cache):
                self._last_fetch_stats['trie_hit'] = True
                cache = copy.deepcopy(cache_entry.prompt_cache)
                prefix = min(len(tokens) - 1, result.common_prefix)
                num_to_trim = len(result.longer) - prefix
                trim_prompt_cache(cache, num_to_trim)
                return cache, tokens[prefix:]

        if short_length > 0:
            self._last_fetch_stats['trie_hit'] = True
            cache_entry = self._trie.get(result.model, result.shorter)
            return copy.deepcopy(cache_entry.prompt_cache), tokens[short_length:]

        # ── Block SSD cache lookup ─────────────────────────────────────
        if self._block_ssd_cache is not None:
            model_str = str(model)
            try:
                # Use Hermes optimizer to skip blocks we know will miss
                check_tokens = tokens
                if self._hermes_optimizer is not None:
                    cacheable_count = self._hermes_optimizer.get_cacheable_token_count(
                        tokens
                    )
                    if cacheable_count > 0:
                        check_tokens = tokens[:cacheable_count]

                matched_hashes, div_idx = self._block_ssd_cache.find_longest_block_chain(
                    check_tokens, model_str
                )

                if matched_hashes:
                    self._last_fetch_stats['block_hit_blocks'] = len(matched_hashes)
                    self._last_fetch_stats['block_hit_tokens'] = div_idx
                    # Load all blocks from SSD in a single batch call
                    all_block_caches = self._block_ssd_cache.load_blocks_batch(matched_hashes)

                    # Filter out None results (missing/corrupt blocks)
                    valid_caches = [bc for bc in all_block_caches if bc is not None]
                    if len(valid_caches) == len(matched_hashes):
                        # Merge blocks into a single cache list
                        merged_cache = self._merge_block_caches(valid_caches)
                        if merged_cache is not None:
                            # Cache the merged result in RAM trie for future hits.
                            # Deep copy for the trie (protected from caller mutations)
                            # and return the original — avoids deep-copying the
                            # full merged result twice on the return path.
                            cached_tokens = tokens[:div_idx]
                            trie_cache = copy.deepcopy(merged_cache)
                            entry = LRUPromptCache.CacheEntry(
                                trie_cache,
                                sum(c.nbytes for c in trie_cache),
                                "system",  # System prompt blocks
                            )
                            self._trie.add(model, cached_tokens, entry)
                            self._n_bytes += entry.nbytes
                            self._n_bytes_by_type[entry.cache_type] += entry.nbytes
                            self._lru.push(model, cached_tokens, entry.cache_type)

                            logger.debug(
                                f"BlockSSDCache HIT: {len(matched_hashes)} blocks, "
                                f"{div_idx} tokens from SSD"
                            )
                            return merged_cache, tokens[div_idx:]
                else:
                        self._last_fetch_stats['chain_break'] = True
                        # Chain break: block 0 doesn't match anything in the index.
                        # The system prompt content at the very beginning changed.
                        index_block_count = self._block_ssd_cache.block_count
                        if index_block_count > 0:
                            # Decode the first tokens to show what text is new
                            preamble_text = ""
                            try:
                                if self._hermes_optimizer is not None:
                                    tok = getattr(self._hermes_optimizer, '_tokenizer', None)
                                    if tok is not None:
                                        preamble_text = tok.decode(tokens[:min(64, len(tokens))])[:300]
                            except Exception:
                                pass
                            logger.warning(
                                f"BlockSSD chain break at block 0: "
                                f"{len(tokens)} tokens in prompt, "
                                f"{index_block_count} blocks in SSD index. "
                                f"System prompt prefix text changed. "
                                f"First 64 token IDs: {list(tokens[:min(64, len(tokens))])}"
                            )
                            if preamble_text:
                                logger.warning(
                                    f"New system prompt start: {preamble_text!r}"
                                )

            except Exception as e:
                logger.warning(
                    f"BlockSSDCache lookup failed: {e}, "
                    f"falling back to progressive scan"
                )

        # SSD fallback: try progressively shorter prefixes on disk
        if self._ssd_cache is not None:
            model_str = str(model)
            for i in range(len(tokens), 0, -1):
                prefix = tokens[:i]
                cache = self._ssd_cache.load_entry(model_str, prefix)
                if cache is not None:
                    # Restore into RAM trie
                    entry = LRUPromptCache.CacheEntry(
                        cache,
                        sum(c.nbytes for c in cache),
                        "user",  # conservative cache_type
                    )
                    self._trie.add(model, prefix, entry)
                    self._n_bytes += entry.nbytes
                    self._n_bytes_by_type[entry.cache_type] += entry.nbytes
                    self._lru.push(model, prefix, entry.cache_type)
                    logger.debug(
                        f"SsdCache HIT: restored {len(prefix)} tokens "
                        f"({entry.nbytes / 1e6:.1f} MB) from disk to RAM trie"
                    )
                    return copy.deepcopy(cache), tokens[i:]

        return None, tokens

    def insert_cache(
        self,
        model: Any,
        tokens: List[int],
        prompt_cache: List[Any],
        *,
        cache_type: str = "assistant",
    ):
        # Make the cache entry
        entry = LRUPromptCache.CacheEntry(
            prompt_cache, sum(c.nbytes for c in prompt_cache), cache_type
        )

        # Insert into the trie and update the byte counter and lru position
        self._n_bytes += entry.nbytes
        self._n_bytes_by_type[cache_type] += entry.nbytes
        prev = self._trie.add(model, tokens, entry)
        if prev is not None:
            self._n_bytes -= prev.nbytes
            self._n_bytes_by_type[prev.cache_type] -= prev.nbytes
            self._lru.remove(model, tokens)
        self._lru.push(model, tokens, cache_type)

        # Remove all prefix entries that are taking space for this branch,
        # but preserve intermediate checkpoints (system/user) created by
        # segment-aware caching — they are strategic cache boundaries, not
        # accidental short entries.
        if can_trim_prompt_cache(prompt_cache):
            for prefix_len, entry in self._trie.pop_prefixes(model, tokens):
                if entry.cache_type in ("system", "user"):
                    # pop_prefixes destructively removed __value__ from this
                    # trie node. Since we're preserving this checkpoint,
                    # re-insert __value__ so the trie stays in sync with
                    # the LRU — otherwise eviction later will hit a
                    # KeyError('__value__') in _trie.pop().
                    self._trie.add(model, tokens[:prefix_len], entry)
                    continue
                self._n_bytes -= entry.nbytes
                self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
                self._lru.remove(model, tokens[:prefix_len])

        # Immediately persist "system" cache blocks to SSD for cross-session reuse.
        if cache_type == "system" and self._block_ssd_cache is not None:
            self._save_blocks_to_ssd(str(model), tokens, prompt_cache)

        # Ensure we match the constraints
        if len(self._lru) > self.max_size:
            model, tokens = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            if self._ssd_cache is not None and entry.prompt_cache:
                self._ssd_cache.save_entry(str(model), tokens, entry.prompt_cache)
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
            # Optionally save to block SSD cache — only system entries
            # are reusable across sessions. Non-system entries (user/assistant)
            # pollute the cache with unreusable conversation cruft.
            if (
                self._block_ssd_cache is not None
                and entry.prompt_cache
                and entry.cache_type == "system"
            ):
                self._save_blocks_to_ssd(str(model), tokens, entry.prompt_cache)
        while self._n_bytes > self.max_bytes:
            model, tokens = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            if self._ssd_cache is not None and entry.prompt_cache:
                self._ssd_cache.save_entry(str(model), tokens, entry.prompt_cache)
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
            # Optionally save to block SSD cache
            if (
                self._block_ssd_cache is not None
                and entry.prompt_cache
                and entry.cache_type == "system"
            ):
                self._save_blocks_to_ssd(str(model), tokens, entry.prompt_cache)

    def trim_to(
        self, *, n_sequences: Optional[int] = None, n_bytes: Optional[int] = None
    ):
        n_sequences = max(0, n_sequences) if n_sequences is not None else 1 << 63
        n_bytes = max(0, n_bytes) if n_bytes is not None else 1 << 63

        while len(self._lru) > n_sequences:
            model, tokens = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            if self._ssd_cache is not None and entry.prompt_cache:
                self._ssd_cache.save_entry(str(model), tokens, entry.prompt_cache)
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
            # Optionally save to block SSD cache
            if (
                self._block_ssd_cache is not None
                and entry.prompt_cache
                and entry.cache_type == "system"
            ):
                self._save_blocks_to_ssd(str(model), tokens, entry.prompt_cache)
        while self._n_bytes > n_bytes:
            model, tokens = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            if self._ssd_cache is not None and entry.prompt_cache:
                self._ssd_cache.save_entry(str(model), tokens, entry.prompt_cache)
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
            # Optionally save to block SSD cache
            if (
                self._block_ssd_cache is not None
                and entry.prompt_cache
                and entry.cache_type == "system"
            ):
                self._save_blocks_to_ssd(str(model), tokens, entry.prompt_cache)

    def stats_by_type(self):
        result = {}
        for cache_type in self._lru._ordering:
            result[cache_type] = {
                "n_sequences": len(self._lru._lrus[cache_type]),
                "n_bytes": self._n_bytes_by_type[cache_type],
            }
        return result

    def warm_from_ssd(self, n: int = 0):
        """Warm RAM cache by loading SSD entries.

        This is a placeholder for future enhancement. SSD entries are
        loaded lazily on fetch miss in ``fetch_nearest_cache``, which
        is sufficient for most use cases. A full warm-up would require
        storing token sequences in SSD metadata for reconstruction.

        Args:
            n: Number of entries to load. 0 = all that fit within max_bytes.
        """
        pass

    # ── Block-level SSD cache methods ─────────────────────────────────

    def _merge_block_caches(
        self, block_caches: List[List[Any]]
    ) -> Optional[List[Any]]:
        """
        Merge consecutive block caches into a single contiguous KV cache.

        For sliceable layers (KVCache): concatenates keys/values along seq dimension.
        For boundary_only layers: takes the state from the LAST block only.

        Args:
            block_caches: List of per-layer cache lists, one per block.

        Returns:
            Merged per-layer cache list, or None if merge fails.
        """
        import mlx.core as mx
        from mlx_lm.models.block_cache_utils import classify_cache_layer

        if not block_caches or len(block_caches) == 0:
            return None

        num_layers = len(block_caches[0])
        merged = []

        for layer_idx in range(num_layers):
            # Get this layer's cache from the first block to determine type
            first_cache = block_caches[0][layer_idx]
            layer_type = classify_cache_layer(first_cache)

            if layer_type == "sliceable":
                # Concatenate along seq dim (axis=2)
                all_keys = []
                all_values = []
                total_tokens = 0

                for block in block_caches:
                    layer = block[layer_idx]
                    state = layer.state  # (keys, values), [B, n_kv_heads, seq, dim]
                    all_keys.append(state[0])
                    all_values.append(state[1])
                    total_tokens += state[0].shape[2]

                merged_keys = mx.concatenate(all_keys, axis=2)
                merged_values = mx.concatenate(all_values, axis=2)

                # Reuse the first block's layer object: replace its state
                # and offset in-place.  Blocks from load_blocks_batch are
                # already unique copies (deep-copied from hot cache or
                # freshly deserialized from disk), so no sharing risk.
                merged_layer = first_cache
                merged_layer.state = (merged_keys, merged_values)
                merged_layer.offset = total_tokens
                merged.append(merged_layer)

            elif layer_type in ("boundary_only", "unknown"):
                # Only the LAST block's state is valid for non-sliceable layers
                last_block = block_caches[-1][layer_idx]
                merged.append(last_block)

        return merged


    def get_last_fetch_stats(self) -> Dict[str, Any]:
        """Return per-request SSD cache observability stats.

        Called by server.py after each request to populate the PERF log.
        Returns a dict with keys:
          - block_hit_blocks: blocks loaded from SSD this request
          - block_hit_tokens: tokens covered by SSD blocks
          - blocks_written: new blocks persisted to SSD this request
          - chain_break: True if block-0 prefix mismatch was detected
          - trie_hit: True if the RAM trie matched before SSD lookup
        """
        return dict(self._last_fetch_stats)

    def _save_blocks_to_ssd(
        self, model_str: str, tokens: List[int], prompt_cache: List[Any]
    ) -> None:
        """
        Queue block SSD save for later deferred batch processing.

        Instead of saving immediately (which triggers per-block GPU sync points
        via mx.eval), defers to flush_pending_ssd_saves() which batches
        ALL tensor evals into a single mx.eval() call.
        """
        if self._block_ssd_cache is None:
            return
        self._pending_ssd_saves.append((model_str, tokens, prompt_cache))

    def flush_pending_ssd_saves(self) -> None:
        """
        Process all queued SSD saves with a single batch mx.eval().

        Deferred from _save_blocks_to_ssd() to move GPU sync points
        out of the prefill hot path. For each pending save:
         1. Compute block hashes and slice caches (lazy ops, no GPU sync)
         2. Collect ALL tensors from ALL block caches
         3. Call ONE mx.eval() to materialize everything
         4. Save each block (tensor extraction is now a no-op eval)
         5. Final flush + shrink once
        """
        if not self._pending_ssd_saves:
            return

        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm.models.block_cache_utils import (
            slice_cache_for_block,
            get_cache_layer_info,
        )
        from mlx_lm.models.block_ssd_cache import compute_block_hash, compute_content_hash, BLOCK_SIZE

        # Phase 1: Compute all block slices (lazy ops, no GPU sync)
        block_tasks: List[Tuple[bytes, List[Any], Dict[str, str]]] = []
        for model_str, tokens, prompt_cache in self._pending_ssd_saves:
            info = get_cache_layer_info(prompt_cache)
            num_full_blocks = len(tokens) // BLOCK_SIZE
            if num_full_blocks == 0:
                continue

            parent_hash: Optional[bytes] = None
            for block_idx in range(num_full_blocks):
                start_tok = block_idx * BLOCK_SIZE
                block_tokens = tokens[start_tok : start_tok + BLOCK_SIZE]
                block_hash = compute_block_hash(parent_hash, block_tokens, model_str)
                content_hash = compute_content_hash(block_tokens, model_str, block_idx)

                # Deduplicate against already-cached blocks
                if self._block_ssd_cache.contains(block_hash):
                    parent_hash = block_hash
                    continue

                block_cache = slice_cache_for_block(
                    prompt_cache, start_tok, BLOCK_SIZE
                )
                metadata = {
                    "model_name": model_str,
                    "num_layers": info["num_layers"],
                    "token_count": BLOCK_SIZE,
                    "layer_cache_types": info["layer_cache_types"],
                    "content_hash": content_hash.hex(),
                }
                block_tasks.append((block_hash, block_cache, metadata))
                parent_hash = block_hash

        # Clear the queue — all work is captured in block_tasks
        self._pending_ssd_saves.clear()

        # Record how many new blocks were written this flush cycle (Fix B)
        self._last_fetch_stats['blocks_written'] = len(block_tasks)

        if not block_tasks:
            return

        # Phase 2: Collect ALL tensors from ALL block caches for ONE big eval
        all_tensors: List[mx.array] = []
        for _, block_cache, _ in block_tasks:
            for layer_cache in block_cache:
                state = layer_cache.state
                flat_state = tree_flatten(state)
                for _, t in flat_state:
                    if hasattr(t, "dtype") and hasattr(t, "shape"):
                        all_tensors.append(t)

        # Single batch eval — the whole optimization.
        # Replaces ~120 per-block mx.eval() calls with one.
        if all_tensors:
            mx.eval(all_tensors)

        # Phase 3: Save all blocks.
        # _extract_cache_bytes will call mx.eval() again, but since
        # all tensors are already evaluated it's a no-op.
        for block_hash, block_cache, metadata in block_tasks:
            self._block_ssd_cache.save_block(block_hash, block_cache, metadata)

            logger.debug(
                f"Deferred save of block {block_hash.hex()[:12]} "
                f"({metadata.get('token_count', '?')} tokens)"
            )

        # One flush + shrink for the whole batch
        self._block_ssd_cache.flush()
        self._block_ssd_cache.shrink()


@dataclass
class SsdCacheEntry:
    """Metadata for a single SSD-cached prompt cache entry."""

    key: str  # sha256 hex digest
    path: str  # full path to .safetensors file
    nbytes: int  # file size in bytes
    mtime: float  # os.path.getmtime
    model_key: str  # model identifier
    token_hash: str  # hash of the token list (separate from cache key)


class SsdCache:
    """Disk-backed tier for LRUPromptCache. One safetensors file per entry."""

    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_size_gb: float = 0,
    ):
        """
        Args:
            cache_dir: Directory for safetensors files. Created if not exist.
            max_size_gb: Max disk usage in GB. 0 = unlimited.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_size_gb * 1e9) if max_size_gb > 0 else 0
        self._lock = threading.Lock()
        self._index: dict[str, SsdCacheEntry] = {}
        self._index_path = self.cache_dir / "_index.json"
        self._load_index()

    def _compute_key(self, model_key: str, tokens: list[int]) -> str:
        """Deterministic hash from model_key + tokens for stable filenames."""
        h = hashlib.sha256()
        h.update(model_key.encode("utf-8"))
        h.update(b"\0")
        h.update(pickle.dumps(tokens))
        return h.hexdigest()

    def _entry_path(self, key: str) -> Path:
        """Two-char prefix subdir to avoid flat dir explosion."""
        return self.cache_dir / key[:2] / f"{key}.safetensors"

    def _load_index(self) -> None:
        """Read _index.json if it exists. Falls back to directory scan on error."""
        if self._index_path.exists():
            try:
                with open(self._index_path, "r") as f:
                    data = json.load(f)
                for entry_dict in data:
                    entry = SsdCacheEntry(**entry_dict)
                    self._index[entry.key] = entry
                return
            except (json.JSONDecodeError, OSError, TypeError) as e:
                logger.warning(
                    f"SsdCache index corrupted at {self._index_path}, "
                    f"rebuilding from directory scan: {e}"
                )
                # Remove corrupted index so the scan can regenerate it
                self._index_path.unlink(missing_ok=True)
        self._scan_directory()

    def _scan_directory(self) -> None:
        """Rebuild index by walking cache_dir for .safetensors files."""
        for fpath in self.cache_dir.rglob("*.safetensors"):
            key = fpath.stem
            stat = fpath.stat()
            self._index[key] = SsdCacheEntry(
                key=key,
                path=str(fpath),
                nbytes=stat.st_size,
                mtime=stat.st_mtime,
                model_key="",
                token_hash="",
            )

    def _save_index(self) -> None:
        """Persist index to _index.json atomically."""
        data = [dataclasses.asdict(e) for e in self._index.values()]
        tmp = self._index_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        tmp.rename(self._index_path)

    def save_entry(
        self,
        model_key: str,
        tokens: list[int],
        prompt_cache: list[Any],
    ) -> Optional[str]:
        """Serialize a cache entry to SSD.

        Args:
            model_key: Model identifier (used in hash).
            tokens: Token list that keys this entry.
            prompt_cache: Per-layer KV cache list (List[Any] with .state,
                .meta_state, .offset).

        Returns:
            The cache key string, or None on failure.
        """
        key = self._compute_key(model_key, tokens)
        path = self._entry_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "model_key": model_key,
            "token_count": str(len(tokens)),
            "key": key,
        }

        try:
            save_prompt_cache(str(path), prompt_cache, metadata)
            stat = path.stat()
            entry = SsdCacheEntry(
                key=key,
                path=str(path),
                nbytes=stat.st_size,
                mtime=stat.st_mtime,
                model_key=model_key,
                token_hash=str(hash(tuple(tokens))),
            )
            with self._lock:
                self._index[key] = entry
                self._save_index()
            self._enforce_size_limit()
            logger.debug(
                f"SsdCache saved entry '{key[:12]}' "
                f"({stat.st_size / 1e6:.1f} MB, {len(tokens)} tokens)"
            )
            return key
        except Exception as e:
            logger.warning(f"SsdCache save failed for key {key}: {e}")
            if path.exists():
                path.unlink(missing_ok=True)
            return None

    def load_entry(
        self,
        model_key: str,
        tokens: list[int],
    ) -> Optional[list[Any]]:
        """Deserialize a cache entry from SSD.

        Args:
            model_key: Model identifier.
            tokens: Token list to compute lookup key.

        Returns:
            The per-layer KV cache list, or None if not found/error.
        """
        key = self._compute_key(model_key, tokens)
        path = self._entry_path(key)

        if not path.exists():
            return None

        try:
            cache_layers, metadata = load_prompt_cache(str(path), return_metadata=True)
            stored_model = metadata.get("model_key", "")
            if stored_model and stored_model != model_key:
                logger.warning(
                    f"SsdCache model_key mismatch: {stored_model} != {model_key}"
                )
                return None
            with self._lock:
                if key in self._index:
                    self._index[key].mtime = path.stat().st_mtime
            logger.debug(
                f"SsdCache loaded entry '{key[:12]}' "
                f"({metadata.get('token_count', '?')} tokens, "
                f"{path.stat().st_size / 1e6:.1f} MB)"
            )
            return cache_layers
        except Exception as e:
            logger.warning(f"SsdCache load failed for key {key}: {e}")
            path.unlink(missing_ok=True)
            with self._lock:
                self._index.pop(key, None)
                self._save_index()
            return None

    def delete_entry(self, key: str) -> bool:
        """Remove an entry from disk and index."""
        path = self._entry_path(key)
        try:
            path.unlink(missing_ok=True)
            with self._lock:
                self._index.pop(key, None)
                self._save_index()
            return True
        except Exception as e:
            logger.warning(f"SsdCache delete failed for key {key}: {e}")
            return False

    def _enforce_size_limit(self) -> None:
        """Delete oldest entries until under max_bytes.

        Called after each save_entry. Evicts oldest by mtime.
        """
        if self.max_bytes <= 0:
            return
        with self._lock:
            total = sum(e.nbytes for e in self._index.values())
            if total <= self.max_bytes:
                return
            sorted_entries = sorted(self._index.values(), key=lambda e: e.mtime)
            for entry in sorted_entries:
                if total <= self.max_bytes:
                    break
                fpath = Path(entry.path)
                fpath.unlink(missing_ok=True)
                self._index.pop(entry.key, None)
                total -= entry.nbytes
            self._save_index()

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return sum(e.nbytes for e in self._index.values())

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._index)

    def clear(self) -> None:
        """Delete all cached entries."""
        with self._lock:
            for entry in self._index.values():
                Path(entry.path).unlink(missing_ok=True)
            self._index.clear()
            self._save_index()
