"""
Block-level SSD prefix cache for mlx-lm.

Replaces whole-sequence keying with chain-hash-based block storage.
Each block stores 256 tokens of KV cache as a separate safetensors file.

Design reference: omlx/cache/paged_ssd_cache.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import pickle
import queue
import struct
import threading
import time
from dataclasses import dataclass, field
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
BLOCK_SIZE = 256  # Tokens per block. Tune later based on model profiling.
DEFAULT_ROOT_HASH = b"mlx-lm-block-root"  # Seed hash for first block in chain
_MAX_PENDING_WRITES = 64  # Max async write queue depth

# ── Safetensors dtype mapping (no mx API needed for background writes) ────
_MX_TO_ST_DTYPE: Dict[Any, str] = {}
_ST_DTYPE_TO_NP: Dict[str, Any] = {}

try:
    import mlx.core as _mx

    _MX_TO_ST_DTYPE = {
        _mx.float16: "F16",
        _mx.float32: "F32",
        _mx.bfloat16: "BF16",
        _mx.int8: "I8",
        _mx.int16: "I16",
        _mx.int32: "I32",
        _mx.int64: "I64",
        _mx.uint8: "U8",
        _mx.uint16: "U16",
        _mx.uint32: "U32",
        _mx.uint64: "U64",
        _mx.bool_: "BOOL",
    }
except ImportError:
    pass

_ST_DTYPE_TO_NP = {
    "F16": np.float16,
    "F32": np.float32,
    "BF16": np.uint16,  # bfloat16 handled via uint16 view
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "U8": np.uint8,
    "U16": np.uint16,
    "U32": np.uint32,
    "U64": np.uint64,
    "BOOL": np.bool_,
}


def _has_zero_dim(tensor: Any) -> bool:
    """Check if a tensor has any zero-dimension axis (unsupported by safetensors)."""
    return hasattr(tensor, "shape") and any(d == 0 for d in tensor.shape)


def _extract_tensor_bytes(arr: Any) -> Tuple[bytes, str, List[int]]:
    """Extract raw bytes from an evaluated mlx.core.array.

    Must be called from inference thread after mx.eval() (Metal-safe).
    The returned bytes can be written to disk from any thread without mx API.

    For bfloat16 arrays, uses view(uint16) trick since Python's buffer
    protocol doesn't support bfloat16 directly.

    Args:
        arr: Evaluated MLX array (must have been mx.eval'd).

    Returns:
        Tuple of (raw_bytes, safetensors_dtype_string, shape_list).
    """
    import mlx.core as mx

    dtype_str = _MX_TO_ST_DTYPE[arr.dtype]
    shape = list(arr.shape)
    if arr.dtype == mx.bfloat16:
        u16 = arr.view(mx.uint16)
        mx.eval(u16)
        raw = bytes(memoryview(u16))
    else:
        raw = bytes(memoryview(arr))
    return raw, dtype_str, shape


def _write_safetensors_no_mx(
    path: str,
    tensors_raw: Dict[str, Tuple[bytes, str, List[int]]],
    metadata: Optional[Dict[str, str]] = None,
) -> int:
    """Write a safetensors file without any mx/Metal API calls.

    Safe to call from background threads. Produces files fully compatible
    with mx.load(path, return_metadata=True).

    The safetensors binary format:
      [8 bytes: header_size as little-endian uint64]
      [header_size bytes: JSON header]
      [remaining bytes: concatenated tensor data]

    Args:
        path: Output file path (must include .safetensors extension).
        tensors_raw: Dict of {name: (raw_bytes, dtype_str, shape)}.
        metadata: Optional string-to-string metadata dict.

    Returns:
        Total file size in bytes.
    """
    offset = 0
    header_tensors = {}
    all_data = []

    for name, (raw, dtype_str, shape) in tensors_raw.items():
        header_tensors[name] = {
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw)],
        }
        all_data.append(raw)
        offset += len(raw)

    header_dict: Dict[str, Any] = dict(header_tensors)
    if metadata:
        header_dict["__metadata__"] = metadata

    header_json = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
    # Safetensors spec: header must be 8-byte aligned
    pad = (8 - len(header_json) % 8) % 8
    header_json += b" " * pad

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for d in all_data:
            f.write(d)

    return offset + 8 + len(header_json)


# ── Hash function ──────────────────────────────────────────────────────────

def compute_block_hash(
    parent_hash: Optional[bytes],
    block_tokens: List[int],
    model_name: str,
    extra_keys: Optional[Tuple[Any, ...]] = None,
) -> bytes:
    """
    Compute SHA-256 chain hash for a block of tokens.

    The hash depends on:
      1. model_name — prevents cross-model cache poisoning
      2. parent_hash — links this block to all previous blocks in sequence
      3. block_tokens — the actual token content
      4. extra_keys — optional salt (e.g., for VLM image hashes; not needed for Hermes)

    Why chain hashing? The KV cache at position N depends on ALL tokens 0..N-1.
    If token K changes, KV values for all positions >K are different. Chain
    hashing encodes this: a changed parent_hash produces a different child hash
    for all subsequent blocks.

    Args:
        parent_hash: SHA-256 digest of previous block, or None for first block.
        block_tokens: Token IDs in this block (up to BLOCK_SIZE).
        model_name: Model identifier for cache isolation.
        extra_keys: Optional extra hash salt.

    Returns:
        32-byte SHA-256 digest.
    """
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(parent_hash if parent_hash else DEFAULT_ROOT_HASH)
    # Use pickle for deterministic token list serialization
    h.update(pickle.dumps(tuple(block_tokens)))
    if extra_keys is not None:
        h.update(pickle.dumps(extra_keys))
    return h.digest()


# ── Block metadata ─────────────────────────────────────────────────────────

@dataclass
class BlockMeta:
    """Metadata for one SSD-cached block."""
    block_hash: bytes              # SHA-256 content hash (32 bytes)
    file_path: Path                # Full path to .safetensors file
    file_size: int                 # Bytes on disk (updated after write completes)
    token_count: int               # Typically 256; fewer for the very last block
    num_layers: int                # Number of model layers (for validation on load)
    model_name: str                # Model identifier for cache isolation
    layer_cache_types: List[str]   # e.g. ["KVCache", "KVCache", ..., "ArraysCache"]
                                   # Needed to reconstruct correct cache objects on load
    created_at: float              # Unix timestamp when saved
    last_access: float             # Unix timestamp, touched on every read

    def touch(self) -> None:
        self.last_access = time.time()


# ── Block SSD Cache ────────────────────────────────────────────────────────

class BlockSSDCache:
    """
    SSD-backed cache indexed by block content hash.

    Layout on disk:
        {cache_dir}/
        ├── _index.json          # HashMap[hex_hash → BlockMeta]
        └── blocks/
            ├── ab/
            │   └── abc123....safetensors
            └── cd/
                └── cde456....safetensors

    Features:
    - O(1) block lookup by content hash
    - LRU eviction when total size exceeds max_bytes
    - Thread-safe for concurrent reads from multiple requests
    - Startup scan to index existing files (survives restarts)
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_size_gb: float = 0,
    ) -> None:
        """
        Args:
            cache_dir: Directory for block safetensors files.
            max_size_gb: Max disk usage in GB. 0 = unlimited.
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.blocks_dir = self.cache_dir / "blocks"
        self.blocks_dir.mkdir(parents=True, exist_ok=True)

        self.max_bytes = int(max_size_gb * 1e9) if max_size_gb > 0 else 0

        # Core data structures (thread-safe)
        self._lock = threading.RLock()
        self._index: Dict[bytes, BlockMeta] = {}       # hash → metadata
        self._lru: OrderedDict[bytes, float] = OrderedDict()  # hash → last_access
        self._total_bytes: int = 0

        self._index_path = self.cache_dir / "_index.json"
        self._load_index()
        self._start_async_writer()

        # Hot/in-memory LRU block cache for recently loaded blocks
        self._hot_cache: Dict[bytes, List[Any]] = {}
        self._hot_cache_order: OrderedDict[bytes, None] = OrderedDict()
        self._hot_cache_max_items = 128

        # Stats tracking
        self._stats: Dict[str, int] = {
            "hot_cache_hits": 0,
            "hot_cache_misses": 0,
            "disk_loads": 0,
            "disk_saves": 0,
            "batch_loads": 0,
        }

    # ── Public API ──────────────────────────────────────────────────

    def contains(self, block_hash: bytes) -> bool:
        """Check if a block with this hash exists on disk."""
        with self._lock:
            return block_hash in self._index

    def get_meta(self, block_hash: bytes) -> Optional[BlockMeta]:
        """Get block metadata without loading tensor data."""
        with self._lock:
            meta = self._index.get(block_hash)
            if meta is not None:
                meta.touch()
                self._lru.move_to_end(block_hash)
                self._lru[block_hash] = meta.last_access
            return meta

    def load_block(self, block_hash: bytes) -> Optional[List[Any]]:
        """
        Load KV cache tensors for a block from SSD or in-memory hot cache.

        Checks the LRU hot cache first (O(1) lookup). On cache hit, returns
        a deep copy of the cached layers. On miss, loads from disk via
        mx.load(), promotes to hot cache, and returns.

        Returns:
            List of per-layer cache objects, or None if not found/corrupted.
        """
        # Check hot cache first (fast path)
        with self._lock:
            if block_hash in self._hot_cache:
                self._hot_cache_order.move_to_end(block_hash)
                self._stats["hot_cache_hits"] += 1
                return copy.deepcopy(self._hot_cache[block_hash])

            self._stats["hot_cache_misses"] += 1

        meta = self.get_meta(block_hash)
        if meta is None:
            logger.debug(f"Block {block_hash.hex()[:12]} not in index")
            return None

        if not meta.file_path.exists():
            logger.warning(f"Block file missing: {meta.file_path}, removing from index")
            self._remove_from_index(block_hash)
            return None

        try:
            # Direct mx.load() — avoids load_prompt_cache overhead
            import mlx.core as mx

            arrays, cache_metadata = mx.load(
                str(meta.file_path), return_metadata=True
            )
            cache_layers = self._reconstruct_cache_data(
                arrays, cache_metadata,
                meta.num_layers, meta.layer_cache_types,
            )
            if cache_layers is None:
                return None

            # Promote to hot cache (with eviction if at capacity)
            self._promote_to_hot_cache(block_hash, cache_layers)

            # Track stats
            with self._lock:
                self._stats["disk_loads"] += 1

            return copy.deepcopy(cache_layers)
        except Exception as e:
            logger.warning(f"Failed to load block {block_hash.hex()[:12]}: {e}")
            # Corrupt file — remove it
            meta.file_path.unlink(missing_ok=True)
            self._remove_from_index(block_hash)
            return None

    def _reconstruct_cache_data(
        self,
        arrays: Dict[str, Any],
        cache_metadata: Dict[str, str],
        expected_num_layers: int,
        layer_cache_types: List[str],
    ) -> Optional[List[Any]]:
        """Reconstruct cache layers from raw mx.load() output.

        This mirrors the logic of load_prompt_cache() without importing it:
          arrays, cache_metadata = mx.load(path, return_metadata=True)
          arrays = tree_unflatten(list(arrays.items()))
          cache_metadata = tree_unflatten(list(cache_metadata.items()))
          info, metadata, classes = cache_metadata
          cache = [
              globals()[c].from_state(state, meta_state)
              for c, state, meta_state in zip(classes, arrays, info)
          ]

        Args:
            arrays: Flat dict of tensor name -> array from mx.load().
            cache_metadata: Flat dict of metadata keys from mx.load().
            expected_num_layers: Expected number of cache layers (for validation).
            layer_cache_types: List of class names (e.g. ["KVCache", "KVCache", "ArraysCache"]).

        Returns:
            List of reconstructed cache objects, or None on mismatch/error.
        """
        from mlx.utils import tree_unflatten
        from mlx_lm.models import cache as cache_module

        try:
            # Restore nested structure from flat dicts
            arrays_nested = tree_unflatten(list(arrays.items()))
            meta_nested = tree_unflatten(list(cache_metadata.items()))

            # meta_nested = (info, metadata_dict, classes)
            if not isinstance(meta_nested, (list, tuple)) or len(meta_nested) < 3:
                logger.warning(
                    f"Unexpected metadata structure: {type(meta_nested).__name__}"
                )
                return None
            info, metadata_dict, classes = meta_nested[:3]

            # Validate metadata
            if not isinstance(metadata_dict, dict):
                metadata_dict = {}

            stored_layers = metadata_dict.get("num_layers")
            if stored_layers is not None:
                try:
                    if int(stored_layers) != expected_num_layers:
                        logger.warning(
                            f"Layer count mismatch: expected {expected_num_layers}, "
                            f"got {stored_layers}"
                        )
                        return None
                except (ValueError, TypeError):
                    pass

            # Reconstruct each cache layer
            cache: List[Any] = []
            for i, (class_name, state, meta_state) in enumerate(
                zip(classes, arrays_nested, info)
            ):
                cls = getattr(cache_module, class_name, None)
                if cls is None:
                    logger.warning(
                        f"Unknown cache class {class_name} at layer {i}"
                    )
                    return None
                cache.append(cls.from_state(state, meta_state))

            return cache
        except Exception as e:
            logger.warning(f"Failed to reconstruct cache data: {e}")
            return None

    def _promote_to_hot_cache(
        self, block_hash: bytes, cache_layers: List[Any]
    ) -> None:
        """Insert or update a block in the in-memory LRU hot cache.

        Evicts the LRU entry if at capacity. Thread-safe under self._lock.
        """
        with self._lock:
            if block_hash in self._hot_cache:
                # Already present — just update and move to end
                self._hot_cache[block_hash] = cache_layers
                self._hot_cache_order.move_to_end(block_hash)
                return

            # Evict LRU if at capacity
            if len(self._hot_cache) >= self._hot_cache_max_items:
                oldest, _ = self._hot_cache_order.popitem(last=False)
                self._hot_cache.pop(oldest, None)

            # Insert new entry
            self._hot_cache[block_hash] = cache_layers
            self._hot_cache_order[block_hash] = None

    def load_blocks_batch(
        self, block_hashes: List[bytes]
    ) -> List[Tuple[bytes, Optional[List[Any]]]]:
        """Load multiple blocks in a single batch, checking hot cache first.

        For each hash in the list:
          1. Check hot cache — if found, return cached result immediately.
          2. Otherwise, load safetensors file from disk.
          3. Promote newly loaded blocks to hot cache.

        Args:
            block_hashes: List of content hashes to load.

        Returns:
            List of (block_hash, cache_layers_or_None) tuples in the same
            order as the input list.
        """
        results: List[Tuple[bytes, Optional[List[Any]]]] = []
        disk_hashes: List[bytes] = []

        # Phase 1: Check hot cache for all hashes
        with self._lock:
            for h in block_hashes:
                if h in self._hot_cache:
                    self._hot_cache_order.move_to_end(h)
                    self._stats["hot_cache_hits"] += 1
                    results.append((h, copy.deepcopy(self._hot_cache[h])))
                else:
                    self._stats["hot_cache_misses"] += 1
                    results.append((h, None))
                    disk_hashes.append(h)

        if not disk_hashes:
            return results

        # Phase 2: Load remaining blocks from disk
        with self._lock:
            self._stats["batch_loads"] += 1

        for idx, h in enumerate(disk_hashes):
            # Find the original index in results
            orig_idx = next(
                i for i, (rh, _) in enumerate(results) if rh == h
            )
            loaded = self.load_block(h)
            results[orig_idx] = (h, loaded)

        return results

    def load_blocks_batch(self, block_hashes: List[bytes]) -> List[Optional[List[Any]]]:
        """
        Load multiple blocks from SSD in a single batch call.

        Replaces the serial load_block loop in fetch_nearest_cache with
        a bulk method that can be optimized later (e.g. parallel reads).

        Args:
            block_hashes: List of block content hashes to load.

        Returns:
            List of per-layer cache lists, one per hash. Missing/corrupt
            blocks return None and are reported via logger.
        """
        results: List[Optional[List[Any]]] = []
        for block_hash in block_hashes:
            block_cache = self.load_block(block_hash)
            results.append(block_cache)
        return results

    def save_block(
        self,
        block_hash: bytes,
        cache_data: List[Any],
        metadata: Dict[str, str],
    ) -> bool:
        """Save a block's KV cache to SSD asynchronously.

        Extracts tensor bytes synchronously on the calling thread (Metal-safe,
        fast), then enqueues for background safetensors writing. Returns
        immediately once extraction succeeds.

        If the background write queue is full, falls back to synchronous
        write (blocks calling thread).

        Args:
            block_hash: Content hash for this block.
            cache_data: Per-layer KV cache objects (already sliced to this block).
            metadata: Dict with keys: model_name, num_layers, token_count,
                      layer_cache_types.

        Returns:
            True if extraction+enqueue succeeded.
        """
        file_path = self._block_path(block_hash)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            tensors_raw, safetensors_meta = self._extract_cache_bytes(
                block_hash, cache_data, metadata
            )
        except Exception as e:
            logger.warning(f"Failed to extract block {block_hash.hex()[:12]}: {e}")
            return False

        # Enqueue for background write (non-blocking, with sync fallback)
        try:
            self._write_queue.put_nowait(
                (block_hash, tensors_raw, safetensors_meta, file_path)
            )
        except queue.Full:
            logger.warning(
                f"Write queue full ({self._write_queue.maxsize}), "
                f"writing block {block_hash.hex()[:12]} synchronously"
            )
            self._write_block_sync(block_hash, tensors_raw, safetensors_meta, file_path)

        with self._lock:
            self._stats["disk_saves"] += 1

        return True

    def save_block_async(
        self,
        block_hash: bytes,
        cache_data: List[Any],
        metadata: Dict[str, str],
    ) -> bool:
        """Alias for save_block. Same interface, explicitly named for clarity."""
        return self.save_block(block_hash, cache_data, metadata)

    def _extract_cache_bytes(
        self,
        block_hash: bytes,
        cache_data: List[Any],
        metadata: Dict[str, str],
    ) -> Tuple[Dict[str, Tuple[bytes, str, List[int]]], Dict[str, str]]:
        """Extract raw tensor bytes from cache data for async writing.

        Must be called from inference thread (Metal-safe after mx.eval).
        Returns (tensors_raw, safetensors_metadata).

        The safetensors_metadata dict includes _cache_info and _cache_classes
        keys (JSON-encoded) that _write_block_sync uses to reconstruct the
        full metadata format expected by load_prompt_cache.
        """
        import mlx.core as mx
        from mlx.utils import tree_flatten

        # Extract states like save_prompt_cache does — cache_data is List[Any]
        # where each element is a KVCache/ArraysCache/etc. object
        cache_states = [c.state for c in cache_data]
        flat = tree_flatten(cache_states)

        safetensors_meta = {
            "model_name": metadata.get("model_name", ""),
            "num_layers": str(metadata.get("num_layers", 0)),
            "token_count": str(metadata.get("token_count", 0)),
            "block_hash": block_hash.hex(),
            "layer_cache_types": json.dumps(metadata.get("layer_cache_types", [])),
            # Store cache info and classes for metadata reconstruction
            "_cache_info": json.dumps(
                [c.meta_state for c in cache_data]
                if cache_data else []
            ),
            "_cache_classes": json.dumps(
                [type(c).__name__ for c in cache_data]
                if cache_data
                else []
            ),
        }

        # Evaluate all tensors first (handles lazy/unevaluated arrays)
        eval_list = [t for _, t in flat if hasattr(t, "dtype") and hasattr(t, "shape")]
        if eval_list:
            mx.eval(*eval_list)

        tensors_raw: Dict[str, Tuple[bytes, str, List[int]]] = {}
        for key_path, tensor in flat:
            if not hasattr(tensor, "dtype") or not hasattr(tensor, "shape"):
                continue
            if _has_zero_dim(tensor):
                continue
            if isinstance(key_path, (list, tuple)):
                name = "/".join(str(p) for p in key_path)
            else:
                name = str(key_path)
            tensors_raw[name] = _extract_tensor_bytes(tensor)

        return tensors_raw, safetensors_meta

    def _start_async_writer(self) -> None:
        """Start the background writer thread to consume the write queue."""
        self._write_queue: queue.Queue = queue.Queue(maxsize=_MAX_PENDING_WRITES)
        self._writer_thread = threading.Thread(
            target=self._async_writer_loop,
            daemon=True,
            name="block-ssd-writer",
        )
        self._writer_thread.start()
        logger.debug("BlockSSDCache async writer started")

    def _async_writer_loop(self) -> None:
        """Background thread: consume from queue and write safetensors files."""
        while True:
            try:
                block_hash, tensors_raw, safetensors_meta, file_path = (
                    self._write_queue.get(timeout=1.0)
                )
            except queue.Empty:
                continue
            try:
                self._write_block_sync(
                    block_hash, tensors_raw, safetensors_meta, file_path
                )
            except Exception as e:
                logger.error(
                    f"Async writer failed for block {block_hash.hex()[:12]}: {e}"
                )
            finally:
                self._write_queue.task_done()

    def _write_block_sync(
        self,
        block_hash: bytes,
        tensors_raw: Dict[str, Tuple[bytes, str, List[int]]],
        safetensors_meta: Dict[str, str],
        file_path: Path,
    ) -> None:
        """Write a safetensors file from pre-extracted raw bytes.

        Called from the background writer thread. No mx/Metal API calls.
        Reconstructs the full metadata format expected by load_prompt_cache,
        then updates the index after successful write.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = file_path.with_suffix(".safetensors.tmp")

        # Reconstruct metadata in save_prompt_cache format for load_prompt_cache compat
        cache_info_raw = json.loads(safetensors_meta.pop("_cache_info", "[]"))
        cache_classes_raw = json.loads(safetensors_meta.pop("_cache_classes", "[]"))

        from mlx.utils import tree_flatten

        full_meta_dict = dict(
            tree_flatten([cache_info_raw, safetensors_meta, cache_classes_raw])
        )

        file_size = _write_safetensors_no_mx(
            str(tmp_path), tensors_raw, full_meta_dict
        )

        tmp_path.rename(file_path)

        # Update index (re-add popped keys for index consistency)
        safetensors_meta["_cache_info"] = json.dumps(cache_info_raw)
        safetensors_meta["_cache_classes"] = json.dumps(cache_classes_raw)
        meta = BlockMeta(
            block_hash=block_hash,
            file_path=file_path,
            file_size=file_size,
            token_count=int(safetensors_meta.get("token_count", str(BLOCK_SIZE))),
            num_layers=int(safetensors_meta.get("num_layers", "0")),
            model_name=safetensors_meta.get("model_name", ""),
            layer_cache_types=(
                json.loads(safetensors_meta["layer_cache_types"])
                if safetensors_meta.get("layer_cache_types")
                else []
            ),
            created_at=time.time(),
            last_access=time.time(),
        )
        self._add_to_index(meta)
        # NOTE: eviction deferred to shrink() called after batch saves complete.
        # Calling _maybe_evict() here triggers the eviction thrashing bug:
        # blocks verified by contains() in _save_blocks_to_ssd() get evicted
        # mid-loop, creating holes in the chain.

    def find_longest_block_chain(
        self,
        tokens: List[int],
        model_name: str,
        start_block: int = 0,
    ) -> Tuple[List[bytes], int]:
        """
        Walk the token sequence from start_block, computing chain hashes and
        checking the SSD index. Return all matching block hashes and the index
        of the first token NOT covered by cached blocks.

        Args:
            tokens: Full token sequence.
            model_name: Model identifier.
            start_block: Which block index to start from (0-based block number).

        Returns:
            (matched_hashes, first_uncached_token_index)

        Example:
            # 100K tokens, blocks 0-3 cached, block 4 different
            hashes, div = cache.find_longest_block_chain(tokens, "qwen3.6")
            # hashes = [hash0, hash1, hash2, hash3]
            # div = 1024  (= 4 * 256, first uncached token)
        """
        parent_hash: Optional[bytes] = None
        matched_hashes: List[bytes] = []

        # If starting from a non-zero block, the caller must provide
        # parent_hash via start_block context. Currently only start_block=0
        # is used — non-zero requires index lookup by caller.

        block_idx = start_block
        while block_idx * BLOCK_SIZE < len(tokens):
            start = block_idx * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, len(tokens))
            block_tokens = tokens[start:end]

            block_hash = compute_block_hash(parent_hash, block_tokens, model_name)

            if self.contains(block_hash):
                matched_hashes.append(block_hash)
                parent_hash = block_hash
                block_idx += 1
            else:
                break

        return matched_hashes, block_idx * BLOCK_SIZE

    def warm_hot_cache(self, n: int = 64) -> int:
        """
        Pre-load the n most recently accessed blocks from SSD into the hot cache.

        This should be called at server startup, before any requests arrive,
        so that the first request finds blocks already in memory instead of
        loading them from SSD during prefill (which is slower than recompute).

        Returns the number of blocks actually loaded.
        """
        if n <= 0:
            return 0

        loaded = 0
        # Get the n most recently accessed blocks from the index
        # (sorted by last_access descending)
        with self._lock:
            sorted_blocks = sorted(
                self._index.values(),
                key=lambda m: m.last_access,
                reverse=True,
            )

        for meta in sorted_blocks[:n]:
            try:
                block_cache = self.load_block(meta.block_hash)
                if block_cache is not None:
                    # load_block already populates the hot cache,
                    # so we just need to call it to warm it.
                    loaded += 1
            except Exception:
                # Skip blocks that fail to load — they'll be recomputed
                pass

        if loaded > 0:
            logger.info(
                f"Warmed hot cache with {loaded}/{n} blocks "
                f"from SSD (total index: {len(self._index)})"
            )
        return loaded

    # ── Internal helpers ────────────────────────────────────────────

    def _block_path(self, block_hash: bytes) -> Path:
        """Two-char prefix subdirectory to avoid flat directory explosion."""
        hex_hash = block_hash.hex()
        return self.blocks_dir / hex_hash[:2] / f"{hex_hash}.safetensors"

    def _add_to_index(self, meta: BlockMeta) -> None:
        with self._lock:
            existing = self._index.get(meta.block_hash)
            if existing is not None:
                self._total_bytes -= existing.file_size
                del self._lru[meta.block_hash]
            self._index[meta.block_hash] = meta
            self._lru[meta.block_hash] = meta.last_access
            self._total_bytes += meta.file_size

    def _remove_from_index(self, block_hash: bytes) -> Optional[BlockMeta]:
        with self._lock:
            meta = self._index.pop(block_hash, None)
            if meta is not None:
                self._lru.pop(block_hash, None)
                self._total_bytes -= meta.file_size
            return meta

    def _maybe_evict(self) -> None:
        """Evict LRU blocks until total size is within limit."""
        if self.max_bytes <= 0:
            return
        with self._lock:
            while self._total_bytes > self.max_bytes and self._lru:
                block_hash = next(iter(self._lru))
                meta = self._index.pop(block_hash, None)
                if meta is not None:
                    self._lru.pop(block_hash, None)
                    self._total_bytes -= meta.file_size
                    meta.file_path.unlink(missing_ok=True)
                    logger.debug(
                        f"Evicted block {block_hash.hex()[:12]} "
                        f"({meta.file_size / 1e6:.1f} MB)"
                    )

    def flush(self) -> None:
        """Wait for all pending async writes to complete and index to be updated.

        Must be called before shrink() or any operation that depends on
        accurate _total_bytes after a batch of save_block() calls.
        """
        self._write_queue.join()

    def shrink(self) -> None:
        """Evict LRU blocks until total size is within max_bytes.

        Call this once after a batch of saves (save_block → flush → shrink)
        instead of evicting eagerly inside _write_block_sync, which causes
        the eviction thrashing bug.
        """
        self._maybe_evict()

    def _load_index(self) -> None:
        """Load index from _index.json, or scan blocks directory."""
        if self._index_path.exists():
            try:
                with open(self._index_path, "r") as f:
                    data = json.load(f)
                with self._lock:
                    for entry_dict in data:
                        meta = self._dict_to_meta(entry_dict)
                        # Skip entries whose files no longer exist
                        if meta.file_path.exists():
                            self._index[meta.block_hash] = meta
                            self._lru[meta.block_hash] = meta.last_access
                            self._total_bytes += meta.file_size
                logger.info(
                    f"Loaded {len(self._index)} blocks "
                    f"({self._total_bytes / 1e9:.2f} GB) from index"
                )
                return
            except (json.JSONDecodeError, OSError, KeyError) as e:
                logger.warning(f"Index corrupted, rebuilding: {e}")
                self._index_path.unlink(missing_ok=True)

        self._scan_blocks_directory()

    def _scan_blocks_directory(self) -> None:
        """Walk blocks/ directory to rebuild index from safetensors files."""
        count = 0
        for fpath in self.blocks_dir.rglob("*.safetensors"):
            try:
                stat = fpath.stat()
                hex_hash = fpath.stem
                block_hash = bytes.fromhex(hex_hash)

                # Read metadata using mx.load() directly
                import mlx.core as mx

                _, cache_metadata = mx.load(str(fpath), return_metadata=True)
                from mlx.utils import tree_unflatten

                meta_nested = tree_unflatten(list(cache_metadata.items()))
                if isinstance(meta_nested, (list, tuple)) and len(meta_nested) >= 2:
                    # (info, metadata_dict, classes) or (info, metadata_dict)
                    if len(meta_nested) >= 2:
                        metadata_dict = meta_nested[1]
                    else:
                        metadata_dict = {}
                    if isinstance(metadata_dict, dict):
                        pass
                    else:
                        metadata_dict = {}
                else:
                    metadata_dict = {}

                meta = BlockMeta(
                    block_hash=block_hash,
                    file_path=fpath,
                    file_size=stat.st_size,
                    token_count=int(metadata_dict.get("token_count", "256")),
                    num_layers=int(metadata_dict.get("num_layers", "0")),
                    model_name=metadata_dict.get("model_name", ""),
                    layer_cache_types=(
                        json.loads(metadata_dict["layer_cache_types"])
                        if metadata_dict.get("layer_cache_types")
                        else []
                    ),
                    created_at=stat.st_mtime,
                    last_access=stat.st_atime,
                )
                with self._lock:
                    self._index[block_hash] = meta
                    self._lru[block_hash] = meta.last_access
                    self._total_bytes += meta.file_size
                count += 1
            except Exception as e:
                logger.debug(f"Skipping {fpath.name}: {e}")
        logger.info(f"Scanned {count} blocks from {self.blocks_dir}")

    def _save_index(self) -> None:
        """Persist index to disk atomically."""
        with self._lock:
            data = [self._meta_to_dict(m) for m in self._index.values()]
        tmp = self._index_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.rename(self._index_path)

    @staticmethod
    def _meta_to_dict(meta: BlockMeta) -> Dict[str, Any]:
        return {
            "block_hash": meta.block_hash.hex(),
            "file_path": str(meta.file_path),
            "file_size": meta.file_size,
            "token_count": meta.token_count,
            "num_layers": meta.num_layers,
            "model_name": meta.model_name,
            "layer_cache_types": meta.layer_cache_types,
            "created_at": meta.created_at,
            "last_access": meta.last_access,
        }

    @staticmethod
    def _dict_to_meta(d: Dict[str, Any]) -> BlockMeta:
        return BlockMeta(
            block_hash=bytes.fromhex(d["block_hash"]),
            file_path=Path(d["file_path"]),
            file_size=d["file_size"],
            token_count=d["token_count"],
            num_layers=d["num_layers"],
            model_name=d.get("model_name", ""),
            layer_cache_types=d.get("layer_cache_types", []),
            created_at=d["created_at"],
            last_access=d["last_access"],
        )

    def save_index(self) -> None:
        """Public wrapper for _save_index. Call after batch modifications."""
        self._save_index()

    @property
    def stats(self) -> Dict[str, int]:
        """Return a snapshot of current usage statistics."""
        with self._lock:
            return dict(self._stats)

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    @property
    def block_count(self) -> int:
        with self._lock:
            return len(self._index)

    def clear(self) -> None:
        """Delete all cached blocks and clear in-memory hot cache."""
        with self._lock:
            for meta in self._index.values():
                meta.file_path.unlink(missing_ok=True)
            self._index.clear()
            self._lru.clear()
            self._total_bytes = 0
            self._hot_cache.clear()
            self._hot_cache_order.clear()
            self._save_index()
