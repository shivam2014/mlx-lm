# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Apple Inc.

"""Chat-template boundary detection via token-ID scanning.

Resolves marker tokens (e.g. ``<|im_start|>``, ``<|im_end|>``) once at
server startup, then finds all turn boundaries by scanning the encoded
token array directly — no ``apply_chat_template`` re-encoding needed.

Supports Qwen (``<|im_start|>``/``<|im_end|>``), Llama 3+
(``<|start_header_id|>``/``<|eot_id|>``), and any model family where you
provide the marker token strings.
"""

from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Marker resolution
# ---------------------------------------------------------------------------

MarkerIds = Tuple[int, int, Optional[int]]
"""``(turn_start_id, turn_end_id, system_role_id)`` — the three token IDs
needed by ``find_all_boundaries``.  ``system_role_id`` may be ``None`` for
templates where the role name is multi-token (e.g. ``<|start_header_id|>system``
encodes as two tokens)."""


def resolve_marker_ids(
    tokenizer,
    *,
    turn_start: str = "<|im_start|>",
    turn_end: str = "<|im_end|>",
    role: str = "system",
) -> MarkerIds:
    """Encode marker strings into single token IDs.

    Raises ``ValueError`` if a marker string encodes to zero or multiple
    tokens — this indicates the tokenizer doesn't support the expected
    chat-template format.  Caller should fall back to the legacy
    ``apply_chat_template`` re-encoding path.

    Parameters
    ----------
    tokenizer:
        HuggingFace ``PreTrainedTokenizer`` (or ``TokenizerWrapper``).
    turn_start:
        Token that begins a new role's content (e.g. ``<|im_start|>``).
    turn_end:
        Token that ends a role's content (e.g. ``<|im_end|>``).
    role:
        The system-role string to match after ``turn_start``
        (e.g. ``"system"``).  May be ``None`` to skip role-name verification.

    Returns
    -------
    ``(turn_start_id, turn_end_id, system_role_id)``
    """
    start_ids = tokenizer.encode(turn_start, add_special_tokens=False)
    end_ids = tokenizer.encode(turn_end, add_special_tokens=False)

    if len(start_ids) != 1:
        raise ValueError(
            f"Expected single-token turn_start ({turn_start!r}); "
            f"got {len(start_ids)} tokens"
        )
    if len(end_ids) != 1:
        raise ValueError(
            f"Expected single-token turn_end ({turn_end!r}); "
            f"got {len(end_ids)} tokens"
        )

    if role is not None:
        role_ids = tokenizer.encode(role, add_special_tokens=False)
        role_id = role_ids[0] if len(role_ids) == 1 else None
    else:
        role_id = None

    return start_ids[0], end_ids[0], role_id


# ---------------------------------------------------------------------------
# Boundary scanning
# ---------------------------------------------------------------------------


def find_system_boundary(
    ids: List[int],
    start_id: int,
    end_id: int,
    role_id: Optional[int] = None,
) -> int:
    """Return index *after* the first system-message boundary, or -1.

    Scans ``ids`` for the pattern::

        <start_id> [role_id] ... <end_id> [gap] <start_id>

    and returns the index right after that second ``<start_id>``.
    ``ids[:boundary]`` is the system-prefix cache key; ``ids[boundary:]``
    is the per-request suffix.

    The gap between ``<end_id>`` and the next ``<start_id>`` may be up to
    2 tokens (covers the single ``\\n`` separator that chat templates often
    insert between turns).

    Parameters
    ----------
    ids:
        Encoded token ids (full prompt).
    start_id:
        Token id for ``<|im_start|>``, ``<|start_header_id|>``, etc.
    end_id:
        Token id for ``<|im_end|>``, ``<|eot_id|>``, etc.
    role_id:
        Token id for ``"system"``, or ``None`` to skip role verification.

    Returns
    -------
    Boundary index, or -1 if no system message detected.
    """
    # Locate the opening <start_id> [role_id] sequence.
    sys_idx = -1
    for i in range(len(ids) - 1):
        if ids[i] == start_id:
            if role_id is None or ids[i + 1] == role_id:
                sys_idx = i
                break
    if sys_idx < 0:
        return -1

    # Walk forward: find <end_id>, then locate the next <start_id>
    # within 2 tokens (handles \n separator between turns).
    for i in range(sys_idx + 1, len(ids)):
        if ids[i] == end_id:
            for j in range(i + 1, min(i + 3, len(ids))):
                if ids[j] == start_id:
                    return j + 1
            return -1  # <end_id> without a subsequent <start_id>
    return -1


def find_all_boundaries(
    ids: List[int],
    start_id: int,
    end_id: int,
    role_id: Optional[int] = None,
) -> List[int]:
    """Return *all* turn-boundary cut points in ascending order.

    Each cut point is the index *after* an ``<start_id>`` that begins a new
    role's content.  The first cut is the system-prompt boundary; subsequent
    cuts are at every ``<end_id>`` + ``<start_id>`` pair after that.

    Returns an empty list if no system message is found.
    """
    boundaries: List[int] = []

    # Locate the opening <start_id> [role_id] sequence.
    sys_idx = -1
    for i in range(len(ids) - 1):
        if ids[i] == start_id:
            if role_id is None or ids[i + 1] == role_id:
                sys_idx = i
                break
    if sys_idx < 0:
        return boundaries

    # Walk forward from sys_idx.
    i = sys_idx + 1
    while i < len(ids):
        if ids[i] == end_id:
            found = False
            for j in range(i + 1, min(i + 3, len(ids))):
                if ids[j] == start_id:
                    boundaries.append(j + 1)
                    i = j + 1
                    found = True
                    break
            if not found:
                break  # no more role transitions
        else:
            i += 1
    return boundaries
