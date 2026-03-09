"""
Importance (0.95 threshold), deltas, sparsity per layer/head, and save-condition logic.

All counts are raw (no normalization by seq_len). Uses aggregated attention
(e.g. mean over heads/layers) for importance; sparsity is per (layer, head).
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np

ArrayLike = Union[np.ndarray, Any]


def important_indices(
    weights_1d: ArrayLike,
    threshold: float = 0.95,
) -> tuple[frozenset[int], int]:
    """
    Compute the minimal set of positions whose cumulative weight (descending) >= threshold.

    Args:
        weights_1d: One-dimensional attention weights (e.g. aggregated over heads).
        threshold: Cumulative weight threshold (e.g. 0.95).

    Returns:
        (frozenset of important indices, raw count as int). Count is not normalized.
    """
    w = np.asarray(weights_1d, dtype=np.float64).ravel()
    if w.size == 0:
        return (frozenset(), 0)
    order = np.argsort(w)[::-1]
    cumsum = np.cumsum(w[order])
    n = int(np.searchsorted(cumsum, threshold, side="left")) + 1
    n = min(n, order.size)
    indices = frozenset(int(order[i]) for i in range(n))
    return (indices, len(indices))


def compute_deltas(
    prev_important: set[int] | frozenset[int],
    curr_important: set[int] | frozenset[int],
) -> tuple[frozenset[int], frozenset[int], int, int]:
    """
    Compute newly important and no-longer-important token indices between two steps.

    Args:
        prev_important: Set of important indices at the previous step.
        curr_important: Set of important indices at the current step.

    Returns:
        (newly_important, no_longer_important, count_newly, count_no_longer).
        Counts are raw integers (no normalization).
    """
    curr = frozenset(curr_important)
    prev = frozenset(prev_important)
    newly = curr - prev
    no_longer = prev - curr
    return (newly, no_longer, len(newly), len(no_longer))


def important_indices_per_layer_head(
    attention_row: ArrayLike,
    importance_threshold: float = 0.95,
) -> list[list[frozenset[int]]]:
    """
    Compute important indices for each (layer, head) from the attention row.

    For each layer and head, applies important_indices to that head's row
    (attention_row[layer, head, :]).

    Args:
        attention_row: Array of shape (num_layers, num_heads, seq_len).
        importance_threshold: Cumulative weight threshold (e.g. 0.95).

    Returns:
        Nested list: result[layer][head] is frozenset of important position indices.
    """
    arr = np.asarray(attention_row, dtype=np.float64)
    if arr.ndim != 3:
        return []
    n_layers, n_heads, _ = arr.shape
    out: list[list[frozenset[int]]] = []
    for L in range(n_layers):
        layer_heads: list[frozenset[int]] = []
        for H in range(n_heads):
            indices, _ = important_indices(arr[L, H, :], importance_threshold)
            layer_heads.append(indices)
        out.append(layer_heads)
    return out


def layer_important_union(
    important_per_layer_head: list[list[frozenset[int]]],
) -> list[frozenset[int]]:
    """
    For each layer, compute the union of important indices across all heads.

    Layer-level "important" = important for at least one head in that layer.

    Args:
        important_per_layer_head: result[layer][head] is frozenset of important indices.

    Returns:
        List of length num_layers: union of head sets per layer.
    """
    return [
        frozenset().union(*heads) if heads else frozenset()
        for heads in important_per_layer_head
    ]


def compute_deltas_per_layer(
    prev_per_layer: list[frozenset[int]],
    curr_per_layer: list[frozenset[int]],
) -> tuple[
    list[frozenset[int]],
    list[frozenset[int]],
    list[int],
    list[int],
]:
    """
    Compute newly_important and no_longer_important per layer between two steps.

    For each layer, applies the same logic as compute_deltas to the layer's
    important set (union over heads).

    Args:
        prev_per_layer: List of important index sets per layer at previous step.
        curr_per_layer: List of important index sets per layer at current step.

    Returns:
        (newly_per_layer, no_longer_per_layer, count_new_per_layer, count_no_longer_per_layer).
        Each list has length = number of layers. Counts are raw integers.
    """
    n = max(len(prev_per_layer), len(curr_per_layer))
    newly_per_layer: list[frozenset[int]] = []
    no_longer_per_layer: list[frozenset[int]] = []
    count_new_per_layer: list[int] = []
    count_no_longer_per_layer: list[int] = []
    for i in range(n):
        prev_set = prev_per_layer[i] if i < len(prev_per_layer) else frozenset()
        curr_set = curr_per_layer[i] if i < len(curr_per_layer) else frozenset()
        newly, no_longer, cn, cl = compute_deltas(prev_set, curr_set)
        newly_per_layer.append(newly)
        no_longer_per_layer.append(no_longer)
        count_new_per_layer.append(cn)
        count_no_longer_per_layer.append(cl)
    return (
        newly_per_layer,
        no_longer_per_layer,
        count_new_per_layer,
        count_no_longer_per_layer,
    )


def sparsity_count_above_threshold(
    weights_1d: ArrayLike,
    sparsity_threshold: float,
) -> int:
    """
    Number of positions with weight strictly above sparsity_threshold (raw count).

    Args:
        weights_1d: One-dimensional attention weights (e.g. one head's row).
        sparsity_threshold: Minimum weight to count (e.g. 1e-6).

    Returns:
        Raw count of positions with weight > sparsity_threshold. Not normalized.
    """
    w = np.asarray(weights_1d, dtype=np.float64).ravel()
    if w.size == 0:
        return 0
    return int(np.count_nonzero(w > sparsity_threshold))


def sparsity_per_layer_head(
    attention_row: ArrayLike,
    sparsity_threshold: float,
) -> np.ndarray:
    """
    Compute sparsity (count above threshold) for each (layer, head).

    Args:
        attention_row: Array of shape (num_layers, num_heads, seq_len).
        sparsity_threshold: Threshold for sparsity metric.

    Returns:
        Array of shape (num_layers, num_heads) of dtype int (raw counts).
    """
    arr = np.asarray(attention_row, dtype=np.float64)
    above = arr > sparsity_threshold
    return np.count_nonzero(above, axis=-1).astype(np.int32)


def sparsity_proportion(num_important: int, seq_len: int) -> float:
    """
    Proportion of unimportant tokens to total (sparsity as fraction).

    Args:
        num_important: Number of important token positions (e.g. from importance threshold).
        seq_len: Total sequence length (number of tokens).

    Returns:
        (seq_len - num_important) / seq_len when seq_len > 0, else 0.0.
    """
    if seq_len <= 0:
        return 0.0
    return (seq_len - num_important) / seq_len


def aggregate_attention_row_for_importance(
    attention_row: ArrayLike,
) -> np.ndarray:
    """
    Aggregate attention row over layers and heads to one vector (mean).

    Used to compute a single importance set per step from full (layers, heads, seq_len).

    Args:
        attention_row: Shape (num_layers, num_heads, seq_len) or (num_heads, seq_len).

    Returns:
        One-dimensional array of shape (seq_len,).
    """
    arr = np.asarray(attention_row, dtype=np.float64)
    return arr.mean(axis=tuple(range(arr.ndim - 1)))


def importance_from_attention_row(
    attention_row: ArrayLike,
    importance_threshold: float = 0.95,
) -> tuple[frozenset[int], int]:
    """
    Compute important indices and raw count from full attention row (all layers/heads).

    Aggregates by mean over layers and heads, then applies importance_threshold.

    Args:
        attention_row: Shape (num_layers, num_heads, seq_len) or (num_heads, seq_len).
        importance_threshold: Cumulative weight threshold (e.g. 0.95).

    Returns:
        (frozenset of important indices, raw count). Count is not normalized.
    """
    agg = aggregate_attention_row_for_importance(attention_row)
    return important_indices(agg, importance_threshold)


def should_save_on_step(
    step: int,
    last_saved_step: int | None,
    newly_important_count: int,
    save_every_n_steps: int,
    save_when_new_important_above_k: int,
) -> bool:
    """
    Decide whether to save attention row on this step.

    Save when: (step - last_saved) >= save_every_n_steps, or
    newly_important_count > save_when_new_important_above_k.

    Args:
        step: Current decode step (1-based or 0-based; consistency with caller).
        last_saved_step: Last step at which we saved, or None if none yet.
        newly_important_count: Number of newly important tokens this step.
        save_every_n_steps: Save at least every N steps.
        save_when_new_important_above_k: Save when new important count > K.

    Returns:
        True if we should save on this step.
    """
    if newly_important_count > save_when_new_important_above_k:
        return True
    if last_saved_step is None:
        return True
    return (step - last_saved_step) >= save_every_n_steps


def step_importance_and_sparsity(
    attention_row: ArrayLike,
    importance_threshold: float,
    sparsity_threshold: float,
) -> tuple[frozenset[int], int, np.ndarray]:
    """
    Compute importance set, num_important_tokens (raw), and sparsity per (layer, head).

    Args:
        attention_row: (num_layers, num_heads, seq_len).
        importance_threshold: For important_indices (e.g. 0.95).
        sparsity_threshold: For sparsity count per head.

    Returns:
        (important_indices, num_important_tokens_raw, sparsity_array (num_layers, num_heads)).
    """
    indices, num_important = importance_from_attention_row(
        attention_row, importance_threshold
    )
    sparsity = sparsity_per_layer_head(attention_row, sparsity_threshold)
    return (indices, num_important, sparsity)
