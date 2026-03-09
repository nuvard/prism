"""Tests for importance (0.95 threshold) and sparsity on artificial data."""

from __future__ import annotations

import numpy as np
import pytest

from attention_scores.importance import (
    important_indices,
    importance_from_attention_row,
    sparsity_count_above_threshold,
    sparsity_per_layer_head,
    step_importance_and_sparsity,
)


def test_important_indices_single_peak() -> None:
    """One peak: single position gets weight 1.0 -> one important token."""
    w = np.zeros(10)
    w[3] = 1.0
    indices, count = important_indices(w, threshold=0.95)
    assert indices == frozenset({3})
    assert count == 1
    assert count == len(indices)


def test_important_indices_uniform() -> None:
    """Uniform weights: need all positions to reach 0.95."""
    w = np.ones(5) / 5
    indices, count = important_indices(w, threshold=0.95)
    assert len(indices) == 5
    assert count == 5
    assert count == len(indices)


def test_important_indices_empty() -> None:
    """Empty array returns empty set and 0."""
    w = np.array([])
    indices, count = important_indices(w, threshold=0.95)
    assert indices == frozenset()
    assert count == 0


def test_important_indices_sorted_descending() -> None:
    """Positions taken in descending weight order until cumulative >= 0.95."""
    w = np.array([0.1, 0.5, 0.2, 0.15, 0.05])
    indices, count = important_indices(w, threshold=0.95)
    # Order by weight: 0.5, 0.2, 0.15, 0.1, 0.05 -> cumsum 0.5, 0.7, 0.85, 0.95 -> need 4
    assert len(indices) == 4
    assert count == 4
    assert 1 in indices  # 0.5
    assert 2 in indices  # 0.2
    assert 3 in indices  # 0.15
    assert 0 in indices  # 0.1


def test_num_important_tokens_not_normalized() -> None:
    """Count is raw integer, not divided by seq_len."""
    w = np.ones(100) / 100
    _, count = important_indices(w, threshold=0.95)
    # Minimal positions to reach 0.95: 95 (95 * 0.01 == 0.95)
    assert count == 95
    assert isinstance(count, int)


def test_sparsity_count_above_threshold() -> None:
    """Sparsity is count of positions above threshold (raw)."""
    w = np.array([0.0, 1e-5, 1e-4, 0.1, 0.2])
    c = sparsity_count_above_threshold(w, sparsity_threshold=1e-6)
    assert c == 4
    assert isinstance(c, int)


def test_sparsity_count_empty() -> None:
    """Empty vector -> 0."""
    w = np.array([])
    assert sparsity_count_above_threshold(w, 1e-6) == 0


def test_sparsity_per_layer_head_shape() -> None:
    """Sparsity array shape is (num_layers, num_heads)."""
    # (2 layers, 3 heads, 5 positions)
    arr = np.array([
        [[0.1, 0.2, 0.0, 0.0, 0.7]],
        [[0.0, 0.0, 0.0, 0.0, 1.0]],
    ])
    arr = np.broadcast_to(arr, (2, 3, 5)).copy()
    sp = sparsity_per_layer_head(arr, sparsity_threshold=0.05)
    assert sp.shape == (2, 3)
    assert sp.dtype == np.int32


def test_step_importance_and_sparsity() -> None:
    """Full step returns indices, raw count, and sparsity matrix."""
    # (1 layer, 2 heads, 4 positions)
    row = np.array([[[0.5, 0.3, 0.1, 0.1]], [[0.25, 0.25, 0.25, 0.25]]])
    row = np.transpose(row, (0, 1, 2))  # (2, 1, 4) -> (layers, heads, seq)
    indices, num_important, sparsity = step_importance_and_sparsity(
        row, importance_threshold=0.95, sparsity_threshold=0.01
    )
    assert num_important >= 1
    assert len(indices) == num_important
    assert sparsity.shape == (2, 1)
    assert sparsity[0, 0] == 4  # all above 0.01
