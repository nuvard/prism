"""Tests for per-layer importance and deltas (union over heads, deltas per layer)."""

from __future__ import annotations

import numpy as np
import pytest

from attention_scores.importance import (
    compute_deltas_per_layer,
    compute_deltas_per_layer_head,
    important_indices_per_layer_head,
    layer_important_union,
)


def test_important_indices_per_layer_head_shape() -> None:
    """Result is list of list of frozensets; shape (num_layers, num_heads)."""
    # 2 layers, 3 heads, 5 positions
    row = np.random.rand(2, 3, 5).astype(np.float64)
    row /= row.sum(axis=-1, keepdims=True)
    out = important_indices_per_layer_head(row, importance_threshold=0.95)
    assert len(out) == 2
    assert len(out[0]) == 3
    assert len(out[1]) == 3
    for layer in out:
        for head_set in layer:
            assert isinstance(head_set, frozenset)
            assert all(isinstance(i, int) for i in head_set)


def test_layer_important_union_single_head() -> None:
    """One head per layer: union is that head's set."""
    per_lh = [[frozenset({0, 1})], [frozenset({1, 2})]]
    union = layer_important_union(per_lh)
    assert union == [frozenset({0, 1}), frozenset({1, 2})]


def test_layer_important_union_two_heads() -> None:
    """Two heads: union is union of both sets."""
    per_lh = [
        [frozenset({0, 1}), frozenset({1, 2})],
    ]
    union = layer_important_union(per_lh)
    assert union == [frozenset({0, 1, 2})]


def test_layer_important_union_empty() -> None:
    """Empty list of layers."""
    assert layer_important_union([]) == []


def test_compute_deltas_per_layer_first_step() -> None:
    """First step: prev empty, curr has sets -> all newly important per layer."""
    prev = [frozenset(), frozenset()]
    curr = [frozenset({0, 1}), frozenset({2, 3})]
    newly_list, no_longer_list, count_new, count_no_longer = compute_deltas_per_layer(
        prev, curr
    )
    assert count_new == [2, 2]
    assert count_no_longer == [0, 0]
    assert newly_list == [frozenset({0, 1}), frozenset({2, 3})]
    assert no_longer_list == [frozenset(), frozenset()]


def test_compute_deltas_per_layer_same_sets() -> None:
    """Same sets -> no deltas."""
    s0 = frozenset({0, 1})
    s1 = frozenset({2, 3})
    prev = [s0, s1]
    curr = [s0, s1]
    newly_list, no_longer_list, count_new, count_no_longer = compute_deltas_per_layer(
        prev, curr
    )
    assert count_new == [0, 0]
    assert count_no_longer == [0, 0]
    assert newly_list == [frozenset(), frozenset()]
    assert no_longer_list == [frozenset(), frozenset()]


def test_compute_deltas_per_layer_one_new_one_removed() -> None:
    """Layer 0: +3 (new), -0; layer 1: +0, -2 (no longer: 2, since curr has only {3})."""
    prev = [frozenset({0, 1}), frozenset({2, 3})]
    curr = [frozenset({0, 1, 3}), frozenset({3})]
    newly_list, no_longer_list, count_new, count_no_longer = compute_deltas_per_layer(
        prev, curr
    )
    assert count_new == [1, 0]
    assert count_no_longer == [0, 1]  # layer 1: prev - curr = {2, 3} - {3} = {2}
    assert newly_list == [frozenset({3}), frozenset()]
    assert no_longer_list == [frozenset(), frozenset({2})]


def test_compute_deltas_per_layer_length_mismatch() -> None:
    """curr longer than prev: missing prev layers treated as empty."""
    prev = [frozenset({0})]
    curr = [frozenset({0}), frozenset({1})]
    newly_list, no_longer_list, count_new, count_no_longer = compute_deltas_per_layer(
        prev, curr
    )
    assert len(count_new) == 2
    assert count_new == [0, 1]
    assert count_no_longer == [0, 0]


def test_compute_deltas_per_layer_head_first_step() -> None:
    """First step: prev empty -> all newly important per head, no no_longer."""
    prev = [[frozenset(), frozenset()], [frozenset(), frozenset()]]
    curr = [[frozenset({0, 1}), frozenset({2})], [frozenset({3}), frozenset({4, 5})]]
    count_new, count_no_longer = compute_deltas_per_layer_head(prev, curr)
    assert count_new == [[2, 1], [1, 2]]
    assert count_no_longer == [[0, 0], [0, 0]]


def test_compute_deltas_per_layer_head_same_sets() -> None:
    """Same sets per head -> no deltas."""
    s = [[frozenset({0, 1}), frozenset({2})], [frozenset({3})]]
    count_new, count_no_longer = compute_deltas_per_layer_head(s, s)
    assert count_new == [[0, 0], [0]]
    assert count_no_longer == [[0, 0], [0]]


def test_compute_deltas_per_layer_head_one_new_one_removed() -> None:
    """One head gains a token, another loses one."""
    prev = [[frozenset({0, 1}), frozenset({2, 3})]]
    curr = [[frozenset({0, 1, 2}), frozenset({3})]]  # head0: +2; head1: -2
    count_new, count_no_longer = compute_deltas_per_layer_head(prev, curr)
    assert count_new == [[1, 0]]
    assert count_no_longer == [[0, 1]]


def test_compute_deltas_per_layer_head_length_mismatch() -> None:
    """curr has more layers/heads: missing prev treated as empty."""
    prev = [[frozenset({0})]]
    curr = [[frozenset({0}), frozenset({1})], [frozenset({2})]]
    count_new, count_no_longer = compute_deltas_per_layer_head(prev, curr)
    assert count_new == [[0, 1], [1]]
    assert count_no_longer == [[0, 0], [0]]


def test_full_flow_two_steps_artificial() -> None:
    """Two steps: step0 all weight on pos 0; step1 all weight on pos 1."""
    # Step 0: (1 layer, 2 heads, 3 positions) — head0 and head1 both have weight on 0
    row0 = np.zeros((1, 2, 3), dtype=np.float64)
    row0[0, 0, 0] = 1.0
    row0[0, 1, 0] = 1.0
    # Step 1: weight on position 1
    row1 = np.zeros((1, 2, 3), dtype=np.float64)
    row1[0, 0, 1] = 1.0
    row1[0, 1, 1] = 1.0

    lh0 = important_indices_per_layer_head(row0, 0.95)
    lh1 = important_indices_per_layer_head(row1, 0.95)
    curr0 = layer_important_union(lh0)
    curr1 = layer_important_union(lh1)
    prev = [frozenset() for _ in curr0]
    _, _, count_new_0, count_no_0 = compute_deltas_per_layer(prev, curr0)
    _, _, count_new_1, count_no_1 = compute_deltas_per_layer(curr0, curr1)

    assert count_new_0 == [1]
    assert count_no_0 == [0]
    assert count_new_1 == [1]
    assert count_no_1 == [1]
