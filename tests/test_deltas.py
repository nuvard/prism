"""Tests for importance deltas (newly_important, no_longer_important)."""

from __future__ import annotations

import pytest

from attention_scores.importance import compute_deltas


def test_deltas_empty_prev() -> None:
    """First step: prev empty -> all current are newly_important."""
    prev: set[int] = set()
    curr = {0, 1, 2}
    newly, no_longer, cnt_new, cnt_no = compute_deltas(prev, curr)
    assert newly == frozenset({0, 1, 2})
    assert no_longer == frozenset()
    assert cnt_new == 3
    assert cnt_no == 0


def test_deltas_same_sets() -> None:
    """Same sets -> no deltas."""
    s = {1, 2, 3}
    newly, no_longer, cnt_new, cnt_no = compute_deltas(s, s)
    assert newly == frozenset()
    assert no_longer == frozenset()
    assert cnt_new == 0
    assert cnt_no == 0


def test_deltas_disjoint() -> None:
    """Fully different: all new, all no longer."""
    prev = {0, 1}
    curr = {2, 3}
    newly, no_longer, cnt_new, cnt_no = compute_deltas(prev, curr)
    assert newly == frozenset({2, 3})
    assert no_longer == frozenset({0, 1})
    assert cnt_new == 2
    assert cnt_no == 2


def test_deltas_partial_overlap() -> None:
    """Some overlap: only diff is new/no_longer."""
    prev = {0, 1, 2}
    curr = {1, 2, 3}
    newly, no_longer, cnt_new, cnt_no = compute_deltas(prev, curr)
    assert newly == frozenset({3})
    assert no_longer == frozenset({0})
    assert cnt_new == 1
    assert cnt_no == 1


def test_deltas_frozenset_input() -> None:
    """Accept frozenset as input."""
    prev = frozenset({0, 1})
    curr = frozenset({1, 2})
    newly, no_longer, cnt_new, cnt_no = compute_deltas(prev, curr)
    assert newly == frozenset({2})
    assert no_longer == frozenset({0})
