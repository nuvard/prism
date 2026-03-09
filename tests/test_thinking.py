"""Tests for thinking marker detection."""

from __future__ import annotations

import pytest

from attention_scores.thinking import (
    detect_new_markers_at_step,
    thinking_events_from_step_text_pairs,
)


def test_detect_new_markers_first_appearance() -> None:
    """Marker appears for the first time -> one new event."""
    seen: set[str] = set()
    events, seen = detect_new_markers_at_step("Hello Wait, there", step=1, thinking_markers=["Wait,"], seen_markers=seen)
    assert len(events) == 1
    assert events[0]["marker"] == "Wait,"
    assert events[0]["step"] == 1
    assert "Wait," in seen


def test_detect_new_markers_no_duplicate() -> None:
    """Same marker in longer text -> not reported again."""
    seen: set[str] = set()
    detect_new_markers_at_step("Wait,", step=0, thinking_markers=["Wait,"], seen_markers=seen)
    events, _ = detect_new_markers_at_step("Wait, and more", step=1, thinking_markers=["Wait,"], seen_markers=seen)
    assert len(events) == 0


def test_thinking_events_from_pairs() -> None:
    """Full history yields first-seen events in order."""
    pairs = [
        (0, "Hello"),
        (1, "Hello \\think"),
        (2, "Hello \\think yes"),
    ]
    events = thinking_events_from_step_text_pairs(pairs, thinking_markers=["\\think"])
    assert len(events) == 1
    assert events[0]["marker"] == "\\think"
    assert events[0]["step"] == 1
