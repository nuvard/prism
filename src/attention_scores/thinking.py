"""Detection of thinking markers (e.g. \\think, Wait,) in generated text."""

from __future__ import annotations

from typing import TypedDict


class ThinkingEvent(TypedDict):
    """First appearance of a thinking marker at a given step."""

    marker: str
    step: int


def detect_new_markers_at_step(
    current_text: str,
    step: int,
    thinking_markers: list[str],
    seen_markers: set[str],
) -> tuple[list[ThinkingEvent], set[str]]:
    """
    Find thinking markers that appear for the first time in current_text at this step.

    Args:
        current_text: Accumulated generated text up to and including this step.
        step: Current decode step (0-based or 1-based; preserved in events).
        thinking_markers: List of marker substrings to detect (e.g. ["\\think", "Wait,"]).
        seen_markers: Set of markers already seen in previous steps (mutated in place).

    Returns:
        (list of new events {"marker": ..., "step": ...}, updated seen_markers).
        The returned set is the same object as seen_markers, updated.
    """
    new_events: list[ThinkingEvent] = []
    for marker in thinking_markers:
        if not marker or marker in seen_markers:
            continue
        if marker in current_text:
            seen_markers.add(marker)
            new_events.append(ThinkingEvent(marker=marker, step=step))
    return (new_events, seen_markers)


def thinking_events_from_step_text_pairs(
    step_text_pairs: list[tuple[int, str]],
    thinking_markers: list[str],
) -> list[ThinkingEvent]:
    """
    Compute first-seen thinking events from a full history of (step, text_after_step).

    Args:
        step_text_pairs: Ordered list of (step, accumulated_text_after_that_step).
        thinking_markers: List of marker substrings to detect.

    Returns:
        List of {"marker": str, "step": int} for first occurrence of each marker.
    """
    seen: set[str] = set()
    all_events: list[ThinkingEvent] = []
    for step, text in step_text_pairs:
        new_events, seen = detect_new_markers_at_step(
            text, step, thinking_markers, seen
        )
        all_events.extend(new_events)
    return all_events
