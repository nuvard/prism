"""Tests for should_save_on_step (N steps and K new important)."""

from __future__ import annotations

import pytest

from attention_scores.importance import should_save_on_step


def test_should_save_first_step() -> None:
    """When last_saved is None, always save (first step)."""
    assert should_save_on_step(0, None, 0, save_every_n_steps=5, save_when_new_important_above_k=3) is True


def test_should_save_every_n() -> None:
    """Save when (step - last_saved) >= N."""
    assert should_save_on_step(5, 0, 0, save_every_n_steps=5, save_when_new_important_above_k=10) is True
    assert should_save_on_step(4, 0, 0, save_every_n_steps=5, save_when_new_important_above_k=10) is False
    assert should_save_on_step(10, 5, 0, save_every_n_steps=5, save_when_new_important_above_k=10) is True


def test_should_save_when_new_above_k() -> None:
    """Save when newly_important_count > K even if N not reached."""
    assert should_save_on_step(1, 0, 5, save_every_n_steps=10, save_when_new_important_above_k=3) is True
    assert should_save_on_step(1, 0, 3, save_every_n_steps=10, save_when_new_important_above_k=3) is False  # not > 3
    assert should_save_on_step(1, 0, 4, save_every_n_steps=10, save_when_new_important_above_k=3) is True
