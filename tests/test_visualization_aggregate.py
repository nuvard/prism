"""Tests for visualization aggregate: discover requests and aggregate metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from attention_scores.io import write_metadata
from attention_scores.thinking import ThinkingEvent
from visualization.aggregate import (
    aggregate_request_metrics,
    discover_request_ids,
)


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary pipeline output directory."""
    return tmp_path / "output"


def test_discover_request_ids_empty(tmp_output_dir: Path) -> None:
    """discover_request_ids returns empty list when dir has no request subdirs."""
    tmp_output_dir.mkdir(parents=True, exist_ok=True)
    assert discover_request_ids(tmp_output_dir) == []


def test_discover_request_ids_nonexistent(tmp_path: Path) -> None:
    """discover_request_ids returns empty list for non-existent directory."""
    out = tmp_path / "nonexistent"
    assert discover_request_ids(out) == []


def test_discover_request_ids_one_request(tmp_output_dir: Path) -> None:
    """discover_request_ids finds one subdir with metadata.json."""
    tmp_output_dir.mkdir(parents=True, exist_ok=True)
    (tmp_output_dir / "request_0").mkdir(parents=True)
    (tmp_output_dir / "request_0" / "metadata.json").write_text("{}", encoding="utf-8")
    assert discover_request_ids(tmp_output_dir) == ["request_0"]


def test_discover_request_ids_two_requests(tmp_output_dir: Path) -> None:
    """discover_request_ids finds multiple subdirs and returns sorted list."""
    tmp_output_dir.mkdir(parents=True, exist_ok=True)
    for rid in ("request_1", "request_0"):
        (tmp_output_dir / rid).mkdir(parents=True)
        (tmp_output_dir / rid / "metadata.json").write_text("{}", encoding="utf-8")
    assert discover_request_ids(tmp_output_dir) == ["request_0", "request_1"]


def test_discover_request_ids_skips_dir_without_metadata(tmp_output_dir: Path) -> None:
    """Subdirs without metadata.json are ignored."""
    tmp_output_dir.mkdir(parents=True, exist_ok=True)
    (tmp_output_dir / "request_0").mkdir(parents=True)
    (tmp_output_dir / "request_0" / "metadata.json").write_text("{}", encoding="utf-8")
    (tmp_output_dir / "other").mkdir(parents=True)
    assert discover_request_ids(tmp_output_dir) == ["request_0"]


def test_aggregate_request_metrics(tmp_output_dir: Path) -> None:
    """aggregate_request_metrics extracts per-step lists from metadata."""
    tmp_output_dir.mkdir(parents=True, exist_ok=True)
    request_id = "request_0"
    per_step = [
        {
            "step": 0,
            "num_important_tokens": 5,
            "newly_important_count": 5,
            "no_longer_important_count": 0,
            "sparsity": [[2, 3], [4, 5]],
            "seq_len": 20,
            "sparsity_proportion": 0.75,
        },
        {
            "step": 5,
            "num_important_tokens": 7,
            "newly_important_count": 2,
            "no_longer_important_count": 1,
            "sparsity": [[3, 4], [5, 6]],
            "seq_len": 25,
            "sparsity_proportion": 0.72,
        },
    ]
    write_metadata(
        tmp_output_dir,
        request_id,
        importance_threshold=0.95,
        save_every_n_steps=5,
        save_when_new_important_above_k=3,
        save_prefill_attention=False,
        thinking_events=[],
        per_step=per_step,
        num_layers=2,
        num_heads=2,
    )
    metrics = aggregate_request_metrics(tmp_output_dir, request_id)
    assert metrics["request_id"] == request_id
    assert metrics["steps"] == [0, 5]
    assert metrics["num_important_tokens"] == [5, 7]
    assert metrics["newly_important_count"] == [5, 2]
    assert metrics["no_longer_important_count"] == [0, 1]
    assert metrics["sparsity"] == [[[2, 3], [4, 5]], [[3, 4], [5, 6]]]
    assert metrics["sparsity_proportion"] == [0.75, 0.72]


def test_aggregate_request_metrics_empty_per_step(tmp_output_dir: Path) -> None:
    """aggregate_request_metrics handles empty per_step."""
    tmp_output_dir.mkdir(parents=True, exist_ok=True)
    request_id = "req_empty"
    write_metadata(
        tmp_output_dir,
        request_id,
        importance_threshold=0.95,
        save_every_n_steps=5,
        save_when_new_important_above_k=3,
        save_prefill_attention=False,
        thinking_events=[],
        per_step=[],
        num_layers=1,
        num_heads=1,
    )
    metrics = aggregate_request_metrics(tmp_output_dir, request_id)
    assert metrics["request_id"] == request_id
    assert metrics["steps"] == []
    assert metrics["num_important_tokens"] == []
    assert metrics["sparsity"] == []
    assert metrics["sparsity_proportion"] == []


def test_aggregate_request_metrics_backfill_sparsity_proportion(tmp_output_dir: Path) -> None:
    """When sparsity_proportion is missing, it is computed from seq_len and num_important_tokens."""
    tmp_output_dir.mkdir(parents=True, exist_ok=True)
    request_id = "request_backfill"
    per_step = [
        {
            "step": 0,
            "num_important_tokens": 2,
            "newly_important_count": 2,
            "no_longer_important_count": 0,
            "sparsity": [[1], [2]],
            "seq_len": 10,
        },
    ]
    write_metadata(
        tmp_output_dir,
        request_id,
        importance_threshold=0.95,
        save_every_n_steps=5,
        save_when_new_important_above_k=3,
        save_prefill_attention=False,
        thinking_events=[],
        per_step=per_step,
        num_layers=1,
        num_heads=2,
    )
    metrics = aggregate_request_metrics(tmp_output_dir, request_id)
    assert metrics["sparsity_proportion"] == [0.8]  # (10 - 2) / 10


def test_aggregate_request_metrics_per_step_every_step(tmp_output_dir: Path) -> None:
    """When per_step has one entry per decode step (N entries), aggregate returns length N."""
    tmp_output_dir.mkdir(parents=True, exist_ok=True)
    request_id = "request_full_steps"
    n_steps = 10
    per_step = [
        {
            "step": i,
            "num_important_tokens": 3 + i,
            "newly_important_count": 1,
            "no_longer_important_count": 0,
            "sparsity": [[1], [1]],
            "seq_len": 20,
            "sparsity_proportion": 0.85,
        }
        for i in range(n_steps)
    ]
    write_metadata(
        tmp_output_dir,
        request_id,
        importance_threshold=0.95,
        save_every_n_steps=5,
        save_when_new_important_above_k=3,
        save_prefill_attention=False,
        thinking_events=[],
        per_step=per_step,
        num_layers=1,
        num_heads=2,
    )
    metrics = aggregate_request_metrics(tmp_output_dir, request_id)
    assert len(metrics["steps"]) == n_steps
    assert len(metrics["num_important_tokens"]) == n_steps
    assert metrics["steps"] == list(range(n_steps))
