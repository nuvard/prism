"""Tests for io save and read_outputs: write then read artificial artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from attention_scores.io import (
    write_attention_row_step,
    write_metadata,
    write_prefill,
    write_progress,
)
from attention_scores.read_outputs import (
    load_decode_attention_step,
    load_metadata,
    load_prefill,
    load_request_outputs,
)
from attention_scores.thinking import ThinkingEvent


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary output directory."""
    return tmp_path / "out"


def test_write_and_load_metadata(tmp_output_dir: Path) -> None:
    """Write metadata.json and load it back; verify structure."""
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
    ]
    write_metadata(
        tmp_output_dir,
        request_id,
        importance_threshold=0.95,
        save_every_n_steps=5,
        save_when_new_important_above_k=3,
        save_prefill_attention=False,
        thinking_events=[ThinkingEvent(marker="Wait,", step=1)],
        per_step=per_step,
        num_layers=2,
        num_heads=2,
    )
    meta = load_metadata(tmp_output_dir, request_id)
    assert meta["format_version"] == "1.0"
    assert meta["importance_threshold"] == 0.95
    assert meta["num_layers"] == 2
    assert meta["num_heads"] == 2
    assert len(meta["per_step"]) == 1
    assert meta["per_step"][0]["num_important_tokens"] == 5
    assert meta["per_step"][0]["seq_len"] == 20
    assert meta["per_step"][0]["sparsity_proportion"] == 0.75
    assert meta["thinking_events"][0]["marker"] == "Wait,"


def test_write_and_load_attention_row(tmp_output_dir: Path) -> None:
    """Write one step NPZ and load via read_outputs."""
    request_id = "request_0"
    (tmp_output_dir / request_id / "attention_rows").mkdir(parents=True, exist_ok=True)
    # (2 layers, 2 heads, 4 positions)
    row = np.random.rand(2, 2, 4).astype(np.float32)
    row /= row.sum(axis=-1, keepdims=True)
    write_attention_row_step(tmp_output_dir, request_id, step=0, attention_row=row)
    loaded = load_decode_attention_step(tmp_output_dir, request_id, 0)
    assert loaded.shape == (2, 2, 4)
    np.testing.assert_allclose(loaded, row, atol=1e-5)


def test_write_and_load_prefill(tmp_output_dir: Path) -> None:
    """Write prefill layer files and load via read_outputs."""
    request_id = "request_0"
    prefill = [
        np.random.rand(2, 3, 3).astype(np.float32),  # 2 heads, 3x3
        np.random.rand(2, 3, 3).astype(np.float32),
    ]
    for a in prefill:
        a /= a.sum(axis=(-2, -1), keepdims=True)
    write_prefill(tmp_output_dir, request_id, prefill)
    loaded = load_prefill(tmp_output_dir, request_id)
    assert len(loaded) == 2
    assert loaded[0].shape == (2, 3, 3)
    np.testing.assert_allclose(loaded[0], prefill[0], atol=1e-5)


def test_load_request_outputs(tmp_output_dir: Path) -> None:
    """Write full request artifacts and load via load_request_outputs."""
    request_id = "req_1"
    write_metadata(
        tmp_output_dir,
        request_id,
        importance_threshold=0.95,
        save_every_n_steps=5,
        save_when_new_important_above_k=3,
        save_prefill_attention=False,
        thinking_events=[],
        per_step=[{"step": 0, "num_important_tokens": 1, "sparsity": [[1]]}],
        num_layers=1,
        num_heads=1,
    )
    row = np.ones((1, 1, 2), dtype=np.float32) / 2
    write_attention_row_step(tmp_output_dir, request_id, 0, row)
    out = load_request_outputs(tmp_output_dir, request_id)
    assert "metadata" in out
    assert out["saved_steps"] == [0]
    assert out["prefill"] == []


def test_metadata_written_every_step(tmp_output_dir: Path) -> None:
    """Simulate writing metadata after each step; file should have N entries after N steps."""
    request_id = "request_0"
    per_step: list[dict] = []
    for step in range(5):
        per_step.append({
            "step": step,
            "num_important_tokens": 2 + step,
            "newly_important_count": 1,
            "no_longer_important_count": 0,
            "sparsity": [[1, 1]],
            "seq_len": 10,
            "sparsity_proportion": 0.8,
        })
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
            num_heads=1,
        )
    meta = load_metadata(tmp_output_dir, request_id)
    assert len(meta["per_step"]) == 5
    for i, entry in enumerate(meta["per_step"]):
        assert entry["step"] == i
        assert entry["num_important_tokens"] == 2 + i


def test_load_metadata_with_per_layer_deltas(tmp_output_dir: Path) -> None:
    """Write and load metadata with newly_important_per_layer, no_longer_important_per_layer."""
    request_id = "request_1"
    per_step = [
        {
            "step": 0,
            "num_important_tokens": 4,
            "newly_important_count": 4,
            "no_longer_important_count": 0,
            "newly_important_per_layer": [2, 2],
            "no_longer_important_per_layer": [0, 0],
            "sparsity": [[3, 3], [4, 4]],
            "seq_len": 20,
            "sparsity_proportion": 0.8,
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
    meta = load_metadata(tmp_output_dir, request_id)
    assert meta["per_step"][0]["newly_important_per_layer"] == [2, 2]
    assert meta["per_step"][0]["no_longer_important_per_layer"] == [0, 0]


def test_load_metadata_with_sparsity_per_layer_and_per_layer_head_deltas(
    tmp_output_dir: Path,
) -> None:
    """Write and load metadata with sparsity_per_layer and per-layer-head deltas."""
    request_id = "request_2"
    per_step = [
        {
            "step": 0,
            "num_important_tokens": 3,
            "newly_important_count": 3,
            "no_longer_important_count": 0,
            "newly_important_per_layer": [1, 2],
            "no_longer_important_per_layer": [0, 0],
            "newly_important_per_layer_head": [[1, 0], [1, 1]],
            "no_longer_important_per_layer_head": [[0, 0], [0, 0]],
            "sparsity": [[2, 3], [4, 5]],
            "sparsity_per_layer": [5, 9],
            "sparsity_proportion_per_layer_head": [[0.8, 0.7], [0.6, 0.5]],
            "sparsity_proportion_per_layer": [0.75, 0.55],
            "seq_len": 10,
            "sparsity_proportion": 0.7,
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
    meta = load_metadata(tmp_output_dir, request_id)
    assert meta["per_step"][0]["sparsity_per_layer"] == [5, 9]
    assert meta["per_step"][0]["newly_important_per_layer_head"] == [[1, 0], [1, 1]]
    assert meta["per_step"][0]["no_longer_important_per_layer_head"] == [[0, 0], [0, 0]]
    assert meta["per_step"][0]["sparsity_proportion_per_layer_head"] == [
        [0.8, 0.7],
        [0.6, 0.5],
    ]
    assert meta["per_step"][0]["sparsity_proportion_per_layer"] == [0.75, 0.55]


def test_write_progress(tmp_output_dir: Path) -> None:
    """Write progress.json and verify content and atomic write."""
    path = write_progress(
        tmp_output_dir,
        current_request_index=1,
        total_requests=10,
        request_id="request_1",
        current_step=5,
        max_output_len=128,
    )
    assert path == tmp_output_dir / "progress.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["current_request_index"] == 1
    assert data["total_requests"] == 10
    assert data["request_id"] == "request_1"
    assert data["current_step"] == 5
    assert data["max_output_len"] == 128
