"""Integration tests for visualization generate: run_visualization with fake output dir."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from attention_scores.config import Config
from attention_scores.io import write_attention_row_step, write_metadata
from attention_scores.thinking import ThinkingEvent
from visualization.generate import run_visualization


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary pipeline output directory with two request subdirs."""
    out = tmp_path / "output"
    out.mkdir(parents=True, exist_ok=True)

    for idx, request_id in enumerate(("request_0", "request_1")):
        per_step = [
            {
                "step": 0,
                "num_important_tokens": 5 + idx,
                "newly_important_count": 5,
                "no_longer_important_count": 0,
                "sparsity": [[2, 3], [4, 5]],
                "seq_len": 20,
                "sparsity_proportion": 0.75 - idx * 0.05,
            },
            {
                "step": 5,
                "num_important_tokens": 7 + idx,
                "newly_important_count": 2,
                "no_longer_important_count": 1,
                "sparsity": [[3, 4], [5, 6]],
                "seq_len": 25,
                "sparsity_proportion": 0.72 - idx * 0.05,
            },
        ]
        write_metadata(
            out,
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
        # One attention row so score distribution can load something
        row = np.ones((2, 2, 4), dtype=np.float32) / 4
        write_attention_row_step(out, request_id, step=0, attention_row=row)
        write_attention_row_step(out, request_id, step=5, attention_row=row)

    return out


@pytest.fixture
def vis_config(tmp_path: Path, tmp_output_dir: Path) -> Config:
    """Config with output_dir and visualization_output_dir set."""
    vis_dir = tmp_path / "visualization"
    return Config(
        dataset_path="",
        model_path="",
        batch_size=1,
        max_output_len=128,
        save_every_n_steps=5,
        save_when_new_important_above_k=3,
        importance_threshold=0.95,
        thinking_markers=["\\think"],
        output_dir=str(tmp_output_dir),
        save_prefill_attention=False,
        sparsity_threshold=1.0e-6,
        device="auto",
        visualization_output_dir=str(vis_dir),
        visualization_enabled=True,
        visualization_formats=["png"],
    )


def test_run_visualization_creates_files(
    vis_config: Config,
    tmp_path: Path,
) -> None:
    """run_visualization creates expected plot files in visualization_output_dir."""
    paths = run_visualization(vis_config)
    vis_dir = Path(vis_config.visualization_output_dir)
    assert vis_dir.exists()
    assert len(paths) > 0
    for p in paths:
        assert p.exists(), f"Expected file {p}"


def test_run_visualization_per_request_plots(vis_config: Config) -> None:
    """run_visualization produces per-request importance, sparsity, and score_distribution."""
    run_visualization(vis_config)
    vis_dir = Path(vis_config.visualization_output_dir)
    assert (vis_dir / "request_0_importance_dynamics.png").exists()
    assert (vis_dir / "request_0_sparsity_stats.png").exists()
    assert (vis_dir / "request_0_score_distribution.png").exists()
    assert (vis_dir / "request_1_importance_dynamics.png").exists()
    assert (vis_dir / "request_1_sparsity_stats.png").exists()
    assert (vis_dir / "request_1_score_distribution.png").exists()
    assert (vis_dir / "score_distribution_all.png").exists()


def test_run_visualization_disabled_returns_empty(vis_config: Config) -> None:
    """When visualization_enabled is False, run_visualization returns empty list."""
    vis_config.visualization_enabled = False
    paths = run_visualization(vis_config)
    assert paths == []


def test_run_visualization_empty_output_dir(tmp_path: Path) -> None:
    """run_visualization with no request subdirs creates no per-request files."""
    out_dir = tmp_path / "empty_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = tmp_path / "vis"
    config = Config(
        dataset_path="",
        model_path="",
        output_dir=str(out_dir),
        visualization_output_dir=str(vis_dir),
        visualization_enabled=True,
        visualization_formats=["png"],
    )
    paths = run_visualization(config)
    assert vis_dir.exists()
    assert len(paths) == 0
