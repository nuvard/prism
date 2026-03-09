"""Tests for visualization plots: save figures to disk with artificial data."""

from __future__ import annotations

from pathlib import Path

import pytest

from visualization.aggregate import RequestMetrics
from visualization.plots import (
    plot_importance_and_deltas_dynamics,
    plot_score_distribution,
    plot_sparsity_stats,
)


@pytest.fixture
def sample_metrics() -> RequestMetrics:
    """Sample RequestMetrics for plotting."""
    return RequestMetrics(
        request_id="request_0",
        steps=[0, 5, 10, 15],
        num_important_tokens=[5, 7, 6, 8],
        newly_important_count=[5, 2, 1, 3],
        no_longer_important_count=[0, 1, 2, 1],
        sparsity=[
            [[10, 12], [11, 13]],
            [[11, 13], [12, 14]],
            [[9, 11], [10, 12]],
            [[12, 14], [13, 15]],
        ],
        sparsity_proportion=[0.7, 0.65, 0.72, 0.6],
    )


@pytest.fixture
def tmp_plot_dir(tmp_path: Path) -> Path:
    """Temporary directory for saving plots."""
    d = tmp_path / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_plot_importance_and_deltas_dynamics_saves_files(
    sample_metrics: RequestMetrics,
    tmp_plot_dir: Path,
) -> None:
    """plot_importance_and_deltas_dynamics creates plot files when save_path and formats given."""
    save_path = tmp_plot_dir / "importance_dynamics"
    plot_importance_and_deltas_dynamics(
        sample_metrics,
        save_path=save_path,
        formats=["png"],
    )
    assert (tmp_plot_dir / "importance_dynamics.png").exists()


def test_plot_importance_and_deltas_dynamics_empty_metrics(tmp_plot_dir: Path) -> None:
    """plot_importance_and_deltas_dynamics handles empty steps without error."""
    metrics = RequestMetrics(
        request_id="empty",
        steps=[],
        num_important_tokens=[],
        newly_important_count=[],
        no_longer_important_count=[],
        sparsity=[],
        sparsity_proportion=[],
    )
    save_path = tmp_plot_dir / "empty_dynamics"
    plot_importance_and_deltas_dynamics(metrics, save_path=save_path, formats=["png"])
    assert (tmp_plot_dir / "empty_dynamics.png").exists()


def test_plot_score_distribution_saves_file(tmp_plot_dir: Path) -> None:
    """plot_score_distribution creates histogram file."""
    weights = [0.1, 0.2, 0.3, 0.15, 0.25] * 20
    save_path = tmp_plot_dir / "score_dist"
    plot_score_distribution(
        weights,
        request_id="req0",
        save_path=save_path,
        formats=["png"],
    )
    assert (tmp_plot_dir / "score_dist.png").exists()


def test_plot_score_distribution_empty_weights(tmp_plot_dir: Path) -> None:
    """plot_score_distribution handles empty weights without error."""
    save_path = tmp_plot_dir / "score_empty"
    plot_score_distribution([], request_id="empty", save_path=save_path, formats=["png"])
    assert (tmp_plot_dir / "score_empty.png").exists()


def test_plot_sparsity_stats_saves_file(
    sample_metrics: RequestMetrics,
    tmp_plot_dir: Path,
) -> None:
    """plot_sparsity_stats creates plot files."""
    save_path = tmp_plot_dir / "sparsity_stats"
    plot_sparsity_stats(sample_metrics, save_path=save_path, formats=["png"])
    assert (tmp_plot_dir / "sparsity_stats.png").exists()


def test_plot_sparsity_stats_empty(tmp_plot_dir: Path) -> None:
    """plot_sparsity_stats handles empty sparsity without error."""
    metrics = RequestMetrics(
        request_id="empty",
        steps=[],
        num_important_tokens=[],
        newly_important_count=[],
        no_longer_important_count=[],
        sparsity=[],
        sparsity_proportion=[],
    )
    save_path = tmp_plot_dir / "sparsity_empty"
    plot_sparsity_stats(metrics, save_path=save_path, formats=["png"])
    assert (tmp_plot_dir / "sparsity_empty.png").exists()


def test_plot_functions_return_figure_without_save(sample_metrics: RequestMetrics) -> None:
    """Plot functions return a Figure when save_path is None (no save, no close)."""
    fig = plot_importance_and_deltas_dynamics(sample_metrics, save_path=None, formats=None)
    assert fig is not None
    assert fig.number is not None


def test_plot_importance_and_deltas_dynamics_three_subplots(sample_metrics: RequestMetrics) -> None:
    """plot_importance_and_deltas_dynamics creates three subplots (importance, deltas, sparsity proportion)."""
    fig = plot_importance_and_deltas_dynamics(sample_metrics, save_path=None, formats=None)
    assert len(fig.axes) == 3
