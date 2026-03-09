"""
Plotting functions for importance dynamics, score distribution, and sparsity.

Uses matplotlib; figures are saved to disk by the caller or via save path.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visualization.aggregate import RequestMetrics

if TYPE_CHECKING:
    from typing import Any


def plot_importance_and_deltas_dynamics(
    metrics: RequestMetrics,
    save_path: Path | None = None,
    formats: list[str] | None = None,
) -> plt.Figure:
    """
    Plot step vs num_important_tokens, deltas, and sparsity proportion.

    Three subplots: (1) importance count, (2) delta counts, (3) sparsity proportion (unimportant/total).

    Args:
        metrics: Aggregated per-step metrics for one request.
        save_path: If set, save figure to this path (without extension).
        formats: List of extensions (e.g. ['png', 'svg']). Used only if save_path is set.

    Returns:
        The created matplotlib Figure.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    steps = metrics["steps"]
    if not steps:
        fig.suptitle(f"Request {metrics['request_id']} (no saved steps)")
        if save_path and formats:
            for ext in formats:
                fig.savefig(save_path.with_suffix(f".{ext}"), bbox_inches="tight")
                plt.close(fig)
        return fig

    ax1.plot(steps, metrics["num_important_tokens"], marker="o", markersize=3, label="num_important_tokens")
    ax1.set_ylabel("Count")
    ax1.set_title("Important tokens (0.95 cumulative) per step")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, metrics["newly_important_count"], marker="s", markersize=3, label="newly important")
    ax2.plot(steps, metrics["no_longer_important_count"], marker="^", markersize=3, label="no longer important")
    ax2.set_ylabel("Count")
    ax2.set_title("Delta: new and removed important tokens per step")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    sparsity_prop = metrics.get("sparsity_proportion", [])
    if sparsity_prop and len(sparsity_prop) == len(steps):
        ax3.plot(steps, sparsity_prop, marker="o", markersize=3, color="green", label="sparsity (unimportant/total)")
        ax3.set_ylim(0, 1)
    else:
        ax3.set_title("Sparsity proportion (no data)")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Sparsity proportion")
    ax3.set_title("Sparsity proportion (unimportant/total) per step")
    ax3.grid(True, alpha=0.3)
    if sparsity_prop and len(sparsity_prop) == len(steps):
        ax3.legend(loc="upper right")

    fig.suptitle(f"Request {metrics['request_id']}")
    fig.tight_layout()

    if save_path and formats:
        for ext in formats:
            fig.savefig(save_path.with_suffix(f".{ext}"), bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_score_distribution(
    weights: list[float],
    request_id: str = "",
    save_path: Path | None = None,
    formats: list[str] | None = None,
) -> plt.Figure:
    """
    Plot histogram of attention weights (score distribution).

    Args:
        weights: Flat list of attention weight values.
        request_id: Optional label for the title.
        save_path: If set, save figure to this path (without extension).
        formats: List of extensions. Used only if save_path is set.

    Returns:
        The created matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    if not weights:
        ax.set_title(f"Score distribution {request_id} (no data)")
        if save_path and formats:
            for ext in formats:
                fig.savefig(save_path.with_suffix(f".{ext}"), bbox_inches="tight")
            plt.close(fig)
        return fig

    arr = np.array(weights, dtype=np.float64)
    arr = arr[arr > 0]  # exclude exact zeros for log scale if needed
    if arr.size == 0:
        ax.set_title(f"Score distribution {request_id} (all zeros)")
    else:
        ax.hist(arr, bins=min(100, max(20, arr.size // 50)), alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.set_xlabel("Attention weight")
        ax.set_ylabel("Count")
        ax.set_title(f"Score distribution {request_id}".strip() or "Score distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path and formats:
        for ext in formats:
            fig.savefig(save_path.with_suffix(f".{ext}"), bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_sparsity_stats(
    metrics: RequestMetrics,
    save_path: Path | None = None,
    formats: list[str] | None = None,
) -> plt.Figure:
    """
    Plot sparsity statistics over steps: mean over layer/head, heatmap, and sparsity proportion.

    For each saved step, compute mean sparsity across all layer/head cells;
    plot mean over steps, heatmap for last step, and sparsity proportion (unimportant/total).

    Args:
        metrics: Aggregated per-step metrics (sparsity per step, sparsity_proportion).
        save_path: If set, save figure to this path (without extension).
        formats: List of extensions. Used only if save_path is set.

    Returns:
        The created matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ax1, ax2, ax3 = axes
    steps = metrics["steps"]
    sparsity = metrics["sparsity"]
    sparsity_prop = metrics.get("sparsity_proportion", [])

    if not steps or not sparsity:
        ax1.set_title("Sparsity mean per step (no data)")
        ax2.set_title("Sparsity heatmap (no data)")
        ax3.set_title("Sparsity proportion (no data)")
        fig.suptitle(f"Request {metrics['request_id']}")
        if save_path and formats:
            for ext in formats:
                fig.savefig(save_path.with_suffix(f".{ext}"), bbox_inches="tight")
            plt.close(fig)
        return fig

    means: list[float] = []
    for layer_head_grid in sparsity:
        total = 0.0
        count = 0
        for row in layer_head_grid:
            for v in row:
                total += v
                count += 1
        means.append(total / count if count else 0.0)

    ax1.plot(steps, means, marker="o", markersize=3)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Mean sparsity (count above threshold)")
    ax1.set_title("Sparsity mean over layer/head per step")
    ax1.grid(True, alpha=0.3)

    # Heatmap for last step (or first) sparsity per layer/head
    last_sparsity = sparsity[-1]
    if last_sparsity and any(last_sparsity):
        mat = np.array(last_sparsity, dtype=np.float64)
        im = ax2.imshow(mat, aspect="auto", cmap="viridis")
        plt.colorbar(im, ax=ax2, label="Sparsity count")
        ax2.set_xlabel("Head")
        ax2.set_ylabel("Layer")
        ax2.set_title(f"Sparsity at step {steps[-1]}")
    else:
        ax2.set_title("Sparsity heatmap (no data)")

    # Sparsity proportion (unimportant/total) per step
    if sparsity_prop and len(sparsity_prop) == len(steps):
        ax3.plot(steps, sparsity_prop, marker="o", markersize=3, color="green")
        ax3.set_ylim(0, 1)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Sparsity proportion")
    ax3.set_title("Sparsity proportion (unimportant/total) per step")
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f"Request {metrics['request_id']}")
    fig.tight_layout()

    if save_path and formats:
        for ext in formats:
            fig.savefig(save_path.with_suffix(f".{ext}"), bbox_inches="tight")
        plt.close(fig)
    return fig
