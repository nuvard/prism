"""
Load and aggregate pipeline output data for visualization.

Discovers request directories (with metadata.json), loads per-step metrics,
and optionally loads attention rows for score distribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np

from attention_scores.read_outputs import load_metadata


class RequestMetrics(TypedDict):
    """Aggregated per-step metrics for one request."""

    request_id: str
    steps: list[int]
    num_important_tokens: list[int]
    newly_important_count: list[int]
    no_longer_important_count: list[int]
    sparsity: list[list[list[int]]]
    sparsity_proportion: list[float]


def discover_request_ids(output_dir: str | Path) -> list[str]:
    """
    Find all request subdirectories that contain metadata.json.

    Args:
        output_dir: Root pipeline output directory.

    Returns:
        Sorted list of request_id directory names (e.g. request_0, request_1).
    """
    out = Path(output_dir)
    if not out.is_dir():
        return []
    ids: list[str] = []
    for child in out.iterdir():
        if child.is_dir() and (child / "metadata.json").exists():
            ids.append(child.name)
    return sorted(ids)


def aggregate_request_metrics(
    output_dir: str | Path,
    request_id: str,
) -> RequestMetrics:
    """
    Load metadata for one request and extract per-step lists for plotting.

    Args:
        output_dir: Root pipeline output directory.
        request_id: Request subdirectory name.

    Returns:
        RequestMetrics with steps, num_important_tokens, deltas, sparsity, sparsity_proportion.
    """
    meta = load_metadata(output_dir, request_id)
    per_step = meta.get("per_step") or []
    steps: list[int] = []
    num_important_tokens: list[int] = []
    newly_important_count: list[int] = []
    no_longer_important_count: list[int] = []
    sparsity: list[list[list[int]]] = []
    sparsity_proportion_list: list[float] = []

    for entry in per_step:
        step = entry.get("step")
        if step is None:
            continue
        steps.append(int(step))
        num_important_tokens.append(int(entry.get("num_important_tokens", 0)))
        newly_important_count.append(int(entry.get("newly_important_count", 0)))
        no_longer_important_count.append(int(entry.get("no_longer_important_count", 0)))
        raw_sparsity = entry.get("sparsity")
        if isinstance(raw_sparsity, list):
            # Normalize to list[list[int]] (layer -> head -> value)
            layer_list: list[list[int]] = []
            for row in raw_sparsity:
                if isinstance(row, list):
                    layer_list.append([int(x) for x in row])
                else:
                    layer_list.append([])
            sparsity.append(layer_list)
        else:
            sparsity.append([])

        # sparsity_proportion: use stored value or backfill from seq_len and num_important_tokens
        prop = entry.get("sparsity_proportion")
        if prop is not None:
            sparsity_proportion_list.append(float(prop))
        else:
            seq_len = entry.get("seq_len")
            n_imp = int(entry.get("num_important_tokens", 0))
            if seq_len is not None and seq_len > 0:
                sparsity_proportion_list.append((int(seq_len) - n_imp) / int(seq_len))
            else:
                sparsity_proportion_list.append(0.0)

    return RequestMetrics(
        request_id=request_id,
        steps=steps,
        num_important_tokens=num_important_tokens,
        newly_important_count=newly_important_count,
        no_longer_important_count=no_longer_important_count,
        sparsity=sparsity,
        sparsity_proportion=sparsity_proportion_list,
    )


def load_attention_weights_for_distribution(
    output_dir: str | Path,
    request_id: str,
    saved_steps: list[int],
    *,
    max_steps: int = 20,
    max_weights_per_step: int = 50_000,
) -> list[float]:
    """
    Load attention row weights from saved steps for score distribution histogram.

    Samples steps and flattens weights to a 1D list. Caps steps and total size
    to avoid loading huge data.

    Args:
        output_dir: Root pipeline output directory.
        request_id: Request subdirectory name.
        saved_steps: List of step indices that have saved npz files.
        max_steps: Maximum number of steps to load.
        max_weights_per_step: Maximum weights to take per step (sample if larger).

    Returns:
        Flat list of attention weights (may be sampled).
    """
    from attention_scores.read_outputs import load_decode_attention_step

    steps_to_load = saved_steps[:max_steps] if len(saved_steps) > max_steps else saved_steps
    all_weights: list[float] = []
    for step in steps_to_load:
        try:
            arr = load_decode_attention_step(output_dir, request_id, step)
        except (FileNotFoundError, OSError):
            continue
        if arr.size == 0:
            continue
        flat = arr.ravel()
        if flat.size > max_weights_per_step:
            rng = np.random.default_rng(42)
            idx = rng.choice(flat.size, size=max_weights_per_step, replace=False)
            flat = flat[idx]
        all_weights.extend(flat.tolist())
    return all_weights
