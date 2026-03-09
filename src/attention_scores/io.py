"""
Save metadata, attention rows (decode), prefill, generated answers, and dataset copy.

Uses a single key scheme: layer_<L>_head_<H>. NPZ for arrays; JSON for metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .thinking import ThinkingEvent

FORMAT_VERSION = "1.0"


def _request_dir(output_dir: str | Path, request_id: str) -> Path:
    """Return path to request subdirectory (request_id is e.g. request_0 or custom id)."""
    return Path(output_dir) / request_id


def _attention_row_to_npz_dict(
    attention_row: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Build dict of layer_L_head_H -> array (seq_len,) for NPZ.

    attention_row: (num_layers, num_heads, seq_len).
    """
    out: dict[str, np.ndarray] = {}
    n_layers, n_heads, _ = attention_row.shape
    for L in range(n_layers):
        for H in range(n_heads):
            key = f"layer_{L}_head_{H}"
            out[key] = np.asarray(attention_row[L, H], dtype=np.float32)
    return out


def write_metadata(
    output_dir: str | Path,
    request_id: str,
    *,
    importance_threshold: float,
    save_every_n_steps: int,
    save_when_new_important_above_k: int,
    save_prefill_attention: bool,
    thinking_events: list[ThinkingEvent],
    per_step: list[dict[str, Any]],
    num_layers: int,
    num_heads: int,
) -> Path:
    """
    Write metadata.json for one request.

    per_step: list of dicts, one per decode step (call after each step to keep file updated).
    Each entry may contain: step, num_important_tokens, newly_important_count, no_longer_important_count,
    newly_important_per_layer, no_longer_important_per_layer (lists of length num_layers),
    newly_important_per_layer_head, no_longer_important_per_layer_head (list[layer][head] counts),
    sparsity (list of list layer -> head -> value), sparsity_per_layer (list of length num_layers),
    sparsity_proportion_per_layer_head (list[layer][head] float), sparsity_proportion_per_layer
    (list of length num_layers, float), seq_len, sparsity_proportion.
    """
    dir_path = _request_dir(output_dir, request_id)
    dir_path.mkdir(parents=True, exist_ok=True)
    meta = {
        "format_version": FORMAT_VERSION,
        "importance_threshold": importance_threshold,
        "save_every_n_steps": save_every_n_steps,
        "save_when_new_important_above_k": save_when_new_important_above_k,
        "save_prefill_attention": save_prefill_attention,
        "thinking_events": thinking_events,
        "per_step": per_step,
        "num_layers": num_layers,
        "num_heads": num_heads,
    }
    path = dir_path / "metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return path


def write_format_spec(
    output_dir: str | Path,
    request_id: str,
    *,
    num_layers: int,
    num_heads: int,
    decode_steps: list[int],
    has_prefill: bool,
    prefill_seq_len: int | None = None,
) -> Path:
    """Write format_spec.json for validation when reading."""
    dir_path = _request_dir(output_dir, request_id)
    dir_path.mkdir(parents=True, exist_ok=True)
    spec = {
        "format_version": FORMAT_VERSION,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "decode_saved_steps": decode_steps,
        "has_prefill": has_prefill,
        "prefill_seq_len": prefill_seq_len,
    }
    path = dir_path / "format_spec.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    return path


def write_attention_row_step(
    output_dir: str | Path,
    request_id: str,
    step: int,
    attention_row: np.ndarray,
) -> Path:
    """
    Write one decode step's current row to attention_rows/step_<k>.npz.

    attention_row: (num_layers, num_heads, seq_len). Keys: layer_<L>_head_<H>.
    """
    dir_path = _request_dir(output_dir, request_id)
    rows_dir = dir_path / "attention_rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    d = _attention_row_to_npz_dict(attention_row)
    path = rows_dir / f"step_{step}.npz"
    np.savez_compressed(path, **d)
    return path


def write_prefill(
    output_dir: str | Path,
    request_id: str,
    prefill_attentions: list[np.ndarray],
) -> Path:
    """
    Write prefill attention matrices to prefill/ directory.

    prefill_attentions: list of length num_layers; each element (num_heads, seq_len, seq_len).
    Saves one file per layer: layer_<L>.npz with keys head_0, head_1, ...
    """
    dir_path = _request_dir(output_dir, request_id)
    prefill_dir = dir_path / "prefill"
    prefill_dir.mkdir(parents=True, exist_ok=True)
    for L, arr in enumerate(prefill_attentions):
        # arr: (num_heads, seq_len, seq_len)
        d = {f"head_{H}": np.asarray(arr[H], dtype=np.float32) for H in range(arr.shape[0])}
        path = prefill_dir / f"layer_{L}.npz"
        np.savez_compressed(path, **d)
    return prefill_dir


def write_generated_answers(
    output_dir: str | Path,
    answers: list[dict[str, Any]],
) -> Path:
    """Write generated answers to output_dir/generated_answers.json (or .jsonl)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "generated_answers.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    return path


def write_dataset_used(
    output_dir: str | Path,
    dataset_items: list[dict[str, Any]],
) -> Path:
    """Write copy of dataset to output_dir/dataset_used.json."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "dataset_used.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset_items, f, ensure_ascii=False, indent=2)
    return path


def write_progress(
    output_dir: str | Path,
    *,
    current_request_index: int,
    total_requests: int,
    request_id: str,
    current_step: int,
    max_output_len: int | None = None,
) -> Path:
    """
    Write progress.json atomically (temp file + rename) for external progress tracking.

    Args:
        output_dir: Root output directory.
        current_request_index: 0-based index of current request.
        total_requests: Total number of requests.
        request_id: Current request id.
        current_step: Current decode step.
        max_output_len: Max decode steps (optional).

    Returns:
        Path to progress.json.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "current_request_index": current_request_index,
        "total_requests": total_requests,
        "request_id": request_id,
        "current_step": current_step,
    }
    if max_output_len is not None:
        payload["max_output_len"] = max_output_len
    path = out / "progress.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)
    return path
