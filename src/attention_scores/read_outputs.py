"""
Load and parse pipeline outputs: metadata, decode attention by step, prefill.

Returns typed structures and numpy arrays for analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import numpy as np


class ThinkingEventDict(TypedDict):
    """Thinking event as stored in metadata."""

    marker: str
    step: int


class PerStepDict(TypedDict, total=False):
    """Per-step entry in metadata."""

    step: int
    num_important_tokens: int
    newly_important_count: int
    no_longer_important_count: int
    sparsity: list[list[int]]


class RequestMetadata(TypedDict):
    """Metadata for one request."""

    format_version: str
    importance_threshold: float
    save_every_n_steps: int
    save_when_new_important_above_k: int
    save_prefill_attention: bool
    thinking_events: list[ThinkingEventDict]
    per_step: list[PerStepDict]
    num_layers: int
    num_heads: int


def request_dir(output_dir: str | Path, request_id: str) -> Path:
    """Return path to request subdirectory."""
    return Path(output_dir) / request_id


def load_metadata(
    output_dir: str | Path,
    request_id: str,
) -> RequestMetadata:
    """
    Load metadata.json for one request.

    Returns:
        Typed dict with format_version, importance_threshold, thinking_events,
        per_step (list of step data with num_important_tokens, sparsity, etc.),
        num_layers, num_heads.
    """
    import json

    path = request_dir(output_dir, request_id) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_decode_attention_step(
    output_dir: str | Path,
    request_id: str,
    step: int,
) -> np.ndarray:
    """
    Load decode attention row for one step as (num_layers, num_heads, seq_len).

    Reads attention_rows/step_<k>.npz and reassembles by layer/head.
    """
    dir_path = request_dir(output_dir, request_id) / "attention_rows"
    path = dir_path / f"step_{step}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Step file not found: {path}")
    data = np.load(path)
    def _key_parts(k: str) -> tuple[int, int]:
        p = k.split("_")
        return (int(p[1]), int(p[3]))

    keys = sorted(data.files, key=_key_parts)
    if not keys:
        return np.array([]).reshape(0, 0, 0)
    first = data[keys[0]]
    seq_len = first.size
    n_layers = max(int(k.split("_")[1]) for k in keys) + 1
    n_heads = max(int(k.split("_")[3]) for k in keys) + 1
    out = np.zeros((n_layers, n_heads, seq_len), dtype=np.float32)
    for k in keys:
        parts = k.split("_")
        L, H = int(parts[1]), int(parts[3])
        out[L, H] = data[k]
    return out


def load_decode_attention_layer_head(
    output_dir: str | Path,
    request_id: str,
    step: int,
) -> dict[int, dict[int, np.ndarray]]:
    """
    Load decode attention row for one step as dict[layer][head] -> array (seq_len,).

    Returns:
        Nested dict: result[layer][head] is 1D numpy array.
    """
    arr = load_decode_attention_step(output_dir, request_id, step)
    result: dict[int, dict[int, np.ndarray]] = {}
    for L in range(arr.shape[0]):
        result[L] = {H: arr[L, H].copy() for H in range(arr.shape[1])}
    return result


def load_prefill(
    output_dir: str | Path,
    request_id: str,
) -> list[np.ndarray]:
    """
    Load prefill attention matrices if present.

    Returns:
        List of length num_layers; each element (num_heads, seq_len, seq_len).
    """
    prefill_dir = request_dir(output_dir, request_id) / "prefill"
    if not prefill_dir.exists():
        return []
    out: list[np.ndarray] = []
    layer_idx = 0
    while True:
        path = prefill_dir / f"layer_{layer_idx}.npz"
        if not path.exists():
            break
        data = np.load(path)
        keys = sorted(data.files, key=lambda k: int(k.split("_")[1]))
        heads = [data[k] for k in keys]
        out.append(np.stack(heads, axis=0))
        layer_idx += 1
    return out


def load_format_spec(
    output_dir: str | Path,
    request_id: str,
) -> dict[str, Any] | None:
    """Load format_spec.json if present."""
    import json

    path = request_dir(output_dir, request_id) / "format_spec.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_request_outputs(
    output_dir: str | Path,
    request_id: str,
) -> dict[str, Any]:
    """
    Load all outputs for one request: metadata, format_spec, list of saved steps.

    Does not load full attention arrays; use load_decode_attention_step per step.
    Prefill is loaded only if present.

    Returns:
        Dict with keys: metadata, format_spec (or None), saved_steps (from metadata per_step),
        prefill (list of arrays or empty list).
    """
    meta = load_metadata(output_dir, request_id)
    saved_steps = [p["step"] for p in meta["per_step"]]
    prefill = load_prefill(output_dir, request_id)
    spec = load_format_spec(output_dir, request_id)
    return {
        "metadata": meta,
        "format_spec": spec,
        "saved_steps": saved_steps,
        "prefill": prefill,
    }
