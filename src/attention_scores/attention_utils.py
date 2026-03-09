"""Extract current attention row from model outputs and prepare for saving."""

from __future__ import annotations

from typing import Any

import numpy as np


def extract_current_row_from_attentions(
    attentions: tuple[Any, ...] | list[Any],
    batch_index: int = 0,
) -> np.ndarray:
    """
    Extract the current (last) token's attention row from transformers attentions.

    attentions: tuple of tensors, one per layer; each shape (batch, num_heads, seq_len, seq_len).
    We take [batch_index, :, -1, :] per layer and stack -> (num_layers, num_heads, seq_len).

    Args:
        attentions: Output of model forward (e.g. outputs.attentions).
        batch_index: Which batch item to take (0 for single batch).

    Returns:
        numpy array (num_layers, num_heads, seq_len), float.
    """
    layers: list[np.ndarray] = []
    for layer_attn in attentions:
        # layer_attn: (batch, num_heads, seq_len, seq_len)
        arr = np.asarray(layer_attn.detach().cpu(), dtype=np.float64)
        row = arr[batch_index, :, -1, :]  # (num_heads, seq_len)
        layers.append(row)
    return np.stack(layers, axis=0)


def extract_prefill_attentions(
    attentions: tuple[Any, ...] | list[Any],
    batch_index: int = 0,
) -> list[np.ndarray]:
    """
    Extract full attention matrix per layer for prefill (seq_len, seq_len) per head.

    attentions: tuple of tensors, one per layer; each (batch, num_heads, seq_len, seq_len).
    Returns list of (num_heads, seq_len, seq_len) numpy arrays.

    Args:
        attentions: Prefill forward pass attentions.
        batch_index: Which batch item.

    Returns:
        List of length num_layers; each element (num_heads, seq_len, seq_len).
    """
    out: list[np.ndarray] = []
    for layer_attn in attentions:
        arr = np.asarray(layer_attn.detach().cpu(), dtype=np.float64)
        # (batch, num_heads, seq_len, seq_len) -> (num_heads, seq_len, seq_len)
        out.append(arr[batch_index])
    return out
