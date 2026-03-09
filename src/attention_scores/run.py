"""
Main pipeline: load config/dataset/model, run generation with attention collection.

Prefill (optional), then decode step-by-step; at each step compute importance,
deltas, sparsity; save conditionally; track thinking markers; write outputs via io.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from .attention_utils import (
    extract_current_row_from_attentions,
    extract_prefill_attentions,
)
from .config import Config
from .dataset_schema import DatasetItem, load_dataset
from .device import get_device
from .importance import (
    compute_deltas,
    compute_deltas_per_layer,
    compute_deltas_per_layer_head,
    important_indices_per_layer_head,
    layer_important_union,
    should_save_on_step,
    sparsity_per_layer,
    sparsity_proportion,
    sparsity_proportion_per_layer,
    sparsity_proportion_per_layer_head,
    step_importance_and_sparsity,
)
from .io import (
    write_attention_row_step,
    write_dataset_used,
    write_format_spec,
    write_generated_answers,
    write_metadata,
    write_prefill,
    write_progress,
)
from .thinking import ThinkingEvent, detect_new_markers_at_step


def run_pipeline(config_path: str | Path) -> None:
    """
    Run the full pipeline: load config, dataset, model; generate; save attention and metadata.

    Args:
        config_path: Path to YAML or JSON config file.
    """
    config = Config.from_file(config_path)
    device = get_device(config.device)
    dataset = load_dataset(config.dataset_path)
    tokenizer = _load_tokenizer(config.model_path)
    model = _load_model(config.model_path, device)

    answers: list[dict[str, Any]] = []

    total_requests = len(dataset)
    for idx, item in enumerate(dataset):
        request_id = item.get_request_id(idx)
        prompt = item.question
        logging.info(
            "Processing request %s/%s: %s",
            idx + 1,
            total_requests,
            request_id,
        )
        result = _process_one(
            request_id=request_id,
            prompt=prompt,
            item=item,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device=device,
            output_dir=config.output_dir,
            total_requests=total_requests,
            request_index=idx,
        )
        answers.append(result["answer_record"])

    write_generated_answers(config.output_dir, answers)
    dataset_serializable = [_item_to_dict(d) for d in dataset]
    write_dataset_used(config.output_dir, dataset_serializable)

    if getattr(config, "visualization_enabled", True):
        from visualization.generate import run_visualization
        run_visualization(config)


def _item_to_dict(item: DatasetItem) -> dict[str, Any]:
    """Convert DatasetItem to JSON-serializable dict."""
    return item.model_dump(mode="json")


def _load_tokenizer(model_path: str) -> Any:
    """Load HuggingFace tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path)


def _load_model(model_path: str, device: torch.device) -> Any:
    """Load HuggingFace causal LM and move to device."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model.to(device)


def _process_one(
    request_id: str,
    prompt: str,
    item: DatasetItem,
    config: Config,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    output_dir: str,
    *,
    total_requests: int,
    request_index: int,
) -> dict[str, Any]:
    """
    Process one request: tokenize, optional prefill save, decode loop, save metadata and rows.
    """
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=model.config.max_position_embeddings if hasattr(model.config, "max_position_embeddings") else 2048,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    saved_steps: list[int] = []
    num_layers = 0
    num_heads = 0
    per_step_meta: list[dict[str, Any]] = []
    thinking_events: list[ThinkingEvent] = []
    seen_markers: set[str] = set()
    prev_important_per_layer: list[frozenset[int]] = []
    prev_important_per_layer_head: list[list[frozenset[int]]] = []

    # Prefill
    if config.save_prefill_attention and input_ids.size(1) > 0:
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
        prefill_list = extract_prefill_attentions(out.attentions, batch_index=0)
        write_prefill(output_dir, request_id, prefill_list)

    # Decode loop
    generated = input_ids.clone()
    prev_important: frozenset[int] = frozenset()
    last_saved_step: int | None = None
    max_new = config.max_output_len
    n_steps = 0
    progress_log_every = getattr(config, "progress_log_every_n_steps", None)
    write_progress_file = getattr(config, "progress_file", True)

    for step in range(max_new):
        with torch.no_grad():
            out = model(
                input_ids=generated,
                attention_mask=None,
                output_attentions=True,
            )
        logits = out.logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        n_steps += 1

        if not out.attentions:
            continue

        if num_layers == 0:
            num_layers = len(out.attentions)
            num_heads = out.attentions[0].shape[1]
            prev_important_per_layer = [frozenset() for _ in range(num_layers)]
            prev_important_per_layer_head = [
                [frozenset() for _ in range(num_heads)] for _ in range(num_layers)
            ]

        current_row = extract_current_row_from_attentions(out.attentions, batch_index=0)
        important_set, num_important, sparsity_arr = step_importance_and_sparsity(
            current_row,
            config.importance_threshold,
            config.sparsity_threshold,
        )
        newly, no_longer, count_new, count_no_longer = compute_deltas(
            prev_important, important_set
        )
        prev_important = important_set

        # Per-layer importance (union over heads) and deltas
        important_lh = important_indices_per_layer_head(
            current_row, config.importance_threshold
        )
        curr_important_per_layer = layer_important_union(important_lh)
        (
            _newly_per_layer,
            _no_longer_per_layer,
            count_new_per_layer,
            count_no_longer_per_layer,
        ) = compute_deltas_per_layer(prev_important_per_layer, curr_important_per_layer)
        prev_important_per_layer = curr_important_per_layer

        count_new_per_layer_head, count_no_longer_per_layer_head = (
            compute_deltas_per_layer_head(prev_important_per_layer_head, important_lh)
        )
        prev_important_per_layer_head = [list(layer_heads) for layer_heads in important_lh]

        text_so_far = tokenizer.decode(generated[0], skip_special_tokens=False)
        new_evs, seen_markers = detect_new_markers_at_step(
            text_so_far, step, config.thinking_markers, seen_markers
        )
        thinking_events.extend(new_evs)

        do_save = should_save_on_step(
            step,
            last_saved_step,
            count_new,
            config.save_every_n_steps,
            config.save_when_new_important_above_k,
        )
        if do_save:
            write_attention_row_step(output_dir, request_id, step, current_row)
            saved_steps.append(step)
            last_saved_step = step

        sparsity_list = sparsity_arr.tolist()
        sparsity_per_layer_list = sparsity_per_layer(sparsity_arr).tolist()
        seq_len = int(current_row.shape[-1])
        sparsity_proportion_per_layer_head_list = sparsity_proportion_per_layer_head(
            sparsity_arr, seq_len
        ).tolist()
        sparsity_proportion_per_layer_list = sparsity_proportion_per_layer(
            sparsity_arr, seq_len
        ).tolist()
        step_entry: dict[str, Any] = {
            "step": step,
            "num_important_tokens": num_important,
            "newly_important_count": count_new,
            "no_longer_important_count": count_no_longer,
            "newly_important_per_layer": count_new_per_layer,
            "no_longer_important_per_layer": count_no_longer_per_layer,
            "newly_important_per_layer_head": count_new_per_layer_head,
            "no_longer_important_per_layer_head": count_no_longer_per_layer_head,
            "sparsity": sparsity_list,
            "sparsity_per_layer": sparsity_per_layer_list,
            "sparsity_proportion_per_layer_head": sparsity_proportion_per_layer_head_list,
            "sparsity_proportion_per_layer": sparsity_proportion_per_layer_list,
            "seq_len": seq_len,
            "sparsity_proportion": sparsity_proportion(num_important, seq_len),
        }
        per_step_meta.append(step_entry)
        write_metadata(
            output_dir,
            request_id,
            importance_threshold=config.importance_threshold,
            save_every_n_steps=config.save_every_n_steps,
            save_when_new_important_above_k=config.save_when_new_important_above_k,
            save_prefill_attention=config.save_prefill_attention,
            thinking_events=thinking_events,
            per_step=per_step_meta,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        if write_progress_file:
            write_progress(
                output_dir,
                current_request_index=request_index,
                total_requests=total_requests,
                request_id=request_id,
                current_step=step,
                max_output_len=max_new,
            )
        if progress_log_every is not None and step % progress_log_every == 0:
            logging.info("Request %s, step %s/%s", request_id, step, max_new)
        elif progress_log_every is None and do_save:
            logging.info("Request %s, step %s (saved)", request_id, step)

        if next_token.item() == tokenizer.eos_token_id:
            break

    if num_layers == 0:
        num_layers = getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", 0))
        num_heads = getattr(model.config, "num_attention_heads", getattr(model.config, "n_head", 0))

    write_metadata(
        output_dir,
        request_id,
        importance_threshold=config.importance_threshold,
        save_every_n_steps=config.save_every_n_steps,
        save_when_new_important_above_k=config.save_when_new_important_above_k,
        save_prefill_attention=config.save_prefill_attention,
        thinking_events=thinking_events,
        per_step=per_step_meta,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    write_format_spec(
        output_dir,
        request_id,
        num_layers=num_layers,
        num_heads=num_heads,
        decode_steps=saved_steps,
        has_prefill=config.save_prefill_attention,
        prefill_seq_len=input_ids.size(1) if config.save_prefill_attention else None,
    )

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt) :].lstrip()

    return {
        "answer_record": {
            "request_id": request_id,
            "prompt": prompt,
            "generated_text": generated_text,
            "steps": n_steps,
        },
        "saved_steps": saved_steps,
    }


def main() -> None:
    """CLI entry: load config path from argv and run pipeline."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m attention_scores.run <config.yaml|config.json>", file=sys.stderr)
        sys.exit(1)
    run_pipeline(sys.argv[1])


if __name__ == "__main__":
    main()
