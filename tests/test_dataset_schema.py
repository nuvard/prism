"""Tests for dataset schema and load_dataset with generated data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from attention_scores.dataset_schema import DatasetItem, load_dataset


def _make_dataset_json(
    tmp_path: Path,
    items: list[dict],
) -> Path:
    """Write a JSON dataset file and return its path."""
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_load_dataset_minimal_fields(tmp_path: Path) -> None:
    """Load dataset with only required fields (question, answer, split)."""
    data = [
        {"question": "What is 2+2?", "answer": "4", "split": "test"},
        {"question": "Say hello.", "answer": "Hello.", "split": "val"},
    ]
    path = _make_dataset_json(tmp_path, data)
    loaded = load_dataset(str(path))
    assert len(loaded) == 2
    assert loaded[0].question == "What is 2+2?"
    assert loaded[0].answer == "4"
    assert loaded[0].split == "test"
    assert loaded[0].id is None
    assert loaded[0].max_token_len is None
    assert loaded[0].token_len is None
    assert loaded[1].question == "Say hello."


def test_load_dataset_with_optional_fields(tmp_path: Path) -> None:
    """Load dataset with id, max_token_len, token_len."""
    data = [
        {
            "question": "Explain gravity.",
            "answer": "Gravity is a force.",
            "split": "train",
            "id": "q1",
            "max_token_len": 512,
            "token_len": 42,
        },
        {
            "question": "Short one.",
            "answer": "Ok.",
            "split": "test",
            "max_token_len": 128,
            "token_len": 10,
        },
    ]
    path = _make_dataset_json(tmp_path, data)
    loaded = load_dataset(str(path))
    assert len(loaded) == 2
    assert loaded[0].id == "q1"
    assert loaded[0].max_token_len == 512
    assert loaded[0].token_len == 42
    assert loaded[1].id is None
    assert loaded[1].max_token_len == 128
    assert loaded[1].token_len == 10


def test_get_request_id_uses_id_when_set() -> None:
    """get_request_id returns item id when set."""
    item = DatasetItem(question="Q", id="my_id")
    assert item.get_request_id(0) == "my_id"
    assert item.get_request_id(99) == "my_id"


def test_get_request_id_fallback_to_index() -> None:
    """get_request_id returns request_<index> when id is None or empty."""
    item = DatasetItem(question="Q", id=None)
    assert item.get_request_id(0) == "request_0"
    assert item.get_request_id(5) == "request_5"
    item_empty = DatasetItem(question="Q", id="")
    assert item_empty.get_request_id(3) == "request_3"


def test_load_dataset_raises_on_missing_file() -> None:
    """load_dataset raises FileNotFoundError when path does not exist."""
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_dataset("/nonexistent/path/dataset.json")


def test_load_dataset_raises_on_not_array(tmp_path: Path) -> None:
    """load_dataset raises ValueError when JSON is not an array."""
    path = tmp_path / "not_array.json"
    path.write_text('{"question": "only one"}', encoding="utf-8")
    with pytest.raises(ValueError, match="must be an array"):
        load_dataset(str(path))


def test_load_dataset_empty_array(tmp_path: Path) -> None:
    """Load empty array returns empty list."""
    path = _make_dataset_json(tmp_path, [])
    loaded = load_dataset(str(path))
    assert loaded == []
