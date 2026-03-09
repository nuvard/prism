"""Dataset schema for JSON input."""

from typing import Any

from pydantic import BaseModel, Field


class DatasetItem(BaseModel):
    """Single item in the dataset JSON array. question, answer, split required; rest optional."""

    question: str = Field(..., description="Input question/prompt")
    answer: str = Field("", description="Reference answer (optional for inference)")
    split: str = Field("", description="Split label e.g. train/val/test")
    id: str | None = Field(None, description="Optional request id")
    max_token_len: int | None = Field(None, description="Maximum token length (e.g. for truncation or generation cap)")
    token_len: int | None = Field(None, description="Token length for this item (e.g. prompt or prompt+answer)")

    model_config = {"extra": "allow"}

    def get_request_id(self, index: int) -> str:
        """Return id field if set, else request_<index>."""
        if self.id is not None and str(self.id).strip():
            return str(self.id).strip()
        return f"request_{index}"


def load_dataset(path: str) -> list[DatasetItem]:
    """
    Load dataset from JSON file. Expects a JSON array of objects with question, answer, split.

    Args:
        path: Path to JSON file.

    Returns:
        List of validated DatasetItem.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If JSON is not an array or items fail validation.
    """
    import json

    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    text = p.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"Dataset JSON must be an array, got {type(data)}")
    return [DatasetItem.model_validate(item) for item in data]
