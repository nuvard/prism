"""Configuration loading and validation."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class Config(BaseModel):
    """Pipeline configuration. Load from YAML/JSON; override via env or CLI."""

    dataset_path: str = Field(..., description="Path to JSON dataset")
    model_path: str = Field(..., description="Path to model (HF or local)")
    batch_size: int = Field(1, ge=1, description="Batch size")
    max_output_len: int = Field(128, ge=1, description="Max generation length in tokens")
    save_every_n_steps: int = Field(5, ge=1, description="Save attention row every N steps")
    save_when_new_important_above_k: int = Field(
        3, ge=0, description="Save when count of newly important tokens > K"
    )
    importance_threshold: float = Field(
        0.95, ge=0.0, le=1.0, description="Cumulative weight threshold for important tokens"
    )
    thinking_markers: list[str] = Field(
        default_factory=lambda: ["\\think", "Wait,", "Hmm,"],
        description="Strings to detect as thinking markers",
    )
    output_dir: str = Field("./output", description="Root output directory")
    save_prefill_attention: bool = Field(
        False, description="Save prefill attention separately"
    )
    sparsity_threshold: float = Field(
        1.0e-6, ge=0.0, description="Threshold for sparsity metric (weight above)"
    )
    device: str = Field(
        "auto",
        description="Device: auto, cpu, cuda, cuda:0, npu, npu:0",
    )
    # Visualization (post-processing of pipeline outputs)
    visualization_output_dir: str = Field(
        "",
        description="Directory to save visualization plots; if empty, uses output_dir/visualization",
    )
    visualization_enabled: bool = Field(
        True,
        description="Whether to build visualizations when running the pipeline",
    )
    visualization_formats: list[str] = Field(
        default_factory=lambda: ["png"],
        description="File formats for saved plots (e.g. png, svg)",
    )
    progress_file: bool = Field(
        True,
        description="Write progress.json to output_dir with current request and step",
    )
    progress_log_every_n_steps: int | None = Field(
        None,
        description="Log step progress every N steps; if None, log only when saving attention row",
    )

    @field_validator("dataset_path", "model_path", "output_dir", "visualization_output_dir", mode="before")
    @classmethod
    def coerce_path_str(cls, v: Any) -> str:
        """Coerce Path to str for serialization."""
        if isinstance(v, Path):
            return str(v)
        return str(v) if v is not None else ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"Config must be a mapping, got {type(data)}")
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        """Load config from a JSON file."""
        import json

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"Config must be a mapping, got {type(data)}")
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """Load config from YAML or JSON based on extension."""
        path = Path(path)
        suf = path.suffix.lower()
        if suf in (".yaml", ".yml"):
            return cls.from_yaml(path)
        if suf == ".json":
            return cls.from_json(path)
        raise ValueError(f"Unsupported config extension: {suf}. Use .yaml or .json")
