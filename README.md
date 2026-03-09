# Attention Scores Pipeline

Pipeline for collecting attention scores (current row of the attention matrix) during autoregressive generation, together with importance statistics (tokens needed to reach 0.95 cumulative weight), deltas (newly important / no longer important), sparsity per layer/head, and thinking-marker detection. Output is written per request for later analysis.

## Requirements

- Python 3.10+
- Virtual environment (venv) and dependencies from `requirements.txt`

## Installation

1. Clone or enter the project directory:
   ```bash
   cd attention_scores
   ```

2. Create and activate a virtual environment:
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the pipeline and tests using the Python from the activated venv (e.g. `.venv\Scripts\python.exe` on Windows).

## Project structure

```
attention_scores/
├── configs/
│   └── default.yaml          # Default config (paths, N, K, device, etc.)
├── src/
│   └── attention_scores/     # Main package
│       ├── __init__.py
│       ├── config.py        # Config load/validation (pydantic)
│       ├── dataset_schema.py # JSON dataset schema
│       ├── device.py        # CPU / CUDA / NPU (torch_npu) selection
│       ├── importance.py    # Importance (0.95), deltas, sparsity, save condition
│       ├── thinking.py     # Thinking marker detection
│       ├── attention_utils.py # Extract current row / prefill from model output
│       ├── io.py            # Save metadata, attention rows, prefill, answers
│       ├── read_outputs.py  # Load and parse saved outputs
│       └── run.py           # Main pipeline and CLI entry
├── tests/                   # Pytest tests (artificial data, no model)
├── design.md                # Project requirements
├── requirements.txt
├── pytest.ini                # pytest pythonpath = src
└── README.md
```

Output (configurable via `output_dir`) contains one subfolder per request and shared artifacts (generated answers, dataset copy).

## Running the pipeline

Set `dataset_path` and `model_path` in the config, then run:

```bash
# From project root, with venv activated
python -m attention_scores.run configs/default.yaml
```

Or with explicit interpreter:

```bash
.venv\Scripts\python.exe -m attention_scores.run configs/default.yaml
```

Ensure `PYTHONPATH` includes `src` if the package is not installed (e.g. `set PYTHONPATH=src` on Windows, or install in editable mode).

## Configuration

| Parameter | Description |
|-----------|-------------|
| `dataset_path` | Path to JSON dataset (array of objects with `question`, `answer`, `split`) |
| `model_path` | Hugging Face model name or local path |
| `batch_size` | Batch size (default 1) |
| `max_output_len` | Max new tokens per request |
| `save_every_n_steps` | Save attention row every N decode steps |
| `save_when_new_important_above_k` | Also save when newly important token count > K |
| `importance_threshold` | Cumulative weight threshold for “important” tokens (default 0.95) |
| `thinking_markers` | List of strings to detect (e.g. `\think`, "Wait,") |
| `output_dir` | Root directory for request folders and artifacts |
| `save_prefill_attention` | If true, save prefill attention in `prefill/` per request |
| `sparsity_threshold` | Threshold for sparsity metric (count of weights above) |
| `device` | `auto`, `cpu`, `cuda`, `cuda:0`, `npu`, `npu:0` (NPU via torch_npu) |

## Output format

- **Per request** (`output_dir / <request_id>/`):
  - `metadata.json` — config snapshot, per-step stats (num_important_tokens, deltas, sparsity), thinking events.
  - `attention_rows/step_<k>.npz` — current attention row for saved decode steps (keys `layer_<L>_head_<H>`).
  - `prefill/` (if enabled) — `layer_<L>.npz` with keys `head_<H>`, full attention matrix per layer/head.
  - `format_spec.json` — optional spec for validation when reading.

- **Shared**:
  - `generated_answers.json` — request_id, prompt, generated_text, steps.
  - `dataset_used.json` — copy of the dataset used.

## Reading results

Use the `read_outputs` module to load metadata and attention data:

```python
from pathlib import Path
from attention_scores.read_outputs import (
    load_decode_attention_step,
    load_decode_attention_layer_head,
    load_metadata,
    load_prefill,
    load_request_outputs,
)

output_dir = Path("./output")
request_id = "request_0"

# Full request summary
out = load_request_outputs(output_dir, request_id)
meta = out["metadata"]
saved_steps = out["saved_steps"]
prefill = out["prefill"]

# One decode step as (num_layers, num_heads, seq_len)
row = load_decode_attention_step(output_dir, request_id, step=0)

# Per layer/head dict
layer_head = load_decode_attention_layer_head(output_dir, request_id, step=0)
```

## Tests

From the project root, with venv activated and `src` on `PYTHONPATH`:

```bash
# Using pytest.ini (pythonpath = src)
pytest tests -v

# Or set PYTHONPATH explicitly (e.g. Windows PowerShell)
$env:PYTHONPATH="src"; pytest tests -v
```

Tests use artificial data only (no model or GPU). They cover importance, deltas, save condition, sparsity, io save/load, and read_outputs parsing.
