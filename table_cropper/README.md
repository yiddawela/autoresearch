# table-cropper

Crop semantically complete table units (body + caption + footnotes) from scientific PDFs using fine-tuned detection models.

## Project Structure

```
table_cropper/
├── src/               # Core library modules (metrics, heuristics, adapters)
├── scripts/           # Runnable experiment and inference scripts
├── results/           # Experiment output JSONs
├── docs/              # Research plan, annotation protocol, training log
├── paper/             # LaTeX manuscript and figures
├── data/              # Working data (PDFs, extracted crops)
├── pyproject.toml     # Project dependencies
└── uv.lock            # Dependency lock file
```

## Setup

```bash
# Install dependencies
uv sync

# Download SCI-3000 dataset
uv run python scripts/download_data.py
```

## Key Scripts

| Script | Purpose |
|---|---|
| `scripts/crop_tables.py` | Detect and crop tables from PDFs |
| `scripts/fine_tune.py` | Fine-tune TATR on SCI-3000 |
| `scripts/experiment_runner.py` | Run full formulation comparison (WP2) |
| `scripts/downstream_extraction.py` | LLM-based downstream extraction (WP3) |
| `scripts/yolo_pipeline.py` | YOLO training and evaluation pipeline |

## Models

- **TATR (Formulation B)**: Fine-tuned Table Transformer — checkpoint at `~/.cache/table_cropper/checkpoints/phase5_resumed_final`
- **YOLOv26m**: Exploratory comparison — trained via `scripts/yolo_pipeline.py`

## Citation

If using this code, please cite the accompanying paper: *Beyond Table Bodies: A Semantic Completeness Approach to Table Detection in Scientific Documents*.
