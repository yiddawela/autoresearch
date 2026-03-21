# Portability Guide

How to pick up this project on a new machine.

## What's in the Repo (Committed)

| Directory | Contents | Size |
|-----------|----------|------|
| `src/` | Core library: completeness metrics, linked region builder, PubTables adapter | ~27 KB |
| `scripts/` | All experiment scripts (training, evaluation, sweeps, downstream extraction) | ~180 KB |
| `paper/` | LaTeX source, Springer class files, figure PDFs | ~1.5 MB |
| `results/` | Small JSON result files (< 3 MB total; large caches gitignored) | ~3 MB |
| `docs/` | Research plan, annotation protocol, training progress notes | small |
| `data/` | Gitignored. Contains 72 included article PDFs for reference only | small |

## What's NOT in the Repo (Must Be Restored)

Everything below lives in `~/.cache/table_cropper/` and totals ~25 GB.

### 1. SCI-3000 Dataset (~7.1 GB)

Download from Zenodo: https://doi.org/10.5281/zenodo.8357124

Or use the built-in download script:
```bash
uv run python scripts/download_data.py
```

This populates `~/.cache/table_cropper/SCI-3000/`.

### 2. TATR Checkpoint (~115 MB)

The fine-tuned TATR (Formulation B) checkpoint is stored on HuggingFace:
https://huggingface.co/yohani/tatr-formulation-b (private)

To restore:
```python
from transformers import AutoModelForObjectDetection, AutoImageProcessor
model = AutoModelForObjectDetection.from_pretrained("yohani/tatr-formulation-b")
processor = AutoImageProcessor.from_pretrained("yohani/tatr-formulation-b")
```

Or download manually and place at:
```
~/.cache/table_cropper/checkpoints/phase5_resumed_final/
```

To retrain from scratch:
```bash
uv run python scripts/prepare_finetune_data.py
uv run python scripts/fine_tune.py
```

### 3. YOLO Checkpoints (~130 MB total)

YOLO weights for Formulations A, B, C are stored on HuggingFace:
https://huggingface.co/yohani/yolo-table-detection (private)

To restore:
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("yohani/yolo-table-detection", "formulation_B.pt")
```

Or download manually and place at:
```
~/.cache/table_cropper/yolo_data/runs/formulation_A/weights/best.pt
~/.cache/table_cropper/yolo_data/runs/formulation_B/weights/best.pt
~/.cache/table_cropper/yolo_data/runs/formulation_C/weights/best.pt
```

To retrain: use `scripts/yolo_pipeline.py`.

### 4. Fine-Tune Data (~4.5 GB)

Prepared training images and COCO-format annotations. To regenerate:
```bash
uv run python scripts/prepare_finetune_data.py
```

### 5. PubTables-1M Subset (~7.9 GB)

Used for cross-dataset transfer evaluation only. The `cross_dataset_eval.py` script handles downloading automatically.

## Quick Setup on a New Machine

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd table_cropper

# 2. Install dependencies
uv sync

# 3. Set up .env (copy from backup or create with your API keys)
cp .env.example .env  # then edit with your OPENROUTER and HUGGING_FACE tokens

# 4. Log in to HuggingFace
python3 -c "from huggingface_hub import login; login(token='YOUR_TOKEN')"

# 5. Download SCI-3000
uv run python scripts/download_data.py

# 6. Restore checkpoints from HuggingFace (or retrain)
# TATR: auto-downloaded when scripts reference the HF model
# YOLO: download manually or use hf_hub_download

# 7. Regenerate fine-tune data (if needed for retraining)
uv run python scripts/prepare_finetune_data.py

# 8. Run experiments
uv run python scripts/experiment_runner.py --formulation B

# 9. Compile paper
cd paper && pdflatex main.tex && pdflatex main.tex
```

## Uploading / Updating Checkpoints

To push updated checkpoints to HuggingFace:
```bash
python scripts/upload_to_hf.py --hf-user yohani
```

To make repos public (e.g. after paper acceptance):
```bash
python scripts/upload_to_hf.py --hf-user yohani --public
```
