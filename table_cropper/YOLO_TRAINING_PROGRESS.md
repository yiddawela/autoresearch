# YOLO26 Training Progress

> Last updated: 2026-03-18

## Overview

Training YOLO26m (medium) models on the **SCI-3000** dataset for table detection across three formulations. The goal is to compare detection approaches for capturing semantically complete table regions (table body + caption).

### Formulations

| Formulation | Description | Classes |
|---|---|---|
| **A** | Table-only detection (baseline) | 1 class: `table` |
| **B** | Merged table+caption bounding box | 1 class: `table+caption` |
| **C** | Multi-class detection | 2 classes: `table`, `caption` |

### Training Config

- **Model**: YOLO26m (medium)
- **Epochs**: 50
- **Batch size**: 16
- **Image size**: 640px
- **Patience**: 10 (early stopping)
- **Dataset**: SCI-3000 converted to YOLO format via `yolo_pipeline.py`

---

## Formulation A — ✅ Complete

**Status**: Finished all 50 epochs on Mac (MPS).

### Final Results (Epoch 50)

| Metric | Value |
|---|---|
| Precision | 0.985 |
| Recall | 0.983 |
| mAP@50 | 0.993 |
| mAP@50-95 | **0.976** |

### Notes
- Converged well, high performance as expected for single-class table detection.
- Training time: ~46,584 seconds (~12.9 hours) on MPS.
- Weights saved at `~/.cache/table_cropper/yolo_data/runs/formulation_A/weights/best.pt`

---

## Formulation B — ✅ Complete

**Status**: Finished all 50 epochs on Mac (MPS). Initially interrupted mid-training, then resumed using `resume_training.sh`.

### Final Results (Epoch 50)

| Metric | Value |
|---|---|
| Precision | 0.980 |
| Recall | 0.983 |
| mAP@50 | 0.991 |
| mAP@50-95 | **0.980** |

### Notes
- Slightly higher mAP@50-95 than Formulation A, suggesting merged boxes are learnable.
- Training was interrupted partway through and resumed from last checkpoint.
- Resume script: `resume_training.sh`
- Weights saved at `~/.cache/table_cropper/yolo_data/runs/formulation_B/weights/best.pt`

---

## Formulation C — ❌ Interrupted (Epoch 17/50)

**Status**: Crashed at epoch 17 with `torch.AcceleratorError: index out of bounds` on MPS.

### Last Known Results (Epoch 16)

| Metric | Value |
|---|---|
| Precision | 0.944 |
| Recall | 0.937 |
| mAP@50 | 0.974 |
| mAP@50-95 | **0.865** |

### Error
```
torch.AcceleratorError: index 3728 is out of bounds: 0, range 0 to 16
```
This was an MPS-specific bug in the YOLO task-aligned assigner. **Should not occur on CUDA.**

### To Resume on VM
```bash
cd ~/autoresearch/table_cropper

# Fix device from MPS to CUDA
sed -i 's/device="mps"/device="0"/' yolo_pipeline.py

# Train from scratch (cleaner than resuming with MPS crash state)
uv run python yolo_pipeline.py train --formulation C --epochs 50 --model-size m
```

---

## Data Locations

| Data | Path |
|---|---|
| YOLO formatted datasets | `~/.cache/table_cropper/yolo_data/{A,B,C}/` |
| Training runs & weights | `~/.cache/table_cropper/yolo_data/runs/` |
| Raw SCI-3000 dataset | `~/.cache/table_cropper/SCI-3000/` |
| Dataset YAML configs | `~/.cache/table_cropper/yolo_data/{A,B,C}/dataset.yaml` |

## Key Scripts

| Script | Purpose |
|---|---|
| `yolo_pipeline.py` | Prepare data + train YOLO models |
| `train_all_yolo.sh` | Train all formulations sequentially |
| `resume_training.sh` | Resume B + train C (used when B was interrupted) |

## Next Steps

1. **Re-train Formulation C on CUDA VM** — the MPS crash won't happen on CUDA
2. Run `experiment_runner.py` to compare all three formulations with COCO AP + semantic completeness metrics
3. Cross-dataset evaluation on PubTables-1M subset
