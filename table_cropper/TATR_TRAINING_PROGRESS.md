# Table Transformer (TATR) Fine-Tuning Progress

> Last updated: 2026-03-18

## Overview

Fine-tuning the Microsoft **Table Transformer** (DETR-based) on the SCI-3000 dataset to detect table+caption regions as a single merged bounding box. This is the model used for **Formulation B** in the experiment comparison.

The training uses an **agentic loop** (`fine_tune.py`): train → evaluate → decide (keep/discard/adjust) → repeat until targets are met.

### Target Metrics

| Metric | Target |
|---|---|
| Recall | ≥ 0.95 |
| Precision | ≥ 0.97 |
| Mean IoU | ≥ 0.93 |
| FP rate on negatives | ≤ 3% |

---

## Phase 1-3: Initial Fine-Tuning — ✅ Complete

**Dataset**: `finetune_data` (original GT annotations)

### Training History

| Round | Recall | Precision | Mean IoU | F1 | Status |
|---|---|---|---|---|---|
| Baseline | 0.9891 | 0.9616 | 0.9622 | 0.9752 | ✅ Keep |
| R01 | 0.9946 | 0.9625 | 0.9578 | 0.9783 | ✅ Keep |
| R02 | 0.9938 | 0.9654 | 0.9584 | 0.9794 | ✅ Keep |
| R03 | 0.9922 | 0.9639 | 0.9609 | 0.9778 | ✅ Keep |
| R04 | 0.9946 | 0.9639 | 0.9614 | 0.9790 | ✅ Keep |
| R05 | 0.9930 | 0.9697 | 0.9631 | 0.9812 | ✅ Keep |
| R06 | 0.9907 | 0.9674 | 0.9639 | 0.9789 | ❌ Discard |
| R07 | 0.9922 | 0.9682 | 0.9630 | 0.9801 | ❌ Discard |
| **R08** | **0.9930** | **0.9756** | **0.9639** | **0.9842** | **🏆 Final** |

**Result**: All targets met at R08 with LR bump to 1.5e-5.
**Checkpoint**: `~/.cache/table_cropper/checkpoints/round08_final`

---

## Phase 4: Cleaner GT Data — ✅ Complete

**Dataset**: `finetune_data_v2` — tighter expansion (gap_tolerance 25→15, pad_down 150→100)

Retrained from pre-trained TATR with the cleaner ground truth data.

### Training History

| Round | Recall | Precision | Mean IoU | F1 | Status |
|---|---|---|---|---|---|
| Baseline | 0.9902 | 0.9812 | 0.7269 | 0.9857 | ✅ Keep |
| Phase4 R1 | 0.9902 | 0.9798 | 0.8694 | 0.9849 | ✅ Keep |
| Phase4 R2 | 0.9932 | 0.9762 | 0.8715 | 0.9846 | ✅ Keep |
| Phase4 R3 | 0.9909 | 0.9791 | 0.8742 | 0.9850 | ✅ Keep |

Continued with LR bump (1.5e-5):

| Round | Recall | Precision | Mean IoU | F1 | Status |
|---|---|---|---|---|---|
| Phase4r4 R1 | 0.9924 | 0.9798 | 0.8785 | 0.9861 | ✅ Keep |
| Phase4r4 R2 | 0.9909 | 0.9805 | 0.8814 | 0.9857 | ✅ Keep |
| **Phase4r4 R3** | **0.9909** | **0.9820** | **0.8832** | **0.9864** | ✅ Keep |

**Checkpoint**: `~/.cache/table_cropper/checkpoints/phase4r4_best`

> **Note**: Mean IoU improved significantly (0.727 → 0.883) but still below the 0.93 target. The tighter GT annotations made IoU evaluation stricter.

---

## Phase 5: Latest GT Data — 🔶 In Progress (Interrupted)

**Dataset**: `finetune_data_v3` — latest and cleanest GT annotations

### What Changed Before Training

1. **Cleaner GT data** (`finetune_data_v3`) — tighter expansion, better gap tolerance
2. **Smarter FP filters** in `crop_tables.py`:
   - Requires ≥2 horizontal rules
   - Detects vertical lines for flow diagrams
   - Abstract position checks
3. **Training config**: LR=2e-5, Epochs=10 per round

### Training History

| Round | Loss (start→end) | Mean IoU | F1 | Status |
|---|---|---|---|---|
| Baseline | — | 0.826 | 0.9902 | ✅ Keep |
| **R1** | 0.503 → 0.455 | **0.847** (+2.1pts) | 0.9913 | ✅ Saved |
| R2 | 0.450 → ? | ? | ? | ❌ Cancelled at E3/10 |

**R1 was a big success** — IoU jumped from 0.826 to 0.847 in one round.

R2 was making good progress (loss declining to 0.445) when it was interrupted.

**Checkpoint**: `~/.cache/table_cropper/checkpoints/phase5_best` (R1 result)

### To Resume on VM

```bash
cd ~/autoresearch/table_cropper

# Resume from the Phase 5 R1 checkpoint
uv run python fine_tune.py \
    --resume ~/.cache/table_cropper/checkpoints/phase5_best \
    --data-dir ~/.cache/table_cropper/finetune_data_v3 \
    --epochs 10 \
    --lr 2e-5 \
    --batch-size 16 \
    --max-rounds 8
```

At the current rate of ~2pts/round, we need roughly **4-5 more rounds** to hit the 0.93 IoU target.

---

## Data Locations

| Data | Path | Size |
|---|---|---|
| Finetune data v3 (latest) | `~/.cache/table_cropper/finetune_data_v3/` | 4.5 GB |
| Finetune data v2 | `~/.cache/table_cropper/finetune_data_v2/` | 3.8 GB |
| Finetune data v1 | `~/.cache/table_cropper/finetune_data/` | 3.8 GB |
| TATR checkpoints | `~/.cache/table_cropper/checkpoints/` | 1.1 GB |
| SCI-3000 raw | `~/.cache/table_cropper/SCI-3000/` | 7.0 GB |

### Available Checkpoints

| Checkpoint | Phase | Description |
|---|---|---|
| `round01_best` | Phase 1-3 | First improvement |
| `round08_final` | Phase 1-3 | All targets met (original GT) |
| `phase4_best` | Phase 4 | Best on v2 GT (LR=5e-6) |
| `phase4r4_best` | Phase 4 | Best on v2 GT (LR=1.5e-5) |
| `phase5_best` | Phase 5 | Best on v3 GT (IoU=0.847) — **resume from here** |

## Key Scripts

| Script | Purpose |
|---|---|
| `fine_tune.py` | Agentic fine-tuning loop (train → eval → adjust) |
| `prepare_finetune_data.py` | Render PDF pages → images + COCO annotations |
| `evaluate.py` | Standalone evaluation on val set |
| `crop_tables.py` | Inference: detect + crop tables from PDFs |

## Next Steps

1. **Resume Phase 5 training** from `phase5_best` checkpoint on CUDA VM
2. Target: Mean IoU ≥ 0.93 (currently 0.847, need ~8.3pts more)
3. Consider increasing epochs per round or adding learning rate warmup
4. Once IoU target met, run full `experiment_runner.py` comparison
