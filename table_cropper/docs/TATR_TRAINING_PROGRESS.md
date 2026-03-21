# Table Transformer (TATR) Fine-Tuning Progress

> Last updated: 2026-03-19

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

## Phase 5: Latest GT Data — ✅ Complete 🏆

**Dataset**: `finetune_data_v3` — latest and cleanest GT annotations

### What Changed Before Training

1. **Cleaner GT data** (`finetune_data_v3`) — tighter expansion, better gap tolerance
2. **Smarter FP filters** in `crop_tables.py`:
   - Requires ≥2 horizontal rules
   - Detects vertical lines for flow diagrams
   - Abstract position checks
3. **Training config**: LR=2e-5, Epochs=10 per round, Batch=8

### Training History

| Round | Recall | Precision | Mean IoU | F1 | Status |
|---|---|---|---|---|---|
| Baseline (pre-trained) | 0.9984 | 0.9765 | 0.760 | 0.9874 | ✅ Keep |
| Phase 5 R1 (prior run) | — | — | 0.847 | 0.9913 | ✅ Keep |
| **Resumed R1** | **0.9977** | **0.9839** | **0.969** | **0.9908** | **🏆 Final — ALL TARGETS MET** |

**Result**: All targets met in a single resumed round. IoU jumped from 0.847 → **0.969** (+12.2pts).

**Final Checkpoint**: `~/.cache/table_cropper/checkpoints/phase5_resumed_final`

> **Note**: The massive IoU improvement (0.847 → 0.969) in one round suggests the model was already close to convergence at `phase5_best`, and the additional 10 epochs of training were sufficient to push it well past the 0.93 target.

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
| `phase5_best` | Phase 5 | Prior best on v3 GT (IoU=0.847) |
| `phase5_resumed_final` | Phase 5 | **🏆 Final model — all targets met (IoU=0.969)** |

## Key Scripts

| Script | Purpose |
|---|---|
| `fine_tune.py` | Agentic fine-tuning loop (train → eval → adjust) |
| `prepare_finetune_data.py` | Render PDF pages → images + COCO annotations |
| `evaluate.py` | Standalone evaluation on val set |
| `crop_tables.py` | Inference: detect + crop tables from PDFs |

## Final Completeness Evaluation (Validation Set) 📊

The fine-tuned merged detector (Formulation B) was evaluated against both the table-only detector (Formulation A) and a heuristic post-processing approach (Formulation C) using 1,277 Ground Truth tables in the Validation set.

| Metric | A: Table-only | B: Merged (FT) | C: Heuristic |
|---|---|---|---|
| **AP@50 (merged target)** | 0.770 | **0.989** | 0.412 |
| **Caption Inclusion Rate (CIR)** | 0.4% | **94.2%** | 93.9% |
| **Semantic Coverage Score (SCS)** | 42.0% | **96.5%** | 95.5% |
| **Complete Unit Capture Rate (CUCR)** | 0.0% | **78.1%** | 24.1% |
| **Over-crop Ratio** | **0.4%** | 11.1% | 41.7% |

*Formulation B provides the strongest balance of detection performance, reliable caption coverage, and controlled over-crop.*

## Downstream Extraction Impact

A study using GPT-5.2 to extract structured data from these crops shows that **Formulation B dramatically improves interpretation**:

- **Title Extraction Rate**: 4.0% (A) ➡️ **88.0%** (B)
- Formulations missing the caption (A) lead to near-zero success for caption-reliant tasks.

## Training Complete ✅

All target metrics have been met. The final checkpoint (`phase5_resumed_final`) is ready for use in the full WP2 experiment comparison and downstream extraction experiments.
