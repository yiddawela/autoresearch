# Semantically Complete Table-Region Extraction from Scientific PDFs

> Last updated: 2026-03-18
>
> **Venue**: IJDAR (International Journal on Document Analysis and Recognition) — primary; Pattern Recognition Letters — backup
>
> **Title direction**: *"Beyond Table Bodies: Representations and Evaluation for Semantically Complete Table Extraction from Scientific Documents"*

---

## 1. Research Question & Thesis

> *How should tables and their associated captions be represented, detected, and evaluated when the real downstream goal is scientific information extraction rather than table-body localisation alone?*

**Thesis**: Standard table detectors optimise for localising the visual table body. Downstream scientific extraction requires a semantically complete unit including captions. We define this task, compare alternative representations, evaluate them with completeness-aware metrics, and show their effect on downstream extraction.

---

## 2. Four Contribution Layers

1. **Task definition**: Formalise semantically complete table-region extraction with explicit inclusion/exclusion rules
2. **Representation comparison**: Compare three target formulations experimentally rather than assuming one is correct
3. **Evaluation reform**: Introduce semantic completeness metrics alongside standard AP/IoU, and show where they diverge
4. **Downstream validation**: Show that better semantic completeness measurably improves scientific data extraction

---

## 3. Task Definition & Three Formulations

### 3.1 Target Unit: Formal Rule Set

| Rule | Decision | Rationale |
|------|----------|----------|
| **Table body** | Always included | Core visual object |
| **Caption/title** | Included when linked to exactly one table | Captions provide interpretive context |
| **Footnotes** | Included when directly attached below table body | Qualify numeric values |
| **Nearby body text** | Excluded | Not part of table unit |
| **Visually distant captions** | Included if linked by `parent_id`; excluded if distance > 1.5× table height | Prevents unreliable spatial guessing |
| **Multi-part captions** | All fragments merged | Partial captions are informationally incomplete |
| **Shared captions** | Caption included in each table's unit; flagged as edge case | Rare but handled consistently |
| **Continued tables** (spanning pages) | Excluded | Future work |

### 3.2 Three Formulations

| Formulation | Target | Role in paper |
|---|---|---|
| **A. Table-only** | Visual table body only | Baseline — how existing detectors are trained |
| **B. Merged semantically complete crop** | Single box: table + caption(s) + footnotes | Practical candidate for crop-based VLM pipelines |
| **C. Linked-region detection** | Table and caption detected separately, then associated | Structurally faithful candidate |

---

## 4. Model Strategy (Two Families)

The paper's contribution is **representation comparison**, not architecture comparison. Two families test whether conclusions are representation-dependent vs architecture-specific.

| Family | Model | Notes |
|---|---|---|
| **DETR-based** | Microsoft Table Transformer (TATR) | Pre-trained on PubTables-1M |
| **YOLO-based** | YOLO26m (Ultralytics, Jan 2026) | NMS-free, latest single-stage |

---

## 5. Dataset Strategy

| Dataset | Role | Caption annotations? |
|---|---|---|
| **SCI-3000** (Zenodo) | Primary: in-domain training & evaluation | ✅ Table, figure, caption with `parent_id` linking |
| **PubTables-1M** | Transfer: cross-dataset generalisation | ❌ Table-only — evaluate localisation quality |

Cross-dataset transfer tests whether caption-aware training harms, preserves, or improves table-body localisation under domain shift.

---

## 6. Evaluation Metrics

### Standard localisation
- AP@50, AP@75 (COCO-standard, `coco_eval.py`)
- Precision, recall, F1
- IoU distribution (histogram)

### Semantic completeness (novel, `completeness_metrics.py`)
- **CIR** (Caption Inclusion Rate): fraction of GT caption area captured
- **SCS** (Semantic Coverage Score): coverage of body + caption + footnotes
- **CUCR** (Complete Unit Capture Rate): binary — both body and caption above IoU threshold
- **Over-crop ratio**: proportion of irrelevant area beyond the target
- **Linking accuracy** (Formulation C): correct table-caption associations

### Statistical rigour
- Multiple random seeds (≥3) for training
- Bootstrap confidence intervals on key metrics
- Significance tests for formulation comparisons

---

## 7. Downstream Extraction Experiment (Two-Tier)

### Tier 1: Simple semantic task
- **Task**: Table title extraction + study arm identification
- **100 tables**, GPT-4o, exact-match + fuzzy-match F1
- If caption-aware crops don't help here, the thesis is weak

### Tier 2: Full field extraction
- **Task**: Extract all numeric values with row/column headers
- **100–150 tables** with manually verified ground truth
- **One LLM** (GPT-4o) to isolate representation effects
- **Three inputs**: table-only crop, merged crop, linked table+caption input
- **Metric**: Field extraction F1

**Success criterion**: The representation scoring best on SCS also scores best on extraction F1.

---

## 8. Paper Structure (7 Sections)

| Section | Content |
|---|---|
| **Abstract** | Problem, task, formulation comparison, key finding, downstream result |
| **1. Introduction** | Downstream motivation; why table-body localisation is insufficient; research question; four contributions |
| **2. Related Work** | (a) table detection as localisation, (b) richer structural representations, (c) dataset quality, (d) limits of IoU evaluation, (e) scientific/biomedical extraction, (f) gap: semantically complete extraction |
| **3. Task Definition & Formulations** | Formal definition; three formulations; annotation derivation protocol |
| **4. Methods & Datasets** | Two model families; training details; heuristic refinement; dataset descriptions |
| **5. Experiments & Results** | Representation comparison, model comparison, cross-dataset transfer, downstream extraction, robustness |
| **6. Discussion & Limitations** | Metric disagreements; limitations (non-English, continued tables, single-domain completeness eval) |
| **7. Conclusion** | Summary, future work (learned linking, richer annotations) |

---

## 9. Current Progress

### WP0: Pilot Validation — ✅ Complete

**Script**: `pilot_evaluation.py` — 50 diverse pages, 66 GT tables.

| Metric | A: Table-only | B: Merged (FT) | C: Heuristic |
|---|:---:|:---:|:---:|
| Caption Inclusion Rate | 1.3% | 76.1% | 86.5% |
| Semantic Coverage | 41.3% | 84.2% | 90.2% |
| Complete Unit Capture | 1.5% | 68.2% | 27.3% |
| Over-crop Ratio | 0.8% | 8.7% | 34.6% |

**Gate: PASSED** — 61 divergence examples, CIR differences >85%.

### WP1: Infrastructure & Training — ⏳ In Progress

#### Evaluation modules ✅
- `coco_eval.py`, `completeness_metrics.py`, `linked_region.py`, `experiment_runner.py` — all implemented and smoke-tested

#### TATR fine-tuning (see [TATR_TRAINING_PROGRESS.md](./TATR_TRAINING_PROGRESS.md))
- Phase 5 R1: IoU=0.847 (target: 0.93), checkpoint: `phase5_best`
- **Resume on VM** to continue training

#### YOLO26 training
| Formulation | mAP@50-95 | Status |
|---|---|---|
| A (table-only) | 0.976 | ✅ Complete |
| B (merged) | 0.980 | ✅ Complete |
| C (multi-class) | 0.865 (ep.16) | ❌ MPS crash — retrain on CUDA VM |

### WP2: Core Experiments — 📋 Next
- 3 formulations × 2 model families on full SCI-3000 val set (~1,106 pages)
- Cross-dataset transfer on PubTables-1M
- Heuristic sensitivity sweep (5 configs)
- 3 random seeds, bootstrap CIs

### WP3: Downstream Validation — 📋 Planned
- Curate 100–150 test tables with verified GT
- GPT-4o extraction on table-only / merged / linked crops
- Field extraction F1

### WP4: Writing — 📋 Planned
- LaTeX draft, figures, related work, all results with CIs

### WP5: Packaging & Submission — 📋 Planned
- GitHub repo + README, HuggingFace model cards, submit to IJDAR

---

## 10. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Representation differences too small | Medium | Pilot passed (CIR diff >85%); completeness metrics are sensitive |
| Transfer datasets lack caption annotations | High (known) | Frame as localisation transfer; acknowledge limitation |
| Linked-region detection is complex | Medium | Simple rule-based linking baseline |
| Reviewers see work as incremental | Low | Lead with task formulation; four-layer contribution |
| Heuristic parameters appear fragile | Medium | Sensitivity sweep; present as optional domain-adaptation |

---

## 11. Immediate Next Steps (VM)

1. **Re-train YOLO Formulation C** on CUDA (fix `device="mps"` → `device="0"`)
2. **Resume TATR Phase 5** training from `phase5_best` checkpoint
3. Run `experiment_runner.py` for full WP2 comparison once training is complete
4. Download PubTables-1M for cross-dataset evaluation

---

## 12. File Index

| File | Purpose |
|---|---|
| `yolo_pipeline.py` | YOLO data prep + training |
| `fine_tune.py` | TATR agentic fine-tuning loop |
| `experiment_runner.py` | Full WP2 formulation comparison |
| `pilot_evaluation.py` | WP0 50-page pilot validation |
| `coco_eval.py` | COCO-standard AP evaluation |
| `completeness_metrics.py` | Semantic completeness metrics (CIR, SCS, CUCR) |
| `crop_tables.py` | Table detection + cropping inference |
| `linked_region.py` | Heuristic table-caption linking (Form. C) |
| `pubtables_adapter.py` | PubTables-1M format adapter |
| `prepare_finetune_data.py` | PDF → image + annotation rendering |
| `TATR_TRAINING_PROGRESS.md` | TATR fine-tuning detailed progress |
| `ANNOTATION_PROTOCOL.md` | SCI-3000 annotation guidelines |

## 13. Data Locations

| Data | Path | Size |
|---|---|---|
| SCI-3000 raw | `~/.cache/table_cropper/SCI-3000/` | 7 GB |
| Finetune data v3 | `~/.cache/table_cropper/finetune_data_v3/` | 4.5 GB |
| YOLO datasets | `~/.cache/table_cropper/yolo_data/{A,B,C}/` | 400 MB |
| YOLO runs & weights | `~/.cache/table_cropper/yolo_data/runs/` | 344 MB |
| TATR checkpoints | `~/.cache/table_cropper/checkpoints/` | 1.1 GB |
