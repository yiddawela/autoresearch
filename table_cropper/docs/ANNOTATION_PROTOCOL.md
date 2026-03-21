# Annotation Derivation Protocol

This document describes how training targets for each formulation are
derived from public SCI-3000 annotations. Required for reproducibility
and for the paper's methods section.

## Source Dataset: SCI-3000

- **URL**: https://zenodo.org/records/7544824
- **Format**: W3C Web Annotation (JSON), one file per page
- **Classes annotated**: Table, Figure, Caption
- **Linking**: Captions have a `parent` body pointing to the table/figure `id`
- **Coordinate system**: `xywh=pixel:x,y,w,h` relative to canvas (not image)

## Coordinate Conversion

SCI-3000 annotations use canvas coordinates. When rendering PDFs at a
specific DPI, we must scale:

```
scale_x = rendered_image_width / canvas_width
scale_y = rendered_image_height / canvas_height
```

All experiments use DPI=200 for consistency.

## Formulation A: Table-Only

**Rule**: Each `Table` annotation → one bounding box.
- COCO format: `[x_center, y_center, width, height]` (normalised)
- YOLO format: `class_id cx cy w h` (normalised, class_id=0)
- Captions are ignored

**Edge cases**: None. Direct 1:1 mapping from annotations.

## Formulation B: Merged Semantically Complete Crop

**Rule**: For each `Table`, find all `Caption` annotations whose
`parent_id` matches the table's `id`. Merge into a single encompassing
bounding box.

```
merged_x0 = min(table_x0, caption_1_x0, caption_2_x0, ...)
merged_y0 = min(table_y0, caption_1_y0, caption_2_y0, ...)
merged_x1 = max(table_x1, caption_1_x1, caption_2_x1, ...)
merged_y1 = max(table_y1, caption_1_y1, caption_2_y1, ...)
```

**Edge cases handled**:

| Case | Rule | Count in SCI-3000 |
|------|------|-------------------|
| Table with no linked caption | Use table-only box (no merge) | ~15% of tables |
| Multiple captions for one table | Merge all into one box | ~3% of tables |
| Caption linked to non-existent table | Skip (orphan) | Rare |
| Caption with no parent_id | Skip (unlinked) | ~5% of captions |

## Formulation C: Multi-Class Detection

**Rule**: Tables and captions are separate detection targets.
- Class 0: Table body
- Class 1: Caption (only if linked to a table via `parent_id`)

Unlinked captions (those without a `parent_id` pointing to a table)
are excluded from training to avoid noise.

**Linking at inference time**: After detection, tables and captions
are associated using spatial rules:
1. Horizontal overlap ≥ 30% of the narrower region's width
2. Vertical gap ≤ 1.5× table height
3. Greedy assignment: each caption → highest-scoring table

## Filtering Applied

1. **Pages without tables**: Excluded (no training signal)
2. **Duplicate annotations**: De-duplicated by IoU > 0.9
3. **Tiny annotations**: Boxes with area < 100 pixels (in canvas coords)
   are excluded as likely annotation errors

## Validation

- **Manual audit**: 50 randomly selected derived labels (per formulation)
  will be visually inspected against rendered page images
- **Consistency check**: Merged boxes (B) should always encompass the
  table-only box (A) — verified programmatically
- **Caption coverage**: Verified that ≥95% of `parent_id`-linked captions
  are successfully resolved

## PubTables-1M (Transfer Dataset)

- **Source**: bsmock/pubtables-1m (HuggingFace)
- **Format**: PASCAL VOC XML → converted to YOLO format
- **Classes**: `table`, `table rotated` (mapped to single class 0)
- **Caption annotations**: None — cross-dataset evaluation is
  limited to table-body localisation metrics
- **Subset used**: 500 validation images (streamed, not full download)
