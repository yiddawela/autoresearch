"""
Semantic completeness metrics for table-region extraction.

Defines metrics that go beyond standard IoU to measure whether a predicted
region captures the full interpretable table unit (body + caption + footnotes).

Metrics:
    CIR   — Caption Inclusion Rate: fraction of GT caption area captured
    SCS   — Semantic Coverage Score: coverage of body + caption + footnotes
    CUCR  — Complete Unit Capture Rate: binary per-sample, both body and caption
             captured above IoU threshold
    OCR   — Over-crop Ratio: fraction of predicted area that falls outside GT
"""

from __future__ import annotations


def _intersection_area(a: tuple, b: tuple) -> float:
    """Intersection area of two (x0, y0, x1, y1) boxes."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def _box_area(box: tuple) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _iou(a: tuple, b: tuple) -> float:
    inter = _intersection_area(a, b)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _coverage(gt_box: tuple, pred_box: tuple) -> float:
    """What fraction of gt_box is covered by pred_box?"""
    inter = _intersection_area(gt_box, pred_box)
    area = _box_area(gt_box)
    return inter / area if area > 0 else 0.0


def caption_inclusion_rate(
    gt_caption_boxes: list[tuple],
    pred_box: tuple,
) -> float:
    """
    CIR: fraction of total GT caption area captured by the predicted region.

    Args:
        gt_caption_boxes: list of (x0, y0, x1, y1) ground-truth caption boxes
        pred_box: (x0, y0, x1, y1) predicted region

    Returns:
        CIR in [0, 1]. Returns 1.0 if there are no captions (vacuously true).
    """
    if not gt_caption_boxes:
        return 1.0

    total_caption_area = sum(_box_area(c) for c in gt_caption_boxes)
    if total_caption_area <= 0:
        return 1.0

    captured_area = sum(
        _intersection_area(c, pred_box) for c in gt_caption_boxes
    )
    return captured_area / total_caption_area


def semantic_coverage_score(
    gt_table_box: tuple,
    gt_caption_boxes: list[tuple],
    pred_box: tuple,
    gt_footnote_boxes: list[tuple] | None = None,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    SCS: weighted coverage of table body, caption(s), and footnote(s).

    Args:
        gt_table_box: (x0, y0, x1, y1) ground-truth table body
        gt_caption_boxes: list of (x0, y0, x1, y1) caption boxes
        pred_box: (x0, y0, x1, y1) predicted region
        gt_footnote_boxes: optional list of footnote boxes
        weights: (table_weight, caption_weight, footnote_weight)

    Returns:
        SCS in [0, 1]. Equal-weight by default.
    """
    components = []
    w_table, w_caption, w_footnote = weights

    # Table body coverage
    table_cov = _coverage(gt_table_box, pred_box)
    components.append((table_cov, w_table))

    # Caption coverage (aggregate)
    if gt_caption_boxes:
        caption_cov = caption_inclusion_rate(gt_caption_boxes, pred_box)
        components.append((caption_cov, w_caption))

    # Footnote coverage (aggregate)
    if gt_footnote_boxes:
        total_fn_area = sum(_box_area(f) for f in gt_footnote_boxes)
        if total_fn_area > 0:
            fn_captured = sum(
                _intersection_area(f, pred_box) for f in gt_footnote_boxes
            )
            fn_cov = fn_captured / total_fn_area
            components.append((fn_cov, w_footnote))

    total_weight = sum(w for _, w in components)
    if total_weight <= 0:
        return 0.0
    return sum(c * w for c, w in components) / total_weight


def complete_unit_capture_rate(
    gt_table_box: tuple,
    gt_caption_boxes: list[tuple],
    pred_box: tuple,
    iou_threshold: float = 0.7,
) -> bool:
    """
    CUCR: binary metric — True if both the table body and ALL associated
    captions are captured above the IoU threshold.

    For table body: uses IoU between GT table and predicted region.
    For captions: uses coverage (fraction of caption captured) ≥ threshold.

    Args:
        gt_table_box: (x0, y0, x1, y1) ground-truth table body
        gt_caption_boxes: list of caption boxes
        pred_box: predicted region
        iou_threshold: minimum IoU/coverage threshold

    Returns:
        True if the complete unit is captured.
    """
    # Table body must have sufficient IoU
    table_iou = _iou(gt_table_box, pred_box)
    if table_iou < iou_threshold:
        return False

    # Every caption must be sufficiently covered
    for cap_box in gt_caption_boxes:
        if _coverage(cap_box, pred_box) < iou_threshold:
            return False

    return True


def over_crop_ratio(
    gt_table_box: tuple,
    gt_caption_boxes: list[tuple],
    pred_box: tuple,
    gt_footnote_boxes: list[tuple] | None = None,
) -> float:
    """
    Over-crop ratio: fraction of predicted area that falls outside all
    ground-truth components (table + captions + footnotes).

    Lower is better. 0.0 = perfect tight crop. 1.0 = entirely outside GT.

    Args:
        gt_table_box: ground-truth table body
        gt_caption_boxes: caption boxes
        pred_box: predicted region
        gt_footnote_boxes: optional footnote boxes

    Returns:
        Over-crop ratio in [0, 1].
    """
    pred_area = _box_area(pred_box)
    if pred_area <= 0:
        return 0.0

    # Compute area of pred that overlaps with any GT component
    # This is approximate — components may overlap each other
    all_gt = [gt_table_box] + list(gt_caption_boxes)
    if gt_footnote_boxes:
        all_gt.extend(gt_footnote_boxes)

    covered_area = sum(_intersection_area(pred_box, gt) for gt in all_gt)

    # Clamp: can't cover more than pred_area (overlapping GT components)
    covered_area = min(covered_area, pred_area)

    return 1.0 - (covered_area / pred_area)


def compute_all_metrics(
    gt_table_box: tuple,
    gt_caption_boxes: list[tuple],
    pred_box: tuple,
    gt_footnote_boxes: list[tuple] | None = None,
    cucr_threshold: float = 0.7,
) -> dict:
    """
    Compute all semantic completeness metrics for one sample.

    Returns dict with keys: cir, scs, scs_equal, cucr, over_crop, table_iou
    """
    return {
        "cir": caption_inclusion_rate(gt_caption_boxes, pred_box),
        "scs": semantic_coverage_score(
            gt_table_box, gt_caption_boxes, pred_box, gt_footnote_boxes
        ),
        "cucr": complete_unit_capture_rate(
            gt_table_box, gt_caption_boxes, pred_box, cucr_threshold
        ),
        "over_crop": over_crop_ratio(
            gt_table_box, gt_caption_boxes, pred_box, gt_footnote_boxes
        ),
        "table_iou": _iou(gt_table_box, pred_box),
        "table_coverage": _coverage(gt_table_box, pred_box),
        "caption_coverage": caption_inclusion_rate(gt_caption_boxes, pred_box),
    }
