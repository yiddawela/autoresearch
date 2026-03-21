"""
Linked-region detection: Formulation C for the paper.

Detects tables and captions SEPARATELY, then links them via rule-based
spatial association. This is the "structurally faithful" formulation
that respects tables and captions as distinct document objects.

Linking rules (in priority order):
  1. Spatial proximity within column bounds
  2. Caption positioned directly above or below table
  3. Maximum distance constraint (1.5× table height)

Usage:
    from linked_region import link_tables_captions, detect_and_link

    # Link GT annotations
    links = link_tables_captions(table_boxes, caption_boxes, page_size)

    # Or detect + link in one step
    results = detect_and_link(image, processor, model, device)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class DetectedRegion:
    """A detected table or caption region."""
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    score: float = 1.0
    label: str = "table"  # "table" or "caption"

    @property
    def x0(self): return self.bbox[0]
    @property
    def y0(self): return self.bbox[1]
    @property
    def x1(self): return self.bbox[2]
    @property
    def y1(self): return self.bbox[3]
    @property
    def width(self): return self.x1 - self.x0
    @property
    def height(self): return self.y1 - self.y0
    @property
    def cx(self): return (self.x0 + self.x1) / 2
    @property
    def cy(self): return (self.y0 + self.y1) / 2
    @property
    def area(self): return self.width * self.height


@dataclass
class LinkedUnit:
    """A semantically complete table unit: table + linked captions."""
    table: DetectedRegion
    captions: list[DetectedRegion] = field(default_factory=list)
    link_scores: list[float] = field(default_factory=list)

    @property
    def merged_bbox(self) -> tuple[float, float, float, float]:
        """Merged bounding box of table + all linked captions."""
        all_boxes = [self.table] + self.captions
        return (
            min(r.x0 for r in all_boxes),
            min(r.y0 for r in all_boxes),
            max(r.x1 for r in all_boxes),
            max(r.y1 for r in all_boxes),
        )

    @property
    def table_bbox(self) -> tuple[float, float, float, float]:
        return self.table.bbox

    @property
    def caption_bboxes(self) -> list[tuple[float, float, float, float]]:
        return [c.bbox for c in self.captions]


def _horizontal_overlap(a: DetectedRegion, b: DetectedRegion) -> float:
    """
    Fraction of horizontal overlap between two regions.
    Returns overlap_width / min(a.width, b.width).
    """
    overlap_x0 = max(a.x0, b.x0)
    overlap_x1 = min(a.x1, b.x1)
    overlap_w = max(0, overlap_x1 - overlap_x0)
    min_w = min(a.width, b.width)
    return overlap_w / min_w if min_w > 0 else 0.0


def _vertical_gap(table: DetectedRegion, caption: DetectedRegion) -> float:
    """
    Signed vertical gap between table and caption.
    Negative = caption overlaps table vertically.
    Positive = gap between them.
    """
    if caption.cy < table.cy:
        # Caption is above table
        return table.y0 - caption.y1
    else:
        # Caption is below table
        return caption.y0 - table.y1


def _position(table: DetectedRegion, caption: DetectedRegion) -> str:
    """Is caption 'above' or 'below' the table?"""
    return "above" if caption.cy < table.cy else "below"


def compute_link_score(
    table: DetectedRegion,
    caption: DetectedRegion,
    max_distance_ratio: float = 1.5,
    min_horizontal_overlap: float = 0.3,
) -> float:
    """
    Compute a linking score between a table and a candidate caption.

    Score in [0, 1]. Higher = more likely to be linked. 0 = rejected.

    Factors:
    1. Horizontal overlap (must share column bounds)
    2. Vertical proximity (closer = better, max distance = 1.5× table height)
    3. Caption should be directly above or below (no intervening content)
    """
    # Check horizontal overlap
    h_overlap = _horizontal_overlap(table, caption)
    if h_overlap < min_horizontal_overlap:
        return 0.0

    # Check vertical distance
    v_gap = _vertical_gap(table, caption)
    max_distance = table.height * max_distance_ratio

    if v_gap > max_distance:
        return 0.0  # Too far away

    if v_gap < -min(table.height, caption.height) * 0.5:
        return 0.0  # Too much overlap (probably a detection error)

    # Score: combination of proximity and overlap
    # Proximity score: exponential decay with distance
    proximity_score = math.exp(-max(0, v_gap) / max(1, table.height * 0.5))

    # Horizontal alignment score
    alignment_score = min(1.0, h_overlap)

    # Combined score
    return proximity_score * alignment_score


def link_tables_captions(
    tables: list[DetectedRegion],
    captions: list[DetectedRegion],
    max_distance_ratio: float = 1.5,
    min_horizontal_overlap: float = 0.3,
) -> list[LinkedUnit]:
    """
    Link detected tables with detected captions using spatial rules.

    Each caption is linked to at most one table (the best match).
    A table can have multiple linked captions (above and below).

    Args:
        tables: detected table regions
        captions: detected caption regions
        max_distance_ratio: max gap as fraction of table height
        min_horizontal_overlap: minimum horizontal overlap fraction

    Returns:
        list of LinkedUnit, one per table
    """
    # Compute all pairwise link scores
    scores: list[tuple[int, int, float]] = []  # (table_idx, caption_idx, score)
    for t_idx, table in enumerate(tables):
        for c_idx, caption in enumerate(captions):
            score = compute_link_score(
                table, caption, max_distance_ratio, min_horizontal_overlap
            )
            if score > 0:
                scores.append((t_idx, c_idx, score))

    # Greedy assignment: each caption goes to its best-scoring table
    scores.sort(key=lambda x: -x[2])
    caption_assigned = set()
    table_captions: dict[int, list[tuple[int, float]]] = {
        i: [] for i in range(len(tables))
    }

    for t_idx, c_idx, score in scores:
        if c_idx in caption_assigned:
            continue
        table_captions[t_idx].append((c_idx, score))
        caption_assigned.add(c_idx)

    # Build linked units
    units = []
    for t_idx, table in enumerate(tables):
        linked_caps = []
        linked_scores = []
        for c_idx, score in table_captions[t_idx]:
            linked_caps.append(captions[c_idx])
            linked_scores.append(score)

        units.append(LinkedUnit(
            table=table,
            captions=linked_caps,
            link_scores=linked_scores,
        ))

    return units


def detect_and_link(
    image,
    processor,
    model,
    device,
    confidence: float = 0.5,
    max_distance_ratio: float = 1.5,
) -> list[LinkedUnit]:
    """
    Run detection and linking in one step.

    Uses the model to detect all objects, separates tables from captions
    by label, then links them spatially.

    NOTE: This requires a model trained with multi-class labels
    (label 0 = table, label 1 = caption). If the model only outputs
    one class, all detections are treated as tables with no linking.
    """
    import torch

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)
    results = processor.post_process_object_detection(
        outputs, threshold=confidence, target_sizes=target_sizes
    )[0]

    tables = []
    captions = []

    for score, label, box in zip(
        results["scores"].cpu().tolist(),
        results["labels"].cpu().tolist(),
        results["boxes"].cpu().tolist(),
    ):
        region = DetectedRegion(
            bbox=(box[0], box[1], box[2], box[3]),
            score=score,
            label="table" if label == 0 else "caption",
        )
        if label == 0:
            tables.append(region)
        elif label == 1:
            captions.append(region)

    if not captions:
        # Model doesn't output captions — return tables without linking
        return [LinkedUnit(table=t) for t in tables]

    return link_tables_captions(tables, captions, max_distance_ratio)


def link_from_gt_annotations(
    table_boxes: list[tuple],
    caption_boxes: list[tuple],
    max_distance_ratio: float = 1.5,
    min_horizontal_overlap: float = 0.3,
) -> list[LinkedUnit]:
    """
    Convenience function: link from raw (x0,y0,x1,y1) boxes.

    Useful for evaluating linking accuracy against GT annotations
    where parent_id associations are known.
    """
    tables = [DetectedRegion(bbox=b, label="table") for b in table_boxes]
    captions = [DetectedRegion(bbox=b, label="caption") for b in caption_boxes]
    return link_tables_captions(
        tables, captions, max_distance_ratio, min_horizontal_overlap
    )
