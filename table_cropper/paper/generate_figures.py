"""
Generate paper figures from real SCI-3000 data.

Produces:
  fig1_semantic_unit.pdf   — Example of semantically complete table unit
  fig2_formulation_comparison.pdf — Formulations A, B, C side by side
  fig3_heuristic_failures.pdf     — Over-crop and under-capture examples
  fig4_metric_schematic.pdf       — Why AP and completeness disagree
"""

import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

# ── Paths ──
DATA_DIR = os.path.join(os.path.expanduser("~"), ".cache", "table_cropper", "SCI-3000")
VAL_IMG_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "table_cropper", "finetune_data_v3", "val", "images"
)
CHECKPOINT = os.path.join(
    os.path.expanduser("~"), ".cache", "table_cropper", "checkpoints", "phase5_resumed_final"
)
OUT_DIR = "/root/autoresearch/table_cropper/paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)


# ── Annotation parsing ──
def parse_bbox(value):
    m = re.match(r"xywh=pixel:([\d.]+),([\d.]+),([\d.]+),([\d.]+)", value)
    if not m:
        return None
    return tuple(float(m.group(i)) for i in range(1, 5))


def parse_page(json_path):
    with open(json_path) as f:
        data = json.load(f)
    cw, ch = data.get("canvasWidth", 0), data.get("canvasHeight", 0)
    tables, captions = [], []
    for ann in data.get("annotations", []):
        ann_id = ann.get("id", "")
        bodies = ann.get("body", [])
        ann_type, parent_id = None, None
        for b in bodies:
            if b.get("purpose") == "img-cap-enum":
                ann_type = b.get("value")
            elif b.get("purpose") == "parent":
                parent_id = b.get("value")
        try:
            bbox = parse_bbox(ann["target"]["selector"]["value"])
        except (KeyError, ValueError, TypeError):
            continue
        if bbox is None:
            continue
        info = {"id": ann_id, "type": ann_type, "bbox": bbox, "parent_id": parent_id}
        if ann_type == "Table":
            tables.append(info)
        elif ann_type == "Caption":
            captions.append(info)
    return {"canvas": (cw, ch), "tables": tables, "captions": captions}


def xywh_to_xyxy(bbox, sx, sy):
    x, y, w, h = bbox
    return (x * sx, y * sy, (x + w) * sx, (y + h) * sy)


def find_linked_caption(table, captions):
    tid = table["id"]
    for c in captions:
        if c.get("parent_id") == tid:
            return c
    return None


def merge_boxes(boxes):
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return (x0, y0, x1, y1)


def find_good_example(pages, img_dir):
    """Find a page with a table+caption where caption is clearly separate."""
    best = None
    best_score = 0
    for ann_path, parsed in pages:
        pid = Path(ann_path).stem
        img_path = os.path.join(img_dir, f"{pid}.png")
        if not os.path.exists(img_path):
            continue
        cw, ch = parsed["canvas"]
        if cw == 0 or ch == 0:
            continue
        img = Image.open(img_path)
        iw, ih = img.size
        sx, sy = iw / cw, ih / ch

        for t in parsed["tables"]:
            cap = find_linked_caption(t, parsed["captions"])
            if cap is None:
                continue
            tb = xywh_to_xyxy(t["bbox"], sx, sy)
            cb = xywh_to_xyxy(cap["bbox"], sx, sy)
            # Caption should be above table and clearly separated
            cap_h = cb[3] - cb[1]
            tab_h = tb[3] - tb[1]
            gap = tb[1] - cb[3]
            if gap > 5 and cap_h > 20 and tab_h > 100:
                score = tab_h + cap_h
                if score > best_score:
                    best_score = score
                    best = (img_path, tb, cb, t, cap, sx, sy, parsed)
    return best


# ── Load pages ──
def load_pages():
    annot_dir = os.path.join(DATA_DIR, "Annotations")
    page_ids = [Path(f).stem for f in os.listdir(VAL_IMG_DIR) if f.endswith(".png")]
    results = []
    for pid in sorted(page_ids):  # Scan all pages for best example
        ann_path = os.path.join(annot_dir, f"{pid}.json")
        if not os.path.exists(ann_path):
            continue
        parsed = parse_page(ann_path)
        if parsed["tables"]:
            results.append((ann_path, parsed))
    return results


print("Loading pages...")
pages = load_pages()
print(f"Loaded {len(pages)} pages with tables")


# ═══════════════════════════════════════════════════════════════════
# Figure 1: Semantically Complete Table Unit
# ═══════════════════════════════════════════════════════════════════
print("Generating Figure 1...")
example = find_good_example(pages, VAL_IMG_DIR)

if example:
    img_path, tb, cb, table_ann, cap_ann, sx, sy, parsed = example
    img = Image.open(img_path)

    merged = merge_boxes([tb, cb])

    fig, ax = plt.subplots(1, 1, figsize=(7, 10))
    ax.imshow(img, cmap="gray")

    # Table body (blue dashed)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    rect_table = patches.Rectangle(
        (tb[0], tb[1]), tw, th,
        linewidth=2.5, edgecolor="#2196F3", facecolor="none",
        linestyle="--", label="Table body"
    )
    ax.add_patch(rect_table)

    # Title (orange dashed)
    cw_px, ch_px = cb[2] - cb[0], cb[3] - cb[1]
    rect_cap = patches.Rectangle(
        (cb[0], cb[1]), cw_px, ch_px,
        linewidth=2.5, edgecolor="#FF9800", facecolor="none",
        linestyle="--", label="Title"
    )
    ax.add_patch(rect_cap)

    # Merged box (green solid)
    mw, mh = merged[2] - merged[0], merged[3] - merged[1]
    rect_merged = patches.Rectangle(
        (merged[0], merged[1]), mw, mh,
        linewidth=3.0, edgecolor="#4CAF50", facecolor=(0.3, 0.8, 0.3, 0.08),
        linestyle="-", label="Merged (semantic unit)"
    )
    ax.add_patch(rect_merged)

    ax.set_xlim(max(0, merged[0] - 50), min(img.size[0], merged[2] + 50))
    ax.set_ylim(min(img.size[1], merged[3] + 80), max(0, cb[1] - 80))
    ax.set_axis_off()
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_semantic_unit.pdf"), bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(OUT_DIR, "fig1_semantic_unit.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  Fig 1 saved")
else:
    print("  WARNING: No suitable example found for Fig 1")


# ═══════════════════════════════════════════════════════════════════
# Figure 2: Formulation A vs B vs C Comparison
# ═══════════════════════════════════════════════════════════════════
print("Generating Figure 2...")
if example:
    img_path, tb, cb, table_ann, cap_ann, sx, sy, parsed = example
    img = Image.open(img_path)
    merged = merge_boxes([tb, cb])

    # Simulate heuristic expansion (wider than merged, includes some extra)
    heuristic_pad_top = 40
    heuristic_pad_bottom = 60
    heuristic_pad_side = 20
    heuristic_box = (
        max(0, tb[0] - heuristic_pad_side),
        max(0, cb[1] - heuristic_pad_top),
        min(img.size[0], tb[2] + heuristic_pad_side),
        min(img.size[1], tb[3] + heuristic_pad_bottom),
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    titles = [
        "Formulation A\n(Table-only)",
        "Formulation B\n(Merged / learned)",
        "Formulation C\n(Heuristic expansion)"
    ]
    boxes = [tb, merged, heuristic_box]
    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    for ax, title, box, color in zip(axes, titles, boxes, colors):
        ax.imshow(img, cmap="gray")
        bw, bh = box[2] - box[0], box[3] - box[1]
        rect = patches.Rectangle(
            (box[0], box[1]), bw, bh,
            linewidth=3.0, edgecolor=color, facecolor=(*matplotlib.colors.to_rgb(color), 0.08),
            linestyle="-"
        )
        ax.add_patch(rect)

        # Also show GT elements lightly
        tw, th2 = tb[2] - tb[0], tb[3] - tb[1]
        gt_table = patches.Rectangle(
            (tb[0], tb[1]), tw, th2,
            linewidth=1.0, edgecolor="gray", facecolor="none", linestyle=":"
        )
        ax.add_patch(gt_table)
        cw2, ch2 = cb[2] - cb[0], cb[3] - cb[1]
        gt_cap = patches.Rectangle(
            (cb[0], cb[1]), cw2, ch2,
            linewidth=1.0, edgecolor="gray", facecolor="none", linestyle=":"
        )
        ax.add_patch(gt_cap)

        pad = 80
        ax.set_xlim(max(0, merged[0] - pad), min(img.size[0], merged[2] + pad))
        ax.set_ylim(min(img.size[1], max(tb[3], heuristic_box[3]) + pad),
                     max(0, min(cb[1], heuristic_box[1]) - pad))
        ax.set_axis_off()
        ax.set_title(title, fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_formulation_comparison.pdf"), bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(OUT_DIR, "fig2_formulation_comparison.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  Fig 2 saved")


# ═══════════════════════════════════════════════════════════════════
# Figure 3: Heuristic Failure Cases (Over-crop & Under-capture)
# ═══════════════════════════════════════════════════════════════════
print("Generating Figure 3...")

# Find two examples: one where aggressive heuristic over-crops, one where tight under-captures
overcrop_example = None
undercapture_example = None

for ann_path, parsed in pages:
    pid = Path(ann_path).stem
    img_path = os.path.join(VAL_IMG_DIR, f"{pid}.png")
    if not os.path.exists(img_path):
        continue
    cw, ch = parsed["canvas"]
    if cw == 0 or ch == 0:
        continue
    img = Image.open(img_path)
    iw, ih = img.size
    sx2, sy2 = iw / cw, ih / ch

    for t in parsed["tables"]:
        cap = find_linked_caption(t, parsed["captions"])
        if cap is None:
            continue
        tb2 = xywh_to_xyxy(t["bbox"], sx2, sy2)
        cb2 = xywh_to_xyxy(cap["bbox"], sx2, sy2)
        gap = tb2[1] - cb2[3]
        cap_h = cb2[3] - cb2[1]
        tab_h = tb2[3] - tb2[1]

        # Over-crop: caption far above table (large gap = aggressive heuristic grabs too much)
        if gap > 80 and cap_h > 20 and tab_h > 150 and overcrop_example is None:
            overcrop_example = (img_path, tb2, cb2, sx2, sy2)

        # Under-capture: caption very close or overlapping (tight heuristic misses it)
        if 0 < gap < 20 and cap_h > 20 and tab_h > 100 and undercapture_example is None:
            undercapture_example = (img_path, tb2, cb2, sx2, sy2)

        if overcrop_example and undercapture_example:
            break
    if overcrop_example and undercapture_example:
        break

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 7))

for ax, (ex, subtitle, pad_top, pad_bot) in zip(axes3, [
    (overcrop_example, "(a) Aggressive expansion:\nexcessive over-crop", 200, 100),
    (undercapture_example or overcrop_example, "(b) Tight expansion:\ncaption under-capture", 30, 10),
]):
    if ex is None:
        ax.text(0.5, 0.5, "No suitable example", ha="center", va="center", transform=ax.transAxes)
        continue
    ip, tb3, cb3, _, _ = ex
    im = Image.open(ip)
    ax.imshow(im, cmap="gray")

    # GT table body
    tw3, th3 = tb3[2] - tb3[0], tb3[3] - tb3[1]
    ax.add_patch(patches.Rectangle(
        (tb3[0], tb3[1]), tw3, th3,
        linewidth=2, edgecolor="#2196F3", facecolor="none", linestyle="--", label="Table body"
    ))
    # GT caption
    cw3, ch3 = cb3[2] - cb3[0], cb3[3] - cb3[1]
    ax.add_patch(patches.Rectangle(
        (cb3[0], cb3[1]), cw3, ch3,
        linewidth=2, edgecolor="#FF9800", facecolor="none", linestyle="--", label="Title"
    ))

    # Heuristic box
    hbox = (
        max(0, tb3[0] - 15),
        max(0, tb3[1] - pad_top),
        min(im.size[0], tb3[2] + 15),
        min(im.size[1], tb3[3] + pad_bot),
    )
    hw, hh = hbox[2] - hbox[0], hbox[3] - hbox[1]
    ax.add_patch(patches.Rectangle(
        (hbox[0], hbox[1]), hw, hh,
        linewidth=2.5, edgecolor="#FF5722", facecolor=(1, 0.34, 0.13, 0.06),
        linestyle="-", label="Heuristic crop"
    ))

    merged3 = merge_boxes([tb3, cb3])
    pad = 60
    ax.set_xlim(max(0, min(hbox[0], merged3[0]) - pad),
                min(im.size[0], max(hbox[2], merged3[2]) + pad))
    ax.set_ylim(min(im.size[1], max(hbox[3], merged3[3]) + pad),
                max(0, min(hbox[1], merged3[1]) - pad))
    ax.set_axis_off()
    ax.set_title(subtitle, fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, "fig3_heuristic_failures.pdf"), bbox_inches="tight", dpi=200)
fig3.savefig(os.path.join(OUT_DIR, "fig3_heuristic_failures.png"), bbox_inches="tight", dpi=200)
plt.close(fig3)
print("  Fig 3 saved")


# ═══════════════════════════════════════════════════════════════════
# Figure 4: Why AP and Completeness Disagree (Schematic)
# ═══════════════════════════════════════════════════════════════════
print("Generating Figure 4...")

fig4, axes4 = plt.subplots(1, 2, figsize=(10, 5))

for ax, (title, show_caption_included) in zip(axes4, [
    ("(a) High table-body IoU\nbut zero caption inclusion", False),
    ("(b) High caption inclusion\nand complete unit capture", True),
]):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 130)
    ax.set_aspect("equal")

    # Page background
    ax.add_patch(patches.FancyBboxPatch(
        (5, 5), 90, 120, boxstyle="round,pad=2",
        facecolor="#f5f5f5", edgecolor="#ccc", linewidth=1
    ))

    # Caption region
    ax.add_patch(patches.Rectangle(
        (15, 95), 70, 18,
        facecolor="#FFF3E0", edgecolor="#FF9800", linewidth=2, linestyle="--"
    ))
    ax.text(50, 104, "Table 3: Primary endpoints...", ha="center", va="center",
            fontsize=7, fontstyle="italic", color="#E65100")

    # Table body region
    ax.add_patch(patches.Rectangle(
        (15, 25), 70, 65,
        facecolor="#E3F2FD", edgecolor="#2196F3", linewidth=2, linestyle="--"
    ))
    ax.text(50, 57, "Table Body\n(data grid)", ha="center", va="center",
            fontsize=9, color="#1565C0")

    # Detection box
    if show_caption_included:
        # Merged box covering both
        ax.add_patch(patches.Rectangle(
            (13, 23), 74, 93,
            facecolor=(0.3, 0.8, 0.3, 0.1), edgecolor="#4CAF50", linewidth=3
        ))
        ax.text(50, 14, "IoU(body)=0.80  CIR=0.94  CUCR=1",
                ha="center", fontsize=7, color="#2E7D32", fontweight="bold")
    else:
        # Table-body-only box
        ax.add_patch(patches.Rectangle(
            (14, 24), 72, 67,
            facecolor=(0.13, 0.59, 0.95, 0.1), edgecolor="#2196F3", linewidth=3
        ))
        ax.text(50, 14, "IoU(body)=0.94  CIR=0.00  CUCR=0",
                ha="center", fontsize=7, color="#1565C0", fontweight="bold")

    ax.set_axis_off()
    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)

fig4.suptitle("Standard AP rewards (a) while completeness metrics reveal (b) as more useful",
              fontsize=10, y=0.02, fontstyle="italic", color="#555")
fig4.tight_layout(rect=[0, 0.05, 1, 1])
fig4.savefig(os.path.join(OUT_DIR, "fig4_metric_schematic.pdf"), bbox_inches="tight", dpi=200)
fig4.savefig(os.path.join(OUT_DIR, "fig4_metric_schematic.png"), bbox_inches="tight", dpi=200)
plt.close(fig4)
print("  Fig 4 saved")

print(f"\nAll figures saved to {OUT_DIR}")
