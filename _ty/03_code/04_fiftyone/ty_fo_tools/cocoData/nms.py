# ty_fo_tools/coco_nms.py
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple


def _bbox_iou(b1, b2) -> float:
    """
    IoU for COCO bboxes: [x, y, w, h]
    """
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    xa1, ya1 = x1, y1
    xa2, ya2 = x1 + w1, y1 + h1

    xb1, yb1 = x2, y2
    xb2, yb2 = x2 + w2, y2 + h2

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = w1 * h1
    area_b = w2 * h2
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _nms_simple(anns: List[dict], iou_thresh: float) -> List[dict]:
    """
    Simple NMS within a group of anns (same image_id, same category_id).
    Strategy: sort by bbox area desc, greedy keep if IoU <= thresh w.r.t kept.
    """
    def area(ann: dict) -> float:
        x, y, w, h = ann["bbox"]
        return float(w) * float(h)

    anns_sorted = sorted(anns, key=area, reverse=True)
    kept: List[dict] = []
    for ann in anns_sorted:
        keep_this = True
        for k in kept:
            if _bbox_iou(ann["bbox"], k["bbox"]) > iou_thresh:
                keep_this = False
                break
        if keep_this:
            kept.append(ann)
    return kept


def coco_nms_json(
    *,
    input_json: Path | str,
    output_json: Path | str,
    iou_thresh: float = 0.5,
    per_category: bool = True,
) -> Dict[str, int]:
    """
    Run simple NMS de-dup on COCO annotations.

    Grouping:
        - if per_category=True: group by (image_id, category_id)
        - else: group by (image_id)

    Returns:
        efore": int, "after": int}
    """
    input_json = Path(input_json)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    coco = json.loads(input_json.read_text(encoding="utf-8"))
    annotations = coco.get("annotations", [])

    groups: Dict[Tuple[int, int] | int, List[dict]] = {}
    for ann in annotations:
        if per_category:
            key = (ann["image_id"], ann["category_id"])
        else:
            key = ann["image_id"]
        groups.setdefault(key, []).append(ann)

    new_annotations: List[dict] = []
    for _, anns in groups.items():
        new_annotations.extend(_nms_simple(anns, iou_thresh=iou_thresh))

    new_coco = deepcopy(coco)
    new_coco["annotations"] = new_annotations

    output_json.write_text(json.dumps(new_coco, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"before": len(annotations), "after": len(new_annotations)}
