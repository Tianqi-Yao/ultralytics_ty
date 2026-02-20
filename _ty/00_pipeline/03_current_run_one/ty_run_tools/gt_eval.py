"""
GT vs 预测结果评估工具：基于 COCO JSON，按文件名计算 per-image TP/FP/FN。
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def _load_json(path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _build_id2name(coco: dict) -> Dict[int, str]:
    return {img["id"]: img["file_name"] for img in coco.get("images", [])}


def _annotations_to_filename_df(coco: dict, id2name: Dict[int, str]) -> pd.DataFrame:
    """将 COCO annotations 转为 (file_name, bbox, score) DataFrame。"""
    ann = coco.get("annotations", [])
    if not ann:
        return pd.DataFrame(columns=["file_name", "bbox", "score"])
    df = pd.DataFrame(ann)
    df = df[["image_id", "bbox"] + (["score"] if "score" in df.columns else [])].copy()
    if "score" not in df.columns:
        df["score"] = float("nan")
    df["file_name"] = df["image_id"].map(id2name)
    return df.dropna(subset=["file_name"])[["file_name", "bbox", "score"]]


def _iou_coco(b1, b2) -> float:
    """计算两个 COCO 格式 [x, y, w, h] 框的 IoU。"""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    union = max(0.0, w1) * max(0.0, h1) + max(0.0, w2) * max(0.0, h2) - inter
    return inter / union if union > 0 else 0.0


def _match_pred_to_gt(
    pred_boxes: list, gt_boxes: list, iou_thr: float
) -> Tuple[int, int, Set[int]]:
    """贪心匹配预测框到 GT 框，返回 (tp, fp, matched_gt_indices_set)。"""
    matched_gt: Set[int] = set()
    tp = fp = 0
    for pb in pred_boxes:
        best_iou, best_idx = 0.0, -1
        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = _iou_coco(pb, gb)
            if iou > best_iou:
                best_iou, best_idx = iou, gi
        if best_iou >= iou_thr and best_idx != -1:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1
    return tp, fp, matched_gt


def per_image_eval_by_filename(
    gt_path,
    pred_path,
    iou_thr: float = 0.9,
    score_thr: Optional[float] = None,
) -> pd.DataFrame:
    """加载 GT/pred COCO JSON，按文件名匹配，返回 per-image 评估指标 DataFrame。"""
    gt, pred = _load_json(gt_path), _load_json(pred_path)
    gt_id2name, pred_id2name = _build_id2name(gt), _build_id2name(pred)
    gt_df = _annotations_to_filename_df(gt, gt_id2name)
    pred_df = _annotations_to_filename_df(pred, pred_id2name)
    if score_thr is not None:
        pred_df = pred_df[pred_df["score"] >= score_thr]
    gt_group   = gt_df.groupby("file_name")["bbox"].apply(list).to_dict()
    pred_group = pred_df.groupby("file_name")["bbox"].apply(list).to_dict()
    all_files  = sorted(set(gt_id2name.values()) | set(pred_id2name.values()) | set(gt_group) | set(pred_group))
    rows = []
    for fn in all_files:
        gt_boxes, pred_boxes = gt_group.get(fn, []), pred_group.get(fn, [])
        tp, fp, matched = _match_pred_to_gt(pred_boxes, gt_boxes, iou_thr)
        fn_count  = len(gt_boxes) - len(matched)
        precision = tp / len(pred_boxes) if pred_boxes else 0.0
        recall    = tp / len(gt_boxes)   if gt_boxes   else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({
            "file_name": fn, "gt_count": len(gt_boxes), "pred_count": len(pred_boxes),
            "tp (Hit)": tp, "fp (Error Warning)": fp, "fn (Miss)": fn_count,
            "precision": precision, "recall": recall, "f1": f1,
        })
    return pd.DataFrame(rows).sort_values("file_name").reset_index(drop=True)
