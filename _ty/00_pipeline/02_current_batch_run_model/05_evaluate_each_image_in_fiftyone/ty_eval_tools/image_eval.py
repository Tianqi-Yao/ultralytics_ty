"""
Per-image 评估导出工具：
将 FiftyOne evaluate_detections 的结果拆解为逐图统计行。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import fiftyone as fo

logger = logging.getLogger(__name__)


def _safe_dets(sample: fo.Sample, field: str) -> list:
    obj = sample.get_field(field)
    if obj is None:
        return []
    dets = getattr(obj, "detections", None)
    return dets if dets else []


def _confidence_stats(confidences: List[float], ious: List[float]) -> Dict[str, float]:
    """计算置信度和 IoU 的分布统计量。"""
    def stats(arr: List[float], prefix: str) -> Dict[str, float]:
        if not arr:
            return {f"{prefix}_{k}": 0.0 for k in
                    ["avg", "max", "min", "std", "median", "q1", "q3", "iqr"]}
        a = np.asarray(arr)
        q1, q3 = float(np.percentile(a, 25)), float(np.percentile(a, 75))
        return {
            f"{prefix}_avg": float(np.mean(a)),
            f"{prefix}_max": float(np.max(a)),
            f"{prefix}_min": float(np.min(a)),
            f"{prefix}_std": float(np.std(a)),
            f"{prefix}_median": float(np.median(a)),
            f"{prefix}_q1": q1,
            f"{prefix}_q3": q3,
            f"{prefix}_iqr": q3 - q1,
        }

    result = {}
    result.update(stats(confidences, "confidence"))
    result.update(stats(ious, "iou"))
    return result


def collect_image_stats(
    sample: fo.Sample,
    pred_field: str,
    gt_field: str,
    eval_key: str,
) -> Dict[str, Any]:
    """
    从单张图像中收集 TP/FP/FN、置信度列表、IoU 列表等统计数据。
    返回供 export_image_level_rows 使用的字典。
    """
    gt_dets   = _safe_dets(sample, gt_field)
    pred_dets = _safe_dets(sample, pred_field)

    confidences = [d.confidence for d in pred_dets if d.confidence is not None]
    ious = [
        getattr(d, f"{eval_key}_iou")
        for d in pred_dets
        if hasattr(d, f"{eval_key}_iou") and getattr(d, f"{eval_key}_iou") is not None
    ]

    tp_img = getattr(sample, f"{eval_key}_tp", 0)
    fp_img = getattr(sample, f"{eval_key}_fp", 0)
    fn_img = getattr(sample, f"{eval_key}_fn", 0)

    gt_count   = len(gt_dets)
    pred_count = len(pred_dets)
    gt_present   = gt_count > 0
    pred_present = pred_count > 0

    return {
        "gt_count_img":         gt_count,
        "pred_count_img":       pred_count,
        "tp_img":               tp_img,
        "fp_img":               fp_img,
        "fn_img":               fn_img,
        "tp_ratio":             tp_img / gt_count if gt_count > 0 else np.nan,
        "fp_ratio":             fp_img / pred_count if pred_count > 0 else np.nan,
        "fn_ratio":             fn_img / gt_count if gt_count > 0 else np.nan,
        "pred_gt_ratio":        pred_count / gt_count if gt_count > 0 else np.nan,
        "hit_img":              int(gt_present and pred_present),
        "miss_img":             int(gt_present and not pred_present),
        "false_alarm_img":      int(not gt_present and pred_present),
        "correct_reject_img":   int(not gt_present and not pred_present),
        "confidences":          confidences,
        "ious":                 ious,
        **_confidence_stats(confidences, ious),
    }


def export_image_level_rows(
    view: fo.DatasetView,
    dataset_name: str,
    subdir_name: str,
    subdir_path: str,
    model_tag: str,
    ckpt_path: str,
    pred_field: str,
    conf_thr: float,
    eval_key: str,
    version: str,
    iou_thr: float,
    gt_field: str,
    parse_dt_focus_fn,
) -> pd.DataFrame:
    """
    遍历 view 中所有样本，汇集逐图评估统计为 DataFrame。
    parse_dt_focus_fn: 接受 (filepath, year) 返回 (timestamp, focus) 的函数。
    """
    rows: List[Dict[str, Any]] = []

    for s in view.iter_samples(progress=True):
        stats = collect_image_stats(s, pred_field, gt_field, eval_key)

        capture_dt, focus = parse_dt_focus_fn(s.filepath)
        capture_date = None if pd.isna(capture_dt) else capture_dt.date()
        capture_time = None if pd.isna(capture_dt) else capture_dt.time()

        rows.append({
            "dataset_name":        dataset_name,
            "subdir_name":         subdir_name,
            "subdir_path":         subdir_path,
            "version":             version,
            "model_tag":           model_tag,
            "ckpt_path":           ckpt_path,
            "pred_field":          pred_field,
            "confidence_threshold": conf_thr,
            "iou_threshold":       iou_thr,
            "sample_id":           str(s.id),
            "filepath":            s.filepath,
            "capture_datetime":    capture_dt,
            "capture_date":        capture_date,
            "capture_time":        capture_time,
            "focus":               focus,
            **stats,
        })

    return pd.DataFrame(rows)


def summarize_from_image_df(img_df: pd.DataFrame) -> Dict[str, Any]:
    """从 per-image DataFrame 生成全局汇总统计。"""
    tp = int(img_df["tp_img"].sum())
    fp = int(img_df["fp_img"].sum())
    fn = int(img_df["fn_img"].sum())

    gt_total   = int(img_df["gt_count_img"].sum())
    pred_total = int(img_df["pred_count_img"].sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else float("nan")

    hit_img           = int(img_df["hit_img"].sum())
    miss_img          = int(img_df["miss_img"].sum())
    false_alarm_img   = int(img_df["false_alarm_img"].sum())
    correct_reject_img = int(img_df["correct_reject_img"].sum())

    denom_pos    = hit_img + miss_img
    img_recall   = hit_img / denom_pos if denom_pos > 0 else float("nan")
    denom_pred   = hit_img + false_alarm_img
    img_precision = hit_img / denom_pred if denom_pred > 0 else float("nan")
    img_f1 = (
        2 * img_precision * img_recall / (img_precision + img_recall)
        if (img_precision + img_recall) > 0 else float("nan")
    )

    return {
        "gt_total": gt_total, "pred_total": pred_total,
        "tp_total": tp, "fp_total": fp, "fn_total": fn,
        "precision_iou": precision, "recall_iou": recall, "f1_iou": f1,
        "hit_images": hit_img, "miss_images": miss_img,
        "false_alarm_images": false_alarm_img, "correct_reject_images": correct_reject_img,
        "img_precision": img_precision, "img_recall": img_recall, "img_f1": img_f1,
        "hit_rate_img": float(img_df["hit_img"].mean()) if len(img_df) else float("nan"),
    }
