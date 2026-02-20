"""
FiftyOne 预测结果导出：生成 per-image 和 per-detection DataFrame。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import fiftyone as fo

logger = logging.getLogger(__name__)


def _safe_dets(sample: fo.Sample, field: str) -> list:
    """安全获取 sample 中 field 的 detections 列表。"""
    obj = sample.get_field(field)
    if obj is None:
        return []
    return getattr(obj, "detections", None) or []


def _conf_stats(
    confs: List[float], high_thr: float = 0.85, low_thr: float = 0.50
) -> Dict[str, float]:
    """计算置信度分布统计量（count/mass/min/max/mean/std/分位数/高低置信统计）。"""
    if not confs:
        return {
            "conf_count": 0, "conf_mass": 0.0, "conf_min": 0.0, "conf_max": 0.0,
            "conf_mean": 0.0, "conf_std": 0.0, "conf_q05": 0.0, "conf_q25": 0.0,
            "conf_q50": 0.0, "conf_q75": 0.0, "conf_q95": 0.0, "conf_iqr": 0.0,
            f"high_conf_count_c{int(high_thr*100)}": 0,
            f"high_conf_mass_c{int(high_thr*100)}": 0.0,
            f"low_conf_ratio_c{int(low_thr*100)}": 0.0,
        }
    arr = np.asarray(confs, dtype=float)
    q05, q25, q50, q75, q95 = np.percentile(arr, [5, 25, 50, 75, 95])
    high = arr[arr >= high_thr]
    return {
        "conf_count": int(arr.size), "conf_mass": float(arr.sum()),
        "conf_min": float(arr.min()), "conf_max": float(arr.max()),
        "conf_mean": float(arr.mean()), "conf_std": float(arr.std()),
        "conf_q05": float(q05), "conf_q25": float(q25), "conf_q50": float(q50),
        "conf_q75": float(q75), "conf_q95": float(q95), "conf_iqr": float(q75 - q25),
        f"high_conf_count_c{int(high_thr*100)}": int(high.size),
        f"high_conf_mass_c{int(high_thr*100)}": float(high.sum()) if high.size else 0.0,
        f"low_conf_ratio_c{int(low_thr*100)}": float(np.mean(arr < low_thr)),
    }


def _build_image_row(
    sample: fo.Sample,
    pred_field: str,
    dataset_name: str,
    parse_dt_fn: Callable,
    high_thr: float,
    low_thr: float,
) -> Dict[str, Any]:
    """构建单张图像的统计行 dict。"""
    dets = _safe_dets(sample, pred_field)
    confs = [d.confidence for d in dets if d.confidence is not None]
    capture_dt, focus = parse_dt_fn(sample.filepath)
    date = None if pd.isna(capture_dt) else capture_dt.date()
    time = None if pd.isna(capture_dt) else capture_dt.time()
    return {
        "dataset_name": dataset_name, "pred_field": pred_field,
        "sample_id": str(sample.id), "filepath": sample.filepath,
        "datetime": capture_dt, "date": date, "time": time, "focus": focus,
        "pred_count": len(dets),
        **_conf_stats(confs, high_thr, low_thr),
        "confidences_list": confs,
    }


def _build_det_rows(
    sample: fo.Sample,
    pred_field: str,
    dataset_name: str,
    parse_dt_fn: Callable,
) -> List[Dict[str, Any]]:
    """构建单张图像所有检测框的行 list。"""
    dets = _safe_dets(sample, pred_field)
    capture_dt, focus = parse_dt_fn(sample.filepath)
    date = None if pd.isna(capture_dt) else capture_dt.date()
    time = None if pd.isna(capture_dt) else capture_dt.time()
    rows = []
    for i, d in enumerate(dets):
        bb = getattr(d, "bounding_box", None)
        bx, by, bw, bh = (bb if bb and len(bb) == 4 else [np.nan] * 4)
        rows.append({
            "dataset_name": dataset_name, "pred_field": pred_field,
            "sample_id": str(sample.id), "filepath": sample.filepath,
            "datetime": capture_dt, "date": date, "time": time, "focus": focus,
            "det_idx": i, "confidence": getattr(d, "confidence", np.nan),
            "bbox_x": bx, "bbox_y": by, "bbox_w": bw, "bbox_h": bh,
            "bbox_area": float(bw * bh) if not np.isnan(float(bx)) else np.nan,
        })
    return rows


def export_per_image_and_detection_dfs(
    ds: fo.Dataset,
    pred_field: str,
    parse_dt_fn: Callable,
    high_conf_thr: float = 0.85,
    low_conf_thr: float = 0.50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """遍历 dataset，返回 (per_image_df, per_detection_df)。"""
    dataset_name = ds.name
    image_rows: List[Dict[str, Any]] = []
    det_rows: List[Dict[str, Any]] = []
    for s in ds.iter_samples(progress=True):
        image_rows.append(
            _build_image_row(s, pred_field, dataset_name, parse_dt_fn, high_conf_thr, low_conf_thr)
        )
        det_rows.extend(_build_det_rows(s, pred_field, dataset_name, parse_dt_fn))
    return pd.DataFrame(image_rows), pd.DataFrame(det_rows)
