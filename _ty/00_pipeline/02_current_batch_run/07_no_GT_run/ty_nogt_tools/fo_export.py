"""
无 GT 预测结果导出工具：将 FiftyOne datasets 的预测结果汇总为 Excel。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import fiftyone as fo

logger = logging.getLogger(__name__)


def _safe_dets(sample: fo.Sample, pred_field: str) -> list:
    """安全获取 sample 中 pred_field 的 detections 列表。"""
    if not sample.has_field(pred_field):
        return []
    val = sample[pred_field]
    if val is None:
        return []
    return getattr(val, "detections", None) or []


def _conf_stats(confs: List[float]) -> Dict[str, float]:
    """计算置信度统计量（max/mean/p50/p90）。"""
    if not confs:
        return {"conf_max": np.nan, "conf_mean": np.nan, "conf_p50": np.nan, "conf_p90": np.nan}
    arr = np.asarray(confs, dtype=float)
    return {
        "conf_max": float(np.nanmax(arr)),
        "conf_mean": float(np.nanmean(arr)),
        "conf_p50": float(np.nanpercentile(arr, 50)),
        "conf_p90": float(np.nanpercentile(arr, 90)),
    }


def _iter_sample_rows(
    ds: fo.Dataset, model_tag: str, pred_field: str
) -> Tuple[List[Dict[str, Any]], int, List[float]]:
    """遍历 dataset 样本，收集每图统计行、有预测图数量、所有置信度。"""
    per_image_rows: List[Dict[str, Any]] = []
    images_with_pred = 0
    confs_all: List[float] = []

    for sample in ds.iter_samples(progress=True):
        det_list = _safe_dets(sample, pred_field)
        n_det = len(det_list)
        if n_det > 0:
            images_with_pred += 1
        confs = [float(d.confidence) for d in det_list if d.confidence is not None]
        confs_all.extend(confs)
        row = {
            "dataset": ds.name,
            "sample_id": str(sample.id),
            "filepath": sample.filepath,
            "filename": Path(sample.filepath).name,
            "model_tag": model_tag,
            "pred_field": pred_field,
            "pred_count": n_det,
        }
        row.update(_conf_stats(confs))
        per_image_rows.append(row)

    return per_image_rows, images_with_pred, confs_all


def export_two_excels_for_dataset(
    ds: fo.Dataset,
    pred_fields: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """对 dataset 中每个 (model_tag, pred_field) 生成 summary + per_image 行。"""
    summary_rows: List[Dict[str, Any]] = []
    all_per_image_rows: List[Dict[str, Any]] = []
    n_samples = len(ds)

    for model_tag, pred_field in pred_fields.items():
        if pred_field not in ds.get_field_schema():
            logger.warning(f"{ds.name} 缺少 pred_field={pred_field}，跳过")
            continue
        total_pred = int(ds.count(f"{pred_field}.detections"))
        per_img_rows, images_with_pred, confs_all = _iter_sample_rows(ds, model_tag, pred_field)
        all_per_image_rows.extend(per_img_rows)
        s = {
            "dataset": ds.name,
            "model_tag": model_tag,
            "pred_field": pred_field,
            "num_samples": n_samples,
            "total_pred_count": total_pred,
            "images_with_pred": images_with_pred,
            "pct_images_with_pred": (images_with_pred / n_samples * 100.0) if n_samples else np.nan,
            "avg_pred_per_image": (total_pred / n_samples) if n_samples else np.nan,
        }
        s.update(_conf_stats(confs_all))
        summary_rows.append(s)

    return pd.DataFrame(summary_rows), pd.DataFrame(all_per_image_rows)
