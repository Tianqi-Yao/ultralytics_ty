"""
FiftyOne 评估结果提取工具
模拟 GUI 中的数据获取方式，支持批量 eval_key 遍历。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import fiftyone as fo

logger = logging.getLogger(__name__)


def _compute_avg_confidence(results) -> float:
    """从 results.ypred / results.confs 计算各类别平均置信度的均值。"""
    if not (hasattr(results, "confs") and hasattr(results, "ypred")):
        return 0.0

    counts: Dict[str, int] = {}
    sums: Dict[str, float] = {}
    for yp, conf in zip(results.ypred, results.confs):
        counts[yp] = counts.get(yp, 0) + 1
        sums[yp] = sums.get(yp, 0.0) + (conf if conf is not None else 0.0)

    avg_confs = {
        c: sums[c] / counts[c] if counts.get(c, 0) > 0 else 0.0
        for c in results.classes
    }
    return float(np.mean(list(avg_confs.values()))) if avg_confs else 0.0


def _get_tp_fp_fn(dataset: fo.Dataset, eval_key: str) -> Tuple[int, int, int]:
    """从 dataset 字段读取 TP/FP/FN 汇总值，字段不存在时返回 0。"""
    schema = dataset.get_field_schema()

    def safe_sum(field: str) -> int:
        return int(dataset.sum(field)) if field in schema else 0

    return (
        safe_sum(f"{eval_key}_tp"),
        safe_sum(f"{eval_key}_fp"),
        safe_sum(f"{eval_key}_fn"),
    )


def get_evaluation_data_like_gui(
    dataset: fo.Dataset,
    eval_key: str,
) -> Tuple[Dict[str, Any], Any]:
    """
    模拟 GUI 中的评估数据获取方式。

    Returns:
        (metrics, info)
        metrics 包含 precision/recall/f1/tp/fp/fn/map/mar/average_confidence
        info 是 FiftyOne EvaluationInfo 对象
    """
    results = dataset.load_evaluation_results(eval_key)
    info = dataset.get_evaluation_info(eval_key)
    metrics = results.metrics()

    metrics["average_confidence"] = _compute_avg_confidence(results)

    tp, fp, fn = _get_tp_fp_fn(dataset, eval_key)
    metrics["tp"] = tp
    metrics["fp"] = fp
    metrics["fn"] = fn

    try:
        metrics["map"] = results.mAP()
    except Exception:
        metrics["map"] = None

    try:
        metrics["mar"] = results.mAR()
    except Exception:
        metrics["mar"] = None

    return metrics, info
