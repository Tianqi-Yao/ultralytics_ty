"""
04_evaluate.py
评估 CLF_FIELD 的结果：
  - 有 GT：evaluate_detections → TP/FP/FN，打 detection-level tag，计算指标
  - 无 GT：统计框数/置信度分布，输出候选框数量
结果保存 RESULTS_DIR/eval_r{ROUND}.json
"""

from __future__ import annotations
import json
import logging
from pathlib import Path

import fiftyone as fo
from fiftyone import ViewField as F

from pipeline_config import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def eval_with_gt(ds: fo.Dataset) -> dict:
    """有 GT 时，用 FiftyOne 官方 evaluate_detections 评估。"""
    logger.info(f"评估（有 GT）: {CLF_FIELD} vs {GT_FIELD}  eval_key={EVAL_KEY}")

    results = ds.evaluate_detections(
        CLF_FIELD,
        gt_field=GT_FIELD,
        eval_key=EVAL_KEY,
        iou=0.5,
        compute_mAP=True,
    )

    # 打印内置报告
    results.print_report()

    # 提取关键指标
    report = results.report()
    # report 结构：{"swd": {"precision": .., "recall": .., "f1-score": .., "support": ..}, ...}
    cls_report = report.get("swd", report.get(CLASS_NAMES[0], {}))
    precision  = cls_report.get("precision", 0.0)
    recall     = cls_report.get("recall", 0.0)
    f1         = cls_report.get("f1-score", 0.0)
    support    = cls_report.get("support", 0)
    map50      = results.mAP()

    # 统计 TP/FP/FN 数量（detection-level）
    tp_view = ds.filter_labels(CLF_FIELD, F(f"{EVAL_KEY}_iou") > 0)
    tp = len(ds.filter_labels(CLF_FIELD, F(f"{EVAL_KEY}") == "tp").values(f"{CLF_FIELD}.detections", unwind=True))
    fp = len(ds.filter_labels(CLF_FIELD, F(f"{EVAL_KEY}") == "fp").values(f"{CLF_FIELD}.detections", unwind=True))
    fn = len(ds.filter_labels(GT_FIELD,  F(f"{EVAL_KEY}") == "fn").values(f"{GT_FIELD}.detections",  unwind=True))

    metrics = {
        "round":     ROUND,
        "mode":      "with_gt",
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "support":   support,
        "mAP50":     round(map50, 4),
    }

    logger.info(f"TP={tp}  FP={fp}  FN={fn}  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}  mAP@0.5={map50:.4f}")
    return metrics


def eval_without_gt(ds: fo.Dataset) -> dict:
    """无 GT 时，统计 CLF_FIELD 的框分布。"""
    logger.info(f"评估（无 GT）: 统计 {CLF_FIELD} 框分布")

    view = ds.exists(CLF_FIELD)
    all_dets = view.values(f"{CLF_FIELD}.detections", unwind=True)

    total_boxes    = len(all_dets)
    images_w_boxes = sum(1 for s in view if s[CLF_FIELD] and s[CLF_FIELD].detections)

    # 置信度统计（clf_score 属性）
    scores = []
    for det in all_dets:
        sc = det.get_attribute_value("clf_score", None)
        if sc is not None:
            scores.append(float(sc))

    import numpy as np
    metrics = {
        "round":          ROUND,
        "mode":           "no_gt",
        "total_boxes":    total_boxes,
        "images_w_boxes": images_w_boxes,
        "score_mean":     round(float(np.mean(scores)),   4) if scores else None,
        "score_p50":      round(float(np.median(scores)), 4) if scores else None,
        "score_p90":      round(float(np.percentile(scores, 90)), 4) if scores else None,
    }

    logger.info(f"候选框总数: {total_boxes}  含框图片: {images_w_boxes}")
    if scores:
        logger.info(f"置信度: mean={metrics['score_mean']}  p50={metrics['score_p50']}  p90={metrics['score_p90']}")

    return metrics


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ds = fo.load_dataset(FO_DATASET)

    if not ds.has_sample_field(CLF_FIELD):
        raise RuntimeError(f"请先运行 03_run_classifier.py（{CLF_FIELD} 字段不存在）")

    has_gt = ds.has_sample_field(GT_FIELD) and len(ds.exists(GT_FIELD)) > 0

    if has_gt:
        metrics = eval_with_gt(ds)
    else:
        metrics = eval_without_gt(ds)

    # 保存 JSON
    out_path = RESULTS_DIR / f"eval_r{ROUND}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"评估结果已保存: {out_path}")

    if has_gt:
        logger.info(f"在 FiftyOne App 中可按 TP/FP/FN 过滤：")
        logger.info(f"  TP: ds.filter_labels('{CLF_FIELD}', F('{EVAL_KEY}') == 'tp')")
        logger.info(f"  FP: ds.filter_labels('{CLF_FIELD}', F('{EVAL_KEY}') == 'fp')")
        logger.info(f"  FN: ds.filter_labels('{GT_FIELD}',  F('{EVAL_KEY}') == 'fn')")


if __name__ == "__main__":
    main()
