"""
分类模型二次过滤工具：
在已有 YOLO 检测框基础上，用分类模型对每个 patch 进行二次过滤。
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

from PIL import Image
import fiftyone as fo

logger = logging.getLogger(__name__)


def crop_with_padding(pil_img: Image.Image, det: fo.Detection, pad_ratio: float = 0.2) -> Optional[Image.Image]:
    """从图像中裁出一个检测框对应的 patch（带边缘 padding）。"""
    W, H = pil_img.size
    x, y, w, h = det.bounding_box
    x1, y1 = x * W, y * H
    x2, y2 = (x + w) * W, (y + h) * H

    pad_w = (x2 - x1) * pad_ratio
    pad_h = (y2 - y1) * pad_ratio
    x1 = max(0, int(x1 - pad_w))
    y1 = max(0, int(y1 - pad_h))
    x2 = min(W, int(x2 + pad_w))
    y2 = min(H, int(y2 + pad_h))

    if x2 <= x1 or y2 <= y1:
        return None
    return pil_img.crop((x1, y1, x2, y2))


def _batch_classify(
    crops: List[Image.Image],
    idxs: List[int],
    clf_predict_fn: Callable,
    batch_size: int,
) -> dict[int, Tuple[str, float]]:
    """对 crops 批量调用分类函数，返回 {det_idx: (label, score)} 映射。"""
    pred_map: dict[int, Tuple[str, float]] = {}
    for s in range(0, len(crops), batch_size):
        batch = crops[s:s + batch_size]
        try:
            batch_preds = clf_predict_fn(batch)
        except Exception as e:
            logger.error(f"分类推理失败 batch[{s}:{s+batch_size}]: {e}")
            continue
        for j, (lab, sc) in enumerate(batch_preds):
            pred_map[idxs[s + j]] = (lab, float(sc))
    return pred_map


def classify_and_filter_sample(
    sample: fo.Sample,
    pred_field: str,
    out_field: str,
    clf_predict_fn: Callable,
    target_label: str = "swd",
    clf_thresh: float = 0.8,
    pad_ratio: float = 0.2,
    batch_size: int = 64,
) -> None:
    """
    对 sample 的 pred_field 中每个检测框裁 patch，用 clf_predict_fn 分类，
    保留 label==target_label 且 score>=clf_thresh 的框，写入 out_field。
    """
    dets_obj = getattr(sample, pred_field, None)
    if dets_obj is None or len(dets_obj.detections) == 0:
        sample[out_field] = fo.Detections(detections=[])
        return

    try:
        img = Image.open(sample.filepath).convert("RGB")
    except Exception as e:
        logger.error(f"打开图像失败 {sample.filepath}: {e}")
        sample[out_field] = fo.Detections(detections=[])
        return

    with img:
        detections = dets_obj.detections
        crops: List[Image.Image] = []
        idxs: List[int] = []

        for i, det in enumerate(detections):
            crop = crop_with_padding(img, det, pad_ratio=pad_ratio)
            if crop is None:
                det["clf_label"] = None
                det["clf_score"] = None
            else:
                crops.append(crop)
                idxs.append(i)

        pred_map = _batch_classify(crops, idxs, clf_predict_fn, batch_size)

        filtered = []
        for i, det in enumerate(detections):
            if i not in pred_map:
                continue
            lab, sc = pred_map[i]
            det["clf_label"] = lab
            det["clf_score"] = sc
            if lab == target_label and sc >= clf_thresh:
                filtered.append(det)

        sample[out_field] = fo.Detections(detections=filtered)
