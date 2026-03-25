"""
v2_06_run_classify_on_fo_dataset.py
对 FiftyOne 数据集中的检测框做二次分类过滤。

流程：
  1. 加载 FiftyOne 数据集
  2. 对每个 sample 的 PRED_FIELD 检测框裁 patch
  3. 用训练好的 YOLO 分类器判断每个 patch 是否为 SWD
  4. 将通过阈值的框写入新字段 OUT_FIELD（同时保留 clf_label / clf_score 属性）

运行后可在 FiftyOne App 里对比 PRED_FIELD（原始）和 OUT_FIELD（过滤后）。
"""

from __future__ import annotations

import sys
from pathlib import Path
import logging

import fiftyone as fo
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ===== 依赖：复用项目内已有的 classify 工具 =====
_TOOLS_DIR = Path(
    "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/02_current_batch_run_model/07_no_GT_run"
)
sys.path.insert(0, str(_TOOLS_DIR))
from ty_nogt_tools.classify import classify_and_filter_sample  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================
# 配置（按需修改）
# ========================

# 目标 FiftyOne 数据集
FO_DATASET = "del_test_cls_2025_north_v1__jeff"

# 原始检测框字段（YOLO / SAHI 推理后写入的字段）
PRED_FIELD = "pred_yolo11m_20pct_null_images_add_rawData_batch_16_final"

# 二次分类结果写入的字段（原字段不会被修改）
OUT_FIELD  = PRED_FIELD + "_clfs"

# 分类器模型路径（选择 val accuracy 最高的那个）
CLF_MODEL_PATH = Path(
    # "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/03_code/03_train_model/yolo_cls/output/swd_cls_v2_deployment_noAug_seed42/split_0.8_0.2_yolo11n-cls.pt_b16/weights/best.pt",
    # "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/03_code/03_train_model/yolo_cls/output/swd_cls_v2_deployment_noAug_seed42/split_0.8_0.2_yolo11l-cls.pt_b8/weights/best.pt",
    "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/03_code/03_train_model/yolo_cls/output/swd_cls_v2_deployment_noAug_seed42/split_0.8_0.2_yolo11s-cls.pt_b16/weights/best.pt",
)

# 分类阈值：只保留分类器认为是 swd 且置信度 >= 此值的框
CLF_THRESH = 0.8

# 其他参数
PAD_RATIO  = 0.5   # 与训练时保持一致
BATCH_SIZE = 64   # 每次批量送入分类器的 patch 数
DEVICE     = 0    # GPU id，CPU 用 "cpu"
# ========================


def apply_clahe(pil_img: Image.Image, clip_limit: float = 2.0, tile_size: int = 8) -> Image.Image:
    """CLAHE 自适应直方图均衡化：增强局部对比度，让黑点在不同光照下都更突出。"""
    img = np.array(pil_img.convert("RGB"))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)


def make_clf_predict_fn(model: YOLO, device) -> callable:
    """
    包装 YOLO 分类器，返回符合 classify_and_filter_sample 接口的函数。
    推理前先做 CLAHE 预处理，与训练时保持一致。
    输入：PIL Image 列表
    输出：[(label, score), ...] 列表
    """
    def predict(crops: list[Image.Image]) -> list[tuple[str, float]]:
        crops_clahe = [apply_clahe(c) for c in crops]
        results = model.predict(
            source=crops_clahe,
            device=device,
            verbose=False,
            imgsz=224,   # 与训练时保持一致
        )
        out = []
        for r in results:
            top1_idx  = r.probs.top1
            top1_conf = float(r.probs.top1conf)
            label     = r.names[top1_idx]
            out.append((label, top1_conf))
        return out

    return predict


def main():
    # 加载模型
    logger.info(f"加载分类器: {CLF_MODEL_PATH}")
    clf_model = YOLO(str(CLF_MODEL_PATH))
    clf_predict_fn = make_clf_predict_fn(clf_model, DEVICE)

    # 加载数据集
    logger.info(f"加载数据集: {FO_DATASET}")
    ds = fo.load_dataset(FO_DATASET)

    # 只处理有检测框的 sample
    view = ds.exists(PRED_FIELD)
    total = len(view)
    logger.info(f"有 {PRED_FIELD} 字段的 sample: {total}")

    # 确保输出字段存在
    if not ds.has_sample_field(OUT_FIELD):
        ds.add_sample_field(
            OUT_FIELD,
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )

    # 逐 sample 处理
    before_total = after_total = 0

    for sample in view.iter_samples(progress=True, autosave=True):
        dets_obj = sample.get_field(PRED_FIELD)
        n_before = len(dets_obj.detections) if dets_obj and dets_obj.detections else 0

        classify_and_filter_sample(
            sample=sample,
            pred_field=PRED_FIELD,
            out_field=OUT_FIELD,
            clf_predict_fn=clf_predict_fn,
            target_label="swd",
            clf_thresh=CLF_THRESH,
            pad_ratio=PAD_RATIO,
            batch_size=BATCH_SIZE,
        )

        n_after = len(sample[OUT_FIELD].detections) if sample[OUT_FIELD] else 0
        before_total += n_before
        after_total  += n_after

    logger.info(
        f"完成！"
        f"检测框总数: {before_total} → 过滤后: {after_total} "
        f"（移除了 {before_total - after_total} 个，"
        f"保留率 {after_total/max(1,before_total)*100:.1f}%）"
    )
    logger.info(f"结果已写入字段: {OUT_FIELD}")
    logger.info("在 FiftyOne App 中对比：")
    logger.info(f"  原始：{PRED_FIELD}")
    logger.info(f"  过滤后：{OUT_FIELD}")


if __name__ == "__main__":
    main()
