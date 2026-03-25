"""
v2_03_crop_negatives_from_deployment.py
从 FiftyOne 部署数据集裁取 non-SWD 负样本（false positive 框）。

两种情况：
  - 有 GT：预测框中 IoU(pred, gt) < IOU_FP_THRESH 的框 → FP，裁出作为负样本
  - 无 GT：所有预测框直接作为负样本（因 FP 严重，大多数预测都是错的）

输出：data_v2/raw_crops/non_swd/fp_from_deployment/
"""

from pathlib import Path
from PIL import Image
import fiftyone as fo
from fiftyone import ViewField as F
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ===== 配置 =====
# 部署数据集（可多个）
FO_DATASETS = [
    "sahi_null_v2_ms2_0726-0809_13_ok",   # ← 待补充
]

# 预测框字段名（运行 YOLO/SAHI 推理后写入 FiftyOne 的字段）
PRED_FIELD = "small_slices_a03_yolo11s_custom7null_cv1_ms2_0809-0823_10_ok_16"   # ← 根据实际字段名修改

# GT 字段名（如果没有 GT，留空字符串 ""）
GT_FIELD   = "ground_truth"

# 判定 FP 的 IoU 阈值（pred 与所有 GT box 的最大 IoU < 此值 → FP）
IOU_FP_THRESH = 0.5

OUTPUT_DIR = Path(
    "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/03_code/03_train_model/yolo_cls/data_v2/raw_crops/non_swd/fp_from_deployment"
)
PAD_RATIO  = 0.5   # 扩大到 0.5，与正样本裁图保持一致
# ================


def crop_with_padding(img: Image.Image, det: fo.Detection, pad_ratio: float = 0.2) -> Image.Image | None:
    W, H = img.size
    x, y, w, h = det.bounding_box
    x1, y1 = x * W, y * H
    x2, y2 = (x + w) * W, (y + h) * H

    pw = (x2 - x1) * pad_ratio
    ph = (y2 - y1) * pad_ratio
    x1 = max(0, int(x1 - pw))
    y1 = max(0, int(y1 - ph))
    x2 = min(W, int(x2 + pw))
    y2 = min(H, int(y2 + ph))

    if x2 <= x1 or y2 <= y1:
        return None
    return img.crop((x1, y1, x2, y2))


def calc_iou(a: fo.Detection, b: fo.Detection) -> float:
    """计算两个 fo.Detection 的 IoU（归一化坐标）。"""
    ax, ay, aw, ah = a.bounding_box
    bx, by, bw, bh = b.bounding_box
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / (aw * ah + bw * bh - inter)


def is_fp(pred: fo.Detection, gt_dets: list) -> bool:
    """如果 pred 与所有 GT 框的 IoU 都小于阈值，则认为是 FP。"""
    if not gt_dets:
        return True  # 没有 GT → 所有预测都视为 FP
    return all(calc_iou(pred, gt) < IOU_FP_THRESH for gt in gt_dets)


def process_dataset(ds_name: str, output_dir: Path) -> int:
    logger.info(f"加载数据集: {ds_name}")
    ds = fo.load_dataset(ds_name)

    # 只处理有预测框的 sample
    view = ds.exists(PRED_FIELD)
    logger.info(f"有预测框的 sample: {len(view)}")

    count = 0
    for sample in view.iter_samples(progress=True):
        pred_obj = sample.get_field(PRED_FIELD)
        if pred_obj is None or not pred_obj.detections:
            continue

        # 获取 GT（如有）
        gt_obj = sample.get_field(GT_FIELD) if GT_FIELD else None
        gt_dets = gt_obj.detections if (gt_obj and gt_obj.detections) else []

        try:
            img = Image.open(sample.filepath).convert("RGB")
        except Exception as e:
            logger.warning(f"读取失败 {sample.filepath}: {e}")
            continue

        stem = Path(sample.filepath).stem
        for i, pred in enumerate(pred_obj.detections):
            if not is_fp(pred, gt_dets):
                continue  # 与 GT 匹配的真正 TP，跳过

            patch = crop_with_padding(img, pred, pad_ratio=PAD_RATIO)
            if patch is None:
                continue

            out_name = f"{ds_name[:20]}_{stem}_{i}.jpg"
            patch.save(output_dir / out_name, quality=95)
            count += 1

    logger.info(f"[{ds_name}] 裁出 {count} 个 FP 负样本")
    return count


def main():
    if not FO_DATASETS:
        logger.error("请在脚本顶部 FO_DATASETS 填写数据集名称！\n"
                     "可用数据集：" + str(fo.list_datasets()))
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = sum(process_dataset(name, OUTPUT_DIR) for name in FO_DATASETS)
    logger.info(f"完成！共 {total} 个 FP 负样本 → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
