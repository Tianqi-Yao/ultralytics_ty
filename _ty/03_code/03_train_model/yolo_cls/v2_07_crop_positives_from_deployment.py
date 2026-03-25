"""
v2_07_crop_positives_from_deployment.py
从部署域数据集裁取 SWD 正样本，解决域偏移问题。

背景：
  分类器 val_acc=0.97 但部署推理效果差，原因是训练正样本只来自训练域，
  部署域（不同光照/季节）的 SWD 外观与训练域不同。
  只需加入少量部署域正样本（50~100张）即可大幅改善迁移效果。

使用方法：
  方法A（推荐）：在 FiftyOne App 里手动给真正的 SWD 框打 tag "swd_confirmed"，
                 然后运行本脚本，自动裁出带该 tag 的框。
  方法B：直接用 GT 字段（如果部署数据集有 ground_truth 标注）。

输出：data_v2/raw_crops/swd/deploy_domain/
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image
import fiftyone as fo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================
# 配置
# ========================

# 部署域数据集
FO_DATASET = "del_test_cls_2025_north_v1__jeff"

# 方法A：从某个检测字段里，只裁出打了指定 tag 的框
#   → 在 FiftyOne App 里选中真正的 SWD 框，右键打 tag "swd_confirmed"
PRED_FIELD    = "pred_yolo11m_20pct_null_images_add_rawData_batch_16_final"
CONFIRM_TAG   = "swd_confirmed"   # FiftyOne detection-level tag

# 方法B：直接用 GT 字段（有标注时用这个，留空则走方法A）
GT_FIELD = ""   # 例如 "ground_truth"，没有则留 ""

OUTPUT_DIR = Path(
    "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/03_code/03_train_model/yolo_cls/data_v2/raw_crops/swd/deploy_domain"
)
PAD_RATIO = 0.5
# ========================


def crop_with_padding(img: Image.Image, det: fo.Detection, pad_ratio: float) -> Image.Image | None:
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"加载数据集: {FO_DATASET}")
    ds = fo.load_dataset(FO_DATASET)

    count = 0

    if GT_FIELD:
        # 方法B：直接用 GT 标注
        logger.info(f"方法B：使用 GT 字段 '{GT_FIELD}'")
        view = ds.exists(GT_FIELD)
        logger.info(f"有 GT 的 sample: {len(view)}")

        for sample in view.iter_samples(progress=True):
            gt_obj = sample.get_field(GT_FIELD)
            if not gt_obj or not gt_obj.detections:
                continue

            try:
                img = Image.open(sample.filepath).convert("RGB")
            except Exception as e:
                logger.warning(f"读取失败 {sample.filepath}: {e}")
                continue

            stem = Path(sample.filepath).stem
            for i, det in enumerate(gt_obj.detections):
                patch = crop_with_padding(img, det, PAD_RATIO)
                if patch is None:
                    continue
                out_name = f"deploy_{stem}_{i}.jpg"
                patch.save(OUTPUT_DIR / out_name, quality=95)
                count += 1

    else:
        # 方法A：裁出打了 CONFIRM_TAG 的检测框
        logger.info(f"方法A：裁出 tag='{CONFIRM_TAG}' 的检测框（来自 {PRED_FIELD}）")
        logger.info("请先在 FiftyOne App 里对真正的 SWD 检测框打上该 tag，再运行本脚本。")

        view = ds.exists(PRED_FIELD)

        for sample in view.iter_samples(progress=True):
            pred_obj = sample.get_field(PRED_FIELD)
            if not pred_obj or not pred_obj.detections:
                continue

            confirmed = [
                d for d in pred_obj.detections
                if d.tags and CONFIRM_TAG in d.tags
            ]
            if not confirmed:
                continue

            try:
                img = Image.open(sample.filepath).convert("RGB")
            except Exception as e:
                logger.warning(f"读取失败 {sample.filepath}: {e}")
                continue

            stem = Path(sample.filepath).stem
            for i, det in enumerate(confirmed):
                patch = crop_with_padding(img, det, PAD_RATIO)
                if patch is None:
                    continue
                out_name = f"deploy_confirmed_{stem}_{i}.jpg"
                patch.save(OUTPUT_DIR / out_name, quality=95)
                count += 1

    logger.info(f"完成！裁出 {count} 张部署域正样本 → {OUTPUT_DIR}")
    logger.info("下一步：重新运行 v2_04_split_dataset.py → v2_05_train.py / v2_05b_train_timm.py")


if __name__ == "__main__":
    main()
