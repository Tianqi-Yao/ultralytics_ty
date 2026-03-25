"""
v2_01_crop_positives.py
从 FiftyOne 数据集裁取 SWD 正样本 patch（GT detections）。

用法：在配置区填写 FO_DATASETS，直接运行。
"""

from pathlib import Path
from PIL import Image
import fiftyone as fo
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ===== 配置 =====
# 包含 GT 标注的 FiftyOne 数据集名称列表（可多个）
FO_DATASETS = [
    "sahi_null_v2_ms2_0726-0809_13_ok",   # ← 填写实际数据集名
]

GT_FIELD   = "ground_truth"   # GT 检测框字段名
GT_LABEL   = "swd"            # 正样本类别名

OUTPUT_DIR = Path("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/03_code/03_train_model/yolo_cls/data_v2/raw_crops/swd")
PAD_RATIO  = 0.5   # 扩大到 0.5，保留翅膀上下文以学习黑点特征
# ================


def crop_with_padding(img: Image.Image, det: fo.Detection, pad_ratio: float = 0.2) -> Image.Image | None:
    """裁出 FiftyOne Detection 对应的 patch（带 padding）。"""
    W, H = img.size
    x, y, w, h = det.bounding_box          # COCO 格式，归一化
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
    return img.crop((x1, y1, x2, y2))


def process_dataset(ds_name: str, output_dir: Path) -> int:
    logger.info(f"加载数据集: {ds_name}")
    ds = fo.load_dataset(ds_name)

    # 只处理有 GT 的 sample
    view = ds.exists(GT_FIELD)
    logger.info(f"有 GT 的 sample: {len(view)}")

    count = 0
    for sample in view.iter_samples(progress=True):
        dets_obj = sample[GT_FIELD]
        if dets_obj is None or not dets_obj.detections:
            continue

        try:
            img = Image.open(sample.filepath).convert("RGB")
        except Exception as e:
            logger.warning(f"读取图片失败 {sample.filepath}: {e}")
            continue

        for i, det in enumerate(dets_obj.detections):
            if det.label != GT_LABEL:
                continue

            patch = crop_with_padding(img, det, pad_ratio=PAD_RATIO)
            if patch is None:
                continue

            stem = Path(sample.filepath).stem
            out_name = f"{ds_name[:20]}_{stem}_{i}.jpg"
            patch.save(output_dir / out_name, quality=95)
            count += 1

    logger.info(f"[{ds_name}] 裁出 {count} 个正样本")
    return count


def main():
    if not FO_DATASETS:
        logger.error("请在脚本顶部 FO_DATASETS 填写数据集名称！\n"
                     "可用数据集：" + str(fo.list_datasets()))
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = sum(process_dataset(name, OUTPUT_DIR) for name in FO_DATASETS)
    logger.info(f"完成！共 {total} 个正样本 → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
