"""
v2_02_crop_negatives_from_train.py
从 FiftyOne 训练数据集裁取 non-SWD 负样本 patch：
  B1：GT 为空的 sample（null 图）→ 随机裁取
  B2：有 GT 的 sample → 在 bbox 以外区域随机裁取

输出：data_v2/raw_crops/non_swd/bg_from_train/
"""

from pathlib import Path
from PIL import Image
import fiftyone as fo
import random
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ===== 配置 =====
FO_DATASETS = [
    "sahi_null_v2_ms2_0726-0809_13_ok",   # ← 填写实际数据集名
]

GT_FIELD = "ground_truth"

OUTPUT_DIR = Path(
    "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/03_code/03_train_model/yolo_cls/data_v2/raw_crops/non_swd/bg_from_train"
)

NULL_CROPS_PER_IMAGE = 2    # B1：每张 null 图裁几个 patch
ANNO_CROPS_PER_IMAGE = 1    # B2：每张有标注图裁几个背景 patch
CROP_SIZE_RATIO      = 0.08 # patch 大小约为图宽的 8%（参考 SWD bbox 平均尺寸）
PAD_RATIO            = 0.5   # 扩大到 0.5，与正样本裁图保持一致
IOU_THRESH           = 0.1  # 与 GT bbox 的 IoU 超过此值则重试
MAX_RETRY            = 50
SEED                 = 42
# ================

random.seed(SEED)


def calc_iou(a, b):
    """a, b: [x1, y1, x2, y2] 像素坐标"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def fo_det_to_pixel(det: fo.Detection, W: int, H: int) -> list:
    x, y, w, h = det.bounding_box
    return [int(x * W), int(y * H), int((x + w) * W), int((y + h) * H)]


def random_crop(img: Image.Image, crop_w: int, crop_h: int,
                gt_boxes: list) -> Image.Image | None:
    W, H = img.size
    if crop_w >= W or crop_h >= H:
        return None
    for _ in range(MAX_RETRY):
        x1 = random.randint(0, W - crop_w)
        y1 = random.randint(0, H - crop_h)
        candidate = [x1, y1, x1 + crop_w, y1 + crop_h]
        if all(calc_iou(candidate, gt) < IOU_THRESH for gt in gt_boxes):
            return img.crop(candidate)
    return None


def process_dataset(ds_name: str, output_dir: Path):
    logger.info(f"加载数据集: {ds_name}")
    ds = fo.load_dataset(ds_name)
    b1 = b2 = 0

    for sample in ds.iter_samples(progress=True):
        dets_obj = sample.get_field(GT_FIELD)
        gt_dets = dets_obj.detections if (dets_obj and dets_obj.detections) else []

        try:
            img = Image.open(sample.filepath).convert("RGB")
        except Exception as e:
            logger.warning(f"读取失败 {sample.filepath}: {e}")
            continue

        W, H = img.size
        crop_w = max(16, int(W * CROP_SIZE_RATIO * (1 + PAD_RATIO)))
        crop_h = max(16, int(H * CROP_SIZE_RATIO * (1 + PAD_RATIO)))
        gt_boxes = [fo_det_to_pixel(d, W, H) for d in gt_dets]

        n_crops = NULL_CROPS_PER_IMAGE if not gt_dets else ANNO_CROPS_PER_IMAGE
        tag = "null" if not gt_dets else "anno"
        stem = Path(sample.filepath).stem

        for i in range(n_crops):
            patch = random_crop(img, crop_w, crop_h, gt_boxes)
            if patch is None:
                continue
            out_name = f"{tag}_{ds_name[:20]}_{stem}_{i}.jpg"
            patch.save(output_dir / out_name, quality=95)
            if not gt_dets:
                b1 += 1
            else:
                b2 += 1

    logger.info(f"[{ds_name}] B1(null)={b1}  B2(anno_bg)={b2}")
    return b1, b2


def main():
    if not FO_DATASETS:
        logger.error("请在脚本顶部 FO_DATASETS 填写数据集名称！\n"
                     "可用数据集：" + str(fo.list_datasets()))
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_b1 = total_b2 = 0
    for name in FO_DATASETS:
        b1, b2 = process_dataset(name, OUTPUT_DIR)
        total_b1 += b1
        total_b2 += b2
    logger.info(f"完成！B1={total_b1}  B2={total_b2}  合计={total_b1+total_b2} → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
