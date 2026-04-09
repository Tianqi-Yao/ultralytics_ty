"""
01_create_dataset.py
从图片目录（+ 可选 YOLO txt 标注）创建 FiftyOne 数据集。

若数据集已存在则跳过创建，只打印统计信息。
若 LABELS_DIR 已配置，则解析 YOLO txt 写入 ground_truth 字段。
"""

from __future__ import annotations
import logging
from pathlib import Path
import fiftyone as fo
from pipeline_config import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_yolo_txt(label_path: Path, img_w: int, img_h: int,
                   class_names: list[str]) -> list[fo.Detection]:
    """解析单个 YOLO txt 文件，返回 fo.Detection 列表。"""
    dets = []
    if not label_path.exists():
        return dets
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        dets.append(fo.Detection(label=label, bounding_box=[x1, y1, bw, bh]))
    return dets


def create_dataset() -> fo.Dataset:
    """创建 FiftyOne 数据集（若已存在则直接加载）。"""
    if fo.dataset_exists(FO_DATASET):
        logger.info(f"数据集已存在，直接加载: {FO_DATASET}")
        return fo.load_dataset(FO_DATASET)

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"图片目录不存在: {IMAGES_DIR}")

    logger.info(f"创建数据集: {FO_DATASET}  图片目录: {IMAGES_DIR}")

    # 收集所有图片
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    img_paths = sorted(p for p in IMAGES_DIR.rglob("*") if p.suffix.lower() in exts)
    if not img_paths:
        raise FileNotFoundError(f"目录下没有图片: {IMAGES_DIR}")

    logger.info(f"找到 {len(img_paths)} 张图片")

    ds = fo.Dataset(name=FO_DATASET, persistent=True)
    samples = [fo.Sample(filepath=str(p)) for p in img_paths]
    ds.add_samples(samples)
    logger.info(f"数据集创建完成，共 {len(ds)} 个 sample")

    return ds


def import_ground_truth(ds: fo.Dataset):
    """将 YOLO txt 标注写入 ground_truth 字段。"""
    if LABELS_DIR is None:
        logger.info("LABELS_DIR 未配置，跳过 GT 导入。")
        return

    if not LABELS_DIR.exists():
        logger.warning(f"LABELS_DIR 不存在，跳过 GT 导入: {LABELS_DIR}")
        return

    logger.info(f"导入 GT 标注: {LABELS_DIR}")

    gt_count = img_with_gt = 0

    for sample in ds.iter_samples(progress=True, autosave=True):
        img_path   = Path(sample.filepath)
        label_path = LABELS_DIR / (img_path.stem + ".txt")

        # 获取图片尺寸（FiftyOne 会自动填充 metadata）
        if sample.metadata is None:
            sample.compute_metadata()
        w = sample.metadata.width  if sample.metadata else 1
        h = sample.metadata.height if sample.metadata else 1

        dets = parse_yolo_txt(label_path, w, h, CLASS_NAMES)
        sample[GT_FIELD] = fo.Detections(detections=dets)

        if dets:
            img_with_gt += 1
            gt_count    += len(dets)

    logger.info(f"GT 导入完成：有标注图片 {img_with_gt}/{len(ds)}，GT 框总数 {gt_count}")


def print_stats(ds: fo.Dataset):
    total = len(ds)
    has_gt = len(ds.exists(GT_FIELD)) if ds.has_sample_field(GT_FIELD) else 0
    has_pred = {
        field: len(ds.exists(field))
        for field in ds.get_field_schema()
        if field.startswith("pred_r") or field.startswith("clf_r")
    }

    logger.info("=" * 50)
    logger.info(f"数据集：{FO_DATASET}")
    logger.info(f"  总图片数：{total}")
    logger.info(f"  有 GT 的图片：{has_gt}")
    for field, cnt in has_pred.items():
        logger.info(f"  {field}：{cnt} 张有框")
    logger.info("=" * 50)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ds = create_dataset()
    import_ground_truth(ds)
    print_stats(ds)

    logger.info(f"完成！可在 FiftyOne App 中查看: fo.launch_app(fo.load_dataset('{FO_DATASET}'))")


if __name__ == "__main__":
    main()
