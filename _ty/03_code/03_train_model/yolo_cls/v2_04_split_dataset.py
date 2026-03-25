"""
v2_04_split_dataset.py
将 raw_crops/swd/ 和 raw_crops/non_swd/ 划分为 train/val，
生成 ImageNet-style 目录结构供 YOLO 分类模型训练。

输出结构：
  data_v2/split_0.8_0.2/
    train/swd/
    train/non_swd/
    val/swd/
    val/non_swd/

注意：
  - 类别均衡策略：non_swd 样本数 > swd 样本数时，随机下采样到 MAX_NEG_RATIO 倍
  - 随机 seed=42 保证可复现
"""

from pathlib import Path
import random
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ===== 配置 =====
RAW_CROPS_DIR = Path("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/03_code/03_train_model/yolo_cls/data_v2/raw_crops")
OUTPUT_DIR    = Path("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/03_code/03_train_model/yolo_cls/data_v2/split_0.8_0.2")

VAL_RATIO     = 0.2    # 20% 作为 val
MAX_NEG_RATIO = 3.0    # 负样本最多是正样本的 N 倍（防止严重不均衡）
SEED          = 42
# ================

random.seed(SEED)

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def collect_images(root: Path) -> list[Path]:
    """递归收集目录下所有图片文件。"""
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]


def split_and_copy(files: list[Path], class_name: str, output_dir: Path, val_ratio: float):
    random.shuffle(files)
    n_val = max(1, int(len(files) * val_ratio))
    val_files  = files[:n_val]
    train_files = files[n_val:]

    for split, split_files in [("train", train_files), ("val", val_files)]:
        dst_dir = output_dir / split / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in split_files:
            shutil.copy2(src, dst_dir / src.name)

    logger.info(f"  {class_name}: total={len(files)}  train={len(train_files)}  val={len(val_files)}")
    return len(train_files), len(val_files)


def main():
    swd_dir     = RAW_CROPS_DIR / "swd"
    non_swd_dir = RAW_CROPS_DIR / "non_swd"

    if not swd_dir.exists():
        logger.error(f"找不到正样本目录: {swd_dir}")
        return
    if not non_swd_dir.exists():
        logger.error(f"找不到负样本目录: {non_swd_dir}")
        return

    swd_files     = collect_images(swd_dir)
    non_swd_files = collect_images(non_swd_dir)

    logger.info(f"正样本（swd）：{len(swd_files)} 张")
    logger.info(f"负样本（non_swd）：{len(non_swd_files)} 张（下采样前）")

    # 类别均衡：负样本最多为正样本的 MAX_NEG_RATIO 倍
    max_neg = int(len(swd_files) * MAX_NEG_RATIO)
    if len(non_swd_files) > max_neg:
        random.shuffle(non_swd_files)
        non_swd_files = non_swd_files[:max_neg]
        logger.info(f"负样本下采样至 {len(non_swd_files)} 张（{MAX_NEG_RATIO}× 正样本）")

    logger.info(f"最终：swd={len(swd_files)}  non_swd={len(non_swd_files)}  "
                f"比例={len(non_swd_files)/max(1,len(swd_files)):.2f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("划分 swd ...")
    split_and_copy(swd_files, "swd", OUTPUT_DIR, VAL_RATIO)

    logger.info("划分 non_swd ...")
    split_and_copy(non_swd_files, "non_swd", OUTPUT_DIR, VAL_RATIO)

    # 打印最终统计
    for split in ["train", "val"]:
        for cls in ["swd", "non_swd"]:
            d = OUTPUT_DIR / split / cls
            n = len(list(d.glob("*"))) if d.exists() else 0
            logger.info(f"  {split}/{cls}: {n} 张")

    logger.info(f"完成！数据集输出至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
