"""
v2_05_train.py
SWD 分类器 v2 训练脚本。

数据集：data_v2/split_0.8_0.2（包含部署域 false positive 负样本）
模型：yolo11n-cls / yolo11s-cls
无数据增强，seed=42
"""

from pathlib import Path
from ultralytics import YOLO
import wandb
wandb.login(key="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")

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================
# 配置
# ========================

PROJECT_NAME = "swd_cls_v2_deployment"

DATASETS = [
    "/workspace/_ty/03_code/03_train_model/yolo_cls/data_v2/split_0.8_0.2",
]

MODELS = [
    "yolo11n-cls.pt",   # 优先：最轻量，部署推理快
    "yolo11s-cls.pt",   # 备选：稍大，精度更高
    "yolo11m-cls.pt",   # 备选：更大，精度更高，但训练和推理慢
    "yolo11l-cls.pt",   # 过大，训练和推理都慢，不建议
]

BATCH_SIZES = [16, 8]

IMGSZ = 224   # 分类 patch 尺寸（提升到 224 以保留翅膀黑点细节）
# ========================


def dataset_name(path: str) -> str:
    return Path(path).name


def main():
    for data_path in DATASETS:
        ds_name = dataset_name(data_path)

        for model_file in MODELS:
            for batch in BATCH_SIZES:
                logger.info(f"开始训练: model={model_file}, dataset={ds_name}, batch={batch}")

                model = YOLO(model_file)

                try:
                    model.train(
                        task="classify",
                        data=data_path,
                        epochs=1000,
                        imgsz=IMGSZ,
                        batch=batch,
                        device=0,
                        workers=4,
                        project=f"output/{PROJECT_NAME}_noAug_seed42",
                        name=f"{ds_name}_{model_file}_b{batch}",

                        # 关闭所有数据增强（部署推理时无增强，训练保持一致）
                        degrees=0.0,
                        scale=0.0,
                        shear=0.0,
                        translate=0.0,
                        mixup=0.0,
                        cutmix=0.0,
                        erasing=0.0,
                        flipud=0.0,
                        fliplr=0.5,   # 保留左右翻转（SWD 黑点左右对称）

                        seed=42,
                        deterministic=True,
                    )

                    logger.info(f"训练完成，开始 val 评估: model={model_file}, batch={batch}")
                    # model.val(
                    #     task="classify",
                    #     data=data_path,
                    #     split="test",
                    #     name=f"{ds_name}_{model_file}_b{batch}_val",
                    # )

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logger.warning(f"显存不足，跳过: model={model_file}, batch={batch}")
                        continue
                    logger.error(f"训练失败: {e}")
                    raise


if __name__ == "__main__":
    main()
