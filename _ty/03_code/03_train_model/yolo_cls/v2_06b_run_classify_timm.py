"""
v2_06b_run_classify_timm.py
用 timm 训练的分类器（efficientnet_b0 / mobilenetv3 / resnet50）
对 FiftyOne 数据集中的检测框做二次分类过滤。

流程：
  1. 加载 FiftyOne 数据集
  2. 对每个 sample 的 PRED_FIELD 检测框裁 patch（pad_ratio=0.5）
  3. 用 timm 分类器判断每个 patch 是否为 SWD
  4. 将通过阈值的框写入新字段 OUT_FIELD（同时保留 clf_label / clf_score 属性）

运行后可在 FiftyOne App 里对比 PRED_FIELD（原始）和 OUT_FIELD（过滤后）。
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import fiftyone as fo
import timm
import wandb

# 复用项目内已有的 classify 工具
import sys
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

# 原始检测框字段
PRED_FIELD = "pred_yolo11m_20pct_null_images_add_rawData_batch_16_final"

# timm 模型名称（必须与训练时一致）
# 可选：efficientnet_b0 / mobilenetv3_large_100 / resnet50
# TIMM_MODEL_NAME = "efficientnet_b0"
# TIMM_MODEL_NAME = "mobilenetv3_large_100"
TIMM_MODEL_NAME = "resnet50"



# 模型权重路径（v2_05b_train_timm.py 训练输出的 best.pt）
CLF_MODEL_PATH = Path(
    "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/03_code/03_train_model/yolo_cls/output/swd_cls_v2_deployment_noAug_seed42"
) / f"split_0.8_0.2_{TIMM_MODEL_NAME}_b32/weights/best.pt"

# 数据集类别顺序（与训练时 ImageFolder 的字母序一致）
# ImageFolder 按文件夹名字母排序：non_swd=0, swd=1
CLASSES = ["non_swd", "swd"]

# 分类阈值：只保留分类器认为是 swd 且置信度 >= 此值的框
CLF_THRESH = 0.5

# 二次分类结果写入的字段（原字段不会被修改）
OUT_FIELD  = PRED_FIELD + "_clfs_timm_" + TIMM_MODEL_NAME + f"_thresh{CLF_THRESH*100:.0f}"  # e.g. pred_yolo11m_20pct_null_images_add_rawData_batch_16_final_clfs_timm_resnet50_thresh50

# 其他参数
IMGSZ      = 224   # 与训练时保持一致
PAD_RATIO  = 0.5   # 与训练时保持一致
BATCH_SIZE = 64
DEVICE     = "cuda:0"   # CPU 用 "cpu"

# ========================


def apply_clahe(pil_img: Image.Image, clip_limit: float = 2.0, tile_size: int = 8) -> Image.Image:
    """CLAHE 自适应直方图均衡化：增强局部对比度，让黑点在不同光照下都更突出。"""
    img = np.array(pil_img.convert("RGB"))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)


class CLAHETransform:
    def __init__(self, clip_limit: float = 2.0, tile_size: int = 8):
        self.clip_limit = clip_limit
        self.tile_size  = tile_size

    def __call__(self, img: Image.Image) -> Image.Image:
        return apply_clahe(img, self.clip_limit, self.tile_size)


def build_transform() -> transforms.Compose:
    """与训练时 val transform 保持一致：CLAHE → Resize → ToTensor → Normalize。"""
    return transforms.Compose([
        transforms.Resize((IMGSZ, IMGSZ)),
        CLAHETransform(clip_limit=2.0, tile_size=8),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_timm_model(model_name: str, weights_path: Path, num_classes: int, device: str) -> torch.nn.Module:
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    logger.info(f"加载模型 {model_name} from {weights_path}")
    return model


def make_clf_predict_fn(model: torch.nn.Module, transform: transforms.Compose,
                        classes: list[str], device: str) -> callable:
    """
    包装 timm 模型，返回符合 classify_and_filter_sample 接口的函数。
    输入：PIL Image 列表
    输出：[(label, score), ...] 列表
    """
    def predict(crops: list[Image.Image]) -> list[tuple[str, float]]:
        tensors = torch.stack([transform(c.convert("RGB")) for c in crops]).to(device)
        with torch.no_grad():
            logits = model(tensors)
            probs  = F.softmax(logits, dim=1)
        results = []
        for prob in probs:
            idx   = prob.argmax().item()
            score = prob[idx].item()
            results.append((classes[idx], score))
        return results

    return predict


def main():
    if not CLF_MODEL_PATH.exists():
        logger.error(f"模型文件不存在: {CLF_MODEL_PATH}")
        logger.error("请先运行 v2_05b_train_timm.py 完成训练，或修改 CLF_MODEL_PATH")
        return

    # 初始化 wandb
    wandb.login(key="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")
    run = wandb.init(
        project="swd_cls_v2_deployment_noAug_seed42_timm",
        name=f"inference_{TIMM_MODEL_NAME}_thresh{CLF_THRESH}_{FO_DATASET}",
        config={
            "model":      TIMM_MODEL_NAME,
            "weights":    str(CLF_MODEL_PATH),
            "dataset":    FO_DATASET,
            "pred_field": PRED_FIELD,
            "out_field":  OUT_FIELD,
            "clf_thresh": CLF_THRESH,
            "pad_ratio":  PAD_RATIO,
            "imgsz":      IMGSZ,
            "batch_size": BATCH_SIZE,
        },
    )

    # 加载模型
    model = load_timm_model(TIMM_MODEL_NAME, CLF_MODEL_PATH, len(CLASSES), DEVICE)
    transform = build_transform()
    clf_predict_fn = make_clf_predict_fn(model, transform, CLASSES, DEVICE)

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

    for i, sample in enumerate(view.iter_samples(progress=True, autosave=True)):
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

        # 每 100 张记录一次进度
        if (i + 1) % 100 == 0:
            wandb.log({
                "samples_processed": i + 1,
                "boxes_before":      before_total,
                "boxes_after":       after_total,
                "retention_rate":    after_total / max(1, before_total),
            })

    retention = after_total / max(1, before_total)
    removed   = before_total - after_total

    logger.info(
        f"完成！"
        f"检测框总数: {before_total} → 过滤后: {after_total} "
        f"（移除了 {removed} 个，"
        f"保留率 {retention*100:.1f}%）"
    )
    logger.info(f"结果已写入字段: {OUT_FIELD}")
    logger.info("在 FiftyOne App 中对比：")
    logger.info(f"  原始：      {PRED_FIELD}")
    logger.info(f"  YOLO 过滤： {PRED_FIELD}_clfs")
    logger.info(f"  timm 过滤： {OUT_FIELD}")

    # 最终汇总写入 wandb
    wandb.summary.update({
        "total_samples":   total,
        "boxes_before":    before_total,
        "boxes_after":     after_total,
        "boxes_removed":   removed,
        "retention_rate":  retention,
    })
    wandb.finish()


if __name__ == "__main__":
    main()

