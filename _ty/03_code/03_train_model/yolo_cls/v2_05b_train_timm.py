"""
v2_05b_train_timm.py
SWD 分类器 v2 — 非 YOLO 主流分类模型训练脚本。

模型：
  - efficientnet_b0   精度/参数比最优
  - mobilenetv3_large_100  轻量部署友好
  - resnet50          经典强 baseline

数据集：data_v2/split_0.8_0.2（ImageNet-style 目录结构）
输入尺寸：224×224（与 v2_05_train.py 的 YOLO-cls 一致）
无数据增强（仅保留 fliplr），seed=42，early stopping patience=50
"""

from __future__ import annotations

import os
import random
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import wandb

# ========================
# 配置
# ========================

PROJECT_NAME = "swd_cls_v3_deployment"

DATA_DIR = Path(
    "/workspace/_ty/03_code/03_train_model/yolo_cls/data_v2/split_0.8_0.2"
)

OUTPUT_ROOT = Path(
    "/workspace/_ty/03_code/03_train_model/yolo_cls/output/swd_cls_v2_deployment_noAug_seed42"
)

MODELS = [
    "efficientnet_b0",       # 5.3M params，精度/参数比最优
    "mobilenetv3_large_100", # 5.5M params，轻量部署友好
    "resnet50",              # 25.6M params，经典强 baseline
]

IMGSZ      = 224
BATCH_SIZE = 32
EPOCHS     = 200
LR         = 1e-4        # fine-tune 用小学习率
PATIENCE   = 50          # early stopping patience（val acc 不再提升的轮数）
SEED       = 42
DEVICE     = "cuda:0"
NUM_WORKERS = 4

# ========================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_clahe(pil_img: Image.Image, clip_limit: float = 2.0, tile_size: int = 8) -> Image.Image:
    """CLAHE 自适应直方图均衡化：增强局部对比度，让黑点在不同光照下都更突出。"""
    img = np.array(pil_img.convert("RGB"))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])   # 只对亮度通道均衡化
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)


class CLAHETransform:
    """torchvision-compatible CLAHE wrapper。"""
    def __init__(self, clip_limit: float = 2.0, tile_size: int = 8):
        self.clip_limit = clip_limit
        self.tile_size  = tile_size

    def __call__(self, img: Image.Image) -> Image.Image:
        return apply_clahe(img, self.clip_limit, self.tile_size)


def get_transforms(split: str) -> transforms.Compose:
    """
    训练：CLAHE → ColorJitter → fliplr → ToTensor → Normalize
    推理：CLAHE → ToTensor → Normalize
    CLAHE 增强局部对比度（黑点更突出），ColorJitter 模拟光照域偏移。
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMGSZ, IMGSZ)),
            CLAHETransform(clip_limit=2.0, tile_size=8),
            transforms.ColorJitter(
                brightness=0.4,   # 模拟不同曝光
                contrast=0.4,     # 模拟不同光照
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.1),   # 强迫学形状而非颜色
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMGSZ, IMGSZ)),
            CLAHETransform(clip_limit=2.0, tile_size=8),
            transforms.ToTensor(),
            normalize,
        ])


def build_dataloaders():
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=get_transforms("train"))
    val_ds   = datasets.ImageFolder(DATA_DIR / "val",   transform=get_transforms("val"))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    logger.info(f"数据集类别: {train_ds.classes}")
    logger.info(f"train: {len(train_ds)}  val: {len(val_ds)}")
    return train_loader, val_loader, train_ds.classes


def train_one_model(model_name: str, train_loader, val_loader, classes):
    run_name = f"{DATA_DIR.name}_{model_name}_b{BATCH_SIZE}_clahe_aug"
    save_dir = OUTPUT_ROOT / run_name / "weights"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"===== 开始训练: {model_name} =====")

    # 初始化 wandb run
    wandb.init(
        project=f"{PROJECT_NAME}_clahe_aug_seed42_timm",
        name=run_name,
        config={
            "model": model_name,
            "imgsz": IMGSZ,
            "batch": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "patience": PATIENCE,
            "seed": SEED,
        },
        reinit=True,
    )

    # 构建模型（ImageNet 预训练，替换最后一层）
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=len(classes),
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    no_improve   = 0

    for epoch in range(1, EPOCHS + 1):
        # ---------- train ----------
        model.train()
        train_loss = train_correct = train_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * imgs.size(0)
            preds          = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += imgs.size(0)

        scheduler.step()

        train_loss /= train_total
        train_acc   = train_correct / train_total

        # ---------- val ----------
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                loss   = criterion(logits, labels)

                val_loss    += loss.item() * imgs.size(0)
                preds        = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total

        wandb.log({
            "train/loss": train_loss,
            "train/acc":  train_acc,
            "val/loss":   val_loss,
            "val/acc":    val_acc,
            "lr":         scheduler.get_last_lr()[0],
            "epoch":      epoch,
        })

        if epoch % 10 == 0 or val_acc > best_val_acc:
            logger.info(
                f"Epoch {epoch:4d}/{EPOCHS} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        # ---------- checkpoint ----------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save(model.state_dict(), save_dir / "best.pt")
            logger.info(f"  ✓ 保存 best.pt  val_acc={val_acc:.4f}")
        else:
            no_improve += 1

        # ---------- early stopping ----------
        if no_improve >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}（{PATIENCE} 轮无提升）")
            break

    # 保存最后一个 epoch
    torch.save(model.state_dict(), save_dir / "last.pt")
    logger.info(f"训练完成: {model_name}  best_val_acc={best_val_acc:.4f}")
    logger.info(f"模型保存至: {save_dir}")

    wandb.summary["best_val_acc"] = best_val_acc
    wandb.finish()

    return best_val_acc


def main():
    set_seed(SEED)

    wandb.login(key="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")

    train_loader, val_loader, classes = build_dataloaders()

    results = {}
    for model_name in MODELS:
        best_acc = train_one_model(model_name, train_loader, val_loader, classes)
        results[model_name] = best_acc

    logger.info("===== 训练结果汇总 =====")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        logger.info(f"  {name:30s}  best_val_acc={acc:.4f}")


if __name__ == "__main__":
    main()
