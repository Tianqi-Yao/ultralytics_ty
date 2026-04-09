"""
v2_09_train_preprocess_search.py
对 14 种图像预处理方法分别训练分类器，在 test set 上评估，
输出最优预处理方法排名。

设计原则：
  - 固定一个模型（efficientnet_b0）排除模型差异
  - 关闭所有模型自带增强（包括 fliplr），只让预处理本身产生差异
  - 每种方法训练完成后立即在 test set 评估，记录 accuracy / per-class recall
  - 最终输出 CSV 排名表 + 控制台总结

用法：
  python v2_09_train_preprocess_search.py

⚠️ 修改 TEST_DIR 为实际 test 文件夹路径（ImageNet-style，含 swd/ non_swd/ 子目录）。
"""

from __future__ import annotations

import csv
import logging
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import wandb

# ========================
# 配置（按需修改）
# ========================

DATA_DIR = Path("/workspace/_ty/03_code/03_train_model/yolo_cls/data_v2/split_0.8_0.2")

# ⚠️ 修改为实际 test 文件夹路径（同样需要 swd/ non_swd/ 子目录）
TEST_DIR = Path("/workspace/_ty/03_code/03_train_model/yolo_cls/data_v2/test")

OUTPUT_ROOT  = Path("/workspace/_ty/03_code/03_train_model/yolo_cls/output/preprocess_search")
RESULTS_CSV  = OUTPUT_ROOT / "results.csv"

PROJECT_NAME = "swd_cls_preprocess_search"
MODEL_NAME   = "resnet50"   # 固定模型，排除模型差异

IMGSZ        = 224
BATCH_SIZE   = 32
EPOCHS       = 150
PATIENCE     = 40    # early stopping
LR           = 1e-4
SEED         = 42
DEVICE       = "cuda:0"
NUM_WORKERS  = 4

WANDB_KEY = "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"
# ========================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ==================== 14 种预处理方法（纯 numpy/cv2，作用于 BGR ndarray） ====================

def pp_original(img: np.ndarray) -> np.ndarray:
    return img.copy()

def pp_clahe(img: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[:, :, 0] = c.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def pp_clahe_strong(img: np.ndarray) -> np.ndarray:
    return pp_clahe(img, clip=4.0)

def pp_clahe_small_tile(img: np.ndarray) -> np.ndarray:
    return pp_clahe(img, clip=3.0, tile=4)

def pp_unsharp(img: np.ndarray, radius: int = 3, amount: float = 2.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    out = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)

def pp_gamma_dark(img: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)

def pp_gamma_very_dark(img: np.ndarray) -> np.ndarray:
    return pp_gamma_dark(img, gamma=0.3)

def pp_clahe_unsharp(img: np.ndarray) -> np.ndarray:
    return pp_unsharp(pp_clahe(img, clip=2.0))

def pp_grayscale(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def pp_local_norm(img: np.ndarray, kernel: int = 15) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = kernel if kernel % 2 == 1 else kernel + 1
    local_mean = cv2.blur(gray, (k, k))
    diff = gray - local_mean
    local_sq_mean = cv2.blur(gray ** 2, (k, k))
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
    local_std = np.sqrt(local_var) + 1e-6
    norm = diff / local_std
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-6) * 255
    return cv2.cvtColor(norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def pp_tophat(img: np.ndarray, ksize: int = 7) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    enhanced = np.clip(gray.astype(np.int32) - blackhat.astype(np.int32) * 2, 0, 255).astype(np.uint8)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def pp_blackhat_viz(img: np.ndarray, ksize: int = 7) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    return cv2.cvtColor(cv2.equalizeHist(blackhat), cv2.COLOR_GRAY2BGR)

def pp_bilateral(img: np.ndarray) -> np.ndarray:
    smooth = cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)
    return pp_unsharp(smooth, radius=2, amount=1.5)

def pp_clahe_gamma(img: np.ndarray) -> np.ndarray:
    return pp_gamma_dark(pp_clahe(img, clip=2.0), gamma=0.6)


PREPROCESS_METHODS: list[tuple[str, callable]] = [
    ("01_Original",         pp_original),
    ("02_CLAHE",            pp_clahe),
    ("03_CLAHE_Strong",     pp_clahe_strong),
    ("04_CLAHE_SmallTile",  pp_clahe_small_tile),
    ("05_Unsharp_Mask",     pp_unsharp),
    ("06_Gamma_Dark",       pp_gamma_dark),
    ("07_Gamma_VeryDark",   pp_gamma_very_dark),
    ("08_CLAHE_Unsharp",    pp_clahe_unsharp),
    ("09_Grayscale",        pp_grayscale),
    ("10_Local_Norm",       pp_local_norm),
    ("11_TopHat",           pp_tophat),
    ("12_BlackHat_Viz",     pp_blackhat_viz),
    ("13_Bilateral_Sharp",  pp_bilateral),
    ("14_CLAHE_Gamma",      pp_clahe_gamma),
]


# ==================== Dataset ====================

class PreprocessDataset(torch.utils.data.Dataset):
    """在 ImageFolder 基础上，先做指定预处理，再走标准 normalize 流程。"""

    def __init__(self, root: Path, preprocess_fn: callable, augment: bool = False):
        self.base = datasets.ImageFolder(str(root))
        self.preprocess_fn = preprocess_fn
        self.augment = augment   # 训练时可选 fliplr，val/test 关闭
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @property
    def classes(self):
        return self.base.classes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        path, label = self.base.samples[idx]

        # 读取 → BGR ndarray
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            img_bgr = np.zeros((64, 64, 3), dtype=np.uint8)

        # 预处理
        img_bgr = self.preprocess_fn(img_bgr)

        # Resize
        img_bgr = cv2.resize(img_bgr, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)

        # BGR → RGB PIL
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # 唯一保留的增强：左右翻转（不影响黑点位置，仅数据多样性）
        # 推理时关闭
        if self.augment and random.random() < 0.5:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

        return self.normalize(pil_img), label


# ==================== 训练工具 ====================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loaders(preprocess_fn: callable):
    train_ds = PreprocessDataset(DATA_DIR / "train", preprocess_fn, augment=False)
    val_ds   = PreprocessDataset(DATA_DIR / "val",   preprocess_fn, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, train_ds.classes


def build_test_loader(preprocess_fn: callable, classes: list[str]):
    test_ds = PreprocessDataset(TEST_DIR, preprocess_fn, augment=False)
    # 验证类别顺序一致
    if test_ds.classes != classes:
        logger.warning(f"test 类别顺序与 train 不同: {test_ds.classes} vs {classes}")
    return DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, classes: list[str]) -> dict:
    model.eval()
    correct = total = 0
    per_class_correct = [0] * len(classes)
    per_class_total   = [0] * len(classes)

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        for c in range(len(classes)):
            mask = labels == c
            per_class_correct[c] += (preds[mask] == labels[mask]).sum().item()
            per_class_total[c]   += mask.sum().item()

    acc = correct / max(1, total)
    recall = {classes[c]: per_class_correct[c] / max(1, per_class_total[c])
              for c in range(len(classes))}
    return {"acc": acc, "recall": recall, "correct": correct, "total": total}


def train_one(method_name: str, preprocess_fn: callable) -> dict:
    set_seed(SEED)

    run_name = f"{method_name}"
    save_dir = OUTPUT_ROOT / run_name / "weights"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"===== [{method_name}] 开始训练 =====")

    wandb.init(
        project=PROJECT_NAME,
        name=run_name,
        config={
            "model":      MODEL_NAME,
            "preprocess": method_name,
            "imgsz":      IMGSZ,
            "batch":      BATCH_SIZE,
            "epochs":     EPOCHS,
            "lr":         LR,
            "patience":   PATIENCE,
            "augment":    False,
        },
        reinit=True,
    )

    train_loader, val_loader, classes = build_loaders(preprocess_fn)

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    no_improve   = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        train_correct = train_total = 0
        train_loss_sum = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * imgs.size(0)
            train_correct  += (logits.argmax(1) == labels).sum().item()
            train_total    += imgs.size(0)
        scheduler.step()

        train_acc  = train_correct / train_total
        train_loss = train_loss_sum / train_total

        # val
        val_metrics = evaluate(model, val_loader, classes)
        val_acc = val_metrics["acc"]

        wandb.log({
            "train/loss": train_loss,
            "train/acc":  train_acc,
            "val/acc":    val_acc,
            "epoch":      epoch,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save(model.state_dict(), save_dir / "best.pt")
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            logger.info(f"  Early stop @ epoch {epoch}，best_val_acc={best_val_acc:.4f}")
            break

    # ---- 加载 best.pt 在 test set 评估 ----
    model.load_state_dict(torch.load(str(save_dir / "best.pt"), map_location=DEVICE))

    if TEST_DIR.exists():
        test_loader  = build_test_loader(preprocess_fn, classes)
        test_metrics = evaluate(model, test_loader, classes)
        test_acc     = test_metrics["acc"]
        test_recall  = test_metrics["recall"]
        logger.info(f"  test_acc={test_acc:.4f}  recall={test_recall}")
    else:
        logger.warning(f"  TEST_DIR 不存在，跳过 test 评估: {TEST_DIR}")
        test_acc    = float("nan")
        test_recall = {}

    elapsed = time.time() - t0
    logger.info(f"  完成，耗时 {elapsed/60:.1f} min")

    wandb.summary.update({
        "best_val_acc": best_val_acc,
        "test_acc":     test_acc,
        **{f"test_recall_{k}": v for k, v in test_recall.items()},
    })
    wandb.finish()

    return {
        "method":       method_name,
        "best_val_acc": best_val_acc,
        "test_acc":     test_acc,
        **{f"test_recall_{k}": v for k, v in test_recall.items()},
        "elapsed_min":  round(elapsed / 60, 1),
    }


# ==================== 主流程 ====================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    wandb.login(key=WANDB_KEY)

    if not TEST_DIR.exists():
        logger.warning(f"⚠️  TEST_DIR 不存在: {TEST_DIR}")
        logger.warning("   训练会正常进行，但 test 评估会跳过。")
        logger.warning("   请在脚本顶部修改 TEST_DIR 为正确路径。")

    logger.info(f"模型: {MODEL_NAME}，预处理方法数: {len(PREPROCESS_METHODS)}")
    logger.info(f"训练集: {DATA_DIR}，测试集: {TEST_DIR}")

    all_results: list[dict] = []

    for method_name, preprocess_fn in PREPROCESS_METHODS:
        result = train_one(method_name, preprocess_fn)
        all_results.append(result)

        # 每跑完一个就写一次 CSV（防止中途崩溃丢失结果）
        _write_csv(all_results)
        logger.info(f"  → 已写入 {RESULTS_CSV}")

    # ---- 最终排名 ----
    valid = [r for r in all_results if not (isinstance(r["test_acc"], float)
                                             and r["test_acc"] != r["test_acc"])]
    if valid:
        ranked = sorted(valid, key=lambda x: x["test_acc"], reverse=True)
        print("\n" + "=" * 60)
        print(f"{'排名':>4}  {'方法':<25}  {'test_acc':>10}  {'val_acc':>10}")
        print("-" * 60)
        for i, r in enumerate(ranked, 1):
            marker = " ← 最优" if i == 1 else ""
            print(f"{i:>4}  {r['method']:<25}  {r['test_acc']:>10.4f}  {r['best_val_acc']:>10.4f}{marker}")
        print("=" * 60)
        print(f"\n🏆 最优预处理方法: {ranked[0]['method']}  test_acc={ranked[0]['test_acc']:.4f}")
        print(f"完整结果: {RESULTS_CSV}")
    else:
        print("所有方法均未完成 test 评估，请检查 TEST_DIR 路径。")


def _write_csv(results: list[dict]):
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
