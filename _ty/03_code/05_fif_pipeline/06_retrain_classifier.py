"""
06_retrain_classifier.py
用新增 crops 重新训练分类器。

快速模式（FULL_SEARCH=False）：只用上轮 ensemble_results.csv 最优预处理方法训练单个模型
完整模式（FULL_SEARCH=True）：重跑全部 14 种预处理搜索（更准确但耗时更长）

训练完自动更新 ensemble_results.csv。
"""

from __future__ import annotations
import csv
import logging
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import cv2
import numpy as np
from PIL import Image
import timm
import wandb

from pipeline_config import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ==================== 数据集划分（复用 v2_04 逻辑） ====================

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def collect_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]

def split_and_copy(files: list[Path], cls_name: str, out_dir: Path,
                   val_ratio: float = 0.2):
    random.shuffle(files)
    n_val = max(1, int(len(files) * val_ratio))
    val_files   = files[:n_val]
    train_files = files[n_val:]
    for split, split_files in [("train", train_files), ("val", val_files)]:
        dst = out_dir / split / cls_name
        dst.mkdir(parents=True, exist_ok=True)
        for src in split_files:
            shutil.copy2(src, dst / src.name)
    logger.info(f"  {cls_name}: total={len(files)}  train={len(train_files)}  val={len(val_files)}")

def rebuild_split_dataset():
    """重新划分 raw_crops → split_0.8_0.2（清空后重建）。"""
    logger.info("重建数据集划分...")
    random.seed(SEED)

    swd_files     = collect_images(CROPS_DIR / "swd")
    non_swd_files = collect_images(CROPS_DIR / "non_swd")

    # 负样本下采样（最多3倍正样本）
    max_neg = int(len(swd_files) * 3.0)
    if len(non_swd_files) > max_neg:
        random.shuffle(non_swd_files)
        non_swd_files = non_swd_files[:max_neg]

    logger.info(f"swd={len(swd_files)}  non_swd={len(non_swd_files)}")

    # 清空并重建
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)
    SPLIT_DIR.mkdir(parents=True)

    split_and_copy(swd_files,     "swd",     SPLIT_DIR)
    split_and_copy(non_swd_files, "non_swd", SPLIT_DIR)
    logger.info(f"数据集划分完成: {SPLIT_DIR}")


# ==================== 预处理方法（同 v2_09） ====================

def pp_original(img):     return img.copy()
def pp_clahe(img, clip=2.0, tile=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[:, :, 0] = c.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
def pp_clahe_strong(img):     return pp_clahe(img, clip=4.0)
def pp_clahe_small_tile(img): return pp_clahe(img, clip=3.0, tile=4)
def pp_unsharp(img, r=3, a=2.5):
    b = cv2.GaussianBlur(img, (0,0), r)
    return np.clip(cv2.addWeighted(img, 1+a, b, -a, 0), 0, 255).astype(np.uint8)
def pp_gamma_dark(img, g=0.5):
    lut = np.array([((i/255.)**g)*255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)
def pp_gamma_very_dark(img):  return pp_gamma_dark(img, g=0.3)
def pp_clahe_unsharp(img):    return pp_unsharp(pp_clahe(img))
def pp_grayscale(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
def pp_local_norm(img, k=15):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    kk = k if k%2==1 else k+1
    lm = cv2.blur(gray,(kk,kk))
    diff = gray-lm
    lstd = np.sqrt(np.maximum(cv2.blur(gray**2,(kk,kk))-lm**2,0))+1e-6
    n = diff/lstd; n=(n-n.min())/(n.max()-n.min()+1e-6)*255
    return cv2.cvtColor(n.astype(np.uint8), cv2.COLOR_GRAY2BGR)
def pp_tophat(img, ks=7):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ks,ks))
    bh=cv2.morphologyEx(g,cv2.MORPH_BLACKHAT,k)
    e=np.clip(g.astype(np.int32)-bh.astype(np.int32)*2,0,255).astype(np.uint8)
    return cv2.cvtColor(e,cv2.COLOR_GRAY2BGR)
def pp_blackhat_viz(img, ks=7):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ks,ks))
    bh=cv2.morphologyEx(g,cv2.MORPH_BLACKHAT,k)
    return cv2.cvtColor(cv2.equalizeHist(bh),cv2.COLOR_GRAY2BGR)
def pp_bilateral(img):
    return pp_unsharp(cv2.bilateralFilter(img,7,50,50),r=2,a=1.5)
def pp_clahe_gamma(img): return pp_gamma_dark(pp_clahe(img),g=0.6)

ALL_METHODS = [
    ("01_Original", pp_original), ("02_CLAHE", pp_clahe),
    ("03_CLAHE_Strong", pp_clahe_strong), ("04_CLAHE_SmallTile", pp_clahe_small_tile),
    ("05_Unsharp_Mask", pp_unsharp), ("06_Gamma_Dark", pp_gamma_dark),
    ("07_Gamma_VeryDark", pp_gamma_very_dark), ("08_CLAHE_Unsharp", pp_clahe_unsharp),
    ("09_Grayscale", pp_grayscale), ("10_Local_Norm", pp_local_norm),
    ("11_TopHat", pp_tophat), ("12_BlackHat_Viz", pp_blackhat_viz),
    ("13_Bilateral_Sharp", pp_bilateral), ("14_CLAHE_Gamma", pp_clahe_gamma),
]


# ==================== Dataset ====================

class PreprocessDataset(torch.utils.data.Dataset):
    def __init__(self, root, pp_fn, augment=False):
        self.base = datasets.ImageFolder(str(root))
        self.pp_fn = pp_fn
        self.augment = augment
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    @property
    def classes(self): return self.base.classes
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        path, label = self.base.samples[idx]
        bgr = cv2.imread(path)
        if bgr is None: bgr = np.zeros((64,64,3),dtype=np.uint8)
        bgr = self.pp_fn(bgr)
        bgr = cv2.resize(bgr,(IMGSZ_CLF,IMGSZ_CLF),interpolation=cv2.INTER_LINEAR)
        pil = Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        return self.norm(pil), label


# ==================== 训练单个方法 ====================

def train_one(method_name, pp_fn, classes) -> dict:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    run_name = f"r{ROUND}_{method_name}"
    save_dir = CLF_WEIGHTS_DIR / method_name / "weights"
    save_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(project=WANDB_PROJECT, name=run_name,
               config={"round": ROUND, "method": method_name,
                       "model": CLF_MODEL_NAME, "epochs": CLF_EPOCHS}, reinit=True)

    train_ds = PreprocessDataset(SPLIT_DIR/"train", pp_fn)
    val_ds   = PreprocessDataset(SPLIT_DIR/"val",   pp_fn)
    train_ld = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = timm.create_model(CLF_MODEL_NAME, pretrained=True, num_classes=len(classes))
    model = model.to(DEVICE)
    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.AdamW(model.parameters(), lr=CLF_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CLF_EPOCHS)

    best_val = 0.0; no_improve = 0

    for epoch in range(1, CLF_EPOCHS+1):
        model.train()
        correct = total = 0
        for imgs, labels in train_ld:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad(); loss = crit(model(imgs), labels); loss.backward(); opt.step()
            correct += (model(imgs).detach().argmax(1)==labels).sum().item(); total += labels.size(0)
        sched.step()

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for imgs, labels in val_ld:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                vc += (model(imgs).argmax(1)==labels).sum().item(); vt += labels.size(0)
        val_acc = vc/vt

        wandb.log({"val/acc": val_acc, "epoch": epoch})

        if val_acc > best_val:
            best_val = val_acc; no_improve = 0
            torch.save(model.state_dict(), save_dir/"best.pt")
        else:
            no_improve += 1
        if no_improve >= CLF_PATIENCE:
            break

    wandb.summary["best_val_acc"] = best_val
    wandb.finish()
    logger.info(f"  {method_name}: best_val_acc={best_val:.4f}")
    return {"method": method_name, "best_val_acc": best_val}


# ==================== 快速模式：只重训最优方法 ====================

def get_best_method_from_csv() -> str:
    if not ENSEMBLE_CSV.exists():
        return ALL_METHODS[0][0]
    rows = []
    with open(ENSEMBLE_CSV, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) if k!="combo" else v for k,v in row.items()})
    best = max(rows, key=lambda r: (r["recall_swd"], r["acc"]))
    # 取组合中第一个方法
    return best["combo"].split("+")[0].strip()


# ==================== 主流程 ====================

def main():
    rebuild_split_dataset()

    # 确定要训练的方法
    if FULL_SEARCH:
        methods_to_train = ALL_METHODS
        logger.info(f"完整模式：训练所有 {len(methods_to_train)} 种预处理方法")
    else:
        best_name = get_best_method_from_csv()
        fn_map = dict(ALL_METHODS)
        methods_to_train = [(best_name, fn_map[best_name])]
        logger.info(f"快速模式：只重训最优方法 {best_name}")

    # 获取类别列表
    dummy_ds = datasets.ImageFolder(str(SPLIT_DIR/"train"))
    classes  = dummy_ds.classes

    wandb.login(key=WANDB_KEY)
    results = []
    for name, fn in methods_to_train:
        r = train_one(name, fn, classes)
        results.append(r)

    # 更新 results.csv（只更新训练过的方法，保留其他方法的旧结果）
    _update_results_csv(results)
    logger.info(f"训练完成，ENSEMBLE_CSV 已更新: {ENSEMBLE_CSV}")


def _update_results_csv(new_results: list[dict]):
    """把新训练结果更新到 ensemble_results.csv（保留其他方法的旧行）。"""
    existing = {}
    if ENSEMBLE_CSV.exists():
        with open(ENSEMBLE_CSV, newline="") as f:
            for row in csv.DictReader(f):
                # key 用 combo 字段（单方法时 combo == method name）
                existing[row["combo"]] = row

    for r in new_results:
        existing[r["method"]] = {
            "combo":          r["method"],
            "n_models":       1,
            "acc":            r["best_val_acc"],
            "recall_swd":     r["best_val_acc"],   # 无 test set 时用 val_acc 估算
            "recall_non_swd": r["best_val_acc"],
        }

    CLF_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ENSEMBLE_CSV, "w", newline="") as f:
        fieldnames = ["combo","n_models","acc","recall_swd","recall_non_swd"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing.values())


if __name__ == "__main__":
    main()
