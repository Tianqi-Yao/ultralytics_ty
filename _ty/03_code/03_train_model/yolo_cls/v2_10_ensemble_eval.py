"""
v2_10_ensemble_eval.py
集成投票评估：将多个预处理方法训练的模型做 soft voting，
在 test set 上评估，寻找最优集成组合。

流程：
  1. 加载多个 best.pt（来自 v2_09 的 preprocess_search 输出）
  2. 对 test set 每张图分别用各自的预处理 + 推理，得到各模型的 softmax 概率
  3. 平均概率后取 argmax（soft voting）
  4. 评估 accuracy / per-class recall
  5. 自动搜索所有可能的 2/3/4 模型组合，输出最优组合排名

用法：
  python v2_10_ensemble_eval.py
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import timm

# ========================
# 配置
# ========================

TEST_DIR = Path("/workspace/_ty/03_code/03_train_model/yolo_cls/data_v2/test")

WEIGHTS_ROOT = Path(
    "/workspace/_ty/03_code/03_train_model/yolo_cls/output/preprocess_search"
)

MODEL_NAME  = "efficientnet_b0"
IMGSZ       = 224
BATCH_SIZE  = 32
DEVICE      = "cuda:0"
NUM_WORKERS = 4

# 类别顺序（与训练时 ImageFolder 字母序一致）
CLASSES = ["non_swd", "swd"]

# 参与集成的候选模型（来自 v2_09 结果，选 swd recall 有差异的互补模型）
# 可以手动增删，或直接用全部14个（AUTO_SEARCH=True 时自动遍历）
CANDIDATE_METHODS = [
    "08_CLAHE_Unsharp",    # 最平衡：swd=0.60, non_swd=0.71
    "12_BlackHat_Viz",     # swd 最强：swd=0.93, non_swd=0.42
    "10_Local_Norm",       # 次平衡：swd=0.53, non_swd=0.75
    "04_CLAHE_SmallTile",  # swd=0.33, non_swd=0.72
    "09_Grayscale",        # swd=0.47, non_swd=0.59
    "06_Gamma_Dark",       # non_swd 强：non_swd=0.875
    "05_Unsharp_Mask",     # non_swd=0.849
    "01_Original",         # baseline
]

# True = 自动搜索 CANDIDATE_METHODS 中所有 2/3/4 模型组合，找最优
AUTO_SEARCH     = True
MAX_COMBO_SIZE  = 4   # 最大组合数量（超过4个收益递减）
TOP_N_RESULTS   = 20  # 打印前N个最优组合

# ========================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ==================== 预处理方法（与 v2_09 完全一致） ====================

def pp_original(img):      return img.copy()
def pp_clahe(img, clip=2.0, tile=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[:, :, 0] = c.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
def pp_clahe_strong(img):      return pp_clahe(img, clip=4.0)
def pp_clahe_small_tile(img):  return pp_clahe(img, clip=3.0, tile=4)
def pp_unsharp(img, radius=3, amount=2.5):
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    return np.clip(cv2.addWeighted(img, 1+amount, blurred, -amount, 0), 0, 255).astype(np.uint8)
def pp_gamma_dark(img, gamma=0.5):
    lut = np.array([((i/255.0)**gamma)*255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)
def pp_gamma_very_dark(img):   return pp_gamma_dark(img, gamma=0.3)
def pp_clahe_unsharp(img):     return pp_unsharp(pp_clahe(img, clip=2.0))
def pp_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
def pp_local_norm(img, kernel=15):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = kernel if kernel % 2 == 1 else kernel + 1
    lm = cv2.blur(gray, (k, k))
    diff = gray - lm
    lsq = cv2.blur(gray**2, (k, k))
    lstd = np.sqrt(np.maximum(lsq - lm**2, 0)) + 1e-6
    norm = diff / lstd
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-6) * 255
    return cv2.cvtColor(norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
def pp_tophat(img, ksize=7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    enhanced = np.clip(gray.astype(np.int32) - bh.astype(np.int32)*2, 0, 255).astype(np.uint8)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
def pp_blackhat_viz(img, ksize=7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    return cv2.cvtColor(cv2.equalizeHist(bh), cv2.COLOR_GRAY2BGR)
def pp_bilateral(img):
    smooth = cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)
    return pp_unsharp(smooth, radius=2, amount=1.5)
def pp_clahe_gamma(img):   return pp_gamma_dark(pp_clahe(img, clip=2.0), gamma=0.6)

METHOD_FN_MAP = {
    "01_Original":         pp_original,
    "02_CLAHE":            pp_clahe,
    "03_CLAHE_Strong":     pp_clahe_strong,
    "04_CLAHE_SmallTile":  pp_clahe_small_tile,
    "05_Unsharp_Mask":     pp_unsharp,
    "06_Gamma_Dark":       pp_gamma_dark,
    "07_Gamma_VeryDark":   pp_gamma_very_dark,
    "08_CLAHE_Unsharp":    pp_clahe_unsharp,
    "09_Grayscale":        pp_grayscale,
    "10_Local_Norm":       pp_local_norm,
    "11_TopHat":           pp_tophat,
    "12_BlackHat_Viz":     pp_blackhat_viz,
    "13_Bilateral_Sharp":  pp_bilateral,
    "14_CLAHE_Gamma":      pp_clahe_gamma,
}


# ==================== Dataset ====================

class PreprocessDataset(Dataset):
    def __init__(self, root: Path, preprocess_fn):
        self.base = datasets.ImageFolder(str(root))
        self.preprocess_fn = preprocess_fn
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):  return len(self.base)

    def __getitem__(self, idx):
        path, label = self.base.samples[idx]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            img_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        img_bgr = self.preprocess_fn(img_bgr)
        img_bgr = cv2.resize(img_bgr, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.normalize(Image.fromarray(img_rgb)), label


# ==================== 模型加载 & 推理 ====================

def load_model(method_name: str) -> torch.nn.Module | None:
    weights_path = WEIGHTS_ROOT / method_name / "weights" / "best.pt"
    if not weights_path.exists():
        logger.warning(f"  权重不存在，跳过: {weights_path}")
        return None
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(str(weights_path), map_location=DEVICE))
    model.eval().to(DEVICE)
    return model


@torch.no_grad()
def get_probs(model: torch.nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """返回 (probs [N, C], labels [N])"""
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.vstack(all_probs), np.concatenate(all_labels)


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    acc   = (preds == labels).mean()
    recall = {}
    for c, name in enumerate(CLASSES):
        mask = labels == c
        recall[name] = (preds[mask] == c).mean() if mask.sum() > 0 else float("nan")
    return {"acc": acc, **{f"recall_{k}": v for k, v in recall.items()}}


# ==================== 主流程 ====================

def main():
    if not TEST_DIR.exists():
        logger.error(f"TEST_DIR 不存在: {TEST_DIR}")
        return

    # ---- 加载所有候选模型 + 预计算概率 ----
    logger.info(f"加载候选模型（共 {len(CANDIDATE_METHODS)} 个）...")
    method_probs: dict[str, np.ndarray] = {}
    labels_ref: np.ndarray | None = None

    for method_name in CANDIDATE_METHODS:
        fn = METHOD_FN_MAP.get(method_name)
        if fn is None:
            logger.warning(f"  未知方法: {method_name}")
            continue

        model = load_model(method_name)
        if model is None:
            continue

        loader = DataLoader(
            PreprocessDataset(TEST_DIR, fn),
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
        probs, labels = get_probs(model, loader)
        method_probs[method_name] = probs

        if labels_ref is None:
            labels_ref = labels

        single = compute_metrics(probs, labels)
        logger.info(
            f"  {method_name:<25}  acc={single['acc']:.4f}  "
            f"recall_swd={single['recall_swd']:.4f}  "
            f"recall_non_swd={single['recall_non_swd']:.4f}"
        )

    available = list(method_probs.keys())
    logger.info(f"\n成功加载 {len(available)} 个模型: {available}")

    if len(available) < 2:
        logger.error("可用模型不足 2 个，无法做集成。")
        return

    # ---- 搜索所有组合 ----
    all_results = []

    if AUTO_SEARCH:
        logger.info(f"\n搜索所有 2~{MAX_COMBO_SIZE} 模型组合...")
        combos = []
        for size in range(2, min(MAX_COMBO_SIZE, len(available)) + 1):
            combos.extend(itertools.combinations(available, size))
        logger.info(f"共 {len(combos)} 种组合")

        for combo in combos:
            avg_probs = np.mean([method_probs[m] for m in combo], axis=0)
            metrics   = compute_metrics(avg_probs, labels_ref)
            all_results.append({
                "combo":          " + ".join(combo),
                "n_models":       len(combo),
                **metrics,
            })
    else:
        # 只评估全部候选模型的集成
        avg_probs = np.mean(list(method_probs.values()), axis=0)
        metrics   = compute_metrics(avg_probs, labels_ref)
        all_results.append({
            "combo":    " + ".join(available),
            "n_models": len(available),
            **metrics,
        })

    # ---- 排名 ----
    # 排序策略：先按 recall_swd 降序，再按 acc 降序（SWD 召回更重要）
    all_results.sort(key=lambda x: (x["recall_swd"], x["acc"]), reverse=True)

    print("\n" + "=" * 100)
    print(f"{'排名':>4}  {'模型数':>5}  {'acc':>8}  {'recall_swd':>12}  {'recall_non_swd':>16}  组合")
    print("-" * 100)

    for i, r in enumerate(all_results[:TOP_N_RESULTS], 1):
        marker = " ← 最优" if i == 1 else ""
        print(
            f"{i:>4}  {r['n_models']:>5}  {r['acc']:>8.4f}  "
            f"{r['recall_swd']:>12.4f}  {r['recall_non_swd']:>16.4f}  "
            f"{r['combo']}{marker}"
        )

    print("=" * 100)

    best = all_results[0]
    print(f"\n🏆 最优集成组合（按 recall_swd 优先）:")
    print(f"   {best['combo']}")
    print(f"   acc={best['acc']:.4f}  recall_swd={best['recall_swd']:.4f}  recall_non_swd={best['recall_non_swd']:.4f}")

    # 也输出 acc 最高的组合（可能不同）
    best_acc = max(all_results, key=lambda x: x["acc"])
    if best_acc["combo"] != best["combo"]:
        print(f"\n📊 acc 最高组合:")
        print(f"   {best_acc['combo']}")
        print(f"   acc={best_acc['acc']:.4f}  recall_swd={best_acc['recall_swd']:.4f}  recall_non_swd={best_acc['recall_non_swd']:.4f}")

    # 保存完整结果
    import csv
    out_csv = WEIGHTS_ROOT / "ensemble_results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n完整结果已保存: {out_csv}")


if __name__ == "__main__":
    main()
