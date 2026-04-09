"""
03_run_classifier.py
读取 ensemble_results.csv，用 recall_swd 最优的集成组合对 PRED_FIELD 做二次过滤。
结果写入 CLF_FIELD。
"""

from __future__ import annotations
import csv
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import fiftyone as fo
import timm
import wandb

from pipeline_config import *
from ty_nogt_tools.classify import classify_and_filter_sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ==================== 预处理方法（与 v2_09/v2_10 完全一致） ====================

def pp_original(img):     return img.copy()
def pp_clahe(img, clip=2.0, tile=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[:, :, 0] = c.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
def pp_clahe_strong(img):     return pp_clahe(img, clip=4.0)
def pp_clahe_small_tile(img): return pp_clahe(img, clip=3.0, tile=4)
def pp_unsharp(img, radius=3, amount=2.5):
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    return np.clip(cv2.addWeighted(img, 1+amount, blurred, -amount, 0), 0, 255).astype(np.uint8)
def pp_gamma_dark(img, gamma=0.5):
    lut = np.array([((i/255.0)**gamma)*255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)
def pp_gamma_very_dark(img):  return pp_gamma_dark(img, gamma=0.3)
def pp_clahe_unsharp(img):    return pp_unsharp(pp_clahe(img, clip=2.0))
def pp_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
def pp_local_norm(img, kernel=15):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = kernel if kernel % 2 == 1 else kernel + 1
    lm = cv2.blur(gray, (k, k))
    diff = gray - lm
    lstd = np.sqrt(np.maximum(cv2.blur(gray**2, (k, k)) - lm**2, 0)) + 1e-6
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
    return pp_unsharp(cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50), radius=2, amount=1.5)
def pp_clahe_gamma(img):  return pp_gamma_dark(pp_clahe(img, clip=2.0), gamma=0.6)

METHOD_FN_MAP = {
    "01_Original": pp_original, "02_CLAHE": pp_clahe,
    "03_CLAHE_Strong": pp_clahe_strong, "04_CLAHE_SmallTile": pp_clahe_small_tile,
    "05_Unsharp_Mask": pp_unsharp, "06_Gamma_Dark": pp_gamma_dark,
    "07_Gamma_VeryDark": pp_gamma_very_dark, "08_CLAHE_Unsharp": pp_clahe_unsharp,
    "09_Grayscale": pp_grayscale, "10_Local_Norm": pp_local_norm,
    "11_TopHat": pp_tophat, "12_BlackHat_Viz": pp_blackhat_viz,
    "13_Bilateral_Sharp": pp_bilateral, "14_CLAHE_Gamma": pp_clahe_gamma,
}

NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_best_combo() -> tuple[list[str], dict]:
    """从 ensemble_results.csv 读取 recall_swd 最优组合。"""
    rows = []
    with open(ENSEMBLE_CSV, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) if k != "combo" else v for k, v in row.items()})
    best = max(rows, key=lambda r: (r["recall_swd"], r["acc"]))
    methods = [m.strip() for m in best["combo"].split("+")]
    return methods, best


def load_model(method_name: str) -> torch.nn.Module:
    wp = CLF_WEIGHTS_DIR / method_name / "weights" / "best.pt"
    if not wp.exists():
        raise FileNotFoundError(f"权重不存在: {wp}")
    model = timm.create_model(CLF_MODEL_NAME, pretrained=False, num_classes=len(CLF_CLASSES))
    model.load_state_dict(torch.load(str(wp), map_location=DEVICE))
    return model.eval().to(DEVICE)


def make_ensemble_predict_fn(models, preprocess_fns):
    swd_idx = CLF_CLASSES.index("swd")

    def predict(crops: list[Image.Image]) -> list[tuple[str, float]]:
        crops_bgr = [cv2.cvtColor(np.array(c.convert("RGB")), cv2.COLOR_RGB2BGR) for c in crops]
        sum_probs = None
        for model, pp_fn in zip(models, preprocess_fns):
            tensors = []
            for bgr in crops_bgr:
                processed = pp_fn(bgr)
                resized   = cv2.resize(processed, (IMGSZ_CLF, IMGSZ_CLF), interpolation=cv2.INTER_LINEAR)
                rgb_pil   = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                tensors.append(NORMALIZE(rgb_pil))
            batch = torch.stack(tensors).to(DEVICE)
            with torch.no_grad():
                probs = F.softmax(model(batch), dim=1).cpu().numpy()
            sum_probs = probs if sum_probs is None else sum_probs + probs

        avg = sum_probs / len(models)
        results = []
        for prob in avg:
            swd_p = float(prob[swd_idx])
            if swd_p >= CLF_THRESH:
                results.append(("swd", swd_p))
            else:
                non_idx = 1 - swd_idx
                results.append((CLF_CLASSES[non_idx], float(prob[non_idx])))
        return results

    return predict


def main():
    if not ENSEMBLE_CSV.exists():
        raise FileNotFoundError(
            f"ensemble_results.csv 不存在: {ENSEMBLE_CSV}\n"
            "请先运行 v2_10_ensemble_eval.py 生成该文件。"
        )

    combo_methods, combo_info = load_best_combo()
    logger.info(f"最优集成组合（recall_swd={combo_info['recall_swd']:.4f}）: {combo_methods}")

    models    = [load_model(m) for m in combo_methods]
    pp_fns    = [METHOD_FN_MAP[m] for m in combo_methods]
    predict_fn = make_ensemble_predict_fn(models, pp_fns)

    ds = fo.load_dataset(FO_DATASET)
    if not ds.has_sample_field(PRED_FIELD):
        raise RuntimeError(f"请先运行 02_run_detection.py（{PRED_FIELD} 字段不存在）")

    if not ds.has_sample_field(CLF_FIELD):
        ds.add_sample_field(CLF_FIELD, fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections)

    view = ds.exists(PRED_FIELD)
    before_total = after_total = 0

    wandb.login(key=WANDB_KEY)
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"classifier_r{ROUND}",
        config={"round": ROUND, "combo": combo_methods,
                "clf_thresh": CLF_THRESH, "pad_ratio": PAD_RATIO},
    )

    for i, sample in enumerate(view.iter_samples(progress=True, autosave=True)):
        dets = sample.get_field(PRED_FIELD)
        n_before = len(dets.detections) if dets and dets.detections else 0

        classify_and_filter_sample(
            sample=sample, pred_field=PRED_FIELD, out_field=CLF_FIELD,
            clf_predict_fn=predict_fn, target_label="swd",
            clf_thresh=CLF_THRESH, pad_ratio=PAD_RATIO, batch_size=BATCH_SIZE_CLF,
        )

        n_after = len(sample[CLF_FIELD].detections) if sample[CLF_FIELD] else 0
        before_total += n_before
        after_total  += n_after

        if (i + 1) % 100 == 0:
            run.log({"samples": i+1, "retention": after_total / max(1, before_total)})

    retention = after_total / max(1, before_total)
    logger.info(f"完成：{before_total} → {after_total} 框（保留率 {retention*100:.1f}%）")
    logger.info(f"结果已写入字段: {CLF_FIELD}")

    run.summary.update({"boxes_before": before_total, "boxes_after": after_total,
                        "retention_rate": retention})
    run.finish()


if __name__ == "__main__":
    main()
