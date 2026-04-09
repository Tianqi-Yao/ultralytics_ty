"""
v2_06b_run_classify_timm.py
用 ensemble_results.csv 中最优的两个集成组合对 FiftyOne 数据集做二次分类过滤。

启动时自动读取 ensemble_results.csv，取：
  - COMBO_A：recall_swd 最高的组合（尽量不漏虫）
  - COMBO_B：acc 最高的组合（整体最准）

对每个组合：
  - 每个 patch 送入组合内所有模型（各自用对应预处理），softmax 概率取平均
  - 平均 swd 概率 >= CLF_THRESH 则保留该框
  - 结果写入独立的 FiftyOne 字段，方便在 App 里对比

FiftyOne 字段命名：
  {PRED_FIELD}_ens_recallA    （recall_swd 最优组合）
  {PRED_FIELD}_ens_accB       （acc 最优组合）
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

FO_DATASET  = "del_test_cls_2025_north_v1__jeff"
PRED_FIELD  = "pred_yolo11m_20pct_null_images_add_rawData_batch_16_final"

ENSEMBLE_CSV = Path(
    "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/03_code/03_train_model/yolo_cls/output/preprocess_search/ensemble_results.csv"
)
WEIGHTS_ROOT = Path(
    "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/03_code/03_train_model/yolo_cls/output/preprocess_search"
)

MODEL_NAME  = "efficientnet_b0"
CLASSES     = ["non_swd", "swd"]   # ImageFolder 字母序
CLF_THRESH  = 0.5
PAD_RATIO   = 0.5
IMGSZ       = 224
BATCH_SIZE  = 32
DEVICE      = "cuda:0"

WANDB_KEY = "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"
# ========================


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
    return pp_unsharp(cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50),
                      radius=2, amount=1.5)

def pp_clahe_gamma(img):  return pp_gamma_dark(pp_clahe(img, clip=2.0), gamma=0.6)

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

NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ==================== 读取 CSV，选出两个最优组合 ====================

def load_best_combos(csv_path: Path) -> tuple[dict, dict]:
    """
    返回 (best_recall_swd_row, best_acc_row)
    每行格式：{"combo": "A + B + C", "acc": ..., "recall_swd": ..., ...}
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if k not in ("combo",) else v
                         for k, v in row.items()})

    best_recall = max(rows, key=lambda r: (r["recall_swd"], r["acc"]))
    best_acc    = max(rows, key=lambda r: (r["acc"], r["recall_swd"]))
    return best_recall, best_acc


def parse_combo(combo_str: str) -> list[str]:
    """'08_CLAHE_Unsharp + 12_BlackHat_Viz' → ['08_CLAHE_Unsharp', '12_BlackHat_Viz']"""
    return [m.strip() for m in combo_str.split("+")]


# ==================== 模型加载 ====================

def load_model(method_name: str) -> torch.nn.Module:
    weights_path = WEIGHTS_ROOT / method_name / "weights" / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"权重不存在: {weights_path}")
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(str(weights_path), map_location=DEVICE))
    model.eval().to(DEVICE)
    logger.info(f"  已加载: {method_name}")
    return model


# ==================== 集成推理函数工厂 ====================

def crop_pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """PIL RGB → BGR ndarray，用于送入 cv2 预处理。"""
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def make_ensemble_predict_fn(
    models: list[torch.nn.Module],
    preprocess_fns: list[callable],
) -> callable:
    """
    返回符合 classify_and_filter_sample 接口的集成推理函数。
    对每个 crop：
      1. 用各自的预处理函数处理
      2. 各自推理得到 softmax 概率
      3. 平均概率，取 argmax 作为最终分类
    """
    swd_idx = CLASSES.index("swd")

    def predict(crops: list[Image.Image]) -> list[tuple[str, float]]:
        # crops: list of PIL images
        # 预先转成 BGR ndarray
        crops_bgr = [crop_pil_to_bgr(c) for c in crops]

        # 每个模型单独推理，累加概率
        sum_probs = None
        for model, pp_fn in zip(models, preprocess_fns):
            # 预处理 + resize + normalize
            tensors = []
            for bgr in crops_bgr:
                processed = pp_fn(bgr)
                resized   = cv2.resize(processed, (IMGSZ, IMGSZ),
                                       interpolation=cv2.INTER_LINEAR)
                rgb_pil   = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                tensors.append(NORMALIZE(rgb_pil))

            batch = torch.stack(tensors).to(DEVICE)
            with torch.no_grad():
                probs = F.softmax(model(batch), dim=1).cpu().numpy()  # [N, C]

            sum_probs = probs if sum_probs is None else sum_probs + probs

        avg_probs = sum_probs / len(models)   # [N, C]

        results = []
        for prob in avg_probs:
            swd_prob = float(prob[swd_idx])
            if swd_prob >= CLF_THRESH:
                results.append(("swd", swd_prob))
            else:
                non_idx = 1 - swd_idx
                results.append((CLASSES[non_idx], float(prob[non_idx])))
        return results

    return predict


# ==================== FiftyOne 推理 ====================

def run_inference(
    ds: fo.Dataset,
    view,
    predict_fn: callable,
    out_field: str,
    combo_name: str,
    wandb_run,
):
    if not ds.has_sample_field(out_field):
        ds.add_sample_field(
            out_field,
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )

    before_total = after_total = 0
    total = len(view)

    for i, sample in enumerate(view.iter_samples(progress=True, autosave=True)):
        dets_obj = sample.get_field(PRED_FIELD)
        n_before = len(dets_obj.detections) if dets_obj and dets_obj.detections else 0

        classify_and_filter_sample(
            sample=sample,
            pred_field=PRED_FIELD,
            out_field=out_field,
            clf_predict_fn=predict_fn,
            target_label="swd",
            clf_thresh=CLF_THRESH,
            pad_ratio=PAD_RATIO,
            batch_size=BATCH_SIZE,
        )

        n_after = len(sample[out_field].detections) if sample[out_field] else 0
        before_total += n_before
        after_total  += n_after

        if (i + 1) % 100 == 0:
            wandb_run.log({
                f"{combo_name}/samples_processed": i + 1,
                f"{combo_name}/retention_rate": after_total / max(1, before_total),
            })

    retention = after_total / max(1, before_total)
    logger.info(
        f"[{combo_name}] 完成  {before_total} → {after_total} 框 "
        f"（保留率 {retention*100:.1f}%）  字段: {out_field}"
    )
    wandb_run.summary.update({
        f"{combo_name}/boxes_before":  before_total,
        f"{combo_name}/boxes_after":   after_total,
        f"{combo_name}/retention_rate": retention,
    })


# ==================== 主流程 ====================

def main():
    if not ENSEMBLE_CSV.exists():
        logger.error(f"找不到 ensemble_results.csv: {ENSEMBLE_CSV}")
        logger.error("请先运行 v2_10_ensemble_eval.py 生成该文件。")
        return

    # 读取最优两个组合
    best_recall_row, best_acc_row = load_best_combos(ENSEMBLE_CSV)

    combo_a_methods = parse_combo(best_recall_row["combo"])
    combo_b_methods = parse_combo(best_acc_row["combo"])

    logger.info(f"COMBO A（recall_swd 最优）: {combo_a_methods}")
    logger.info(f"  recall_swd={best_recall_row['recall_swd']:.4f}  acc={best_recall_row['acc']:.4f}")
    logger.info(f"COMBO B（acc 最优）: {combo_b_methods}")
    logger.info(f"  acc={best_acc_row['acc']:.4f}  recall_swd={best_acc_row['recall_swd']:.4f}")

    # FiftyOne 输出字段名（取组合方法名缩写，避免太长）
    def short_field(methods: list[str], suffix: str) -> str:
        abbr = "_".join(m.split("_")[0] for m in methods)   # 取序号部分
        return f"{PRED_FIELD}_ens_{suffix}_{abbr}_t{int(CLF_THRESH*100)}"

    out_field_a = short_field(combo_a_methods, "recallA")
    out_field_b = short_field(combo_b_methods, "accB")

    # 如果两个组合完全相同，只跑一次
    same_combo = set(combo_a_methods) == set(combo_b_methods)
    if same_combo:
        logger.info("两个最优组合完全相同，只跑一次。")
        out_field_b = None

    # 加载模型
    logger.info("\n加载 COMBO A 模型...")
    models_a = [load_model(m) for m in combo_a_methods]
    fns_a    = [METHOD_FN_MAP[m] for m in combo_a_methods]

    if not same_combo:
        logger.info("\n加载 COMBO B 模型...")
        models_b = [load_model(m) for m in combo_b_methods]
        fns_b    = [METHOD_FN_MAP[m] for m in combo_b_methods]
    else:
        models_b, fns_b = models_a, fns_a

    predict_a = make_ensemble_predict_fn(models_a, fns_a)
    predict_b = make_ensemble_predict_fn(models_b, fns_b) if not same_combo else predict_a

    # wandb
    wandb.login(key=WANDB_KEY)
    run = wandb.init(
        project="swd_cls_ensemble_inference",
        name=f"ensemble_{FO_DATASET}_thresh{int(CLF_THRESH*100)}",
        config={
            "dataset":     FO_DATASET,
            "pred_field":  PRED_FIELD,
            "clf_thresh":  CLF_THRESH,
            "pad_ratio":   PAD_RATIO,
            "combo_a":     combo_a_methods,
            "combo_b":     combo_b_methods,
            "recall_swd_a": best_recall_row["recall_swd"],
            "acc_b":        best_acc_row["acc"],
        },
    )

    # 加载 FiftyOne 数据集
    logger.info(f"\n加载数据集: {FO_DATASET}")
    ds   = fo.load_dataset(FO_DATASET)
    view = ds.exists(PRED_FIELD)
    logger.info(f"有 {PRED_FIELD} 字段的 sample: {len(view)}")

    # 运行 COMBO A
    logger.info(f"\n===== COMBO A（recall_swd 最优）=====")
    run_inference(ds, view, predict_a, out_field_a, "combo_a", run)

    # 运行 COMBO B（如果不同）
    if not same_combo and out_field_b:
        logger.info(f"\n===== COMBO B（acc 最优）=====")
        run_inference(ds, view, predict_b, out_field_b, "combo_b", run)

    logger.info("\n在 FiftyOne App 中对比：")
    logger.info(f"  原始检测框：    {PRED_FIELD}")
    logger.info(f"  Ensemble A：   {out_field_a}  （recall_swd 优先）")
    if not same_combo and out_field_b:
        logger.info(f"  Ensemble B：   {out_field_b}  （acc 优先）")

    run.finish()


if __name__ == "__main__":
    main()
