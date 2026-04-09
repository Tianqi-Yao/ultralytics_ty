"""
v2_08_preview_preprocessing.py
可视化对比多种图像预处理方法，帮助判断哪种最能突出 SWD 翅膀黑点。

输出目录结构：
  preprocess_preview/
  ├── swd/          — 正样本对比图（raw_crops/swd/）
  ├── fp/           — FP 负样本对比图（non_swd/fp_from_deployment/）
  ├── bg/           — 背景负样本对比图（non_swd/bg_from_train/）
  └── _overview_*.png — 各类别总览网格

使用方法：
  python v2_08_preview_preprocessing.py
  然后打开 preprocess_preview/ 目录查看结果。
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ========================
# 配置
# ========================

BASE_DIR = Path(
    "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/03_code/03_train_model/yolo_cls/data_v2"
)

INPUT_SOURCES = {
    "swd": BASE_DIR / "raw_crops/swd",
    "fp":  BASE_DIR / "raw_crops/non_swd/fp_from_deployment",
    "bg":  BASE_DIR / "raw_crops/non_swd/bg_from_train",
}

OUTPUT_DIR   = BASE_DIR / "preprocess_preview"
N_SAMPLES    = 20      # 每类随机抽取张数
DISPLAY_SIZE = 200     # 每个格子放大尺寸（原图很小，放大便于查看）
SEED         = 42
# ========================

random.seed(SEED)


# ==================== 预处理方法（12种） ====================

def method_original(img_bgr: np.ndarray) -> np.ndarray:
    """1. 原图（基准）"""
    return img_bgr.copy()


def method_clahe(img_bgr: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """2. CLAHE — LAB色空间自适应直方图均衡"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[:, :, 0] = c.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def method_clahe_strong(img_bgr: np.ndarray) -> np.ndarray:
    """3. CLAHE 强 — clipLimit=4.0，更激进均衡"""
    return method_clahe(img_bgr, clip=4.0)


def method_clahe_small_tile(img_bgr: np.ndarray) -> np.ndarray:
    """4. CLAHE 细粒度 — tile=4×4，更小局部窗口，黑点区域独立均衡"""
    return method_clahe(img_bgr, clip=3.0, tile=4)


def method_unsharp(img_bgr: np.ndarray, radius: int = 3, amount: float = 2.5) -> np.ndarray:
    """5. Unsharp Mask — 锐化，黑点边界更清晰"""
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), radius)
    out = cv2.addWeighted(img_bgr, 1 + amount, blurred, -amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def method_gamma_dark(img_bgr: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """6. Gamma 变暗 — gamma=0.5，压暗整体增大黑点对比"""
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img_bgr, lut)


def method_gamma_very_dark(img_bgr: np.ndarray) -> np.ndarray:
    """7. Gamma 极暗 — gamma=0.3，更强压暗"""
    return method_gamma_dark(img_bgr, gamma=0.3)


def method_clahe_unsharp(img_bgr: np.ndarray) -> np.ndarray:
    """8. CLAHE + Unsharp 组合 — 先均衡后锐化"""
    return method_unsharp(method_clahe(img_bgr, clip=2.0))


def method_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    """9. 灰度 — 去色，只看亮度信息"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def method_local_norm(img_bgr: np.ndarray, kernel: int = 15) -> np.ndarray:
    """10. 局部归一化 — 减去局部均值除以局部标准差，突出局部暗区"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = kernel if kernel % 2 == 1 else kernel + 1
    local_mean = cv2.blur(gray, (k, k))
    diff = gray - local_mean
    local_sq_mean = cv2.blur(gray ** 2, (k, k))
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
    local_std = np.sqrt(local_var) + 1e-6
    norm = diff / local_std
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-6) * 255
    norm = norm.astype(np.uint8)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)


def method_tophat(img_bgr: np.ndarray, ksize: int = 7) -> np.ndarray:
    """11. Top-Hat 变换 — 形态学，提取比周围更暗的小区域（直接突出黑点）"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    # Black-hat：比周围更暗的区域（即黑点）
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    # 将 black-hat 叠加到原图上增强暗区
    enhanced = cv2.subtract(gray, blackhat * 2)
    enhanced = np.clip(enhanced.astype(np.int32) - blackhat.astype(np.int32) * 2, 0, 255).astype(np.uint8)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def method_blackhat_highlight(img_bgr: np.ndarray, ksize: int = 7) -> np.ndarray:
    """12. Black-Hat 高亮 — 直接可视化 Black-Hat 结果（黑点区域变白，其他变黑）"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    # 增强对比度后可视化
    blackhat_eq = cv2.equalizeHist(blackhat)
    return cv2.cvtColor(blackhat_eq, cv2.COLOR_GRAY2BGR)


def method_bilateral(img_bgr: np.ndarray) -> np.ndarray:
    """13. 双边滤波 + 锐化 — 保边去噪后锐化，减少噪点干扰黑点识别"""
    smooth = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=50, sigmaSpace=50)
    return method_unsharp(smooth, radius=2, amount=1.5)


def method_clahe_gamma(img_bgr: np.ndarray) -> np.ndarray:
    """14. CLAHE + Gamma 组合 — 均衡后再压暗，双重增强黑点"""
    return method_gamma_dark(method_clahe(img_bgr, clip=2.0), gamma=0.6)


# 方法注册表（顺序即列顺序）
METHODS: list[tuple[str, callable]] = [
    ("01_Original",          method_original),
    ("02_CLAHE",             method_clahe),
    ("03_CLAHE_Strong",      method_clahe_strong),
    ("04_CLAHE_SmallTile",   method_clahe_small_tile),
    ("05_Unsharp_Mask",      method_unsharp),
    ("06_Gamma_Dark",        method_gamma_dark),
    ("07_Gamma_VeryDark",    method_gamma_very_dark),
    ("08_CLAHE+Unsharp",     method_clahe_unsharp),
    ("09_Grayscale",         method_grayscale),
    ("10_Local_Norm",        method_local_norm),
    ("11_TopHat",            method_tophat),
    ("12_BlackHat_Viz",      method_blackhat_highlight),
    ("13_Bilateral+Sharp",   method_bilateral),
    ("14_CLAHE+Gamma",       method_clahe_gamma),
]

N_METHODS = len(METHODS)


# ==================== 拼图工具 ====================

def load_font(size: int = 11):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def add_label(img_rgb: np.ndarray, text: str, cell_size: int,
              bg_color=(240, 240, 240)) -> np.ndarray:
    """在图片顶部添加方法名标签。"""
    label_h = 20
    canvas = Image.new("RGB", (cell_size, cell_size + label_h), bg_color)
    canvas.paste(Image.fromarray(img_rgb), (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((3, 3), text, fill=(30, 30, 30), font=load_font(10))
    return np.array(canvas)


def make_comparison_row(img_path: Path, cell_size: int,
                        label_bg=(240, 240, 240)) -> np.ndarray:
    """对单张图生成 1行×N列对比图（RGB numpy）。"""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取: {img_path}")

    cells = []
    for name, fn in METHODS:
        processed = fn(img_bgr)
        resized = cv2.resize(processed, (cell_size, cell_size),
                             interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        labeled = add_label(rgb, name, cell_size, label_bg)
        cells.append(labeled)

    return np.hstack(cells)


# ==================== 主流程 ====================

# 各类别标签颜色（label背景色）用于区分
CATEGORY_STYLE = {
    "swd": {"bg": (220, 240, 220), "title": "SWD 正样本"},   # 浅绿
    "fp":  {"bg": (240, 220, 220), "title": "FP 负样本（部署域 false positive）"},  # 浅红
    "bg":  {"bg": (220, 225, 240), "title": "背景负样本（训练集背景随机裁取）"},  # 浅蓝
}


def process_category(category: str, input_dir: Path, out_dir: Path):
    style = CATEGORY_STYLE[category]
    out_dir.mkdir(parents=True, exist_ok=True)

    all_imgs = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    if not all_imgs:
        print(f"  [{category}] 找不到图片：{input_dir}")
        return

    n = min(N_SAMPLES, len(all_imgs))
    samples = random.sample(all_imgs, n)
    samples.sort()

    print(f"  [{category}] 共 {len(all_imgs)} 张，抽取 {n} 张 → {out_dir}")

    cell_h = DISPLAY_SIZE + 20
    rows_for_overview = []

    for i, img_path in enumerate(samples):
        row = make_comparison_row(img_path, DISPLAY_SIZE, label_bg=style["bg"])
        rows_for_overview.append(row)
        Image.fromarray(row).save(str(out_dir / (img_path.stem + ".png")))

    # 总览图：加左侧序号列 + 顶部标题行
    label_w = 36
    title_h = 28
    row_h   = rows_for_overview[0].shape[0]
    row_w   = rows_for_overview[0].shape[1]

    # 标题行
    title_row = np.full((title_h, label_w + row_w, 3), 60, dtype=np.uint8)
    pil_title = Image.fromarray(title_row)
    draw = ImageDraw.Draw(pil_title)
    draw.text((6, 6), f"{style['title']}  (n={n})", fill=(255, 255, 200), font=load_font(13))
    title_row = np.array(pil_title)

    labeled_rows = []
    for idx, row in enumerate(rows_for_overview):
        label_col = np.full((row_h, label_w, 3), 210, dtype=np.uint8)
        pil_lc = Image.fromarray(label_col)
        draw = ImageDraw.Draw(pil_lc)
        draw.text((4, row_h // 2 - 8), f"#{idx+1:02d}", fill=(30, 30, 30), font=load_font(12))
        labeled_rows.append(np.hstack([np.array(pil_lc), row]))

    overview = np.vstack([title_row] + labeled_rows)
    overview_path = out_dir / "_overview.png"
    Image.fromarray(overview).save(str(overview_path))
    print(f"    总览图：{overview_path}")


def make_legend():
    """生成方法说明图例。"""
    descriptions = [
        "原图\n（基准）",
        "CLAHE\nclip=2.0\n均衡对比度",
        "CLAHE 强\nclip=4.0\n更激进均衡",
        "CLAHE\ntile=4×4\n更细粒度",
        "Unsharp\nMask 锐化\n黑点边缘清",
        "Gamma\n变暗 γ=0.5\n压暗增对比",
        "Gamma\n极暗 γ=0.3\n更强压暗",
        "CLAHE\n+Unsharp\n组合增强",
        "灰度\n去色\n只看亮度",
        "局部\n归一化\n突出暗区",
        "Top-Hat\n形态学\n暗区增强",
        "Black-Hat\n黑点区域\n直接可视化",
        "双边滤波\n+锐化\n去噪后增强",
        "CLAHE\n+Gamma\n双重增强",
    ]

    fig, axes = plt.subplots(1, N_METHODS, figsize=(N_METHODS * 1.8, 2.8))
    fig.suptitle("SWD 翅膀黑点增强方法说明（共14种）", fontsize=11, fontweight="bold", y=1.02)
    colors = plt.cm.tab20(np.linspace(0, 1, N_METHODS))

    for ax, (name, _), desc, color in zip(axes, METHODS, descriptions, colors):
        ax.set_facecolor((*color[:3], 0.15))
        ax.text(0.5, 0.55, desc, ha="center", va="center",
                fontsize=7.5, transform=ax.transAxes, linespacing=1.4)
        short = name.split("_", 1)[1].replace("_", "\n")
        ax.set_title(short, fontsize=7, pad=2, fontweight="bold")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(1.5)

    plt.tight_layout()
    legend_path = OUTPUT_DIR / "_legend.png"
    plt.savefig(str(legend_path), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  方法说明：{legend_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"输出根目录：{OUTPUT_DIR}")
    print(f"每类抽取：{N_SAMPLES} 张，预处理方法：{N_METHODS} 种\n")

    for category, input_dir in INPUT_SOURCES.items():
        out_subdir = OUTPUT_DIR / category
        process_category(category, input_dir, out_subdir)
        print()

    print("生成方法说明图例 ...")
    make_legend()

    print(f"\n===== 完成 =====")
    print(f"目录结构：")
    print(f"  {OUTPUT_DIR}/")
    print(f"  ├── swd/          正样本对比图 + _overview.png")
    print(f"  ├── fp/           FP负样本对比图 + _overview.png")
    print(f"  ├── bg/           背景负样本对比图 + _overview.png")
    print(f"  └── _legend.png   14种方法说明")
    print(f"\n建议查看：先看各类的 _overview.png，找出黑点最突出的列编号。")


if __name__ == "__main__":
    main()
