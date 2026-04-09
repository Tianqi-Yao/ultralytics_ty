"""
pipeline_config.py
全局配置文件，所有 pipeline 脚本通过 `from pipeline_config import *` 使用。

每次迭代只需修改 ROUND，字段名自动更新。
"""

from pathlib import Path

# ============================================================
# 迭代控制 — 每轮 +1
# ============================================================
ROUND = 1

# ============================================================
# FiftyOne 数据集
# ============================================================
FO_DATASET = "swd_pipeline_v1"         # FiftyOne 数据集名称

# FiftyOne 字段名（含轮次，避免覆盖历史结果）
PRED_FIELD = f"pred_r{ROUND}"          # YOLO/SAHI 推理结果
CLF_FIELD  = f"clf_r{ROUND}"           # 分类器过滤后结果
EVAL_KEY   = f"eval_r{ROUND}"          # evaluate_detections 的 eval_key

GT_FIELD   = "ground_truth"            # GT 字段名（有标注时）

# ============================================================
# 数据路径 — ⚠️ 按实际情况修改
# ============================================================

# 原始图片目录（创建数据集时使用）
IMAGES_DIR = Path("/workspace/data/images")

# YOLO txt 标注目录（可选，无则留 None，跳过 GT 写入）
LABELS_DIR = None   # 例如：Path("/workspace/data/labels")

# 类别列表（与 YOLO txt 的 class_id 对应）
CLASS_NAMES = ["swd"]

# ============================================================
# 模型路径
# ============================================================

# YOLO 检测模型（用于 02_run_detection.py）
DET_MODEL_PATH = Path(
    "/workspace/_ty/00_pipeline/02_current_batch_run_model/02_TrainingModel"
    "/output/swd_model_v5_nullImagesAdded_final_noAug_seed42"
    "/yolo11m_batch16/weights/best.pt"
)

# 分类器权重目录（来自 v2_09 preprocess_search 输出）
CLF_WEIGHTS_DIR = Path(
    "/workspace/_ty/03_code/03_train_model/yolo_cls/output/preprocess_search"
)
ENSEMBLE_CSV = CLF_WEIGHTS_DIR / "ensemble_results.csv"

# 分类器 timm 模型名（需与训练时一致）
CLF_MODEL_NAME = "efficientnet_b0"
CLF_CLASSES    = ["non_swd", "swd"]    # ImageFolder 字母序

# ============================================================
# 数据管理路径
# ============================================================

_BASE = Path("/workspace/_ty/03_code/03_train_model/yolo_cls")

CROPS_DIR     = _BASE / "data_v2/raw_crops"
SPLIT_DIR     = _BASE / "data_v2/split_0.8_0.2"
RESULTS_DIR   = _BASE / "output/pipeline_results"

# ============================================================
# 推理 / 训练参数
# ============================================================

# SAHI 推理参数
SAHI_SLICE_SIZE   = 640
SAHI_OVERLAP      = 0.2
CONF_THRESH       = 0.1     # 检测置信度阈值

# 分类器参数
CLF_THRESH        = 0.5     # 分类置信度阈值（swd 概率 >= 此值才保留）
PAD_RATIO         = 0.5     # 裁 patch 时的 padding 比例
IMGSZ_CLF         = 224     # 分类器输入尺寸
BATCH_SIZE_CLF    = 32

# 重训参数
CLF_EPOCHS        = 150
CLF_PATIENCE      = 40
CLF_LR            = 1e-4
FULL_SEARCH       = False   # False=快速模式（只训最优预处理），True=全14种搜索

SEED              = 42
DEVICE            = "cuda:0"

# ============================================================
# 工具路径（复用现有代码）
# ============================================================
import sys
_TOOLS_DIR = Path(
    "/workspace/_ty/00_pipeline/02_current_batch_run_model/07_no_GT_run"
)
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

# wandb
WANDB_KEY     = "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"
WANDB_PROJECT = f"swd_pipeline_{FO_DATASET}"
