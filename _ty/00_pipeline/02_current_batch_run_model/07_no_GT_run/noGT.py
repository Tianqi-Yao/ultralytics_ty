# %% [markdown]
# # ── 0. 配置区 ──────────────────────────────────────────────

# %%
from pathlib import Path

# 路径
DATA_ROOT = Path("/media/tianqi/16tb/SWD/01_Data/2025_SWD_data_RAW/01_North")
MODEL_DIR = Path("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/02_fine-tuned_checkpoint/best_models/04_swd_hbb/null_image_trained_final_checkpoint")

# 推理版本（用于 FiftyOne dataset 命名 和 Excel 导出前缀）
VERSION = "2025_north_v1"

# 模型列表（注释掉不运行的）
MODEL_NAMES = [
    # "yolo11x_20pct_null_images_add_rawData_batch_8_final.pt",
    "yolo11m_20pct_null_images_add_rawData_batch_16_final.pt",
    # "yolo11m_20pct_null_images_add_rawData_batch_8_final.pt",
    # "yolo11l_20pct_null_images_add_rawData_batch_4_final.pt",
    # "yolo11s_20pct_null_images_add_rawData_batch_4_final.pt",
    # "yolo11n_20pct_null_images_add_rawData_batch_4_final.pt",
    # "yolo11x_20pct_null_images_add_rawData_batch_4_final.pt",
    # "yolo11s_20pct_null_images_add_rawData_batch_16_final.pt",
    # "yolo11n_20pct_null_images_add_rawData_batch_16_final.pt",
    # "yolo11n_20pct_null_images_add_rawData_batch_8_final.pt",
    # "yolo11l_20pct_null_images_add_rawData_batch_16_final.pt",
    # "yolo11l_20pct_null_images_add_rawData_batch_8_final.pt",
    # "yolo11s_20pct_null_images_add_rawData_batch_8_final.pt",
    # "yolo11m_20pct_null_images_add_rawData_batch_4_final.pt",
]

# 推理参数
CONF    = 0.01
DEVICE  = "cuda"
SLICE   = 640
OVERLAP = 0.2

# 开关
SKIP_IF_PRED_EXISTS = True

# Excel 导出路径
EXPORT_OUT_DIR = Path("./_pred_exports_2025_north_datasets")
EXPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # ── 1. 初始化日志 ──────────────────────────────────────────

# %%
import logging
# import ipynbname

log_file_name = "noGT.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_name, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
logger.info("============ Notebook logging 初始化完成 ============")

# %% [markdown]
# # ── 2. Step 1：扫描数据目录子目录 ──────────────────────────

# %%
# ── Step 1：扫描 DATA_ROOT 下的站点子目录 ───────────────────
# 输入：DATA_ROOT
# 输出：打印子目录列表供 Master 确认

logger.info("Step 1 开始：扫描数据子目录")


# %%
site_dirs = [
    # "/media/tianqi/16tb/SWD/01_Data/2025_SWD_data_RAW/01_North/air1/air1_0427-0611/output/ManualFocus_air1_0427-0611",
    "/media/tianqi/16tb/SWD/01_Data/2025_SWD_data_RAW/01_North/air1/air1_0611-0727/output/ManualFocus_air1_0611-0727",
    "/media/tianqi/16tb/SWD/01_Data/2025_SWD_data_RAW/01_North/air1/air1_0728-0921/output/ManualFocus_air1_0728-0921",
    # "/media/tianqi/16tb/SWD/01_Data/2025_SWD_data_RAW/01_North/air2/air2_0427-0611/output/ManualFocus_air2_0427-0611",
    # "/media/tianqi/16tb/SWD/01_Data/2025_SWD_data_RAW/01_North/air2/air2_0611-0725/output/ManualFocus_air2_0611-0725",
    # "/media/tianqi/16tb/SWD/01_Data/2025_SWD_data_RAW/01_North/jeff/jeff_0425-0611/output/ManualFocus_jeff_0425-0611",
    # "/media/tianqi/16tb/SWD/01_Data/2025_SWD_data_RAW/01_North/jeff/jeff_0611-0711/output/ManualFocus_jeff_0611-0711",
    # "/media/tianqi/16tb/SWD/01_Data/2025_SWD_data_RAW/01_North/jeff/jeff_0717-0915/output/ManualFocus_jeff_0717-0915",
    # "/media/tianqi/16tb/SWD/01_Data/2025_SWD_data_RAW/01_North/jeff/jeff_0921-1108/output/ManualFocus_jeff_0921-1108",
    # "/media/tianqi/16tb/SWD/01_Data/2025_SWD_data_RAW/01_North/lloyd1/lloyd1_0425-0619/output/ManualFocus_lloyd1_0425-0619",
    # "/media/tianqi/16tb/SWD/01_Data/2025_SWD_data_RAW/01_North/lloyd1/lloyd1_0619-0727/output/ManualFocus_lloyd1_0619-0727",
    # "/media/tianqi/16tb/SWD/01_Data/2025_SWD_data_RAW/01_North/lloyd2/lloyd2_0425-0507/output/ManualFocus_lloyd2_0425-0507",
    # "/media/tianqi/16tb/SWD/01_Data/2025_SWD_data_RAW/01_North/lloyd2/lloyd2_0425-0628/output/ManualFocus_lloyd2_0425-0628",
    # "/media/tianqi/16tb/SWD/01_Data/2025_SWD_data_RAW/01_North/southfarm1/southfarm1_0426-0608/output/ManualFocus_southfarm1_0426-0608",
    # "/media/tianqi/16tb/SWD/01_Data/2025_SWD_data_RAW/01_North/southfarm1/southfarm1_0611-0725/output/ManualFocus_southfarm1_0611-0725",
    # "/media/tianqi/16tb/SWD/01_Data/2025_SWD_data_RAW/01_North/southfarm1/southfarm1_0728-0921/output/ManualFocus_southfarm1_0728-0921",
    # "/media/tianqi/16tb/SWD/01_Data/2025_SWD_data_RAW/01_North/southfarm2/southfarm2_0426-0611/output/ManualFocus_southfarm2_0426-0611",
    # "/media/tianqi/16tb/SWD/01_Data/2025_SWD_data_RAW/01_North/southfarm2/southfarm2_0611-0727/output/ManualFocus_southfarm2_0611-0727",
    # "/media/tianqi/16tb/SWD/01_Data/2025_SWD_data_RAW/01_North/southfarm2/southfarm2_0728-0921/output/ManualFocus_southfarm2_0728-0921"
]

# %% [markdown]
# # ── 3. Step 2：批量 SAHI 推理，写入 FiftyOne ───────────────

# %%
# ── Step 2 辅助函数 ───────────────────────────────────────────
import fiftyone as fo
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def ensure_dataset(site_dir: Path) -> fo.Dataset:
    name = f"{VERSION}_{site_dir.name}"
    if name in fo.list_datasets():
        return fo.load_dataset(name)
    return fo.Dataset.from_images_dir(str(site_dir), name=name)


def ensure_field(ds: fo.Dataset, field: str) -> None:
    if field not in ds.get_field_schema():
        ds.add_sample_field(field, fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections)


def run_model(ds: fo.Dataset, ckpt_path: Path, pred_field: str) -> None:
    ensure_field(ds, pred_field)
    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(ckpt_path),
        confidence_threshold=CONF,
        image_size=640,
        device=DEVICE,
    )
    for sample in ds.iter_samples(progress=True, autosave=True):
        res = get_sliced_prediction(
            sample.filepath, model,
            slice_height=SLICE, slice_width=SLICE,
            overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP,
            verbose=0,
        )
        sample[pred_field] = fo.Detections(detections=res.to_fiftyone_detections())

# %%
# ── Step 2：批量推理 ─────────────────────────────────────────
# 输入：site_dirs 下各站点图像
# 输出：FiftyOne datasets，预测字段 pred_{model_tag}

logger.info("Step 2 开始：批量 SAHI 推理")

for site in site_dirs:
    try:
        site = Path(site)
        ds = ensure_dataset(site)
        logger.info(f"  [DATASET] {ds.name}，共 {len(ds)} 张")
        for m in MODEL_NAMES:
            ckpt = MODEL_DIR / m
            tag = Path(m).stem
            pred_field = f"pred_{tag}"
            if SKIP_IF_PRED_EXISTS and pred_field in ds.get_field_schema():
                if ds.count(f"{pred_field}.detections") > 0:
                    logger.info(f"  [SKIP] {tag}")
                    continue
            logger.info(f"  [RUN] {tag}")
            run_model(ds, ckpt, pred_field)
    except Exception as e:
        logger.error(f"推理失败 {site.name}: {e}")

logger.info("Step 2 完成：批量推理结束")

# %% [markdown]
# # ── 4. Step 3：导出预测结果到 Excel ─────────────────────────

# %%
# ── Step 3：导出预测结果到 Excel ─────────────────────────────
# 输入：VERSION 前缀的 FiftyOne datasets
# 输出：EXPORT_OUT_DIR 下的 summary + per_image Excel

logger.info("Step 3 开始：导出预测结果")

from ty_nogt_tools import export_two_excels_for_dataset
from datetime import datetime
import pandas as pd
import fiftyone as fo

PRED_FIELDS = {Path(n).stem: f"pred_{Path(n).stem}" for n in MODEL_NAMES}
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_summary_xlsx   = EXPORT_OUT_DIR / f"pred_summary__{VERSION}__{stamp}.xlsx"
out_per_image_xlsx = EXPORT_OUT_DIR / f"pred_per_image__{VERSION}__{stamp}.xlsx"

all_summary, all_per_image = [], []
ds_names = sorted([n for n in fo.list_datasets() if n.startswith(f"{VERSION}_")])
logger.info(f"发现 {len(ds_names)} 个 datasets")

for ds_name in ds_names:
    try:
        ds = fo.load_dataset(ds_name)
        df_sum, df_img = export_two_excels_for_dataset(ds, PRED_FIELDS)
        all_summary.append(df_sum)
        all_per_image.append(df_img)
    except Exception as e:
        logger.error(f"导出失败 {ds_name}: {e}")

df_summary   = pd.concat(all_summary,   ignore_index=True) if all_summary   else pd.DataFrame()
df_per_image = pd.concat(all_per_image, ignore_index=True) if all_per_image else pd.DataFrame()

try:
    with pd.ExcelWriter(out_summary_xlsx,   engine="openpyxl") as w: df_summary.to_excel(w,   sheet_name="summary",   index=False)
    with pd.ExcelWriter(out_per_image_xlsx, engine="openpyxl") as w: df_per_image.to_excel(w, sheet_name="per_image", index=False)
    logger.info(f"已保存: {out_summary_xlsx}, {out_per_image_xlsx}")
except Exception as e:
    logger.error(f"保存 Excel 失败: {e}")

logger.info(f"Step 3 完成：summary={df_summary.shape}, per_image={df_per_image.shape}")

# %% [markdown]
# # ── 5. 验证 ────────────────────────────────────────────────

# %%
# ── 验证：抽查 summary 和 per_image 结果 ─────────────────────
from IPython.display import display

logger.info(f"验证：summary={df_summary.shape}, per_image={df_per_image.shape}")
display(df_summary.head())
display(df_per_image.head())
