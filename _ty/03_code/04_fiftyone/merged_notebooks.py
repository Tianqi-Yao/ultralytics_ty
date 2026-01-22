
################################################################################
# NOTEBOOK: 31_sahi.ipynb
################################################################################


# --- Cell 1 from 31_sahi.ipynb ---

import fiftyone as fo  

# 加载已存在的数据集  
session = fo.launch_app()


# --- Cell 2 from 31_sahi.ipynb ---

from pathlib import Path


subdir_path_list = [
    Path("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/01_data/03_16mp_fiftyone_dataset/ms2_0726-0809_13_ok/images_640_ov20")

]

subdir_name_list = [
    path.name for path in subdir_path_list
]

version = "sahi_v1"

display(subdir_path_list)
display(subdir_name_list)
len(subdir_path_list), len(subdir_name_list)


# --- Cell 3 from 31_sahi.ipynb ---

from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction

ckpt_path = "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/02_models/best_models/04_swd_hbb/model_v2_4datasets_noAug_seed0_yolo11s_data_split_custom_8.pt"

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=ckpt_path,
    confidence_threshold=0.1, ## same as the default value for our base model
    image_size=640,
    device="cuda", # or 'cuda'
)

# 定义一个函数来预测切片结果
def predict_with_slicing(sample, label_field, **kwargs):
    result = get_sliced_prediction(
        sample.filepath, detection_model, verbose=0, **kwargs
    )
    sample[label_field] = fo.Detections(detections=result.to_fiftyone_detections())


# --- Cell 4 from 31_sahi.ipynb ---

import fiftyone as fo  
import fiftyone.utils.coco as fouc
from pathlib import Path

count = 0
for subdir_path, subdir_name in zip(subdir_path_list, subdir_name_list):
    # 第二次退出
    # if count >= 2:
    #     break
    # count += 1
    print(f"Processing subdir: {subdir_name},{subdir_path}")
    if f"{subdir_name}_{version}" in fo.list_datasets():
        fo.delete_dataset(f"{subdir_name}_{version}") 
    dataset = fo.Dataset.from_images_dir(  
        str(subdir_path / "data"),
        name=f"{subdir_name}_{version}"  
    ) 

    # 添加COCO标签到现有数据集  
    fouc.add_coco_labels(  
        dataset,   
        label_field="ground_truth",  # 存储标签的字段名  
        labels_or_path=f"{subdir_path}/org_label_no_overlap.json",  # COCO JSON文件路径  
        categories={1: "swd"},  # 可选：类别信息，会自动从JSON中提取  
        label_type="detections"  # 标签类型：detections, segmentations, keypoints  
    )

    kwargs = {"overlap_height_ratio": 0.2, "overlap_width_ratio": 0.2}

    for sample in dataset.iter_samples(progress=True, autosave=True):
        predict_with_slicing(sample, label_field="small_slices", slice_height=640, slice_width=640, **kwargs)
    # break


# --- Cell 5 from 31_sahi.ipynb ---

for subdir_path in subdir_path_list:
    print(subdir_path) 
for subdir_name in subdir_name_list:
    print(subdir_name)


# --- Cell 6 from 31_sahi.ipynb ---

subdir_path_list2 = [
    "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/01_data/03_16mp_fiftyone_dataset/ms2_0726-0809_13_ok",
    "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/01_data/03_16mp_fiftyone_dataset/sw1_0605-0613_07_ok",
    # "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/01_data/03_16mp_fiftyone_dataset/ms1_0809-0823_34_ok",
    # "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/01_data/03_16mp_fiftyone_dataset/ms1_0710-0726_36_ok",
]
subdir_name_list2 = [
    "ms2_0726-0809_13_ok_sahi_v1",
    "sw1_0605-0613_07_ok_sahi_v1",
    # "ms1_0809-0823_34_ok_sahi_v1",
    # "ms1_0710-0726_36_ok_sahi_v1",
]

# for subdir_path, subdir_name in zip(subdir_path_list2, subdir_name_list2):
#     small_slice_results = session.view.evaluate_detections("small_slices", gt_field="ground_truth", eval_key="eval_small_slices")
#     print(f"{subdir_name} Small slice results:")
#     small_slice_results.print_report()


# --- Cell 7 from 31_sahi.ipynb ---

import fiftyone as fo
from fiftyone import ViewField as F

# 定义不同的置信度阈值
confidence_thresholds = [
                            0.5, 0.6, 0.7, 0.8, 0.85,
                            0.9, 0.91, 0.92, 0.93
                        ]

# for subdir_path, subdir_name in zip(subdir_path_list2, subdir_name_list2):
for subdir_path, subdir_name in zip(subdir_path_list, subdir_name_list):
    subdir_name = f"{subdir_name}_{version}"
    print(f"\n=== {subdir_name} ===")

    dataset = fo.load_dataset(subdir_name)

    print("GT 框数量:", dataset.count("ground_truth.detections"))
    print("预测框数量:", dataset.count("small_slices.detections"))


    # 创建不同置信度的视图
    views = {}
    results = {}

    for threshold in confidence_thresholds:
        views[threshold] = dataset.filter_labels(
            "small_slices", F("confidence") > threshold, only_matches=False
        )

        eval_key = f"eval_conf_{int(threshold * 100)}"
        results[threshold] = views[threshold].evaluate_detections(
            "small_slices", gt_field="ground_truth", compute_mAP=True, 
            eval_key=eval_key
        )

    # 打印比较报告
    print(f"\n{subdir_name} 详细比较:")
    print("置信度 | mAP    | 精确度 | 召回率 | 预测数")
    print("-" * 50)

    for threshold in confidence_thresholds:
        res = results[threshold]
        
        pred_count = views[threshold].count("small_slices.detections")
        print(
            f"{threshold:6.2f} | {res.mAP():6.4f} | {res.metrics()['precision']:6.4f} | {res.metrics()['recall']:6.4f} | {pred_count:6d}"
        )


################################################################################
# NOTEBOOK: 32_sahi_batch.ipynb
################################################################################


# --- Cell 1 from 32_sahi_batch.ipynb ---

import fiftyone as fo  

# 加载已存在的数据集  
dataset = fo.load_dataset("my-dataset")
session = fo.launch_app(dataset)


# --- Cell 2 from 32_sahi_batch.ipynb ---

from pathlib import Path

def fetch_subsequent_dir(data_root: Path, target_subdir_name: Path):
    data_paths = list(data_root.glob(f"*/{target_subdir_name}"))
    # display(data_paths)
    # get sub dir - no target_subdir_name
    subdir_path_list = [data_path.parent for data_path in data_paths]
    # display(subdir_path_list)
    subdir_name_list = [subdir.name for subdir in subdir_path_list]
    # display(subdir_name_list)
    return subdir_path_list, subdir_name_list


# data_root = Path("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/01_data/00_test")
data_root = Path("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/01_data/a02_16mp_2024_datasets_fiftyone")
version = "sahi_v1"

target_subdir_name = Path("data")
subdir_path_list, subdir_name_list = fetch_subsequent_dir(data_root, target_subdir_name)
display(subdir_path_list)
display(subdir_name_list)
len(subdir_path_list), len(subdir_name_list)


# --- Cell 3 from 32_sahi_batch.ipynb ---

from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction

ckpt_path = "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/02_models/best_models/04_swd_hbb/model_v2_4datasets_noAug_seed0_yolo11s_data_split_custom_8.pt"

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=ckpt_path,
    confidence_threshold=0.1, ## same as the default value for our base model
    image_size=640,
    device="cuda", # or 'cuda'
)

# 定义一个函数来预测切片结果
def predict_with_slicing(sample, label_field, **kwargs):
    result = get_sliced_prediction(
        sample.filepath, detection_model, verbose=0, **kwargs
    )
    sample[label_field] = fo.Detections(detections=result.to_fiftyone_detections())


# --- Cell 4 from 32_sahi_batch.ipynb ---

import fiftyone as fo  
import fiftyone.utils.coco as fouc
from pathlib import Path

count = 0
for subdir_path, subdir_name in zip(subdir_path_list, subdir_name_list):
    # 第二次退出
    # if count >= 2:
    #     break
    # count += 1
    print(f"Processing subdir: {subdir_name},{subdir_path}")
    if f"{subdir_name}_{version}" in fo.list_datasets():
        fo.delete_dataset(f"{subdir_name}_{version}") 
    dataset = fo.Dataset.from_images_dir(  
        str(subdir_path / "data"),
        name=f"{version}_{subdir_name}"  
    ) 

    # 添加COCO标签到现有数据集  
    fouc.add_coco_labels(  
        dataset,   
        label_field="ground_truth",  # 存储标签的字段名  
        labels_or_path=f"{subdir_path}/org_label_no_overlap.json",  # COCO JSON文件路径  
        categories={1: "swd"},  # 可选：类别信息，会自动从JSON中提取  
        label_type="detections"  # 标签类型：detections, segmentations, keypoints  
    )

    kwargs = {"overlap_height_ratio": 0.2, "overlap_width_ratio": 0.2}

    for sample in dataset.iter_samples(progress=True, autosave=True):
        predict_with_slicing(sample, label_field="small_slices", slice_height=640, slice_width=640, **kwargs)
    # break


# --- Cell 5 from 32_sahi_batch.ipynb ---

for subdir_path in subdir_path_list:
    print(subdir_path) 
for subdir_name in subdir_name_list:
    print(subdir_name)


# --- Cell 6 from 32_sahi_batch.ipynb ---

subdir_path_list2 = [
    "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/01_data/03_16mp_fiftyone_dataset/ms2_0726-0809_13_ok",
    "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/01_data/03_16mp_fiftyone_dataset/sw1_0605-0613_07_ok",
    # "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/01_data/03_16mp_fiftyone_dataset/ms1_0809-0823_34_ok",
    # "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/01_data/03_16mp_fiftyone_dataset/ms1_0710-0726_36_ok",
]
subdir_name_list2 = [
    "ms2_0726-0809_13_ok_sahi_v1",
    "sw1_0605-0613_07_ok_sahi_v1",
    # "ms1_0809-0823_34_ok_sahi_v1",
    # "ms1_0710-0726_36_ok_sahi_v1",
]

# for subdir_path, subdir_name in zip(subdir_path_list2, subdir_name_list2):
#     small_slice_results = session.view.evaluate_detections("small_slices", gt_field="ground_truth", eval_key="eval_small_slices")
#     print(f"{subdir_name} Small slice results:")
#     small_slice_results.print_report()


# --- Cell 7 from 32_sahi_batch.ipynb ---

import fiftyone as fo
from fiftyone import ViewField as F

# 定义不同的置信度阈值
confidence_thresholds = [
                            0.5, 0.6, 0.7, 0.8, 0.85,
                            0.9, 0.91, 0.92, 0.93
                        ]

# for subdir_path, subdir_name in zip(subdir_path_list2, subdir_name_list2):
for subdir_path, subdir_name in zip(subdir_path_list, subdir_name_list):
    subdir_name = f"{subdir_name}_{version}"
    print(f"\n=== {subdir_name} ===")

    dataset = fo.load_dataset(subdir_name)

    print("GT 框数量:", dataset.count("ground_truth.detections"))
    print("预测框数量:", dataset.count("small_slices.detections"))


    # 创建不同置信度的视图
    views = {}
    results = {}

    for threshold in confidence_thresholds:
        views[threshold] = dataset.filter_labels(
            "small_slices", F("confidence") > threshold, only_matches=False
        )

        eval_key = f"eval_conf_{int(threshold * 100)}"
        results[threshold] = views[threshold].evaluate_detections(
            "small_slices", gt_field="ground_truth", compute_mAP=True, 
            eval_key=eval_key
        )

    # 打印比较报告
    print(f"\n{subdir_name} 详细比较:")
    print("置信度 | mAP    | 精确度 | 召回率 | 预测数")
    print("-" * 50)

    for threshold in confidence_thresholds:
        res = results[threshold]
        
        pred_count = views[threshold].count("small_slices.detections")
        print(
            f"{threshold:6.2f} | {res.mAP():6.4f} | {res.metrics()['precision']:6.4f} | {res.metrics()['recall']:6.4f} | {pred_count:6d}"
        )


################################################################################
# NOTEBOOK: 32_sahi_batch_v2.ipynb
################################################################################


# --- Cell 1 from 32_sahi_batch_v2.ipynb ---

import fiftyone as fo  

# 加载已存在的数据集  
dataset = fo.load_dataset("my-dataset")
session = fo.launch_app(dataset)


# --- Cell 2 from 32_sahi_batch_v2.ipynb ---

# %% =========================
# 0) Imports
# =========================
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import fiftyone as fo
import fiftyone.utils.coco as fouc
from fiftyone import ViewField as F

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


# --- Cell 3 from 32_sahi_batch_v2.ipynb ---

# %% =========================
# 1) User Config
# =========================
data_root = Path("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/01_data/a02_16mp_2024_datasets_fiftyone")
target_subdir_name = Path("data")

version = "sahi_null_v2"

# 多模型：直接在这里放 .pt
model_root = Path("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/02_models/best_models/04_swd_hbb") 
model_names = [
    # "a02_yolo11s_custom7_v2-13_7_34_36-40_10-11_8.pt",
    # "a02_yolo11s_custom7_v2-13_7_34_36-40_10-11_4.pt",
    # "a02_yolo11s_custom7_v1-34_36_40_11_-13-10_16.pt",
    # "a02_yolo11s_custom7_v4-36_40_10_11-7_34-13_16.pt",
    # "a02_yolo11s_custom7_v4-36_40_10_11-7_34-13_8.pt",
    "a03_yolo11s_custom7null_cv1_ms2_0809-0823_10_ok_16.pt",
    "a03_yolo11n_custom7null_cv1_ms2_0809-0823_10_ok_8.pt",
    "a03_yolo11s_custom7null_cv1_ms2_0809-0823_10_ok_8.pt",
    "a03_yolo11s_custom7null_cv1_ms2_0809-0823_10_ok_4.pt",
    "a04_yolo11s_custom7null_cv1_ms2_0809-0823_10_ok_16.pt",
    "a04_yolo11n_custom7null_cv1_ms2_0809-0823_10_ok_4.pt",
    "a04_yolo11n_custom7null_cv1_ms2_0809-0823_10_ok_8.pt",
    "a04_yolo11s_custom7null_cv1_ms2_0809-0823_10_ok_8.pt",
]

ckpt_paths = [str(model_root / name) for name in model_names]

# SAHI slicing overlap
sahi_kwargs = {"overlap_height_ratio": 0.2, "overlap_width_ratio": 0.2}
slice_height = 640
slice_width = 640

# Evaluate thresholds
confidence_thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93]

# COCO label path under each subdir_path (you used this name)
coco_json_name = "org_label_no_overlap.json"

# category map (optional; COCO can also infer)
categories = {1: "swd"}

# 输出汇总文件
out_dir = Path("./_eval_exports")
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / f"eval_summary__{version}.csv"
out_xlsx = out_dir / f"eval_summary__{version}.xlsx"


# --- Cell 4 from 32_sahi_batch_v2.ipynb ---

# %% =========================
# 2) Helpers
# =========================
def fetch_subsequent_dir(data_root: Path, target_subdir_name: Path) -> Tuple[List[Path], List[str]]:
    data_paths = list(data_root.glob(f"*/{target_subdir_name}"))
    subdir_path_list = [data_path.parent for data_path in data_paths]
    subdir_name_list = [subdir.name for subdir in subdir_path_list]
    return subdir_path_list, subdir_name_list


def model_tag_from_path(p: str) -> str:
    # 字段名建议只用 stem，避免空格和奇怪字符
    return Path(p).stem

def ensure_pred_field(dataset: fo.Dataset, pred_field: str) -> None:
    if pred_field not in dataset.get_field_schema():
        dataset.add_sample_field(
            pred_field,
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )

def predict_with_slicing(sample: fo.Sample, detection_model: AutoDetectionModel, label_field: str, **kwargs) -> None:
    result = get_sliced_prediction(
        sample.filepath, detection_model, verbose=0, **kwargs
    )
    ensure_pred_field(dataset, label_field)
    sample[label_field] = fo.Detections(detections=result.to_fiftyone_detections())


def ensure_dataset_with_gt(subdir_path: Path, subdir_name: str, version: str) -> fo.Dataset:
    """
    方案B：dataset 只创建一次，GT 也只导入一次
    """
    ds_name = f"{version}_{subdir_name}"

    if ds_name in fo.list_datasets():
        dataset = fo.load_dataset(ds_name)
        return dataset

    dataset = fo.Dataset.from_images_dir(
        str(subdir_path / "data"),
        name=ds_name
    )

    coco_path = subdir_path / coco_json_name
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO JSON not found: {coco_path}")

    fouc.add_coco_labels(
        dataset,
        label_field="ground_truth",
        labels_or_path=str(coco_path),
        categories=categories,
        label_type="detections",
    )

    return dataset

def run_inference_for_model(dataset: fo.Dataset, ckpt_path: str, pred_field: str) -> None:
    """
    对 dataset 跑一个模型的 SAHI slicing 推理，并写入 pred_field
    """
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=ckpt_path,
        confidence_threshold=0.1,
        image_size=640,
        device="cuda",
    )

    for sample in dataset.iter_samples(progress=True, autosave=True):
        predict_with_slicing(
            sample,
            detection_model=detection_model,
            label_field=pred_field,
            slice_height=slice_height,
            slice_width=slice_width,
            **sahi_kwargs
        )


def evaluate_model_fields(
    dataset: fo.Dataset,
    ds_name: str,
    subdir_path: Path,
    ckpt_path: str,
    model_tag: str,
    pred_field: str,
    thresholds: List[float],
) -> List[Dict[str, Any]]:
    """
    对某个 pred_field（某个模型）在多个置信度阈值下评估，返回 rows
    """
    rows: List[Dict[str, Any]] = []

    gt_count = dataset.count("ground_truth.detections")
    total_pred_count = dataset.count(f"{pred_field}.detections")

    for thr in thresholds:
        view = dataset.filter_labels(
            pred_field, F("confidence") > thr, only_matches=False
        )

        new_model_tag = model_tag.replace("-", "__")

        eval_key = f"eval_{new_model_tag}_conf_{int(thr * 100)}"
        res = view.evaluate_detections(
            pred_field,
            gt_field="ground_truth",
            compute_mAP=True,
            eval_key=eval_key
        )

        pred_count_thr = view.count(f"{pred_field}.detections")
        metrics = res.metrics()

        row = {
            "dataset_name": ds_name,
            "subdir_name": ds_name.replace(f"_{version}", ""),
            "subdir_path": str(subdir_path),
            "version": version,

            "model_tag": model_tag,
            "ckpt_path": ckpt_path,
            "pred_field": pred_field,

            "confidence_threshold": thr,

            "gt_count": int(gt_count),
            "pred_count_total": int(total_pred_count),
            "pred_count_at_threshold": int(pred_count_thr),

            "mAP": float(res.mAP()),
            "precision": float(metrics.get("precision", float("nan"))),
            "recall": float(metrics.get("recall", float("nan"))),
            "f1": float(metrics.get("f1", float("nan"))),
        }
        rows.append(row)

    return rows


# --- Cell 5 from 32_sahi_batch_v2.ipynb ---

# %% =========================
# 3) Discover subdirs
# =========================
subdir_path_list, subdir_name_list = fetch_subsequent_dir(data_root, target_subdir_name)
print("Found subdirs:", len(subdir_name_list))
for n, p in zip(subdir_name_list, subdir_path_list):
    print(n, p)


# --- Cell 6 from 32_sahi_batch_v2.ipynb ---

# %% =========================
# 4) Build datasets (GT once) + run inference (multi-model fields)
# =========================
for subdir_path, subdir_name in zip(subdir_path_list, subdir_name_list):
    dataset = ensure_dataset_with_gt(subdir_path, subdir_name, version)
    ds_name = dataset.name

    print(f"\n==================== Dataset: {ds_name} ====================")
    print("GT 框数量:", dataset.count("ground_truth.detections"))

    for ckpt_path in ckpt_paths:
        model_tag = model_tag_from_path(ckpt_path)
        pred_field = f"small_slices_{model_tag}"

        # 如果你想“已存在就跳过”，打开这段：
        # if dataset.count(f"{pred_field}.detections") > 0:
        #     print(f"[SKIP] {model_tag} already has predictions in {pred_field}")
        #     continue

        print(f"\n[INFER] Model: {model_tag}")
        print(f"        Field: {pred_field}")
        run_inference_for_model(dataset, ckpt_path, pred_field)
        print("预测框数量:", dataset.count(f"{pred_field}.detections"))


# --- Cell 7 from 32_sahi_batch_v2.ipynb ---

# %% =========================
# 5) Evaluate all models -> DataFrame summary
# =========================
all_rows: List[Dict[str, Any]] = []

for subdir_path, subdir_name in zip(subdir_path_list, subdir_name_list):
    ds_name = f"{version}_{subdir_name}"
    dataset = fo.load_dataset(ds_name)

    print(f"\n==================== EVAL Dataset: {ds_name} ====================")

    for ckpt_path in ckpt_paths:
        model_tag = model_tag_from_path(ckpt_path)
        pred_field = f"small_slices_{model_tag}"

        if pred_field not in dataset.get_field_schema():
            print(f"[WARN] Missing pred_field={pred_field} in dataset={ds_name}, skip.")
            continue

        print(f"\n[EVAL] Model: {model_tag} | Field: {pred_field}")
        rows = evaluate_model_fields(
            dataset=dataset,
            ds_name=ds_name,
            subdir_path=subdir_path,
            ckpt_path=ckpt_path,
            model_tag=model_tag,
            pred_field=pred_field,
            thresholds=confidence_thresholds,
        )
        all_rows.extend(rows)

df = pd.DataFrame(all_rows)

# 一个更好用的排序：按 dataset、model、threshold
if not df.empty:
    df = df.sort_values(["dataset_name", "model_tag", "confidence_threshold"]).reset_index(drop=True)

print("\nDone. Summary shape:", df.shape)
display(df.head(20))


# --- Cell 8 from 32_sahi_batch_v2.ipynb ---

# %% =========================
# 6) Add "best" columns (Best F1 / Best Recall@Precision>=X etc.)
# =========================
import numpy as np
import pandas as pd

if not df.empty:
    group_cols = ["dataset_name", "model_tag"]

    # --- 1) 确保 precision/recall 是数值 ---
    df["precision"] = pd.to_numeric(df["precision"], errors="coerce")
    df["recall"] = pd.to_numeric(df["recall"], errors="coerce")

    # --- 2) 可靠地计算 F1（因为 df["f1"] 全是 NaN）---
    den = df["precision"] + df["recall"]
    df["f1_calc"] = np.where(den > 0, 2 * df["precision"] * df["recall"] / den, np.nan)

    # --- 3) Best F1（只在 f1_calc 有效的行上取 idxmax）---
    df["is_best_f1_in_group"] = False
    valid = df[df["f1_calc"].notna()]
    if not valid.empty:
        idx_best = valid.groupby(group_cols)["f1_calc"].idxmax()
        df.loc[idx_best, "is_best_f1_in_group"] = True
    else:
        print("[WARN] All f1_calc are NaN; cannot mark best F1")

    # --- 4) Best Recall @ Precision >= gate ---
    PREC_GATE = 0.90
    df["is_best_recall_at_prec_gate"] = False

    tmp = df[(df["precision"] >= PREC_GATE) & (df["recall"].notna())].copy()
    if not tmp.empty:
        idx_best_recall = tmp.groupby(group_cols)["recall"].idxmax()
        df.loc[idx_best_recall, "is_best_recall_at_prec_gate"] = True
    else:
        print(f"[WARN] No rows meet precision >= {PREC_GATE}")

# 看 best F1 的行
display(df[df.get("is_best_f1_in_group", False)].head(50))


# --- Cell 9 from 32_sahi_batch_v2.ipynb ---

# %% =========================
# 7) Export
# =========================
df.to_csv(out_csv, index=False)
print("Saved CSV:", out_csv)

# Excel 可选（如果 df 很大也能存，但会慢点）
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="summary", index=False)

    # 也顺便给一个 “best-only” sheet
    best_f1 = df[df["is_best_f1_in_group"]].copy()
    best_f1.to_excel(writer, sheet_name="best_f1", index=False)

    best_recall_gate = df[df["is_best_recall_at_prec_gate"]].copy()
    best_recall_gate.to_excel(writer, sheet_name="best_recall_at_prec", index=False)

print("Saved XLSX:", out_xlsx)


# %% =========================
# 8) Launch FiftyOne App (optional)
# =========================
# 你想看某个 dataset:
# ds = fo.load_dataset(f"{subdir_name_list[0]}_{version}")
# session = fo.launch_app(ds)
# session.view = ds
