# 项目快速地图 MAP

> 忘了去哪找？扫一眼这里。详细说明看 [README.md](README.md)。在 VS Code 预览模式（`Ctrl+Shift+V`）下相对路径链接可直接点击跳转。

---

## 1. 工作流一览

```
FiftyOne 标注库     │     ▼[01 数据准备]  FiftyOne 当前视图 → 切割 640 瓦片 → YOLO 格式标签     │     ▼[02 模型训练]  train.py → 输出 best.pt     │     ▼[03 SAHI 推理] 滑窗推理（有 GT），生成预测结果     │     ▼[04 真实场景评估] 与 Ground Truth 对比 → Excel 报告     │     ▼[05 逐图评估]  每张图单独评估，结果写回 FiftyOne     │     ▼[06 绘图]     汇总图表可视化
```

---

## 2. 完整 Batch 训练流程

> 目录：[00_pipeline/02_current_batch_run/](00_pipeline/02_current_batch_run/)

步骤

做什么

入口文件

01 数据准备

FiftyOne 当前视图 → 640瓦片 + YOLO标签

[01_fiftyone_current_view_export_640_sub_images_and_labels.ipynb](00_pipeline/02_current_batch_run/01_PrepareTrainingData__fiftyone_to_trainingData/01_fiftyone_current_view_export_640_sub_images_and_labels.ipynb)

02 训练

YOLO训练，产物为 best.pt

[train.py](00_pipeline/02_current_batch_run/02_TrainingModel/train.py)

03 SAHI 推理

批量滑窗推理 + 写入 FiftyOne

[01_sahi_batch.ipynb](00_pipeline/02_current_batch_run/03_run_checkpoint_predict_raw_image_SAHI_way_and_evaluate/32_sahi_batch.ipynb)

04 真实场景评估

与GT对比 → Excel 报告

[evaluation_to_excel.ipynb](00_pipeline/02_current_batch_run/04_evaluate_Real-world_Performance_in_fiftyone/evaluation_to_excel.ipynb)

05 逐图评估

每张图单独评估 → FiftyOne

[33_per_image_export_fiftyone_evaluate_result.ipynb](00_pipeline/02_current_batch_run/05_evaluate_each_image_in_fiftyone/33_per_image_export_fiftyone_evaluate_result.ipynb)

06 绘图

汇总图表

[34_draw_plot.ipynb](00_pipeline/02_current_batch_run/06_draw_plot/34_draw_plot.ipynb) · [37_补充图](00_pipeline/02_current_batch_run/06_draw_plot/37_draw_plot.ipynb)

07 无GT推理

无标注图批量推理

[01_sahi_batch_noGT.ipynb](00_pipeline/02_current_batch_run/07_no_GT_run/01_sahi_batch_noGT.ipynb)

---

## 3. 生产推理流程

> 目录：[00_pipeline/05_trained_model_predict_raw_images/](00_pipeline/05_trained_model_predict_raw_images/)
> 
> 用途：不做评估，直接对原始图做推理并输出结果。

顺序

做什么

入口文件

01

原始图分割预测

[01_raw_image_seg_prediction.ipynb](00_pipeline/05_trained_model_predict_raw_images/01_raw_image_seg_prediction.ipynb)

02

去除重复预测（NMS）

[02_remove_duplicate_predictions.ipynb](00_pipeline/05_trained_model_predict_raw_images/02_remove_duplicate_predictions.ipynb)

03

裁出目标区域后二次推理

[03_cut_out_the_object_in_the_image_and_then_perform_inference.ipynb](00_pipeline/05_trained_model_predict_raw_images/03_cut_out_the_object_in_the_image_and_then_perform_inference.ipynb)

---

## 4. 核心工具库 ty_fo_tools

> 路径：[01_PrepareTrainingData.../ty_fo_tools/](00_pipeline/02_current_batch_run/01_PrepareTrainingData__fiftyone_to_trainingData/ty_fo_tools/)

模块

文件

主要函数/功能

COCO 瓦片切割

[cocoData/tiles.py](00_pipeline/02_current_batch_run/01_PrepareTrainingData__fiftyone_to_trainingData/ty_fo_tools/cocoData/tiles.py)

`export_labeled_tiles_from_coco()` 切割+导出标注瓦片

COCO NMS

[cocoData/nms.py](00_pipeline/02_current_batch_run/01_PrepareTrainingData__fiftyone_to_trainingData/ty_fo_tools/cocoData/nms.py)

`coco_nms_json()` 按 IoU 去重

FiftyOne 导出

[fiftyone/export_view.py](00_pipeline/02_current_batch_run/01_PrepareTrainingData__fiftyone_to_trainingData/ty_fo_tools/fiftyone/export_view.py)

FiftyOne 视图 → COCO JSON 格式

YOLO 数据集构建

[yoloData/build_trainning_dataset.py](00_pipeline/02_current_batch_run/01_PrepareTrainingData__fiftyone_to_trainingData/ty_fo_tools/yoloData/build_trainning_dataset.py)

COCO → YOLO，生成 dataset.yaml

---

## 5. 模型权重在哪里

> 路径：[02_fine-tuned_checkpoint/best_models/](02_fine-tuned_checkpoint/best_models/)

子目录

任务类型

`01_swd_seg/`

SWD 分割（Segmentation）

`02_swd_pose/`

SWD 姿态估计（Pose Estimation）

`03_dot_det/`

点检测（Dot Detection）

`04_swd_hbb/`

SWD 水平框检测（HBB）

`05_swd_obb/`

SWD 旋转框检测（OBB）

`11_swd_seg_64mp/`

SWD 分割，64MP 高分辨率版本

`91_swd_cls/`

SWD 分类（Classification）

---

## 6. 数据集在哪里

> 路径：[01_data/](01_data/)

目录

内容说明

`a01_20dataset/`

20类数据集

`a02_16mp_2024_datasets_fiftyone/`

16MP 2024年数据，FiftyOne 格式

`a03_raw_data/`

原始未处理数据

`a04_whole_raw_data/`

全量原始数据

`a05_2024_data/`

2024年整理后的数据

---

## 7. 可复用工具代码

> 路径：[03_code/](03_code/)

工具

路径

用途

FiftyOne 操作速查

[04_fiftyone/tools/00_note.ipynb](03_code/04_fiftyone/tools/00_note.ipynb)

常用 FiftyOne API 笔记

添加 focus 字段

[04_fiftyone/tools/36_fiftyone_add_focus_field.ipynb](03_code/04_fiftyone/tools/36_fiftyone_add_focus_field.ipynb)

给样本打 focus 标记

导出数据 patch

[04_fiftyone/tools/export_data_patch.ipynb](03_code/04_fiftyone/tools/export_data_patch.ipynb)

从 FiftyOne 导出图像补丁

YOLO11 水平框训练

[03_train_model/yolo11_hbb/](03_code/03_train_model/yolo11_hbb/)

HBB 任务训练脚本

YOLO 旋转框训练

[03_train_model/yolo_obb/](03_code/03_train_model/yolo_obb/)

OBB 任务训练脚本

YOLO 分类训练

[03_train_model/yolo_cls/](03_code/03_train_model/yolo_cls/)

分类任务训练脚本

---

## 8. 单次运行工具

> 路径：[00_pipeline/03_current_run_one/](00_pipeline/03_current_run_one/)
> 
> **与 Batch 流程的区别**：针对单次/临时任务的一次性脚本，不走完整 batch 流程，通常直接出结论。

文件

用途

[01_Data%20Cleaning_Preprocessing.ipynb](00_pipeline/03_current_run_one/01_Data%20Cleaning_Preprocessing.ipynb)

数据清洗预处理

[02_Sensor_Data%20Cleaning_Preprocessing.ipynb](00_pipeline/03_current_run_one/02_Sensor_Data%20Cleaning_Preprocessing.ipynb)

传感器数据清洗

[03_get_predict_excel_from_fiftyone.ipynb](00_pipeline/03_current_run_one/03_get_predict_excel_from_fiftyone.ipynb)

从 FiftyOne 导出预测结果到 Excel

[11_compare_gt_with_swd.ipynb](00_pipeline/03_current_run_one/11_compare_gt_with_swd.ipynb)

GT 与 SWD 预测结果对比分析

---

## 9. 关键参数速查

参数

值

来源位置

瓦片尺寸

**640 × 640**

`tiles.py` → `TileSpec.crop_size = 640`

瓦片重叠率

**20%**

`tiles.py` → `TileSpec.overlap_ratio = 0.2`

NMS IoU 阈值

**0.5**

`nms.py` → `coco_nms_json(iou_thresh=0.5)`

训练数据 YAML

按批次命名

[02_TrainingModel/02_yaml/](00_pipeline/02_current_batch_run/02_TrainingModel/02_yaml/)

训练输出目录

wandb + output

[02_TrainingModel/output/](00_pipeline/02_current_batch_run/02_TrainingModel/output/)