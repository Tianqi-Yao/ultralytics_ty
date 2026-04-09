# 05_paper_material — 论文资料总索引

本目录集中整理 SWD Paper 1（`doc/paper1/00_manuscript/manuscript_v3_lean.md`）所需的全部可复用资料。小文件直接复制，大数据（GB 级以上）用 `*_pointer.txt` 记录原路径。

**最后整理日期**：2026-04-09
**总体积**：~575 MB

---

## 目录索引

| 子目录 | 论文对应 | 体积 | 主要内容 |
|---|---|---|---|
| `01_models/` | §3.1 / §3.4 | 552 M | 14 个 YOLO11 变体训练权重 `.pt` + manifest |
| `02_training_runs/` | §3.1 Table 1 / Fig 4 / Kruskal-Wallis | 13 M | 14 个训练 run 的 results.csv + 曲线图 + 混淆矩阵 |
| `03_counting_validation/` | §3.2 Fig 6 / Table 2 | 44 K | 36 张图 AI vs 人工 计数 xlsx + 评估工具 |
| `04_timeseries_population/` | §3.3 Fig 7 / §4.3 Fig 8 | 6.5 M | 时序 CSV + 预处理 notebooks（2024 ms2 + 2025 北区 3 站点） |
| `05_plotting/` | 所有主图 | 68 K | 主绘图 notebook `34_draw_plot.ipynb` + 工具代码 |
| `06_edge_deployment/` | §3.4 Table 3 | 4.9 M | NCNN 导出脚本 + RPi4 推理脚本（ncnn_models 二进制用 pointer） |
| `07_large_data_pointers/` | — | 4 K | 原始图像数据集 / 代码 / final pipeline 的 pointer (共 ~500 G) |

---

## 各子目录说明

### 01_models/
- `all_14_variants/`：14 个 `yolo11{n,s,m,l,x}_..._batch_{4,8,16}_final.pt`（x_batch_16 未训练）+ `manifest.csv`
- 仅含 `*_final.pt` 主版本；`_v1.pt` / `_v2.pt` 重跑备份见 `source_pointer.txt`

### 02_training_runs/
- 14 个子目录按 `yolo11{size}_batch_{batch}` 简化命名
- 每个子目录仅保留 9 个论文用文件：
  - `results.csv` — **§3.1 Table 1 与 Kruskal-Wallis 的直接数据源**
  - `args.yaml` — 训练超参
  - `results.png`, `BoxF1/PR/P/R_curve.png`, `confusion_matrix(_normalized).png` — 曲线与混淆矩阵
- **未复制**：`weights/` 已在 01_models 独立归档；`train_batch*.jpg`/`val_batch*.jpg`/`labels*.jpg` 为数据可视化样例，可从 01_data 重生

### 03_counting_validation/
- `evaluation_sahi_null_v2_ms1_0605-0621_40_ok.xlsx`：36 张 ms2 站点田间图像的 AI 计数 vs 人工巡检对比
  - ⚠ 文件名含 `ms1` 但实际数据为 ms2 站点（历史命名遗留）
  - 使用模型：`yolo11n_..._batch_8_final.pt`，conf=0.40
- `evaluation_to_excel.ipynb`：从 fiftyone 导出为 xlsx 的 notebook
- `ty_eval_tools/`：评估工具代码

### 04_timeseries_population/
- `notebooks/`：4 个预处理与时序聚合 notebook（见 source_pointer.txt 详解）
- `csv/`：
  - `ms2_0605_0923.csv` / `ms2_0605_0923_proc.csv` — ms2 站点 2024 生长季
  - `2025_north_v1__{lloyd2,southfarm1,southfarm2}_count_over_time_*.csv` — 2025 跨年 3 站点
  - `table_3_per_timepoint_fused_max.csv` — **§3.3 主时序数据源**（多焦距融合）

### 05_plotting/
- `34_draw_plot.ipynb`：主绘图 notebook
- `37_draw_plot.ipynb`：备用
- `ty_plot_tools/`：样式与辅助函数

### 06_edge_deployment/
- `convert/export_ncnn{,2}.py`：PyTorch → NCNN 导出脚本
- `predict/`：RPi4 端推理脚本（ncnn + pt 对比基线 + sahi 验证 notebook + 样例图）
- `ncnn_models_pointer.txt`：指向 437 MB 的 NCNN 二进制目录

### 07_large_data_pointers/
- `01_data_pointer.txt` — 207 G 原始图像数据集
- `03_code_pointer.txt` — 149 G 杂项代码
- `04_final_pipeline_pointer.txt` — 143 G final pipeline

---

## 与论文的交叉引用

| 论文位置 | 本目录对应 |
|---|---|
| §2.5 HRSI 切片 | `06_edge_deployment/predict/sahi.ipynb` |
| §2.8 评估指标 | `03_counting_validation/ty_eval_tools/` |
| §3.1 Table 1 | `02_training_runs/*/results.csv` |
| §3.1 Fig 4 | `05_plotting/34_draw_plot.ipynb` + `02_training_runs/` |
| §3.1 Kruskal-Wallis | `paper1/04_results/4a_model_performance/scripts/kruskal_saturation_test.py`（输入来自 `02_training_runs/`） |
| §3.2 Fig 6 / Table 2 | `03_counting_validation/evaluation_sahi_null_v2_ms1_0605-0621_40_ok.xlsx` |
| §3.3 Fig 7 | `04_timeseries_population/csv/table_3_per_timepoint_fused_max.csv` |
| §4.3 Fig 8 | `04_timeseries_population/csv/2025_north_v1__*.csv` |
| §3.4 Table 3 | `06_edge_deployment/` + `paper1/04_results/4d_edge_computing/scripts/` |

---

## 维护约定

- 本目录**只增不删**：新增实验输出可加子目录，但不应移动或删除现有文件
- 所有 `*_pointer.txt` 在原路径变动后需同步更新
- 如果某个子目录在论文定稿后不再需要，移动到 `99_archive/` 而非删除

---

## 🗂️ 完整路径映射表（当前路径 ↔ 原路径）

> **用途**：每一条记录都是一次 `cp` 操作的溯源。Master 在用任何文件前都可以查这张表，**核对自己拿的是不是最新版**。
> 所有"原路径"的根 = `/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/`（下面简写为 `$ROOT`）
> 所有"当前路径"的根 = `$ROOT/05_paper_material/`（下面简写为 `$HERE`）

### 01_models/ — 模型权重（共 14 个 .pt + 1 个 manifest）

| 当前路径 | ← | 原路径 |
|---|---|---|
| `$HERE/01_models/all_14_variants/yolo11n_..._batch_4_final.pt` | ← | `$ROOT/02_fine-tuned_checkpoint/best_models/04_swd_hbb/null_image_trained_final_checkpoint/yolo11n_20pct_null_images_add_rawData_batch_4_final.pt` |
| `$HERE/01_models/all_14_variants/yolo11n_..._batch_8_final.pt` | ← | `$ROOT/02_fine-tuned_checkpoint/.../yolo11n_20pct_null_images_add_rawData_batch_8_final.pt` |
| `$HERE/01_models/all_14_variants/yolo11n_..._batch_16_final.pt` | ← | `$ROOT/02_fine-tuned_checkpoint/.../yolo11n_20pct_null_images_add_rawData_batch_16_final.pt` |
| `$HERE/01_models/all_14_variants/yolo11s_..._batch_{4,8,16}_final.pt` | ← | `$ROOT/02_fine-tuned_checkpoint/.../yolo11s_20pct_null_images_add_rawData_batch_{4,8,16}_final.pt` |
| `$HERE/01_models/all_14_variants/yolo11m_..._batch_{4,8,16}_final.pt` | ← | `$ROOT/02_fine-tuned_checkpoint/.../yolo11m_20pct_null_images_add_rawData_batch_{4,8,16}_final.pt` |
| `$HERE/01_models/all_14_variants/yolo11l_..._batch_{4,8,16}_final.pt` | ← | `$ROOT/02_fine-tuned_checkpoint/.../yolo11l_20pct_null_images_add_rawData_batch_{4,8,16}_final.pt` |
| `$HERE/01_models/all_14_variants/yolo11x_..._batch_{4,8}_final.pt` | ← | `$ROOT/02_fine-tuned_checkpoint/.../yolo11x_20pct_null_images_add_rawData_batch_{4,8}_final.pt`（x_batch_16 未训练） |
| `$HERE/01_models/all_14_variants/manifest.csv` | ← | `$ROOT/02_fine-tuned_checkpoint/best_models/04_swd_hbb/null_image_trained_final_checkpoint/manifest.csv` |

**⚠ 版本提醒**：原目录下每个 `_final.pt` 都有 `_v1.pt` / `_v2.pt` 两份备份（重跑不同 seed）。本目录只复制 `_final.pt` 主版本。如果 Master 怀疑主版本用错，请直接对比原目录下三份文件的 mtime 与大小（或查 `manifest.csv`）。

### 02_training_runs/ — 训练输出（14 runs × 9 files = 126 文件）

**规则**：`$HERE/02_training_runs/yolo11{size}_batch_{batch}/{file}` ← `$ROOT/00_pipeline/02_current_batch_run_model/02_TrainingModel/output/swd_model_v5_nullImagesAdded_final_noAug_seed42/yolo11{size}.pt_20pct_null_images_add_rawData_list_train_val_test_{batch}/{file}`

| 本地子目录 | 原子目录名 |
|---|---|
| `yolo11n_batch_4` | `yolo11n.pt_20pct_null_images_add_rawData_list_train_val_test_4` |
| `yolo11n_batch_8` | `yolo11n.pt_..._list_train_val_test_8` |
| `yolo11n_batch_16` | `yolo11n.pt_..._list_train_val_test_16` |
| `yolo11s_batch_{4,8,16}` | `yolo11s.pt_..._list_train_val_test_{4,8,16}` |
| `yolo11m_batch_{4,8,16}` | `yolo11m.pt_..._list_train_val_test_{4,8,16}` |
| `yolo11l_batch_{4,8,16}` | `yolo11l.pt_..._list_train_val_test_{4,8,16}` |
| `yolo11x_batch_{4,8}` | `yolo11x.pt_..._list_train_val_test_{4,8}` |

**每个子目录内的 9 个白名单文件**（其余未复制）：
`results.csv` / `args.yaml` / `results.png` / `BoxF1_curve.png` / `BoxPR_curve.png` / `BoxP_curve.png` / `BoxR_curve.png` / `confusion_matrix.png` / `confusion_matrix_normalized.png`

**⚠ 版本提醒**：训练 run 目录名 `swd_model_v5_nullImagesAdded_final_noAug_seed42` 是**最终版**（final + seed42）。同级还有 `swd_model_v5_nullImagesAdded_final_test_only` 是测试性产出，本目录**未复制**。如果 Master 在原目录看到多个 run 目录，请确认用的是带 `seed42` 的那个。

### 03_counting_validation/

| 当前路径 | ← | 原路径 |
|---|---|---|
| `$HERE/03_counting_validation/evaluation_sahi_null_v2_ms1_0605-0621_40_ok.xlsx` | ← | `$ROOT/00_pipeline/02_current_batch_run_model/04_evaluate_Real-world_Performance_in_fiftyone/evaluation_sahi_null_v2_ms1_0605-0621_40_ok.xlsx` |
| `$HERE/03_counting_validation/evaluation_to_excel.ipynb` | ← | `$ROOT/00_pipeline/02_current_batch_run_model/04_evaluate_Real-world_Performance_in_fiftyone/evaluation_to_excel.ipynb` |
| `$HERE/03_counting_validation/ty_eval_tools/` | ← | `$ROOT/00_pipeline/02_current_batch_run_model/04_evaluate_Real-world_Performance_in_fiftyone/ty_eval_tools/` |

**⚠ 版本提醒**：
1. 文件名含 `ms1` 但**实际数据为 ms2 站点**（历史命名遗留）。
2. `v2` / `null_v2` 表示用"加了 null image 训练的第 2 版模型"导出的评估。如果原目录出现 `_v1` / `_v3` 后续版本，请确认 Master 需要的是哪一版。
3. 评估中使用的模型是 `yolo11n_..._batch_8_final.pt`，conf=0.40。**不是** §3.1 饱和论证里的 n_batch_4。这一点在 §3.2 重跑 5 变体时要注意。

### 04_timeseries_population/

| 当前路径 | ← | 原路径 |
|---|---|---|
| `$HERE/04_timeseries_population/notebooks/01_image_filename_timestamp_capture_log_and_daily_focus_statistics.ipynb` | ← | `$ROOT/00_pipeline/03_current_run_one/01_image_filename_timestamp_capture_log_and_daily_focus_statistics.ipynb` |
| `$HERE/04_timeseries_population/notebooks/02_sensor_csv_preprocessing_daily_aggregation_and_temperature_humidity_visualization.ipynb` | ← | `$ROOT/00_pipeline/03_current_run_one/02_sensor_csv_preprocessing_daily_aggregation_and_temperature_humidity_visualization.ipynb` |
| `$HERE/04_timeseries_population/notebooks/03_fiftyone_prediction_export_statistics_time_focus_aggregation_and_multifocus_fusion.ipynb` | ← | `$ROOT/00_pipeline/03_current_run_one/03_fiftyone_prediction_export_statistics_time_focus_aggregation_and_multifocus_fusion.ipynb` |
| `$HERE/04_timeseries_population/notebooks/11_compare_gt_with_swd.ipynb` | ← | `$ROOT/00_pipeline/03_current_run_one/11_compare_gt_with_swd.ipynb` |
| `$HERE/04_timeseries_population/csv/ms2_0605_0923.csv` | ← | `$ROOT/00_pipeline/03_current_run_one/ms2_0605_0923.csv` |
| `$HERE/04_timeseries_population/csv/ms2_0605_0923_proc.csv` | ← | `$ROOT/00_pipeline/03_current_run_one/ms2_0605_0923_proc.csv` |
| `$HERE/04_timeseries_population/csv/2025_north_v1__lloyd2_count_over_time_conf70_focus775.csv` | ← | `$ROOT/00_pipeline/03_current_run_one/2025_north_v1__lloyd2_count_over_time_conf70_focus775.csv` |
| `$HERE/04_timeseries_population/csv/2025_north_v1__southfarm1_count_over_time_conf70_focus700.csv` | ← | `$ROOT/00_pipeline/03_current_run_one/2025_north_v1__southfarm1_count_over_time_conf70_focus700.csv` |
| `$HERE/04_timeseries_population/csv/2025_north_v1__southfarm2_count_over_time_conf70_focus600.csv` | ← | `$ROOT/00_pipeline/03_current_run_one/2025_north_v1__southfarm2_count_over_time_conf70_focus600.csv` |
| `$HERE/04_timeseries_population/csv/table_3_per_timepoint_fused_max.csv` | ← | `$ROOT/00_pipeline/03_current_run_one/_deploy_exports/sahi_null_run_whole_rawData_v5/sahi_null_run_whole_rawData_v5_ms2_0605_0923/pred_yolo11n_20pct_null_images_add_rawData_batch_8_final/table_3_per_timepoint_fused_max.csv` |

**⚠ 版本提醒**：
1. `table_3_per_timepoint_fused_max.csv` 是用 **`yolo11n_batch_8_final`** 预测的——与 §3.2 计数验证同一个模型。如果 §3.1 的"饱和论证"把部署模型锁到 `n_batch_4`，§3.3 这个时序**严格来说用的是 n_batch_8**，需要在论文中核对一致性（或重跑 n_batch_4 的时序）。
2. `2025_north_v1_` 前缀表示 2025 年北区第一版部署；如果原目录后续出现 `2025_north_v2_`，说明有新版本。
3. 每个 2025 csv 文件名里含 conf/focus 参数（`conf70_focus{600,700,775}`），与论文 §4.3 中的设置需对照。

### 05_plotting/

| 当前路径 | ← | 原路径 |
|---|---|---|
| `$HERE/05_plotting/34_draw_plot.ipynb` | ← | `$ROOT/00_pipeline/02_current_batch_run_model/06_draw_plot/34_draw_plot.ipynb` |
| `$HERE/05_plotting/37_draw_plot.ipynb` | ← | `$ROOT/00_pipeline/02_current_batch_run_model/06_draw_plot/37_draw_plot.ipynb` |
| `$HERE/05_plotting/ty_plot_tools/` | ← | `$ROOT/00_pipeline/02_current_batch_run_model/06_draw_plot/ty_plot_tools/` |

**⚠ 版本提醒**：`34_` 与 `37_` 两个编号暗示可能还有 `35_` / `36_` 中间版本被跳过。如果 Master 发现这两个 notebook 里的代码不对得上，请回原目录查是否有遗漏。

### 06_edge_deployment/

| 当前路径 | ← | 原路径 |
|---|---|---|
| `$HERE/06_edge_deployment/convert/export_ncnn.py` | ← | `$ROOT/00_pipeline/06_rpi/convert/export_ncnn.py` |
| `$HERE/06_edge_deployment/convert/export_ncnn2.py` | ← | `$ROOT/00_pipeline/06_rpi/convert/export_ncnn2.py` |
| `$HERE/06_edge_deployment/predict/yolo_ncnn.py` | ← | `$ROOT/00_pipeline/06_rpi/predict/yolo_ncnn.py` |
| `$HERE/06_edge_deployment/predict/yolo_pt.py` | ← | `$ROOT/00_pipeline/06_rpi/predict/yolo_pt.py` |
| `$HERE/06_edge_deployment/predict/sahi.ipynb` | ← | `$ROOT/00_pipeline/06_rpi/predict/sahi.ipynb` |
| `$HERE/06_edge_deployment/predict/0727_0836_640.jpg` | ← | `$ROOT/00_pipeline/06_rpi/predict/0727_0836_640.jpg` |
| `$HERE/06_edge_deployment/ncnn_models_pointer.txt` → | → | `$ROOT/00_pipeline/06_rpi/convert/ncnn_models/`（437 M，未复制） |

**⚠ 版本提醒**：
1. `export_ncnn.py` 与 `export_ncnn2.py` 并存——后者是第 2 版导出逻辑，Master 用前需确认哪个对应 §3.4 的最终结果。
2. 原 `predict/` 目录里还有 `yolo_ncnn copy.py` 与 `yolo_pt copy.py`（带空格的副本文件），本目录**未复制**。如果主文件和副本内容不同，可能意味着正在开发中的修改，请 Master 自行核对。

### 07_large_data_pointers/（仅 pointer，未复制）

| 当前 pointer | → | 原路径 |
|---|---|---|
| `$HERE/07_large_data_pointers/01_data_pointer.txt` | → | `$ROOT/01_data/` （207 G） |
| `$HERE/07_large_data_pointers/03_code_pointer.txt` | → | `$ROOT/03_code/` （149 G） |
| `$HERE/07_large_data_pointers/04_final_pipeline_pointer.txt` | → | `$ROOT/04_final_pipeline/` （143 G） |

---

## 🔍 如何核对"是不是用对了版本"

1. **对每个文件**：在表中找到原路径 → `ls -l <原路径>` 看 mtime → 与本地 `ls -l $HERE/<对应路径>` 对比，mtime/大小一致说明本地是当时的快照
2. **对 14 个 .pt**：查 `01_models/all_14_variants/manifest.csv` 里每行的训练元信息（超参、epoch、最终 metric）
3. **对训练 run**：查 `02_training_runs/yolo11*/args.yaml` 里的 `project` / `name` 字段，应含 `swd_model_v5_nullImagesAdded_final_noAug_seed42`
4. **对计数 xlsx**：文件名末尾的 `v2` / `null_v2` / `ms1_0605-0621_40_ok` 是版本标识，改动即会变化
5. **对时序 csv**：`table_3_per_timepoint_fused_max.csv` 的生成路径里嵌入了模型名 `pred_yolo11n_..._batch_8_final`，直接体现了用哪个模型预测的

**如果发现本地快照与原路径不一致**（例如 Master 后续在原目录重跑了训练覆盖了老的 run），有两个选择：
- (a) 更新本目录对应文件，并在 README 末尾加一行"YYYY-MM-DD: 更新 xxx"
- (b) 把旧版本移到 `99_archive/`，保留新旧两份以便回溯
