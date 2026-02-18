# SWD智能杂草检测计算机视觉项目

## 项目简介

基于YOLOv11和FiftyOne的农业SWD（Smart Weed Detection）计算机视觉项目，专注于自动化杂草检测和识别。

### 项目目标
- 实现高精度的杂草检测和识别
- 支持多种检测模型（分类、HBB、OBB）
- 提供完整的数据处理流水线
- 集成FiftyOne可视化分析工具

### 技术栈
- **深度学习框架**: YOLOv11 (Ultralytics)
- **数据可视化**: FiftyOne
- **数据处理**: COCO格式、YOLO格式
- **模型训练**: PyTorch
- **实验跟踪**: Weights & Biases (wandb)

## 目录结构

```
_ty/
├── README.md                          # 本文件
├── 00_pipeline/                       # 数据处理流水线
│   ├── 00_temp/                       # 临时工作目录
│   ├── 01_README/                     # 项目文档和笔记本
│   ├── 02_current_batch_run/          # 当前批次运行工作流
│   │   ├── 01_PrepareTrainingData__fiftyone_to_trainingData/  # 数据准备流水线
│   │   ├── 02_TrainingModel/          # 模型训练流水线
│   │   ├── 03_run_checkpoint_predict_raw_image_SAHI_way_and_evaluate/  # 模型预测和评估
│   │   ├── 04_evaluate_Real-world_Performance​_in_fiftyone/  # 实际场景评估
│   │   ├── 05_evaluate_each_image_in_fiftyone/  # 每张图片评估
│   │   ├── 06_draw_plot/               # 可视化绘图
│   │   └── 07_no_GT_run/               # 无真实标签运行
│   ├── 03_current_run_one/            # 单次运行流程
│   ├── 05_trained_model_predict_raw_images/  # 训练后模型预测
│   └── 99_archive/                    # 归档历史记录
├── 01_data/                           # 数据集目录
│   ├── a01_20dataset/                 # 2024年原始数据 (20组数据)
│   ├── a02_16mp_2024_datasets_fiftyone/  # 1600万像素处理数据
│   ├── a03_raw_data/                  # 原始数据
│   ├── a04_whole_raw_data/            # 完整原始数据
│   ├── a05_2024_data/                 # 2024年数据
│   └── test/                          # 测试数据
├── 02_fine-tuned_checkpoint/          # 训练好的模型
│   └── best_models/
│       ├── 01_swd_seg/                # 分割模型
│       ├── 02_swd_pose/               # 姿态估计模型
│       ├── 03_dot_det/                # 点检测模型
│       ├── 04_swd_hbb/                # 水平边框检测模型
│       ├── 05_swd_obb/                # 旋转边框检测模型
│       ├── 11_swd_seg_64mp/           # 6400万像素分割模型
│       └── 91_swd_cls/                # 分类模型
├── 03_code/                           # 代码目录
│   ├── 03_train_model/                # 模型训练代码
│   │   ├── yolo_cls/                  # 分类模型训练
│   │   ├── yolo11_hbb/                # 水平边框检测训练
│   │   ├── yolo11_hbb_old/            # 旧版检测训练
│   │   └── yolo_obb/                  # 旋转边框检测训练
│   └── 04_fiftyone/                   # FiftyOne可视化分析
│       └── tools/                     # 分析工具
└── .claude/                           # Claude配置目录
```

## 主要模块说明

### 00_pipeline - 数据处理流水线

数据处理流水线是项目的核心，负责从原始数据到训练数据的完整转换流程。

#### 00_temp - 临时工作目录
- 用于数据预处理中间文件
- 存储导出的COCO JSON文件
- 存储640子图和空图像

#### 01_README - 项目文档
- 项目说明文档
- 数据分析笔记本
- 可视化图表

#### 02_current_batch_run - 当前批次运行工作流
包含完整的数据处理和模型训练流程：

1. **01_PrepareTrainingData__fiftyone_to_trainingData**: 数据准备流水线
   - 从FiftyOne导出数据
   - 生成640子图
   - NMS去重处理
   - 构建空图像数据集

2. **02_TrainingModel**: 模型训练流水线
   - 模型训练脚本
   - 训练配置管理
   - 训练结果记录

3. **03_run_checkpoint_predict_raw_image_SAHI_way_and_evaluate**: 模型预测和评估
   - 使用SAHI进行预测
   - 模型性能评估
   - 结果分析

4. **04_evaluate_Real-world_Performance​_in_fiftyone**: 实际场景评估
   - 在实际场景中评估模型
   - 真实世界性能分析

5. **05_evaluate_each_image_in_fiftyone**: 每张图片评估
   - 细粒度图像级评估
   - 错误分析

6. **06_draw_plot**: 可视化绘图
   - 结果可视化
   - 图表生成

7. **07_no_GT_run**: 无真实标签运行
   - 无标签数据推理
   - 预测结果导出

#### 03_current_run_one - 单次运行流程
- 单次运行的完整流程
- 包含部署导出功能

#### 05_trained_model_predict_raw_images - 训练后模型预测
- 使用训练好的模型进行预测
- 配置文件管理
- 预测结果处理

#### 99_archive - 归档历史记录
- 历史数据归档
- 旧模型归档

### 01_data - 数据集目录

#### a01_20dataset - 2024年原始数据
包含20组数据，按卡片分组：
- air2_0729-0813_04
- jeff_0613-0624_04_ok
- jeff_0624-0702_01_ok
- lloyd_0715-0729_04
- ms1_0605-0621_40
- ms1_0710-0726_36
- ms1_0726-0809_11
- ms1_0809-0823_34
- ms2_0605-0621_30
- ms2_0621-0710_01
- ms2_0710-0726_41
- ms2_0726-0809_13
- ms2_0809-0823_10
- ms2_0823-0906_07
- sw1_0605-0613_07_ok
- sw1_0627-0711_02
- sw1_0711-0725_03
- sw1_0808-0823_01
- sw2_0725-0808_02
- sw2_0808-0823_04

#### a02_16mp_2024_datasets_fiftyone - 1600万像素处理数据
处理后的数据集，包含7组训练数据：
- ms1_0605-0621_40_ok
- ms1_0710-0726_36_ok
- ms1_0726-0809_11_ok
- ms1_0809-0823_34_ok
- ms2_0726-0809_13_ok
- ms2_0809-0823_10_ok
- sw1_0605-0613_07_ok

#### a03_raw_data - 原始数据
原始数据备份，包含所有数据组。

#### a04_whole_raw_data - 完整原始数据
完整原始数据，用于实际场景评估。

#### a05_2024_data - 2024年数据
2024年处理后的数据。

#### test - 测试数据
测试数据集。

### 02_fine-tuned_checkpoint - 训练好的模型

#### best_models - 最佳模型检查点
按模型类型分类存储：

1. **01_swd_seg**: 分割模型
   - 杂草分割模型
   - 基于YOLO分割任务

2. **02_swd_pose**: 姿态估计模型
   - 杂草姿态估计
   - 关键点检测

3. **03_dot_det**: 点检测模型
   - 点状目标检测
   - 微小目标检测

4. **04_swd_hbb**: 水平边框检测模型
   - 水平边界框检测
   - 标准目标检测

5. **05_swd_obb**: 旋转边框检测模型
   - 旋转边界框检测
   - 适用于倾斜目标

6. **11_swd_seg_64mp**: 6400万像素分割模型
   - 高分辨率分割模型
   - 6400万像素数据训练

7. **91_swd_cls**: 分类模型
   - 杂草分类模型
   - 二分类/多分类

### 03_code - 代码目录

#### 03_train_model - 模型训练代码

##### yolo_cls - 分类模型训练
- **01_01_split_train_dataset_for_cls.ipynb**: 数据集分割
- **01_02_main_train_And_test.py**: 分类模型训练主程序
- **01_03_export_ONNX_model.ipynb**: ONNX模型导出
- **01_04_pred.py**: 分类模型推理
- **01_05_predonnx.py**: ONNX格式推理
- **02_grad_cam.ipynb**: Grad-CAM可视化

##### yolo11_hbb - 水平边框检测训练
- **01_01_split_train_dataset_for_seg.ipynb**: 数据集分割
- **01_02_main_train_And_test.ipynb**: HBB模型训练
- **01_03_export_ONNX_model.ipynb**: ONNX模型导出
- **01_04_eval.ipynb**: 模型评估
- **01_05_add_null_image_label.ipynb**: 添加空图像标签
- **train.py**: HBB训练脚本

##### yolo_obb - 旋转边框检测训练
- **01_01_split_train_dataset_for_seg.ipynb**: 数据集分割
- **01_02_main_train_And_test.ipynb**: OBB模型训练
- **01_03_export_ONNX_model.ipynb**: ONNX模型导出
- **train.py**: OBB训练脚本

#### 04_fiftyone - FiftyOne可视化分析

##### tools - 分析工具
- **fiftyone_embedding.py**: 嵌入分析工具
- **fiftyone_embedding_patch.py**: Patch嵌入分析工具

## 关键入口文件和脚本

### 数据准备入口

| 文件路径 | 说明 | 使用场景 |
|---------|------|----------|
| `00_pipeline/02_current_batch_run/01_PrepareTrainingData__fiftyone_to_trainingData/01_fiftyone_current_view_export_640_sub_images_and_labels.ipynb` | 从FiftyOne导出640子图和标签 | 数据预处理、生成训练数据 |
| `00_pipeline/02_current_batch_run/01_PrepareTrainingData__fiftyone_to_trainingData/02_x-anylableing.ipynb` | X-Annotable标注处理 | 标签转换和处理 |

### 模型训练入口

| 文件路径 | 说明 | 模型类型 |
|---------|------|----------|
| `03_code/03_train_model/yolo_cls/01_02_main_train_And_test.py` | 分类模型训练主程序 | 分类 (Classification) |
| `03_code/03_train_model/yolo11_hbb/train.py` | 水平边框检测训练 | HBB (Horizontal Bounding Box) |
| `03_code/03_train_model/yolo_obb/train.py` | 旋转边框检测训练 | OBB (Oriented Bounding Box) |

### 评估和推理入口

| 文件路径 | 说明 | 使用场景 |
|---------|------|----------|
| `03_code/03_train_model/yolo_cls/01_04_pred.py` | 分类模型推理 | 模型预测 |
| `03_code/03_train_model/yolo_cls/01_05_predonnx.py` | ONNX格式推理 | 模型部署 |
| `00_pipeline/03_current_run_one/03_get_predict_excel_from_fiftyone.ipynb` | 从FiftyOne获取预测Excel | 结果导出 |

### 五维可视化分析入口

| 文件路径 | 说明 | 分析功能 |
|---------|------|----------|
| `03_code/04_fiftyone/tools/fiftyone_embedding.py` | 嵌入分析工具 | 特征可视化 |
| `03_code/04_fiftyone/tools/fiftyone_embedding_patch.py` | Patch嵌入分析 | 目标区域分析 |

## 自定义工具库 (ty_fo_tools)

ty_fo_tools是项目核心工具库，位于：
`00_pipeline/02_current_batch_run/01_PrepareTrainingData__fiftyone_to_trainingData/ty_fo_tools/`

### 导入方式
```python
import ty_fo_tools as ty
```

### 主要功能模块

#### 1. FiftyOne导出工具 (fiftyone/export_view.py)
- **export_view_to_coco**: 将FiftyOne视图导出为COCO格式
```python
ty.export_view_to_coco(
    view=current_view,
    export_dir=export_dir,
    label_field="pred_field",
    dataset_type=fot.COCODetectionDataset,
    export_media=True
)
```

#### 2. COCO数据处理工具 (cocoData/tiles.py)
- **TileSpec**: 瓦片规格配置类
- **export_labeled_tiles_from_coco**: 从COCO导出标记瓦片
- **export_null_images_tiles_from_coco**: 导出空图像瓦片

#### 3. COCO NMS工具 (cocoData/nms.py)
- **coco_nms_json**: COCO JSON非极大值抑制
- **_bbox_iou**: 计算边界框IoU

#### 4. YOLO数据集构建工具 (yoloData/build_trainning_dataset.py)
- **build_yolo_null_images_dataset**: 构建空图像数据集
- **YoloPair**: YOLO图像-标签对类

## 工作流程指南

### 整体工作流程
```
FiftyOne数据集 → 数据导出 → 数据预处理 → 模型训练 → 模型评估 → 可视化分析
    ↓           ↓          ↓         ↓         ↓         ↓
  原始标注   COCO格式   640子图    训练脚本   mAP/F1    embedding
              导出     生成       执行      评估      可视化
```

### 详细工作流步骤

#### 步骤1: 数据准备 (Data Preparation)
```
00_pipeline/02_current_batch_run/01_PrepareTrainingData__fiftyone_to_trainingData/
├── 01_fiftyone_current_view_export_640_sub_images_and_labels.ipynb
│   ├── 1. 从FiftyOne导出原图和标签到COCO格式
│   ├── 2. 原图切成640子图和对应标签
│   ├── 3. NMS去除重叠标注
│   ├── 4. 导出640子图的null images和空标签
│   └── 5. 随机抽取k对null images
└── 02_x-anylableing.ipynb
    └── X-Annotable标注文件处理
```

#### 步骤2: 模型训练 (Model Training)
根据模型类型选择不同的训练脚本：

1. **分类模型**: `03_code/03_train_model/yolo_cls/01_02_main_train_And_test.py`
   - 支持数据集分割比例配置
   - 支持多batch size实验
   - 支持不同YOLO模型系列 (yolo11n-cls, yolo11s-cls, yolo11m-cls)

2. **HBB检测模型**: `03_code/03_train_model/yolo11_hbb/train.py`
   - 使用YAML配置文件定义数据集
   - 支持多数据集组合训练
   - 支持wandb实验跟踪

3. **OBB检测模型**: `03_code/03_train_model/yolo_obb/train.py`
   - 旋转边框检测
   - 适用于倾斜目标检测

#### 步骤3: 模型评估和推理 (Evaluation & Inference)
```
关键入口文件：
1. 分类模型推理: 03_code/03_train_model/yolo_cls/01_04_pred.py
2. ONNX模型推理: 03_code/03_train_model/yolo_cls/01_05_predonnx.py
3. 实际场景评估: 00_pipeline/03_current_run_one/03_get_predict_excel_from_fiftyone.ipynb
```

#### 步骤4: FiftyOne可视化分析 (Visualization Analysis)
```
03_code/04_fiftyone/tools/
├── fiftyone_embedding.py      # 嵌入分析工具
│   ├── CLIP嵌入
│   ├── DINOv2嵌入
│   ├── ResNet50嵌入
│   ├── MobileNet嵌入
│   └── UMAP/t-SNE降维可视化
└── fiftyone_embedding_patch.py # Patch嵌入分析
```

## 快速开始指南

### 1. 环境要求
```bash
# 核心依赖
ultralytics == 8.3.178
fiftyone >= 0.15.0
pytorch >= 1.13
python >= 3.8
```

### 2. 数据准备流程

#### 步骤1: 启动FiftyOne查看数据
```python
import fiftyone as fo

# 加载数据集
dataset = fo.load_dataset(fo.list_datasets()[0])

# 启动GUI
session = fo.launch_app(dataset, port=5151)
```

#### 步骤2: 运行数据预处理笔记本
在Jupyter中执行：
```
00_pipeline/02_current_batch_run/01_PrepareTrainingData__fiftyone_to_trainingData/
01_fiftyone_current_view_export_640_sub_images_and_labels.ipynb
```

关键参数设置：
```python
export_dir = Path("00_pipeline/00_temp")  # 导出目录
current_view = session.view               # 当前视图
spec = ty.TileSpec(crop_size=640, overlap_ratio=0.2, keep_ratio=0.9)
```

### 3. 模型训练流程

#### 分类模型训练
```bash
# 运行Python脚本
cd 03_code/03_train_model/yolo_cls
python 01_02_main_train_And_test.py
```

#### 检测模型训练
```bash
# 运行Python脚本
cd 03_code/03_train_model/yolo11_hbb
python train.py
```

#### 使用YAML配置文件
模型训练使用YAML配置文件定义数据集路径：
```yaml
# 示例配置文件结构
path: /path/to/dataset
train: images/train
val: images/val
names:
  0: swd
  1: ok
  2: others
```

### 4. 模型评估流程

#### 设置置信度阈值
支持多阈值评估：[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93]

#### 关键评估指标
- **mAP**: 平均精度均值
- **F1-score**: 精确率和召回率的调和平均值
- **PR Curve**: 精确率-召回率曲线

### 5. 可视化分析流程

#### 嵌入分析步骤
```python
# 运行嵌入分析脚本
cd 03_code/04_fiftyone/tools
python fiftyone_embedding.py
```

分析流程：
1. 选择嵌入模型 (CLIP, DINOv2, ResNet50, MobileNet)
2. 计算patch嵌入
3. 降维可视化 (UMAP/t-SNE)
4. 在FiftyOne中查看结果

### 6. 常用命令速查

#### 数据处理
```python
# 导出FiftyOne视图到COCO
ty.export_view_to_coco(view, export_dir, label_field)

# 生成训练瓦片
ty.export_labeled_tiles_from_coco(img_dir, coco_json, out_dir, spec)

# NMS去重
ty.coco_nms_json(input_json, output_json, iou_thresh=0.5)

# 构建空图像数据集
ty.build_yolo_null_images_dataset(src_images_dir, src_labels_dir, out_dir, k=200)
```

#### 模型训练
```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n.pt")  # 预训练模型

# 训练模型
results = model.train(
    data="path/to/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    project="swd_model",
    name="best_model"
)

# 评估模型
metrics = model.val()
```

#### 模型推理
```python
# 单张图片推理
results = model("path/to/image.jpg")

# 批量推理
results = model(["image1.jpg", "image2.jpg"])

# 导出ONNX
model.export(format="onnx")
```

## 数据预处理工作流详解

### 输入数据格式
- **原始数据**: 位于`01_data/a01_20dataset/`
- **处理后数据**: 位于`01_data/a02_16mp_2024_datasets_fiftyone/`

### 子图生成参数
- **瓦片大小**: 640x640像素
- **重叠率**: 20%
- **保持比例**: 90%
- **NMS阈值**: 0.5 IoU

### 数据分割方案
| 分割比例 | 训练集 | 验证集 | 测试集 | 说明 |
|---------|--------|--------|--------|------|
| 标准分割 | 60% | 40% | 0% | 常用配置 |
| 严格分割 | 80% | 20% | 0% | 大训练集 |
| 三路分割 | 50% | 30% | 20% | 包含测试集 |

## 模型类型说明

### 分类模型 (Classification)
- **用途**: 判断图像中是否包含杂草
- **模型**: YOLO分类模型 (yolo11n-cls, yolo11s-cls, yolo11m-cls)
- **输出**: 二分类或多分类结果

### 水平边框检测 (HBB)
- **用途**: 检测杂草的水平边界框
- **模型**: YOLO检测模型 (yolo11n, yolo11s, yolo11m)
- **输出**: [x1, y1, x2, y2, class_id]

### 旋转边框检测 (OBB)
- **用途**: 检测杂草的旋转边界框
- **模型**: YOLO OBB模型 (yolo11n-obb, yolo11s-obb)
- **输出**: [x1, y1, x2, y2, x3, y3, x4, y4, class_id]

### 分割模型 (Segmentation)
- **用途**: 精确分割杂草区域
- **模型**: YOLO分割模型
- **输出**: 掩码和边界框

## 故障排除和最佳实践

### 常见问题

#### 1. FiftyOne连接失败
```python
# 解决方案
import fiftyone as fo

# 重新启动FiftyOne
session = fo.launch_app(dataset, port=5151)

# 检查端口占用
# netstat -tulpn | grep 5151
```

#### 2. 内存不足错误
```python
# 减少batch size
batch_size = 4  # 从16减少到4

# 减小图片尺寸
imgsz = 320  # 从640减小到320

# 使用梯度累积
model.train(..., batch=8, accumulate=2)  # 实际batch=16
```

#### 3. 数据集路径错误
```bash
# 确保路径正确
ls /path/to/your/dataset  # 检查目录存在

# 使用绝对路径
data_path = "/home/tianqi/.../dataset.yaml"
```

#### 4. 内存泄露问题
```python
# 定期清理内存
import gc
gc.collect()

# 在循环中定期调用
for epoch in range(epochs):
    # 训练代码...
    if epoch % 10 == 0:
        gc.collect()
```

### 最佳实践

#### 1. 数据管理
- **保持原始数据副本**: 始终保留原始数据的备份
- **版本控制**: 对处理后的数据添加版本号
- **文档记录**: 记录每次数据处理的参数和结果

#### 2. 模型训练
- **实验记录**: 使用wandb记录所有实验
- **超参数调优**: 系统化尝试不同的超参数组合
- **模型版本管理**: 对训练好的模型进行版本控制

#### 3. 评估策略
- **多阈值评估**: 在多个置信度阈值下评估模型
- **实际场景测试**: 在真实场景中测试模型性能
- **误分析**: 分析模型的错误案例，指导改进

#### 4. 代码组织
- **注释完整**: 关键代码添加详细注释
- **模块化设计**: 将功能封装成可重用的函数/类
- **错误处理**: 添加适当的异常处理

## 项目历史和版本

### 数据版本
- **v1**: 原始数据集，精炼，4+3组数据
- **v2**: 扩大数据量，3&4dataset
- **v3**: 扩大数据集，7datasets
- **v4**: 加入null_images，7datasets

### 模型版本
- **第一版**: 手动标记，Roboflow，759张含SWD的640子图
- **第二版**: 扩大数据量，3&4dataset
- **第三版**: 扩大数据集，7datasets
- **第四版**: 加入null_images，7datasets

## 相关资源

### 项目文档
- `00_pipeline/01_README/01_README_files_img600.ipynb`: 项目说明文档
- `00_pipeline/01_README/01_README_old.ipynb`: 旧版项目文档

### 数据集信息
- 2024年可用数据：20组数据
- 训练数据：7组数据（来自不同卡片）
- 测试数据：1组数据（held-out test set）

### 模型评估结果
- 总共20个模型评估结果
- 最佳模型：yolo11n_custom7null_cv1_ms2_0809-0823_10_ok_8
- 最佳置信度阈值：0.7, 0.8, 0.85

## 贡献指南

### 代码规范
- 使用Python 3.8+
- 遵循PEP 8代码风格
- 添加必要的文档字符串
- 使用类型注解

### 提交规范
- 使用有意义的提交信息
- 关联相关issue
- 添加测试代码

## 许可证

本项目采用开源许可证，具体信息请查看项目根目录的LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目维护者：Tianqi
- 项目地址：`/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/`

---

**最后更新**: 2026-02-18
**版本**: 1.0.0
