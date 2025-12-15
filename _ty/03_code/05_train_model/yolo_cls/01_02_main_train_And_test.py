# %% [markdown]
# # 训练分类模型

# %%
from ultralytics import YOLO
import wandb
import os

# ========================
# 1. 基本配置
# ========================
wandb.login(key="957096cc564005d5332d45e2da6a75838e1cc9ac")

PROJECT_NAME = "swd_cls_v1_4datasets"

# 你的分类数据集根目录列表
# 每个路径下面需要是：
#   root/
#     train/classA/...
#     val/classA/...
#     test/classA/...   (可选)
DATASETS = [
    # "/workspace/_ty/03_code/05_train_model/yolo_cls/data/data_split_0.4_0.3_0.3", 
    # "/workspace/_ty/03_code/05_train_model/yolo_cls/data/data_split_0.5_0.3_0.2",
    # "/workspace/_ty/03_code/05_train_model/yolo_cls/data/data_split_0.6_0.2_0.2",
    "/workspace/_ty/03_code/05_train_model/yolo_cls/data/data_split_0.6_0.4_0.0",
    "/workspace/_ty/03_code/05_train_model/yolo_cls/data/data_split_0.8_0.2_0.0",
]

batchSizes = [4, 8, 16]

# 使用 YOLO 分类模型（注意是 -cls.pt）
models = [
    "yolo11n-cls.pt",
    "yolo11s-cls.pt",
    "yolo11m-cls.pt"
    # 需要可以加： , "yolo11l-cls.pt", "yolo11x-cls.pt",
]


# 小工具：从路径里提取一个好读的名字
def dataset_name_from_path(path: str) -> str:
    return os.path.basename(path.rstrip("/")) or path.replace("/", "_")


# ========================
# 2. 循环1：默认增强 + seed=42
# ========================
# for data_path in DATASETS:
#     ds_name = dataset_name_from_path(data_path)

#     for modelFile in models:
#         for batch in batchSizes:
#             print(f"\n🚀 [CLS-DEFAULT] model={modelFile}, dataset={ds_name}, batch={batch}")

#             model = YOLO(modelFile)

#             try:
#                 model.train(
#                     task="classify",        # 明确指定分类任务
#                     data=data_path,        # ⭐ 直接给根目录路径，不是 yaml
#                     epochs=1000,
#                     imgsz=640,
#                     batch=batch,
#                     device=0,
#                     workers=4,
#                     project=f"output/{PROJECT_NAME}_seed42",
#                     name=f"{ds_name}_{modelFile}_{batch}",
#                     seed=42,
#                     deterministic=True,
#                 )

#                 # 如果数据里有 test/ 目录，可以这样评估
#                 model.val(
#                     task="classify",
#                     data=data_path,
#                     split="test",   # 没有 test 文件夹就删掉这个参数
#                     name=f"{ds_name}_{modelFile}_{batch}_test",
#                 )

#             except RuntimeError as e:
#                 if "CUDA out of memory" in str(e):
#                     print(f"⚠️ 跳过: model={modelFile}, dataset={ds_name}, batch={batch} —— 显存不足")
#                     continue
#                 else:
#                     raise


# ========================
# 3. 循环2：基本无增强 + seed=0
# ========================
# for data_path in DATASETS:
#     ds_name = dataset_name_from_path(data_path)

#     for modelFile in models:
#         for batch in batchSizes:
#             print(f"\n🚀 [CLS-noAug-seed0] model={modelFile}, dataset={ds_name}, batch={batch}")

#             model = YOLO(modelFile)

#             try:
#                 model.train(
#                     task="classify",
#                     data=data_path,
#                     epochs=1000,
#                     # imgsz=640,
#                     batch=batch,
#                     device=0,
#                     workers=4,
#                     project=f"output/{PROJECT_NAME}_noAug_seed0",
#                     name=f"{ds_name}_{modelFile}_{batch}",

#                     # ====== 分类增强：尽量关掉 ======
#                     degrees=0.0,
#                     scale=0.0,
#                     shear=0.0,
#                     translate=0.0,

#                     mixup=0.0,
#                     cutmix=0.0,
#                     erasing=0.0,

#                     flipud=0.0,
#                     fliplr=0.0,

#                     seed=0,
#                     deterministic=True,
#                 )

#                 model.val(
#                     task="classify",
#                     data=data_path,
#                     split="test",
#                     name=f"{ds_name}_{modelFile}_{batch}_test",
#                 )

#             except RuntimeError as e:
#                 if "CUDA out of memory" in str(e):
#                     print(f"⚠️ 跳过: model={modelFile}, dataset={ds_name}, batch={batch} —— 显存不足")
#                     continue
#                 else:
#                     raise


# ========================
# 4. 循环3：基本无增强 + seed=42
# ========================
for data_path in DATASETS:
    ds_name = dataset_name_from_path(data_path)

    for modelFile in models:
        for batch in batchSizes:
            print(f"\n🚀 [CLS-noAug-seed42] model={modelFile}, dataset={ds_name}, batch={batch}")

            model = YOLO(modelFile)

            try:
                model.train(
                    task="classify",
                    data=data_path,
                    epochs=1000,
                    # imgsz=640,
                    batch=batch,
                    device=0,
                    workers=4,
                    project=f"output/{PROJECT_NAME}_noAug_seed42",
                    name=f"{ds_name}_{modelFile}_{batch}",

                    degrees=0.0,
                    scale=0.0,
                    shear=0.0,
                    translate=0.0,

                    mixup=0.0,
                    cutmix=0.0,
                    erasing=0.0,

                    flipud=0.0,
                    fliplr=0.0,

                    seed=42,
                    deterministic=True,
                )

                model.val(
                    task="classify",
                    data=data_path,
                    split="test",
                    name=f"{ds_name}_{modelFile}_{batch}_test",
                )

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"⚠️ 跳过: model={modelFile}, dataset={ds_name}, batch={batch} —— 显存不足")
                    continue
                else:
                    raise



