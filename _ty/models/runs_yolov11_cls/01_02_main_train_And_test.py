# %% [markdown]
# # 训练分类模型

# %%
from ultralytics import YOLO

# ==== 你需要手动填写的列表 ====
dataFloderNameList = [  # 数据集 yaml 文件名（不要写后缀 .yaml）
    "data_split_0.4_0.3_0.3",
    "data_split_0.5_0.2_0.3",
    "data_split_0.5_0.3_0.2",
    "data_split_0.6_0.2_0.2",
]
batchSizes = [
    # 8, 
    16, 32, 
    # 64
]
models = [
    "yolo11n-cls.pt",
    "yolo11s-cls.pt",
    "yolo11m-cls.pt",
    # "yolo11l-cls.pt",
]
# =================================

runPath =  "/workspace/models/runs_yolov11_cls/data/"

# 循环遍历
for modelFile in models:
    for dataFloderName in dataFloderNameList:
        dataPath = runPath + dataFloderName
        for batch in batchSizes:
            print(f"\n🚀 Training model={modelFile}, dataset={dataFloderName}, batch={batch}")

            model = YOLO(modelFile)

            try:
                model.train(
                    data=dataPath,
                    epochs=2200,
                    imgsz=640,
                    batch=batch,
                    device=0,
                    workers=4,
                    project=f"output1/{modelFile}",
                    name=f"{dataFloderName}_{batch}",

                    translate = 0,
                    scale = 0,
                    erasing = 0,

                    optimizer='AdamW',
                    lr0=1e-3 * (batch/64),
                    lrf=0.01,                 # 余弦衰减到初始LR的1%
                    weight_decay=5e-4,
                    patience=30,              # 早停

                    # 轻度、保守的增广
                    mixup=0.0,
                    fliplr=0.5, flipud=0.0,
                    hsv_h=0.0, hsv_s=0.20, hsv_v=0.20,

                    dropout=0.05              # 分类头很小的dropout
                    # trainer=NoCropTrainer, validator=NoCropValidator  # 若已实现无裁剪管线

                )



                # 测试集验证
                model.val(
                    data=dataPath,
                    split="test",
                    name=f"{dataFloderName}_{batch}_test"
                )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"⚠️  跳过: model={modelFile}, yaml={dataFloderName}, batch={batch} —— 显存不足")
                    continue
                else:
                    raise  # 不是 OOM 的错误则继续抛出

# %% [markdown]
# # 2

# %%
from ultralytics import YOLO

# ==== 你需要手动填写的列表 ====
dataFloderNameList = [  # 数据集 yaml 文件名（不要写后缀 .yaml）
    "data_split_0.4_0.3_0.3",
    "data_split_0.5_0.2_0.3",
    "data_split_0.5_0.3_0.2",
    "data_split_0.6_0.2_0.2",
]
batchSizes = [
    # 8, 
    16, 32, 
    # 64
]
models = [
    "yolo11n-cls.pt",
    "yolo11s-cls.pt",
    "yolo11m-cls.pt",
    # "yolo11l-cls.pt",
]
# =================================

runPath =  "/workspace/models/runs_yolov11_cls/data/"

# 循环遍历
for modelFile in models:
    for dataFloderName in dataFloderNameList:
        dataPath = runPath + dataFloderName
        for batch in batchSizes:
            print(f"\n🚀 Training model={modelFile}, dataset={dataFloderName}, batch={batch}")

            model = YOLO(modelFile)

            try:
                model.train(
                    data=dataPath,
                    epochs=2080,
                    imgsz=640,
                    batch=batch,
                    device=0,
                    workers=4,
                    project=f"output2/{modelFile}",
                    name=f"{dataFloderName}_{batch}",
                    
                    translate = 0,
                    scale = 0,
                    erasing = 0,

                    optimizer='AdamW',
                    lr0=7e-4 * (batch/64),    # 略低的起始LR
                    lrf=0.01,
                    weight_decay=1e-3,        # 更强WD抑制过拟合
                    label_smoothing=0.05,     # 若版本支持分类平滑
                    patience=40,

                    mixup=0.0, 
                    fliplr=0.5, flipud=0.0,
                    hsv_h=0.0, hsv_s=0.15, hsv_v=0.15,  # 更保守的颜色扰动

                    dropout=0.15
                    # trainer=NoCropTrainer, validator=NoCropValidator
                )


                # 测试集验证
                model.val(
                    data=dataPath,
                    split="test",
                    name=f"{dataFloderName}_{batch}_test"
                )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"⚠️  跳过: model={modelFile}, yaml={dataFloderName}, batch={batch} —— 显存不足")
                    continue
                else:
                    raise  # 不是 OOM 的错误则继续抛出

# %% [markdown]
# # 3

# %%
from ultralytics import YOLO

# ==== 你需要手动填写的列表 ====
dataFloderNameList = [  # 数据集 yaml 文件名（不要写后缀 .yaml）
    "data_split_0.4_0.3_0.3",
    "data_split_0.5_0.2_0.3",
    "data_split_0.5_0.3_0.2",
    "data_split_0.6_0.2_0.2",
]
batchSizes = [
    # 8, 
    16, 32, 
    # 64
]
models = [
    "yolo11n-cls.pt",
    "yolo11s-cls.pt",
    "yolo11m-cls.pt",
    # "yolo11l-cls.pt",
]
# =================================

runPath =  "/workspace/models/runs_yolov11_cls/data/"

# 循环遍历
for modelFile in models:
    for dataFloderName in dataFloderNameList:
        dataPath = runPath + dataFloderName
        for batch in batchSizes:
            print(f"\n🚀 Training model={modelFile}, dataset={dataFloderName}, batch={batch}")

            model = YOLO(modelFile)

            try:
                model.train(
                    data=dataPath,
                    epochs=2060,
                    imgsz=640,
                    batch=batch,
                    device=0,
                    workers=4,
                    project=f"output3/{modelFile}",
                    name=f"{dataFloderName}_{batch}",

                    translate = 0,
                    scale = 0,
                    erasing = 0,

                    optimizer='SGD',          # m=0.9, nesterov 默认即可
                    lr0=0.01 * (batch/64),    # SGD 常用起点
                    lrf=0.01,
                    weight_decay=1e-4,        # SGD 下WD略小一点
                    patience=30,

                    mixup=0.0,
                    fliplr=0.5, flipud=0.0,
                    hsv_h=0.0, hsv_s=0.20, hsv_v=0.20,

                    dropout=0.0               # 先不开，观察对比
                    # trainer=NoCropTrainer, validator=NoCropValidator
                )


                # 测试集验证
                model.val(
                    data=dataPath,
                    split="test",
                    name=f"{dataFloderName}_{batch}_test"
                )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"⚠️  跳过: model={modelFile}, yaml={dataFloderName}, batch={batch} —— 显存不足")
                    continue
                else:
                    raise  # 不是 OOM 的错误则继续抛出

# %% [markdown]
# # 4

# %%
from ultralytics import YOLO

# ==== 你需要手动填写的列表 ====
dataFloderNameList = [  # 数据集 yaml 文件名（不要写后缀 .yaml）
    "data_split_0.4_0.3_0.3",
    "data_split_0.5_0.2_0.3",
    "data_split_0.5_0.3_0.2",
    "data_split_0.6_0.2_0.2",
]
batchSizes = [
    # 8, 
    16, 32, 
    # 64
]
models = [
    "yolo11n-cls.pt",
    "yolo11s-cls.pt",
    "yolo11m-cls.pt",
    # "yolo11l-cls.pt",
]
# =================================

runPath =  "/workspace/models/runs_yolov11_cls/data/"

# 循环遍历
for modelFile in models:
    for dataFloderName in dataFloderNameList:
        dataPath = runPath + dataFloderName
        for batch in batchSizes:
            print(f"\n🚀 Training model={modelFile}, dataset={dataFloderName}, batch={batch}")

            model = YOLO(modelFile)

            try:
                model.train(
                    data=dataPath,
                    epochs=2200,
                    imgsz=640,                # 提升分辨率以观察细节
                    batch=batch,
                    device=0,
                    workers=4,
                    project=f"output4/{modelFile}",
                    name=f"{dataFloderName}_{batch}",

                    translate = 0,
                    scale = 0,
                    erasing = 0,

                    optimizer='AdamW',
                    lr0=8e-4 * (batch/64),    # 分辨率更高，LR略保守
                    lrf=0.01,
                    weight_decay=7e-4,
                    patience=35,

                    mixup=0.0, 
                    fliplr=0.5, flipud=0.0,
                    hsv_h=0.0, hsv_s=0.18, hsv_v=0.18,

                    dropout=0.10
                    # trainer=NoCropTrainer, validator=NoCropValidator
                )


                # 测试集验证
                model.val(
                    data=dataPath,
                    split="test",
                    name=f"{dataFloderName}_{batch}_test"
                )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"⚠️  跳过: model={modelFile}, yaml={dataFloderName}, batch={batch} —— 显存不足")
                    continue
                else:
                    raise  # 不是 OOM 的错误则继续抛出

# %%
from ultralytics import YOLO

# ==== 你需要手动填写的列表 ====
dataFloderNameList = [  # 数据集 yaml 文件名（不要写后缀 .yaml）
    "data_split_0.4_0.3_0.3",
    "data_split_0.5_0.2_0.3",
    "data_split_0.5_0.3_0.2",
    "data_split_0.6_0.2_0.2",
]
batchSizes = [
    # 8, 
    16, 32, 
    # 64
]
models = [
    "yolo11n-cls.pt",
    "yolo11s-cls.pt",
    "yolo11m-cls.pt",
    # "yolo11l-cls.pt",
]
# =================================

runPath =  "/workspace/models/runs_yolov11_cls/data/"

# 循环遍历
for modelFile in models:
    for dataFloderName in dataFloderNameList:
        dataPath = runPath + dataFloderName
        for batch in batchSizes:
            print(f"\n🚀 Training model={modelFile}, dataset={dataFloderName}, batch={batch}")

            model = YOLO(modelFile)

            try:
                model.train(
                    data=dataPath,
                    epochs=2000,
                    imgsz=640,                # 略低分辨率加快迭代，观察趋势
                    batch=batch,
                    device=0,
                    workers=4,
                    project=f"output5/{modelFile}",
                    name=f"{dataFloderName}_{batch}",

                    translate = 0,
                    scale = 0,
                    erasing = 0,

                    optimizer='AdamW',
                    lr0=1.2e-3 * (batch/64),  # 稍大一点LR加快前期收敛
                    lrf=0.02,                 # 末端稍微高一点
                    weight_decay=3e-4,        # 正则更轻
                    patience=25,

                    mixup=0.0, 
                    fliplr=0.5, flipud=0.0,
                    hsv_h=0.0, hsv_s=0.15, hsv_v=0.15,

                    dropout=0.0
                    # trainer=NoCropTrainer, validator=NoCropValidator
                )



                # 测试集验证
                model.val(
                    data=dataPath,
                    split="test",
                    name=f"{dataFloderName}_{batch}_test"
                )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"⚠️  跳过: model={modelFile}, yaml={dataFloderName}, batch={batch} —— 显存不足")
                    continue
                else:
                    raise  # 不是 OOM 的错误则继续抛出

# %% [markdown]
# # 5

# %%
# from ultralytics import YOLO

# # ==== 你需要手动填写的列表 ====
# dataFloderNameList = [  # 数据集 yaml 文件名（不要写后缀 .yaml）
#     "data_split_0.4_0.3_0.3",
#     "data_split_0.5_0.2_0.3",
#     "data_split_0.5_0.3_0.2",
#     "data_split_0.6_0.2_0.2",
# ]
# batchSizes = [
#     # 8, 
#     16, 32, 
#     # 64
# ]
# models = [
#     "yolo11n-cls.pt",
#     "yolo11s-cls.pt",
#     "yolo11m-cls.pt",
#     # "yolo11l-cls.pt",
# ]
# # =================================

# runPath =  "/workspace/models/runs_yolov11_cls/data/"

# # 循环遍历
# for modelFile in models:
#     for dataFloderName in dataFloderNameList:
#         dataPath = runPath + dataFloderName
#         for batch in batchSizes:
#             print(f"\n🚀 Training model={modelFile}, dataset={dataFloderName}, batch={batch}")

#             model = YOLO(modelFile)

#             try:
#                 model.train(
#                     data=dataPath,
#                     epochs=1000,
#                     imgsz=640,
#                     batch=batch,
#                     device=0,
#                     workers=4,
#                     project=f"output/{modelFile}",   # 输出目录
#                     name=f"{dataFloderName}_{batch}",  # run 名字更清晰
#                 )

#                 # 测试集验证
#                 model.val(
#                     data=dataPath,
#                     split="test",
#                     name=f"{dataFloderName}_{batch}_test"
#                 )
#             except RuntimeError as e:
#                 if "CUDA out of memory" in str(e):
#                     print(f"⚠️  跳过: model={modelFile}, yaml={dataFloderName}, batch={batch} —— 显存不足")
#                     continue
#                 else:
#                     raise  # 不是 OOM 的错误则继续抛出


