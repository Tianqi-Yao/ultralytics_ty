# %% [markdown]
# # 训练instance segmentation模型

# %%
import wandb
wandb.login(key="957096cc564005d5332d45e2da6a75838e1cc9ac")

# %%
from lark import logger
from ultralytics import YOLO

# ===== project name =====
PROJECT_NAME = "swd_model_obb_v1"
runPath = "/workspace/_ty/03_code/05_train_model/yolo_obb/yaml/"
# ========================

# ==== 你需要手动填写的列表 ====
yamlFileNames = [  # 数据集 yaml 文件名（不要写后缀 .yaml）
    "data_split_0.6_0.4_0.0",
    "data_split_0.8_0.2_0.0",
]
batchSizes = [
    4, 8, 16, 
]

models = [
    "yolo11s-obb.yaml",
]

# =================================

# 循环遍历
for yamlFileName in yamlFileNames:
    yamlPath = runPath + yamlFileName + ".yaml"
    for modelFile in models:
        for batch in batchSizes:
            print(f"\n🚀 Training model={modelFile}, dataset={yamlFileName}, batch={batch}")

            model = YOLO(modelFile)
            model = model.load("/workspace/_ty/02_models/best_models/04_swd_hbb/model_v2_4datasets_noAug_seed0_yolo11s_data_split_custom_8.pt")


            try:
                model.train(
                    data=yamlPath,
                    epochs=1000,
                    imgsz=640,
                    batch=batch,
                    device=0,
                    workers=4,
                    freeze=10,
                    project=f"output/{PROJECT_NAME}_noAug_freeze10_hbbBest",   # 输出目录
                    name=f"{modelFile}_{yamlFileName}_{batch}",
                    
                    # ========= 关键：根据trap关闭/弱化的图像增强 =========
                    # 不需要“拼图场景”
                    mosaic=0.0,         # 默认 1.0，强烈建议你改成 0

                    # 这些本来默认就几乎不用，但显式关掉更安心
                    mixup=0.0,          # 文档里默认 0.0
                    cutmix=0.0,         # 文档里默认 0.0
                    copy_paste=0.0,     # 你是检测，不是实例分割，可直接关

                    # 几何变换：你的板子几乎不旋转、不歪，不希望改变虫子绝对大小
                    degrees=0.0,        # 不随机旋转
                    shear=0.0,          # 不剪切
                    perspective=0.0,    # 不做透视变换
                    scale=0.0,          # 关键：不做随机缩放，保护虫子的 “真实像素大小”
                    translate=0.02,     # 保留一点点平移(2%)，模拟安装微小偏差即可

                    # 颜色增强：只轻微动一动亮度/饱和度，别把红板改成奇怪颜色
                    hsv_h=0.0,          # 不动色相（Hue）
                    hsv_s=0.1,          # 轻微改饱和度（原默认 0.7 对你太猛）
                    hsv_v=0.1,          # 轻微改亮度（原默认 0.4 也比较大）

                    # 翻转：虫子方向不重要的话可以保留水平翻转
                    flipud=0.0,         # 不上下翻转
                    fliplr=0.5,         # 左右翻转 50% 概率

                    # 多尺度训练：你已经用 SAHI 固定 640×640，再多尺度会破坏大小信息
                    multi_scale=False,

                    # （可选）方便复现实验
                    seed=0,
                    deterministic=True,
                )

                # 测试集验证
                # model.val(
                #     data=yamlPath,
                #     split="test",
                #     name=f"{modelFile}_{yamlFileName}_{batch}_test",
                # )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"⚠️  跳过: model={modelFile}, yaml={yamlFileName}, batch={batch} —— 显存不足")
                    continue
                else:
                    raise  # 不是 OOM 的错误则继续抛出


# 循环遍历
for yamlFileName in yamlFileNames:
    yamlPath = runPath + yamlFileName + ".yaml"
    for modelFile in models:
        for batch in batchSizes:
            print(f"\n🚀 Training model={modelFile}, dataset={yamlFileName}, batch={batch}")

            model = YOLO(modelFile)
            model = model.load("/workspace/_ty/02_models/best_models/04_swd_hbb/model_v2_4datasets_noAug_seed0_yolo11s_data_split_custom_8.pt")

            try:
                model.train(
                    data=yamlPath,
                    epochs=1000,
                    imgsz=640,
                    batch=batch,
                    device=0,
                    workers=4,
                    project=f"output/{PROJECT_NAME}_noAug_hbbBest",   # 输出目录
                    name=f"{modelFile}_{yamlFileName}_{batch}",

                    # ========= 关键：根据trap关闭/弱化的图像增强 =========
                    # 不需要“拼图场景”
                    mosaic=0.0,         # 默认 1.0，强烈建议你改成 0

                    # 这些本来默认就几乎不用，但显式关掉更安心
                    mixup=0.0,          # 文档里默认 0.0
                    cutmix=0.0,         # 文档里默认 0.0
                    copy_paste=0.0,     # 你是检测，不是实例分割，可直接关

                    # 几何变换：你的板子几乎不旋转、不歪，不希望改变虫子绝对大小
                    degrees=0.0,        # 不随机旋转
                    shear=0.0,          # 不剪切
                    perspective=0.0,    # 不做透视变换
                    scale=0.0,          # 关键：不做随机缩放，保护虫子的 “真实像素大小”
                    translate=0.02,     # 保留一点点平移(2%)，模拟安装微小偏差即可

                    # 颜色增强：只轻微动一动亮度/饱和度，别把红板改成奇怪颜色
                    hsv_h=0.0,          # 不动色相（Hue）
                    hsv_s=0.1,          # 轻微改饱和度（原默认 0.7 对你太猛）
                    hsv_v=0.1,          # 轻微改亮度（原默认 0.4 也比较大）

                    # 翻转：虫子方向不重要的话可以保留水平翻转
                    flipud=0.0,         # 不上下翻转
                    fliplr=0.5,         # 左右翻转 50% 概率

                    # 多尺度训练：你已经用 SAHI 固定 640×640，再多尺度会破坏大小信息
                    multi_scale=False,

                    # （可选）方便复现实验
                    seed=0,
                    deterministic=True,
                )

                # 测试集验证
                # model.val(
                #     data=yamlPath,
                #     split="test",
                #     name=f"{modelFile}_{yamlFileName}_{batch}_test",
                # )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"⚠️  跳过: model={modelFile}, yaml={yamlFileName}, batch={batch} —— 显存不足")
                    continue
                else:
                    raise  # 不是 OOM 的错误则继续抛出


models = [
    "yolo11n-obb.pt",
    "yolo11s-obb.pt",
    "yolo11m-obb.pt",
]

for yamlFileName in yamlFileNames:
    yamlPath = runPath + yamlFileName + ".yaml"
    for modelFile in models:
        for batch in batchSizes:
            print(f"\n🚀 Training model={modelFile}, dataset={yamlFileName}, batch={batch}")

            model = YOLO(modelFile)

            try:
                model.train(
                    data=yamlPath,
                    epochs=1000,
                    imgsz=640,
                    batch=batch,
                    device=0,
                    workers=4,
                    freeze=10,
                    project=f"output/{PROJECT_NAME}_noAug_freeze10",   # 输出目录
                    name=f"{modelFile}_{yamlFileName}_{batch}",
                    
                    # ========= 关键：根据trap关闭/弱化的图像增强 =========
                    # 不需要“拼图场景”
                    mosaic=0.0,         # 默认 1.0，强烈建议你改成 0

                    # 这些本来默认就几乎不用，但显式关掉更安心
                    mixup=0.0,          # 文档里默认 0.0
                    cutmix=0.0,         # 文档里默认 0.0
                    copy_paste=0.0,     # 你是检测，不是实例分割，可直接关

                    # 几何变换：你的板子几乎不旋转、不歪，不希望改变虫子绝对大小
                    degrees=0.0,        # 不随机旋转
                    shear=0.0,          # 不剪切
                    perspective=0.0,    # 不做透视变换
                    scale=0.0,          # 关键：不做随机缩放，保护虫子的 “真实像素大小”
                    translate=0.02,     # 保留一点点平移(2%)，模拟安装微小偏差即可

                    # 颜色增强：只轻微动一动亮度/饱和度，别把红板改成奇怪颜色
                    hsv_h=0.0,          # 不动色相（Hue）
                    hsv_s=0.1,          # 轻微改饱和度（原默认 0.7 对你太猛）
                    hsv_v=0.1,          # 轻微改亮度（原默认 0.4 也比较大）

                    # 翻转：虫子方向不重要的话可以保留水平翻转
                    flipud=0.0,         # 不上下翻转
                    fliplr=0.5,         # 左右翻转 50% 概率

                    # 多尺度训练：你已经用 SAHI 固定 640×640，再多尺度会破坏大小信息
                    multi_scale=False,

                    # （可选）方便复现实验
                    seed=0,
                    deterministic=True,
                )

                # 测试集验证
                # model.val(
                #     data=yamlPath,
                #     split="test",
                #     name=f"{modelFile}_{yamlFileName}_{batch}_test",
                # )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"⚠️  跳过: model={modelFile}, yaml={yamlFileName}, batch={batch} —— 显存不足")
                    continue
                else:
                    raise  # 不是 OOM 的错误则继续抛出


# 循环遍历
for yamlFileName in yamlFileNames:
    yamlPath = runPath + yamlFileName + ".yaml"
    for modelFile in models:
        for batch in batchSizes:
            print(f"\n🚀 Training model={modelFile}, dataset={yamlFileName}, batch={batch}")

            model = YOLO(modelFile)

            try:
                model.train(
                    data=yamlPath,
                    epochs=1000,
                    imgsz=640,
                    batch=batch,
                    device=0,
                    workers=4,
                    project=f"output/{PROJECT_NAME}_noAug",   # 输出目录
                    name=f"{modelFile}_{yamlFileName}_{batch}",

                    # ========= 关键：根据trap关闭/弱化的图像增强 =========
                    # 不需要“拼图场景”
                    mosaic=0.0,         # 默认 1.0，强烈建议你改成 0

                    # 这些本来默认就几乎不用，但显式关掉更安心
                    mixup=0.0,          # 文档里默认 0.0
                    cutmix=0.0,         # 文档里默认 0.0
                    copy_paste=0.0,     # 你是检测，不是实例分割，可直接关

                    # 几何变换：你的板子几乎不旋转、不歪，不希望改变虫子绝对大小
                    degrees=0.0,        # 不随机旋转
                    shear=0.0,          # 不剪切
                    perspective=0.0,    # 不做透视变换
                    scale=0.0,          # 关键：不做随机缩放，保护虫子的 “真实像素大小”
                    translate=0.02,     # 保留一点点平移(2%)，模拟安装微小偏差即可

                    # 颜色增强：只轻微动一动亮度/饱和度，别把红板改成奇怪颜色
                    hsv_h=0.0,          # 不动色相（Hue）
                    hsv_s=0.1,          # 轻微改饱和度（原默认 0.7 对你太猛）
                    hsv_v=0.1,          # 轻微改亮度（原默认 0.4 也比较大）

                    # 翻转：虫子方向不重要的话可以保留水平翻转
                    flipud=0.0,         # 不上下翻转
                    fliplr=0.5,         # 左右翻转 50% 概率

                    # 多尺度训练：你已经用 SAHI 固定 640×640，再多尺度会破坏大小信息
                    multi_scale=False,

                    # （可选）方便复现实验
                    seed=0,
                    deterministic=True,
                )

                # 测试集验证
                # model.val(
                #     data=yamlPath,
                #     split="test",
                #     name=f"{modelFile}_{yamlFileName}_{batch}_test",
                # )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"⚠️  跳过: model={modelFile}, yaml={yamlFileName}, batch={batch} —— 显存不足")
                    continue
                else:
                    raise  # 不是 OOM 的错误则继续抛出