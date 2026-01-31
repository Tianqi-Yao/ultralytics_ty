from ultralytics import YOLO

import wandb
wandb.login(key="957096cc564005d5332d45e2da6a75838e1cc9ac")

# ===== project name =====
PROJECT_NAME = "swd_model_v4_7datasets_null_image_full"
runPath = "/workspace/_ty/03_code/05_train_model/yolo/yaml/"
# ========================

# ==== 你需要手动填写的列表 ====
yamlFileNames = [  # 数据集 yaml 文件名（不要写后缀 .yaml）
    # "data_split_0.4_0.3_0.3",
    # "data_split_0.5_0.3_0.2",
    # "data_split_0.6_0.2_0.2",
    # "data_split_0.7_0.2_0.1",
    # "data_split_0.8_0.2_0",
    # "data_split_custom",
    # "custom7_v1-34_36_40_11-13-10",
    # "custom7_v2-13_7_34_36-40_10-11",
    # "custom7_v3-13_7_34-36_40-10_11",
    # "custom7_v4-36_40_10_11-7_34-13",
    # "custom7_v5-36_40-13_7_34-10_11",
    # "custom7null_cv1_ms2_0809-0823_10_ok",
    "custom7null_cv2_ms1_0710-0726_36_ok",
    "custom7null_cv3_ms1_0809-0823_34_ok",
    "custom7null_cv4_ms1_0605-0621_40_ok",
    "custom7null_cv5_ms2_0726-0809_13_ok",
]
batchSizes = [
    4, 
    8, 16,
]
models = [
    # "yolo11n-seg.pt",
    # "yolo11s-seg.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    # "yolo11m-seg.pt",
    # "yolo11m.pt",
    # "yolo11l-seg.pt", "yolo11x-seg.pt",
    # "yolo11l.pt", "yolo11x.pt",
]
# models = [
#     "yolo11n-seg.pt",
#     "/workspace/_ty/models/runs_yolov11/output_16mp/good/yolo11m-seg.pt---data_split_0.6_0.2_0.2_8-----0.909/weights/best.pt",
#     "/workspace/_ty/models/runs_yolov11/output_16mp/good/yolo11n-seg.pt---data_split_0.6_0.2_0.2_4---0.913/weights/best.pt",
#     "/workspace/_ty/models/runs_yolov11/output_16mp/good/yolo11n.pt---data_split_0.6_0.1_0.3_4----0.906/weights/best.pt",
# ]
# =================================

# # 循环遍历
# for yamlFileName in yamlFileNames:
#     yamlPath = runPath + yamlFileName + ".yaml"
#     for modelFile in models:
#         for batch in batchSizes:
#             print(f"\n🚀 Training model={modelFile}, dataset={yamlFileName}, batch={batch}")

#             model = YOLO(modelFile)

#             try:
#                 model.train(
#                     data=yamlPath,
#                     epochs=300,
#                     imgsz=640,
#                     batch=batch,
#                     device=0,
#                     workers=4,
#                     project=f"output/{PROJECT_NAME}_seed42",   # 输出目录
#                     name=f"{modelFile}_{yamlFileName}_{batch}",
#                     # （可选）方便复现实验
#                     seed=42,
#                     deterministic=True,
#                 )

#                 # 测试集验证
#                 model.val(
#                     data=yamlPath,
#                     split="test",
#                     name=f"{modelFile}_{yamlFileName}_{batch}_test",
#                 )
#             except RuntimeError as e:
#                 if "CUDA out of memory" in str(e):
#                     print(f"⚠️  跳过: model={modelFile}, yaml={yamlFileName}, batch={batch} —— 显存不足")
#                     continue
#                 else:
#                     raise  # 不是 OOM 的错误则继续抛出

# # == 循环遍历2 ==
# for yamlFileName in yamlFileNames:
#     yamlPath = runPath + yamlFileName + ".yaml"
#     for modelFile in models:
#         for batch in batchSizes:
#             print(f"\n🚀 Training model={modelFile}, dataset={yamlFileName}, batch={batch}")

#             model = YOLO(modelFile)

#             try:
#                 model.train(
#                     data=yamlPath,
#                     epochs=1000,
#                     imgsz=640,
#                     batch=batch,
#                     device=0,
#                     workers=4,
#                     project=f"output/{PROJECT_NAME}_noAug_seed0",   #
#                     name=f"{modelFile}_{yamlFileName}_{batch}",

#                     # ========= 关键：根据trap关闭/弱化的图像增强 =========
#                     # 不需要“拼图场景”
#                     mosaic=0.0,         # 默认 1.0，强烈建议你改成 0

#                     # 这些本来默认就几乎不用，但显式关掉更安心
#                     mixup=0.0,          # 文档里默认 0.0
#                     cutmix=0.0,         # 文档里默认 0.0
#                     copy_paste=0.0,     # 你是检测，不是实例分割，可直接关

#                     # 几何变换：你的板子几乎不旋转、不歪，不希望改变虫子绝对大小
#                     degrees=0.0,        # 不随机旋转
#                     shear=0.0,          # 不剪切
#                     perspective=0.0,    # 不做透视变换
#                     scale=0.0,          # 关键：不做随机缩放，保护虫子的 “真实像素大小”
#                     translate=0.02,     # 保留一点点平移(2%)，模拟安装微小偏差即可

#                     # 颜色增强：只轻微动一动亮度/饱和度，别把红板改成奇怪颜色
#                     hsv_h=0.0,          # 不动色相（Hue）
#                     hsv_s=0.1,          # 轻微改饱和度（原默认 0.7 对你太猛）
#                     hsv_v=0.1,          # 轻微改亮度（原默认 0.4 也比较大）

#                     # 翻转：虫子方向不重要的话可以保留水平翻转
#                     flipud=0.0,         # 不上下翻转
#                     fliplr=0.5,         # 左右翻转 50% 概率

#                     # 多尺度训练：你已经用 SAHI 固定 640×640，再多尺度会破坏大小信息
#                     multi_scale=False,

#                     # （可选）方便复现实验
#                     seed=0,
#                     deterministic=True,
#                 )

#                 # 测试集验证
#                 model.val(
#                     data=yamlPath,
#                     split="test",
#                     name=f"{modelFile}_{yamlFileName}_{batch}_test",
#                 )
#             except RuntimeError as e:
#                 if "CUDA out of memory" in str(e):
#                     print(f"⚠️  跳过: model={modelFile}, yaml={yamlFileName}, batch={batch} —— 显存不足")
#                     continue
#                 else:
#                     raise  # 不是 OOM 的错误则继续抛出


# == 循环遍历3 ==
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
                    project=f"output/{PROJECT_NAME}_noAug_seed42",
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
                    seed=42,
                    deterministic=True,
                )

                # 测试集验证
                model.val(
                    data=yamlPath,
                    split="test",
                    name=f"{modelFile}_{yamlFileName}_{batch}_test",
                )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"⚠️  跳过: model={modelFile}, yaml={yamlFileName}, batch={batch} —— 显存不足")
                    continue
                else:
                    raise  # 不是 OOM 的错误则继续抛出