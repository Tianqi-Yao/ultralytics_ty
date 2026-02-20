from pathlib import Path
import logging

from ultralytics import YOLO

import wandb
wandb.login(key="957096cc564005d5332d45e2da6a75838e1cc9ac")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("train.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ===== project name =====
PROJECT_NAME = "swd_model_v5_nullImagesAdded_final"
run_path = Path("/workspace/_ty/00_pipeline/02_current_batch_run/02_TrainingModel/02_yaml")
# ========================

# ==== 你需要手动填写的列表 ====
yaml_file_names = [  # 数据集 yaml 文件名（不要写后缀 .yaml）
    "20pct_null_images_add_rawData_list_train_val_test",
]
batch_sizes = [
    8, 4, 16,
]
models = [
    # "yolo11n-seg.pt",
    # "yolo11s-seg.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    # "yolo11m-seg.pt",
    "yolo11m.pt",
    # "yolo11l-seg.pt", "yolo11x-seg.pt",
    "yolo11l.pt", "yolo11x.pt",
]
# models = [
#     "yolo11n-seg.pt",
#     "/workspace/_ty/models/runs_yolov11/output_16mp/good/yolo11m-seg.pt---data_split_0.6_0.2_0.2_8-----0.909/weights/best.pt",
#     "/workspace/_ty/models/runs_yolov11/output_16mp/good/yolo11n-seg.pt---data_split_0.6_0.2_0.2_4---0.913/weights/best.pt",
#     "/workspace/_ty/models/runs_yolov11/output_16mp/good/yolo11n.pt---data_split_0.6_0.1_0.3_4----0.906/weights/best.pt",
# ]
# =================================

# # 循环遍历
# for yaml_file_name in yaml_file_names:
#     yaml_path = run_path / (yaml_file_name + ".yaml")
#     for model_file in models:
#         for batch in batch_sizes:
#             logger.info(f"开始训练: model={model_file}, dataset={yaml_file_name}, batch={batch}")
#
#             model = YOLO(model_file)
#
#             try:
#                 model.train(
#                     data=str(yaml_path),
#                     epochs=300,
#                     imgsz=640,
#                     batch=batch,
#                     device=0,
#                     workers=4,
#                     project=f"output/{PROJECT_NAME}_seed42",   # 输出目录
#                     name=f"{model_file}_{yaml_file_name}_{batch}",
#                     # （可选）方便复现实验
#                     seed=42,
#                     deterministic=True,
#                 )
#
#                 # 测试集验证
#                 model.val(
#                     data=str(yaml_path),
#                     split="test",
#                     name=f"{model_file}_{yaml_file_name}_{batch}_test",
#                 )
#                 logger.info(f"训练+验证完成: model={model_file}, batch={batch}")
#             except RuntimeError as e:
#                 if "CUDA out of memory" in str(e):
#                     logger.warning(f"跳过: model={model_file}, yaml={yaml_file_name}, batch={batch} —— 显存不足")
#                     continue
#                 else:
#                     raise  # 不是 OOM 的错误则继续抛出

# # == 循环遍历2 ==
# for yaml_file_name in yaml_file_names:
#     yaml_path = run_path / (yaml_file_name + ".yaml")
#     for model_file in models:
#         for batch in batch_sizes:
#             logger.info(f"开始训练: model={model_file}, dataset={yaml_file_name}, batch={batch}")
#
#             model = YOLO(model_file)
#
#             try:
#                 model.train(
#                     data=str(yaml_path),
#                     epochs=1000,
#                     imgsz=640,
#                     batch=batch,
#                     device=0,
#                     workers=4,
#                     project=f"output/{PROJECT_NAME}_noAug_seed0",
#                     name=f"{model_file}_{yaml_file_name}_{batch}",
#
#                     # ========= 关键：根据trap关闭/弱化的图像增强 =========
#                     # 不需要"拼图场景"
#                     mosaic=0.0,         # 默认 1.0，强烈建议你改成 0
#
#                     # 这些本来默认就几乎不用，但显式关掉更安心
#                     mixup=0.0,          # 文档里默认 0.0
#                     cutmix=0.0,         # 文档里默认 0.0
#                     copy_paste=0.0,     # 你是检测，不是实例分割，可直接关
#
#                     # 几何变换：你的板子几乎不旋转、不歪，不希望改变虫子绝对大小
#                     degrees=0.0,        # 不随机旋转
#                     shear=0.0,          # 不剪切
#                     perspective=0.0,    # 不做透视变换
#                     scale=0.0,          # 关键：不做随机缩放，保护虫子的 "真实像素大小"
#                     translate=0.02,     # 保留一点点平移(2%)，模拟安装微小偏差即可
#
#                     # 颜色增强：只轻微动一动亮度/饱和度，别把红板改成奇怪颜色
#                     hsv_h=0.0,          # 不动色相（Hue）
#                     hsv_s=0.1,          # 轻微改饱和度（原默认 0.7 对你太猛）
#                     hsv_v=0.1,          # 轻微改亮度（原默认 0.4 也比较大）
#
#                     # 翻转：虫子方向不重要的话可以保留水平翻转
#                     flipud=0.0,         # 不上下翻转
#                     fliplr=0.5,         # 左右翻转 50% 概率
#
#                     # 多尺度训练：你已经用 SAHI 固定 640×640，再多尺度会破坏大小信息
#                     multi_scale=False,
#
#                     # （可选）方便复现实验
#                     seed=0,
#                     deterministic=True,
#                 )
#
#                 # 测试集验证
#                 model.val(
#                     data=str(yaml_path),
#                     split="test",
#                     name=f"{model_file}_{yaml_file_name}_{batch}_test",
#                 )
#                 logger.info(f"训练+验证完成: model={model_file}, batch={batch}")
#             except RuntimeError as e:
#                 if "CUDA out of memory" in str(e):
#                     logger.warning(f"跳过: model={model_file}, yaml={yaml_file_name}, batch={batch} —— 显存不足")
#                     continue
#                 else:
#                     raise  # 不是 OOM 的错误则继续抛出


# == 循环遍历3 ==
for yaml_file_name in yaml_file_names:
    yaml_path = run_path / (yaml_file_name + ".yaml")
    for model_file in models:
        for batch in batch_sizes:
            logger.info(f"开始训练: model={model_file}, dataset={yaml_file_name}, batch={batch}")

            model = YOLO(model_file)

            try:
                model.train(
                    data=str(yaml_path),
                    epochs=1000,
                    imgsz=640,
                    batch=batch,
                    device=0,
                    workers=4,
                    project=f"output/{PROJECT_NAME}_noAug_seed42",
                    name=f"{model_file}_{yaml_file_name}_{batch}",

                    # ========= 关键：根据trap关闭/弱化的图像增强 =========
                    # 不需要"拼图场景"
                    mosaic=0.0,         # 默认 1.0，强烈建议你改成 0

                    # 这些本来默认就几乎不用，但显式关掉更安心
                    mixup=0.0,          # 文档里默认 0.0
                    cutmix=0.0,         # 文档里默认 0.0
                    copy_paste=0.0,     # 你是检测，不是实例分割，可直接关

                    # 几何变换：你的板子几乎不旋转、不歪，不希望改变虫子绝对大小
                    degrees=0.0,        # 不随机旋转
                    shear=0.0,          # 不剪切
                    perspective=0.0,    # 不做透视变换
                    scale=0.0,          # 关键：不做随机缩放，保护虫子的 "真实像素大小"
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
                logger.info(f"训练完成: model={model_file}, dataset={yaml_file_name}, batch={batch}")

                # 测试集验证
                # model.val(
                #     data=str(yaml_path),
                #     split="test",
                #     name=f"{model_file}_{yaml_file_name}_{batch}_test",
                # )
                # logger.info(f"验证完成: model={model_file}, dataset={yaml_file_name}, batch={batch}")
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.warning(f"跳过: model={model_file}, yaml={yaml_file_name}, batch={batch} —— 显存不足")
                    continue
                else:
                    raise  # 不是 OOM 的错误则继续抛出
