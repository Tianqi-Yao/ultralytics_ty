from lark import logger
from ultralytics import YOLO

import wandb
wandb.login(key="957096cc564005d5332d45e2da6a75838e1cc9ac")

# ===== project name =====
PROJECT_NAME = "swd_model_v2_3datasets"
runPath = "/workspace/_ty/models/runs_yolov11/yaml/"
# ========================

# ==== 你需要手动填写的列表 ====
yamlFileNames = [  # 数据集 yaml 文件名（不要写后缀 .yaml）
    "data_split_0.4_0.3_0.3",
    "data_split_0.5_0.3_0.2",
    "data_split_0.6_0.2_0.2",
    "data_split_0.7_0.2_0.1",
    "data_split_0.8_0.2_0",
]
batchSizes = [
    4, 
    8, 16,
    # 64,
]
models = [
    # "yolo11n-seg.pt",
    # "yolo11s-seg.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    # "yolo11m-seg.pt",
    "yolo11m.pt",
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
                    epochs=10,
                    imgsz=640,
                    batch=batch,
                    device=0,
                    workers=4,
                    project=f"output/{PROJECT_NAME}",   # 输出目录
                    name=f"{modelFile}_{yamlFileName}_{batch}",
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