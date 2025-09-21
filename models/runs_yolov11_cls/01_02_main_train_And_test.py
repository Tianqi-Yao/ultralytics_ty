from ultralytics import YOLO

# ==== 你需要手动填写的列表 ====
dataFloderNameList = [  # 数据集 yaml 文件名（不要写后缀 .yaml）
    "data_split_0.4_0.3_0.3",
    "data_split_0.5_0.2_0.3",
    "data_split_0.5_0.3_0.2",
    "data_split_0.6_0.2_0.2",
]
batchSizes = [
    8, 
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
                    epochs=1000,
                    imgsz=640,
                    batch=batch,
                    device=0,
                    workers=4,
                    project=f"output/{modelFile}",   # 输出目录
                    name=f"{dataFloderName}_{batch}",  # run 名字更清晰
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