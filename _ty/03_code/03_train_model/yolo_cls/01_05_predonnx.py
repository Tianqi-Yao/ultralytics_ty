from unittest import result
import onnxruntime as ort
import numpy as np
import cv2
import os
import yaml  # 如果你要从 data.yaml 里加载类名

onnx_path = "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/02_models/best_models/91_swd_cls/data_split_0.6_0.4_0.0_yolo11m-cls_4_224.onnx"
session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# 找到输入名
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name  # 通常只有一个输出

# 假设 imgsz = 224（要和训练时一致）
imgsz = 224

def preprocess(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # [1,3,H,W]
    return img

# class names（你可以从 data.yaml 读）
# with open("/path/to/data.yaml", "r") as f:
#     names = yaml.safe_load(f)["names"]
names = {0: "class0", 1: "class1", 2: "class2"}

images_folder_path = "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/01_data/00_test/00_try/del/debug"

results = []

for filename in os.listdir(images_folder_path):
    if not filename.lower().endswith((".jpg", ".png")):
        continue
    image_path = os.path.join(images_folder_path, filename)
    inp = preprocess(image_path)

    # ONNX 推理
    logits = session.run([output_name], {input_name: inp})[0]  # shape [1, num_classes]
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    top1_id = int(np.argmax(probs, axis=1)[0])
    top1_conf = float(probs[0, top1_id])
    top1_name = names.get(top1_id, str(top1_id))

    # print(filename, "->", top1_name, top1_conf)
    results.append((filename, top1_name, top1_conf))

# sort filename
results.sort(key=lambda x: x[0])
for r in results:
    print(r)
