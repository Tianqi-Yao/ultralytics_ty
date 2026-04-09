from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

model = YOLO("/home/paalab/Documents/yolo/ncnn_models/yolo11n_b8/model_ncnn", task="detect")

img_path = "/home/paalab/Documents/yolo/images/0727_0836_640.jpg"
img = cv2.imread(img_path)
H, W = img.shape[:2]

tile_size = 640
overlap = 0.2
stride = int(tile_size * (1 - overlap))  # 512

all_boxes = []
all_scores = []
all_classes = []

# Precompute tiles so we can show an accurate overall progress bar.
tiles = [(x, y) for y in range(0, H, stride) for x in range(0, W, stride)]
total_tiles = len(tiles)

if tqdm is not None:
    tile_iter = tqdm(tiles, total=total_tiles, desc="Inference", unit="tile")
else:
    print(f"Total tiles: {total_tiles}")
    tile_iter = tiles

for idx, (x, y) in enumerate(tile_iter, start=1):
    progress = idx / total_tiles * 100
    print(f"\rInference progress: {idx}/{total_tiles} ({progress:.1f}%)", end="", flush=True)

    x2 = min(x + tile_size, W)
    y2 = min(y + tile_size, H)

    tile = img[y:y2, x:x2]

    # 右下角不足 640 时补黑边到 640x640
    pad_h = tile_size - tile.shape[0]
    pad_w = tile_size - tile.shape[1]
    if pad_h > 0 or pad_w > 0:
        tile = cv2.copyMakeBorder(
            tile, 0, pad_h, 0, pad_w,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

    # save tile to results
    save_dir = Path("/home/paalab/Documents/yolo/results/tiles")
    save_dir.mkdir(parents=True, exist_ok=True)
    tile_path = save_dir / f"tile_{idx:04d}.jpg"
    cv2.imwrite(str(tile_path), tile)

    results = model(tile, imgsz=640, conf=0.25, verbose=False)

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        continue

    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()

    for box, score, cls_id in zip(boxes, scores, classes):
        x1, y1, x2b, y2b = box

        # 映射回原图坐标
        x1 += x
        x2b += x
        y1 += y
        y2b += y

        # 裁回原图边界
        x1 = max(0, min(x1, W))
        x2b = max(0, min(x2b, W))
        y1 = max(0, min(y1, H))
        y2b = max(0, min(y2b, H))

        all_boxes.append([x1, y1, x2b, y2b])
        all_scores.append(float(score))
        all_classes.append(int(cls_id))


print(f"Total raw detections: {len(all_boxes)}")