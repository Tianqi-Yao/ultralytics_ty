"""
yolo_pt.py
==========
Run on the server (GPU).
Benchmarks all 5 YOLO11 .pt models on a full sticky-card image
using manual SAHI tiling + NMS. Outputs results/pt_results.csv.

Usage:
    python yolo_pt.py
"""

import csv
import gc
import statistics
import time
from pathlib import Path

import cv2
import psutil
import torch
from ultralytics import YOLO

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_BASE = Path("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/04_final_pipeline/29_Model_Performance/model/swd_model_v5_nullImagesAdded_final_noAug_seed42")
IMG_PATH   = Path("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/06_rpi/predict/0727_0836_640.jpg")

TILE_SIZE      = 640
OVERLAP        = 0.2
CONF_THRESHOLD = 0.7
NMS_IOU        = 0.5

WARMUP_RUNS = 2
TIMED_RUNS  = 10

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(__file__).parent / "results"

MODELS = {
    "yolo11n_b8":  MODEL_BASE / "yolo11npt_20pct_null_images_add_rawData_list_train_val_test_8"  / "weights" / "best.pt",
    "yolo11s_b4":  MODEL_BASE / "yolo11spt_20pct_null_images_add_rawData_list_train_val_test_4"  / "weights" / "best.pt",
    "yolo11m_b16": MODEL_BASE / "yolo11mpt_20pct_null_images_add_rawData_list_train_val_test_16" / "weights" / "best.pt",
    "yolo11l_b4":  MODEL_BASE / "yolo11lpt_20pct_null_images_add_rawData_list_train_val_test_4"  / "weights" / "best.pt",
    "yolo11x_b8":  MODEL_BASE / "yolo11xpt_20pct_null_images_add_rawData_list_train_val_test_8"  / "weights" / "best.pt",
}

MAP50 = {
    "yolo11n_b8":  0.979,
    "yolo11s_b4":  0.974,
    "yolo11m_b16": 0.985,
    "yolo11l_b4":  0.982,
    "yolo11x_b8":  0.978,
}
# ──────────────────────────────────────────────────────────────────────────────


def get_model_size_mb(pt_path: Path) -> float:
    return round(pt_path.stat().st_size / 1024 / 1024, 2)


def get_rss_mb() -> float:
    return round(psutil.Process().memory_info().rss / 1024 / 1024, 1)


def run_inference(model, img) -> int:
    """Run full tiling + inference + NMS on one image. Returns detection count."""
    H, W = img.shape[:2]
    stride = int(TILE_SIZE * (1 - OVERLAP))

    raw_boxes, raw_scores = [], []

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            x2 = min(x + TILE_SIZE, W)
            y2 = min(y + TILE_SIZE, H)
            tile = img[y:y2, x:x2]

            pad_h = TILE_SIZE - tile.shape[0]
            pad_w = TILE_SIZE - tile.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile = cv2.copyMakeBorder(
                    tile, 0, pad_h, 0, pad_w,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )

            results = model(tile, imgsz=TILE_SIZE, conf=CONF_THRESHOLD,
                            device=DEVICE, verbose=False)
            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue

            for box, score in zip(r.boxes.xyxy.cpu().numpy(),
                                  r.boxes.conf.cpu().numpy()):
                x1, y1, x2b, y2b = box
                raw_boxes.append([
                    float(x1 + x), float(y1 + y),
                    float(x2b + x), float(y2b + y)
                ])
                raw_scores.append(float(score))

    if not raw_boxes:
        return 0

    # NMS across all tiles
    boxes_xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in raw_boxes]
    indices = cv2.dnn.NMSBoxes(boxes_xywh, raw_scores, CONF_THRESHOLD, NMS_IOU)
    return len(indices)


def benchmark_model(name: str, pt_path: Path, img) -> dict:
    print(f"\n{'='*50}")
    print(f"[{name}] starting ... (device={DEVICE})")

    if not pt_path.exists():
        print(f"  SKIP: not found -> {pt_path}")
        return {}

    model_size = get_model_size_mb(pt_path)
    model = YOLO(str(pt_path), task="detect")

    # Warmup
    for _ in range(WARMUP_RUNS):
        run_inference(model, img)

    # Timed runs
    times_ms, peak_mem, last_count = [], 0.0, 0
    for i in range(TIMED_RUNS):
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        mem_before = get_rss_mb()
        t0 = time.perf_counter()

        count = run_inference(model, img)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        mem_after = get_rss_mb()

        times_ms.append(elapsed_ms)
        peak_mem = max(peak_mem, mem_after - mem_before)
        last_count = count
        print(f"  run {i+1:2d}: {elapsed_ms:7.1f} ms  detections={count}")

    avg_ms = statistics.mean(times_ms)
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

    row = {
        "model":         name,
        "format":        "pt",
        "device":        DEVICE,
        "mAP@0.5":       MAP50.get(name, ""),
        "avg_ms":        round(avg_ms, 1),
        "std_ms":        round(std_ms, 1),
        "fps":           round(1000 / avg_ms, 3),
        "peak_mem_mb":   round(peak_mem, 1),
        "model_size_mb": model_size,
        "detections":    last_count,
        "tiled_runs":    TIMED_RUNS,
        "tile_size":     TILE_SIZE,
        "overlap":       OVERLAP,
        "conf":          CONF_THRESHOLD,
        "nms_iou":       NMS_IOU,
    }
    print(f"  DONE: avg={avg_ms:.1f}ms  FPS={row['fps']}  "
          f"mem={peak_mem:.1f}MB  size={model_size}MB  det={last_count}")
    return row


def save_csv(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "pt_results.csv"
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {path}")

    # Summary table
    print(f"\n{'Model':<16} {'mAP@0.5':>8} {'Avg(ms)':>10} {'FPS':>7} "
          f"{'Mem(MB)':>9} {'Size(MB)':>10} {'Det':>5}")
    print("-" * 70)
    for r in rows:
        print(f"{r['model']:<16} {r['mAP@0.5']:>8} {r['avg_ms']:>10} "
              f"{r['fps']:>7} {r['peak_mem_mb']:>9} {r['model_size_mb']:>10} "
              f"{r['detections']:>5}")


def main():
    print(f"YOLO11 .pt Benchmark — {DEVICE.upper()}")
    print(f"Image : {IMG_PATH}")
    print(f"Tile  : {TILE_SIZE}px  overlap={OVERLAP}  conf={CONF_THRESHOLD}  NMS_IOU={NMS_IOU}")

    img = cv2.imread(str(IMG_PATH))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {IMG_PATH}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    rows = []
    for name, pt_path in MODELS.items():
        row = benchmark_model(name, pt_path, img)
        if row:
            rows.append(row)

    if rows:
        save_csv(rows)
    else:
        print("No models completed. Check paths.")


if __name__ == "__main__":
    main()
