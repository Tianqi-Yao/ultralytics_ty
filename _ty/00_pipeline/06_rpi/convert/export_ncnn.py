"""
export_ncnn.py
==============
在服务器上运行，将 5 个最佳 YOLO11 .pt 模型导出为 ncnn 格式。
导出完成后，将整个 ncnn_models/ 目录复制到 RPi 上运行 benchmark。

运行方式：
    python export_ncnn.py

输出目录结构：
    ncnn_models/
    ├── yolo11n_b8/
    │   ├── best.ncnn.param
    │   └── best.ncnn.bin
    ├── yolo11s_b4/
    ├── yolo11m_b16/
    ├── yolo11l_b4/
    └── yolo11x_b8/
"""

from pathlib import Path
from ultralytics import YOLO

# ─── 请修改为实际路径 ──────────────────────────────────────────────────────────
MODEL_BASE = Path("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/04_final_pipeline/29_Model_Performance/model/swd_model_v5_nullImagesAdded_final_noAug_seed42")

# 5 个代表性模型：(模型名, batch, 训练目录子路径)
# 子路径填实际目录名（含 batch 后缀），例如 "yolo11n.pt_batch8_xxxxx"
MODELS = {
    "yolo11n_b8": MODEL_BASE / "yolo11npt_20pct_null_images_add_rawData_list_train_val_test_8" / "weights" / "best.pt",
    "yolo11s_b4": MODEL_BASE / "yolo11spt_20pct_null_images_add_rawData_list_train_val_test_4" / "weights" / "best.pt",
    "yolo11m_b16": MODEL_BASE / "yolo11mpt_20pct_null_images_add_rawData_list_train_val_test_16" / "weights" / "best.pt",
    "yolo11l_b4": MODEL_BASE / "yolo11lpt_20pct_null_images_add_rawData_list_train_val_test_4" / "weights" / "best.pt",
    "yolo11x_b8": MODEL_BASE / "yolo11xpt_20pct_null_images_add_rawData_list_train_val_test_8" / "weights" / "best.pt",
}

OUTPUT_DIR = Path("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/06_rpi/convert/ncnn_models")
# ──────────────────────────────────────────────────────────────────────────────


def export_model(name: str, pt_path: Path, output_dir: Path) -> None:
    out = output_dir / name
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[{name}] 导出 {pt_path} → ncnn ...")
    model = YOLO(str(pt_path))
    # half=False：RPi 4B CPU 不支持 FP16
    model.export(format="ncnn")

    # Ultralytics 默认把 ncnn 文件输出在 pt 同级目录，移动到目标路径
    ncnn_dir = pt_path.parent / (pt_path.stem + "_ncnn_model")
    if ncnn_dir.exists():
        import shutil
        shutil.move(str(ncnn_dir), str(out / "best_ncnn_model"))
        print(f"[{name}] 已移动到 {out / 'best_ncnn_model'}")
    else:
        print(f"[{name}] ⚠️  未找到 ncnn 目录，请手动检查：{ncnn_dir}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, pt_path in MODELS.items():
        if not pt_path.exists():
            print(f"[{name}] ⚠️  跳过：文件不存在 → {pt_path}")
            continue
        export_model(name, pt_path, OUTPUT_DIR)

    print(f"\n✅ 全部完成，ncnn 模型目录：{OUTPUT_DIR}")
    print("请将整个 ncnn_models/ 目录复制到 RPi 后运行 rpi/benchmark.py")


if __name__ == "__main__":
    main()
