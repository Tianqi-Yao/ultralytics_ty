# %% [markdown]
# # 训练分类模型

# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ultralytics import YOLO
from pathlib import Path

# ===== 你自己改的部分 =====
dataFolderNameList = [
    "swd_pose_split_0.7_0.2_0.1",
    "swd_pose_split_0.6_0.2_0.2",
    "swd_pose_split_0.5_0.3_0.2",
    "swd_pose_split_0.5_0.2_0.3",
    "swd_pose_split_0.4_0.3_0.3",
]
batchSizes = [8, 16, 32]
models = ["yolo11n-pose.pt", "yolo11s-pose.pt", 
        "yolo11m-pose.pt", "yolo11l-pose.pt", "yolo11x-pose.pt"
        ]

# 你的数据集根（每个数据集文件夹里应该有 swd_pose.yaml）
RUN_DATA_ROOT = Path("/workspace/models/runs_yolov11_pose/datasets")
# 训练输出根
PROJECT_ROOT = Path("/workspace/models/runs_yolov11_pose/outputs")

IMGSZ_INIT = 640      # 小黑点建议 >= 896/1024/1280
EPOCHS = 1000           # 一般 150~300 就够；太长反而过拟合
WORKERS = 4
SEED = 42
# =========================


def find_yaml(folder: Path) -> Path:
    """在数据集文件夹下自动找 YAML。优先 swd_pose.yaml，其次唯一 *.yaml。"""
    if folder.is_file() and folder.suffix == ".yaml":
        return folder
    if not folder.exists():
        raise FileNotFoundError(f"数据集目录不存在: {folder}")

    y1 = folder / "swd_pose.yaml"
    if y1.exists(): 
        return y1

    yamls = list(folder.glob("*.yaml"))
    if len(yamls) == 1:
        return yamls[0]
    elif len(yamls) == 0:
        raise FileNotFoundError(f"找不到 YAML: {folder} 下没有 *.yaml")
    else:
        raise FileNotFoundError(f"找到多个 YAML，请保留一个或改名 swd_pose.yaml: {yamls}")


def has_nonempty_dir(p: Path) -> bool:
    return p.exists() and any(p.iterdir())


def train_one_combo(model_file: str, yaml_path: Path, batch: int):
    """带 OOM 自动降级重试：先降 batch，再降 imgsz。"""
    model = YOLO(model_file)
    current_bs = batch
    current_imgsz = IMGSZ_INIT

    run_name_base = f"{Path(model_file).stem}_{yaml_path.parent.name}_bs{batch}_sz{IMGSZ_INIT}"

    while True:
        print(f"\n🚀 Training model={model_file}  data={yaml_path}  batch={current_bs}  imgsz={current_imgsz}")
        try:
            model.train(
                data=str(yaml_path),
                epochs=EPOCHS,
                imgsz=current_imgsz,
                batch=current_bs,
                device=0,
                workers=WORKERS,
                seed=SEED,
                project=str(PROJECT_ROOT / Path(model_file).stem),
                name=f"{run_name_base}",
                # —— pose 任务更稳的增强 —— #
                mosaic=0,              # 马赛克关，避免糊小点
                perspective=0.0,
                erasing=0.0,
                fliplr=0.5,            # 和 flip_idx 配合
                degrees=5.0,
                scale=0.2,
                # —— 早停/保存 —— #
                patience=30,           # 早停容忍
                save=True,
                save_period=25,        # 每隔 N epoch 存一次
                plots=True,
            )
            break  # 训练成功
        except RuntimeError as e:
            msg = str(e)
            if "CUDA out of memory" in msg:
                # 先减半 batch；如果已经 1，再降 imgsz（到不低于 640，32 对齐）
                if current_bs > 1:
                    current_bs = max(1, current_bs // 2)
                    print(f"⚠️  OOM，降 batch 重试：batch={current_bs}")
                    continue
                elif current_imgsz > 640:
                    current_imgsz = max(640, int(current_imgsz * 0.8 // 32 * 32))
                    print(f"⚠️  OOM，降 imgsz 重试：imgsz={current_imgsz}")
                    continue
                else:
                    print(f"⏭️  跳过该组合（仍 OOM）：{model_file} @ {yaml_path}")
                    return
            else:
                raise  # 其它错误直接抛出

    # —— 评测（有 test 就测 test，没有就测 val）——
    ds_root = yaml_path.parent
    test_dir = ds_root / "images" / "test"
    split = "test" if has_nonempty_dir(test_dir) else "val"
    print(f"🧪 Evaluating on split={split} ...")
    model.val(
        data=str(yaml_path),
        split=split,
        imgsz=current_imgsz,
        batch=min(current_bs, 16),
        project=str(PROJECT_ROOT / Path(model_file).stem),
        name=f"{run_name_base}_{split}",
    )


if __name__ == "__main__":
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

    for modelFile in models:
        for dataFolderName in dataFolderNameList:
            yaml_path = find_yaml(RUN_DATA_ROOT / dataFolderName)
            for bs in batchSizes:
                train_one_combo(modelFile, yaml_path, bs)

    print("\n✅ All jobs finished.")



