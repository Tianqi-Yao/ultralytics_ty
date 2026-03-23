# run_batch.py
import papermill as pm
import os
from pathlib import Path

datasets = [
    "TEST2_air2_0701_0823",
    "TEST2_southfarm2_0712_0823",
]

outputs_dir = Path("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/a02_image_quaility_DL/a02_1_apply_cls_model/outputs")
input_path = Path("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/a02_image_quaility_DL/a02_1_apply_cls_model/a02_1_apply_cls_model.ipynb")
os.makedirs(outputs_dir, exist_ok=True)  # 确保输出目录存在

for ds_name in datasets:
    print(f"▶ 正在执行: {ds_name}")
    
    pm.execute_notebook(
        input_path=str(input_path),   # 原始 notebook
        output_path=str(outputs_dir / f"{input_path.stem}_{ds_name}.ipynb"),  # 输出 notebook
        parameters={"DATASET_NAME": ds_name, "OUTPUT_PATH": str(outputs_dir)},  # 传入参数
        log_output=True,  # 显示 notebook 输出日志
        progress_bar=False,  # 关闭 papermill 的进度条显示
    )
    print(f"✅ 完成: {ds_name}\n")