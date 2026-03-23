# run_batch.py
import papermill as pm
import os
from pathlib import Path

datasets = [
    "sahi_null_v2_ms1_0605-0621_40_ok",
    "sahi_null_v2_ms1_0710-0726_36_ok",
    "sahi_null_v2_ms1_0726-0809_11_ok",
]

outputs_dir = Path("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/03_code/04_fiftyone/tools/outputs")
input_path = Path("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/03_code/04_fiftyone/tools/12_data_quality_check.ipynb")
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