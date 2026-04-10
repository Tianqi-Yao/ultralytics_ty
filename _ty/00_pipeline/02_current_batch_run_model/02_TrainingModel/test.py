from pathlib import Path
import logging

from ultralytics import YOLO

import wandb
wandb.login(key="957096cc564005d5332d45e2da6a75838e1cc9ac")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("test.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

# ===== project name =====
PROJECT_NAME = "swd_model_v5_nullImagesAdded_final"
run_path = Path("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/02_current_batch_run_model/02_TrainingModel/02_yaml")
# ========================

# ==== 你需要手动填写的列表 ====
yaml_file_names = [  # 数据集 yaml 文件名（不要写后缀 .yaml）
    "20pct_null_images_add_rawData_list_train_val_test",
]
val_batch_sizes = [16]  # batch size 不影响评估指标，只跑一次即可

# 建议填写训练完成后的 best.pt 路径
# 每个模型变体（架构 × 训练batch）各一个，用于生成 test set 对比表
models = {
    # yolo11n
    "yolo11n_4": "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/02_current_batch_run_model/02_TrainingModel/output/swd_model_v5_nullImagesAdded_final_noAug_seed42/yolo11n.pt_20pct_null_images_add_rawData_list_train_val_test_4/weights/best.pt",
    "yolo11n_8": "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/02_current_batch_run_model/02_TrainingModel/output/swd_model_v5_nullImagesAdded_final_noAug_seed42/yolo11n.pt_20pct_null_images_add_rawData_list_train_val_test_8/weights/best.pt",
    "yolo11n_16": "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/02_current_batch_run_model/02_TrainingModel/output/swd_model_v5_nullImagesAdded_final_noAug_seed42/yolo11n.pt_20pct_null_images_add_rawData_list_train_val_test_16/weights/best.pt",
    # yolo11s
    "yolo11s_4": "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/02_current_batch_run_model/02_TrainingModel/output/swd_model_v5_nullImagesAdded_final_noAug_seed42/yolo11s.pt_20pct_null_images_add_rawData_list_train_val_test_4/weights/best.pt",
    "yolo11s_8": "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/02_current_batch_run_model/02_TrainingModel/output/swd_model_v5_nullImagesAdded_final_noAug_seed42/yolo11s.pt_20pct_null_images_add_rawData_list_train_val_test_8/weights/best.pt",
    "yolo11s_16": "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/02_current_batch_run_model/02_TrainingModel/output/swd_model_v5_nullImagesAdded_final_noAug_seed42/yolo11s.pt_20pct_null_images_add_rawData_list_train_val_test_16/weights/best.pt",
    # yolo11m
    "yolo11m_4": "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/02_current_batch_run_model/02_TrainingModel/output/swd_model_v5_nullImagesAdded_final_noAug_seed42/yolo11m.pt_20pct_null_images_add_rawData_list_train_val_test_4/weights/best.pt",
    "yolo11m_8": "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/02_current_batch_run_model/02_TrainingModel/output/swd_model_v5_nullImagesAdded_final_noAug_seed42/yolo11m.pt_20pct_null_images_add_rawData_list_train_val_test_8/weights/best.pt",
    "yolo11m_16": "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/02_current_batch_run_model/02_TrainingModel/output/swd_model_v5_nullImagesAdded_final_noAug_seed42/yolo11m.pt_20pct_null_images_add_rawData_list_train_val_test_16/weights/best.pt",
    # yolo11l
    "yolo11l_4": "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/02_current_batch_run_model/02_TrainingModel/output/swd_model_v5_nullImagesAdded_final_noAug_seed42/yolo11l.pt_20pct_null_images_add_rawData_list_train_val_test_4/weights/best.pt",
    "yolo11l_8": "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/02_current_batch_run_model/02_TrainingModel/output/swd_model_v5_nullImagesAdded_final_noAug_seed42/yolo11l.pt_20pct_null_images_add_rawData_list_train_val_test_8/weights/best.pt",
    "yolo11l_16": "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/02_current_batch_run_model/02_TrainingModel/output/swd_model_v5_nullImagesAdded_final_noAug_seed42/yolo11l.pt_20pct_null_images_add_rawData_list_train_val_test_16/weights/best.pt",
    # yolo１１x（无 b１６ 训练结果，用 b４/b８）
    "yolo11x_4": "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/02_current_batch_run_model/02_TrainingModel/output/swd_model_v5_nullImagesAdded_final_noAug_seed42/yolo11x.pt_20pct_null_images_add_rawData_list_train_val_test_4/weights/best.pt",
    "yolo11x_8": "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/02_current_batch_run_model/02_TrainingModel/output/swd_model_v5_nullImagesAdded_final_noAug_seed42/yolo11x.pt_20pct_null_images_add_rawData_list_train_val_test_8/weights/best.pt",
}
# =================================

for yaml_file_name in yaml_file_names:
    yaml_path = run_path / (yaml_file_name + ".yaml")
    if not yaml_path.exists():
        logger.error(f"数据集配置不存在，跳过: {yaml_path}")
        continue

    for model_name, model_file in models.items():
        model_path = Path(model_file)
        if not model_path.exists():
            logger.warning(f"权重不存在，跳过: {model_file}")
            continue

        for val_batch in val_batch_sizes:
            logger.info(
                f"开始测试: model={model_file}, dataset={yaml_file_name}, batch={val_batch}"
            )

            model = YOLO(str(model_path))

            try:
                model.val(
                    data=str(yaml_path),
                    split="test",
                    imgsz=640,
                    batch=val_batch,
                    device=0,
                    workers=4,
                    project=f"/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/00_pipeline/02_current_batch_run_model/02_TrainingModel/output/{PROJECT_NAME}_test_only",
                    name=f"{model_name}_test_b{val_batch}",
                )
                logger.info(
                    f"测试完成: model={model_file}, dataset={yaml_file_name}, batch={val_batch}"
                )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.warning(
                        f"跳过: model={model_file}, yaml={yaml_file_name}, batch={val_batch} —— 显存不足"
                    )
                    continue
                logger.error(f"测试失败: model={model_file}, error={e}")
                raise
