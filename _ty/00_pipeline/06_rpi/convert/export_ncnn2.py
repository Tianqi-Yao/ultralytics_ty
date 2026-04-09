from ultralytics import YOLO

# model = YOLO("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/04_final_pipeline/29_Model_Performance/model/swd_model_v5_nullImagesAdded_final_noAug_seed42/yolo11npt_20pct_null_images_add_rawData_list_train_val_test_4/weights/best.pt")
model = YOLO("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/04_final_pipeline/29_Model_Performance/model/swd_model_v5_nullImagesAdded_final_noAug_seed42/yolo11npt_20pct_null_images_add_rawData_list_train_val_test_8/weights/best.pt")
model.export(format="ncnn")
