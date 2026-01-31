from ultralytics import YOLO

# Load a model
# model = YOLO("/workspace/_ty/02_models/best_models/91_swd_cls/data_split_0.6_0.4_0.0_yolo11m-cls_4.pt")  # pretrained YOLO11n model
model = YOLO("/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/02_models/best_models/91_swd_cls/data_split_0.6_0.4_0.0_yolo11m-cls_4.pt")  # pretrained YOLO11n model


# images_folder_path = "/workspace/_ty/01_data/00_test/00_try/del/debug"
images_folder_path = "/home/tianqi/D/01_Projects/01_swd/02_code/pipeline/ultralytics_ty/_ty/01_data/00_test/00_try/del/debug"

images_path_list = []
import os
for filename in os.listdir(images_folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        images_path_list.append(os.path.join(images_folder_path, filename))
# print(f"images_path_list: {images_path_list}")

# Run batched inference on a list of images
results = model(images_path_list)  # return a list of Results objects

# Process results list
for i,result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    result.save(filename=f"{images_folder_path}/pred1/result_{i}.jpg")  # save to disk