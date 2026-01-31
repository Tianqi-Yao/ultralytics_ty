import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob

label_field = "01_swd_seg_results_coco"

# embedding模型
models = ["clip-vit-base32-torch", "dinov2-vits14-torch", "resnet50-imagenet-torch", "mobilenet-v2-imagenet-torch"]
# 降维方法
dim_reduction_methods = ["umap", 
                            # "tsne", "pca"
                        ]

embeddings_fields = ["emb_clip", "emb_dinov2", "emb_resnet50", "emb_mobilenet"]


datasets = [  'ms1_0710-0726_36_ok_v22',
 'ms1_0809-0823_34_ok_v22',
 'ms2_0726-0809_13_ok_v22',
 'sw1_0605-0613_07_ok_v22']

# 清理已有的 brain runs，避免冲突
for dataset_name in datasets:
    dataset = fo.load_dataset(dataset_name)
    for key in dataset.list_brain_runs():
        dataset.delete_brain_run(key)


for model_name, emb_field in zip(models, embeddings_fields):
    model = foz.load_zoo_model(model_name)

    for dataset_name in datasets:
        print(f"Dataset: {dataset_name}")
        dataset = fo.load_dataset(dataset_name)

        # 1) 对每个 ann 直接算 patch embedding（按 bbox/mask 裁剪，不导出图片）
        dataset.compute_patch_embeddings(       # ***************关键区别****************
            model,
            patches_field=label_field,   # 关键：按这个字段里的 bbox/mask 作为 patch
            embeddings_field=emb_field,      # embedding 存在每个 ann 的 .emb 里
        )

        for method in dim_reduction_methods:
            brain_key = f"patches_{model_name.split('-')[0]}_{method}_v1"

            # 2) 对所有 patch 做 降维 可视化
            fob.compute_visualization(
                dataset,
                patches_field=label_field,   # 告诉 brain 这是 patch 字段
                embeddings=emb_field,            # 用上一步算好的 embedding 字段
                method=method,                # 先用 pca，规避 umap/numba 问题
                seed=51,
                brain_key=brain_key,  # 每个 dataset 自己有一份同名 brain_key 就行
            )