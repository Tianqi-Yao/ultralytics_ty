import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob

# embedding 模型
models = [
    "clip-vit-base32-torch",
    "dinov2-vits14-torch",
    "resnet50-imagenet-torch",
    "mobilenet-v2-imagenet-torch",
]

# 降维方法
dim_reduction_methods = ["umap", "tsne", "pca"]

# 对应存 embedding 的字段名（存在 sample 上）
embeddings_fields = ["emb_clip", "emb_dinov2", "emb_resnet50", "emb_mobilenet"]

# 现在是 patch 小图 dataset
datasets = [
    # "ms1_0809-0823_34_ok_v22_conf09_patches",
    "ms1_0710-0726_36_ok_v22_conf09_patches"
]

# 先清理 brain runs，避免重名冲突
for dataset_name in datasets:
    dataset = fo.load_dataset(dataset_name)
    for key in dataset.list_brain_runs():
        dataset.delete_brain_run(key)

# 主循环：模型 × 数据集 × 降维方法
for model_name, emb_field in zip(models, embeddings_fields):
    print(f"\n========== Embedding model: {model_name} ==========")
    model = foz.load_zoo_model(model_name)

    for dataset_name in datasets:
        print(f"\nDataset: {dataset_name}")
        dataset = fo.load_dataset(dataset_name)

        # 1) 对每张 patch 小图计算 embedding（存在 sample 级别）
        print(f"  -> Computing embeddings into field `{emb_field}` ...")
        dataset.compute_embeddings(             # ***************关键区别****************
            model,
            embeddings_field=emb_field,
        )

        # 2) 对 embedding 做降维可视化
        for method in dim_reduction_methods:
            brain_key = f"{model_name.split('-')[0]}_{method}_v1"
            print(f"  -> Computing visualization: {brain_key}")

            fob.compute_visualization(
                dataset,
                embeddings=emb_field,   # 使用上面算好的 embedding 字段
                method=method,
                seed=51,
                brain_key=brain_key,
            )

print("\n✅ All embeddings & visualizations done.")
