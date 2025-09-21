# %% [markdown]
# # 查看根目录下需要运行的文件夹

# %%
from pathlib import Path

version = "v9"

if __name__ == "__main__":
    # === 1) 给定一个根目录 ===
    root_dir = Path("/workspace/models/SAHI/run_v7")
    # end_with = "_sliced"
    end_with = "_data"

    # === 2) 遍历所有子目录 ===
    sub_dirs = list(root_dir.glob("**/*" + end_with))


    if not sub_dirs:
        print(f"没有找到 *{end_with} 目录")
        exit(0)

    print(f"找到以下 {end_with} 数据集：")
    for i, d in enumerate(sub_dirs):
        print(f"[{i}] {d}")

    # === 3) 让你选择要跑的目录 ===
    idx_str = input("请输入要处理的编号 (多个用逗号分隔, 回车默认全选): ").strip()
    if idx_str:
        indices = [int(x) for x in idx_str.split(",")]
        chosen_dirs = [sub_dirs[i] for i in indices]
    else:
        chosen_dirs = sub_dirs
    
    print(f"将处理以下 {end_with} 目录：")
    for i, d in enumerate(chosen_dirs):
        print(f"- {i+1}. {d}")

    # 如果有的文件夹的raw_data里面没有图片，就移除
    chosen_dirs = [d for d in chosen_dirs if (d.parent / "raw_data").exists() and any((d.parent / "raw_data").glob("*.jpg"))]
    if not chosen_dirs:
        print(f"没有找到包含图片的 *{end_with} 目录")
        exit(0)

# %%
import os
import re
import json
from collections import Counter
from ultralytics import YOLO
from pathlib import Path


MODEL_PATH = "/workspace/models/best_model/yolo11m-cls-best_v8.pt"

# 解析文件名的正则表达式
UUID_RE = re.compile(r"uuid_([a-f0-9\-]+)\.jpg", re.IGNORECASE)
ORIG_RE = re.compile(r"^(\d+_\d+_\d+)_obj", re.IGNORECASE)

def get_probs_fields(res, name_map):
    """安全获取 top1/top5 字段（缺失时给空值/空列表）。"""
    probs = getattr(res, "probs", None)
    if probs is None:
        return None, None, None, [], [], []

    # top1
    try:
        top1_id = int(probs.top1)
    except Exception:
        top1_id = None
    top1_name = name_map.get(top1_id) if top1_id is not None else None

    # top1 conf
    try:
        top1_conf = float(getattr(probs.top1conf, "item", lambda: probs.top1conf)())
    except Exception:
        top1_conf = None

    # top5
    try:
        top5_id = [int(x) for x in list(probs.top5)]
    except Exception:
        top5_id = []
    top5_name = [name_map.get(i, str(i)) for i in top5_id]
    try:
        top5_conf = [float(x) for x in list(probs.top5conf)]
    except Exception:
        top5_conf = []

    return top1_id, top1_name, top1_conf, top5_id, top5_name, top5_conf


# ==== 主流程 ====
if __name__ == "__main__":
    # 1) 加载模型
    model = YOLO(MODEL_PATH)
    class_names = model.names  # dict: {0: "...", 1: "..."}

    for d in chosen_dirs:
        print(f"\n=== 处理目录: {d} ===")
        input_dir  = str(d) + "_sliced_merge/cropped_objects/"
        output_json = str(d) + f"_sliced_merge/classification_predicted_results_{version}.json"

        # 确保输入目录不为空
        if not any(Path(input_dir).glob("*.jpg")):
            print(f"⚠️ 输入目录 {input_dir} 没有 JPG 文件，跳过")
            continue

        # 2) 执行预测（Ultralytics 支持目录）
        results = model(input_dir)

        data = []
        counts = Counter()

        for res in results:
            path = getattr(res, "path", "")
            fname = os.path.basename(path)

            # 提取 uuid / 原图名
            uuid_match = UUID_RE.search(fname)
            uuid_str = uuid_match.group(1) if uuid_match else None
            orig_match = ORIG_RE.match(fname)
            original_name = orig_match.group(1) if orig_match else None

            # 概率字段
            top1_id, top1_name, top1_conf, top5_id, top5_name, top5_conf = get_probs_fields(res, class_names)
            if top1_id is not None:
                counts[top1_id] += 1

            # 记录一条
            data.append({
                "path": path,
                "uuid": uuid_str,
                "original_name": original_name,
                "top1_id": top1_id,
                "top1_name": top1_name,
                "top1_conf": top1_conf,
                "top5_id": top5_id,
                "top5_name": top5_name,
                "top5_conf": top5_conf,
            })

        # 3) 打印统计
        print("分类统计结果：")
        for cls_id, num in counts.items():
            print(f"{class_names.get(cls_id, cls_id)}: {num}")
        total = sum(counts.values())
        print(f"总计: {total}")

        # 4) 保存 JSON（保持与你原来一致的结构）
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ 已保存到 {output_json}")


# %% [markdown]
# # 0302

# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import orjson
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter

# ========= JSON I/O =========
def jload(fp: Path):
    with fp.open("rb") as f:
        return orjson.loads(f.read())

def jdump_to_file(obj, fp: Path):
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("wb") as f:
        f.write(orjson.dumps(obj))

# ========= 工具函数 =========
def load_cls_map(path: Path) -> Dict[str, Dict]:
    """
    读取分类结果，返回 uuid -> item 的映射。
    允许两种结构：
    1) {"results": [...]}
    2) [ ... ]
    item 至少应包含: {"uuid": "...", "top1_name": "...",  ...}
    """
    data = jload(path)
    if isinstance(data, dict) and "results" in data:
        items = data["results"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("classification_predicted_results.json 结构不支持")
    return {it["uuid"]: it for it in items if "uuid" in it}

def build_image_index(img_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in img_dir.glob(ext):
            index[p.stem] = p
    return index

def filter_annotations(
    merged_annotations: Dict[str, List[dict]],
    cls_map: Dict[str, dict],
    keep: Optional[List[str]],
    drop: Optional[List[str]],
    relabel: bool
) -> Tuple[Dict[str, List[dict]], Tuple[int, int, int]]:
    out: Dict[str, List[dict]] = {}
    kept = dropped = not_found = 0
    keep_set = set(keep) if keep else None
    drop_set = set(drop) if drop else None

    for img, anns in tqdm(merged_annotations.items(), desc="筛选标注中", ncols=80):
        new_list: List[dict] = []
        for ann in anns:
            u = ann.get("uuid")
            if not u:
                continue
            pred = cls_map.get(u)
            if pred is None:
                not_found += 1
                continue

            top1 = pred.get("top1_name")
            if keep_set is not None and top1 not in keep_set:
                dropped += 1
                continue
            if drop_set is not None and top1 in drop_set:
                dropped += 1
                continue

            if relabel and top1:
                ann2 = dict(ann)
                ann2["label"] = top1
                new_list.append(ann2)
            else:
                new_list.append(ann)
            kept += 1

        if new_list:
            out[img] = new_list

    return out, (kept, dropped, not_found)

# ========= 可视化 =========
LINE_COLOR_BGR = (0, 255, 255)
LINE_WIDTH = 1
TEXT_SCALE = 0.7
TEXT_THICKNESS = 2
TEXT_COLOR_BGR = (0, 255, 255)
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_DY = 18

def _draw_one_image(image_name: str, anns: List[dict], img_index: Dict[str, str], out_dir: str):
    img_path = img_index.get(image_name) or img_index.get(Path(image_name).stem)
    if not img_path:
        return None
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None

    for i, ann in enumerate(anns):
        pts = ann["points"]
        pts_np = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts_np], True, LINE_COLOR_BGR, LINE_WIDTH)

        label = str(ann.get("label", ""))
        s = ann.get("score")
        label_txt = f"{label} {s:.3f}" if isinstance(s, (float, int)) else label
        x0, y0 = int(pts[0][0]), int(pts[0][1])
        cv2.putText(img, label_txt, (x0 + 6, y0 + 6 + (i % 3) * TEXT_DY),
                    TEXT_FONT, TEXT_SCALE, TEXT_COLOR_BGR, TEXT_THICKNESS, cv2.LINE_AA)

    out_path = str(Path(out_dir) / f"{Path(image_name).stem}_vis.jpg")
    cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return out_path

def visualize_parallel(annotations: Dict[str, List[dict]], original_dir: Path, out_dir: Path, max_workers: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    img_index = {k: str(v) for k, v in build_image_index(original_dir).items()}
    total = len(annotations)
    if total == 0:
        return
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_draw_one_image, image_name, anns, img_index, str(out_dir))
            for image_name, anns in annotations.items()
        ]
        for _ in tqdm(as_completed(futures), total=total, desc="可视化中", ncols=80):
            pass

# ========= 小图整理（含 others）=========
_UUID_RE = re.compile(r"uuid_([a-f0-9\-]+)\.(jpg|jpeg|png)$", re.IGNORECASE)

def sort_cropped_objects(
    cropped_dir: Path,
    out_root: Path,
    cls_map: Dict[str, dict],
    keep: Optional[List[str]],
    drop: Optional[List[str]],
    max_workers: int
) -> Tuple[int, int, int, int, Counter]:
    """
    将 cropped_objects 里的图片按 top1_name 归类复制到 out_root/<class_name>/ 下，
    未命中 keep/drop 或无预测 / 文件名不匹配的，统一放到 out_root/others/ 下。
    返回: (总文件数, 符合条件复制数, 无预测数, others数, 各类别计数)
    """
    if not cropped_dir.exists():
        print(f"⚠️ 未找到裁剪目录: {cropped_dir}")
        return 0, 0, 0, 0, Counter()

    files: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(cropped_dir.glob(ext))
    total = len(files)
    if total == 0:
        print(f"⚠️ 裁剪目录为空: {cropped_dir}")
        return 0, 0, 0, 0, Counter()

    copied_kept = 0
    no_pred = 0
    others = 0
    per_class = Counter()
    keep_set = set(keep) if keep else None
    drop_set = set(drop) if drop else None

    def to_dir(d: Path):
        d.mkdir(parents=True, exist_ok=True)
        return d

    def task(path: Path):
        nonlocal copied_kept, no_pred, others
        m = _UUID_RE.search(path.name)
        if not m:
            # 文件名不符合规则 → others
            dst = to_dir(out_root / "others")
            shutil.copy2(path, dst / path.name)
            per_class.update(["others"])
            others += 1
            return

        uuid = m.group(1)
        pred = cls_map.get(uuid)
        if pred is None:
            # 无预测 → others
            dst = to_dir(out_root / "others")
            shutil.copy2(path, dst / path.name)
            per_class.update(["others"])
            others += 1
            no_pred += 1
            return

        top1 = pred.get("top1_name")
        # 规则判断
        if keep_set is not None and top1 not in keep_set:
            dst = to_dir(out_root / "others")
            shutil.copy2(path, dst / path.name)
            per_class.update(["others"])
            others += 1
            return
        if drop_set is not None and top1 in drop_set:
            dst = to_dir(out_root / "others")
            shutil.copy2(path, dst / path.name)
            per_class.update(["others"])
            others += 1
            return

        # ✅ 符合条件，复制到对应类别
        subdir = top1 or "unknown"
        dst = to_dir(out_root / subdir)
        shutil.copy2(path, dst / path.name)
        per_class.update([subdir])
        copied_kept += 1

    out_root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(tqdm(ex.map(task, files), total=total, desc="整理小图中", ncols=80))

    return total, copied_kept, no_pred, others, per_class

# %%
def main():
    # ---- 过滤策略 ----
    DROP_CLASSES: Optional[List[str]] = None
    KEEP_CLASSES: Optional[List[str]] = ["swd", "mayswd"]
    RENAME_WITH_PRED: bool = True

    # ---- 开关与并发 ----
    DO_VISUALIZE = True
    DO_SORT_CROPPED = True
    MAX_WORKERS = min(max(1, (os.cpu_count() or 2) - 1), 8)


    for d in chosen_dirs:
        print(f"\n=== 处理目录: {d} ===")   
        
        # ---- 路径配置 ----
        CLASSIFICATION_JSON = d.parent / (d.name + "_sliced_merge") / "classification_predicted_results.json"
        MERGED_ANN_JSON     = d.parent / (d.name + "_sliced_merge") / "merged_annotations.json"
        FILTERED_ANN_JSON   = d.parent / (d.name + "_sliced_merge") / f"filtered_annotations_{version}.json"
        ORIGINAL_IMAGE_DIR  = d.parent / (d.name)
        OUT_VIS_DIR         = d.parent / (d.name + "_sliced_merge") / f"filtered_visualizations_{version}"
        CROPPED_OBJECTS_DIR = d.parent / (d.name + "_sliced_merge") / "cropped_objects"
        FILTERED_CROPPED_DIR= d.parent / (d.name + "_sliced_merge") / f"filtered_cropped_objects_{version}"

        # 确保路径存在
        if not any(ORIGINAL_IMAGE_DIR.glob("*.jpg")):
            print(f"⚠️ 跳过，原图目录不存在或无图片: {ORIGINAL_IMAGE_DIR}")
            continue

        # 读取分类与合并标注
        cls_map = load_cls_map(CLASSIFICATION_JSON)
        merged = jload(MERGED_ANN_JSON)

        # 筛选标注并可选重标
        filtered, (kept, dropped, not_found) = filter_annotations(
            merged_annotations=merged,
            cls_map=cls_map,
            keep=KEEP_CLASSES,
            drop=DROP_CLASSES,
            relabel=RENAME_WITH_PRED
        )
        print(f"\n[标注筛选统计] kept: {kept}, dropped: {dropped}, uuid_without_pred: {not_found}")

        jdump_to_file(filtered, FILTERED_ANN_JSON)
        print(f"✅ 已保存筛选后的标注: {FILTERED_ANN_JSON}")

        # 可视化
        if DO_VISUALIZE:
            visualize_parallel(filtered, ORIGINAL_IMAGE_DIR, OUT_VIS_DIR, max_workers=MAX_WORKERS)
            print(f"✅ 已保存可视化结果: {OUT_VIS_DIR}")

        # 小图分类（含 others）
        if DO_SORT_CROPPED:
            total, copied_kept, no_pred, others, per_class = sort_cropped_objects(
                CROPPED_OBJECTS_DIR,
                FILTERED_CROPPED_DIR,
                cls_map=cls_map,
                keep=KEEP_CLASSES,
                drop=DROP_CLASSES,
                max_workers=MAX_WORKERS
            )
            print("\n[裁剪小图整理统计]")
            print(f"  总计扫描: {total}")
            print(f"  符合规则并归类: {copied_kept}")
            print(f"  无预测放入 others: {no_pred}")
            print(f"  其余（不在 keep 或命中 drop、文件名不合规等）放入 others: {others - no_pred}")
            if per_class:
                print("  各类别计数：")
                for k, v in per_class.most_common():
                    print(f"    {k}: {v}")
            print(f"✅ 已分类保存到: {FILTERED_CROPPED_DIR}")

if __name__ == "__main__":
    main()


