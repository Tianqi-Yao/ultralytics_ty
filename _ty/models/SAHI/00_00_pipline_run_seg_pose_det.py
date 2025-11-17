# %% [markdown]
# # 通过seg模型将疑似SWD的object标注出来。

# %%
version = "v1"
run_type = "pose_and_det"  # "pose_and_det" or "cls"

# 需要运行的step列表
steps_to_run = [
    "run_clean_and_slice_images_on_dirs",           # Step 1 清理坏图并切片大图到640*640小图
    "process_sliced_images_with_yolo_seg",          # Step 2 使用YOLO-seg模型处理640*640切片图像
    "combine_sliced_predictions",                   # Step 3 合并seg预测结果，回到原图。 同时切出objects小图
]

# %% [markdown]
# # Step_0 查看根目录下需要运行的文件夹 
# 选择含有 raw_data 图片的 *_data 目录
# 
# ![image.png](attachment:image.png)
# 

# %%
# 选择数据目录的核心代码
from pathlib import Path

def select_data_dirs(root_dir: Path, end_with: str = "_data"):
    # === 1) 遍历所有子目录 ===
    sub_dirs = list(root_dir.glob("*/*" + end_with))

    if not sub_dirs:
        print(f"没有找到 *{end_with} 目录")
        return []

    print(f"找到以下 {end_with} 数据集：")
    for i, d in enumerate(sub_dirs):
        print(f"[{i}] {d}")

    # === 2) 让用户选择要跑的目录 ===
    idx_str = input("请输入要处理的编号 (多个用逗号分隔, 回车默认全选): ").strip()
    if idx_str:
        indices = [int(x) for x in idx_str.split(",")]
        chosen_dirs = [sub_dirs[i] for i in indices]
    else:
        chosen_dirs = sub_dirs

    print(f"将处理以下 {end_with} 目录：")
    for i, d in enumerate(chosen_dirs):
        print(f"- {i+1}. {d}")

    # === 3) 筛选掉没有 raw_data 图片的目录 ===
    chosen_dirs = [
        d for d in chosen_dirs
        if (d.parent / "raw_data").exists() and any((d.parent / "raw_data").glob("*.jpg"))
    ]

    if not chosen_dirs:
        print(f"没有找到包含图片的 *{end_with} 目录")
        return []

    return chosen_dirs

# %%
root_dir = Path("/workspace/models/SAHI/run_v8")
chosen_dirs = select_data_dirs(root_dir, end_with="_data")
print("最终确认的目录：", chosen_dirs)
if not chosen_dirs:
    raise ValueError("没有选择任何目录，程序终止。")

# %% [markdown]
# # Step_1 将文件夹中的RAW图片全部切片640x640并保存
# 
# ### 输入
# ![image.png](attachment:image.png)
# 
# ### 输出
# ![image-2.png](attachment:image-2.png)

# %%
# Step 1 清理坏图并切片

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import cv2

# ============================================================
# 基本配置
# ============================================================
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ============================================================
# 1) 单张图无损切片（输出 PNG）
# ============================================================
def slice_image_cv2(
    image_path: Path,
    output_dir: Path,
    tile_h: int = 640,
    tile_w: int = 640,
    overlap: float = 0.2,
    out_ext: str = ".jpg",           # ✅ JPG：有损
    jpeg_quality: int = 95,          # 对 PNG 无效，保留参数便于兼容
    png_compression: int = 3,        # 0(最快,大)~9(最慢,小)，3~5 较均衡
    keep_small_edge: bool = True,    # 末端不足一片时仍保存小片
) -> int:
    """
    读取一张图并切片到 output_dir，返回保存的切片数。
    文件命名：<stem>_x0_y0_x1_y1.<ext>（与 SAHI 基本兼容）
    """
    # 读图：np.fromfile + imdecode 更稳更快（兼容中文路径等）
    buf = np.fromfile(str(image_path), dtype=np.uint8)
    if buf.size == 0:
        return 0
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return 0

    H, W = img.shape[:2]
    out_ext = out_ext.lower()
    os.makedirs(output_dir, exist_ok=True)

    # 步长（带重叠）
    overlap = max(0.0, min(0.99, overlap))
    sh = max(1, int(round(tile_h * (1.0 - overlap))))
    sw = max(1, int(round(tile_w * (1.0 - overlap))))

    # 生成起点，确保右/下边缘覆盖
    ys = list(range(0, max(1, H - tile_h + 1), sh))
    xs = list(range(0, max(1, W - tile_w + 1), sw))
    if keep_small_edge:
        if ys[-1] != max(0, H - tile_h):
            ys.append(max(0, H - tile_h))
        if xs[-1] != max(0, W - tile_w):
            xs.append(max(0, W - tile_w))

    # 写图参数
    if out_ext in (".jpg", ".jpeg"):
        imwrite_params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    elif out_ext == ".png":
        imwrite_params = [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)]
    else:
        imwrite_params = []

    stem = image_path.stem
    saved = 0

    for y0 in ys:
        y1 = min(y0 + tile_h, H)
        for x0 in xs:
            x1 = min(x0 + tile_w, W)
            crop = img[y0:y1, x0:x1]
            out_name = f"{stem}_{x0}_{y0}_{x1}_{y1}{out_ext}"
            out_path = output_dir / out_name
            try:
                cv2.imwrite(str(out_path), crop, imwrite_params)
                saved += 1
            except Exception:
                pass
    return saved


# ============================================================
# 2) 文件夹批量切片（并行按“图”）
# ============================================================
def slice_folder_cv2(
    input_folder: Path,
    output_folder: Optional[Path] = None,
    tile_h: int = 640,
    tile_w: int = 640,
    overlap: float = 0.2,
    out_ext: str = ".jpg",           # ✅ 默认 JPG：有损
    jpeg_quality: int = 95,
    png_compression: int = 3,
    recurse: bool = False,
    max_workers: Optional[int] = None,
) -> Dict[str, int]:
    """
    批量切图；返回 {'images':N, 'tiles':M, 'failed':K}
    """
    if output_folder is None:
        output_folder = input_folder.parent / f"{input_folder.name}_sliced"
    output_folder.mkdir(parents=True, exist_ok=True)

    it = input_folder.rglob("*") if recurse else input_folder.iterdir()
    images = [p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not images:
        print(f"[slice] no images in {input_folder}")
        return {"images": 0, "tiles": 0, "failed": 0}

    if max_workers is None:
        cpu = os.cpu_count() or 8
        max_workers = max(2, min(16, cpu * 4))  # 留出余量，封顶 16

    def _one(p: Path) -> Tuple[Path, int]:
        try:
            return p, slice_image_cv2(
                p, output_folder,
                tile_h=tile_h, tile_w=tile_w, overlap=overlap,
                out_ext=out_ext, jpeg_quality=jpeg_quality, png_compression=png_compression
            )
        except Exception:
            return p, 0

    tiles = failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_one, p) for p in images]
        for fut in as_completed(futures):
            _, saved = fut.result()
            tiles += saved
            if saved == 0:
                failed += 1

    print(f"[slice] images={len(images)} tiles={tiles} failed={failed} -> {output_folder}")
    return {"images": len(images), "tiles": tiles, "failed": failed}


# ============================================================
# 3) 快速删除坏图（并行）
# ============================================================
def delete_corrupt_images_fast(
    root_dir: Path | str,
    recurse: bool = False,
    exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),  # 主要 JPG/PNG
    min_bytes: int = 32,                                # 小于这个大小直接判坏图
    max_workers: Optional[int] = None,                  # 默认=CPU*4
    dry_run: bool = False,                              # 仅统计不删除
) -> Dict[str, int]:
    root = Path(root_dir)
    if recurse:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]

    if max_workers is None:
        cpu = os.cpu_count() or 8
        max_workers = max(2, min(16, cpu * 4))  # 留出余量，封顶 16

    def is_bad(p: Path) -> Tuple[Path, bool, str]:
        # 1) 空/超小文件：直接坏
        try:
            if p.stat().st_size < min_bytes:
                return p, True, "too_small"
        except Exception:
            return p, True, "stat_error"

        # 2) OpenCV 高速解码校验
        try:
            data = np.fromfile(str(p), dtype=np.uint8)
            if data.size == 0:
                return p, True, "empty"
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if img is None:
                return p, True, "imdecode_none"
            h, w = img.shape[:2]
            if h == 0 or w == 0:
                return p, True, "zero_dim"
            return p, False, ""
        except Exception as e:
            return p, True, f"decode_error:{type(e).__name__}"

    scanned = 0
    deleted = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fut in as_completed([ex.submit(is_bad, p) for p in files]):
            p, bad, _reason = fut.result()
            scanned += 1
            if bad:
                if not dry_run:
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass
                deleted += 1

    kept = scanned - deleted
    print(f"[clean-fast] scanned={scanned} kept={kept} deleted={deleted}")
    return {"scanned": scanned, "kept": kept, "deleted": deleted}


# ============================================================
# 4) 一键：按目录执行【清理坏图 → 切片】
# ============================================================
def run_clean_and_slice_images_on_dirs(
    dirs: List[Path],
    *,
    # —— 清理坏图参数 ——
    clean_recurse: bool = False,
    clean_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    clean_min_bytes: int = 32,
    clean_max_workers: Optional[int] = None,
    clean_dry_run: bool = False,

    # —— 切片参数——
    tile_h: int = 640,
    tile_w: int = 640,
    overlap: float = 0.2,
    out_ext: str = ".jpg",          
    jpeg_quality: int = 95,        
    png_compression: int = 3,
    slice_recurse: bool = False,
    slice_max_workers: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    对每个目录依次执行：
      1) delete_corrupt_images_fast 清理坏图
      2) slice_folder_cv2 进行切片（输出到同级 *_sliced，PNG 无损）

    返回汇总字典：
      {
        "<dir>": {
          "clean": {"scanned":..., "kept":..., "deleted":...},
          "slice": {"images":..., "tiles":..., "failed":...},
          "out_dir": "<dir>_sliced"
        },
        ...
      }
    """
    summary: Dict[str, Dict[str, Any]] = {}

    total = len(dirs)
    for idx, folder in enumerate(dirs, 1):
        print(f"\n[{idx}/{total}] Processing: {folder}")

        # 1) 清理坏图
        clean_stats = delete_corrupt_images_fast(
            root_dir=folder,
            recurse=clean_recurse,
            exts=clean_exts,
            min_bytes=clean_min_bytes,
            max_workers=clean_max_workers,
            dry_run=clean_dry_run,
        )

        print(f"--- 切片 {folder} ---")
        # 2) 切片（PNG 无损）
        slice_stats = slice_folder_cv2(
            input_folder=folder,
            output_folder=None,       # None: 自动 <folder>_sliced
            tile_h=tile_h,
            tile_w=tile_w,
            overlap=overlap,
            out_ext=out_ext,          # ✅ PNG
            jpeg_quality=jpeg_quality,
            png_compression=png_compression,
            recurse=slice_recurse,
            max_workers=slice_max_workers,
        )

        out_dir = folder.parent / f"{folder.name}_sliced"
        summary[str(folder)] = {
            "clean": clean_stats,
            "slice": slice_stats,
            "out_dir": str(out_dir),
        }

    print("\n✅ 清理与切片完成")
    return summary


# %%
if "run_clean_and_slice_images_on_dirs" in steps_to_run:
    summary = run_clean_and_slice_images_on_dirs(
        chosen_dirs,
        clean_recurse=False,      # True=子目录也清理，False=仅当前目录
        clean_exts=(".jpg", ".jpeg", ".png"),
        clean_min_bytes=32,
        clean_max_workers=None,
        clean_dry_run=False,      # True=仅统计不删除，False=实际删除
        tile_h=640, tile_w=640, overlap=0.2,
        out_ext=".jpg",
        jpeg_quality=95,          # 对 PNG 无效
        png_compression=3,        # 3~5 较平衡
        slice_recurse=False,      # True=子目录也切片，False=仅当前目录
        slice_max_workers=None,
    )
    print(summary)
else:
    print("跳过 Step 1: 清理坏图并切片")

# %% [markdown]
# # Step_2 0202 运行YOLO分割模型，给被分割的子图数据标记掩码
# 使用YOLO模型处理切片图像并生成LabelMe格式的标注文件
# 
# ### 输入
# 
# ![image-3.png](attachment:image-3.png)
# ### 输出
# 
# ![image-4.png](attachment:image-4.png)
# 
# ### 效果
# 
# ![image-2.png](attachment:image-2.png)
# 

# %%
# Step 2 使用YOLO分割模型处理切片图像
from ultralytics import YOLO
from pathlib import Path
import os, gc
import orjson as jsonlib
import torch

def process_sliced_images_with_yolo_seg(chosen_dirs, model_path, COMMON_KWARGS):
    """
    使用YOLO模型处理切片图像并生成LabelMe格式的标注文件
    """
    model = YOLO(model_path)

    _dumps = lambda obj: jsonlib.dumps(obj, option=jsonlib.OPT_INDENT_2 | jsonlib.OPT_NON_STR_KEYS)
    _loads = jsonlib.loads

    for directory in chosen_dirs:
        print(f"\n=== 处理目录: {directory} ===")
        src_dir = Path(str(directory) + "_sliced")
        
        if not src_dir.exists() or not any(src_dir.iterdir()):
            print(f"跳过空目录: {src_dir}")
            continue

        # 获取所有图片文件
        image_files = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
        if not image_files:
            print(f"无图片: {src_dir}")
            continue

        # 分片处理避免内存溢出
        CHUNK_SIZE = 100
        for chunk_index in range(0, len(image_files), CHUNK_SIZE):
            image_chunk = image_files[chunk_index:chunk_index + CHUNK_SIZE]
            print(f" -> 处理分片 {chunk_index}-{chunk_index + len(image_chunk) - 1} / {len(image_files)}")

            # 批量预测
            results_generator = model.predict(image_chunk, **COMMON_KWARGS)

            # 逐图像处理结果
            for result_index, result in enumerate(results_generator, 1):
                try:
                    detections_list = _loads(result.to_json())

                    height, width = map(int, result.orig_shape[:2])
                    image_name = os.path.basename(getattr(result, "path", "")) or f"image_{result_index}.png"

                    shapes = []
                    for detection in detections_list:
                        segmentation = detection.get("segments", {})
                        xs, ys = segmentation.get("x", []), segmentation.get("y", [])
                        if not xs or not ys:
                            continue
                        points = [[float(x), float(y)] for x, y in zip(xs, ys)]
                        shapes.append({
                            "label": detection.get("name", ""),
                            "score": float(detection.get("confidence", 0.0)),
                            "points": points,
                            "shape_type": "polygon",
                        })

                    labelme_annotation = {
                        "shapes": shapes,
                        "imagePath": image_name,
                        "imageHeight": height,
                        "imageWidth": width,
                    }

                    output_path = src_dir / f"{Path(image_name).stem}.json"
                    output_path.write_bytes(_dumps(labelme_annotation))

                finally:
                    # 及时释放内存
                    del result
                    if result_index % 64 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

            # 分片结束后清理内存
            torch.cuda.empty_cache()
            gc.collect()

        print(f"✅ 完成。保存至: {src_dir}")

# %%
if "process_sliced_images_with_yolo_seg" in steps_to_run:
    seg_model_path = f"/workspace/models/best_model/yolo11n-seg-best.pt"
    COMMON_KWARGS = dict(
        imgsz=640,
        conf=0.25,
        iou=0.45,
        device=0,
        batch=3,
        retina_masks=False,
        workers=2,
        verbose=False,
        save=False,
    )
    process_sliced_images_with_yolo_seg(chosen_dirs, seg_model_path, COMMON_KWARGS  = COMMON_KWARGS)
else:
    print("跳过 Step 1: 使用YOLO分割模型标记掩码")

# %% [markdown]
# # Step_3 0203  合并子图到大图。也将segmentation信息整合
# 
# ### 输入输出
# ![image.png](attachment:image.png)
# 
# ### 展开输出文件夹
# 
# ![image-4.png](attachment:image-4.png)
# 
# 
# ### 效果
# 
# ![image-5.png](attachment:image-5.png)

# %%
# Step 3 合并和去重切片预测结果
import os
import math
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional, Iterable, List, Dict, Any, Tuple

import orjson
from tqdm import tqdm
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============== JSON 工具函数 ==============
def json_load(path: str):
    with open(path, "rb") as f:
        return orjson.loads(f.read())

def json_dump(obj, path: str):
    with open(path, "wb") as f:
        f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))

# ============== shapely（可选） ==============
try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    _HAVE_SHAPELY = True
except Exception:
    _HAVE_SHAPELY = False

# ============== 工具函数 ==============
def _build_image_index(original_image_dir: str) -> Dict[str, str]:
    """建立原图文件名到路径的索引"""
    idx: Dict[str, str] = {}
    for p in Path(original_image_dir).glob("*.jpg"):
        idx[p.stem] = str(p)
    return idx

def _draw_annotations_on_image(args: Tuple[str, List[Dict[str, Any]], str, str, bool, int]) -> bool:
    """在单张图像上绘制多边形标注"""
    image_name, annotations, image_path, out_dir, draw_text, jpeg_quality = args
    img = cv2.imread(image_path)
    if img is None:
        return False

    for ann in annotations:
        pts = np.asarray(ann["points"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=True, thickness=1, color=(0, 255, 255))

        if draw_text:
            label = ann.get("label", "")
            score = float(ann.get("score", 0.0))
            x0, y0 = int(ann["points"][0][0]) + 12, int(ann["points"][0][1]) + 12
            txt = f"{label} {score:.3f}"
            cv2.putText(img, txt, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    out_path = os.path.join(out_dir, f"{image_name}_vis.jpg")
    cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    return True

def parse_slice_filename(filename: str):
    """解析切片文件名获取原图名和偏移坐标"""
    parts = Path(filename).stem.split("_")
    name = "_".join(parts[:-4])
    x1, y1, x2, y2 = map(int, parts[-4:])
    return name, x1, y1

# ============== 主要处理函数 ==============
def merge_slice_annotations(sliced_label_dir: str, output_json_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """合并所有切片标注到原图坐标系"""
    merged_annotations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    json_files = list(Path(sliced_label_dir).rglob("*.json"))

    for json_file in tqdm(json_files, desc="合并切片标注", unit="file"):
        data = json_load(str(json_file))
        image_path = data["imagePath"]
        original_name, offset_x, offset_y = parse_slice_filename(image_path)
        for shape in data.get("shapes", []):
            points = shape["points"]
            label = shape.get("label", "")
            new_points = [[x + offset_x, y + offset_y] for x, y in points]
            merged_annotations[original_name].append({
                "uuid": str(uuid.uuid4()),
                "original_name": original_name,
                "label": label,
                "points": new_points,
                "offset_x": offset_x,
                "offset_y": offset_y,
                "score": float(shape.get("score", 0.0)),
            })

    json_dump(merged_annotations, output_json_path)
    print(f"✅ 合并完成，共处理 {len(merged_annotations)} 张原图")
    print(f"✔️ 合并标注已保存到 {output_json_path}")
    return merged_annotations

def deduplicate_annotations(
    merged_annotations: Dict[str, List[Dict[str, Any]]],
    output_json_path: str,
    method: str = "GREEDYNMM",
    metric: str = "IOS",
    threshold: float = 0.5,
    class_agnostic: bool = False,
    center_threshold: Optional[float] = 20.0,
    keep_strategy: str = "REP"
) -> Dict[str, List[Dict[str, Any]]]:
    """去除重复标注（多种去重算法）"""
    
    # 内部工具函数定义
    def polygon_to_bbox(points: Iterable[Iterable[float]]) -> List[float]:
        xs, ys = zip(*points)
        return [min(xs), min(ys), max(xs), max(ys)]

    def bbox_area(bbox):
        w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
        return (w if w > 0 else 0) * (h if h > 0 else 0)

    def bbox_iou(bbox_a, bbox_b):
        xA = max(bbox_a[0], bbox_b[0]); yA = max(bbox_a[1], bbox_b[1])
        xB = min(bbox_a[2], bbox_b[2]); yB = min(bbox_a[3], bbox_b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter <= 0: return 0.0
        u = bbox_area(bbox_a) + bbox_area(bbox_b) - inter
        return inter / u if u > 0 else 0.0

    def bbox_ios(bbox_a, bbox_b):
        xA = max(bbox_a[0], bbox_b[0]); yA = max(bbox_a[1], bbox_b[1])
        xB = min(bbox_a[2], bbox_b[2]); yB = min(bbox_a[3], bbox_b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter <= 0: return 0.0
        smaller = min(bbox_area(bbox_a), bbox_area(bbox_b))
        return inter / smaller if smaller > 0 else 0.0

    def center_distance(bbox_a, bbox_b):
        cxA = (bbox_a[0] + bbox_a[2]) * 0.5; cyA = (bbox_a[1] + bbox_a[3]) * 0.5
        cxB = (bbox_b[0] + bbox_b[2]) * 0.5; cyB = (bbox_b[1] + bbox_b[3]) * 0.5
        return math.hypot(cxA - cxB, cyA - cyB)

    match_metric = bbox_iou if metric.upper() == "IOU" else bbox_ios

    def is_same_group(ann1: Dict, ann2: Dict) -> bool:
        if (not class_agnostic) and (ann1["label"] != ann2["label"]):
            return False
        bbox1 = polygon_to_bbox(ann1["points"])
        bbox2 = polygon_to_bbox(ann2["points"])
        if (center_threshold is not None) and center_distance(bbox1, bbox2) > center_threshold:
            return False
        return match_metric(bbox1, bbox2) >= threshold

    def get_annotation_score(ann: Dict) -> float:
        try:
            return float(ann.get("score", 0.0))
        except Exception:
            return 0.0

    def select_representative_polygon(group: List[Dict]) -> Dict:
        best_ann = None
        best_key = (-1e9, -1e9)
        for i, ann in enumerate(group):
            bbox = polygon_to_bbox(ann["points"])
            overlap_sum = 0.0
            for j, other_ann in enumerate(group):
                if i == j: continue
                overlap_sum += match_metric(bbox, polygon_to_bbox(other_ann["points"]))
            key = (get_annotation_score(ann), overlap_sum)
            if key > best_key:
                best_key = key; best_ann = ann
        representative = dict(best_ann)
        representative["uuid"] = str(uuid.uuid4())
        return representative

    def merge_polygon_group(group: List[Dict]) -> Dict:
        if not _HAVE_SHAPELY:
            return select_representative_polygon(group)
        polygons = []
        for ann in group:
            points = ann["points"]
            if len(points) >= 3:
                try:
                    polygons.append(Polygon(points))
                except Exception:
                    pass
        if not polygons:
            return select_representative_polygon(group)
        merged_polygon = unary_union(polygons)
        if merged_polygon.geom_type == "MultiPolygon":
            merged_polygon = max(list(merged_polygon.geoms), key=lambda p: p.area)
        coordinates = list(merged_polygon.exterior.coords)[:-1]
        base_ann = dict(group[0])
        base_ann["uuid"] = str(uuid.uuid4())
        base_ann["points"] = [[float(x), float(y)] for (x, y) in coordinates] or group[0]["points"]
        base_ann["score"] = max(get_annotation_score(ann) for ann in group)
        return base_ann

    def process_annotation_group(group: List[Dict]) -> Dict:
        return merge_polygon_group(group) if keep_strategy.upper() == "UNION_POLY" else select_representative_polygon(group)

    # 去重算法实现
    def nms_algorithm(annotations: List[Dict]) -> List[Dict]:
        sorted_annotations = sorted(annotations, key=lambda a: get_annotation_score(a), reverse=True)
        kept_annotations: List[Dict] = []
        for ann in sorted_annotations:
            should_suppress = False
            ann_bbox = None
            for kept_ann in kept_annotations:
                if (not class_agnostic) and ann["label"] != kept_ann["label"]:
                    continue
                if ann_bbox is None:
                    ann_bbox = polygon_to_bbox(ann["points"])
                kept_bbox = polygon_to_bbox(kept_ann["points"])
                if (center_threshold is None or center_distance(ann_bbox, kept_bbox) <= center_threshold) and \
                   match_metric(ann_bbox, kept_bbox) >= threshold:
                    should_suppress = True; break
            if not should_suppress:
                kept_annotations.append(ann)
        result = []
        for ann in kept_annotations:
            new_ann = dict(ann); new_ann["uuid"] = str(uuid.uuid4())
            result.append(new_ann)
        return result

    def greedy_grouping_algorithm(annotations: List[Dict]) -> List[List[Dict]]:
        used = [False] * len(annotations)
        indices_sorted = sorted(range(len(annotations)), key=lambda i: get_annotation_score(annotations[i]), reverse=True)
        groups: List[List[Dict]] = []
        for idx in indices_sorted:
            if used[idx]: continue
            seed_ann = annotations[idx]
            group = [seed_ann]; used[idx] = True
            changed = True
            while changed:
                changed = False
                for j, other_ann in enumerate(annotations):
                    if used[j]: continue
                    if any(is_same_group(other_ann, group_ann) for group_ann in group):
                        group.append(other_ann); used[j] = True; changed = True
            groups.append(group)
        return groups

    # 主处理逻辑
    cleaned_annotations: Dict[str, List[Dict]] = {}
    total_before = sum(len(v) for v in merged_annotations.values())
    total_after = 0

    for image_name, annotations in tqdm(merged_annotations.items(), desc="去重处理", unit="image"):
        if not class_agnostic:
            label_buckets = defaultdict(list)
            for ann in annotations:
                label_buckets[ann["label"]].append(ann)
            result_annotations: List[Dict] = []
            for _, bucket in label_buckets.items():
                algorithm = method.upper()
                if algorithm == "NMS":
                    result_annotations.extend(nms_algorithm(bucket))
                elif algorithm == "NMM":
                    result_annotations.extend(process_annotation_group(g) for g in pairwise_grouping(bucket))
                elif algorithm == "LSNMS":
                    result_annotations.extend(lsnms_algorithm(bucket))
                else:  # GREEDYNMM
                    result_annotations.extend(process_annotation_group(g) for g in greedy_grouping_algorithm(bucket))
        else:
            algorithm = method.upper()
            if algorithm == "NMS":
                result_annotations = nms_algorithm(annotations)
            elif algorithm == "NMM":
                result_annotations = [process_annotation_group(g) for g in pairwise_grouping(annotations)]
            elif algorithm == "LSNMS":
                result_annotations = lsnms_algorithm(annotations)
            else:
                result_annotations = [process_annotation_group(g) for g in greedy_grouping_algorithm(annotations)]

        cleaned_annotations[image_name] = result_annotations
        total_after += len(result_annotations)

    json_dump(cleaned_annotations, output_json_path)

    print(f"🔁 去重完成（{method}, metric={metric}, threshold={threshold}, class_agnostic={class_agnostic}, strategy={keep_strategy}）")
    print(f"    目标数：{total_before} → {total_after}")
    if keep_strategy.upper() == "UNION_POLY" and not _HAVE_SHAPELY:
        print("⚠️ 未安装shapely，已退回REP模式")
    print(f"✔️ 已保存到 {output_json_path}")
    return cleaned_annotations

def visualize_annotations(
    merged_annotations: Dict[str, List[Dict[str, Any]]],
    original_image_dir: str,
    output_visual_dir: str,
    draw_text: bool = True,
    jpeg_quality: int = 95,
    parallel: bool = True,
    max_workers: Optional[int] = None
):
    """可视化标注结果"""
    os.makedirs(output_visual_dir, exist_ok=True)
    image_index = _build_image_index(original_image_dir)

    tasks = []
    for image_name, annotations in merged_annotations.items():
        image_path = image_index.get(image_name)
        if image_path is None or not os.path.exists(image_path):
            found = None
            for p in Path(original_image_dir).glob(f"{image_name}*.jpg"):
                found = str(p); break
            image_path = found
        if image_path is None or not os.path.exists(image_path):
            continue
        tasks.append((image_name, annotations, image_path, output_visual_dir, draw_text, jpeg_quality))

    if not tasks:
        print("⚠️ 没有可视化任务")
        return

    if not parallel:
        for task in tqdm(tasks, desc="可视化处理（串行）", unit="image"):
            _draw_annotations_on_image(task)
    else:
        if max_workers is None:
            max_workers = max(2, (os.cpu_count() or 8) // 2)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_draw_annotations_on_image, task) for task in tasks]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="可视化处理（并行）", unit="image"):
                pass

    print(f"🖼 可视化图片已保存到 {output_visual_dir}/")

def _crop_single_object(args: Tuple[str, List[Dict[str, Any]], str, str, int, int]) -> int:
    """裁剪单个目标对象"""
    image_name, annotations, image_path, out_dir, margin, jpeg_quality = args
    img = cv2.imread(image_path)
    if img is None:
        return 0
    height, width = img.shape[:2]
    saved_count = 0

    for idx, ann in enumerate(annotations):
        points = np.asarray(ann["points"], dtype=np.float32)
        xs = points[:, 0]; ys = points[:, 1]
        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())

        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        side_length = max(bbox_width, bbox_height)

        center_x = (min_x + max_x) * 0.5
        center_y = (min_y + max_y) * 0.5

        left = int(round(center_x - side_length * 0.5)) - margin
        top = int(round(center_y - side_length * 0.5)) - margin
        right = int(round(center_x + side_length * 0.5)) + margin
        bottom = int(round(center_y + side_length * 0.5)) + margin

        left = max(0, left)
        top = max(0, top)
        right = min(width, right)
        bottom = min(height, bottom)

        if right - left <= 1 or bottom - top <= 1:
            continue

        crop = img[top:bottom, left:right]
        save_name = f"{image_name}_obj{idx}_{ann.get('label','')}_uuid_{ann['uuid']}.jpg"
        out_path = os.path.join(out_dir, save_name)
        cv2.imwrite(out_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        saved_count += 1

    return saved_count

def export_cropped_objects(
    merged_annotations: Dict[str, List[Dict[str, Any]]],
    original_image_dir: str,
    cropped_object_dir: str,
    margin: int = 0,
    jpeg_quality: int = 95,
    parallel: bool = True,
    max_workers: Optional[int] = None
):
    """导出裁剪的目标对象"""
    os.makedirs(cropped_object_dir, exist_ok=True)
    image_index = _build_image_index(original_image_dir)

    tasks = []
    for image_name, annotations in merged_annotations.items():
        image_path = image_index.get(image_name)
        if image_path is None or not os.path.exists(image_path):
            found = None
            for p in Path(original_image_dir).glob(f"{image_name}*.jpg"):
                found = str(p); break
            image_path = found
        if image_path is None or not os.path.exists(image_path):
            continue
        tasks.append((image_name, annotations, image_path, cropped_object_dir, margin, jpeg_quality))

    total_saved = 0
    if not tasks:
        print("⚠️ 没有可导出的裁剪任务")
        return

    if not parallel:
        for task in tqdm(tasks, desc="导出裁剪（串行）", unit="image"):
            total_saved += _crop_single_object(task)
    else:
        if max_workers is None:
            max_workers = max(2, (os.cpu_count() or 8) // 2)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_crop_single_object, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="导出裁剪（并行）", unit="image"):
                total_saved += future.result()

    print(f"📦 个体裁剪图像已保存到 {cropped_object_dir}/ （共导出 {total_saved} 张）")



# %%
def combine_sliced_predictions(chosen_dirs):
    """处理切片预测结果的主函数"""
    for directory in chosen_dirs:
        print(f"\n=== 处理目录: {directory} ===")
        original_image_dir = str(directory)
        sliced_label_dir = str(directory) + "_sliced"
        output_json_path = str(directory) + "_sliced_merge/01_merged_annotations.json"
        output_visual_dir = str(directory) + "_sliced_merge/01_visualizations"
        cropped_object_dir = str(directory) + "_sliced_merge/01_cropped_objects"

        print(f"原图目录: {original_image_dir}")
        print(f"切片标注目录: {sliced_label_dir}")
        print(f"输出合并标注: {output_json_path}")
        print(f"输出可视化目录: {output_visual_dir}")
        print(f"输出裁剪目录: {cropped_object_dir}")

        os.makedirs(output_visual_dir, exist_ok=True)
        os.makedirs(cropped_object_dir, exist_ok=True)

        # 1) 合并切片标注
        merged_annotations = merge_slice_annotations(sliced_label_dir, output_json_path)

        # 2) 去重处理
        merged_annotations = deduplicate_annotations(
            merged_annotations,
            output_json_path,
            method="NMS",               # 'NMM'/'GREEDYNMM'/'LSNMS'/'NMS' -- NMS每个目标最多 1 个重复（完整 + 小碎片），直接保留高分的那一个就行。
            metric="IOS",               # 'IOU'/'IOS'  当可能出现"小框被大框包含"时，推荐使用 IOS，当两个框大小相近，且你想知道"整体重叠程度"时，推荐使用 IOU
            threshold=0.5,              # 两个边界框被认为是重复的阈值
            class_agnostic=False,
            center_threshold=20,
            keep_strategy="REP"         # 'REP'/'UNION_POLY' -- 不需要 UNION_POLY（并集）去“粘合碎片”，因为我们只保留完整的那份就好。
        )

        # 3) 可视化
        visualize_annotations(
            merged_annotations,
            original_image_dir,
            output_visual_dir,
            draw_text=True,         # 是否绘制标签和置信度，关掉可以提升速度
            jpeg_quality=95,
            parallel=True,
            max_workers=None
        )

        # 4) 导出裁剪
        export_cropped_objects(
            merged_annotations,
            original_image_dir,
            cropped_object_dir,
            margin=15,
            jpeg_quality=95,
            parallel=True,
            max_workers=None
        )

# %%
if "combine_sliced_predictions" in steps_to_run:
    combine_sliced_predictions(chosen_dirs)
else:
    print("跳过 Step 2: 合并和去重切片预测结果")


# %% [markdown]
# # 精炼检测v3--用seg+det模型对所有疑似SWD的object进一步筛选

# %%
version = "v1"
run_type = "pose_and_det"  # "pose_and_det" or "cls"

# 需要运行的step列表
steps_to_run = [
    # "run_clean_and_slice_images_on_dirs",           # Step 1 清理坏图并切片大图到640*640小图
    # "process_sliced_images_with_yolo_seg",          # Step 2 使用YOLO-seg模型处理640*640切片图像
    # "combine_sliced_predictions",                   # Step 3 合并seg预测结果，回到原图。 同时切出objects小图
    "run_pose_on_chosen_dirs",                        # Step 4 运行 Pose Estimation（在 cropped_objects 上）
    "run_batch_dot_det",                              # Step 5 运行 det 模型 检测“小黑点”（在 cropped_objects 上）
    "process_swd_matching",                          # Step 6 判定 SWD ⇒ 匹配规则：两翼关键点分别落入两个不同的小黑点框
]

# %%
def free_gpu():
    import gc, torch
    gc.collect()                    # 触发 Python 垃圾回收
    torch.cuda.empty_cache()        # 释放未使用的 GPU 缓存到驱动
    torch.cuda.ipc_collect()        # 清理跨进程缓存（偶尔有用）


# %% [markdown]
# # Step_0 查看根目录下需要运行的文件夹 
# 选择含有 raw_data 图片的 *_data 目录
# 
# ![image.png](attachment:image.png)
# 

# %%
# 选择数据目录的核心代码
from pathlib import Path

def select_data_dirs(root_dir: Path, end_with: str = "_data"):
    # === 1) 遍历所有子目录 ===
    sub_dirs = list(root_dir.glob("*/*" + end_with))

    if not sub_dirs:
        print(f"没有找到 *{end_with} 目录")
        return []

    print(f"找到以下 {end_with} 数据集：")
    for i, d in enumerate(sub_dirs):
        print(f"[{i}] {d}")

    # === 2) 让用户选择要跑的目录 ===
    idx_str = input("请输入要处理的编号 (多个用逗号分隔, 回车默认全选): ").strip()
    if idx_str:
        indices = [int(x) for x in idx_str.split(",")]
        chosen_dirs = [sub_dirs[i] for i in indices]
    else:
        chosen_dirs = sub_dirs

    print(f"将处理以下 {end_with} 目录：")
    for i, d in enumerate(chosen_dirs):
        print(f"- {i+1}. {d}")

    # === 3) 筛选掉没有 raw_data 图片的目录 ===
    chosen_dirs = [
        d for d in chosen_dirs
        if (d.parent / "raw_data").exists() and any((d.parent / "raw_data").glob("*.jpg"))
    ]

    if not chosen_dirs:
        print(f"没有找到包含图片的 *{end_with} 目录")
        return []

    return chosen_dirs

# %%
# root_dir = Path("/workspace/models/SAHI/run_v8")
# chosen_dirs = select_data_dirs(root_dir, end_with="_data")
# print("最终确认的目录：", chosen_dirs)
# if not chosen_dirs:
#     raise ValueError("没有选择任何目录，程序终止。")

# %% [markdown]
# # Step_4 运行 Pose Estimation（在 cropped_objects 上）
# 读取每张小图（带 uuid_... 命名），输出头(h)、左翼(lp)、右翼(rp) 三关键点
# 
# ### 输入输出
# ![image.png](attachment:image.png)
# 
# ### 效果
# ![image-2.png](attachment:image-2.png)

# %%
# Step 4: 运行 Pose Estimation
import re, os, json
from pathlib import Path
from typing import List, Dict, Any

from shapely import box
from ultralytics import YOLO

# 文件名解析：..._uuid_<uuid>.jpg
UUID_RE = re.compile(r"uuid_([a-f0-9\-]+)\.(jpg|jpeg|png)$", re.IGNORECASE)
ORIG_RE = re.compile(r"^(\d+_\d+_\d+)_obj", re.IGNORECASE)

def run_pose_on_dir(
    model_path: str,
    input_dir: Path,
    out_json: Path,
    kpt_names: List[str],
    predict_args: Dict[str, Any]
):
    """
    对单个目录运行姿态估计
    
    Args:
        model_path: 模型路径
        input_dir: 输入图片目录
        out_json: 输出JSON文件路径
        kpt_names: 关键点名称列表
        predict_args: model.predict的参数（必须传入；如果 imgsz=None 则读取模型默认值）
    """
    if not any(input_dir.glob("*.jpg")) and not any(input_dir.glob("*.png")):
        print(f"⚠️ 输入目录无图片：{input_dir}")
        return
    
    print(f"加载姿态模型：{model_path}")
    model = YOLO(model_path)

    # 处理 imgsz=None -> 使用模型默认
    args = dict(predict_args)  # 复制一份
    if "imgsz" in args and args["imgsz"] is None:
        args["imgsz"] = model.overrides.get("imgsz")
        print(f"ℹ️ 使用模型默认 imgsz = {args['imgsz']}")

    results = model.predict(source=str(input_dir), **args)

    out: List[Dict[str, Any]] = []
    for res in results:
        fpath = getattr(res, "path", "")
        fname = os.path.basename(fpath)
        m_uuid = UUID_RE.search(fname)
        uuid_str = m_uuid.group(1) if m_uuid else None
        m_orig = ORIG_RE.match(fname)
        original_name = m_orig.group(1) if m_orig else None

        dets = []
        kpts = getattr(res, "keypoints", None)
        if kpts is not None and kpts.data is not None:
            arr = kpts.data.cpu().numpy()
            conf_arr = getattr(kpts, "conf", None)
            conf_arr = conf_arr.cpu().numpy() if conf_arr is not None else None

            for i in range(arr.shape[0]):
                pts = arr[i]
                item = []
                for ki in range(min(len(kpt_names), pts.shape[0])):
                    x, y = float(pts[ki][0]), float(pts[ki][1])
                    c = float(conf_arr[i][ki]) if (
                        conf_arr is not None and conf_arr.shape == (arr.shape[0], pts.shape[0])
                    ) else None
                    item.append({"name": kpt_names[ki], "x": x, "y": y, "conf": c})
                dets.append({"kpts": item})
        boxes = getattr(res, "boxes", None)
        if boxes is not None and boxes.data is not None:
            arr = boxes.data.cpu().numpy()
            for i in range(arr.shape[0]):
                x1, y1, x2, y2, conf, cls = arr[i]
                item = {
                    "box": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                    "conf": float(conf),
                    "cls": int(cls)
                }
                if i < len(dets):
                    dets[i]["box"] = item["box"]
                    dets[i]["box_conf"] = item["conf"]
                    dets[i]["box_cls"] = item["cls"]
                else:
                    dets.append(item)

        out.append({
            "path": fpath,
            "file": fname,
            "uuid": uuid_str,
            "original_name": original_name,
            "instances": dets
        })

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"✅ Pose 结果保存：{out_json}")

def run_pose_on_chosen_dirs(
    chosen_dirs: List[Path],
    model_path: str,
    kpt_names: List[str],
    predict_args: Dict[str, Any]
):
    """
    对选定的目录列表批量运行姿态估计
    
    Args:
        chosen_dirs: 包含图片的目录列表
        model_path: 模型路径
        kpt_names: 关键点名称列表
        predict_args: model.predict的参数（必须传入；imgsz=None 表示自适应模型默认）
    """
    for d in chosen_dirs:
        crops_dir = d.parent / (d.name + "_sliced_merge") / "01_cropped_objects"
        pose_json = d.parent / (d.name + "_sliced_merge") / f"pose_and_det_{version}" / "02_pose_predicted_results.json"
        print(f"\n=== Pose on: {crops_dir} ===")
        run_pose_on_dir(model_path, crops_dir, pose_json, kpt_names, predict_args)


# %%
if "run_pose_on_chosen_dirs" in steps_to_run:
    custom_kpt_names = ["h", "lp", "rp"]
    custom_predict_args = {
        "imgsz": None,  # None 表示使用模型默认值
        "conf": 0.88,
        "iou": 0.6,
        "device": 0,
        "verbose": True,
        # "stream": True,
        "batch": 128,
    }

    run_pose_on_chosen_dirs(
        chosen_dirs, 
        model_path="/workspace/models/best_model/yolo11n-pose-best_v2.pt",
        kpt_names=custom_kpt_names,
        predict_args=custom_predict_args
    )
    free_gpu()
else:
    print("Step_4 运行 Pose Estimation（在 cropped_objects 上） 被跳过")

# %% [markdown]
# # Step_5 运行 det 模型 检测“小黑点”（在 cropped_objects 上）
# 读取每张小图（带 uuid_... 命名），输出头(h)、左翼(lp)、右翼(rp) 三关键点
# 
# ### 输入输出
# ![image-2.png](attachment:image-2.png)
# 
# ### 效果
# ![image-3.png](attachment:image-3.png)

# %%
# Step 5: 运行 det 模型 检测“小黑点”
import os, re, json
from pathlib import Path
from typing import Dict, Any, List
from ultralytics import YOLO
import numpy as np

UUID_RE = re.compile(r"uuid_([a-f0-9\-]+)\.(jpg|jpeg|png)$", re.IGNORECASE)

def _to_float_list(x):
    # 把 numpy/tensor 标量安全转成 python float
    return [float(v) for v in x]

def run_dot_det_on_dir(
    model_path: str,
    input_dir: Path,
    out_json: Path,
    custom_predict_args: Dict[str, Any]
):
    if not any(input_dir.glob("*.jpg")) and not any(input_dir.glob("*.png")):
        print(f"⚠️ 输入目录无图片：{input_dir}")
        return

    print(f"加载小黑点检测模型：{model_path}")
    model = YOLO(model_path)

    if custom_predict_args['imgsz'] is None:
        custom_predict_args['imgsz'] = model.overrides.get("imgsz")
        print(f"ℹ️ 使用模型默认 imgsz = {custom_predict_args['imgsz']}")

    # 必须传 custom_predict_args
    results = model.predict(**custom_predict_args, source=str(input_dir))

    out: List[Dict[str, Any]] = []
    for res in results:
        fpath = getattr(res, "path", "")
        fname = os.path.basename(fpath)
        m_uuid = UUID_RE.search(fname)
        uuid_str = m_uuid.group(1) if m_uuid else None

        det_list = []
        boxes = getattr(res, "boxes", None)
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else None
            confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") and boxes.conf is not None else None
            clses = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") and boxes.cls is not None else None

            if xyxy is not None:
                n = xyxy.shape[0]
                for i in range(n):
                    x1, y1, x2, y2 = _to_float_list(xyxy[i].tolist())
                    conf_score = float(confs[i]) if confs is not None and i < len(confs) else None
                    cls_id = int(clses[i]) if clses is not None and i < len(clses) else 0
                    det_list.append({
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf_score,
                        "cls": cls_id
                    })

        out.append({
            "path": fpath,
            "file": fname,
            "uuid": uuid_str,
            "boxes": det_list
        })

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"✅ 小黑点检测结果保存：{out_json}")

def run_batch_dot_det(
    chosen_dirs: List[Path],
    dot_model: str,
    custom_predict_args: Dict[str, Any]
):
    """批处理入口：遍历 chosen_dirs，运行小黑点检测；predict 参数必须通过 custom_predict_args 提供"""
    for d in chosen_dirs:
        crops_dir = d.parent / (d.name + "_sliced_merge") / "01_cropped_objects"
        dot_json  = d.parent / (d.name + "_sliced_merge") / f"pose_and_det_{version}" / "03_dot_predicted_results.json"
        print(f"\n=== Dot-Det on: {crops_dir} ===")
        run_dot_det_on_dir(dot_model, crops_dir, dot_json, custom_predict_args=custom_predict_args)


# %%
if "run_batch_dot_det" in steps_to_run:
    dot_det_model_path = "/workspace/models/best_model/yolo11n-det-best_v1.pt"
    custom_predict_args = {
        "imgsz": None,  # None 则使用模型默认值
        "conf": 0.3,
        "iou": 0.5,
        "device": 0,
        "verbose": True,
        # "stream": True,
        "batch": 128
    }
    run_batch_dot_det(chosen_dirs, dot_model=dot_det_model_path, custom_predict_args=custom_predict_args)

    free_gpu()
else:
    print("Step_5 运行 det 模型 检测“小黑点”（在 cropped_objects 上） 被跳过")


# %% [markdown]
# # Step_6 判定 SWD ⇒ 匹配规则：两翼关键点分别落入两个不同的小黑点框
# 
# ### 输入
# ![image-3.png](attachment:image-3.png)
# ### 输出
# ![image-4.png](attachment:image-4.png)
# 
# ### 效果
# ![image-2.png](attachment:image-2.png)
# ![image-8.png](attachment:image-8.png)
# ![image-10.png](attachment:image-10.png)

# %%
# Step 6: 判定 SWD ⇒ 匹配规则：两翼关键点分别落入两个不同的小黑点框

import os, re, shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np

# ---------- 可选：orjson 加速 ----------
try:
    import orjson as _fastjson
    def _loads(b: bytes): return _fastjson.loads(b)
    def _dumps(obj): return _fastjson.dumps(obj, option=_fastjson.OPT_INDENT_2)
except Exception:
    import json as _fastjson
    def _loads(b: bytes): return _fastjson.loads(b.decode("utf-8"))
    def _dumps(obj): return _fastjson.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")

# ====== 参数区（按需调整）======
REQUIRE_DIFFERENT_BOXES = True   # ✅ 两翼必须落入不同框
USE_POINT_CONF = True            # 若关键点带 conf，则应用阈值过滤
KPT_CONF_THR = 0.88
DO_VIS = False                    # 输出可视化图
LIMIT_VIS = None                 # 仅可视化前 N 张（None 为全部）
COPY_UNMATCHED = False            # 也拷贝未匹配样本，便于人工复核

UUID_RE = re.compile(r"uuid_([a-f0-9\-]+)\.(jpg|jpeg|png)$", re.IGNORECASE)

# ---------- I/O ----------
def load_json(path: Path):
    with path.open("rb") as f:
        data = f.read()
    return _loads(data)

def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(_dumps(obj))

# ---------- 向量化点落框 ----------
# 输入：pt=(x,y), boxes: np.ndarray (N,4) [x1,y1,x2,y2] float32
# 输出：命中的索引数组（np.int32）
def hits_for_point(pt: Tuple[float, float], boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int32)
    x, y = pt
    # (N,)
    cond = (x >= boxes[:, 0]) & (x <= boxes[:, 2]) & (y >= boxes[:, 1]) & (y <= boxes[:, 3])
    return np.flatnonzero(cond).astype(np.int32, copy=False)

def find_kpt(items: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    # 保持 O(K) 扫描，但避开多次 dict.get
    for it in items:
        if it.get("name") == name:
            return it
    return None

def kpt_ok(k: Optional[Dict[str, Any]], use_point_conf: bool, thr: float) -> bool:
    if not k:
        return False
    if use_point_conf:
        conf = k.get("conf")
        if conf is not None:
            # 避免 try/except，直接按常见数值/字符串转 float
            try:
                return float(conf) >= thr
            except Exception:
                return False
    return True

def match_one(uuid_: str,
              pose_items: List[Dict[str, Any]],
              dot_boxes_np: np.ndarray,
              require_different_boxes: bool = True,
              use_point_conf: bool = True,
              kpt_conf_thr: float = 0.15) -> Dict[str, Any]:
    """对一张小图进行匹配判断：lp 与 rp 必须分别命中两个不同的小黑点框（向量化版）"""
    lp = find_kpt(pose_items, "lp")
    rp = find_kpt(pose_items, "rp")

    lp_ok = kpt_ok(lp, use_point_conf, kpt_conf_thr)
    rp_ok = kpt_ok(rp, use_point_conf, kpt_conf_thr)

    matched = False
    lp_in_idx = None
    rp_in_idx = None

    if lp_ok and rp_ok and dot_boxes_np.size:
        lp_hits = hits_for_point((lp["x"], lp["y"]), dot_boxes_np)
        rp_hits = hits_for_point((rp["x"], rp["y"]), dot_boxes_np)

        if lp_hits.size and rp_hits.size:
            if require_different_boxes:
                # 找一对不同索引：利用广播快速找到第一对
                # 等价原逻辑的 "第一对" —— 取 lexicographically 最小的一对
                # 生成笛卡尔积最省事但可能大；这里用集合优化：
                rp_set = set(int(i) for i in rp_hits.tolist())
                for i in lp_hits.tolist():
                    # 寻找 rp_set 中 != i 的任意元素
                    if i in rp_set:
                        # 如果 rp 还有其他不同于 i 的命中，选其一
                        # 这里继续尝试找 rp 中第一个 != i 的
                        for j in rp_hits.tolist():
                            if j != i:
                                lp_in_idx, rp_in_idx = i, j
                                matched = True
                                break
                        if matched:
                            break
                    else:
                        # 直接取 rp_hits[0]
                        lp_in_idx, rp_in_idx = i, int(rp_hits[0])
                        matched = True
                        break
                # 若上面没找到，再尝试反向
                if not matched and lp_hits.size > 1 and rp_hits.size > 1:
                    i = int(lp_hits[0]); j = int(rp_hits[1] if rp_hits[0] == i else rp_hits[0])
                    if i != j:
                        lp_in_idx, rp_in_idx = i, j
                        matched = True
            else:
                matched = True
                lp_in_idx = int(lp_hits[0])
                rp_in_idx = int(rp_hits[0])

    return {
        "uuid": uuid_,
        "matched": matched,
        "lp": lp,
        "rp": rp,
        "lp_box_idx": lp_in_idx,
        "rp_box_idx": rp_in_idx,
    }

def visualize_match(img_path: Path,
                    pose_items: List[Dict[str, Any]],
                    boxes_np: np.ndarray,
                    match_info: Dict[str, Any],
                    out_path: Path):
    """可视化：小黑点框（细线），关键点小圆点"""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return
    vis = img.copy()

    lp_idx = match_info.get("lp_box_idx")
    rp_idx = match_info.get("rp_box_idx")

    # 画框：绿色；lp 命中红；rp 命中蓝；两者同框紫色（虽然默认不允许）
    if boxes_np.size:
        for i in range(boxes_np.shape[0]):
            x1, y1, x2, y2 = boxes_np[i]
            lp_hit = (lp_idx == i)
            rp_hit = (rp_idx == i)
            if lp_hit and rp_hit:
                color = (255, 0, 255)
            elif lp_hit:
                color = (0, 0, 255)
            elif rp_hit:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

    # 画关键点：h=蓝，其余=红（与你现有颜色保持一致）
    for k in pose_items:
        if not k:
            continue
        px, py = int(k["x"]), int(k["y"])
        name = k.get("name", "?")
        color = (255, 0, 0) if name == "h" else (0, 0, 255)
        cv2.circle(vis, (px, py), 2, color, -1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)

def process_swd_matching(chosen_dirs, 
                         require_different_boxes=REQUIRE_DIFFERENT_BOXES,
                         use_point_conf=USE_POINT_CONF,
                         kpt_conf_thr=KPT_CONF_THR,
                         do_vis=DO_VIS,
                         limit_vis=LIMIT_VIS,
                         copy_unmatched=COPY_UNMATCHED):
    """
    批处理 chosen_dirs 列表中的目录进行 SWD 匹配
    
    Args:
        chosen_dirs: 目录路径列表
        require_different_boxes: 两翼必须落入不同框
        use_point_conf: 若关键点带 conf，则应用阈值过滤
        kpt_conf_thr: 关键点置信度阈值
        do_vis: 输出可视化图
        limit_vis: 仅可视化前 N 张（None 为全部）
        copy_unmatched: 也拷贝未匹配样本，便于人工复核
    """
    for d in chosen_dirs:
        base = d.parent / (d.name + "_sliced_merge")
        crops_dir = base / "01_cropped_objects"

        pose_json = base / f"pose_and_det_{version}" / "02_pose_predicted_results.json"
        dot_json  = base / f"pose_and_det_{version}" / "03_dot_predicted_results.json"
        out_json  = base / f"pose_and_det_{version}" / "04_pose_wing_matched_dot_results.json"
        vis_dir   = base / f"pose_and_det_{version}" / "04_detected_swd_pose_vis"

        if not pose_json.exists() or not dot_json.exists():
            print(f"⚠️ 缺少输入：{pose_json} 或 {dot_json}，跳过 {d}")
            continue

        pose_list: List[Dict[str, Any]] = load_json(pose_json)
        dot_list:  List[Dict[str, Any]] = load_json(dot_json)

        # ---------- 预处理 dot_list：uuid -> np.ndarray(N,4) ----------
        dot_map_np: Dict[str, np.ndarray] = {}
        for item in dot_list:
            u = item.get("uuid")
            if not u:
                fname = item.get("file") or os.path.basename(item.get("path", ""))
                m = UUID_RE.search(str(fname))
                u = m.group(1) if m else None
            if not u:
                continue

            boxes = item.get("boxes", [])
            if not boxes:
                dot_map_np[u] = np.empty((0, 4), dtype=np.float32)
                continue

            # 只收 bbox，过滤非 list/tuple
            arr = [b["bbox"] for b in boxes if isinstance(b, dict) and isinstance(b.get("bbox"), (list, tuple)) and len(b["bbox"]) == 4]
            if arr:
                dot_map_np[u] = np.asarray(arr, dtype=np.float32)
            else:
                dot_map_np[u] = np.empty((0, 4), dtype=np.float32)

        out_rows: List[Dict[str, Any]] = []
        matched_cnt = 0
        total_cnt = 0
        vis_written = 0
        passcount = 0

        if do_vis:
            vis_dir.mkdir(parents=True, exist_ok=True)

        # ---------- 主循环（热点路径只走 Python 最少分支） ----------
        for item in pose_list:
            uuid_ = item.get("uuid")
            total_cnt += 1

            insts = item.get("instances", [])
            if not insts:
                out_rows.append({"uuid": uuid_, "matched": False, "reason": "no_pose", "path": item.get("path"), "file": item.get("file")})
                passcount += 1
                continue

            pose_items = insts[0].get("kpts") or []
            boxes_np = dot_map_np.get(uuid_, np.empty((0, 4), dtype=np.float32))

            info = match_one(
                uuid_=uuid_,
                pose_items=pose_items,
                dot_boxes_np=boxes_np,
                require_different_boxes=require_different_boxes,
                use_point_conf=use_point_conf,
                kpt_conf_thr=kpt_conf_thr,
            )
            info["boxes"] = boxes_np.tolist() if boxes_np.size else []
            info["path"]  = item.get("path")
            info["file"]  = item.get("file")
            info["pose_boxes"] = insts[0].get("box") if insts and isinstance(insts[0], dict) else None
            out_rows.append(info)

            if info["matched"]:
                matched_cnt += 1

            if do_vis and (limit_vis is None or vis_written < limit_vis):
                fname = item.get("file")
                if fname:
                    img_path = crops_dir / fname
                    if img_path.exists():
                        out_img = vis_dir / f"{Path(fname).stem}_vis.jpg"
                        visualize_match(img_path, pose_items, boxes_np, info, out_img)
                        vis_written += 1

        # 保存 JSON（orjson 更快；无则回退）
        save_json(out_rows, out_json)

        # === 拷贝 matched 与（可选）unmatched 到独立文件夹（含原图与可视化） ===
        confirmed_raw = base / f"pose_and_det_{version}" / "04_confirmed_swd" / "raw"
        confirmed_vis = base / f"pose_and_det_{version}" / "04_confirmed_swd" / "vis"
        review_raw    = base / f"pose_and_det_{version}" / "04_review_unmatched" / "raw"
        review_vis    = base / f"pose_and_det_{version}" / "04_review_unmatched" / "vis"
        confirmed_raw.mkdir(parents=True, exist_ok=True)
        confirmed_vis.mkdir(parents=True, exist_ok=True)
        if copy_unmatched:
            review_raw.mkdir(parents=True, exist_ok=True)
            review_vis.mkdir(parents=True, exist_ok=True)

        copied_match = copied_unmatch = 0
        for info in out_rows:
            fname = info.get("file")
            if not fname:
                continue
            src_raw = crops_dir / fname
            if not src_raw.exists():
                continue
            src_vis = vis_dir / f"{Path(fname).stem}_vis.jpg"

            if info.get("matched", False):
                shutil.copy2(src_raw, confirmed_raw / fname)
                if src_vis.exists():
                    shutil.copy2(src_vis, confirmed_vis / src_vis.name)
                copied_match += 1
            elif copy_unmatched:
                shutil.copy2(src_raw, review_raw / fname)
                if src_vis.exists():
                    shutil.copy2(src_vis, review_vis / src_vis.name)
                copied_unmatch += 1

        # 统计打印
        match_ratio = (matched_cnt / total_cnt * 100.0) if total_cnt else 0.0
        print(f"\n=== 匹配完成: {d.name} ===")
        print(f"总小图: {total_cnt}")
        print(f"匹配为 SWD: {matched_cnt}")
        print(f"匹配率: {match_ratio:.2f}%")
        print(f"结果 JSON: {out_json}")
        print(f"已拷贝 matched: {copied_match} 张 -> {confirmed_raw} | {confirmed_vis}")
        if copy_unmatched:
            print(f"已拷贝 unmatched: {copied_unmatch} 张 -> {review_raw} | {review_vis}")
        if do_vis:
            print(f"可视化目录: {vis_dir} （已写 {vis_written} 张）")
            print(f"跳过无姿态图片: {passcount} 张")

# %%
if "process_swd_matching" in steps_to_run:
    stats = process_swd_matching(
        chosen_dirs=chosen_dirs,
        require_different_boxes=True,   # True = 强制 lp 与 rp 命中不同小黑点框, False = 可命中同一框
        use_point_conf=True,            # True = 使用关键点置信度阈值, False = 不使用   
        kpt_conf_thr=0.88,              # 关键点置信度阈值（仅在 use_point_conf=True 时生效）
        do_vis=True,                    # False = 所有文件均不输出可视化图，True = 输出可视化图
        limit_vis=None,                 # 仅可视化前 N 张；None 为全部
        copy_unmatched=False,           # 是否拷贝未匹配样本，便于人工复核
    )
else:
    print("Step_6 判定 SWD ⇒ 匹配规则：两翼关键点分别落入两个不同的小黑点框 被跳过")

# %% [markdown]
# # Step_7  根据判定结果，过滤错误数据
# 

# %%
import json
from pathlib import Path
from typing import Dict, Any


def filter_annotations_by_matched_uuid(
    matched_file: Path, annotations_file: Path, output_file: Path
) -> None:
    """
    根据 matched_file 中标记 matched=True 的 uuid，
    从 annotations_file 中筛选对应的标注，并保存到 output_file。

    Args:
        matched_file (str): 匹配结果 JSON 文件路径，例如 '04_pose_wing_matched_dot_results.json'
        annotations_file (str): 原始标注 JSON 文件路径，例如 '01_merged_annotations.json'
        output_file (str): 输出 JSON 文件路径，例如 '09_filtered_annotations.json'
    """

    # 读取数据
    with matched_file.open("r", encoding="utf-8") as f:
        matched_data = json.load(f)

    with annotations_file.open("r", encoding="utf-8") as f:
        annotations_data: Dict[str, Any] = json.load(f)

    # 获取 matched 的 UUID
    matched_uuids = {item["uuid"] for item in matched_data if item.get("matched")}

    # 过滤标注
    filtered_annotations = {
        name: [ann for ann in anns if ann["uuid"] in matched_uuids]
        for name, anns in annotations_data.items()
    }
    # 移除空的类别
    filtered_annotations = {
        name: anns for name, anns in filtered_annotations.items() if anns
    }

    # 保存结果
    output_path = output_file
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(filtered_annotations, f, indent=2, ensure_ascii=False)

    print(f"✅ 过滤完成！结果已保存到 {output_path}")

# %%
for chosen_dir in chosen_dirs:
    print(f"选择的目录: {chosen_dir}")
    filter_annotations_by_matched_uuid(
        matched_file= chosen_dir.parent / (chosen_dir.name + "_sliced_merge") / f"pose_and_det_{version}" / "04_pose_wing_matched_dot_results.json",
        annotations_file=chosen_dir.parent / (chosen_dir.name + "_sliced_merge") / "01_merged_annotations.json",
        output_file=chosen_dir.parent / (chosen_dir.name + "_sliced_merge") / f"pose_and_det_{version}" / "09_filtered_annotations.json"
    )


# %% [markdown]
# # 查看根目录下需要运行的文件夹

# %%
from pathlib import Path

version = "v1"
run_type = "pose_and_det"  # "pose_and_det" or "cls"

# path = Path("/workspace/models/SAHI/run_v8")
# chosen_dirs = [d / "raw_data" for d in path.iterdir() if d.is_dir()]
# print("子文件夹路径列表：", chosen_dirs)

# %% [markdown]
# # objects去重，结果可视化
# ### 输入输出
# ![image-4.png](attachment:image-4.png)
# 
# 
# ### 效果
# ![image.png](attachment:image.png)![image-2.png](attachment:image-2.png)
# 
# ![image-3.png](attachment:image-3.png)![image-5.png](attachment:image-5.png)

# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
时间序列目标跟踪与可视化系统（仅多边形 + poly_iou）
- 输入：类似 09_filtered_annotations.json 的 dict[name] -> List[annotation]，annotation 必须包含:
        label, points(多边形), original_name(可选), uuid(可选), score(可选)
- 原图目录：raw_data/ 下的 *.jpg，文件名以键名为前缀（如 0801_1034_880*.jpg）
- 命名解析：MMDD_HHMM（示例：0801_1034_880）

功能：
1) assign_persistent_ids：跨时刻匹配分配稳定 ID（poly IoU）
2) draw_overlays：原图叠加可视化（新=绿，重复=红、幽灵轨迹、右上角 NOW/SUM 徽标）
3) build_track_galleries：按 ID 裁剪时间序列小图
4) export_stats_B：导出统计（Slots.csv & IDs.csv）
5) 保存完整时间线 JSON（timeline.json）

依赖：shapely, numpy, opencv-python, orjson
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Iterable
from collections import defaultdict, Counter

import numpy as np
import cv2
import orjson
import json as pyjson

# ============== 强制依赖 shapely（仅多边形 + poly_iou） ==============
try:
    from shapely.geometry import Polygon
except Exception as e:
    raise ImportError(
        "本脚本仅支持多边形 + poly_iou，请先安装 shapely：\n"
        "  pip install shapely\n"
    ) from e


# ============== 时间解析：文件名 MMDD_HHMM[...] ==============
_FN_RE = re.compile(r"(?P<mm>\d{2})(?P<dd>\d{2})_(?P<hh>\d{2})(?P<mi>\d{2})")

def parse_mmdd_hhmm(name: str) -> Optional[Tuple[str, str]]:
    """
    返回 (date_str 'MM-DD', time_str 'HH:MM')，失败返回 None
    """
    m = _FN_RE.search(name)
    if not m:
        return None
    mm, dd, hh, mi = m.group("mm", "dd", "hh", "mi")
    return f"{mm}-{dd}", f"{hh}:{mi}"

def slot_sort_key(date_str: str, time_str: str) -> Tuple[int,int,int,int]:
    return (int(date_str[:2]), int(date_str[3:]), int(time_str[:2]), int(time_str[3:]))


# ============== IO ==============
def json_load(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return orjson.loads(f.read())

def json_dump(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))


# ============== 原图索引 ==============
def build_image_index(image_dir: Path) -> Dict[str, str]:
    """
    返回 {stem -> path}，若找不到精确 stem，后续会尝试前缀匹配
    """
    idx = {}
    for p in image_dir.glob("*.jpg"):
        idx[p.stem] = str(p)
    return idx


# ============== 多边形 IoU（仅 shapely） ==============
def poly_iou(poly_a: List[List[float]], poly_b: List[List[float]]) -> float:
    try:
        A = Polygon(poly_a)
        B = Polygon(poly_b)
        if not (A.is_valid and B.is_valid):
            return 0.0
        inter = A.intersection(B).area
        if inter <= 0:
            return 0.0
        u = A.area + B.area - inter
        return float(inter / u) if u > 0 else 0.0
    except Exception:
        return 0.0


# ============== 标签标准化 / 过滤（可选） ==============
def make_label_normalizer(label_map: Optional[Dict[str, str]] = None,
                          whitelist: Optional[Iterable[str]] = None):
    """
    返回 normalize(label) -> 标准化后的 label
    - label_map: 别名到统一名，如 {'SWD':'swd','MAYSWD':'mayswd','may_swd':'mayswd'}
    - whitelist: 只保留白名单中的标签；不在白名单则返回 'other'（或返回 '' 表示忽略）
    """
    label_map = {k.lower(): v for k, v in (label_map or {}).items()}
    wl = set(x.lower() for x in whitelist) if whitelist else None

    def normalize(label: str) -> str:
        if label is None:
            return ""
        s = str(label).strip()
        if not s:
            return ""
        s_lo = s.lower()
        s_std = label_map.get(s_lo, s_lo)
        if wl is not None and s_std not in wl:
            return "other"  # 如需忽略可改为返回 ""
        return s_std
    return normalize


# ============== 数据读取：09_filtered_annotations.json（支持 normalize_label） ==============
def load_annotations_json(json_path: Path, 
                            normalize_label=None) -> Dict[str, List[Dict[str, Any]]]:
    """
    期望结构：{ img_key: [ {label, points, ...}, ... ], ... }
    仅使用 points 多边形；若缺失则跳过该条 annotation
    """
    if normalize_label is None:
        normalize_label = lambda x: ("" if x is None else str(x))

    data = json_load(json_path)
    cleaned: Dict[str, List[Dict[str, Any]]] = {}
    for img_key, anns in data.items():
        keep = []
        for a in anns or []:
            pts = a.get("points")
            if isinstance(pts, list) and len(pts) >= 3:
                lab = normalize_label(a.get("label"))
                keep.append({
                    "label": lab,
                    "points": [[float(x), float(y)] for x, y in pts],
                    "uuid": a.get("uuid"),
                    "original_name": a.get("original_name", img_key),
                    "score": a.get("score"),
                })
        if keep:
            cleaned[img_key] = keep
    return cleaned


# ============== 稳定 ID 分配（仅 poly_iou） ==============
def assign_persistent_ids(
    cleaned_annotations: Dict[str, List[Dict[str, Any]]],
    iou_threshold: float = 0.5,
    class_agnostic: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    """
    返回：
      - timeline: List[ {date,time,img,label,id,is_new,repeat_idx,points} ]
      - id_tracks: Dict[id] -> List[occurrence(dict)]
    匹配策略：
      - 同时刻内部去重：相同 label（或 class_agnostic=True）之间 IoU >= 阈值视为重复，仅保留一个
      - 跨时刻匹配：与“已见库”中 IoU 最高且 >= 阈值者匹配，否则分配新 ID
    """
    by_slot: Dict[Tuple[str, str], List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for img_key, anns in cleaned_annotations.items():
        ts = parse_mmdd_hhmm(img_key)
        if not ts:
            continue
        d, t = ts
        for a in anns:
            by_slot[(d, t)].append((img_key, a))

    slots = sorted(by_slot.keys(), key=lambda k: slot_sort_key(k[0], k[1]))

    next_id = 1
    pool_by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    pool_all: List[Dict[str, Any]] = []
    id_tracks: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    timeline: List[Dict[str, Any]] = []

    for d, t in slots:
        obs = by_slot[(d, t)]

        # 1) 同时刻内部去重
        unique_obs: List[Tuple[str, Dict[str, Any]]] = []
        for img_key, det in obs:
            label, pts = det.get("label", ""), det.get("points")
            if not pts:
                continue
            dup = False
            for _, u in unique_obs:
                if (not class_agnostic) and (u.get("label", "") != label):
                    continue
                s = poly_iou(pts, u["points"])
                if s >= iou_threshold:
                    dup = True
                    break
            if not dup:
                unique_obs.append((img_key, det))

        # 2) 跨时刻匹配
        for img_key, det in unique_obs:
            label, pts = det.get("label", ""), det.get("points")
            candidates = pool_all if class_agnostic else pool_by_label[label]
            best = None
            best_s = -1.0
            for c in candidates:
                s = poly_iou(pts, c["points"])
                if s >= iou_threshold and s > best_s:
                    best = c
                    best_s = s

            if best is None:
                cur_id = next_id
                next_id += 1
                entry = {"id": cur_id, "points": pts, "last_dt": (d, t)}
                if class_agnostic:
                    pool_all.append(entry)
                else:
                    pool_by_label[label].append(entry)
                repeat_idx = 1
                is_new = True
            else:
                cur_id = best["id"]
                best["points"] = pts or best["points"]
                best["last_dt"] = (d, t)
                repeat_idx = len(id_tracks[cur_id]) + 1
                is_new = False

            occ = {
                "date": d, "time": t, "img": img_key,
                "label": label, "id": cur_id, "is_new": is_new,
                "repeat_idx": repeat_idx, "points": pts
            }
            id_tracks[cur_id].append(occ)
            timeline.append(occ)

    return timeline, id_tracks


# ============== 可视化：右上角计数徽标 ==============
def draw_top_right_counter(canvas: np.ndarray, now: int, cum: int):
    H, W = canvas.shape[:2]
    margin = max(8, W // 200)
    pad = max(8, W // 300)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.8, min(2.2, W / 800.0))
    thickness = max(2, int(round(font_scale + 1)))

    line1 = f"NOW: {now}"
    line2 = f"SUM: {cum}"

    (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
    (w2, h2), _ = cv2.getTextSize(line2, font, font_scale, thickness)
    box_w = max(w1, w2) + 2 * pad
    line_gap = max(6, int(0.25 * h1))
    box_h = h1 + h2 + line_gap + 2 * pad

    x2 = W - margin
    y1 = margin
    x1 = x2 - box_w
    y2 = y1 + box_h

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, dst=canvas)

    tx = x1 + pad
    ty1 = y1 + pad + h1
    ty2 = ty1 + line_gap + h2
    cv2.putText(canvas, line1, (tx, ty1), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(canvas, line2, (tx, ty2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


# ============== 可视化：叠加绘制（新=绿，重复=红，幽灵轨迹） ==============
def draw_overlays(
    timeline: List[Dict[str, Any]],
    image_dir: Path,
    out_dir: Path,
    ghost_trail_steps: int = 0,
    ghost_alpha: float = 0.25
):
    os.makedirs(out_dir, exist_ok=True)
    index = build_image_index(image_dir)

    # 分图收集
    by_img = defaultdict(list)
    for r in timeline:
        by_img[r["img"]].append(r)

    # 按时间排序的全局时间线，用于回溯轨迹
    def dt_key(r): return slot_sort_key(r["date"], r["time"])
    timeline_sorted = sorted(timeline, key=dt_key)
    hist_by_id = defaultdict(list)
    for r in timeline_sorted:
        hist_by_id[r["id"]].append(r)

    # 为徽标准备 per-slot now/cum，并映射到图
    per_slot_counts = Counter()
    cum_total = 0
    slot_order = sorted({(r["date"], r["time"]) for r in timeline}, key=lambda x: slot_sort_key(*x))
    slot_to_cum = {}
    for d, t in slot_order:
        now = sum(1 for r in timeline if r["date"] == d and r["time"] == t and r["is_new"])
        cum_total += now
        per_slot_counts[(d, t)] = now
        slot_to_cum[(d, t)] = cum_total

    for img_key, rows in by_img.items():
        # 找原图
        img_path = index.get(img_key)
        if img_path is None:
            for p in Path(image_dir).glob(f"{img_key}*.jpg"):
                img_path = str(p); break
        if not img_path or not os.path.exists(img_path):
            continue

        canvas = cv2.imread(img_path)
        if canvas is None:
            continue

        # 幽灵轨迹
        if ghost_trail_steps > 0:
            ghost = canvas.copy()
            for r in rows:
                hist = hist_by_id[r["id"]]
                idx = None
                for i, k in enumerate(hist):
                    if k["img"] == img_key:
                        idx = i; break
                if idx is None: 
                    continue
                start = max(0, idx - ghost_trail_steps)
                for j in range(start, idx):
                    pj = hist[j]
                    pts = np.asarray(pj["points"], dtype=np.int32).reshape(-1,1,2)
                    cv2.polylines(ghost, [pts], True, (200,200,200), 1, cv2.LINE_AA)
            canvas = cv2.addWeighted(ghost, ghost_alpha, canvas, 1-ghost_alpha, 0)

        # 当前多边形
        for r in rows:
            pts = r["points"]
            if not pts:
                continue
            pts_i = np.asarray(pts, dtype=np.int32).reshape(-1,1,2)
            color = (0,255,0) if r["is_new"] else (0,0,255)  # 新=绿，重复=红
            cv2.polylines(canvas, [pts_i], True, color, 2, cv2.LINE_AA)

            # 标注文字（ID/label/出现次数）
            if pts_i.size > 0:
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                x1, y1 = int(min(xs)) + 8, int(min(ys)) + 18
                txt = f"ID#{r['id']} {r.get('label','')} x{r['repeat_idx']}"
                cv2.putText(canvas, txt, (x1 + 64, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # 右上角徽标
        dd, tt = parse_mmdd_hhmm(img_key) or (None, None)
        if dd and tt:
            now = per_slot_counts[(dd, tt)]
            cum = slot_to_cum[(dd, tt)]
            draw_top_right_counter(canvas, now=now, cum=cum)

        out_path = os.path.join(out_dir, f"{img_key}_track_vis.jpg")
        cv2.imwrite(out_path, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


# ============== 轨迹相册（按 ID 裁剪序列） ==============
def _crop_square(img: np.ndarray, pts: List[List[float]], margin: int = 8) -> Optional[np.ndarray]:
    H, W = img.shape[:2]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    l = max(0, cx - side//2 - margin)
    t = max(0, cy - side//2 - margin)
    r = min(W, cx + side//2 + margin)
    b = min(H, cy + side//2 + margin)
    if r - l <= 1 or b - t <= 1:
        return None
    return img[t:b, l:r].copy()

def build_track_galleries(
    id_tracks: Dict[int, List[Dict[str, Any]]],
    image_dir: Path,
    out_dir: Path,
    margin: int = 8,
    workers: Optional[int] = None,
    jpeg_quality: int = 90
):
    os.makedirs(out_dir, exist_ok=True)
    index = build_image_index(image_dir)
    # 每个 id 一个文件夹
    save_dirs = {}
    for tid in id_tracks.keys():
        d = os.path.join(out_dir, f"id_{tid:04d}")
        os.makedirs(d, exist_ok=True)
        save_dirs[tid] = d

    tasks_by_img: Dict[str, List[Tuple[List[List[float]], str]]] = defaultdict(list)
    _fallback: Dict[str, Optional[str]] = {}

    for tid, occs in id_tracks.items():
        occs_sorted = sorted(occs, key=lambda r: slot_sort_key(r["date"], r["time"]))
        for k, r in enumerate(occs_sorted, start=1):
            pts = r.get("points")
            if not pts:
                continue
            img_key = r["img"]
            img_path = index.get(img_key)
            if img_path is None:
                if img_key not in _fallback:
                    hit = None
                    for p in Path(image_dir).glob(f"{img_key}*.jpg"):
                        hit = str(p); break
                    _fallback[img_key] = hit
                img_path = _fallback[img_key]
            if not img_path or not os.path.exists(img_path):
                continue
            fn = f"{k:02d}_{r['date']}_{r['time']}_{img_key}.jpg"
            save_path = os.path.join(save_dirs[tid], fn)
            tasks_by_img[img_path].append((pts, save_path))

    if not tasks_by_img:
        return

    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    def _process(img_path: str, todo: List[Tuple[List[List[float]], str]]):
        img = cv2.imread(img_path)
        if img is None:
            return 0
        ok = 0
        for pts, save_path in todo:
            crop = _crop_square(img, pts, margin=margin)
            if crop is None:
                continue
            cv2.imwrite(save_path, crop, params)
            ok += 1
        return ok

    if workers is None:
        workers = min(32, (os.cpu_count() or 4) + 4)

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_process, p, todo) for p, todo in tasks_by_img.items()]
        for _ in concurrent.futures.as_completed(futs):
            pass


# ============== 统一统计（按你的字段要求改名/精简） ==============
def export_stats_B(
    timeline: list,
    slots_csv_path: str,
    ids_csv_path: str,
    final_label_by_id: dict | None = None,      # 仍保留接口，但不再输出到 IDs.csv
    final_conf_by_id: dict | None = None
):
    """
    输出：
    - Slots.csv：datetime,new,repeat,total,cumulative_total,<label>_new,<label>_repeat,<label>_total,...,new_rate
      （移除 date,time）
    - IDs.csv：id,appearances_times,first_time_slot,last_time_slot,main_label,main_ratio,
               labels_present(对象字符串),label_switch_times
      （移除 span_slots/final_label/final_label_confidence/num_labels/purity/各 <label>_count 动态列）
    """
    import csv
    from collections import defaultdict, Counter

    # 空输入时：写表头
    if not timeline:
        with open(slots_csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["datetime","new","repeat","total","cumulative_total","new_rate"])
        with open(ids_csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "id","appearances_times","first_time_slot","last_time_slot",
                "main_label","main_ratio","labels_present","label_switch_times"
            ])
        return

    def slot_key(d, t):  # d='MM-DD', t='HH:MM'
        return (int(d[:2]), int(d[3:]), int(t[:2]), int(t[3:]))

    # —— 标签全集（按字母序稳定） —— #
    labels = sorted({str(r.get("label","")) for r in timeline if str(r.get("label","")) != ""})

    # =========================
    # 1) SLOTS（每时间槽聚合）
    # =========================
    grp = defaultdict(list)  # (date,time) -> [rows...]
    for r in timeline:
        grp[(r["date"], r["time"])] .append(r)

    ordered_slots = sorted(grp.keys(), key=lambda dt: slot_key(dt[0], dt[1]))

    cum_total = 0
    slot_rows = []
    for d, t in ordered_slots:
        sub = grp[(d, t)]
        new_cnt    = sum(1 for x in sub if x.get("is_new", False) is True)
        repeat_cnt = sum(1 for x in sub if x.get("is_new", False) is False)
        total_cnt  = len(sub)
        cum_total += new_cnt

        row = {
            "datetime": f"{d} {t}",
            "new": new_cnt,
            "repeat": repeat_cnt,
            "total": total_cnt,
            "cumulative_total": cum_total,
            "new_rate": (new_cnt / total_cnt) if total_cnt else 0.0,
        }

        for lab in labels:
            lab_sub = [x for x in sub if str(x.get("label","")) == lab]
            lab_new = sum(1 for x in lab_sub if x.get("is_new", False) is True)
            lab_rep = sum(1 for x in lab_sub if x.get("is_new", False) is False)
            row[f"{lab}_new"]    = lab_new
            row[f"{lab}_repeat"] = lab_rep
            row[f"{lab}_total"]  = lab_new + lab_rep

        slot_rows.append(row)

    slot_header = ["datetime","new","repeat","total","cumulative_total"]
    for lab in labels:
        slot_header += [f"{lab}_new", f"{lab}_repeat", f"{lab}_total"]
    slot_header += ["new_rate"]

    with open(slots_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=slot_header)
        w.writeheader()
        for r in slot_rows:
            w.writerow(r)

    # =========================
    # 2) IDs（每轨迹聚合）
    # =========================
    by_id = defaultdict(list)
    for r in timeline:
        by_id[int(r["id"])] .append(r)

    id_rows = []
    for tid, occs in by_id.items():
        occs_sorted = sorted(occs, key=lambda r: slot_key(r["date"], r["time"]))
        appearances = len(occs_sorted)
        first_slot  = f"{occs_sorted[0]['date']}_{occs_sorted[0]['time']}"
        last_slot   = f"{occs_sorted[-1]['date']}_{occs_sorted[-1]['time']}"

        # —— 计数每个 label —— #
        cnt = Counter(str(o.get("label","")) for o in occs_sorted if str(o.get("label","")) != "")
        main_lab, main_cnt = ("", 0)
        if cnt:
            main_lab, main_cnt = max(cnt.items(), key=lambda kv: kv[1])
        main_ratio = (main_cnt / appearances) if appearances else 0.0

        # —— 类别切换次数（按时间序） —— #
        label_seq = [str(o.get("label","")) for o in occs_sorted]
        label_seq = [x for x in label_seq if x != ""]
        switch_times = sum(1 for i in range(1, len(label_seq)) if label_seq[i] != label_seq[i-1])

        # —— labels_present 以对象字符串输出 —— #
        labels_present_obj = {lab: cnt[lab] for lab in sorted(cnt.keys())}
        labels_present_str = pyjson.dumps(labels_present_obj, ensure_ascii=False, separators=(',',':'))

        row = {
            "id": tid,
            "appearances_times": appearances,
            "first_time_slot": first_slot,
            "last_time_slot": last_slot,
            "main_label": main_lab,
            "main_ratio": main_ratio,
            "labels_present": labels_present_str,
            "label_switch_times": switch_times,
        }

        id_rows.append(row)

    id_header = [
        "id","appearances_times","first_time_slot","last_time_slot",
        "main_label","main_ratio","labels_present","label_switch_times"
    ]

    with open(ids_csv_path, "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=id_header)
        w.writeheader()
        for r in sorted(id_rows, key=lambda x: x["id"]):
            w.writerow(r)



# ============== 主流程（支持标签归一/白名单） ==============
def run_pipeline(
    annotations_json: Path,
    image_dir: Path,
    out_root: Path,
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
    ghost_trail_steps: int = 0,
    ghost_alpha: float = 0.25,
    label_map: Optional[Dict[str,str]] = None,        # 可选：别名到统一名
    label_whitelist: Optional[Iterable[str]] = None   # 可选：只保留这些标签，其它归 'other'
):
    os.makedirs(out_root, exist_ok=True)

    # 0) 标签标准化器（可选）
    normalizer = make_label_normalizer(label_map, label_whitelist)

    # 1) 读注释（加入标准化）
    cleaned = load_annotations_json(annotations_json, normalize_label=normalizer)

    # 2) 分配稳定 ID（poly IoU）
    timeline, id_tracks = assign_persistent_ids(
        cleaned, iou_threshold=iou_threshold, class_agnostic=class_agnostic
    )

    # 3) 可视化叠加
    vis_dir = out_root / "vis"
    draw_overlays(
        timeline, image_dir=image_dir, out_dir=vis_dir,
        ghost_trail_steps=ghost_trail_steps, ghost_alpha=ghost_alpha
    )

    # 4) 轨迹相册
    crops_dir = out_root / "galleries"
    build_track_galleries(id_tracks, image_dir=image_dir, out_dir=crops_dir)

    # 5) 统一统计
    slots_csv = os.path.join(out_root, "Slots.csv")
    ids_csv   = os.path.join(out_root, "IDs.csv")

    export_stats_B(
        timeline,
        slots_csv_path=slots_csv,
        ids_csv_path=ids_csv,
        final_label_by_id=None,
        final_conf_by_id=None
    )

    # 6) 保存时间线
    json_dump(timeline, out_root / "timeline.json")

    # 控制台摘要（含本批标签集合）
    labels_present = sorted({str(r.get("label","")) for r in timeline if str(r.get("label","")) != ""})
    num_ids = len(id_tracks)
    num_slots = len(set((r["date"], r["time"]) for r in timeline))
    num_obs = len(timeline)
    print("=== Summary ===")
    print(f"Observations (timeline rows): {num_obs}")
    print(f"Unique IDs: {num_ids}")
    print(f"Time slots: {num_slots}")
    print(f"Labels in this dataset: {labels_present}  (count={len(labels_present)})")
    print(f"Output:")
    print(f"  - Overlays:   {vis_dir}")
    print(f"  - Galleries:  {crops_dir}")
    print(f"  - Slots CSV:  {slots_csv}")
    print(f"  - IDs CSV:    {ids_csv}")
    print(f"  - Timeline:   {os.path.join(out_root, 'timeline.json')}")

# %%
for chosen_dir in chosen_dirs:
    print(f"Processing directory: {chosen_dir}")
    annotations_file = chosen_dir.parent / "raw_data_sliced_merge"  / f"{run_type}_{version}" / "09_filtered_annotations.json"
    if not annotations_file.exists():
        print(f"Annotations file not found: {annotations_file}")
        continue

    raw_image_dir = chosen_dir.parent / "raw_data"
    if not raw_image_dir.exists():
        print(f"Raw image directory not found: {raw_image_dir}")
        continue

    output_directory = chosen_dir.parent / "raw_data_sliced_merge" / f"{run_type}_{version}" / "10_visualization_tracking_results"
    run_pipeline(
        annotations_json=annotations_file,
        image_dir=raw_image_dir,
        out_root=output_directory,
        iou_threshold=0.5,
        class_agnostic=True,  # True=类别如果不同也可匹配为重复objects，False=必须同类才记为重复objects 
        ghost_trail_steps=1,
        ghost_alpha=0.25,
        label_map=None,
        label_whitelist=['swd', 'mayswd']  # 只保留这些标签，其它归 'other'
    )

# %%



# %% [markdown]
# # 查看根目录下需要运行的文件夹

# %%
from pathlib import Path

version = "v1"
run_type = "pose_and_det"  # "pose_and_det" or "cls"

# path = Path("/workspace/models/SAHI/run_v8")
# chosen_dirs = [d / "raw_data" for d in path.iterdir() if d.is_dir()]
# print("子文件夹路径列表：", chosen_dirs)

# %% [markdown]
# # 简单统计 -- 统计各类别objects数量，并导出为CSV文件
# ### 输入输出
# ![image.png](attachment:image.png)
# 
# ### 效果
# ![image-2.png](attachment:image-2.png)

# %%
# 统计每个类别在每个时间点的数量并导出为 CSV

"""
统计: 各类别(label)在每个时刻(original_name解析)的数量 & 按时间排序后的累积数量
输入: 03_filtered_annotations.json
输出: 04_filtered_annotations_counts.csv  |  04_filtered_annotations_cumulative.csv
"""

import json, csv, re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Iterable, Optional

# ───────────────────────────────────────────────────────────────────────────────
# 1) original_name → 解析时间键
#    假设命名类似: "0801_1203_840" → MMDD=0801, HHMM=1203
#    若解析失败: 回退到原字符串并按自然排序
# ───────────────────────────────────────────────────────────────────────────────

_TIME_RE = re.compile(r"^(\d{4})_(\d{4})(?:_.+)?$")  # e.g. 0801_1203_...

def parse_time_key(name: str) -> Tuple[int,int,int,int,str]:
    """
    返回一个可排序的键 (MM, DD, HH, mm, display_str)
    解析失败时, 返回 (9999, 9999, 99, 99, name) 确保排在最后并用原名展示
    """
    m = _TIME_RE.match(name)
    if not m:
        # 回退: 把名字放最后，display 就用原名
        return (9999, 9999, 99, 99, name)
    mmdd, hhmm = m.group(1), m.group(2)
    try:
        MM  = int(mmdd[:2])
        DD  = int(mmdd[2:])
        HH  = int(hhmm[:2])
        mm  = int(hhmm[2:])
        disp = f"{MM:02d}-{DD:02d} {HH:02d}:{mm:02d}"
        return (MM, DD, HH, mm, disp)
    except Exception:
        return (9999, 9999, 99, 99, name)

# ───────────────────────────────────────────────────────────────────────────────
# 2) 读取与计数
#    JSON 结构: { original_name: [ { label: "...", ... }, ... ], ... }
# ───────────────────────────────────────────────────────────────────────────────

def load_annotations(fp: Path) -> Dict[str, List[dict]]:
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)

def compute_counts_by_time(data: Dict[str, List[dict]]) -> Tuple[List[str], List[Tuple[str, Dict[str,int]]]]:
    """
    返回:
      - 所有出现过的 label 列表(按字母序)
      - 按时间排序的 [(display_time, {label: count, ...}), ...]
    """
    # 收集 label 集
    all_labels = set()
    # 临时记录: time_key → Counter(label)
    time_counters: Dict[Tuple[int,int,int,int,str], Counter] = {}  # tuple: (MM, DD, HH, mm, display_str)

    # 遍历数据, 统计每个时间点的类别数量
    for original_name, ann_list in data.items():
        key = parse_time_key(original_name)
        ctr = time_counters.setdefault(key, Counter())
        for ann in ann_list:
            lbl = ann.get("label")
            if not lbl:
                continue
            all_labels.add(lbl)
            ctr[lbl] += 1
    # 按字母序排序所有 label
    labels_sorted = sorted(all_labels)
    # 按解析后的时间键排序
    entries: List[Tuple[str, Dict[str,int]]] = []
    for k in sorted(time_counters.keys()):
        disp = k[4]                         # display time 部分 -- display_str
        ctr  = time_counters[k] # Counter
        # print(f"[debug] {disp} 计数: {ctr}")
        row_counts = {lbl: ctr.get(lbl, 0) for lbl in labels_sorted} # 保持列顺序
        entries.append((disp, row_counts)) # (display, {label: count, ...})
    return labels_sorted, entries

# ───────────────────────────────────────────────────────────────────────────────
# 3) 写出 CSV: 原始计数 & 累积计数
# ───────────────────────────────────────────────────────────────────────────────

def write_counts_and_cumu_onefile(
    labels: List[str],
    entries: List[Tuple[str, Dict[str, int]]],
    out_path: Path,
) -> None:
    """
    输出一个 CSV，每个类别有两列：<label>_count, <label>_cumu
    """
    # 累积计数器
    cumu = {lbl: 0 for lbl in labels}

    # 构造表头
    headers = ["time"]
    for lbl in labels:
        headers.append(f"{lbl}_count")
        headers.append(f"{lbl}_cumu")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for disp, row_counts in entries:
            row = [disp]
            for lbl in labels:
                count = row_counts.get(lbl, 0)
                cumu[lbl] += count
                row.extend([count, cumu[lbl]])
            writer.writerow(row)



# ───────────────────────────────────────────────────────────────────────────────
# 4) 主流程
# ───────────────────────────────────────────────────────────────────────────────

def run_analysis(
    in_path: Path,
    out_path: Path,
):
    if not in_path.exists():
        raise FileNotFoundError(f"未找到输入文件: {in_path.resolve()}")

    data = load_annotations(in_path)
    labels, entries = compute_counts_by_time(data)

    # 输出一个 CSV
    write_counts_and_cumu_onefile(labels, entries, out_path)

    # 友好打印预览
    print(f"✅ 统计完成，共发现 {len(labels)} 个类别: {labels}")
    print(f"✅ 时间点数量: {len(entries)}")
    print(f"💾 已写出: {out_path}")



# %%
for chosen_dir in chosen_dirs:
    print(f"选择的目录: {chosen_dir}")
    run_analysis(
        in_path = chosen_dir.parent / "raw_data_sliced_merge" / f"{run_type}_{version}" / "09_filtered_annotations.json",
        out_path = chosen_dir.parent / "raw_data_sliced_merge" / f"{run_type}_{version}" / "11_statistics_filtered_annotations_counts.csv",
    )

# %% [markdown]
# # 去重统计 -- 统计各类别除去多张照片之间重复位置的objects数量，并导出为CSV文件

# %%