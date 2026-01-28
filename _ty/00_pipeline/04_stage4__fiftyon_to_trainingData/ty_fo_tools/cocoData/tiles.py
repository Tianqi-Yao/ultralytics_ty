# ty_fo_tools/coco_tiles.py
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


@dataclass(frozen=True)
class TileSpec:
    """
    Tiling configuration.

    crop_size:
        tile size, e.g. 640 -> 640x640
    overlap_ratio:
        e.g. 0.2 -> 20% overlap
    keep_ratio:
        min intersection/original area ratio to keep a bbox in a tile
        (your notebook used 0.9)
    """
    crop_size: int = 640
    overlap_ratio: float = 0.2
    keep_ratio: float = 0.9

    @property
    def stride(self) -> int:
        return int(self.crop_size * (1 - self.overlap_ratio))


def _make_positions(length: int, crop_size: int, stride: int) -> List[int]:
    """
    Generate tile top-left positions along one axis.
    Guarantees last tile touches the end (like your notebook).
    """
    if length <= crop_size:
        return [0]

    pos_list: List[int] = []
    pos = 0
    while pos + crop_size < length:
        pos_list.append(pos)
        pos += stride

    last = length - crop_size
    if not pos_list or pos_list[-1] != last:
        pos_list.append(last)

    return pos_list


def _crop_bbox_to_tile(
    bbox: List[float],
    tile_x: int,
    tile_y: int,
    tile_size: int,
) -> Tuple[List[float] | None, float]:
    """
    Crop a COCO bbox [x,y,w,h] to a tile and return (new_bbox, keep_ratio).
    new_bbox is in tile-local coordinates.
    """
    x, y, w, h = bbox
    x2, y2 = x + w, y + h

    tx1, ty1 = tile_x, tile_y
    tx2, ty2 = tile_x + tile_size, tile_y + tile_size

    ix1 = max(x, tx1)
    iy1 = max(y, ty1)
    ix2 = min(x2, tx2)
    iy2 = min(y2, ty2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)

    if inter_w <= 0 or inter_h <= 0:
        return None, 0.0

    inter_area = inter_w * inter_h
    orig_area = max(w * h, 1e-6)
    keep_ratio = inter_area / orig_area

    new_bbox = [ix1 - tx1, iy1 - ty1, inter_w, inter_h]
    return new_bbox, keep_ratio


def export_labeled_tiles_from_coco(
    *,
    img_dir: Path | str,
    coco_json: Path | str,
    out_img_dir: Path | str,
    out_json: Path | str,
    spec: TileSpec = TileSpec(),
    remove_empty_tiles: bool = True,
) -> Dict[str, int]:
    """
    Slice images into tiles and export a NEW COCO containing ONLY tiles with kept bboxes.
    (Matches your first tiling main(): it deleted empty tiles.)

    Output:
        out_img_dir: tile images
        out_json: COCO annotation for kept tiles

    Returns:
        {"tiles": int, "annotations": int, "source_images": int}
    """
    img_dir = Path(img_dir)
    coco_json = Path(coco_json)
    out_img_dir = Path(out_img_dir)
    out_json = Path(out_json)

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    coco = json.loads(coco_json.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    # group anns by image_id
    anns_by_img: Dict[int, List[dict]] = {}
    for ann in annotations:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    new_images: List[dict] = []
    new_annotations: List[dict] = []
    new_image_id = 1
    new_ann_id = 1

    for img_info in images:
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        img_path = img_dir / file_name

        img = cv2.imread(str(img_path))
        if img is None:
            # keep behavior minimal: skip unreadable
            continue

        h, w = img.shape[:2]
        xs = _make_positions(w, spec.crop_size, spec.stride)
        ys = _make_positions(h, spec.crop_size, spec.stride)

        orig_anns = anns_by_img.get(img_id, [])

        for ty in ys:
            for tx in xs:
                tile = img[ty:ty + spec.crop_size, tx:tx + spec.crop_size]
                new_file = f"{Path(file_name).stem}_{tx}_{ty}{Path(file_name).suffix}"
                out_path = out_img_dir / new_file

                # write first; may delete later if empty
                cv2.imwrite(str(out_path), tile)

                tile_ann_list: List[dict] = []

                for ann in orig_anns:
                    new_bbox, ratio = _crop_bbox_to_tile(ann["bbox"], tx, ty, spec.crop_size)
                    if new_bbox is None or ratio < spec.keep_ratio:
                        continue

                    new_ann = deepcopy(ann)
                    new_ann["id"] = new_ann_id
                    new_ann["image_id"] = new_image_id
                    new_ann["bbox"] = new_bbox
                    new_ann["area"] = new_bbox[2] * new_bbox[3]
                    tile_ann_list.append(new_ann)
                    new_ann_id += 1

                if remove_empty_tiles and len(tile_ann_list) == 0:
                    out_path.unlink(missing_ok=True)
                    continue

                new_images.append({
                    "id": new_image_id,
                    "file_name": new_file,
                    "width": spec.crop_size,
                    "height": spec.crop_size,
                })
                new_annotations.extend(tile_ann_list)
                new_image_id += 1

    new_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": categories,
    }
    out_json.write_text(json.dumps(new_coco, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "source_images": len(images),
        "tiles": len(new_images),
        "annotations": len(new_annotations),
    }


def export_null_images_tiles_from_coco(
    *,
    img_dir: Path | str,
    coco_json: Path | str,
    out_img_dir: Path | str,
    out_label_dir: Path | str,
    spec: TileSpec = TileSpec(),
    # NOTE: null tiles do not need keep_ratio, but we keep it consistent with your logic:
    # a tile is "has label" if ANY bbox has ratio >= spec.keep_ratio
) -> Dict[str, int]:
    """
    Export NULL tiles: tiles with NO bbox meeting keep_ratio.
    For each null tile, write an empty YOLO txt label file.

    Output:
        out_img_dir/*.jpg
        out_label_dir/*.txt  (empty)

    Returns:
        {"source_images": int, "null_tiles": int}
    """
    img_dir = Path(img_dir)
    coco_json = Path(coco_json)
    out_img_dir = Path(out_img_dir)
    out_label_dir = Path(out_label_dir)

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    coco = json.loads(coco_json.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    anns_by_img: Dict[int, List[dict]] = {}
    for ann in annotations:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    kept_null_tiles = 0

    for img_info in images:
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        img_path = img_dir / file_name

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        xs = _make_positions(w, spec.crop_size, spec.stride)
        ys = _make_positions(h, spec.crop_size, spec.stride)

        orig_anns = anns_by_img.get(img_id, [])

        for ty in ys:
            for tx in xs:
                # check if this tile has any "kept" annotation
                has_label = False
                for ann in orig_anns:
                    _, ratio = _crop_bbox_to_tile(ann["bbox"], tx, ty, spec.crop_size)
                    if ratio >= spec.keep_ratio:
                        has_label = True
                        break
                if has_label:
                    continue

                tile = img[ty:ty + spec.crop_size, tx:tx + spec.crop_size]
                new_file = f"{Path(file_name).stem}_{tx}_{ty}{Path(file_name).suffix}"

                out_img_path = out_img_dir / new_file
                cv2.imwrite(str(out_img_path), tile)

                out_label_path = out_label_dir / f"{Path(new_file).stem}.txt"
                out_label_path.write_text("", encoding="utf-8")

                kept_null_tiles += 1

    return {
        "source_images": len(images),
        "null_tiles": kept_null_tiles,
    }
