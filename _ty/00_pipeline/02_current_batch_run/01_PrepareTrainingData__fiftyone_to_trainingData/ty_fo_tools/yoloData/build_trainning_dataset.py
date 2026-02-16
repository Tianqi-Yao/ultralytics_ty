# ty_fo_tools/yolo_sample.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import random
import shutil


@dataclass(frozen=True)
class YoloPair:
    img: Path
    lbl: Optional[Path]  # allow missing label if strict_pair=False


def _list_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def _match_label_for_image(img_path: Path, labels_dir: Path) -> Optional[Path]:
    lbl = labels_dir / f"{img_path.stem}.txt"
    return lbl if lbl.exists() else None


def build_yolo_null_images_dataset(
    *,
    src_images_dir: Path | str,
    src_labels_dir: Path | str,
    out_root_dir: Path | str,
    ratio: Optional[float] = None,
    k: Optional[int] = None,
    seed: int = 42,
    strict_pair: bool = True,
    strict_count: bool = True,
    copy_labels: bool = True,
) -> int:
    """
    Randomly sample YOLO (image, label) pairs into out_root_dir.

    Choose exactly ONE of:
        - ratio: float in (0,1], samples floor(n * ratio) (min 1 if ratio>0)
        - k: int >= 0

    strict_pair:
        - True  -> sampled image missing a .txt raises
        - False -> allow missing labels (lbl=None)

    strict_count:
        - True  -> if k > n_total, raise
        - False -> if k > n_total, set k = n_total

    Output structure:
        out_root_dir/
            images/
            labels/  (if copy_labels=True)

    Returns:
        f samples copied
    """
    if (ratio is None) == (k is None):
        raise ValueError("Specify exactly one of ratio=... or k=... (not both, not neither).")

    if ratio is not None and not (0 < ratio <= 1):
        raise ValueError(f"ratio must be in (0,1], got {ratio}")

    if k is not None and k < 0:
        raise ValueError(f"k must be >= 0, got {k}")

    src_images_dir = Path(src_images_dir)
    src_labels_dir = Path(src_labels_dir)
    out_root_dir = Path(out_root_dir)

    if not src_images_dir.exists():
        raise FileNotFoundError(f"src_images_dir not found: {src_images_dir}")
    if not src_labels_dir.exists():
        raise FileNotFoundError(f"src_labels_dir not found: {src_labels_dir}")

    all_imgs = _list_images(src_images_dir)
    n_total = len(all_imgs)
    if n_total == 0:
        raise ValueError(f"No images found in {src_images_dir}")

    # decide k_final
    if ratio is not None:
        k_final = int(n_total * ratio)
        k_final = max(1, k_final)  # keep behavior similar to your notebook
    else:
        k_final = int(k)  # type: ignore[arg-type]

    if k_final > n_total:
        if strict_count:
            raise ValueError(f"Requested k={k_final}, but only {n_total} images available in {src_images_dir}")
        k_final = n_total

    rng = random.Random(seed)
    chosen_imgs = rng.sample(all_imgs, k_final)

    pairs: List[YoloPair] = []
    for img in chosen_imgs:
        lbl = _match_label_for_image(img, src_labels_dir)
        if strict_pair and lbl is None:
            raise ValueError(f"Missing label for image: {img.name} (expected {src_labels_dir}/{img.stem}.txt)")
        pairs.append(YoloPair(img=img, lbl=lbl))

    out_images = out_root_dir / "images"
    out_labels = out_root_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    if copy_labels:
        out_labels.mkdir(parents=True, exist_ok=True)

    for p in pairs:
        shutil.copy2(p.img, out_images / p.img.name)
        if copy_labels and p.lbl is not None:
            shutil.copy2(p.lbl, out_labels / p.lbl.name)

    return len(pairs)
