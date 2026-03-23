from pathlib import Path
import logging

log_file_name = "check_broken_img.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_name, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
logger.info("============ Notebook logging 初始化完成 ============")

DATA_ROOT = Path("/media/tianqi/16tb/SWD/01_Data/2025_SWD_data_RAW/02_South")

# 可选：扫描并找出损坏图像
from PIL import Image, ImageFile
import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = False


def is_image_ok(p: Path) -> bool:
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            im.load()
        return True
    except Exception:
        return False


bad_img_scan_root = DATA_ROOT
img_paths = [p for p in bad_img_scan_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
bad_imgs = [p for p in tqdm.tqdm(img_paths) if not is_image_ok(p)]
Path("bad_images.txt").write_text("\n".join(map(str, bad_imgs)))
logger.info(f"损坏图像数量: {len(bad_imgs)}")


import shutil

delete_folder = DATA_ROOT / "delete"
delete_folder.mkdir(exist_ok=True)

# bad_img_list_file = Path("bad_images.txt")
# with open(bad_img_list_file, "r") as f:
#     move_list = [Path(line.strip()) for line in f if line.strip()]
move_list = bad_imgs

counter = 0
for image_path in move_list:
    try:
        shutil.move(str(image_path), delete_folder / image_path.name)
        counter += 1
    except Exception as e:
        logger.error(f"移动失败 {image_path}: {e}")
logger.info(f"总共{len(move_list)}张图片，成功移动{counter}张到 {delete_folder}")