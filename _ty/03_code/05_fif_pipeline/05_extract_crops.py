"""
05_extract_crops.py
从 CLF_FIELD 的评估结果中裁取 patch，扩充分类器训练集：
  - TP → 新正样本（swd/round_{ROUND}/）
  - FP → 新负样本（non_swd/fp_round_{ROUND}/）

无 GT 时：CLF_FIELD 所有保留框作为"候选正样本"（需人工确认后在 FiftyOne App 打 tag）
"""

from __future__ import annotations
import logging
from pathlib import Path

from PIL import Image
import fiftyone as fo
from fiftyone import ViewField as F

from pipeline_config import *
from ty_nogt_tools.classify import crop_with_padding

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 人工确认 tag（无 GT 时，在 FiftyOne App 里对正确的框打此 tag）
MANUAL_CONFIRM_TAG = "swd_confirmed"


def extract_crops(ds: fo.Dataset, field: str, filter_expr,
                  out_dir: Path, label: str) -> int:
    """
    裁取满足 filter_expr 条件的检测框 patch，保存到 out_dir。
    返回裁出的 patch 数量。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    view = ds.filter_labels(field, filter_expr)

    for sample in view.iter_samples(progress=True):
        dets_obj = sample.get_field(field)
        if not dets_obj or not dets_obj.detections:
            continue

        try:
            img = Image.open(sample.filepath).convert("RGB")
        except Exception as e:
            logger.warning(f"读取失败 {sample.filepath}: {e}")
            continue

        stem = Path(sample.filepath).stem
        for i, det in enumerate(dets_obj.detections):
            # 检查是否满足过滤条件（filter_expr 已在 view 层面过滤，这里直接全用）
            patch = crop_with_padding(img, det, pad_ratio=PAD_RATIO)
            if patch is None:
                continue
            fname = f"r{ROUND}_{stem}_{i}.jpg"
            patch.save(str(out_dir / fname), quality=95)
            count += 1

    logger.info(f"  [{label}] 裁出 {count} 个 patch → {out_dir}")
    return count


def main():
    ds = fo.load_dataset(FO_DATASET)

    if not ds.has_sample_field(CLF_FIELD):
        raise RuntimeError(f"请先运行 03_run_classifier.py（{CLF_FIELD} 字段不存在）")

    has_gt     = ds.has_sample_field(GT_FIELD) and len(ds.exists(GT_FIELD)) > 0
    has_eval   = ds.has_sample_field(CLF_FIELD) and EVAL_KEY in (
        ds.get_field_schema().get(CLF_FIELD, {}) or {}
    )

    swd_out    = CROPS_DIR / "swd"     / f"round_{ROUND}"
    non_swd_out = CROPS_DIR / "non_swd" / f"fp_round_{ROUND}"

    if has_gt:
        logger.info("有 GT，从 TP/FP 裁图...")

        # TP → 正样本
        tp_count = extract_crops(
            ds, CLF_FIELD,
            filter_expr=F(EVAL_KEY) == "tp",
            out_dir=swd_out,
            label="TP (新正样本)",
        )

        # FP → 负样本
        fp_count = extract_crops(
            ds, CLF_FIELD,
            filter_expr=F(EVAL_KEY) == "fp",
            out_dir=non_swd_out,
            label="FP (新负样本)",
        )

        logger.info(f"完成：新正样本 {tp_count} 张，新负样本 {fp_count} 张")

    else:
        logger.info("无 GT，使用人工确认 tag 模式...")

        # 检查是否有人工打 tag 的框
        confirmed_view = ds.filter_labels(
            CLF_FIELD,
            F("tags").contains(MANUAL_CONFIRM_TAG),
        )
        confirmed_count = sum(
            len([d for d in s[CLF_FIELD].detections if MANUAL_CONFIRM_TAG in (d.tags or [])])
            for s in confirmed_view.iter_samples()
            if s[CLF_FIELD] and s[CLF_FIELD].detections
        )

        if confirmed_count > 0:
            logger.info(f"发现 {confirmed_count} 个已确认框，开始裁图...")
            tp_count = extract_crops(
                ds, CLF_FIELD,
                filter_expr=F("tags").contains(MANUAL_CONFIRM_TAG),
                out_dir=swd_out,
                label="手动确认正样本",
            )
            logger.info(f"完成：新正样本 {tp_count} 张")
        else:
            logger.info("=" * 60)
            logger.info("⚠️  无 GT 且尚无人工确认框。")
            logger.info("请在 FiftyOne App 中：")
            logger.info(f"  1. 打开数据集 '{FO_DATASET}'")
            logger.info(f"  2. 找到 {CLF_FIELD} 字段中确认是 SWD 的框")
            logger.info(f"  3. 对该框右键 → Add tag → 输入 '{MANUAL_CONFIRM_TAG}'")
            logger.info(f"  4. 重新运行本脚本")
            logger.info("=" * 60)

            # 把所有保留框输出为候选正样本（供参考）
            candidate_dir = CROPS_DIR / "swd" / f"round_{ROUND}_candidates"
            all_count = extract_crops(
                ds, CLF_FIELD,
                filter_expr=F("confidence").exists(),   # 所有框
                out_dir=candidate_dir,
                label="候选正样本（未确认）",
            )
            logger.info(f"候选正样本已输出到: {candidate_dir}（共 {all_count} 张，请人工筛选）")


if __name__ == "__main__":
    main()
