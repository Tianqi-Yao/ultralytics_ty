"""
02_run_detection.py
对 FiftyOne 数据集中的图片运行 YOLO/SAHI 推理，结果写入 PRED_FIELD。

支持断点续跑：已有 PRED_FIELD 的 sample 自动跳过。
"""

from __future__ import annotations
import logging
import fiftyone as fo
import wandb
from pipeline_config import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def ensure_field(ds: fo.Dataset, field: str):
    if not ds.has_sample_field(field):
        ds.add_sample_field(
            field,
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )


def main():
    if not DET_MODEL_PATH.exists():
        raise FileNotFoundError(f"检测模型不存在: {DET_MODEL_PATH}")

    # 延迟导入 SAHI（避免未安装时报错）
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    logger.info(f"加载检测模型: {DET_MODEL_PATH}")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(DET_MODEL_PATH),
        confidence_threshold=CONF_THRESH,
        device=DEVICE,
    )

    logger.info(f"加载数据集: {FO_DATASET}")
    ds = fo.load_dataset(FO_DATASET)
    ensure_field(ds, PRED_FIELD)

    # 只处理尚无该字段的 sample（断点续跑）
    view = ds.match(fo.ViewField(PRED_FIELD).is_none())
    total   = len(ds)
    pending = len(view)
    logger.info(f"总 sample: {total}，待推理: {pending}（已完成: {total - pending}）")

    if pending == 0:
        logger.info(f"{PRED_FIELD} 已全部完成，跳过推理。")
        return

    # wandb
    wandb.login(key=WANDB_KEY)
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"detection_r{ROUND}",
        config={
            "round":        ROUND,
            "pred_field":   PRED_FIELD,
            "model":        str(DET_MODEL_PATH),
            "conf_thresh":  CONF_THRESH,
            "sahi_slice":   SAHI_SLICE_SIZE,
            "sahi_overlap": SAHI_OVERLAP,
        },
    )

    pred_total = 0

    for i, sample in enumerate(view.iter_samples(progress=True, autosave=True)):
        result = get_sliced_prediction(
            sample.filepath,
            detection_model,
            slice_height=SAHI_SLICE_SIZE,
            slice_width=SAHI_SLICE_SIZE,
            overlap_height_ratio=SAHI_OVERLAP,
            overlap_width_ratio=SAHI_OVERLAP,
            verbose=0,
        )
        fo_dets = result.to_fiftyone_detections()
        sample[PRED_FIELD] = fo.Detections(detections=fo_dets)
        pred_total += len(fo_dets)

        if (i + 1) % 50 == 0:
            run.log({"samples_done": i + 1, "total_preds": pred_total})

    avg_per_img = pred_total / max(1, pending)
    logger.info(f"推理完成：{pending} 张图，共 {pred_total} 个框，平均 {avg_per_img:.1f} 框/图")

    run.summary.update({"total_preds": pred_total, "avg_per_img": avg_per_img})
    run.finish()


if __name__ == "__main__":
    main()
