"""
07_score.py
汇总所有轮次的评估结果，输出对比表格，标注最优轮次。
读取 RESULTS_DIR/eval_r*.json，保存 score_report.csv，终端打印排名。
"""

from __future__ import annotations
import csv
import json
import logging
from pathlib import Path

from pipeline_config import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_all_evals() -> list[dict]:
    evals = []
    for f in sorted(RESULTS_DIR.glob("eval_r*.json")):
        with open(f) as fh:
            evals.append(json.load(fh))
    return evals


def print_table(evals: list[dict]):
    if not evals:
        logger.warning("没有找到任何评估结果。请先运行 04_evaluate.py。")
        return

    mode = evals[0].get("mode", "with_gt")

    print("\n" + "=" * 80)
    print(f"  SWD Pipeline 评分报告  （共 {len(evals)} 轮）")
    print("=" * 80)

    if mode == "with_gt":
        header = f"{'轮次':>5}  {'TP':>6}  {'FP':>6}  {'FN':>6}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'mAP@0.5':>9}"
        print(header)
        print("-" * 80)

        best_round = max(evals, key=lambda r: r.get("f1", 0))

        for r in evals:
            marker = " ← 最优" if r["round"] == best_round["round"] else ""
            print(
                f"  R{r['round']:>2}  "
                f"{r.get('tp','-'):>6}  {r.get('fp','-'):>6}  {r.get('fn','-'):>6}  "
                f"{r.get('precision',0):>10.4f}  {r.get('recall',0):>8.4f}  "
                f"{r.get('f1',0):>8.4f}  {r.get('mAP50',0):>9.4f}"
                f"{marker}"
            )

        print("=" * 80)
        print(f"\n🏆 最优轮次: Round {best_round['round']}")
        print(f"   Precision={best_round.get('precision',0):.4f}  "
              f"Recall={best_round.get('recall',0):.4f}  "
              f"F1={best_round.get('f1',0):.4f}  "
              f"mAP@0.5={best_round.get('mAP50',0):.4f}")

        # 进步幅度
        if len(evals) >= 2:
            first, last = evals[0], evals[-1]
            delta_recall = last.get("recall", 0) - first.get("recall", 0)
            delta_f1     = last.get("f1", 0)     - first.get("f1", 0)
            sign_r = "+" if delta_recall >= 0 else ""
            sign_f = "+" if delta_f1     >= 0 else ""
            print(f"\n📈 R1 → R{last['round']} 进步:")
            print(f"   Recall: {sign_r}{delta_recall:+.4f}   F1: {sign_f}{delta_f1:+.4f}")

    else:  # no_gt 模式
        header = f"{'轮次':>5}  {'候选框数':>10}  {'含框图片':>10}  {'score_mean':>12}  {'score_p50':>10}  {'score_p90':>10}"
        print(header)
        print("-" * 80)
        for r in evals:
            print(
                f"  R{r['round']:>2}  "
                f"{r.get('total_boxes','-'):>10}  {r.get('images_w_boxes','-'):>10}  "
                f"{r.get('score_mean','N/A'):>12}  {r.get('score_p50','N/A'):>10}  {r.get('score_p90','N/A'):>10}"
            )
        print("=" * 80)

    print()


def save_csv(evals: list[dict]):
    if not evals:
        return
    out_path = RESULTS_DIR / "score_report.csv"
    fieldnames = list(evals[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evals)
    logger.info(f"评分报告已保存: {out_path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    evals = load_all_evals()
    logger.info(f"找到 {len(evals)} 轮评估结果")

    print_table(evals)
    save_csv(evals)

    if not evals:
        logger.info("提示：请先运行 01~04 步骤完成至少一轮评估。")


if __name__ == "__main__":
    main()
