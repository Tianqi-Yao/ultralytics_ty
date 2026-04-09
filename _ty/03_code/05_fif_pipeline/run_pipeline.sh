#!/bin/bash
# run_pipeline.sh
# 完整迭代 pipeline 一键运行脚本。
#
# 用法：
#   首次运行（Round 1）：
#     bash run_pipeline.sh
#
#   后台挂起（断线不中断）：
#     nohup bash run_pipeline.sh > pipeline_r1.log 2>&1 & echo "PID: $!"
#
#   查看进度：
#     tail -f pipeline_r1.log
#
#   第二轮迭代：
#     1. 修改 pipeline_config.py 中 ROUND = 2
#     2. bash run_pipeline.sh
#
# 注意：
#   - 步骤 05 在无 GT 时会提示在 FiftyOne App 中打 tag，需人工操作后重跑 05
#   - 每步失败会立即停止（set -e），修复后从失败步骤重跑即可

set -e   # 任一步骤失败立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================"
echo " SWD FiftyOne Iterative Pipeline"
echo " 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

echo ""
echo "===== [1/7] 创建 / 更新 FiftyOne 数据集 ====="
python 01_create_dataset.py

echo ""
echo "===== [2/7] YOLO/SAHI 推理 ====="
python 02_run_detection.py

echo ""
echo "===== [3/7] 分类器二次过滤 ====="
python 03_run_classifier.py

echo ""
echo "===== [4/7] 评估 TP/FP/FN ====="
python 04_evaluate.py

echo ""
echo "===== [5/7] 裁取新 crops ====="
python 05_extract_crops.py

echo ""
echo "===== [6/7] 重训分类器 ====="
python 06_retrain_classifier.py

echo ""
echo "===== [7/7] 生成评分报告 ====="
python 07_score.py

echo ""
echo "======================================================"
echo " Pipeline 完成！"
echo " 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo " 下一步（开始新一轮迭代）："
echo "   1. 修改 pipeline_config.py → ROUND += 1"
echo "   2. 重新运行: bash run_pipeline.sh"
echo "======================================================"
