#!/bin/bash
# SCI 完整流程: 数据 -> 训练(含 lambda 记录) -> 评估 -> Figure A & B
# 一步到位生成论文核心图表
set -e

cd "$(dirname "$0")"

echo "=== Step 0: 数据 ==="
DATA="offline_dataset_v2.npz"
if [ ! -f "$DATA" ]; then
  python scripts/generate_data.py -o "$DATA" --preset safe
fi
python scripts/verify_dataset.py "$DATA"

echo ""
SEED="${SEED:-42}"
echo "=== Step 1: 训练三组 Safe CQL (含 lambda 记录), seed=$SEED ==="
mkdir -p checkpoints
for limit in 0.01 0.1 0.5; do
  echo "--- cost_limit=$limit ---"
  python scripts/train.py --algo safe_cql --data "$DATA" --cost-limit "$limit" \
    --seed "$SEED" --log-lambda 1000
done

echo ""
echo "=== Step 2: 评估 (生成 sensitivity.csv) ==="
python scripts/evaluate_sensitivity.py -o results/sensitivity.csv

echo ""
echo "=== Step 3: 绘图 ==="
python scripts/plot_lambda_dynamics.py -o results/lambda_dynamics.png
python scripts/plot_pareto.py -i results/sensitivity.csv -o results/pareto_front.png

echo ""
echo "Done. 查看 results/lambda_dynamics.png (Figure A) 和 results/pareto_front.png (Figure B)"
