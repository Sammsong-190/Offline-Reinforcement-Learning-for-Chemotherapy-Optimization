#!/bin/bash
# SCI 灵敏度分析: 训练三组 cost_limit + 记录 lambda + 评估 + 绘图
set -e

DATA="${1:-offline_dataset_v2.npz}"
if [ ! -f "$DATA" ]; then
  echo "生成数据: $DATA"
  python scripts/generate_data.py -o "$DATA" --preset safe
fi

mkdir -p checkpoints results

echo "=== 1. 训练三组 Safe CQL (含 lambda 记录) ==="
for limit in 0.01 0.1 0.5; do
  echo "--- cost_limit=$limit ---"
  python scripts/train.py --algo safe_cql --data "$DATA" --cost-limit "$limit" \
    --save "checkpoints/safe_cql_limit${limit}.pt" --log-lambda 1000
done

echo ""
echo "=== 2. 评估 (生成 sensitivity.csv) ==="
python scripts/evaluate_sensitivity.py -o results/sensitivity.csv

echo ""
echo "=== 3. 绘图 (Figure A & B) ==="
python scripts/plot_lambda_dynamics.py -o results/lambda_dynamics.png
python scripts/plot_pareto.py -i results/sensitivity.csv -o results/pareto_front.png

echo ""
echo "Done. 查看 results/lambda_dynamics.png 和 results/pareto_front.png"
