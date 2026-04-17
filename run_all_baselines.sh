#!/bin/bash
# SCI 对比实验: BC | Standard CQL | Safe CQL
# 数据: offline_dataset_v2.npz
set -e

DATA="offline_dataset_v2.npz"
if [ ! -f "$DATA" ]; then
  echo "生成数据: $DATA"
  python scripts/generate_data.py -o "$DATA" --preset safe
fi
python scripts/verify_dataset.py "$DATA"

echo ""
echo "=== 1. Behavior Cloning ==="
python scripts/train.py --algo bc --data "$DATA" --save bc_policy.pt

echo ""
echo "=== 2. Standard CQL (无视 Cost) ==="
python scripts/train.py --algo cql --data "$DATA" --save cql_model.d3

echo ""
echo "=== 3. Safe CQL (Lagrangian) ==="
python scripts/train.py --algo safe_cql --data "$DATA" --seed 42

echo ""
echo "=== 4. 评估 (多种子) ==="
python scripts/evaluate.py --seeds 42 123 456 -o results/eval_results.csv

echo ""
echo "=== 5. 画 Return vs Cost 图 ==="
python scripts/plot_return_cost.py -i results/eval_results.csv -o figures/return_vs_cost.png

echo ""
echo "Done. 查看 figures/return_vs_cost.png"
