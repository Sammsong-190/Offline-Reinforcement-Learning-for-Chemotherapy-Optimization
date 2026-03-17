#!/bin/bash
# 多种子实验 - SCI 要求
# 单一种子的 RL 结果在 SCI 一区不被认可
set -e
SEEDS="42 123 456 789 1024"
echo "Training with seeds: $SEEDS"
for seed in $SEEDS; do
  echo "=== Seed $seed ==="
  python scripts/train.py --algo safe_cql --save "checkpoints/safe_cql_seed${seed}.pt"
  # 或: python train_cql.py --seed $seed --save cql_seed${seed}.d3
done
echo "Run: python scripts/evaluate.py --seeds $SEEDS -o results/multi_seed.csv"
