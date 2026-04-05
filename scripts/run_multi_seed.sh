#!/bin/bash
# 命名: checkpoints/safe_cql_limit{COST_LIMIT}_seed{seed}.pt（互不覆盖）
set -e
COST_LIMIT="${COST_LIMIT:-0.1}"
SEEDS="42 123 456 789 1024"
echo "cost_limit=$COST_LIMIT  seeds: $SEEDS"
mkdir -p checkpoints
for seed in $SEEDS; do
  echo "=== Seed $seed ==="
  python scripts/train.py --algo safe_cql --cost-limit "$COST_LIMIT" --seed "$seed"
done
echo "Run: python scripts/evaluate.py --seeds $SEEDS -o results/multi_seed.csv"
