#!/bin/bash
# 兼容旧入口：单 cost_limit 多 seed（见 run_multi_seed_constraint_sweep.sh 做全 ε×seed 网格）
set -e
cd "$(dirname "$0")/.."

COST_LIMIT="${COST_LIMIT:-0.1}"
SEEDS="${SEEDS:-42 123 456 789 1024}"
DATA="${DATA:-data/raw/offline_dataset_v3.npz}"

mkdir -p checkpoints
for seed in $SEEDS; do
  echo "=== cost_limit=$COST_LIMIT seed=$seed ==="
  python scripts/train.py --algo safe_cql --data "$DATA" --cost-limit "$COST_LIMIT" --seed "$seed" --log-lambda 1000
done

echo "聚合: python scripts/aggregate_eval_multi_seed.py --limits $COST_LIMIT -o results/multi_seed_runs.csv -a results/multi_seed_agg.csv"
echo "作图: python scripts/plot_epsilon_shaded.py -i results/multi_seed_agg.csv"
