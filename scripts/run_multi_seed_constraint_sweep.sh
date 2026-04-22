#!/bin/bash
# 实验 A：多 seed × ε 训练（checkpoint 互不覆盖：safe_cql_limit{ε}_seed{seed}.pt）
set -e
cd "$(dirname "$0")/.."

DATA="${1:-data/raw/offline_dataset_v3.npz}"
SEEDS="${SEEDS:-42 123 999 2024 0}"
LIMITS="${LIMITS:-0.0 0.1 0.3 0.5 1.0}"

if [ ! -f "$DATA" ]; then
  echo "缺少数据: $DATA"
  exit 1
fi

mkdir -p checkpoints results
echo "data=$DATA  SEEDS=$SEEDS  LIMITS=$LIMITS"

for SEED in $SEEDS; do
  for LIMIT in $LIMITS; do
    echo "=== train ε=$LIMIT seed=$SEED ==="
    python scripts/train.py --algo safe_cql --data "$DATA" --cost-limit "$LIMIT" \
      --seed "$SEED" --log-lambda 1000
  done
done

echo ""
echo "=== 聚合评估（跨 seed 均值/方差）==="
python scripts/aggregate_eval_multi_seed.py -o results/multi_seed_runs.csv -a results/multi_seed_agg.csv

echo "=== 作图 ==="
python scripts/plot_epsilon_shaded.py -i results/multi_seed_agg.csv -o figures/epsilon_shaded_return.png

echo "Done."
