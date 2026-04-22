#!/bin/bash
# 实验 B：加密 ε 网格（相变区细扫）
set -e
cd "$(dirname "$0")/.."

DATA="${1:-data/raw/offline_dataset_v3.npz}"
SEED="${SEED:-42}"
# 低 ε 区加密；可按需增删
LIMITS="${LIMITS:-0.0 0.01 0.05 0.1 0.2 0.3 0.5 0.8 1.0}"

if [ ! -f "$DATA" ]; then
  echo "缺少数据: $DATA"
  exit 1
fi

mkdir -p checkpoints results figures
echo "data=$DATA seed=$SEED LIMITS=$LIMITS"

for LIMIT in $LIMITS; do
  echo "=== ε=$LIMIT ==="
  python scripts/train.py --algo safe_cql --data "$DATA" --cost-limit "$LIMIT" \
    --seed "$SEED" --log-lambda 1000
done

echo ""
echo "=== 聚合（单 seed 多 ε → 仍用 aggregate 脚本，按 limit 过滤）==="
python scripts/aggregate_eval_multi_seed.py \
  --limits $LIMITS --seeds "$SEED" \
  --data-tag dense_single_seed \
  -o results/epsilon_dense_runs.csv \
  -a results/epsilon_dense_agg.csv

python scripts/plot_phase_transition.py -i results/epsilon_dense_agg.csv \
  -o figures/phase_transition_dual_y.png --log-x

echo "Done."
