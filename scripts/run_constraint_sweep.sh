#!/bin/bash
# 实验二：约束扫描 (Cost limit / ε)，用固定离线数据训练多条 SafeCQL，再评估并画帕累托前沿。
# 用法: bash scripts/run_constraint_sweep.sh [data.npz]
set -e

cd "$(dirname "$0")/.."

DATA="${1:-data/raw/offline_dataset_v3.npz}"
SEED="${SEED:-42}"

if [ ! -f "$DATA" ]; then
  echo "数据不存在: $DATA — 请先放置 offline_dataset_v3.npz 或传入路径。"
  exit 1
fi

mkdir -p checkpoints results

echo "=== SafeCQL 约束扫描 seed=$SEED data=$DATA ==="
for LIMIT in 0.0 0.1 0.3 0.5 1.0; do
  echo "--- cost_limit=$LIMIT ---"
  python scripts/train.py --algo safe_cql --data "$DATA" --cost-limit "$LIMIT" \
    --seed "$SEED" --log-lambda 1000
done

echo ""
echo "=== 评估 → results/sensitivity.csv (可加 --with-cql --with-bc) ==="
python scripts/evaluate_sensitivity.py -o results/sensitivity.csv --with-cql --with-bc

echo ""
echo "=== 帕累托图 ==="
python scripts/plot_pareto.py -i results/sensitivity.csv -o figures/pareto_front.png
