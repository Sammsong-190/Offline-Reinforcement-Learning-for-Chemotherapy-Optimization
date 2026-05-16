#!/bin/bash
# 通用 23 seeds × cost limits；可续跑时用 SKIP_EXISTING=1 跳过已有 checkpoint。
#
# 用法: bash scripts/run_multi_seed_remaining.sh [data.npz]
# 若某 checkpoint 已存在可跳过: SKIP_EXISTING=1 bash scripts/run_multi_seed_remaining.sh
set -e
cd "$(dirname "$0")/.."

DATA="${1:-data/raw/offline_dataset_v3.npz}"
SKIP="${SKIP_EXISTING:-0}"

if [ ! -f "$DATA" ]; then
  echo "缺少数据: $DATA"
  exit 1
fi

mkdir -p checkpoints results

run_one() {
  local LIMIT="$1" SEED="$2"
  local out="checkpoints/safe_cql_limit${LIMIT}_seed${SEED}.pt"
  if [ "$SKIP" = "1" ] && [ -f "$out" ]; then
    echo "=== skip (exists): $out ==="
    return 0
  fi
  echo "=== train ε=$LIMIT seed=$SEED ==="
  python scripts/train.py --algo safe_cql --data "$DATA" --cost-limit "$LIMIT" \
    --seed "$SEED" --log-lambda 1000
}

SEEDS="${SEEDS:-15 500 1200 1800 2500 3200 3900 4600 5300 6000 6700 7400 8100 8800 9500 10200 10900 11600 12300 13000 13700 14400 15000}"
LIMITS="${LIMITS:-0.0 0.1 0.3 0.5 1.0}"

for SEED in $SEEDS; do
  for LIMIT in $LIMITS; do
    run_one "$LIMIT" "$SEED"
  done
done

echo ""
echo "=== 全部训完后执行 ==="
echo "python scripts/aggregate_eval_multi_seed.py -o results/multi_seed_runs.csv -a results/multi_seed_agg.csv"
echo "python scripts/plot_epsilon_shaded.py -i results/multi_seed_agg.csv -o figures/epsilon_shaded_return.png"
echo "Done training."
