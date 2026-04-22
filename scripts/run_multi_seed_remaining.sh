#!/bin/bash
# 续跑实验 A：根据上次进度，只训尚未完成的 (seed × ε)。
# 当前逻辑（与终端日志一致）：
#   - seed 42：五档已齐
#   - seed 123：0.3 中断需重训，补 0.5、1.0
#   - seed 999, 2024, 0：各跑全套五档
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

# seed 123：0.3 被 Ctrl+C 打断未保存，需重跑；再 0.5、1.0
for LIMIT in 0.3 0.5 1.0; do
  run_one "$LIMIT" 123
done

# 其余 seed 全套
for SEED in 999 2024 0; do
  for LIMIT in 0.0 0.1 0.3 0.5 1.0; do
    run_one "$LIMIT" "$SEED"
  done
done

echo ""
echo "=== 全部训完后执行 ==="
echo "python scripts/aggregate_eval_multi_seed.py -o results/multi_seed_runs.csv -a results/multi_seed_agg.csv"
echo "python scripts/plot_epsilon_shaded.py -i results/multi_seed_agg.csv -o figures/epsilon_shaded_return.png"
echo "Done training."
