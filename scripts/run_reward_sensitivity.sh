#!/bin/bash
# 实验一：奖励敏感性 — high_incentive 数据 → CQL vs SafeCQL；评估时沿用 CHEMO_REWARD_PROFILE
set -e
cd "$(dirname "$0")/.."

SEED="${SEED:-42}"
COST_LIMIT="${COST_LIMIT:-0.1}"
DATA_HIGH="data/raw/offline_dataset_high_reward.npz"
mkdir -p data/raw checkpoints results figures

echo "=== 1. 生成高诱惑离线数据 ==="
python scripts/generate_data.py -o "$DATA_HIGH" --reward-profile high_incentive --preset safe

echo ""
echo "=== 2. 训练 CQL → checkpoints/cql_high_reward.d3 ==="
python scripts/train.py --algo cql --data "$DATA_HIGH" --seed "$SEED" \
  --save "$(pwd)/checkpoints/cql_high_reward.d3"

echo ""
echo "=== 3. 训练 SafeCQL (--cost-limit $COST_LIMIT) ==="
python scripts/train.py --algo safe_cql --data "$DATA_HIGH" --seed "$SEED" \
  --cost-limit "$COST_LIMIT" --log-lambda 1000

SAFE_PT="checkpoints/safe_cql_limit${COST_LIMIT}_seed${SEED}.pt"
cp -f "$SAFE_PT" checkpoints/safecql_high_reward.pt

echo ""
echo "=== 4. 评估 (high_incentive profile) ==="
python scripts/evaluate.py --reward-profile high_incentive \
  --ckpt CQL_HR=checkpoints/cql_high_reward.d3 \
  --ckpt SafeCQL_HR=checkpoints/safecql_high_reward.pt \
  --policies CQL_HR SafeCQL_HR \
  -o results/reward_high_incentive.csv

echo ""
echo "=== 5. 柱状图 (avg_dose) ==="
python scripts/plot_reward_bar.py -i results/reward_high_incentive.csv \
  -o figures/reward_sensitivity_dose.png --metric avg_dose \
  --title "High-incentive reward: dose"

echo "Done. CSV: results/reward_high_incentive.csv  图: figures/reward_sensitivity_dose.png"
