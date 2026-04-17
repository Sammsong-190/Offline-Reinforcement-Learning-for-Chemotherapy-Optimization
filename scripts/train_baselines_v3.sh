#!/usr/bin/env bash
# 在同一离线数据集上训练 CQL / BC（离散动作；不写 IQL：d3rlpy 标准 IQL 仅支持连续动作空间）
# 用法: bash scripts/train_baselines_v3.sh [offline_dataset_v3.npz]
set -e
cd "$(dirname "$0")/.."
DATA="${1:-offline_dataset_v3.npz}"
SEED="${SEED:-42}"

if [ ! -f "$DATA" ]; then
  echo "找不到数据: $DATA — 请先生成，例如:"
  echo "  python scripts/generate_data.py -o $DATA --cohorts"
  exit 1
fi

echo "=== CQL (无安全约束) ==="
python scripts/train.py --algo cql --data "$DATA" --seed "$SEED"

echo "=== BC ==="
python scripts/train.py --algo bc --data "$DATA" --seed "$SEED"

echo "完成。权重默认: cql_model.d3, bc_policy.pt（项目根目录）"
