#!/bin/bash
# SCI 完整流程: 数据(v3) -> 训练 SafeCQL 五档 ε -> 评估 -> Figure A & B
set -e
cd "$(dirname "$0")"

DATA="${DATA:-data/raw/offline_dataset_v3.npz}"
SEED="${SEED:-42}"

echo "=== Step 0: 数据 ==="
if [ ! -f "$DATA" ]; then
  echo "请提供 $DATA（或设置 DATA=...）；亦可先用 generate_data.py 生成到该路径"
  exit 1
fi
if [ -f "scripts/verify_dataset.py" ]; then
  python scripts/verify_dataset.py "$DATA"
fi

echo ""
echo "=== Step 1: 训练 SafeCQL ε ∈ {0.0,0.1,0.3,0.5,1.0}, seed=$SEED ==="
mkdir -p checkpoints
for limit in 0.0 0.1 0.3 0.5 1.0; do
  echo "--- cost_limit=$limit ---"
  python scripts/train.py --algo safe_cql --data "$DATA" --cost-limit "$limit" \
    --seed "$SEED" --log-lambda 1000
done

echo ""
echo "=== Step 2: 评估 sensitivity.csv ==="
python scripts/evaluate_sensitivity.py -o results/sensitivity.csv --with-cql --with-bc

echo ""
echo "=== Step 3: 绘图 ==="
python scripts/plot_lambda_dynamics.py -o figures/lambda_dynamics.png
python scripts/plot_pareto.py -i results/sensitivity.csv -o figures/pareto_front.png

echo ""
echo "Done. figures/lambda_dynamics.png (Figure A), figures/pareto_front.png (Figure B)"
