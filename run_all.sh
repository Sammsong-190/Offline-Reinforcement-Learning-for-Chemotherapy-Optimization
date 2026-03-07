#!/bin/bash
# 一键运行完整流程
set -e
echo "1. 生成数据..."
python generate_offline_data.py
echo "2. 训练 BC..."
python train_offline.py
echo "3. 训练 CQL (需 d3rlpy)..."
python train_cql.py || echo "   (跳过)"
echo "4. 训练 IQL/BCQ (需 d3rlpy)..."
python train_iql.py || echo "   (跳过)"
echo "5. 验证..."
python verify_reproduction.py
echo "6. 论文实验 (可选)..."
python -m experiments.run_experiments 2>/dev/null || echo "   (跳过)"
echo "完成."
