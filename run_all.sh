#!/bin/bash
# 一键运行完整流程
set -e
echo "1. 生成数据..."
python generate_offline_data.py
echo "2. 训练 BC..."
python train_offline.py
echo "3. 训练 CQL (可选)..."
python train_cql.py 2>/dev/null || echo "   (跳过: 需 pip install d3rlpy)"
echo "3b. 训练 IQL/BCQ baseline (可选)..."
python train_iql.py 2>/dev/null || echo "   (跳过)"
echo "4. 验证..."
python verify_reproduction.py
echo "5. 论文实验 (可选)..."
python -m experiments.run_experiments 2>/dev/null || echo "   (跳过)"
echo "完成."
