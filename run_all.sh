#!/bin/bash
# 一键运行完整流程
# 使用 v2 流程：reward_v2 + 改进行为策略 + 自实现 CQL
set -e
echo "1. 生成数据 (v2: reward_v2 + balanced/aggressive)..."
python generate_offline_data_v2.py
echo "2. 训练 BC (v2: ImprovedPolicyNet)..."
python train_offline_v2.py
echo "3. 训练 CQL (native, 无 d3rlpy 依赖)..."
python train_cql_native.py
echo "3b. 训练 IQL/BCQ (可选, 需 d3rlpy)..."
python train_iql.py 2>/dev/null || echo "   (跳过)"
echo "4. 验证..."
python verify_reproduction.py
echo "5. 论文实验 (可选)..."
python -m experiments.run_experiments 2>/dev/null || echo "   (跳过)"
echo "完成."
