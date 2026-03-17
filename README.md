# Supervised Optimal Chemotherapy Regimen (Offline RL)

基于 ODE 仿真与离线强化学习复现化疗剂量调度策略。

**论文**：Supervised Optimal Chemotherapy Regimen Based on Offline Reinforcement Learning (IEEE 9798842)

---

## 快速开始

```bash
pip install -r requirements.txt
python scripts/generate_data.py -o offline_dataset.npz
python train_offline.py
python train_cql.py --alpha 5 --lr 1e-4 --seed 42   # d3rlpy 强基线
python train_iql.py   # IQL/BCQ 基线，需 d3rlpy
python scripts/train.py --algo safe_cql
python verify_reproduction.py
```

或一键运行：`bash run_all.sh`

**新结构 (configs + scripts)**：
```bash
python scripts/generate_data.py -o data/raw/offline_dataset.npz
python scripts/train.py --algo safe_cql
python scripts/evaluate.py --seeds 42 123 456 -o results/eval.csv
```

---

## 项目结构

```
├── configs/                  # 配置文件 (YAML)
│   ├── env/chemo_std.yaml    # ODE、奖励参数
│   ├── agent/                # CQL、Safe CQL、BC 超参数
│   └── experiment/           # 训练入口配置
├── data/
│   ├── raw/                  # 原始 npz
│   ├── processed/            # 预处理数据
│   ├── generate.py           # 数据生成
│   └── buffer.py             # ReplayBuffer 加载器
├── src/                      # 核心源码
│   ├── envs/                 # 环境、wrappers
│   ├── models/               # Actor、Critic、SafetyCritic
│   ├── algos/                # Safe CQL 等算法
│   └── utils/                # logger、evaluation
├── scripts/                  # 统一入口
│   ├── generate_data.py
│   ├── train.py              # --algo bc|cql|safe_cql
│   └── evaluate.py
├── notebooks/                # 轨迹分析、Pareto 图
├── env/                      # 环境 (legacy)
├── train_*.py                # 训练脚本 (legacy)
├── verify_reproduction.py    # 策略验证
└── run_all.sh                # 一键运行
```

---

## 流程概览

| 步骤 | 脚本 | 说明 |
|------|------|------|
| 1 | `scripts/generate_data.py` | ODE + 混合策略 → 离线数据 |
| 2 | `train_offline.py` | BC 模仿专家 |
| 3 | `train_cql.py` | CQL 训练（需 d3rlpy） |
| 4 | `train_iql.py` | IQL/BCQ 基线（需 d3rlpy） |
| 5 | `scripts/train.py --algo safe_cql` | Safe CQL（Lagrangian 约束） |
| 6 | `verify_reproduction.py` | 对比 Expert / BC / CQL / Safe CQL / IQL / Random / Fixed |

---

## 核心概念

- **状态**：N（正常细胞）、T（肿瘤）、I（免疫）、C（药物浓度）
- **动作**：4 档剂量 {0, 0.5, 1.0, 2.0}
- **BC**：监督学习，模仿专家
- **CQL**：保守 Q 学习，可超越行为策略
- **Safe CQL**：CMDP 约束，保证 I≥0.3、N≥0.4（见 [docs/SAFE_RL.md](docs/SAFE_RL.md)）

---

## 详细文档

- [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) 项目详解
- [docs/SAFE_RL.md](docs/SAFE_RL.md) Safe CQL 显式安全约束

---

## 依赖

`numpy` `scipy` `torch` `d3rlpy`（见 requirements.txt）
