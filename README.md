# Supervised Optimal Chemotherapy Regimen (Offline RL)

基于 ODE 仿真与离线强化学习复现化疗剂量调度策略。

**论文**：Supervised Optimal Chemotherapy Regimen Based on Offline Reinforcement Learning (IEEE 9798842)

---

## 快速开始

```bash
pip install -r requirements.txt
python generate_offline_data.py
python train_offline.py
python train_cql.py
python verify_reproduction.py
```

或一键运行：`bash run_all.sh`

---

## 项目结构

```
├── env/                      # 环境
│   ├── chemo_env.py          # ODE、奖励、状态归一化
│   └── patient.py            # 病人参数随机化
├── data/
│   └── generate.py           # 数据生成（专家/混合策略）
├── generate_offline_data.py  # 数据生成入口
├── train_offline.py          # BC 训练
├── train_cql.py              # CQL 训练（可选）
├── verify_reproduction.py    # 策略对比验证
├── export_dataset.py         # 导出 npz → csv
├── ode_model.py              # 兼容层
├── run_all.sh                # 一键运行
├── requirements.txt
└── docs/
    └── PROJECT_GUIDE.md      # 详细说明
```

---

## 流程概览

| 步骤 | 脚本 | 说明 |
|------|------|------|
| 1 | `generate_offline_data.py` | ODE + 混合策略 → 离线数据 |
| 2 | `train_offline.py` | BC 模仿专家 |
| 3 | `train_cql.py` | CQL 训练（可选） |
| 4 | `verify_reproduction.py` | 对比 Expert / BC / CQL / Random / Fixed |

---

## 核心概念

- **状态**：N（正常细胞）、T（肿瘤）、I（免疫）、C（药物浓度）
- **动作**：4 档剂量 {0, 0.5, 1.0, 2.0}
- **BC**：监督学习，模仿专家
- **CQL**：保守 Q 学习，可超越行为策略

---

## 详细文档

见 [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)

---

## 依赖

`numpy` `scipy` `torch` `d3rlpy`（见 requirements.txt）
