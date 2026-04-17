# Supervised Optimal Chemotherapy Regimen (Offline RL)

基于 **Safe Offline RL** 的个体化化疗剂量调度：在离线数据上学习策略，在显式毒性/免疫约束下优化长期回报。

**论文**：Supervised Optimal Chemotherapy Regimen Based on Offline Reinforcement Learning (IEEE 9798842)

---

## 核心亮点

- **SDE 与异质性**：环境步进支持随机微分噪声（`sde_sigma`），配合患者上下文模拟治疗不确定性。
- **多亚群虚拟临床试验**：`patient_cohorts` 定义 young_strong / elderly_frail 等队列；评估可用 `--cohort` 或 `--cohort-id` 固定亚群，输出生存步数、死因占比等指标，便于与论文图表对齐。

---

## 快速复现（推荐顺序）

在项目根目录执行：

```bash
pip install -r requirements.txt

# 1) 生成离线数据集（含多亚群 + SDE 风格步进时用 --cohorts）
python scripts/generate_data.py -o data/raw/offline_dataset.npz --cohorts

# 2) 训练 Safe CQL（默认读取 configs 中的 data.path）
python scripts/train.py --algo safe_cql --seed 42

# 3) 多策略、多种子评估，写入 CSV（虚拟试验：按队列）
python scripts/evaluate.py --seeds 42 123 456 --cohort -o results/eval_cohort.csv

# 4) 从 results/*.csv 汇总出图，并 rollout KM / 剂量箱线 / 典型轨迹 → figures/
python scripts/plot_results.py --results-dir results --fig-dir figures
```

仅画 CSV 汇总柱图、不加载策略做 rollout 时：

```bash
python scripts/plot_results.py --no-rollout
```

**说明**：Kaplan-Meier 与轨迹图依赖环境与 checkpoint；请保证已训练 Safe CQL，且默认会查找 `checkpoints/safe_cql_limit0.1_seed42.pt`（可用 `scripts/plot_results.py` 的 `--safe-cql-ckpt` 覆盖）。

---

## 其他入口

**同一数据集 `offline_dataset_v3.npz` 上：基线（CQL + BC）与 Safe CQL**

```bash
# 若尚无数据：生成（与论文一致时用 --cohorts）
python scripts/generate_data.py -o offline_dataset_v3.npz --cohorts

# 无约束 CQL、模仿 BC
python scripts/train.py --algo cql --data offline_dataset_v3.npz --seed 42
python scripts/train.py --algo bc   --data offline_dataset_v3.npz --seed 42
# 或: bash scripts/train_baselines_v3.sh offline_dataset_v3.npz

# Safe CQL（可显式指向同一数据）
python scripts/train.py --algo safe_cql --data offline_dataset_v3.npz --seed 42

# 亚群虚拟试验：主方法 + 基线 + Expert（多种子）
python scripts/evaluate.py --cohort-id young_strong elderly_frail \
  --policies SafeCQL CQL BC Expert \
  --n_ep 20 --seeds 42 123 456 \
  -o results/vtrial_all_algos.csv
```

其它：

```bash
python scripts/generate_data.py -o data/raw/offline_dataset.npz   # 不含 --cohorts 时为均匀参数噪声
python scripts/train.py --algo safe_cql --cost-limit 0.1
```

或一键流水线：`bash run_all.sh` / `bash run_all_sci.sh`（若仓库中提供）。

### 论文补充（算法与主表，无需重训）

```bash
# Fig：SafeCQL vs CQL 同患者轨迹对比（左成功 / 右高剂量失败）
python scripts/plot_trajectory_contrast.py -o figures/trajectory_contrast_young_strong.png

# Fig：λ、Cost 风险、Q_R 损失（需训练时加 --log-lambda 1000 生成 *_lambda.json）
python scripts/plot_training_dynamics.py --lambda-json checkpoints/safe_cql_limit0.1_seed42_lambda.json \
  -o figures/training_dynamics_safecql.png

# Fig（备选）：Return–违规率 帕累托散点
python scripts/plot_pareto.py -i results/vtrial_all_algos.csv -o figures/pareto_front.png

# Table 1（队列参数）+ Table 2（evaluate 导出的 CSV → 宽表/长表 + LaTeX 片段）
python scripts/export_paper_tables.py --eval-csv results/vtrial_all_algos.csv --out-dir tables
```

---

## 项目结构（节选）

```
├── configs/                  # YAML：环境、Safe CQL、实验
├── data/
│   ├── generate.py           # 离线数据生成
│   └── ...
├── env/                      # ODE/SDE 环境、patient_cohorts
├── src/                      # 评估、Safe CQL 等
├── scripts/
│   ├── generate_data.py
│   ├── train.py              # --algo bc|cql|safe_cql
│   ├── evaluate.py
│   ├── plot_results.py       # CSV 汇总 + 科研图 → figures/
│   ├── plot_trajectory_contrast.py
│   ├── plot_training_dynamics.py
│   ├── export_paper_tables.py
│   └── plot_pareto.py
├── results/                  # 评估 CSV、JSON 等数值结果（*.csv 默认 gitignore）
└── figures/                  # 所有论文图（*.png 等；与 results/ 分离，默认 gitignore）
```

---

## 流程概览

| 步骤 | 脚本 | 说明 |
|------|------|------|
| 1 | `scripts/generate_data.py` | ODE（+ 可选 cohort SDE）→ 离线 npz |
| 2 | `scripts/train.py --algo safe_cql` | Safe CQL（Lagrangian 代价约束） |
| 3 | `scripts/evaluate.py` | 多策略、多种子、可选 `--cohort` / `--cohort-id` |
| 4 | `scripts/plot_results.py` | 读 `results/*.csv` + rollout → `figures/` |

---

## 核心概念

- **状态**：N（正常细胞）、T（肿瘤）、I（免疫）、C（药物浓度）
- **动作**：离散剂量档（见环境配置）
- **Safe CQL**：CMDP 约束，控制毒性相关代价（见 [docs/SAFE_RL.md](docs/SAFE_RL.md)）

---

## 详细文档

- [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) 项目详解
- [docs/SAFE_RL.md](docs/SAFE_RL.md) Safe CQL 约束说明

---

## 依赖

`numpy` `scipy` `torch` `d3rlpy` `matplotlib` `lifelines`（见 `requirements.txt`）
