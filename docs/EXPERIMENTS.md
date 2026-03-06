# 论文级实验设计

## 实验 1：Main Performance

| 指标 | 含义 |
|------|------|
| Return | 总治疗效果 |
| FinalTumor | 最终肿瘤大小 |
| AvgDose | 平均给药 |
| MaxTox | 最大毒性 |
| TumorClear | 治愈率 |
| Survival | 生存率 |
| **TrtEff** | Treatment Efficiency: (Tumor_start − Tumor_end) / ∑u_t |
| **Tctrl** | Time to Tumor Control: 肿瘤首次降至阈值所需时间 |

**运行**：`python -m experiments.run_experiments`

---

## 实验 2：Patient Robustness（论文亮点）

模拟不同患者参数变化：
- **tumor growth (r1)** ±20%
- **immune strength (c1)** ±20%
- **drug decay (d2)** ±20%

**输出**：return vs parameter variation 曲线图

**画图**：`python -m experiments.plot_results`

---

## 实验 3：Safety Analysis（医疗 AI 必须）

- **Toxicity constraint**：C(t) > threshold（默认 1.5）
- **Toxicity Violation Rate**：违反约束的 episode 比例

| Policy | Toxicity Violation |
|--------|-------------------|
| Expert | x% |
| BC | x% |
| CQL | x% |
| IQL | x% |

---

## Ablation Study

1. **Dataset size**：10k, 25k, 50k, 100k transitions
2. **Reward weight**：w3 = 0.3, 0.5, 0.8
3. **Behavior noise**：expert ε = 0.05, 0.15, 0.3

**运行**：`python -m experiments.ablation`

---

## 主要图表

- **图1** 系统框架：ODE → Dataset → Offline RL → Evaluation
- **图2** Tumor trajectory：T(t) 不同算法
- **图3** Dose schedule：a(t)
- **图4** Robustness：return vs parameter shift

---

## 创新点

1. **Offline RL for chemotherapy**：避免 online trial，适合医疗
2. **ODE-driven environment**：比普通 RL 更生物真实
3. **Robustness under patient heterogeneity**：策略可泛化到新患者

---

## 目标期刊

- IEEE TNNLS (~14)
- Information Fusion (~18)
- IEEE JBHI (~7)
- Pattern Recognition (~9)
