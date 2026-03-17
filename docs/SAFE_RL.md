# Safe CQL: 显式安全约束 (CMDP)

## 理论

将标准 MDP 升级为 **受限马尔可夫决策过程 (CMDP)**：

$$
\max_\pi \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t r(s_t, a_t)\right]
\quad \text{s.t.} \quad
\mathbb{E}\left[\sum_{t=0}^\infty \gamma^t c(s_t, a_t)\right] \le \varepsilon
$$

- **Reward $r$**：治疗效果（肿瘤下降）
- **Cost $c$**：违规成本（免疫/器官衰竭）
- **$\varepsilon$**：风险预算 (cost_limit)

## 成本定义

```python
c(s, a) = 1  if I < 0.3 (免疫崩溃) or N < 0.4 (器官衰竭)
        = 0  otherwise
```

阈值在 `env/chemo_env.py` 中：`I_SAFE=0.3`, `N_SAFE=0.4`

## Lagrangian 实现

1. **Reward Q ($Q_R$)**：CQL 保守 Q 学习
2. **Cost Q ($Q_C$)**：MSE 拟合未来累积成本
3. **Lagrange 乘子 $\lambda$**：约束违反时增大
4. **Actor**：最大化 $Q_R - \lambda \cdot Q_C$

## 运行

```bash
# 需先生成含 cost 的数据（自动检测并重新生成）
python train_safe_cql.py

# 或完整流程
bash run_all.sh
```

## 实验设计（论文）

| Baseline | 说明 |
|----------|------|
| Unconstrained CQL | 治疗效果好，可能免疫归零 |
| BC | 模仿专家，OOD 时安全性未知 |
| **Safe CQL** | 治疗稍慢，但 I 始终 ≥ 0.3 |

**可视化**：轨迹图展示 Safe CQL 在 I 接近 0.3 时主动降剂量、等待免疫恢复。
