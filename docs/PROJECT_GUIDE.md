# 项目逐步详解：公式与概念

---

## 零、MDP 形式化

### 定义

马尔可夫决策过程 \(\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)\)：

| 要素 | 定义 |
|------|------|
| **State** \(\mathcal{S}\) | \(s_t = (N_t, T_t, I_t, C_t)\)，归一化后 \(s_t^{norm} = (\tilde{N}, \tilde{T}, \tilde{I}, \tilde{C})\) |
| **Action** \(\mathcal{A}\) | 离散，\(\mathcal{A} = \{0, 0.5, 1.0, 2.0\}\) |
| **Transition** \(P\) | \(s_{t+1} = f(s_t, a_t)\)，由 ODE Euler 积分决定 |
| **Reward** \(R\) | \(r_t = R(s_t, a_t, s_{t+1})\)，见第三节 |
| **Discount** \(\gamma\) | 0.99 |

### 离线数据集

$$
\mathcal{D} = \{(s_i, a_i, r_i, s_i')\}_{i=1}^{N}
$$

- 生成：`num_trajectories` 条轨迹，每条 `horizon` 步
- 默认：100 轨迹 × ~100 步 ≈ 1e4 transitions

### Pipeline

```
     ODE Simulator
           │
           ▼
   Behavior Policy (expert + conservative + random)
           │
           ▼
   Offline Dataset D
           │
     ┌─────┴─────┐
     ▼           ▼
    BC         CQL
     │           │
     └─────┬─────┘
           ▼
   Policy Evaluation (ODE rollout)
```

---

## 一、ODE 肿瘤动力学（Padmanabhan et al. 2017）

### 1.1 状态变量

| 符号 | 含义 | 初值 |
|------|------|------|
| \(N\) | 正常细胞密度 | 1.0 |
| \(T\) | 肿瘤细胞密度 | 0.7 |
| \(I\) | 免疫细胞密度 | 1.0 |
| \(C\) | 药物浓度 | 0.0 |

状态向量：\(\mathbf{x} = (N, T, I, C)\)

### 1.2 常微分方程组

$$
\begin{aligned}
\frac{dN}{dt} &= r_2 N(1 - b_2 N) - c_4 N T - a_3 N C \\
\frac{dT}{dt} &= r_1 T(1 - b_1 T) - c_2 I T - c_3 T N - a_2 T C \\
\frac{dI}{dt} &= s + \frac{\rho I T}{\alpha + T} - c_1 I T - d_1 I - a_1 I C \\
\frac{dC}{dt} &= -d_2 C + u
\end{aligned}
$$

- \(u\)：控制输入（给药速率），\(u \in \{0, 0.5, 1.0, 2.0\}\)
- \(r_1, r_2\)：肿瘤/正常细胞增长率
- \(b_1, b_2\)：Logistic 承载力
- \(c_1 \sim c_4\)：细胞间相互作用
- \(a_1, a_2, a_3\)：药物杀伤系数
- \(s, \rho, \alpha, d_1, d_2\)：免疫与药物代谢参数

### 1.3 默认参数（DEFAULT_PARAMS）

```
r1=1.5, r2=1.0, b1=1.0, b2=1.0
c1=1.0, c2=0.05, c3=0.05, c4=0.5
a1=0.35, a2=0.45, a3=0.25  # 增强毒性：高剂量→免疫崩溃
s=0.33, rho=0.3, alpha=0.3
d1=0.2, d2=0.5
```

### 1.4 数值积分（Euler）

$$
\mathbf{x}_{t+1} = \mathbf{x}_t + h \cdot \mathbf{f}(\mathbf{x}_t, u), \quad h = \frac{\Delta t}{n_{sub}}
$$

- \(\Delta t = 0.3\)，\(n_{sub} = 5\)
- 每步后裁剪：\(\mathbf{x} = \text{clip}(\mathbf{x}, 0, 50)\)

---

## 二、状态归一化

### 2.1 Log-scale（N, T, I）

$$
\tilde{N} = \frac{\log(1+N)}{X_{scale,N}}, \quad
\tilde{T} = \frac{\log(1+T)}{X_{scale,T}}, \quad
\tilde{I} = \frac{\log(1+I)}{X_{scale,I}}
$$

- \(\log(1+x)\)：\(\log1p\)，数值稳定
- \(X_{scale} = [1, 1, 1, 3]\)（N,T,I 用 1，C 用 3）

### 2.2 C 的线性缩放

$$
\tilde{C} = \frac{C}{X_{scale,C}} = \frac{C}{3}
$$

### 2.3 反归一化

$$
N = \exp(\tilde{N} \cdot X_{scale,N}) - 1
$$

（T, I 同理；C 为线性逆变换）

---

## 三、奖励函数

### 3.1 奖励函数

$$
r = \left( -2T - 0.3C - 0.5\sigma(I_{th}-I) + 0.3N \right) \Delta t + 10 \cdot \mathbb{1}_{T<0.02}
$$

| 项 | 含义 |
|----|------|
| \(-2T\) | 强烈抑制肿瘤 |
| \(-0.3C\) | 适度毒性 |
| \(+0.3N\) | 保护正常细胞 |
| \(+10\) | 肿瘤清除奖励 |

### 3.2 各项含义

| 项 | 含义 |
|----|------|
| \(-T\) | 肿瘤越小越好 |
| \(-0.8 \tanh(C)\) | 药物毒性惩罚（强） |
| \(-0.5 \sigma(I_{th}-I)\) | \(I < I_{th}\) 时惩罚 |
| \(+5 \cdot \mathbb{1}_{T<0.02}\) | 肿瘤清除奖励 |

---

## 四、轨迹终止条件

### 4.1 自然终止（done）

$$
\text{done} \Leftrightarrow (T < 0.02) \lor (N < 0.1) \lor (I < 0.1) \lor (C > 8) \lor (\max(\mathbf{x}) > 30)
$$

- \(T_{clear}=0.02\)：肿瘤清除
- \(C_{tox}=8\)：毒性上限（安全约束）
- \(\max(\mathbf{x})>30\)：ODE 爆炸保护

### 4.2 超时（timeout）

$$
\text{timeout} \Leftrightarrow (\text{step} = \text{MAX\_STEPS}-1) \land \neg\text{done}
$$

- \(\text{MAX\_STEPS} = 300\)（肿瘤动力学较慢，250 步不足）

---

## 五、动作空间

### 5.1 离散动作

$$
\mathcal{A} = \{0, 0.5, 1.0, 2.0\}
$$

- 动作索引：\(a_{idx} \in \{0, 1, 2, 3\}\)

### 5.2 离散化

$$
a = \arg\min_{a' \in \mathcal{A}} |a' - u|
$$

---

## 六、行为策略

### 6.1 专家策略（Expert，ε-greedy）

以概率 \(\varepsilon\) 随机选动作，否则（T-based 更激进）：

$$
\pi_{expert}(s) = \begin{cases}
0 & T < 0.02 \lor N < 0.2 \lor I < 0.2 \\
2.0 & T > 0.5 \\
1.0 & T > 0.3 \\
0.5 & T > 0.1 \\
0 & \text{else}
\end{cases}
$$

- 默认 \(\varepsilon = 0.2\)，更多 aggressive therapy 利于 Offline RL

### 6.2 保守策略（Conservative）

$$
\pi_{conservative}(s) = \begin{cases}
0.5 & N > 0.5 \land I > 0.5 \land T > 0.3 \\
0 & \text{else}
\end{cases}
$$

### 6.3 混合策略（Behavior）

按轨迹类型采样：60% expert、20% balanced、10% aggressive、10% conservative。

- **expert**：激进专家，追求清除
- **balanced**：按状态加权采样，改善动作分布
- **aggressive**：高剂量探索清除路径
- **conservative**：谨慎低剂量

### 6.4 Noisy Expert（可选）

\(a = \text{discretize}(\pi_{expert}(s) + \mathcal{N}(0, 0.2))\)，更接近真实医疗噪声。

---

## 七、参数随机化（模拟不同病人）

### 7.1 分组

- **PHYS**：\(r_1, r_2, b_1, b_2\)
- **DRUG**：\(a_1, a_2, a_3, c_4\)
- **IMMUNE**：\(c_1, c_2, \rho, \alpha, c_3, s, d_1, d_2\)

### 7.2 随机化公式

$$
\theta_k \leftarrow \text{clip}\left( \theta_k \cdot \mathcal{N}(1, \sigma), \; 0.5\theta_0, \; 1.5\theta_0 \right)
$$

- PHYS / IMMUNE：\(\sigma = 0.15\)（增强随机性）
- DRUG：\(\sigma = 0.075\)
- clip 防止极端值导致 ODE 爆炸

---

## 八、Behavioral Cloning（BC）

### 8.1 策略网络

$$
\pi_\theta(a|s) = \text{softmax}(\text{MLP}(s))
$$

- 输入：\(\mathbf{s}_{norm} \in \mathbb{R}^4\)
- 隐藏层：64 维 × 2，ReLU
- 输出：4 维 logits

### 8.2 损失函数（CrossEntropy）

$$
\mathcal{L}_{BC} = -\frac{1}{N} \sum_{i=1}^{N} \log \pi_\theta(a_i | s_i)
$$

- 等价于多分类交叉熵，\(a_i\) 为专家动作索引

### 8.3 训练

- 优化器：Adam，\(lr = 10^{-3}\)
- Batch size：64
- Epochs：100

---

## 九、CQL（Conservative Q-Learning）

### 9.1 核心思想

- 在数据分布上最大化 Q
- 在策略分布上最小化 Q，抑制分布外动作

### 9.2 损失（离散版，概念）

$$
\mathcal{L}_{CQL} = \alpha \mathbb{E}_{s \sim D} \left[ \log \sum_a e^{Q(s,a)} - \mathbb{E}_{a \sim D}[Q(s,a)] \right] + \mathcal{L}_{DQN}
$$

- \(\alpha = 1.0\)：保守项权重
- \(\mathcal{L}_{DQN}\)：Double DQN 的 TD 损失

### 9.3 超参数（DiscreteCQL）

- \(lr = 10^{-4}\)，\(\gamma = 0.99\)
- batch_size = 64
- target_update_interval = 8000

### 9.4 OOD 动作问题（Offline RL 核心挑战）

- **问题**：\(a \notin \mathcal{D}\) 时，Q 易过估计
- **CQL 解法**：\(\log \sum_a e^{Q(s,a)}\) 惩罚所有动作的 Q，使 OOD 动作的 Q 被压低

---

## 十、累积回报（Return）

### 10.1 定义

$$
G = \sum_{t=0}^{T-1} r_t
$$

- 单条轨迹的累积奖励
- 验证时对多条轨迹取平均：\(\bar{G}\)

### 10.2 评估

- 在 ODE 中 rollout，使用默认参数
- 每次 rollout 最多 100 步，或直到 done

### 10.3 扩展指标

| 指标 | 定义 |
|------|------|
| **Tumor clear** | \(P(T_{end} < 0.02)\) |
| **Survival** | \(P(N_{end} > 0.1 \land I_{end} > 0.1)\) |
| **Avg dose** | \(\frac{1}{T}\sum_t u_t\) |
| **Avg Tumor** | \(\frac{1}{T}\sum_t T_t\) |
| **Max Toxicity** | \(\max_t C_t\) |
| **Drug Usage** | \(\sum_t u_t\)（总给药量） |

### 10.4 训练细节

- Dataset：~2.5e4 transitions（100 轨迹 × ~250 步）
- BC：200 epochs，batch 64
- CQL：~50 epochs × (N/64) steps，上限 50k

### 10.5 鲁棒性评估（Robust Chemotherapy RL）

- **训练**：100 患者（seed 42），每轨迹随机化参数
- **测试**：100 新患者（seed 123），与训练集不重叠
- **RL baseline**：CQL（d3rlpy DiscreteCQL）与 BC；离散动作下不设 IQL（d3rlpy 标准 IQL 为连续动作空间，与本环境不兼容）

---

## 十一、符号对照表

| 符号 | 含义 |
|------|------|
| \(\mathbf{x}\) | 状态 (N,T,I,C) |
| \(u\) | 动作（剂量） |
| \(\Delta t\) | 时间步长 0.3 |
| \(\gamma\) | 折扣因子 0.99 |
| \(\varepsilon\) | ε-greedy 探索率 |
| \(\sigma\) | sigmoid |
| \(\mathcal{D}\) | 离线数据集 |

---

## 十二、FAQ

**Q：为什么用 log1p 归一化？**  
A：N,T,I 可能呈指数变化，log 压缩尺度，利于网络训练。

**Q：reward 为何用 tanh/sigmoid？**  
A：平滑、有界、可导，比阶跃函数更易学习。

**Q：BC 与 CQL 的区别？**  
A：BC 只模仿 (s,a)；CQL 利用 (s,a,r,s') 做价值估计，可超越行为策略。

**Q：复现成功标准？**  
A：BC ≈ Expert，且 CQL > BC 更佳。

**Q：loss 正常但 Return 不提升，是否过拟合？**  
A：有可能。训练 loss 下降而验证/测试 Return 未改善，说明模型在记忆数据。已加：train/val 划分、val-based early stopping、label_smoothing、weight_decay、梯度裁剪。
