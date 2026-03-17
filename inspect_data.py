"""
快速查看 offline_dataset.npz 内容
Run: python inspect_data.py
"""
import numpy as np
import os

path = "offline_dataset.npz"
if not os.path.exists(path):
    print(f"文件不存在: {path}")
    print("请先运行: python scripts/generate_data.py -o offline_dataset.npz")
    exit(1)

d = np.load(path)
print("=" * 50)
print("offline_dataset.npz 内容概览")
print("=" * 50)
print(f"键名: {d.files}")
print()
for k in d.files:
    arr = d[k]
    print(f"  {k:8} shape={arr.shape}  dtype={arr.dtype}")
print()
n = len(d["s"])
print(f"总 transitions: {n:,}")
print(f"约 trajectories: {n // 200} (假设每轨迹 ~200 步)")
print()
# 简单统计
print("动作分布:")
a = np.array(d["a"]).flatten()
for i, val in enumerate([0.0, 0.5, 1.0, 2.0]):
    pct = (a == val).sum() / len(a) * 100
    print(f"  u={val}  {pct:.1f}%")
print()
print("奖励: min={:.3f}  max={:.3f}  mean={:.3f}".format(
    d["r"].min(), d["r"].max(), d["r"].mean()))
print()
# Cost 分布 (SCI: 理想 5%-15% 违规，Safe RL 才能从错误中学习)
if "c" in d.files:
    c = np.array(d["c"]).flatten()
    cost_rate = c.mean() * 100
    print("Cost 分布 (c=1 违规):")
    print(f"  违规率: {cost_rate:.2f}%  (理想 5%-15%)")
    if cost_rate < 1:
        print("  [警告] c=1 过少，Safe RL 学不到危险")
    elif cost_rate > 50:
        print("  [警告] c=1 过多，任务可能过难")
    else:
        print("  [OK] 违规比例适中")
else:
    print("Cost: (无 c 列，需重新生成数据)")
print("=" * 50)
