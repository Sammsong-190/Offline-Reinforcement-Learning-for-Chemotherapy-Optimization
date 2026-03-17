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
print("=" * 50)
