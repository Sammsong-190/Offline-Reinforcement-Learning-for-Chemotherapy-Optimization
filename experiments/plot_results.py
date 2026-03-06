"""
Generate paper figures: Robustness (return vs param shift), Safety table
Run after: python -m experiments.run_experiments
"""
import os
import json
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
except ImportError:
    print("pip install matplotlib for figures")
    exit(1)


def load_results():
    base = "experiments/results"
    exp1 = json.load(open(f"{base}/exp1_main.json")) if os.path.exists(f"{base}/exp1_main.json") else []
    exp2 = json.load(open(f"{base}/exp2_robustness.json")) if os.path.exists(f"{base}/exp2_robustness.json") else {}
    exp3 = json.load(open(f"{base}/exp3_safety.json")) if os.path.exists(f"{base}/exp3_safety.json") else []
    return exp1, exp2, exp3


def plot_robustness(exp2, out_path="experiments/figures/fig_robustness.pdf"):
    """Fig: Return vs parameter variation (tumor/immune/drug)"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    scales = [0.0, 0.1, 0.2, 0.3]
    modes = ["tumor_growth", "immune_strength", "drug_decay"]
    titles = ["Tumor Growth (r1) ±%", "Immune Strength (c1) ±%", "Drug Decay (d2) ±%"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    policies = ["Expert", "BC", "CQL", "IQL"]

    for ax, mode, title in zip(axes, modes, titles):
        if mode not in exp2:
            continue
        for policy in policies:
            means, stds = [], []
            for s in scales:
                key = f"{policy}_scale{s}"
                if key in exp2[mode]:
                    m, std = exp2[mode][key]
                    means.append(float(m))
                    stds.append(float(std))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
            x = [int(scale * 100) for scale in scales]
            ax.errorbar(x, means, yerr=stds, label=policy, marker='o', capsize=3)
        ax.set_xlabel("Parameter variation (%)")
        ax.set_ylabel("Return")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


def plot_safety_table(exp3, out_path="experiments/figures/table_safety.txt"):
    """Save safety table as text"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("Policy\tToxicityViolation(%)\n")
        for r in exp3:
            f.write(f"{r['Policy']}\t{r['ToxicityViolation']:.1f}\n")
    print(f"Saved {out_path}")


def main():
    exp1, exp2, exp3 = load_results()
    if exp2:
        plot_robustness(exp2)
    if exp3:
        plot_safety_table(exp3)
    print("Done.")


if __name__ == "__main__":
    main()
