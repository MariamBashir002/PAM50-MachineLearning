import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

os.makedirs("figures", exist_ok=True)
summary = pd.read_csv("results/summary.csv").set_index("Analysis")
ks = pd.read_csv("results/k_sensitivity.csv")
paired = pd.read_csv("results/paired_tests.csv")

# 1. k sensitivity, with the spread across folds
fig, ax = plt.subplots(figsize=(7, 4))
k_vals = sorted(ks["K"].unique())
for name, color, shift in [("All genes", "#4c78a8", -0.04), ("PAM50 removed", "#e45756", 0.04)]:
    d = ks[ks["Dataset"] == name].sort_values("K")
    ax.errorbar(np.arange(len(k_vals)) + shift, d["Balanced accuracy"], yerr=d["CV SD"],
                marker="o", capsize=3, label=name, color=color)
ax.set_xticks(range(len(k_vals)))
ax.set_xticklabels(k_vals)
ax.set_xlabel("Number of selected genes (k)")
ax.set_ylabel("Balanced accuracy")
ax.set_title(
    "Sensitivity to k (single 5-fold run, bars = SD across folds)", fontsize=10)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("figures/k_sensitivity.png", dpi=150)
plt.close()

# 2. balanced accuracy by gene set, with spread
rows = [
    ("All genes (repeated CV)", "All genes"),
    ("PAM50 genes removed (repeated CV)", "PAM50 genes removed"),
    ("Top-50 non-PAM50 genes removed (repeated CV)",
     "Top-50 non-PAM50\ngenes removed"),
    ("Random 50 genes removed (mean, single CV)", "Random 50 genes\nremoved*"),
    ("PAM50-only logistic, k=50 (single CV, ceiling)", "PAM50 genes only\n(ceiling)*"),
]
vals = [summary.loc[r, "Balanced accuracy"] for r, _ in rows]
sds = [0 if pd.isna(summary.loc[r, "SD"]) else summary.loc[r, "SD"]
       for r, _ in rows]
y = np.arange(len(rows))[::-1]

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(vals, y, xerr=sds, fmt="o", capsize=4, color="#4c78a8")
for yi, v in zip(y, vals):
    ax.text(v, yi + 0.18, f"{v:.3f}", ha="center", fontsize=9)
ax.set_yticks(y)
ax.set_yticklabels([label for _, label in rows])
ax.set_xlim(0.80, 0.90)
ax.set_xlabel("Balanced accuracy")
ax.set_title("Balanced accuracy by gene set (logistic regression)", fontsize=10)
fig.text(0.01, 0.01,
         "Bars: SD across 10 repeats, or across 100 random draws. *Single 5-fold run.",
         fontsize=7)
plt.tight_layout(rect=(0, 0.04, 1, 1))
plt.savefig("figures/balanced_accuracy.png", dpi=150)
plt.close()

# 3. paired differences with 95% intervals
crit = stats.t.ppf(0.975, 49)
y = np.arange(len(paired))[::-1]
fig, ax = plt.subplots(figsize=(8, 3))
for yi, (_, r) in zip(y, paired.iterrows()):
    se = abs(r["Mean difference"] / r["t"])
    ax.errorbar(r["Mean difference"], yi, xerr=crit *
                se, fmt="o", capsize=4, color="#4c78a8")
ax.axvline(0, color="grey", linestyle="--")
ax.set_yticks(y)
ax.set_yticklabels(paired["Comparison"])
ax.set_xlabel("Difference in balanced accuracy (95% interval)")
ax.set_title(
    "Paired comparisons (corrected resampled t-test, 10 x 5-fold CV)", fontsize=10)
plt.tight_layout()
plt.savefig("figures/paired_differences.png", dpi=150)
plt.close()

print("Figures saved.")
