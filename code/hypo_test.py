import pandas as pd
import numpy as np
from scipy import stats
import glob
import os

# =========================
# CONFIG
# =========================
base_path = "../data/backtest/static_roll_analysis"
roll_windows = list(range(1, 5)) + [7, 8, 9, 10, 11, 14]

# =========================
# LOAD DATA
# =========================
all_core = []
all_other = []

for d in roll_windows:
    path = f"{base_path}/roll_{d}d/rolling_spread.csv"

    df = pd.read_csv(path, index_col=0).dropna()
    df = df[["difference"]].rename(columns={"difference": "spread"})

    if d in [3, 4, 7]:
        all_core.append(df["spread"])
    else:
        all_other.append(df["spread"])

core = pd.concat(all_core)
other = pd.concat(all_other)

print(core)

# =========================
# BASIC STATS
# =========================
print("\n========== BASIC STATS ==========")

print("Core (T-3/4/5):")
print(f"Mean: {core.mean():.4f}")
print(f"Std : {core.std():.4f}")
print(f"N   : {len(core)}")

print("\nOther windows:")
print(f"Mean: {other.mean():.4f}")
print(f"Std : {other.std():.4f}")
print(f"N   : {len(other)}")

# =========================
# SIMPLE T-TEST
# =========================
t_stat, p_val = stats.ttest_ind(core, other, equal_var=False)

print("\n========== T-TEST ==========")
print(f"T-stat: {t_stat:.4f}")
print(f"P-val : {p_val:.6f}")

if p_val < 0.05:
    print("→ Significant difference")
else:
    print("→ NOT statistically significant")

# =========================
# EFFECT SIZE (VERY IMPORTANT)
# =========================
effect = core.mean() - other.mean()

print("\n========== EFFECT SIZE ==========")
print(f"Core - Other: {effect:.6f}")