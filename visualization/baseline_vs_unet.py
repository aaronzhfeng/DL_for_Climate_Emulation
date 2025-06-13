# visualization/baseline_vs_unet.py
"""
Make the four summary plots that compare best_unet with the three baselines
(linear, MLP, CNN).  All metrics are read from each model's
results/<model_name>/metrics.csv file.

Expected CSV columns (minimum):
    epoch
    train/loss
    val/tas/rmse
    val/pr/rmse
    val/tas/time_mean_rmse
    val/pr/time_mean_rmse
    test/tas/rmse
    test/pr/rmse
Feel free to add/remove columns – the script ignores anything unused.

Author: <you>
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
# Configuration
###############################################################################
# Friendly labels → folder names under results/
MODELS = {
    "Best UNet":      "best_unet",
    "Linear":         "linear_baseline",
    "MLP":            "mlp_baseline",
    "CNN":            "cnn_baseline",
}

# Where metrics.csv live
THIS_DIR     = Path(__file__).resolve().parent
RESULTS_DIR  = THIS_DIR.parent / "results"
FIG_DIR      = THIS_DIR / "figures" / "baseline_vs_unet"
FIG_DIR.mkdir(parents=True, exist_ok=True)

###############################################################################
# Helpers
###############################################################################
def load_metrics(folder_name: str) -> pd.DataFrame:
    csv_path = RESULTS_DIR / folder_name / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not locate {csv_path}")
    return pd.read_csv(csv_path)

def last_row(df: pd.DataFrame) -> pd.Series:
    """Assume last row = final epoch; if column `epoch` exists, use max epoch."""
    if "epoch" in df.columns:
        return df.loc[df["epoch"].idxmax()]
    return df.iloc[-1]

###############################################################################
# Read all model logs
###############################################################################
raw_logs   = {label: load_metrics(folder) for label, folder in MODELS.items()}
final_rows = {label: last_row(df)         for label, df     in raw_logs.items()}

###############################################################################
# 1. Grouped bar chart – final test RMSE (tas & pr)
###############################################################################
labels          = list(MODELS.keys())
test_rmse_tas   = [final_rows[l]["test/tas/rmse"] for l in labels]
test_rmse_pr    = [final_rows[l]["test/pr/rmse"]  for l in labels]

x  = range(len(labels))
w  = 0.35  # bar width
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar([i - w/2 for i in x], test_rmse_tas, width=w, label="tas (°C)")
ax.bar([i + w/2 for i in x], test_rmse_pr,  width=w, label="pr (mm/day)")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylabel("Final Test RMSE")
ax.set_title("Final Test RMSE – Best UNet vs. Baseline Models")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "bar_final_test_rmse.png", dpi=300)

###############################################################################
# 2. Validation RMSE vs. epoch (tas & pr)
###############################################################################
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 6), sharex=True)

for label, df in raw_logs.items():
    ax1.plot(df["epoch"], df["val/tas/rmse"], label=label)
    ax2.plot(df["epoch"], df["val/pr/rmse"],  label=label)

ax1.set_ylabel("Val RMSE (tas)")
ax1.set_title("Validation RMSE vs. Epoch")
ax1.grid(alpha=0.3)
ax1.legend(fontsize=8)

ax2.set_ylabel("Val RMSE (pr)")
ax2.set_xlabel("Epoch")
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "line_val_rmse_vs_epoch.png", dpi=300)

###############################################################################
# 3. Validation time-mean RMSE (tas & pr) – bar chart
###############################################################################
val_mean_rmse_tas = [final_rows[l]["val/tas/time_mean_rmse"] for l in labels]
val_mean_rmse_pr  = [final_rows[l]["val/pr/time_mean_rmse"]  for l in labels]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar([i - w/2 for i in x], val_mean_rmse_tas, width=w, label="tas (°C)")
ax.bar([i + w/2 for i in x], val_mean_rmse_pr,  width=w, label="pr (mm/day)")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylabel("Validation Time-Mean RMSE")
ax.set_title("Time-Mean Validation RMSE – Best UNet vs. Baseline Models")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "bar_val_time_mean_rmse.png", dpi=300)

###############################################################################
# 4. Scatter – Val PR RMSE vs. Val TAS RMSE
###############################################################################
fig, ax = plt.subplots(figsize=(5, 5))
for label in labels:
    ax.scatter(final_rows[label]["val/tas/rmse"],
               final_rows[label]["val/pr/rmse"],
               label=label, s=80, edgecolor="k")
ax.set_xlabel("Val TAS RMSE")
ax.set_ylabel("Val PR RMSE")
ax.set_title("Validation PR vs. TAS RMSE")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "scatter_val_pr_vs_tas_rmse.png", dpi=300)

print(f"✓ All figures written to {FIG_DIR.relative_to(THIS_DIR.parent)}")

