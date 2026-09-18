"""generate_charts.py — Produce chart PNGs for embedding in the project report."""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_data
from src.preprocessing import FEATURE_COLS, prepare_data

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "house_price_regression_dataset.csv")
OUT_DIR      = os.path.join(PROJECT_ROOT, "report_charts")
os.makedirs(OUT_DIR, exist_ok=True)

COLORS  = ["#3b82d4", "#7c5cd8", "#16a34a"]
MODELS  = ["Linear Regression", "Random Forest", "XGBoost"]
R2      = [0.9984, 0.9939, 0.9952]
MAE     = [8174.58, 16128.62, 14254.42]
RMSE    = [10071.48, 19873.98, 17522.45]

STYLE = dict(edgecolor="white", linewidth=0.8)

df = load_data(DATA_PATH)

# ── 1. R² Comparison Bar Chart ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(MODELS, R2, color=COLORS, **STYLE)
ax.set_ylim(0.98, 1.002)
ax.set_ylabel("R² Score", fontsize=11)
ax.set_title("Model Comparison — R² Score (higher is better)", fontsize=13, fontweight="bold", pad=12)
for bar, val in zip(bars, R2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "chart_r2.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved chart_r2.png")

# ── 2. MAE & RMSE Grouped Bar Chart ──────────────────────────────────────────
x = np.arange(len(MODELS))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 4.5))
b1 = ax.bar(x - width/2, MAE,  width, label="MAE",  color="#3b82d4", **STYLE)
b2 = ax.bar(x + width/2, RMSE, width, label="RMSE", color="#7c5cd8", **STYLE)
ax.set_xticks(x); ax.set_xticklabels(MODELS, fontsize=10)
ax.set_ylabel("Error (USD)", fontsize=11)
ax.set_title("Model Comparison — MAE vs RMSE (lower is better)", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f"${bar.get_height():,.0f}", ha="center", va="bottom", fontsize=8)
ax.spines[["top","right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "chart_mae_rmse.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved chart_mae_rmse.png")

# ── 3. Radar / Spider Chart — normalised metrics ──────────────────────────────
labels   = ["R²", "1 - MAE_norm", "1 - RMSE_norm"]
mae_norm  = np.array(MAE)  / max(MAE)
rmse_norm = np.array(RMSE) / max(RMSE)
r2_norm   = np.array(R2)   # already 0-1

# For each model: [R², 1-mae_norm, 1-rmse_norm]  — higher always = better
data_radar = np.column_stack([r2_norm, 1 - mae_norm, 1 - rmse_norm])
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
for i, (model, row) in enumerate(zip(MODELS, data_radar)):
    values = row.tolist() + [row[0]]
    ax.plot(angles, values, "o-", linewidth=2, color=COLORS[i], label=model)
    ax.fill(angles, values, alpha=0.10, color=COLORS[i])
ax.set_thetagrids(np.degrees(angles[:-1]), ["R² Score", "Low MAE", "Low RMSE"], fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title("Model Performance Radar", fontsize=13, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "chart_radar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved chart_radar.png")

# ── 4. Scatter — Square Footage vs House Price (coloured by Neighbourhood) ────
fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(df["Square_Footage"], df["House_Price"],
                     c=df["Neighborhood_Quality"], cmap="viridis",
                     alpha=0.55, s=18, linewidths=0)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Neighborhood Quality", fontsize=10)
ax.set_xlabel("Square Footage (sq ft)", fontsize=11)
ax.set_ylabel("House Price (USD)", fontsize=11)
ax.set_title("Square Footage vs House Price\n(colour = Neighborhood Quality)", fontsize=13, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}k"))
ax.spines[["top","right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "chart_scatter.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved chart_scatter.png")

# ── 5. Correlation Heatmap ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[FEATURE_COLS + ["House_Price"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 9})
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "chart_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved chart_heatmap.png")

# ── 6. House Price Distribution Histogram ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(df["House_Price"], bins=40, color="#3b82d4", edgecolor="white", linewidth=0.6)
ax.set_xlabel("House Price (USD)", fontsize=11)
ax.set_ylabel("Number of Properties", fontsize=11)
ax.set_title("House Price Distribution", fontsize=13, fontweight="bold", pad=12)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}k"))
ax.spines[["top","right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
mean_price = df["House_Price"].mean()
ax.axvline(mean_price, color="#e11d48", linestyle="--", linewidth=1.4, label=f"Mean: ${mean_price:,.0f}")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "chart_histogram.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved chart_histogram.png")

# ── 7. Box Plot — Price by Number of Bedrooms ─────────────────────────────────
bedroom_groups = [df[df["Num_Bedrooms"] == b]["House_Price"].values for b in sorted(df["Num_Bedrooms"].unique())]
bedroom_labels = [f"{b} Bed" for b in sorted(df["Num_Bedrooms"].unique())]
fig, ax = plt.subplots(figsize=(8, 4.5))
bp = ax.boxplot(bedroom_groups, tick_labels=bedroom_labels, patch_artist=True, notch=False,
                medianprops=dict(color="white", linewidth=2))
palette = ["#3b82d4","#7c5cd8","#16a34a","#f59e0b","#ef4444","#06b6d4"]
for patch, color in zip(bp["boxes"], palette):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax.set_xlabel("Number of Bedrooms", fontsize=11)
ax.set_ylabel("House Price (USD)", fontsize=11)
ax.set_title("House Price Distribution by Bedrooms", fontsize=13, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1e3:.0f}k"))
ax.spines[["top","right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "chart_boxplot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved chart_boxplot.png")

print(f"\nAll charts saved to: {OUT_DIR}")
