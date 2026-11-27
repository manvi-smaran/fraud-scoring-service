"""
EDA script for Credit Card Fraud Detection dataset.
Outputs 4 diagnostic plots to diagrams/.

Usage:  python notebooks/eda.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA     = ROOT / "data" / "creditcard.csv"
DIAGRAMS = ROOT / "diagrams"
DIAGRAMS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")

# ── 1. load ────────────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA)

# ── 2. basic info ──────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nDtypes:\n{df.dtypes.value_counts().to_string()}")
print(f"\nNull counts  : {df.isnull().sum().sum()} total nulls")

total   = len(df)
fraud   = df["Class"].sum()
legit   = total - fraud
rate    = fraud / total

print(f"\n{'='*50}")
print(f"Total transactions : {total:,}")
print(f"Fraud transactions : {fraud:,}")
print(f"Legit transactions : {legit:,}")
print(f"Fraud rate         : {rate:.4%}")

# ── 3. feature engineering ─────────────────────────────────────────────────────
df["Hour"] = (df["Time"] / 3600) % 24

# ── 4. class distribution bar chart ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["Class"].value_counts().sort_index()
bars = ax.bar(["Legitimate", "Fraud"], counts.values,
              color=["steelblue", "tomato"], width=0.5, edgecolor="white")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{val:,}", ha="center", va="bottom", fontsize=11)
ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
ax.set_ylabel("Transaction Count")
ax.set_yscale("log")
ax.set_ylim(1, counts.max() * 3)
plt.tight_layout()
fig.savefig(DIAGRAMS / "class_distribution.png", dpi=150)
plt.close(fig)
print("\n[saved] class_distribution.png")

# ── 5. amount distribution ─────────────────────────────────────────────────────
med_fraud = df.loc[df["Class"] == 1, "Amount"].median()
med_legit = df.loc[df["Class"] == 0, "Amount"].median()
print(f"\nMedian Amount — Fraud:  ${med_fraud:.2f}")
print(f"Median Amount — Legit:  ${med_legit:.2f}")

fig, ax = plt.subplots(figsize=(8, 4))
for label, colour in [(0, "steelblue"), (1, "tomato")]:
    subset = df.loc[df["Class"] == label, "Amount"].clip(upper=500)
    ax.hist(subset, bins=80, alpha=0.6, color=colour,
            label=f"{'Fraud' if label else 'Legit'} (n={len(subset):,})", density=True)
ax.set_title("Transaction Amount Distribution (clipped at $500)", fontsize=13, fontweight="bold")
ax.set_xlabel("Amount ($)")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
fig.savefig(DIAGRAMS / "amount_distribution.png", dpi=150)
plt.close(fig)
print("[saved] amount_distribution.png")

# ── 6. hourly pattern ──────────────────────────────────────────────────────────
hourly = (
    df.groupby(["Hour", "Class"])
    .size()
    .reset_index(name="Count")
)
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
for ax, (cls, colour, title) in zip(
    axes,
    [(0, "steelblue", "Legitimate Transactions"),
     (1, "tomato",    "Fraud Transactions")]
):
    sub = hourly[hourly["Class"] == cls]
    ax.bar(sub["Hour"], sub["Count"], color=colour, edgecolor="white", width=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Count")
    ax.set_xticks(range(0, 24, 2))
plt.suptitle("Hourly Transaction Pattern", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(DIAGRAMS / "hourly_pattern.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[saved] hourly_pattern.png")

# ── 7. feature correlation with Class ─────────────────────────────────────────
v_cols = [c for c in df.columns if c.startswith("V")] + ["Amount", "Hour"]
corrs  = df[v_cols + ["Class"]].corr()["Class"].drop("Class").sort_values()

top5_neg = corrs.head(5)
top5_pos = corrs.tail(5).iloc[::-1]

print("\nTop 5 negatively correlated features with Class:")
print(top5_neg.to_string())
print("\nTop 5 positively correlated features with Class:")
print(top5_pos.to_string())

top10 = pd.concat([top5_neg, top5_pos]).sort_values()
colours = ["tomato" if v < 0 else "steelblue" for v in top10.values]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top10.index, top10.values, color=colours, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Top 10 Feature Correlations with Fraud Class", fontsize=13, fontweight="bold")
ax.set_xlabel("Pearson Correlation")
plt.tight_layout()
fig.savefig(DIAGRAMS / "feature_correlation.png", dpi=150)
plt.close(fig)
print("[saved] feature_correlation.png")

# ── 8. final confirmation ──────────────────────────────────────────────────────
saved = list(DIAGRAMS.glob("*.png"))
print(f"\n{'='*50}")
print(f"All {len(saved)} diagrams saved to {DIAGRAMS}/")
for p in sorted(saved):
    print(f"  ✓ {p.name}")
print("Phase 1 EDA complete.")
