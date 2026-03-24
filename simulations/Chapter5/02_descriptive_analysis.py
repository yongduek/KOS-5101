"""
02_descriptive_analysis.py
==========================
Exploratory / descriptive analysis of the simulated response data.
This mirrors the initial data exploration described in Chapter 5.

What this script does:
  1. Loads the data files created by 01_generate_data.py
  2. Computes item-level statistics (mean scores by topic area & modeling type)
  3. Computes person-level statistics (sum scores by program, gender)
  4. Creates visualizations:
     - Heatmap of mean scores by topic area × modeling type
     - Distribution of total scores
     - Score distributions by program level
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

out_dir = os.path.dirname(os.path.abspath(__file__))

# ── 1. Load data ──────────────────────────────────────────────────────────
items_df = pd.read_csv(os.path.join(out_dir, "data_items.csv"))
persons_df = pd.read_csv(os.path.join(out_dir, "data_persons.csv"))
resp_df = pd.read_csv(os.path.join(out_dir, "data_responses.csv"))

resp_matrix = resp_df.drop(columns=["person_id"]).values
item_names = resp_df.columns[1:].tolist()

# ── 2. Item-level statistics ──────────────────────────────────────────────
item_means = resp_matrix.mean(axis=0)
items_df["p_value"] = item_means

print("=" * 60)
print("ITEM-LEVEL STATISTICS")
print("=" * 60)
print(items_df[["item_name", "topic_area", "modeling_type", "p_value", "beta_true"]].to_string(index=False))

# Mean by topic area
print("\nMean scores by Topic Area:")
for ta in ["Arithmetic", "Geometry", "Algebra"]:
    mask = items_df["topic_area"] == ta
    print(f"  {ta:12s}: {item_means[mask].mean():.3f}")

# Mean by modeling type
print("\nMean scores by Modeling Type:")
for mt in ["TechnicalProcessing", "NumericalModeling", "AbstractModeling"]:
    mask = items_df["modeling_type"] == mt
    print(f"  {mt:22s}: {item_means[mask].mean():.3f}")

# Cross-tabulation: topic_area × modeling_type
print("\nMean scores by Topic Area × Modeling Type:")
cross = items_df.groupby(["topic_area", "modeling_type"])["p_value"].mean().unstack()
cross = cross.reindex(index=["Arithmetic", "Geometry", "Algebra"],
                       columns=["TechnicalProcessing", "NumericalModeling", "AbstractModeling"])
print(cross.round(3).to_string())

# ── 3. Person-level statistics ────────────────────────────────────────────
sum_scores = resp_matrix.sum(axis=1)
persons_df["sum_score"] = sum_scores

print("\n" + "=" * 60)
print("PERSON-LEVEL STATISTICS")
print("=" * 60)
print(f"  Mean sum score:   {sum_scores.mean():.2f}")
print(f"  SD sum score:     {sum_scores.std():.2f}")
print(f"  Min:              {sum_scores.min()}")
print(f"  Max:              {sum_scores.max()}")

print("\nMean sum score by Program:")
for prog in [1, 2, 3, 4]:
    mask = persons_df["program"] == prog
    prog_names = {1: "Hauptschule", 2: "Integrated", 3: "Realschule", 4: "Gymnasium"}
    print(f"  {prog} ({prog_names[prog]:12s}): {sum_scores[mask].mean():.2f}  (n={mask.sum()})")

print("\nMean sum score by Gender:")
for g in [0, 1]:
    mask = persons_df["gender"] == g
    label = "Female" if g == 0 else "Male"
    print(f"  {label:8s}: {sum_scores[mask].mean():.2f}  (n={mask.sum()})")

print("\nMean sum score by Gender × Program:")
for prog in [1, 2, 3, 4]:
    for g in [0, 1]:
        mask = (persons_df["program"] == prog) & (persons_df["gender"] == g)
        label = f"  Prog {prog}, {'F' if g == 0 else 'M'}"
        print(f"{label}: {sum_scores[mask].mean():.2f}  (n={mask.sum()})")

# ── 4. Visualizations ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Heatmap of mean scores
ax = axes[0, 0]
heatdata = cross.values
im = ax.imshow(heatdata, cmap="YlGnBu", aspect="auto", vmin=0.2, vmax=0.8)
ax.set_xticks(range(3))
ax.set_xticklabels(["Technical\nProcessing", "Numerical\nModeling", "Abstract\nModeling"], fontsize=9)
ax.set_yticks(range(3))
ax.set_yticklabels(["Arithmetic", "Geometry", "Algebra"], fontsize=9)
ax.set_title("Mean Scores by Topic × Modeling Type", fontweight="bold")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{heatdata[i, j]:.2f}", ha="center", va="center",
                color="white" if heatdata[i, j] > 0.55 else "black", fontsize=12)
plt.colorbar(im, ax=ax, shrink=0.8)

# (b) Distribution of sum scores
ax = axes[0, 1]
ax.hist(sum_scores, bins=range(0, 20), edgecolor="white", color="#4472C4", alpha=0.8)
ax.set_xlabel("Sum Score (out of 18)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Total Scores", fontweight="bold")
ax.axvline(sum_scores.mean(), color="red", linestyle="--", label=f"Mean = {sum_scores.mean():.1f}")
ax.legend()

# (c) Box plots by program
ax = axes[1, 0]
prog_data = [sum_scores[persons_df["program"] == p] for p in [1, 2, 3, 4]]
bp = ax.boxplot(prog_data, labels=["Hauptschule\n(1)", "Integrated\n(2)",
                                     "Realschule\n(3)", "Gymnasium\n(4)"],
                patch_artist=True)
colors = ["#F28E2B", "#E15759", "#76B7B2", "#4E79A7"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel("Sum Score")
ax.set_title("Scores by School Program", fontweight="bold")

# (d) Mean scores by gender × program
ax = axes[1, 1]
progs = [1, 2, 3, 4]
female_means = [sum_scores[(persons_df["program"] == p) & (persons_df["gender"] == 0)].mean() for p in progs]
male_means = [sum_scores[(persons_df["program"] == p) & (persons_df["gender"] == 1)].mean() for p in progs]
x = np.arange(4)
ax.bar(x - 0.15, female_means, 0.3, label="Female", color="#E15759", alpha=0.8)
ax.bar(x + 0.15, male_means, 0.3, label="Male", color="#4E79A7", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(["Hauptschule", "Integrated", "Realschule", "Gymnasium"], fontsize=8)
ax.set_ylabel("Mean Sum Score")
ax.set_title("Gender × Program Interaction", fontweight="bold")
ax.legend()

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig_descriptive.png"), dpi=150, bbox_inches="tight")
print(f"\nFigure saved: fig_descriptive.png")
plt.close()

# ── 5. Correlation between sum score and true theta ───────────────────────
r = np.corrcoef(sum_scores, persons_df["theta_true"])[0, 1]
print(f"\nCorrelation between sum score and true θ: r = {r:.3f}")
print("(This shows the sum score is a reasonable but imperfect measure of ability)")
