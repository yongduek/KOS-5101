import nbformat as nbf

nb = nbf.v4.new_notebook()

# Add a header markdown cell
nb.cells.append(nbf.v4.new_markdown_cell('# Simulation Analysis Interactive Workspace\nThis notebook converts the `simulation_analysis.py` pipeline into an interactive environment for further experimentation.'))

# We will break the script into structural cells based on comments
# Cell 1: imports and functions
code_imports = """import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def ols(X_mat, y):
    coef, *_ = np.linalg.lstsq(X_mat, y, rcond=None)
    y_hat = X_mat @ coef
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return coef, r2

def design_matrix(x_cols):
    return np.column_stack([np.ones(len(x_cols[0]))] + x_cols)"""

nb.cells.append(nbf.v4.new_code_cell(code_imports))

# Cell 2: DataLoader
code_data = """# 1. Loading integer item responses and recreating totals
df_rses = pd.read_csv("rses_simulated.csv")
df_rssis = pd.read_csv("rssis_simulated.csv")
df_iss = pd.read_csv("iss_simulated.csv")

# Reverse-coding handling for items marked with _REV
for col in df_rses.columns:
    if col.endswith('_REV'):
        df_rses[col] = 5 - df_rses[col]

for col in df_iss.columns:
    if col.endswith('_REV'):
        df_iss[col] = 6 - df_iss[col]

X = df_rses.sum(axis=1).values
Y = df_rssis.sum(axis=1).values
M = df_iss.sum(axis=1).values
N = len(X)

# Store in a single DF for correlation and plotting
df = pd.DataFrame({
    "RSES (X)": X,
    "ISS (M)": M,
    "RSSIS (Y)": Y
})
df.head()"""

nb.cells.append(nbf.v4.new_code_cell(code_data))

# Cell 3: Descriptive Stats
code_desc = """# 2. DESCRIPTIVE STATS
targets = {
    "RSES (X)": {"mean": 29.87, "sd": 5.24},
    "ISS (M)": {"mean": 86.10, "sd": 12.27},
    "RSSIS (Y)": {"mean": 81.84, "sd": 31.49}
}
print(f"{'Variable':<15} | {'Sim Int Mean':>12} | {'Paper Mean':>10} | {'Sim Int SD':>10} | {'Paper SD':>10}")
print("-" * 65)
for var, target in targets.items():
    print(f"{var:<15} | {df[var].mean():>12.2f} | {target['mean']:>10.2f} | {df[var].std():>10.2f} | {target['sd']:>10.2f}")"""

nb.cells.append(nbf.v4.new_code_cell(code_desc))

# Cell 4: Correlations
code_corr = """# 3. CORRELATIONS
target_corrs = {
    ("RSES (X)", "ISS (M)"): 0.46,
    ("RSES (X)", "RSSIS (Y)"): -0.41,
    ("ISS (M)", "RSSIS (Y)"): -0.49
}
print(f"{'Pair':<20} | {'Sim Int Corr':>15} | {'Paper Corr':>15}")
print("-" * 56)
for pair, target in target_corrs.items():
    corr = df[pair[0]].corr(df[pair[1]])
    print(f"{pair[0][:4]}-{pair[1][:4]:<15} | {corr:>15.3f} | {target:>15.3f}")"""

nb.cells.append(nbf.v4.new_code_cell(code_corr))

# Cell 5: Mediation
code_med = """# 4. MEDIATION ANALYSIS
# Step 1: Y = c X  (Total Effect)
c1, r2_total = ols(design_matrix([X]), Y)
total_effect = c1[1]

# Step 2: M = a X
c2, r2_m = ols(design_matrix([X]), M)
a_path = c2[1]

# Step 3: Y = c' X + b M (Direct and B path)
c3, r2_final = ols(design_matrix([X, M]), Y)
direct_effect = c3[1]
b_path = c3[2]

indirect_effect = a_path * b_path

print(f"Path c (Total Effect X->Y)     : {total_effect:8.3f}")
print(f"Path a (X->M)                  : {a_path:8.3f}")
print(f"Path b (M->Y)                  : {b_path:8.3f}")
print(f"Path c' (Direct Effect X->Y)   : {direct_effect:8.3f}")
print(f"Indirect Effect (a * b)        : {indirect_effect:8.3f}")

# Bootstrapping for Indirect Effect
np.random.seed(2025)
boots = 2000
ind_effects = []
for _ in range(boots):
    idx = np.random.choice(N, N, replace=True)
    X_b, Y_b, M_b = X[idx], Y[idx], M[idx]
    
    ca, _ = ols(design_matrix([X_b]), M_b)
    cb, _ = ols(design_matrix([X_b, M_b]), Y_b)
    ind_effects.append(ca[1] * cb[2])
    
ci_lower, ci_upper = np.percentile(ind_effects, [2.5, 97.5])
print(f"\\nBootstrap (N={boots}) Indirect Effect 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")"""

nb.cells.append(nbf.v4.new_code_cell(code_med))

# Cell 6: Plotting
code_plot = """# 5. VISUAL EXPLORATION
# Plot 1: Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Fig 1: Correlation Matrix")
plt.tight_layout()
plt.show()

# Plot 2: Distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df["RSES (X)"], kde=True, ax=axes[0], color="skyblue")
axes[0].set_title("Self-Esteem (RSES)")

sns.histplot(df["ISS (M)"], kde=True, ax=axes[1], color="lightgreen")
axes[1].set_title("Intercultural Sensitivity (ISS)")

sns.histplot(df["RSSIS (Y)"], kde=True, ax=axes[2], color="salmon")
axes[2].set_title("Acculturative Stress (RSSIS)")

plt.suptitle("Fig 2: Variable Distributions (Simulated Integers)", y=1.05)
plt.tight_layout()
plt.show()

# Plot 3: Mediation Scatters
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.regplot(x="RSES (X)", y="ISS (M)", data=df, ax=axes[0], color="purple", line_kws={"color":"black"}, scatter_kws={'alpha':0.5})
axes[0].set_title(f"a Path (X -> M)\\nCoef: {a_path:.2f}")

sns.regplot(x="RSES (X)", y="RSSIS (Y)", data=df, ax=axes[1], color="teal", line_kws={"color":"black"}, scatter_kws={'alpha':0.5})
axes[1].set_title(f"c Path (X -> Y)\\nCoef: {total_effect:.2f}")

sns.regplot(x="ISS (M)", y="RSSIS (Y)", data=df, ax=axes[2], color="orange", line_kws={"color":"black"}, scatter_kws={'alpha':0.5})
axes[2].set_title(f"b Path (M -> Y)\\nUnadjusted Coef: {c2[1]:.2f}")

plt.suptitle("Fig 3: Mediation Relationships (Scatter & OLS)", y=1.05)
plt.tight_layout()
plt.show()

# Plot 4: Bootstrap Distribution
plt.figure(figsize=(8, 5))
sns.histplot(ind_effects, kde=True, color="skyblue")
plt.axvline(np.mean(ind_effects), color="red", linestyle="--", label=f"Mean: {np.mean(ind_effects):.3f}")
plt.axvline(ci_lower, color="green", linestyle=":", label=f"95% CI Lower: {ci_lower:.3f}")
plt.axvline(ci_upper, color="green", linestyle=":", label=f"95% CI Upper: {ci_upper:.3f}")
plt.title(f"Fig 4: Bootstrap Distribution of Indirect Effect ($a \\times b$)\\nN={boots} Resamples")
plt.xlabel("Indirect Effect Size")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()"""

nb.cells.append(nbf.v4.new_code_cell(code_plot))

# Write out the notebook file
nbf.write(nb, "simulation_analysis.ipynb")
