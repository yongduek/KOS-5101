"""
09_bayes_latent_regression_rasch.py
====================================
Bayesian Model 2: Latent Regression Rasch (Person Explanatory)

  η_pi = Σ_j ϑ_j Z_pj + θ_p − β_i

This script:
  1. Builds the person predictor matrix Z (program, gender×program, SES)
  2. Runs MCMC for the latent regression Rasch model
  3. Reports posterior distributions for person-property effects
  4. Computes variance decomposition with uncertainty
  5. Compares Bayesian vs. MML estimates from script 04

Requires: cmdstanpy, numpy, pandas, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmdstanpy
import os
import json

out_dir = os.path.dirname(os.path.abspath(__file__))
stan_dir = os.path.join(out_dir, "stan_models")

# ── 1. Load data ──────────────────────────────────────────────────────────
items_df = pd.read_csv(os.path.join(out_dir, "data_items.csv"))
persons_df = pd.read_csv(os.path.join(out_dir, "data_persons.csv"))
resp_df = pd.read_csv(os.path.join(out_dir, "data_responses.csv"))
resp_matrix = resp_df.drop(columns=["person_id"]).values
N, I = resp_matrix.shape

# ── 2. Build person predictor matrix ──────────────────────────────────────
gender = persons_df["gender"].values
program = persons_df["program"].values
hises = persons_df["hises"].values

prog1 = (program == 1).astype(float)
prog3 = (program == 3).astype(float)
prog4 = (program == 4).astype(float)
female = (1 - gender).astype(float)

Z = np.column_stack([prog1, prog3, prog4,
                     female * prog1, female * prog3, female * prog4,
                     hises])
J = Z.shape[1]
predictor_names = ["Program 1 (Hauptschule)", "Program 3 (Realschule)",
                   "Program 4 (Gymnasium)", "Female × Prog 1",
                   "Female × Prog 3", "Female × Prog 4", "SES (HiSES)"]

# ── 3. Stan data ──────────────────────────────────────────────────────────
stan_data = {
    "N": N, "I": I, "J": J,
    "Y": resp_matrix.tolist(),
    "Z": Z.tolist(),
}

# ── 4. Compile and sample ─────────────────────────────────────────────────
model = cmdstanpy.CmdStanModel(
    stan_file=os.path.join(stan_dir, "latent_regression_rasch.stan")
)

print("Running MCMC for Latent Regression Rasch...")
fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    seed=42,
    show_console=False,
    adapt_delta=0.95,
    max_treedepth=12,
)

# ── 5. Diagnostics ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MCMC DIAGNOSTICS")
print("=" * 60)
print(fit.diagnose())

# ── 6. Posterior summaries ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BAYESIAN LATENT REGRESSION RASCH — POSTERIORS")
print("=" * 60)

vartheta_draws = fit.stan_variable("vartheta")  # (n_draws, J)
sigma_e_draws = fit.stan_variable("sigma_e")    # (n_draws,)
beta_draws = fit.stan_variable("beta")          # (n_draws, I)

vt_mean = vartheta_draws.mean(axis=0)
vt_sd = vartheta_draws.std(axis=0)
vt_q025 = np.percentile(vartheta_draws, 2.5, axis=0)
vt_q975 = np.percentile(vartheta_draws, 97.5, axis=0)

sigma_e_mean = sigma_e_draws.mean()
var_resid_mean = (sigma_e_draws ** 2).mean()

print(f"\nPerson Property Effects (ϑ) — Posterior summaries:")
print(f"{'Predictor':<30} {'Mean':>8} {'SD':>8} {'2.5%':>8} {'97.5%':>8} {'Signif?':>8}")
print("-" * 75)
for j in range(J):
    # "Significant" if 95% CI excludes zero
    sig = "***" if (vt_q025[j] > 0 or vt_q975[j] < 0) else ""
    print(f"{predictor_names[j]:<30} {vt_mean[j]:>8.3f} {vt_sd[j]:>8.3f} "
          f"{vt_q025[j]:>8.3f} {vt_q975[j]:>8.3f} {sig:>8}")

print(f"\nResidual person SD (σ_ε):  {sigma_e_mean:.3f} ± {sigma_e_draws.std():.3f}")
print(f"Residual person variance:  {var_resid_mean:.3f}")

# ── 7. Variance decomposition ────────────────────────────────────────────
# For each MCMC draw, compute explained variance
n_draws = vartheta_draws.shape[0]
var_explained_draws = np.zeros(n_draws)
var_total_draws = np.zeros(n_draws)

for d in range(n_draws):
    theta_fixed = Z @ vartheta_draws[d]
    ve = np.var(theta_fixed)
    vr = sigma_e_draws[d] ** 2
    var_explained_draws[d] = ve
    var_total_draws[d] = ve + vr

pct_explained = var_explained_draws / var_total_draws * 100

print(f"\nVariance Decomposition (posterior):")
print(f"  Explained: {var_explained_draws.mean():.3f} [{np.percentile(var_explained_draws, 2.5):.3f}, {np.percentile(var_explained_draws, 97.5):.3f}]")
print(f"  Residual:  {var_resid_mean:.3f}")
print(f"  % explained: {pct_explained.mean():.1f}% [{np.percentile(pct_explained, 2.5):.1f}%, {np.percentile(pct_explained, 97.5):.1f}%]")

# ── 8. Visualizations ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Forest plot of person property effects
ax = axes[0, 0]
y_pos = np.arange(J)
ax.barh(y_pos, vt_mean, xerr=vt_sd, color="#0D9488", alpha=0.7, edgecolor="white", capsize=3)
for j in range(J):
    ax.plot([vt_q025[j], vt_q975[j]], [j, j], "k-", linewidth=2)
ax.axvline(0, color="red", linestyle="--", alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(predictor_names, fontsize=8)
ax.set_xlabel("Effect (logits)")
ax.set_title("Person Property Effects (95% CI)", fontweight="bold")

# (b) Posterior of residual variance
ax = axes[0, 1]
ax.hist(sigma_e_draws ** 2, bins=50, density=True, color="#4472C4", alpha=0.7, edgecolor="white")
ax.axvline(var_resid_mean, color="red", linestyle="--", label=f"Mean = {var_resid_mean:.3f}")
ax.set_xlabel("σ²_ε (Residual Variance)")
ax.set_ylabel("Density")
ax.set_title("Posterior of Residual Variance", fontweight="bold")
ax.legend()

# (c) Posterior of % variance explained
ax = axes[1, 0]
ax.hist(pct_explained, bins=50, density=True, color="#E15759", alpha=0.7, edgecolor="white")
ax.axvline(pct_explained.mean(), color="red", linestyle="--",
           label=f"Mean = {pct_explained.mean():.1f}%")
ax.set_xlabel("% Variance Explained")
ax.set_ylabel("Density")
ax.set_title("Posterior of Variance Explained", fontweight="bold")
ax.legend()

# (d) Comparison: Bayesian vs MML (if available)
ax = axes[1, 1]
try:
    mml_effects = pd.read_csv(os.path.join(out_dir, "results_person_effects.csv"))
    mml_vals = mml_effects["effect"].values
    ax.scatter(mml_vals, vt_mean, c="#4472C4", s=80, zorder=5, edgecolors="white")
    for j in range(J):
        ax.annotate(predictor_names[j].split("(")[0].strip(),
                    (mml_vals[j], vt_mean[j]),
                    xytext=(5, 5), textcoords="offset points", fontsize=7)
    lims = [min(mml_vals.min(), vt_mean.min()) - 0.2,
            max(mml_vals.max(), vt_mean.max()) + 0.2]
    ax.plot(lims, lims, "r--", alpha=0.5)
    r_mml = np.corrcoef(mml_vals, vt_mean)[0, 1]
    ax.set_xlabel("MML Estimates (Script 04)")
    ax.set_ylabel("Bayesian Posterior Means")
    ax.set_title(f"Bayesian vs MML (r = {r_mml:.3f})", fontweight="bold")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
except FileNotFoundError:
    ax.text(0.5, 0.5, "Run 04_latent_regression_rasch.py\nfor MML comparison",
            ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_title("Bayesian vs MML (not available)", fontweight="bold")

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig_bayes_lat_reg_rasch.png"), dpi=150, bbox_inches="tight")
print(f"\nFigure saved: fig_bayes_lat_reg_rasch.png")
plt.close()

# ── 9. Save ───────────────────────────────────────────────────────────────
log_lik_draws = fit.stan_variable("log_lik")
log_lik_flat = log_lik_draws.reshape(log_lik_draws.shape[0], -1)
np.save(os.path.join(out_dir, "bayes_loglik_lat_reg_rasch.npy"), log_lik_flat)

effects_df = pd.DataFrame({
    "predictor": predictor_names,
    "mean": np.round(vt_mean, 4),
    "sd": np.round(vt_sd, 4),
    "q025": np.round(vt_q025, 4),
    "q975": np.round(vt_q975, 4),
})
effects_df.to_csv(os.path.join(out_dir, "bayes_results_lat_reg_rasch.csv"), index=False)

summary = {
    "model": "Latent Regression Rasch (Bayesian)",
    "sigma_e_mean": round(float(sigma_e_mean), 4),
    "var_resid_mean": round(float(var_resid_mean), 4),
    "pct_explained_mean": round(float(pct_explained.mean()), 1),
}
with open(os.path.join(out_dir, "bayes_summary_lat_reg_rasch.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("Results saved.")
