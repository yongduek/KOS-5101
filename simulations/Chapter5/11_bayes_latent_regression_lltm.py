"""
11_bayes_latent_regression_lltm.py
===================================
Bayesian Model 4: Latent Regression LLTM (Doubly Explanatory)

  η_pi = Σ_j ϑ_j Z_pj + θ_p − Σ_k β_k X_ik

This script:
  1. Fits the combined model (person + item properties)
  2. Reports all posterior distributions
  3. Compares with MML results from script 06

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

# ── 2. Predictors ─────────────────────────────────────────────────────────
# Person predictors
gender = persons_df["gender"].values
program = persons_df["program"].values
hises = persons_df["hises"].values
prog1 = (program == 1).astype(float)
prog3 = (program == 3).astype(float)
prog4 = (program == 4).astype(float)
female = (1 - gender).astype(float)
Z = np.column_stack([prog1, prog3, prog4,
                     female * prog1, female * prog3, female * prog4, hises])
J = Z.shape[1]
person_pred_names = ["Program 1", "Program 3", "Program 4",
                     "Female×Prog1", "Female×Prog3", "Female×Prog4", "SES"]

# Item predictors
topic_areas = ["Arithmetic", "Geometry", "Algebra"]
modeling_types = ["TechnicalProcessing", "NumericalModeling", "AbstractModeling"]
X_item = np.zeros((I, 9)); cell_names = []
for idx, (ta, mt) in enumerate([(ta, mt) for ta in topic_areas for mt in modeling_types]):
    cell_names.append(f"{ta}_{mt}")
    mask = (items_df["topic_area"] == ta) & (items_df["modeling_type"] == mt)
    X_item[mask.values, idx] = 1.0
K = 9

# ── 3. Stan data ──────────────────────────────────────────────────────────
stan_data = {
    "N": N, "I": I, "J": J, "K": K,
    "Y": resp_matrix.tolist(),
    "Z": Z.tolist(),
    "X": X_item.tolist(),
}

# ── 4. Compile and sample ─────────────────────────────────────────────────
model = cmdstanpy.CmdStanModel(
    stan_file=os.path.join(stan_dir, "latent_regression_lltm.stan")
)

print("Running MCMC for Latent Regression LLTM (doubly explanatory)...")
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

print("\n" + "=" * 60)
print("MCMC DIAGNOSTICS")
print("=" * 60)
print(fit.diagnose())

# ── 5. Posteriors ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BAYESIAN LATENT REGRESSION LLTM — POSTERIORS")
print("=" * 60)

beta_k_draws = fit.stan_variable("beta_k")
vartheta_draws = fit.stan_variable("vartheta")
sigma_e_draws = fit.stan_variable("sigma_e")
theta_total_draws = fit.stan_variable("theta_total")

bk_mean = beta_k_draws.mean(axis=0)
bk_sd = beta_k_draws.std(axis=0)
bk_q025 = np.percentile(beta_k_draws, 2.5, axis=0)
bk_q975 = np.percentile(beta_k_draws, 97.5, axis=0)

vt_mean = vartheta_draws.mean(axis=0)
vt_sd = vartheta_draws.std(axis=0)
vt_q025 = np.percentile(vartheta_draws, 2.5, axis=0)
vt_q975 = np.percentile(vartheta_draws, 97.5, axis=0)

sigma_e_mean = sigma_e_draws.mean()
var_resid_mean = (sigma_e_draws ** 2).mean()

print(f"\nItem Property Effects:")
print(f"{'Cell':<35} {'Mean':>8} {'SD':>8} {'95% CI':>20}")
print("-" * 68)
for k in range(K):
    print(f"{cell_names[k]:<35} {bk_mean[k]:>8.3f} {bk_sd[k]:>8.3f} "
          f"[{bk_q025[k]:>7.3f}, {bk_q975[k]:>7.3f}]")

print(f"\nPerson Property Effects:")
print(f"{'Predictor':<25} {'Mean':>8} {'SD':>8} {'95% CI':>20}")
print("-" * 65)
for j in range(J):
    sig = "*" if (vt_q025[j] > 0 or vt_q975[j] < 0) else ""
    print(f"{person_pred_names[j]:<25} {vt_mean[j]:>8.3f} {vt_sd[j]:>8.3f} "
          f"[{vt_q025[j]:>7.3f}, {vt_q975[j]:>7.3f}] {sig}")

# Variance decomposition
n_draws = vartheta_draws.shape[0]
var_expl = np.array([np.var(Z @ vartheta_draws[d]) for d in range(n_draws)])
var_resid = sigma_e_draws ** 2
var_total = var_expl + var_resid
pct_expl = var_expl / var_total * 100

print(f"\nResidual σ_ε: {sigma_e_mean:.3f}")
print(f"Variance: Explained={var_expl.mean():.3f}, Residual={var_resid.mean():.3f}")
print(f"% explained: {pct_expl.mean():.1f}% [{np.percentile(pct_expl, 2.5):.1f}, {np.percentile(pct_expl, 97.5):.1f}]")

# ── 6. Visualization ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Item effects forest plot
ax = axes[0, 0]
y_pos = np.arange(K)
ax.barh(y_pos, bk_mean, color="#0D9488", alpha=0.7, edgecolor="white")
for k in range(K):
    ax.plot([bk_q025[k], bk_q975[k]], [k, k], "k-", linewidth=2)
ax.set_yticks(y_pos); ax.set_yticklabels(cell_names, fontsize=7)
ax.set_xlabel("Item Difficulty Effect (logits)")
ax.set_title("Item Property Effects (95% CI)", fontweight="bold")
ax.axvline(0, color="red", linestyle="--", alpha=0.5)

# (b) Person effects forest plot
ax = axes[0, 1]
y_pos = np.arange(J)
ax.barh(y_pos, vt_mean, color="#E15759", alpha=0.7, edgecolor="white")
for j in range(J):
    ax.plot([vt_q025[j], vt_q975[j]], [j, j], "k-", linewidth=2)
ax.set_yticks(y_pos); ax.set_yticklabels(person_pred_names, fontsize=8)
ax.set_xlabel("Effect on Ability (logits)")
ax.set_title("Person Property Effects (95% CI)", fontweight="bold")
ax.axvline(0, color="red", linestyle="--", alpha=0.5)

# (c) Posterior of % variance explained
ax = axes[1, 0]
ax.hist(pct_expl, bins=50, density=True, color="#F28E2B", alpha=0.7, edgecolor="white")
ax.axvline(pct_expl.mean(), color="red", linestyle="--",
           label=f"Mean = {pct_expl.mean():.1f}%")
ax.set_xlabel("% Variance Explained by Person Predictors")
ax.set_title("Posterior of Variance Explained", fontweight="bold")
ax.legend()

# (d) Person ability posteriors vs true
ax = axes[1, 1]
theta_total_mean = theta_total_draws.mean(axis=0)
r_tt = np.corrcoef(theta_total_mean, persons_df["theta_true"])[0, 1]
# Subsample for readability
idx_sub = np.random.choice(N, min(200, N), replace=False)
ax.scatter(persons_df["theta_true"].values[idx_sub], theta_total_mean[idx_sub],
           alpha=0.3, s=15, c="#4472C4")
lims = [-4, 4]
ax.plot(lims, lims, "r--", alpha=0.5)
ax.set_xlabel("True θ"); ax.set_ylabel("Posterior Mean θ_total")
ax.set_title(f"Recovery of Person Ability (r = {r_tt:.3f})", fontweight="bold")
ax.grid(True, alpha=0.2)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig_bayes_doubly_explanatory.png"), dpi=150, bbox_inches="tight")
print(f"\nFigure saved: fig_bayes_doubly_explanatory.png")
plt.close()

# ── 7. Save ───────────────────────────────────────────────────────────────
log_lik_draws = fit.stan_variable("log_lik")
log_lik_flat = log_lik_draws.reshape(log_lik_draws.shape[0], -1)
np.save(os.path.join(out_dir, "bayes_loglik_lat_reg_lltm.npy"), log_lik_flat)

# Save all effects
item_effects_df = pd.DataFrame({
    "cell": cell_names, "mean": np.round(bk_mean, 4),
    "sd": np.round(bk_sd, 4), "q025": np.round(bk_q025, 4), "q975": np.round(bk_q975, 4)
})
item_effects_df.to_csv(os.path.join(out_dir, "bayes_results_lat_reg_lltm_items.csv"), index=False)

person_effects_df = pd.DataFrame({
    "predictor": person_pred_names, "mean": np.round(vt_mean, 4),
    "sd": np.round(vt_sd, 4), "q025": np.round(vt_q025, 4), "q975": np.round(vt_q975, 4)
})
person_effects_df.to_csv(os.path.join(out_dir, "bayes_results_lat_reg_lltm_persons.csv"), index=False)

summary = {
    "model": "Latent Regression LLTM (Bayesian)",
    "sigma_e_mean": round(float(sigma_e_mean), 4),
    "var_resid_mean": round(float(var_resid_mean), 4),
    "pct_explained_mean": round(float(pct_expl.mean()), 1),
    "r_theta_recovery": round(float(r_tt), 4),
}
with open(os.path.join(out_dir, "bayes_summary_lat_reg_lltm.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("Results saved.")
