"""
08_bayes_rasch.py
=================
Bayesian Model 1: The Rasch Model (Doubly Descriptive)

  η_pi = θ_p − β_i
  θ_p ~ Normal(0, σ²),  σ ~ Half-Cauchy(0, 2.5)
  β_i ~ Normal(0, 25)

This script:
  1. Prepares data for Stan
  2. Runs MCMC sampling (4 chains)
  3. Reports posterior summaries for item parameters, person variance
  4. Computes posterior predictive checks
  5. Creates a Bayesian Wright map with credible intervals
  6. Saves results for later model comparison

Requires: cmdstanpy, numpy, pandas, matplotlib
          Run 01_generate_data.py and 07_bayes_setup.py first.
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
print(f"Data: {N} persons × {I} items")

# ── 2. Prepare Stan data ──────────────────────────────────────────────────
stan_data = {
    "N": N,
    "I": I,
    "Y": resp_matrix.tolist(),
}

# ── 3. Compile and sample ─────────────────────────────────────────────────
model = cmdstanpy.CmdStanModel(stan_file=os.path.join(stan_dir, "rasch.stan"))

print("Running MCMC sampling (4 chains × 1000 warmup + 1000 sampling)...")
fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    seed=42,
    show_console=False,
    adapt_delta=0.9,
)

# ── 4. Diagnostics ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MCMC DIAGNOSTICS")
print("=" * 60)
diag = fit.diagnose()
print(diag)

# ── 5. Posterior summaries ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BAYESIAN RASCH MODEL — POSTERIOR SUMMARIES")
print("=" * 60)

# Item parameters
beta_summary = fit.summary(sig_figs=4)
beta_rows = [f"beta[{i+1}]" for i in range(I)]
beta_sub = beta_summary.loc[beta_rows] if all(b in beta_summary.index for b in beta_rows) else None

# Extract posterior draws
beta_draws = fit.stan_variable("beta")       # shape (n_draws, I)
theta_draws = fit.stan_variable("theta")     # shape (n_draws, N)
sigma_draws = fit.stan_variable("sigma")     # shape (n_draws,)

beta_mean = beta_draws.mean(axis=0)
beta_sd = beta_draws.std(axis=0)
beta_q025 = np.percentile(beta_draws, 2.5, axis=0)
beta_q975 = np.percentile(beta_draws, 97.5, axis=0)

sigma_mean = sigma_draws.mean()
sigma_sd = sigma_draws.std()
var_mean = (sigma_draws ** 2).mean()

print(f"\nPerson ability SD (σ): {sigma_mean:.3f} ± {sigma_sd:.3f}")
print(f"Person variance (σ²):  {var_mean:.3f}")

print(f"\nItem Parameter Posteriors:")
print(f"{'Item':<15} {'Mean':>8} {'SD':>8} {'2.5%':>8} {'97.5%':>8} {'True':>8}")
print("-" * 58)
for i in range(I):
    print(f"{items_df['item_name'].iloc[i]:<15} "
          f"{beta_mean[i]:>8.3f} {beta_sd[i]:>8.3f} "
          f"{beta_q025[i]:>8.3f} {beta_q975[i]:>8.3f} "
          f"{items_df['beta_true'].iloc[i]:>8.3f}")

# Correlation with true values
r_beta = np.corrcoef(beta_mean, items_df["beta_true"])[0, 1]
print(f"\nCorrelation (posterior mean β vs true β): r = {r_beta:.3f}")

# Person ability posteriors
theta_mean = theta_draws.mean(axis=0)
r_theta = np.corrcoef(theta_mean, persons_df["theta_true"])[0, 1]
print(f"Correlation (posterior mean θ vs true θ): r = {r_theta:.3f}")

# ── 6. Bayesian Wright Map with credible intervals ────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Person histogram (left side)
theta_bins = np.arange(-4, 4.1, 0.5)
counts, _ = np.histogram(theta_mean, bins=theta_bins)
max_count = max(counts.max(), 1)
for j in range(len(counts)):
    y_center = (theta_bins[j] + theta_bins[j + 1]) / 2
    bar_width = counts[j] / max_count * 3.5
    ax.barh(y_center, bar_width, height=0.4, left=-4, color="#4472C4", alpha=0.7, edgecolor="white")

# Item locations with 95% credible intervals (right side)
colors_topic = {"Arithmetic": "#E15759", "Geometry": "#76B7B2", "Algebra": "#F28E2B"}
for i in range(I):
    row = items_df.iloc[i]
    color = colors_topic[row["topic_area"]]
    ax.plot(1.5, beta_mean[i], "o", color=color, markersize=9, zorder=5)
    # 95% CI bar
    ax.plot([1.5, 1.5], [beta_q025[i], beta_q975[i]], "-", color=color, linewidth=2, alpha=0.6)
    ax.annotate(row["item_name"], (1.5, beta_mean[i]),
                xytext=(2.0, beta_mean[i]), fontsize=6,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlim(-4.5, 5.5); ax.set_ylim(-3.5, 3.5)
ax.set_ylabel("Logit Scale"); ax.set_title("Bayesian Wright Map (Rasch) with 95% CIs", fontweight="bold")
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#E15759", markersize=8, label="Arithmetic"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#76B7B2", markersize=8, label="Geometry"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#F28E2B", markersize=8, label="Algebra"),
]
ax.legend(handles=legend_elements, loc="upper right", title="Topic Area")

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig_bayes_wright_map.png"), dpi=150, bbox_inches="tight")
print(f"\nFigure saved: fig_bayes_wright_map.png")
plt.close()

# ── 7. Posterior density for σ ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.hist(sigma_draws, bins=50, density=True, color="#4472C4", alpha=0.7, edgecolor="white")
ax.axvline(sigma_mean, color="red", linestyle="--", label=f"Mean = {sigma_mean:.3f}")
ax.set_xlabel("σ (Person SD)"); ax.set_ylabel("Density")
ax.set_title("Posterior of Person SD (σ)", fontweight="bold"); ax.legend()

ax = axes[1]
# Trace plot for sigma (first chain)
n_samples = sigma_draws.shape[0] // 4  # per chain
for c in range(4):
    ax.plot(sigma_draws[c * n_samples:(c + 1) * n_samples], alpha=0.5, linewidth=0.5, label=f"Chain {c+1}")
ax.set_xlabel("Iteration"); ax.set_ylabel("σ")
ax.set_title("Trace Plot for σ", fontweight="bold"); ax.legend(fontsize=7)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig_bayes_rasch_diagnostics.png"), dpi=150, bbox_inches="tight")
print("Figure saved: fig_bayes_rasch_diagnostics.png")
plt.close()

# ── 8. Save results ───────────────────────────────────────────────────────
# Save log-likelihood for LOO-CV later
log_lik_draws = fit.stan_variable("log_lik")  # shape (n_draws, N, I)
# Reshape to (n_draws, N*I) for arviz
log_lik_flat = log_lik_draws.reshape(log_lik_draws.shape[0], -1)
np.save(os.path.join(out_dir, "bayes_loglik_rasch.npy"), log_lik_flat)

# Save posterior summaries
results = pd.DataFrame({
    "item_name": items_df["item_name"],
    "beta_mean": np.round(beta_mean, 4),
    "beta_sd": np.round(beta_sd, 4),
    "beta_q025": np.round(beta_q025, 4),
    "beta_q975": np.round(beta_q975, 4),
    "beta_true": np.round(items_df["beta_true"].values, 4),
})
results.to_csv(os.path.join(out_dir, "bayes_results_rasch.csv"), index=False)

# Save scalar summaries
scalar_summary = {
    "model": "Rasch (Bayesian)",
    "sigma_mean": round(float(sigma_mean), 4),
    "sigma_sd": round(float(sigma_sd), 4),
    "variance_mean": round(float(var_mean), 4),
    "r_beta": round(float(r_beta), 4),
    "r_theta": round(float(r_theta), 4),
}
with open(os.path.join(out_dir, "bayes_summary_rasch.json"), "w") as f:
    json.dump(scalar_summary, f, indent=2)

print("\nResults saved: bayes_results_rasch.csv, bayes_summary_rasch.json, bayes_loglik_rasch.npy")
