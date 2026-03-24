"""
10_bayes_lltm.py
================
Bayesian Model 3: LLTM (Item Explanatory)

  η_pi = θ_p − Σ_k β_k X_ik

This script:
  1. Fits the LLTM with Bayesian inference via Stan
  2. Reports posterior distributions for item property effects
  3. Compares Bayesian LLTM predictions with Rasch item estimates
  4. Creates ICC curves with posterior uncertainty bands

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
resp_df = pd.read_csv(os.path.join(out_dir, "data_responses.csv"))
resp_matrix = resp_df.drop(columns=["person_id"]).values
N, I = resp_matrix.shape

# ── 2. Item predictor matrix ──────────────────────────────────────────────
topic_areas = ["Arithmetic", "Geometry", "Algebra"]
modeling_types = ["TechnicalProcessing", "NumericalModeling", "AbstractModeling"]

X_item = np.zeros((I, 9))
cell_names = []
for idx, (ta, mt) in enumerate([(ta, mt) for ta in topic_areas for mt in modeling_types]):
    cell_names.append(f"{ta}_{mt}")
    mask = (items_df["topic_area"] == ta) & (items_df["modeling_type"] == mt)
    X_item[mask.values, idx] = 1.0
K = 9

# ── 3. Stan data ──────────────────────────────────────────────────────────
stan_data = {
    "N": N, "I": I, "K": K,
    "Y": resp_matrix.tolist(),
    "X": X_item.tolist(),
}

# ── 4. Compile and sample ─────────────────────────────────────────────────
model = cmdstanpy.CmdStanModel(stan_file=os.path.join(stan_dir, "lltm.stan"))

print("Running MCMC for LLTM...")
fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    seed=42,
    show_console=False,
    adapt_delta=0.9,
)

print("\n" + "=" * 60)
print("MCMC DIAGNOSTICS")
print("=" * 60)
print(fit.diagnose())

# ── 5. Posterior summaries ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BAYESIAN LLTM — POSTERIORS")
print("=" * 60)

beta_k_draws = fit.stan_variable("beta_k")     # (n_draws, K)
sigma_draws = fit.stan_variable("sigma")        # (n_draws,)
beta_pred_draws = fit.stan_variable("beta_pred_gq")  # (n_draws, I)

bk_mean = beta_k_draws.mean(axis=0)
bk_sd = beta_k_draws.std(axis=0)
bk_q025 = np.percentile(beta_k_draws, 2.5, axis=0)
bk_q975 = np.percentile(beta_k_draws, 97.5, axis=0)

sigma_mean = sigma_draws.mean()
var_mean = (sigma_draws ** 2).mean()

print(f"\nPerson variance: {var_mean:.3f}")

print(f"\nItem Property Effects:")
print(f"{'Cell':<35} {'Mean':>8} {'SD':>8} {'2.5%':>8} {'97.5%':>8}")
print("-" * 72)
for k in range(K):
    print(f"{cell_names[k]:<35} {bk_mean[k]:>8.3f} {bk_sd[k]:>8.3f} "
          f"{bk_q025[k]:>8.3f} {bk_q975[k]:>8.3f}")

# ── 6. Compare predicted difficulties with Rasch estimates ────────────────
beta_pred_mean = beta_pred_draws.mean(axis=0)
beta_pred_q025 = np.percentile(beta_pred_draws, 2.5, axis=0)
beta_pred_q975 = np.percentile(beta_pred_draws, 97.5, axis=0)

try:
    rasch_results = pd.read_csv(os.path.join(out_dir, "bayes_results_rasch.csv"))
    beta_rasch = rasch_results["beta_mean"].values
    r = np.corrcoef(beta_pred_mean, beta_rasch)[0, 1]
    print(f"\nCorrelation (LLTM pred vs Bayesian Rasch): r = {r:.3f}")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(beta_rasch, beta_pred_mean, c="#4472C4", s=80, zorder=5, edgecolors="white")
    # Add uncertainty bars (LLTM predictions)
    for i in range(I):
        ax.plot([beta_rasch[i], beta_rasch[i]],
                [beta_pred_q025[i], beta_pred_q975[i]],
                "-", color="#4472C4", alpha=0.3, linewidth=2)
        ax.annotate(items_df["item_name"].iloc[i],
                    (beta_rasch[i], beta_pred_mean[i]),
                    xytext=(5, 5), textcoords="offset points", fontsize=6, color="gray")

    lims = [min(beta_rasch.min(), beta_pred_mean.min()) - 0.3,
            max(beta_rasch.max(), beta_pred_mean.max()) + 0.3]
    ax.plot(lims, lims, "r--", alpha=0.5, label="Perfect prediction")
    ax.set_xlabel("Bayesian Rasch β (posterior mean)")
    ax.set_ylabel("Bayesian LLTM β' (posterior mean)")
    ax.set_title(f"Rasch vs LLTM — Bayesian (r = {r:.3f})", fontweight="bold")
    ax.legend(); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_bayes_rasch_vs_lltm.png"), dpi=150, bbox_inches="tight")
    print("Figure saved: fig_bayes_rasch_vs_lltm.png"); plt.close()
except FileNotFoundError:
    print("\n(Run 08_bayes_rasch.py first for comparison)")

# ── 7. ICC curves with posterior uncertainty ──────────────────────────────
def expit(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
theta_range = np.linspace(-3, 3, 100)

for ax_idx, ta in enumerate(topic_areas):
    ax = axes[ax_idx]
    ta_mask = items_df["topic_area"] == ta

    for i in range(I):
        if not ta_mask.iloc[i]:
            continue
        mt = items_df["modeling_type"].iloc[i]
        style = {"TechnicalProcessing": "-", "NumericalModeling": "--", "AbstractModeling": ":"}[mt]
        color = {"TechnicalProcessing": "#4472C4", "NumericalModeling": "#E15759",
                 "AbstractModeling": "#76B7B2"}[mt]

        # Mean ICC
        p_mean = expit(theta_range - beta_pred_mean[i])
        ax.plot(theta_range, p_mean, style, color=color, linewidth=2, alpha=0.8)

        # 95% CI band (from posterior draws — use a subsample for speed)
        n_sub = min(200, beta_pred_draws.shape[0])
        p_draws = np.zeros((n_sub, len(theta_range)))
        idx_sub = np.random.choice(beta_pred_draws.shape[0], n_sub, replace=False)
        for s, d_idx in enumerate(idx_sub):
            p_draws[s] = expit(theta_range - beta_pred_draws[d_idx, i])
        p_lo = np.percentile(p_draws, 2.5, axis=0)
        p_hi = np.percentile(p_draws, 97.5, axis=0)
        ax.fill_between(theta_range, p_lo, p_hi, color=color, alpha=0.1)

    ax.set_xlabel("Person Ability (θ)")
    ax.set_title(ta, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.grid(True, alpha=0.2)

axes[0].set_ylabel("P(correct)")
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="#4472C4", linestyle="-", label="Technical"),
    Line2D([0], [0], color="#E15759", linestyle="--", label="Numerical"),
    Line2D([0], [0], color="#76B7B2", linestyle=":", label="Abstract"),
]
axes[2].legend(handles=legend_elements, loc="lower right", fontsize=8)
plt.suptitle("Bayesian ICCs with 95% Credible Bands", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig_bayes_icc_uncertainty.png"), dpi=150, bbox_inches="tight")
print("Figure saved: fig_bayes_icc_uncertainty.png"); plt.close()

# ── 8. Save ───────────────────────────────────────────────────────────────
log_lik_draws = fit.stan_variable("log_lik")
log_lik_flat = log_lik_draws.reshape(log_lik_draws.shape[0], -1)
np.save(os.path.join(out_dir, "bayes_loglik_lltm.npy"), log_lik_flat)

effects_df = pd.DataFrame({
    "cell": cell_names,
    "mean": np.round(bk_mean, 4),
    "sd": np.round(bk_sd, 4),
    "q025": np.round(bk_q025, 4),
    "q975": np.round(bk_q975, 4),
})
effects_df.to_csv(os.path.join(out_dir, "bayes_results_lltm.csv"), index=False)

summary = {
    "model": "LLTM (Bayesian)",
    "sigma_mean": round(float(sigma_mean), 4),
    "variance_mean": round(float(var_mean), 4),
}
with open(os.path.join(out_dir, "bayes_summary_lltm.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("Results saved.")
