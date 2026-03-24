"""
12_bayes_model_comparison.py
============================
Bayesian Model Comparison using LOO-CV (PSIS-LOO)

This script:
  1. Loads log-likelihood arrays from all four Bayesian models
  2. Computes LOO-CV (Leave-One-Out Cross-Validation) via PSIS
     using the arviz library
  3. Compares all models in a single table
  4. Creates comparison visualizations
  5. Also compares Bayesian posteriors with MML point estimates

Requires: cmdstanpy, arviz, numpy, pandas, matplotlib
          Run scripts 08-11 first to generate log-likelihood files.

Note: If arviz is not available, a simplified WAIC comparison is provided.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

out_dir = os.path.dirname(os.path.abspath(__file__))

# ── 1. Load log-likelihood arrays ─────────────────────────────────────────
print("=" * 60)
print("BAYESIAN MODEL COMPARISON")
print("=" * 60)

models = {
    "Rasch": "bayes_loglik_rasch.npy",
    "Lat. Reg. Rasch": "bayes_loglik_lat_reg_rasch.npy",
    "LLTM": "bayes_loglik_lltm.npy",
    "Lat. Reg. LLTM": "bayes_loglik_lat_reg_lltm.npy",
}

log_liks = {}
for name, fname in models.items():
    path = os.path.join(out_dir, fname)
    if os.path.exists(path):
        log_liks[name] = np.load(path)
        print(f"  Loaded {fname}: shape {log_liks[name].shape}")
    else:
        print(f"  ✗ {fname} not found — run the corresponding script first")

if len(log_liks) == 0:
    print("\nNo log-likelihood files found. Run scripts 08-11 first.")
    import sys; sys.exit(1)

# ── 2. Try arviz LOO-CV ──────────────────────────────────────────────────
use_arviz = False
try:
    import arviz as az
    use_arviz = True
    print(f"\n  Using arviz {az.__version__} for PSIS-LOO")
except ImportError:
    print("\n  arviz not available — falling back to WAIC approximation")

results = []

if use_arviz:
    # ── PSIS-LOO via arviz ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PSIS-LOO CROSS-VALIDATION")
    print("=" * 60)

    loo_results = {}
    for name, ll in log_liks.items():
        # arviz expects log_lik with shape (chains, draws_per_chain, n_obs)
        # or (draws, n_obs). We have (n_draws, N*I).
        # We need to reshape to (chains, draws_per_chain, N*I)
        n_draws = ll.shape[0]
        n_chains = 4
        draws_per_chain = n_draws // n_chains

        # Create an InferenceData-like dict
        ll_reshaped = ll[:draws_per_chain * n_chains].reshape(n_chains, draws_per_chain, -1)

        idata = az.from_dict(log_likelihood={"y": ll_reshaped})
        loo = az.loo(idata, pointwise=True)
        loo_results[name] = loo

        print(f"\n{name}:")
        print(f"  ELPD_LOO = {loo.elpd_loo:.1f} ± {loo.se:.1f}")
        print(f"  p_LOO (effective #params) = {loo.p_loo:.1f}")
        print(f"  LOO-IC = {-2 * loo.elpd_loo:.1f}")

        results.append({
            "Model": name,
            "ELPD_LOO": round(float(loo.elpd_loo), 1),
            "SE": round(float(loo.se), 1),
            "p_LOO": round(float(loo.p_loo), 1),
            "LOO_IC": round(float(-2 * loo.elpd_loo), 1),
        })

    # Pairwise comparison
    if len(loo_results) >= 2:
        print("\n" + "-" * 60)
        print("PAIRWISE LOO COMPARISON")
        print("-" * 60)
        comp = az.compare(loo_results, ic="loo")
        print(comp.to_string())

else:
    # ── WAIC approximation (no arviz) ────────────────────────────────────
    print("\n" + "=" * 60)
    print("WAIC APPROXIMATION")
    print("=" * 60)

    for name, ll in log_liks.items():
        # WAIC = -2 * (lppd - p_waic)
        # lppd = Σ_i log(mean_s exp(log_lik[s,i]))
        # p_waic = Σ_i var_s(log_lik[s,i])

        # Numerically stable lppd
        max_ll = ll.max(axis=0)
        lppd = np.sum(max_ll + np.log(np.mean(np.exp(ll - max_ll), axis=0)))

        # Effective number of parameters
        p_waic = np.sum(np.var(ll, axis=0))

        waic = -2 * (lppd - p_waic)

        print(f"\n{name}:")
        print(f"  lppd    = {lppd:.1f}")
        print(f"  p_WAIC  = {p_waic:.1f}")
        print(f"  WAIC    = {waic:.1f}")

        results.append({
            "Model": name,
            "lppd": round(float(lppd), 1),
            "p_WAIC": round(float(p_waic), 1),
            "WAIC": round(float(waic), 1),
        })

# ── 3. Summary table ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARISON TABLE")
print("=" * 60)
comp_df = pd.DataFrame(results)
print(comp_df.to_string(index=False))
comp_df.to_csv(os.path.join(out_dir, "bayes_model_comparison.csv"), index=False)

# ── 4. Compare Bayesian vs MML results ────────────────────────────────────
print("\n" + "=" * 60)
print("BAYESIAN vs MML COMPARISON")
print("=" * 60)

# Load MML fit indices
try:
    mml_comp = pd.read_csv(os.path.join(out_dir, "results_model_comparison.csv"))
    print("\nMML Results (from scripts 03-06):")
    print(mml_comp[["model", "deviance", "AIC", "BIC"]].to_string(index=False))
    print("\nBayesian Results:")
    print(comp_df.to_string(index=False))
    print("\nNote: MML uses information criteria (AIC/BIC), while Bayesian uses")
    print("      LOO-CV or WAIC — different approaches to the same goal of")
    print("      model comparison. Both should agree on model ranking.")
except FileNotFoundError:
    print("(MML results not found — run scripts 03-06 for comparison)")

# ── 5. Load and compare specific estimates ────────────────────────────────
print("\n" + "=" * 60)
print("PARAMETER RECOVERY: TRUE vs MML vs BAYESIAN")
print("=" * 60)

try:
    items_df = pd.read_csv(os.path.join(out_dir, "data_items.csv"))
    mml_rasch = pd.read_csv(os.path.join(out_dir, "results_rasch.csv"))
    bayes_rasch = pd.read_csv(os.path.join(out_dir, "bayes_results_rasch.csv"))

    true_beta = items_df["beta_true"].values
    mml_beta = mml_rasch["beta_estimated"].values
    bayes_beta = bayes_rasch["beta_mean"].values

    r_true_mml = np.corrcoef(true_beta, mml_beta)[0, 1]
    r_true_bayes = np.corrcoef(true_beta, bayes_beta)[0, 1]
    r_mml_bayes = np.corrcoef(mml_beta, bayes_beta)[0, 1]

    print(f"\nItem difficulty correlations (Rasch model):")
    print(f"  True vs MML:     r = {r_true_mml:.4f}")
    print(f"  True vs Bayesian: r = {r_true_bayes:.4f}")
    print(f"  MML vs Bayesian:  r = {r_mml_bayes:.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.scatter(true_beta, mml_beta, c="#4472C4", s=60, alpha=0.7, edgecolors="white")
    lims = [-2, 1.5]; ax.plot(lims, lims, "r--", alpha=0.5)
    ax.set_xlabel("True β"); ax.set_ylabel("MML β")
    ax.set_title(f"True vs MML (r={r_true_mml:.3f})", fontweight="bold")
    ax.grid(True, alpha=0.2); ax.set_aspect("equal")

    ax = axes[1]
    ax.scatter(true_beta, bayes_beta, c="#E15759", s=60, alpha=0.7, edgecolors="white")
    ax.plot(lims, lims, "r--", alpha=0.5)
    ax.set_xlabel("True β"); ax.set_ylabel("Bayesian β")
    ax.set_title(f"True vs Bayesian (r={r_true_bayes:.3f})", fontweight="bold")
    ax.grid(True, alpha=0.2); ax.set_aspect("equal")

    ax = axes[2]
    ax.scatter(mml_beta, bayes_beta, c="#76B7B2", s=60, alpha=0.7, edgecolors="white")
    ax.plot(lims, lims, "r--", alpha=0.5)
    ax.set_xlabel("MML β"); ax.set_ylabel("Bayesian β")
    ax.set_title(f"MML vs Bayesian (r={r_mml_bayes:.3f})", fontweight="bold")
    ax.grid(True, alpha=0.2); ax.set_aspect("equal")

    plt.suptitle("Item Parameter Recovery: True vs MML vs Bayesian",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_bayes_parameter_recovery.png"),
                dpi=150, bbox_inches="tight")
    print("\nFigure saved: fig_bayes_parameter_recovery.png")
    plt.close()

except FileNotFoundError as e:
    print(f"  Some files missing for comparison: {e}")

# ── 6. Final summary ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPLETE BAYESIAN ANALYSIS PIPELINE FINISHED!")
print("=" * 60)
print("""
Files generated by the Bayesian pipeline:

  Stan models:
    stan_models/rasch.stan
    stan_models/latent_regression_rasch.stan
    stan_models/lltm.stan
    stan_models/latent_regression_lltm.stan

  Results:
    bayes_results_rasch.csv            — Rasch item posterior summaries
    bayes_results_lat_reg_rasch.csv    — Person effect posteriors
    bayes_results_lltm.csv             — LLTM item property posteriors
    bayes_results_lat_reg_lltm_*.csv   — Doubly explanatory posteriors
    bayes_model_comparison.csv         — LOO/WAIC comparison table

  Figures:
    fig_bayes_wright_map.png           — Wright map with 95% CIs
    fig_bayes_rasch_diagnostics.png    — Trace plots & posterior density
    fig_bayes_lat_reg_rasch.png        — Person effect forest plots
    fig_bayes_rasch_vs_lltm.png        — LLTM vs Rasch comparison
    fig_bayes_icc_uncertainty.png      — ICCs with credible bands
    fig_bayes_doubly_explanatory.png   — Doubly explanatory results
    fig_bayes_parameter_recovery.png   — True vs MML vs Bayesian
""")
