"""
05_lltm.py
==========
Model 3: The LLTM (Item Explanatory)

  η_pi = θ_p − Σ β_k X_ik

No scipy required.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss
import os

out_dir = os.path.dirname(os.path.abspath(__file__))

def expit(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def log_expit(x):
    return np.where(x >= 0, -np.log(1 + np.exp(-x)), x - np.log(1 + np.exp(x)))

def minimize_gd(func, x0, maxiter=400, lr=0.01, tol=1e-5):
    x = x0.copy(); best_f = func(x); best_x = x.copy(); eps = 1e-5
    for it in range(maxiter):
        f0 = func(x)
        grad = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy(); xp[i] += eps
            grad[i] = (func(xp) - f0) / eps
        step = lr
        for _ in range(8):
            xn = x - step * grad; fn = func(xn)
            if fn < f0: break
            step *= 0.5
        else: break
        x = xn
        if fn < best_f: best_f = fn; best_x = x.copy()
        if it % 100 == 0: print(f"  Iter {it}: neg_loglik = {fn:.2f}")
        if abs(f0 - fn) < tol: break
    return best_x, best_f

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

# ── 3. MML ────────────────────────────────────────────────────────────────
n_quad = 21
nodes, weights = hermgauss(n_quad)
weights = weights / np.sqrt(np.pi)

def neg_ll(params):
    beta_k = params[:K]; log_sigma = params[K]; sigma = np.exp(log_sigma)
    beta_pred = X_item @ beta_k
    theta_nodes = nodes * sigma * np.sqrt(2)
    eta = theta_nodes[:, None] - beta_pred[None, :]
    lp1 = log_expit(eta); lp0 = log_expit(-eta)
    log_lik_pq = (resp_matrix @ lp1.T) + ((1 - resp_matrix) @ lp0.T)
    mx = log_lik_pq.max(axis=1, keepdims=True)
    log_marg = mx.squeeze() + np.log((weights[None, :] * np.exp(log_lik_pq - mx)).sum(axis=1))
    return -log_marg.sum()

p_means = resp_matrix.mean(axis=0)
beta_cell_init = []
for k in range(K):
    mask = X_item[:, k] == 1
    p_cell = np.clip(p_means[mask].mean(), 0.01, 0.99)
    beta_cell_init.append(-np.log(p_cell / (1 - p_cell)))
params_init = np.array(beta_cell_init + [np.log(1.0)])

print("Fitting LLTM...")
best_params, best_nll = minimize_gd(neg_ll, params_init, maxiter=400, lr=0.02)

beta_k_est = best_params[:K]
sigma_est = np.exp(best_params[K])
variance_est = sigma_est ** 2
log_lik = -best_nll
beta_pred = X_item @ beta_k_est

# ── 4. Results ────────────────────────────────────────────────────────────
n_params = K + 1
deviance = -2 * log_lik
aic = deviance + 2 * n_params
bic = deviance + np.log(N) * n_params

print("\n" + "=" * 60)
print("LLTM RESULTS")
print("=" * 60)

print(f"\nItem Property Effects (β_k):")
print(f"{'Topic Area':<15} {'Modeling Type':<25} {'Effect':>8}")
print("-" * 50)
for k in range(K):
    parts = cell_names[k].split("_", 1)
    print(f"{parts[0]:<15} {parts[1]:<25} {beta_k_est[k]:>8.3f}")

print(f"\nPerson variance: {variance_est:.3f}")
print(f"\nFit: Deviance={deviance:.1f}, AIC={aic:.1f}, BIC={bic:.1f}")

# ── 5. Compare with Rasch ─────────────────────────────────────────────────
try:
    rasch_res = pd.read_csv(os.path.join(out_dir, "results_rasch.csv"))
    beta_rasch = rasch_res["beta_estimated"].values
    r = np.corrcoef(beta_pred, beta_rasch)[0, 1]
    print(f"\nCorrelation (LLTM pred vs Rasch est): r = {r:.3f}")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(beta_rasch, beta_pred, c="#4472C4", s=80, zorder=5, edgecolors="white")
    for i in range(I):
        ax.annotate(items_df["item_name"].iloc[i], (beta_rasch[i], beta_pred[i]),
                    xytext=(5, 5), textcoords="offset points", fontsize=7, color="gray")
    lims = [min(beta_rasch.min(), beta_pred.min()) - 0.2,
            max(beta_rasch.max(), beta_pred.max()) + 0.2]
    ax.plot(lims, lims, "r--", alpha=0.5, label="Perfect prediction")
    ax.set_xlabel("Rasch Estimates (β_i)"); ax.set_ylabel("LLTM Predictions (β'_i)")
    ax.set_title(f"Rasch vs LLTM (r = {r:.3f})", fontweight="bold")
    ax.legend(); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_rasch_vs_lltm.png"), dpi=150, bbox_inches="tight")
    print("Figure saved: fig_rasch_vs_lltm.png"); plt.close()
except FileNotFoundError:
    pass

# Save
fit_df = pd.DataFrame([{"model": "LLTM", "deviance": round(deviance, 1),
    "AIC": round(aic, 1), "BIC": round(bic, 1), "n_params": n_params,
    "person_variance": round(variance_est, 3)}])
fit_df.to_csv(os.path.join(out_dir, "results_fit_lltm.csv"), index=False)

lltm_effects = pd.DataFrame({"cell": cell_names, "beta_k": np.round(beta_k_est, 4)})
lltm_effects.to_csv(os.path.join(out_dir, "results_lltm_effects.csv"), index=False)
print("Results saved.")
