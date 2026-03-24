"""
04_latent_regression_rasch.py
=============================
Model 2: The Latent Regression Rasch Model (Person Explanatory)

  η_pi = Σ ϑ_j Z_pj + θ_p − β_i

No scipy required.
"""

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
import os

out_dir = os.path.dirname(os.path.abspath(__file__))

def expit(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def log_expit(x):
    """Numerically stable log(sigmoid(x))."""
    return np.where(x >= 0, -np.log(1 + np.exp(-x)), x - np.log(1 + np.exp(x)))

def minimize_gd(func, x0, maxiter=400, lr=0.01, tol=1e-5):
    x = x0.copy()
    best_f = func(x)
    best_x = x.copy()
    eps = 1e-5
    for it in range(maxiter):
        f0 = func(x)
        grad = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy(); xp[i] += eps
            grad[i] = (func(xp) - f0) / eps
        step = lr
        for _ in range(8):
            xn = x - step * grad
            fn = func(xn)
            if fn < f0: break
            step *= 0.5
        else:
            break
        x = xn
        if fn < best_f: best_f = fn; best_x = x.copy()
        if it % 100 == 0: print(f"  Iter {it}: neg_loglik = {fn:.2f}")
        if abs(f0 - fn) < tol: break
    return best_x, best_f

# ── 1. Load data ──────────────────────────────────────────────────────────
items_df = pd.read_csv(os.path.join(out_dir, "data_items.csv"))
persons_df = pd.read_csv(os.path.join(out_dir, "data_persons.csv"))
resp_df = pd.read_csv(os.path.join(out_dir, "data_responses.csv"))
resp_matrix = resp_df.drop(columns=["person_id"]).values
N, I = resp_matrix.shape

# ── 2. Person predictors ──────────────────────────────────────────────────
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
predictor_names = ["Program 1 (Hauptschule)", "Program 3 (Realschule)",
                   "Program 4 (Gymnasium)", "Female × Prog 1",
                   "Female × Prog 3", "Female × Prog 4", "SES (HiSES)"]

# ── 3. MML ────────────────────────────────────────────────────────────────
n_quad = 21
nodes, weights = hermgauss(n_quad)
weights = weights / np.sqrt(np.pi)

def neg_ll(params):
    beta = params[:I]
    theta_coefs = params[I:I+J]
    log_sigma = params[I+J]
    sigma = np.exp(log_sigma)
    theta_fixed = Z @ theta_coefs
    theta_nodes = nodes * sigma * np.sqrt(2)

    log_lik_pq = np.zeros((N, n_quad))
    for q in range(n_quad):
        eta = theta_fixed[:, None] + theta_nodes[q] - beta[None, :]
        lp1 = log_expit(eta)
        lp0 = log_expit(-eta)
        log_lik_pq[:, q] = (resp_matrix * lp1 + (1 - resp_matrix) * lp0).sum(axis=1)

    mx = log_lik_pq.max(axis=1, keepdims=True)
    log_marg = mx.squeeze() + np.log((weights[None, :] * np.exp(log_lik_pq - mx)).sum(axis=1))
    return -log_marg.sum()

# Init
p_vals = np.clip(resp_matrix.mean(axis=0), 0.01, 0.99)
beta_init = -np.log(p_vals / (1 - p_vals))
params_init = np.concatenate([beta_init, np.zeros(J), [np.log(0.8)]])

print("Fitting Latent Regression Rasch model...")
best_params, best_nll = minimize_gd(neg_ll, params_init, maxiter=500, lr=0.015)

beta_est = best_params[:I]
theta_coefs = best_params[I:I+J]
sigma_resid = np.exp(best_params[I+J])
var_resid = sigma_resid ** 2
log_lik = -best_nll

# ── 4. Results ────────────────────────────────────────────────────────────
n_params = I + J + 1
deviance = -2 * log_lik
aic = deviance + 2 * n_params
bic = deviance + np.log(N) * n_params

print("\n" + "=" * 60)
print("LATENT REGRESSION RASCH MODEL RESULTS")
print("=" * 60)

print(f"\nPerson Property Effects:")
print(f"{'Predictor':<30} {'Effect':>8}")
print("-" * 40)
for j_idx in range(J):
    print(f"{predictor_names[j_idx]:<30} {theta_coefs[j_idx]:>8.3f}")

print(f"\nResidual person variance (σ²_ε): {var_resid:.3f}")

theta_fixed_all = Z @ theta_coefs
var_fixed = np.var(theta_fixed_all)
var_total = var_fixed + var_resid
pct = var_fixed / var_total * 100

print(f"\nVariance decomposition:")
print(f"  Explained: {var_fixed:.3f}")
print(f"  Residual:  {var_resid:.3f}")
print(f"  Total:     {var_total:.3f}")
print(f"  % expl:    {pct:.1f}%")

print(f"\nFit Indices:")
print(f"  Deviance: {deviance:.1f}")
print(f"  AIC:      {aic:.1f}")
print(f"  BIC:      {bic:.1f}")

try:
    fit_rasch = pd.read_csv(os.path.join(out_dir, "results_fit_rasch.csv"))
    print(f"\nComparison with Rasch Model:")
    for col in ["deviance", "AIC", "BIC"]:
        r_val = fit_rasch[col].values[0]
        lr_val = {"deviance": deviance, "AIC": aic, "BIC": bic}[col]
        print(f"  {col}: {r_val:.1f} → {lr_val:.1f} (Δ = {lr_val - r_val:.1f})")
except FileNotFoundError:
    pass

# Save
fit_df = pd.DataFrame([{"model": "Latent Regression Rasch", "deviance": round(deviance, 1),
    "AIC": round(aic, 1), "BIC": round(bic, 1), "n_params": n_params,
    "person_variance": round(var_resid, 3)}])
fit_df.to_csv(os.path.join(out_dir, "results_fit_lat_reg_rasch.csv"), index=False)

effects_df = pd.DataFrame({"predictor": predictor_names, "effect": np.round(theta_coefs, 4)})
effects_df.to_csv(os.path.join(out_dir, "results_person_effects.csv"), index=False)
print("\nResults saved.")
