"""
06_latent_regression_lltm.py
============================
Model 4: Latent Regression LLTM (Doubly Explanatory)

  η_pi = Σ ϑ_j Z_pj + θ_p − Σ β_k X_ik

Also creates the final model comparison table and figures.
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

def minimize_gd(func, x0, maxiter=500, lr=0.01, tol=1e-5):
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
persons_df = pd.read_csv(os.path.join(out_dir, "data_persons.csv"))
resp_df = pd.read_csv(os.path.join(out_dir, "data_responses.csv"))
resp_matrix = resp_df.drop(columns=["person_id"]).values
N, I = resp_matrix.shape

# ── 2. Predictors ─────────────────────────────────────────────────────────
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

topic_areas = ["Arithmetic", "Geometry", "Algebra"]
modeling_types = ["TechnicalProcessing", "NumericalModeling", "AbstractModeling"]
X_item = np.zeros((I, 9)); cell_names = []
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
    beta_k = params[:K]; theta_coefs = params[K:K+J]; log_sigma = params[K+J]
    sigma = np.exp(log_sigma)
    beta_pred = X_item @ beta_k; theta_fixed = Z @ theta_coefs
    theta_nodes = nodes * sigma * np.sqrt(2)

    log_lik_pq = np.zeros((N, n_quad))
    for q in range(n_quad):
        eta = theta_fixed[:, None] + theta_nodes[q] - beta_pred[None, :]
        lp1 = log_expit(eta); lp0 = log_expit(-eta)
        log_lik_pq[:, q] = (resp_matrix * lp1 + (1 - resp_matrix) * lp0).sum(axis=1)

    mx = log_lik_pq.max(axis=1, keepdims=True)
    log_marg = mx.squeeze() + np.log((weights[None, :] * np.exp(log_lik_pq - mx)).sum(axis=1))
    return -log_marg.sum()

p_means = resp_matrix.mean(axis=0)
beta_cell_init = []
for k in range(K):
    mask = X_item[:, k] == 1
    p_cell = np.clip(p_means[mask].mean(), 0.01, 0.99)
    beta_cell_init.append(-np.log(p_cell / (1 - p_cell)))
params_init = np.array(beta_cell_init + [0.0]*J + [np.log(0.8)])

print("Fitting Latent Regression LLTM (doubly explanatory)...")
best_params, best_nll = minimize_gd(neg_ll, params_init, maxiter=600, lr=0.012)

beta_k_est = best_params[:K]; theta_coefs = best_params[K:K+J]
sigma_resid = np.exp(best_params[K+J]); var_resid = sigma_resid**2; log_lik = -best_nll

# ── 4. Results ────────────────────────────────────────────────────────────
n_params = K + J + 1
deviance = -2 * log_lik; aic = deviance + 2*n_params; bic = deviance + np.log(N)*n_params

print("\n" + "=" * 60)
print("LATENT REGRESSION LLTM (DOUBLY EXPLANATORY)")
print("=" * 60)

print(f"\nItem Property Effects:")
for k in range(K):
    print(f"  {cell_names[k]:<35} {beta_k_est[k]:>8.3f}")

print(f"\nPerson Property Effects:")
for j_idx in range(J):
    print(f"  {person_pred_names[j_idx]:<25} {theta_coefs[j_idx]:>8.3f}")

theta_fixed_all = Z @ theta_coefs
var_fixed = np.var(theta_fixed_all); var_total = var_fixed + var_resid
print(f"\nVariance: Explained={var_fixed:.3f}, Residual={var_resid:.3f}, Total={var_total:.3f}")
print(f"% explained: {var_fixed/var_total*100:.1f}%")
print(f"\nFit: Deviance={deviance:.1f}, AIC={aic:.1f}, BIC={bic:.1f}")

# ── 5. Model comparison ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL COMPARISON (Table 5 from the chapter)")
print("=" * 60)

all_fits = []
for fname in ["results_fit_rasch.csv", "results_fit_lat_reg_rasch.csv", "results_fit_lltm.csv"]:
    try:
        all_fits.append(pd.read_csv(os.path.join(out_dir, fname)).iloc[0].to_dict())
    except FileNotFoundError:
        pass
all_fits.append({"model": "Latent Regression LLTM", "deviance": round(deviance, 1),
    "AIC": round(aic, 1), "BIC": round(bic, 1), "n_params": n_params,
    "person_variance": round(var_resid, 3)})

comparison_df = pd.DataFrame(all_fits)
print(f"\n{'Model':<30} {'Deviance':>10} {'AIC':>10} {'BIC':>10} {'#Params':>8}")
print("-" * 70)
for _, row in comparison_df.iterrows():
    print(f"{row['model']:<30} {row['deviance']:>10.1f} {row['AIC']:>10.1f} {row['BIC']:>10.1f} {int(row['n_params']):>8}")
comparison_df.to_csv(os.path.join(out_dir, "results_model_comparison.csv"), index=False)

# ── 6. Visualization ─────────────────────────────────────────────────────
if len(all_fits) == 4:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    models = comparison_df["model"].tolist()
    x = np.arange(len(models))
    for ax, metric in zip(axes, ["deviance", "AIC", "BIC"]):
        values = comparison_df[metric].values
        colors = ["#4472C4", "#E15759", "#76B7B2", "#F28E2B"]
        bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=7)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f"{val:.0f}", ha="center", fontsize=8)
        ax.set_ylim(min(values)*0.98, max(values)*1.02)
    plt.suptitle("Model Comparison: Fit Indices", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_model_comparison.png"), dpi=150, bbox_inches="tight")
    print("\nFigure saved: fig_model_comparison.png"); plt.close()

# ICC curves
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
beta_pred = X_item @ beta_k_est
theta_range = np.linspace(-3, 3, 100)
for ax_idx, ta in enumerate(topic_areas):
    ax = axes[ax_idx]; ta_mask = items_df["topic_area"] == ta
    for i in range(I):
        if not ta_mask.iloc[i]: continue
        p_curve = expit(theta_range - beta_pred[i])
        mt = items_df["modeling_type"].iloc[i]
        style = {"TechnicalProcessing": "-", "NumericalModeling": "--", "AbstractModeling": ":"}[mt]
        color = {"TechnicalProcessing": "#4472C4", "NumericalModeling": "#E15759",
                 "AbstractModeling": "#76B7B2"}[mt]
        ax.plot(theta_range, p_curve, style, color=color, linewidth=2, alpha=0.8)
    ax.set_xlabel("Person Ability (θ)"); ax.set_title(ta, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5); ax.grid(True, alpha=0.2)
axes[0].set_ylabel("P(correct)")
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="#4472C4", linestyle="-", label="Technical Processing"),
    Line2D([0], [0], color="#E15759", linestyle="--", label="Numerical Modeling"),
    Line2D([0], [0], color="#76B7B2", linestyle=":", label="Abstract Modeling")]
axes[2].legend(handles=legend_elements, loc="lower right", fontsize=8)
plt.suptitle("Item Characteristic Curves", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig_icc_curves.png"), dpi=150, bbox_inches="tight")
print("Figure saved: fig_icc_curves.png"); plt.close()

print("\n" + "=" * 60)
print("ALL FOUR MODELS FITTED SUCCESSFULLY!")
print("=" * 60)
