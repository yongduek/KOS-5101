"""
03_rasch_model.py
=================
Model 1: The Rasch Model (Doubly Descriptive)

  η_pi = θ_p − β_i

Uses Marginal Maximum Likelihood with Gauss-Hermite quadrature.
No scipy required — optimization done with numpy only.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

out_dir = os.path.dirname(os.path.abspath(__file__))

# ── Logistic function ─────────────────────────────────────────────────────
def expit(x):
    """Stable logistic function."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

# ── Gauss-Hermite quadrature (numpy built-in) ────────────────────────────
from numpy.polynomial.hermite import hermgauss

# ── Simple L-BFGS-B replacement: gradient descent with line search ────────
def minimize_lbfgs(func, x0, maxiter=300, tol=1e-6, lr=0.01):
    """Simple gradient-based optimizer using finite differences."""
    x = x0.copy()
    best_f = func(x)
    best_x = x.copy()
    eps = 1e-5

    for iteration in range(maxiter):
        f0 = func(x)
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (func(x_plus) - f0) / eps

        # Adaptive learning rate
        step = lr
        for _ in range(10):
            x_new = x - step * grad
            f_new = func(x_new)
            if f_new < f0:
                break
            step *= 0.5
        else:
            break

        x = x_new
        if f_new < best_f:
            best_f = f_new
            best_x = x.copy()

        if iteration % 50 == 0:
            print(f"  Iter {iteration}: neg_loglik = {f_new:.2f}")

        if np.abs(f0 - f_new) < tol:
            break

    return best_x, best_f


# ── 1. Load data ──────────────────────────────────────────────────────────
items_df = pd.read_csv(os.path.join(out_dir, "data_items.csv"))
persons_df = pd.read_csv(os.path.join(out_dir, "data_persons.csv"))
resp_df = pd.read_csv(os.path.join(out_dir, "data_responses.csv"))
resp_matrix = resp_df.drop(columns=["person_id"]).values

N, I = resp_matrix.shape
print(f"Data: {N} persons × {I} items")

# ── 2. Fit Rasch Model via MML ────────────────────────────────────────────
n_quad = 21
nodes, weights = hermgauss(n_quad)
weights = weights / np.sqrt(np.pi)


def neg_log_marginal_likelihood(params):
    beta = params[:I]
    log_sigma = params[I]
    sigma = np.exp(log_sigma)

    theta_nodes = nodes * sigma * np.sqrt(2)

    eta = theta_nodes[:, None] - beta[None, :]  # (n_quad, I)
    log_p1 = np.where(eta >= 0,
                       -np.log(1 + np.exp(-eta)),
                       eta - np.log(1 + np.exp(eta)))
    log_p0 = np.where(-eta >= 0,
                       -np.log(1 + np.exp(eta)),
                       -eta - np.log(1 + np.exp(-eta)))

    log_lik_pq = (resp_matrix @ log_p1.T) + ((1 - resp_matrix) @ log_p0.T)

    max_ll = log_lik_pq.max(axis=1, keepdims=True)
    log_marg = max_ll.squeeze() + np.log(
        (weights[None, :] * np.exp(log_lik_pq - max_ll)).sum(axis=1)
    )

    return -log_marg.sum()


# Initial values from item p-values
p_vals = np.clip(resp_matrix.mean(axis=0), 0.01, 0.99)
beta_init = -np.log(p_vals / (1 - p_vals))
params_init = np.concatenate([beta_init, [np.log(1.0)]])

print("Fitting Rasch model (MML with Gauss-Hermite quadrature)...")
best_params, best_nll = minimize_lbfgs(neg_log_marginal_likelihood, params_init,
                                        maxiter=400, lr=0.02)

beta_est = best_params[:I]
sigma_est = np.exp(best_params[I])
variance_est = sigma_est ** 2
log_lik = -best_nll

# ── 3. Fit indices ────────────────────────────────────────────────────────
n_params = I + 1
deviance = -2 * log_lik
aic = deviance + 2 * n_params
bic = deviance + np.log(N) * n_params

print("\n" + "=" * 60)
print("RASCH MODEL RESULTS")
print("=" * 60)
print(f"\nEstimated person variance (σ²): {variance_est:.3f}")
print(f"Estimated person SD (σ):        {sigma_est:.3f}")

print(f"\nFit Indices:")
print(f"  Deviance: {deviance:.1f}")
print(f"  AIC:      {aic:.1f}")
print(f"  BIC:      {bic:.1f}")

print(f"\nItem Parameter Estimates:")
print(f"{'Item':<15} {'β_est':>8} {'β_true':>8} {'p-value':>8}")
print("-" * 42)
for i in range(I):
    print(f"{items_df['item_name'].iloc[i]:<15} {beta_est[i]:>8.3f} {items_df['beta_true'].iloc[i]:>8.3f} {expit(-beta_est[i]):>8.3f}")

r_items = np.corrcoef(beta_est, items_df["beta_true"])[0, 1]
print(f"\nCorrelation (est vs true β): r = {r_items:.3f}")

# ── 4. Person ability estimates (EAP) ─────────────────────────────────────
print("\nComputing EAP person ability estimates...")
theta_nodes_scaled = nodes * sigma_est * np.sqrt(2)

eta_all = theta_nodes_scaled[:, None] - beta_est[None, :]
log_p1 = np.where(eta_all >= 0,
                   -np.log(1 + np.exp(-eta_all)),
                   eta_all - np.log(1 + np.exp(eta_all)))
log_p0 = np.where(-eta_all >= 0,
                   -np.log(1 + np.exp(eta_all)),
                   -eta_all - np.log(1 + np.exp(-eta_all)))

log_lik_pq = (resp_matrix @ log_p1.T) + ((1 - resp_matrix) @ log_p0.T)

max_ll = log_lik_pq.max(axis=1, keepdims=True)
posterior = weights[None, :] * np.exp(log_lik_pq - max_ll)
posterior = posterior / posterior.sum(axis=1, keepdims=True)

theta_eap = (posterior * theta_nodes_scaled[None, :]).sum(axis=1)
theta_var = (posterior * (theta_nodes_scaled[None, :] ** 2)).sum(axis=1) - theta_eap ** 2

r_theta = np.corrcoef(theta_eap, persons_df["theta_true"])[0, 1]
print(f"Correlation (EAP θ vs true θ): r = {r_theta:.3f}")

reliability = 1 - theta_var.mean() / np.var(theta_eap)
print(f"EAP Reliability: {reliability:.3f}")

# ── 5. Wright Map ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

theta_bins = np.arange(-4, 4.1, 0.5)
counts, _ = np.histogram(theta_eap, bins=theta_bins)
max_count = max(counts.max(), 1)
for j in range(len(counts)):
    y_center = (theta_bins[j] + theta_bins[j + 1]) / 2
    bar_width = counts[j] / max_count * 3.5
    ax.barh(y_center, bar_width, height=0.4, left=-4, color="#4472C4", alpha=0.7, edgecolor="white")

colors_topic = {"Arithmetic": "#E15759", "Geometry": "#76B7B2", "Algebra": "#F28E2B"}
for i in range(I):
    row = items_df.iloc[i]
    color = colors_topic[row["topic_area"]]
    ax.plot(1.5, beta_est[i], "o", color=color, markersize=10, zorder=5)
    ax.annotate(row["item_name"], (1.5, beta_est[i]),
                xytext=(2.0, beta_est[i]), fontsize=7,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

ax.axvline(0, color="black", linewidth=0.5, linestyle="-")
ax.set_xlim(-4.5, 5.5)
ax.set_ylim(-3.5, 3.5)
ax.set_ylabel("Logit Scale", fontsize=12)
ax.set_title("Wright Map: Rasch Model", fontsize=14, fontweight="bold")

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#E15759", markersize=8, label="Arithmetic"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#76B7B2", markersize=8, label="Geometry"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#F28E2B", markersize=8, label="Algebra"),
]
ax.legend(handles=legend_elements, loc="upper right", title="Topic Area")
ax.text(-2.5, -3.2, "← Students", ha="center", fontsize=10, color="#4472C4")
ax.text(3.0, -3.2, "Items →", ha="center", fontsize=10, color="gray")

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig_wright_map.png"), dpi=150, bbox_inches="tight")
print(f"\nFigure saved: fig_wright_map.png")
plt.close()

# ── 6. Save results ───────────────────────────────────────────────────────
results_rasch = pd.DataFrame({
    "item_name": items_df["item_name"],
    "beta_estimated": np.round(beta_est, 4),
    "beta_true": np.round(items_df["beta_true"].values, 4),
})
results_rasch.to_csv(os.path.join(out_dir, "results_rasch.csv"), index=False)

persons_df["theta_eap"] = np.round(theta_eap, 4)
persons_df.to_csv(os.path.join(out_dir, "results_persons_rasch.csv"), index=False)

fit_df = pd.DataFrame([{
    "model": "Rasch",
    "deviance": round(deviance, 1),
    "AIC": round(aic, 1),
    "BIC": round(bic, 1),
    "n_params": n_params,
    "person_variance": round(variance_est, 3),
}])
fit_df.to_csv(os.path.join(out_dir, "results_fit_rasch.csv"), index=False)

print("\nResults saved.")
