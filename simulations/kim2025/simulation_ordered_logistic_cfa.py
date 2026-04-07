import argparse
import time

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from cmdstanpy import CmdStanModel

from simulation_bsem import load_item_data, write_stan_model as write_pcm_stan_model


def write_ordered_logit_stan_model():
    """Write an ordered logistic CFA mediation model with free item thresholds."""
    stan_code = r"""
data {
  int<lower=1> N;
  int<lower=1> K_x;
  int<lower=1> K_m;
  int<lower=1> K_y;
  int<lower=1> C;
  array[N, K_x] int<lower=1, upper=4> x_items;
  array[N, K_m] int<lower=1, upper=5> m_items;
  array[N, K_y] int<lower=1, upper=5> y_items;
  matrix[N, C] covs;
}

parameters {
  vector[N] theta_x;
  vector[N] theta_m;
  vector[N] theta_y;

  real alpha_m;
  real alpha_y;
  real a;
  real b;
  real cp;
  vector[C] beta_m;
  vector[C] beta_y;
  real<lower=0> sigma_m;
  real<lower=0> sigma_y;

  vector<lower=0>[K_x - 1] lambda_x_free;
  vector<lower=0>[K_m - 1] lambda_m_free;
  vector<lower=0>[K_y - 1] lambda_y_free;

  array[K_x] ordered[3] kappa_x;
  array[K_m] ordered[4] kappa_m;
  array[K_y] ordered[4] kappa_y;
}

transformed parameters {
  vector[N] mu_m = alpha_m + a * theta_x + covs * beta_m;
  vector[N] mu_y = alpha_y + cp * theta_x + b * theta_m + covs * beta_y;
  vector[K_x] lambda_x;
  vector[K_m] lambda_m;
  vector[K_y] lambda_y;

  lambda_x[1] = 1;
  lambda_m[1] = 1;
  lambda_y[1] = 1;
  lambda_x[2:K_x] = lambda_x_free;
  lambda_m[2:K_m] = lambda_m_free;
  lambda_y[2:K_y] = lambda_y_free;
}

model {
  theta_x ~ std_normal();
  theta_m ~ normal(mu_m, sigma_m);
  theta_y ~ normal(mu_y, sigma_y);

  alpha_m ~ normal(0, 1);
  alpha_y ~ normal(0, 1);
  a ~ normal(0, 1);
  b ~ normal(0, 1);
  cp ~ normal(0, 1);
  beta_m ~ normal(0, 0.75);
  beta_y ~ normal(0, 0.75);
  sigma_m ~ exponential(1);
  sigma_y ~ exponential(1);

  lambda_x_free ~ lognormal(0, 0.35);
  lambda_m_free ~ lognormal(0, 0.35);
  lambda_y_free ~ lognormal(0, 0.35);

  for (j in 1:K_x) {
    kappa_x[j] ~ normal(0, 1.5);
  }
  for (j in 1:K_m) {
    kappa_m[j] ~ normal(0, 1.5);
  }
  for (j in 1:K_y) {
    kappa_y[j] ~ normal(0, 1.5);
  }

  for (n in 1:N) {
    for (j in 1:K_x) {
      x_items[n, j] ~ ordered_logistic(lambda_x[j] * theta_x[n], kappa_x[j]);
    }
    for (j in 1:K_m) {
      m_items[n, j] ~ ordered_logistic(lambda_m[j] * theta_m[n], kappa_m[j]);
    }
    for (j in 1:K_y) {
      y_items[n, j] ~ ordered_logistic(lambda_y[j] * theta_y[n], kappa_y[j]);
    }
  }
}

generated quantities {
  real indirect_effect = a * b;
  real total_effect = cp + a * b;
  vector[N] log_lik;

  for (n in 1:N) {
    real lp = 0;
    for (j in 1:K_x) {
      lp += ordered_logistic_lpmf(x_items[n, j] | lambda_x[j] * theta_x[n], kappa_x[j]);
    }
    for (j in 1:K_m) {
      lp += ordered_logistic_lpmf(m_items[n, j] | lambda_m[j] * theta_m[n], kappa_m[j]);
    }
    for (j in 1:K_y) {
      lp += ordered_logistic_lpmf(y_items[n, j] | lambda_y[j] * theta_y[n], kappa_y[j]);
    }
    log_lik[n] = lp;
  }
}
"""
    stan_file = "mediation_ordered_logit_cfa.stan"
    with open(stan_file, "w", encoding="utf-8") as file_handle:
        file_handle.write(stan_code)
    return stan_file


def build_stan_data():
    df_rses, df_iss, df_rssis, centered_covs, raw_cov_means = load_item_data()
    stan_data = {
        "N": len(df_rses),
        "K_x": df_rses.shape[1],
        "K_m": df_iss.shape[1],
        "K_y": df_rssis.shape[1],
        "C": centered_covs.shape[1],
        "x_items": df_rses.to_numpy(dtype=int),
        "m_items": df_iss.to_numpy(dtype=int),
        "y_items": df_rssis.to_numpy(dtype=int),
        "covs": centered_covs.to_numpy(dtype=float),
    }
    return stan_data, raw_cov_means


def fit_model(model_name, stan_file, stan_data, sample_kwargs):
    compile_start = time.perf_counter()
    model = CmdStanModel(stan_file=stan_file)
    compile_seconds = time.perf_counter() - compile_start

    sample_start = time.perf_counter()
    fit = model.sample(data=stan_data, **sample_kwargs)
    sample_seconds = time.perf_counter() - sample_start

    summary = fit.summary()
    key_rows = summary.loc[
        summary.index.intersection(["a", "b", "cp", "indirect_effect", "total_effect"])
    ]
    metrics = {
        "model": model_name,
        "compile_seconds": compile_seconds,
        "sample_seconds": sample_seconds,
        "total_seconds": compile_seconds + sample_seconds,
        "max_rhat": float(key_rows["R_hat"].max()),
        "min_ess_bulk": float(key_rows["ESS_bulk"].min()),
        "min_ess_tail": float(key_rows["ESS_tail"].min()),
    }
    idata = az.from_cmdstanpy(posterior=fit, log_likelihood="log_lik")

    loo_result = az.loo(idata)
    waic_result = az.waic(idata)
    metrics["loo_elpd"] = float(loo_result.elpd_loo)
    metrics["loo_p"] = float(loo_result.p_loo)
    metrics["waic_elpd"] = float(waic_result.elpd_waic)
    metrics["waic_p"] = float(waic_result.p_waic)

    return fit, idata, metrics


def summarize_fit(model_name, fit):
    summary = fit.summary()
    key_params = ["a", "b", "cp", "indirect_effect", "total_effect"]
    lower_label = "2.5%" if "2.5%" in summary.columns else "5%"
    upper_label = "97.5%" if "97.5%" in summary.columns else "95%"
    print(f"\n{model_name} posterior summary:")
    for param in key_params:
        if param not in summary.index:
            continue
        row = summary.loc[param]
        print(
            f"  {param:16s} mean={row['Mean']:8.3f} sd={row['StdDev']:7.3f} "
            f"{lower_label}={row[lower_label]:8.3f} {upper_label}={row[upper_label]:8.3f} Rhat={row['R_hat']:6.3f}"
        )


def save_ordered_logit_figures(idata):
    az.style.use("arviz-darkgrid")

    az.plot_trace(idata, var_names=["a", "b", "cp", "indirect_effect"], compact=True)
    plt.suptitle("Ordered Logistic CFA Trace Plots", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig("fig_ordered_logit_trace.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    az.plot_posterior(
        idata,
        var_names=["indirect_effect"],
        ref_val=0,
        hdi_prob=0.95,
        ax=ax,
        color="#8f5db7",
    )
    ax.set_title("Ordered Logistic CFA Posterior of Indirect Effect", fontsize=13)
    plt.tight_layout()
    plt.savefig("fig_ordered_logit_indirect.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    az.plot_forest(
        idata,
        var_names=["a", "b", "cp", "indirect_effect", "total_effect"],
        combined=True,
        hdi_prob=0.95,
        ax=ax,
    )
    ax.set_title("Ordered Logistic CFA Structural Effects", fontsize=13)
    plt.tight_layout()
    plt.savefig("fig_ordered_logit_forest.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_comparison_artifacts(metrics_rows):
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv("bayesian_item_model_benchmark.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    flat_axes = axes.ravel()
    colors = ["#8f5db7", "#3c8d5a"][: len(metrics_frame)]

    flat_axes[0].bar(metrics_frame["model"], metrics_frame["compile_seconds"], color=colors)
    flat_axes[0].set_title("Compile Time Comparison")
    flat_axes[0].set_ylabel("Seconds")

    flat_axes[1].bar(metrics_frame["model"], metrics_frame["sample_seconds"], color=colors)
    flat_axes[1].set_title("Sampling Time Comparison")
    flat_axes[1].set_ylabel("Seconds")

    flat_axes[2].bar(metrics_frame["model"], metrics_frame["loo_elpd"], color=colors)
    flat_axes[2].set_title("LOO elpd Comparison")
    flat_axes[2].set_ylabel("elpd")

    flat_axes[3].bar(metrics_frame["model"], metrics_frame["waic_elpd"], color=colors)
    flat_axes[3].set_title("WAIC elpd Comparison")
    flat_axes[3].set_ylabel("elpd")

    plt.tight_layout()
    plt.savefig("fig_bayesian_item_model_benchmark.png", dpi=150, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ordered logistic CFA mediation and optionally compare it with PCM BSEM."
    )
    parser.add_argument(
        "--mode",
        choices=["ordered-only", "compare"],
        default="compare",
        help="Run only the ordered logistic model or compare ordered logistic against the PCM model.",
    )
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--iter-warmup", type=int, default=1000)
    parser.add_argument("--iter-sampling", type=int, default=1000)
    parser.add_argument("--adapt-delta", type=float, default=0.95)
    parser.add_argument("--max-treedepth", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Ordered Logistic CFA Mediation")
    print("=" * 60)

    stan_data, raw_cov_means = build_stan_data()
    print("\nCentered covariate means used for preprocessing:")
    print(raw_cov_means.round(3))

    sample_kwargs = {
        "chains": args.chains,
        "parallel_chains": args.chains,
        "iter_warmup": args.iter_warmup,
        "iter_sampling": args.iter_sampling,
        "adapt_delta": args.adapt_delta,
        "max_treedepth": args.max_treedepth,
        "show_progress": True,
    }

    metrics_rows = []

    ordered_stan_file = write_ordered_logit_stan_model()
    ordered_fit, ordered_idata, ordered_metrics = fit_model(
        "ordered_logit_cfa",
        ordered_stan_file,
        stan_data,
        sample_kwargs,
    )
    metrics_rows.append(ordered_metrics)
    summarize_fit("Ordered logistic CFA", ordered_fit)
    save_ordered_logit_figures(ordered_idata)

    if args.mode == "compare":
        pcm_stan_file = write_pcm_stan_model()
        pcm_fit, _, pcm_metrics = fit_model(
            "pcm_bsem",
            pcm_stan_file,
            stan_data,
            sample_kwargs,
        )
        metrics_rows.append(pcm_metrics)
        summarize_fit("PCM BSEM", pcm_fit)

    save_comparison_artifacts(metrics_rows)

    print("\nSaved:")
    print("  mediation_ordered_logit_cfa.stan")
    print("  fig_ordered_logit_trace.png")
    print("  fig_ordered_logit_indirect.png")
    print("  fig_ordered_logit_forest.png")
    print("  bayesian_item_model_benchmark.csv")
    print("  fig_bayesian_item_model_benchmark.png")


if __name__ == "__main__":
    main()