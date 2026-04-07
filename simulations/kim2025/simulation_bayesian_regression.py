import argparse

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel


def load_total_score_data():
    """Load item responses, reverse-code as needed, and reconstruct scale totals."""
    df_rses = pd.read_csv("rses_simulated.csv")
    df_rssis = pd.read_csv("rssis_simulated.csv")
    df_iss = pd.read_csv("iss_simulated.csv")
    df_covs = pd.read_csv("covariates_simulated.csv")

    for col in df_rses.columns:
        if col.endswith("_REV"):
            df_rses[col] = 5 - df_rses[col]

    for col in df_iss.columns:
        if col.endswith("_REV"):
            df_iss[col] = 6 - df_iss[col]

    totals = pd.DataFrame(
        {
            "RSES_Total": df_rses.sum(axis=1).astype(float),
            "ISS_Total": df_iss.sum(axis=1).astype(float),
            "RSSIS_Total": df_rssis.sum(axis=1).astype(float),
            "Gender": df_covs["Gender"].astype(float),
            "Academic_Year": df_covs["Academic_Year"].astype(float),
            "TOPIK_Level": df_covs["TOPIK_Level"].astype(float),
            "Economic_Status": df_covs["Economic_Status"].astype(float),
        }
    )
    return totals


def write_stan_model():
    """Write a Bayesian regression mediation model that mirrors PROCESS Model 4."""
    stan_code = r"""
data {
  int<lower=1> N;
  int<lower=1> C;
  vector[N] X;
  vector[N] M;
  vector[N] Y;
  matrix[N, C] covs;
}
parameters {
  real alpha_m;
  real alpha_y;
  real a;
  real b;
  real cp;
  vector[C] beta_m;
  vector[C] beta_y;
  real<lower=0> sigma_m;
  real<lower=0> sigma_y;
}
transformed parameters {
  vector[N] mu_m = alpha_m + a * X + covs * beta_m;
  vector[N] mu_y = alpha_y + cp * X + b * M + covs * beta_y;
}
model {
  alpha_m ~ normal(0, 20);
  alpha_y ~ normal(0, 20);
  a ~ normal(0, 5);
  b ~ normal(0, 5);
  cp ~ normal(0, 5);
  beta_m ~ normal(0, 5);
  beta_y ~ normal(0, 5);
  sigma_m ~ exponential(1);
  sigma_y ~ exponential(1);

  M ~ normal(mu_m, sigma_m);
  Y ~ normal(mu_y, sigma_y);
}
generated quantities {
  real indirect_effect = a * b;
  real total_effect = cp + a * b;
  vector[N] log_lik_m;
  vector[N] log_lik_y;

  for (n in 1:N) {
    log_lik_m[n] = normal_lpdf(M[n] | mu_m[n], sigma_m);
    log_lik_y[n] = normal_lpdf(Y[n] | mu_y[n], sigma_y);
  }
}
"""
    stan_file = "mediation_bayesian_regression.stan"
    with open(stan_file, "w", encoding="utf-8") as file_handle:
        file_handle.write(stan_code)
    return stan_file


def summarize_fit(fit):
    summary = fit.summary()
    key_params = [
        "a",
        "b",
        "cp",
        "indirect_effect",
        "total_effect",
        "alpha_m",
        "alpha_y",
        "sigma_m",
        "sigma_y",
        "beta_m[1]",
        "beta_m[2]",
        "beta_m[3]",
        "beta_m[4]",
        "beta_y[1]",
        "beta_y[2]",
        "beta_y[3]",
        "beta_y[4]",
    ]

    lower_label = "2.5%" if "2.5%" in summary.columns else "5%"
    upper_label = "97.5%" if "97.5%" in summary.columns else "95%"

    print("\nPosterior summary:")
    for param in key_params:
        if param not in summary.index:
            continue
        row = summary.loc[param]
        print(
            f"  {param:18s} mean={row['Mean']:8.3f} sd={row['StdDev']:7.3f} "
            f"{lower_label}={row[lower_label]:8.3f} {upper_label}={row[upper_label]:8.3f} Rhat={row['R_hat']:6.3f}"
        )


def save_figures(idata):
    az.style.use("arviz-darkgrid")

    az.plot_trace(idata, var_names=["a", "b", "cp", "indirect_effect"], compact=True)
    plt.suptitle("Bayesian Regression Trace Plots", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig("fig_bayes_regression_trace.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    az.plot_posterior(
        idata,
        var_names=["indirect_effect"],
        ref_val=0,
        hdi_prob=0.95,
        ax=ax,
        color="#3d7ea6",
    )
    ax.set_title("Posterior of Indirect Effect (a x b)", fontsize=13)
    plt.tight_layout()
    plt.savefig("fig_bayes_regression_indirect.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    az.plot_forest(
        idata,
        var_names=["a", "b", "cp", "indirect_effect", "total_effect"],
        combined=True,
        hdi_prob=0.95,
        ax=ax,
    )
    ax.set_title("Bayesian Regression Structural Effects", fontsize=13)
    plt.tight_layout()
    plt.savefig("fig_bayes_regression_forest.png", dpi=150, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Bayesian regression mediation model for summed scale scores."
    )
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--iter-warmup", type=int, default=1000)
    parser.add_argument("--iter-sampling", type=int, default=1500)
    parser.add_argument("--adapt-delta", type=float, default=0.9)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Bayesian Regression Mediation (PROCESS Model 4 analogue)")
    print("=" * 60)

    data_frame = load_total_score_data()
    covariate_names = ["Gender", "Academic_Year", "TOPIK_Level", "Economic_Status"]
    stan_data = {
        "N": len(data_frame),
        "C": len(covariate_names),
        "X": data_frame["RSES_Total"].to_numpy(),
        "M": data_frame["ISS_Total"].to_numpy(),
        "Y": data_frame["RSSIS_Total"].to_numpy(),
        "covs": data_frame[covariate_names].to_numpy(),
    }

    print("\nObserved means:")
    print(data_frame[["RSES_Total", "ISS_Total", "RSSIS_Total"]].mean().round(3))
    print("\nObserved correlations:")
    print(data_frame[["RSES_Total", "ISS_Total", "RSSIS_Total"]].corr().round(3))

    stan_file = write_stan_model()
    model = CmdStanModel(stan_file=stan_file)

    fit = model.sample(
        data=stan_data,
        chains=args.chains,
        parallel_chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling,
        adapt_delta=args.adapt_delta,
        show_progress=True,
    )

    summarize_fit(fit)

    idata = az.from_cmdstanpy(posterior=fit)
    save_figures(idata)

    print("\nSaved:")
    print("  mediation_bayesian_regression.stan")
    print("  fig_bayes_regression_trace.png")
    print("  fig_bayes_regression_indirect.png")
    print("  fig_bayes_regression_forest.png")


if __name__ == "__main__":
    main()