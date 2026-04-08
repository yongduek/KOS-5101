import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel


def load_item_data():
        """Load ordinal item responses, reverse-code, and build centered covariates."""
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

        covariates = df_covs[["Gender", "Academic_Year", "TOPIK_Level", "Economic_Status"]].astype(float)
        centered_covariates = covariates - covariates.mean(axis=0)
        centered_covariates.columns = ["gender_c", "year_c", "topik_c", "eco_c"]

        return df_rses.astype(int), df_iss.astype(int), df_rssis.astype(int), centered_covariates, covariates.mean(axis=0)


def write_stan_model():
        """Write Stan code for Graded Response Model. Only recompile if changed."""
        stan_code = r"""
functions {
    real grm_item_lpmf(int y, real theta, vector thresholds) {
        int n_cat = num_elements(thresholds) + 1;
        vector[n_cat + 1] cum_prob;
        
        // Cumulative probabilities: P(Y >= k)
        cum_prob[1] = 1.0;
        for (k in 2:n_cat) {
            cum_prob[k] = inv_logit(thresholds[k - 1] - theta);
        }
        cum_prob[n_cat + 1] = 0.0;
        
        // P(Y = y) = P(Y >= y) - P(Y >= y+1)
        return log(cum_prob[y] - cum_prob[y + 1]);
    }
}
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

    array[K_x] vector[3] threshold_x;
    array[K_m] vector[4] threshold_m;
    array[K_y] vector[4] threshold_y;
}
transformed parameters {
    vector[N] mu_m = alpha_m + a * theta_x + covs * beta_m;
    vector[N] mu_y = alpha_y + cp * theta_x + b * theta_m + covs * beta_y;
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

    for (j in 1:K_x) {
        threshold_x[j] ~ normal(0, 1.5);
    }
    for (j in 1:K_m) {
        threshold_m[j] ~ normal(0, 1.5);
    }
    for (j in 1:K_y) {
        threshold_y[j] ~ normal(0, 1.5);
    }

    for (n in 1:N) {
        for (j in 1:K_x) {
            target += grm_item_lpmf(x_items[n, j] | theta_x[n], to_vector(threshold_x[j]));
        }
        for (j in 1:K_m) {
            target += grm_item_lpmf(m_items[n, j] | theta_m[n], to_vector(threshold_m[j]));
        }
        for (j in 1:K_y) {
            target += grm_item_lpmf(y_items[n, j] | theta_y[n], to_vector(threshold_y[j]));
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
            lp += grm_item_lpmf(x_items[n, j] | theta_x[n], to_vector(threshold_x[j]));
        }
        for (j in 1:K_m) {
            lp += grm_item_lpmf(m_items[n, j] | theta_m[n], to_vector(threshold_m[j]));
        }
        for (j in 1:K_y) {
            lp += grm_item_lpmf(y_items[n, j] | theta_y[n], to_vector(threshold_y[j]));
        }
        log_lik[n] = lp;
    }
}
"""
        stan_name = "mediation_bsem_grm.stan"
        stan_file = Path(stan_name)
        if stan_file.exists():
                existing_code = stan_file.read_text(encoding="utf-8")
                if existing_code == stan_code:
                        return str(stan_file), False

        stan_file.write_text(stan_code, encoding="utf-8")
        return str(stan_file), True


def summarize_fit(fit):
        summary = fit.summary()
        structural_params = [
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

        print("\nPosterior summary for latent structural parameters:")
        for param in structural_params:
                if param not in summary.index:
                        continue
                row = summary.loc[param]
                print(
                        f"  {param:18s} mean={row['Mean']:8.3f} sd={row['StdDev']:7.3f} "
                        f"{lower_label}={row[lower_label]:8.3f} {upper_label}={row[upper_label]:8.3f} Rhat={row['R_hat']:6.3f}"
                )
        return summary


def save_figures(idata):
        az.style.use("arviz-darkgrid")

        az.plot_trace(idata, var_names=["a", "b", "cp", "indirect_effect"], compact=True)
        plt.suptitle("Item-level GRM BSEM Trace Plots", y=1.02, fontsize=14)
        plt.tight_layout()
        plt.savefig("fig_bsem_grm_trace.png", dpi=150, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 5))
        az.plot_posterior(
                idata,
                var_names=["indirect_effect"],
                ref_val=0,
                hdi_prob=0.95,
                ax=ax,
                color="#3c8d5a",
        )
        ax.set_title("GRM BSEM Posterior of Indirect Effect", fontsize=13)
        plt.tight_layout()
        plt.savefig("fig_bsem_grm_indirect.png", dpi=150, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        az.plot_forest(
                idata,
                var_names=["a", "b", "cp", "indirect_effect", "total_effect"],
                combined=True,
                hdi_prob=0.95,
                ax=ax,
        )
        ax.set_title("GRM BSEM Structural Effects", fontsize=13)
        plt.tight_layout()
        plt.savefig("fig_bsem_grm_forest.png", dpi=150, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        az.plot_forest(
                idata,
                var_names=["beta_m", "beta_y"],
                combined=True,
                hdi_prob=0.95,
                ax=ax,
        )
        ax.set_title("Centered Covariate Effects on Latent M and Y", fontsize=12)
        plt.tight_layout()
        plt.savefig("fig_bsem_grm_covariates.png", dpi=150, bbox_inches="tight")
        plt.close()


def parse_args():
        parser = argparse.ArgumentParser(
                description="Run the item-level GRM Bayesian SEM mediation model."
        )
        parser.add_argument("--chains", type=int, default=4)
        parser.add_argument("--iter-warmup", type=int, default=1000)
        parser.add_argument("--iter-sampling", type=int, default=1000)
        parser.add_argument("--adapt-delta", type=float, default=0.95)
        parser.add_argument("--max-treedepth", type=int, default=12)
        parser.add_argument("--output-dir", type=str, default="bsem_outputs_grm")
        parser.add_argument("--posterior-nc", type=str, default="bsem_grm_posterior.nc")
        parser.add_argument("--summary-csv", type=str, default="bsem_grm_summary.csv")
        return parser.parse_args()


def ensure_windows_cmdstan_shell_tools():
        """Ensure cut/expr are available for CmdStan make scripts on Windows."""
        if os.name != "nt":
                return

        if shutil.which("cut") and shutil.which("expr"):
                return

        git_usr_bin = r"C:\Program Files\Git\usr\bin"
        if Path(git_usr_bin).exists():
                os.environ["PATH"] = f"{git_usr_bin};{os.environ.get('PATH', '')}"
                print("\nAdded Git usr/bin to PATH for this run (cut/expr for CmdStan).")


def main():
        args = parse_args()
        ensure_windows_cmdstan_shell_tools()
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("Item-level Bayesian SEM with Graded Response Measurement")
        print("=" * 60)

        df_rses, df_iss, df_rssis, centered_covs, raw_cov_means = load_item_data()
        print("\nCentered covariate means used for preprocessing:")
        print(raw_cov_means.round(3))

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

        print(
                f"\nData loaded: N={stan_data['N']}, K_x={stan_data['K_x']}, "
                f"K_m={stan_data['K_m']}, K_y={stan_data['K_y']}"
        )

        stan_file, stan_updated = write_stan_model()
        if stan_updated:
                print(f"\nStan model updated: {Path(stan_file).name}")
        else:
                print(f"\nStan model unchanged: reusing existing {Path(stan_file).name}")

        model = CmdStanModel(stan_file=stan_file)

        sample_kwargs = {
                "data": stan_data,
                "chains": args.chains,
                "parallel_chains": args.chains,
                "iter_warmup": args.iter_warmup,
                "iter_sampling": args.iter_sampling,
                "adapt_delta": args.adapt_delta,
                "max_treedepth": args.max_treedepth,
                "output_dir": str(output_dir),
                "show_progress": True,
        }

        fit = model.sample(**sample_kwargs)

        summary = None
        try:
                summary = summarize_fit(fit)
        except RuntimeError as summary_error:
                print("\nWarning: could not generate CmdStan summary table.")
                print(f"Reason: {summary_error}")
                print("Posterior draws are still saved for downstream analysis.")

        idata = az.from_cmdstanpy(posterior=fit, log_likelihood="log_lik")
        posterior_nc_path = output_dir / args.posterior_nc
        az.to_netcdf(idata, posterior_nc_path)

        summary_csv_path = output_dir / args.summary_csv
        if summary is not None:
                summary.to_csv(summary_csv_path)

        save_figures(idata)

        print("\nSaved:")
        print(f"  {Path(stan_file).name}")
        print(f"  CmdStan chain CSV files in: {output_dir}")
        print(f"  {posterior_nc_path}")
        if summary is not None:
                print(f"  {summary_csv_path}")
        print("  fig_bsem_grm_trace.png")
        print("  fig_bsem_grm_indirect.png")
        print("  fig_bsem_grm_forest.png")
        print("  fig_bsem_grm_covariates.png")


if __name__ == "__main__":
        main()
