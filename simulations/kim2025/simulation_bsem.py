import argparse

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
        """Write an item-level latent mediation model with partial credit measurement blocks."""
        stan_code = r"""
functions {
    real pcm_item_lpmf(int y, real theta, real item_location, vector step_difficulty) {
        int n_cat = num_elements(step_difficulty) + 1;
        vector[n_cat] eta;
        eta[1] = 0;
        for (c in 2:n_cat) {
            eta[c] = eta[c - 1] + theta - item_location - step_difficulty[c - 1];
        }
        return categorical_logit_lpmf(y | eta);
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

    vector[K_x] item_loc_x;
    vector[K_m] item_loc_m;
    vector[K_y] item_loc_y;

    array[K_x] ordered[3] raw_steps_x;
    array[K_m] ordered[4] raw_steps_m;
    array[K_y] ordered[4] raw_steps_y;
}
transformed parameters {
    vector[N] mu_m = alpha_m + a * theta_x + covs * beta_m;
    vector[N] mu_y = alpha_y + cp * theta_x + b * theta_m + covs * beta_y;
    array[K_x] vector[3] steps_x;
    array[K_m] vector[4] steps_m;
    array[K_y] vector[4] steps_y;

    for (j in 1:K_x) {
        steps_x[j] = to_vector(raw_steps_x[j]) - mean(to_vector(raw_steps_x[j]));
    }
    for (j in 1:K_m) {
        steps_m[j] = to_vector(raw_steps_m[j]) - mean(to_vector(raw_steps_m[j]));
    }
    for (j in 1:K_y) {
        steps_y[j] = to_vector(raw_steps_y[j]) - mean(to_vector(raw_steps_y[j]));
    }
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

    item_loc_x ~ normal(0, 1);
    item_loc_m ~ normal(0, 1);
    item_loc_y ~ normal(0, 1);

    for (j in 1:K_x) {
        raw_steps_x[j] ~ normal(0, 1.5);
    }
    for (j in 1:K_m) {
        raw_steps_m[j] ~ normal(0, 1.5);
    }
    for (j in 1:K_y) {
        raw_steps_y[j] ~ normal(0, 1.5);
    }

    for (n in 1:N) {
        for (j in 1:K_x) {
            target += pcm_item_lpmf(x_items[n, j] | theta_x[n], item_loc_x[j], steps_x[j]);
        }
        for (j in 1:K_m) {
            target += pcm_item_lpmf(m_items[n, j] | theta_m[n], item_loc_m[j], steps_m[j]);
        }
        for (j in 1:K_y) {
            target += pcm_item_lpmf(y_items[n, j] | theta_y[n], item_loc_y[j], steps_y[j]);
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
                        lp += pcm_item_lpmf(x_items[n, j] | theta_x[n], item_loc_x[j], steps_x[j]);
                }
                for (j in 1:K_m) {
                        lp += pcm_item_lpmf(m_items[n, j] | theta_m[n], item_loc_m[j], steps_m[j]);
                }
                for (j in 1:K_y) {
                        lp += pcm_item_lpmf(y_items[n, j] | theta_y[n], item_loc_y[j], steps_y[j]);
                }
                log_lik[n] = lp;
        }
}
"""
        stan_file = "mediation_bsem.stan"
        with open(stan_file, "w", encoding="utf-8") as file_handle:
                file_handle.write(stan_code)
        return stan_file


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


def save_figures(idata):
        az.style.use("arviz-darkgrid")

        az.plot_trace(idata, var_names=["a", "b", "cp", "indirect_effect"], compact=True)
        plt.suptitle("Item-level PCM BSEM Trace Plots", y=1.02, fontsize=14)
        plt.tight_layout()
        plt.savefig("fig_bsem_pcm_trace.png", dpi=150, bbox_inches="tight")
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
        ax.set_title("PCM BSEM Posterior of Indirect Effect", fontsize=13)
        plt.tight_layout()
        plt.savefig("fig_bsem_pcm_indirect.png", dpi=150, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        az.plot_forest(
                idata,
                var_names=["a", "b", "cp", "indirect_effect", "total_effect"],
                combined=True,
                hdi_prob=0.95,
                ax=ax,
        )
        ax.set_title("PCM BSEM Structural Effects", fontsize=13)
        plt.tight_layout()
        plt.savefig("fig_bsem_pcm_forest.png", dpi=150, bbox_inches="tight")
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
        plt.savefig("fig_bsem_pcm_covariates.png", dpi=150, bbox_inches="tight")
        plt.close()


def parse_args():
        parser = argparse.ArgumentParser(
                description="Run the item-level PCM Bayesian SEM mediation model."
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
        print("Item-level Bayesian SEM with Partial Credit Measurement")
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

        stan_file = write_stan_model()
        model = CmdStanModel(stan_file=stan_file)

        fit = model.sample(
                data=stan_data,
                chains=args.chains,
                parallel_chains=args.chains,
                iter_warmup=args.iter_warmup,
                iter_sampling=args.iter_sampling,
                adapt_delta=args.adapt_delta,
                max_treedepth=args.max_treedepth,
                show_progress=True,
        )

        summarize_fit(fit)

        idata = az.from_cmdstanpy(posterior=fit, log_likelihood="log_lik")
        save_figures(idata)

        print("\nSaved:")
        print("  mediation_bsem.stan")
        print("  fig_bsem_pcm_trace.png")
        print("  fig_bsem_pcm_indirect.png")
        print("  fig_bsem_pcm_forest.png")
        print("  fig_bsem_pcm_covariates.png")


if __name__ == "__main__":
        main()
