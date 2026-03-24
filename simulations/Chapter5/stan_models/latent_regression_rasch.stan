// latent_regression_rasch.stan
// ============================================================
// Model 2: Latent Regression Rasch Model (Person Explanatory)
//
//   η_pi = Σ_j ϑ_j Z_pj + θ_p − β_i
//   P(Y_pi = 1) = logit⁻¹(η_pi)
//
//   θ_p     ~ Normal(0, σ_ε²)   [residual person effect]
//   β_i     ~ Normal(0, 5²)
//   ϑ_j     ~ Normal(0, 5²)
//   σ_ε     ~ Half-Cauchy(0, 2.5)
// ============================================================

data {
  int<lower=1> N;                       // persons
  int<lower=1> I;                       // items
  int<lower=1> J;                       // number of person predictors
  array[N, I] int<lower=0, upper=1> Y;  // responses
  matrix[N, J] Z;                       // person predictor matrix
}

parameters {
  vector[N] theta;                // residual person effects
  vector[I] beta;                 // item difficulties
  vector[J] vartheta;            // person predictor effects (ϑ)
  real<lower=0> sigma_e;          // residual person SD
}

model {
  // ── Priors ──
  sigma_e  ~ cauchy(0, 2.5);
  beta     ~ normal(0, 5);
  vartheta ~ normal(0, 5);
  theta    ~ normal(0, sigma_e);

  // ── Likelihood ──
  {
    // Pre-compute fixed person contribution
    vector[N] theta_fixed = Z * vartheta;

    for (p in 1:N) {
      real person_ability = theta_fixed[p] + theta[p];
      for (i in 1:I) {
        Y[p, i] ~ bernoulli_logit(person_ability - beta[i]);
      }
    }
  }
}

generated quantities {
  // Log-likelihood for model comparison
  array[N, I] real log_lik;
  {
    vector[N] theta_fixed = Z * vartheta;
    for (p in 1:N) {
      real person_ability = theta_fixed[p] + theta[p];
      for (i in 1:I) {
        log_lik[p, i] = bernoulli_logit_lpmf(Y[p, i] | person_ability - beta[i]);
      }
    }
  }

  // Total person ability (for posterior summaries)
  vector[N] theta_total;
  {
    vector[N] tf = Z * vartheta;
    for (p in 1:N) {
      theta_total[p] = tf[p] + theta[p];
    }
  }
}
