// latent_regression_lltm.stan
// ============================================================
// Model 4: Latent Regression LLTM (Doubly Explanatory)
//
//   η_pi = Σ_j ϑ_j Z_pj + θ_p − Σ_k β_k X_ik
//   P(Y_pi = 1) = logit⁻¹(η_pi)
//
//   θ_p     ~ Normal(0, σ_ε²)   [residual person effect]
//   β_k     ~ Normal(0, 5²)
//   ϑ_j     ~ Normal(0, 5²)
//   σ_ε     ~ Half-Cauchy(0, 2.5)
// ============================================================

data {
  int<lower=1> N;                       // persons
  int<lower=1> I;                       // items
  int<lower=1> J;                       // person predictors
  int<lower=1> K;                       // item property effects
  array[N, I] int<lower=0, upper=1> Y;  // responses
  matrix[N, J] Z;                       // person predictor matrix
  matrix[I, K] X;                       // item predictor matrix
}

parameters {
  vector[N] theta;            // residual person effects
  vector[K] beta_k;           // item property effects
  vector[J] vartheta;         // person property effects (ϑ)
  real<lower=0> sigma_e;      // residual person SD
}

model {
  // ── Priors ──
  sigma_e  ~ cauchy(0, 2.5);
  beta_k   ~ normal(0, 5);
  vartheta ~ normal(0, 5);
  theta    ~ normal(0, sigma_e);

  // ── Pre-compute ──
  vector[N] theta_fixed = Z * vartheta;
  vector[I] beta_pred   = X * beta_k;

  // ── Likelihood ──
  for (p in 1:N) {
    real person_ability = theta_fixed[p] + theta[p];
    for (i in 1:I) {
      Y[p, i] ~ bernoulli_logit(person_ability - beta_pred[i]);
    }
  }
}

generated quantities {
  vector[I] beta_pred_gq = X * beta_k;

  vector[N] theta_total;
  {
    vector[N] tf = Z * vartheta;
    for (p in 1:N) {
      theta_total[p] = tf[p] + theta[p];
    }
  }

  // Log-likelihood for LOO-CV
  array[N, I] real log_lik;
  {
    vector[N] tf2 = Z * vartheta;
    vector[I] bp = X * beta_k;
    for (p in 1:N) {
      real pa = tf2[p] + theta[p];
      for (i in 1:I) {
        log_lik[p, i] = bernoulli_logit_lpmf(Y[p, i] | pa - bp[i]);
      }
    }
  }
}
