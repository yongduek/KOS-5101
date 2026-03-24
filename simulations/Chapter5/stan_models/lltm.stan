// lltm.stan
// ============================================================
// Model 3: Linear Logistic Test Model — LLTM (Item Explanatory)
//
//   η_pi = θ_p − Σ_k β_k X_ik
//   P(Y_pi = 1) = logit⁻¹(η_pi)
//
//   θ_p   ~ Normal(0, σ²)
//   β_k   ~ Normal(0, 5²)
//   σ     ~ Half-Cauchy(0, 2.5)
// ============================================================

data {
  int<lower=1> N;                       // persons
  int<lower=1> I;                       // items
  int<lower=1> K;                       // number of item property effects
  array[N, I] int<lower=0, upper=1> Y;  // responses
  matrix[I, K] X;                       // item predictor matrix
}

parameters {
  vector[N] theta;            // person abilities
  vector[K] beta_k;           // item property effects
  real<lower=0> sigma;        // person SD
}

model {
  // ── Priors ──
  sigma  ~ cauchy(0, 2.5);
  beta_k ~ normal(0, 5);
  theta  ~ normal(0, sigma);

  // ── Predicted item difficulties ──
  vector[I] beta_pred = X * beta_k;

  // ── Likelihood ──
  for (p in 1:N) {
    for (i in 1:I) {
      Y[p, i] ~ bernoulli_logit(theta[p] - beta_pred[i]);
    }
  }
}

generated quantities {
  // Predicted item difficulties (for comparison with Rasch)
  vector[I] beta_pred_gq = X * beta_k;

  // Log-likelihood
  array[N, I] real log_lik;
  for (p in 1:N) {
    for (i in 1:I) {
      log_lik[p, i] = bernoulli_logit_lpmf(Y[p, i] | theta[p] - beta_pred_gq[i]);
    }
  }
}
