// latent_regression_rasch_mle.stan
// ============================================================
// Model 2: Latent Regression Rasch MLE via Marginal Likelihood (NB9)
//
// Linear predictor:
//   eta_pi = sum_j vartheta_j * Z_pj + theta_p - beta_i
//
// The residual person effect theta_p ~ N(0, sigma_e^2) is
// integrated out using Gauss-Hermite quadrature.
// The fixed part Z_p * vartheta shifts the quadrature center.
//
// Marginal log-likelihood:
//   log L(beta, vartheta, sigma_e) =
//     sum_p log integral P(Y_p | theta, ...) * phi(theta/sigma_e)/sigma_e d(theta)
//
// Parameters: beta_i, vartheta_j, sigma_e
// ============================================================

data {
  int<lower=1> N;
  int<lower=1> I;
  int<lower=1> J;                          // number of person predictors
  array[N, I] int<lower=0, upper=1> Y;
  matrix[N, J] Z;                          // person predictor matrix
  int<lower=1> Q;
  vector[Q] gauss_nodes;
  vector[Q] gauss_weights;
}

parameters {
  vector[I] beta;              // item difficulties
  vector[J] vartheta;          // person predictor effects
  real<lower=0> sigma_e;       // residual person SD
}

model {
  vector[N] theta_fixed = Z * vartheta;    // fixed part of person ability

  for (p in 1:N) {
    vector[Q] lq;
    for (q in 1:Q) {
      real theta_q = theta_fixed[p] + gauss_nodes[q] * sigma_e * sqrt(2.0);
      real ll = 0.0;
      for (i in 1:I) {
        ll += bernoulli_logit_lpmf(Y[p, i] | theta_q - beta[i]);
      }
      lq[q] = log(gauss_weights[q]) + ll;
    }
    target += log_sum_exp(lq);
  }
  // No priors => pure MLE
}
