// latent_regression_lltm_mle.stan
// ============================================================
// Model 4: Latent Regression LLTM MLE via Marginal Likelihood (NB9)
//
// Linear predictor:
//   eta_pi = sum_j vartheta_j * Z_pj + theta_p - sum_k beta_k * X_ik
//
// Both item difficulties and person predictors are explained.
// Residual theta_p ~ N(0, sigma_e^2) is integrated out.
//
// Parameters: beta_k, vartheta_j, sigma_e
// ============================================================

data {
  int<lower=1> N;
  int<lower=1> I;
  int<lower=1> J;
  int<lower=1> K;
  array[N, I] int<lower=0, upper=1> Y;
  matrix[N, J] Z;
  matrix[I, K] X;
  int<lower=1> Q;
  vector[Q] gauss_nodes;
  vector[Q] gauss_weights;
}

parameters {
  vector[K] beta_k;            // item property effects
  vector[J] vartheta;          // person predictor effects
  real<lower=0> sigma_e;       // residual person SD
}

model {
  vector[N] theta_fixed = Z * vartheta;
  vector[I] beta_pred   = X * beta_k;

  for (p in 1:N) {
    vector[Q] lq;
    for (q in 1:Q) {
      real theta_q = theta_fixed[p] + gauss_nodes[q] * sigma_e * sqrt(2.0);
      real ll = 0.0;
      for (i in 1:I) {
        ll += bernoulli_logit_lpmf(Y[p, i] | theta_q - beta_pred[i]);
      }
      lq[q] = log(gauss_weights[q]) + ll;
    }
    target += log_sum_exp(lq);
  }
  // No priors => pure MLE
}
