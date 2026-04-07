// lltm_mle.stan
// ============================================================
// Model 3: LLTM MLE via Marginal Likelihood (NB9)
//
// Linear predictor:
//   eta_pi = theta_p - sum_k beta_k * X_ik
//
// Item difficulties are parameterised as beta_i = X_i * beta_k.
// Person ability theta_p ~ N(0, sigma^2) is integrated out.
//
// Parameters: beta_k (K item property effects), sigma
// ============================================================

data {
  int<lower=1> N;
  int<lower=1> I;
  int<lower=1> K;                          // number of item property effects
  array[N, I] int<lower=0, upper=1> Y;
  matrix[I, K] X;                          // item predictor matrix
  int<lower=1> Q;
  vector[Q] gauss_nodes;
  vector[Q] gauss_weights;
}

parameters {
  vector[K] beta_k;            // item property effects
  real<lower=0> sigma;         // person SD
}

model {
  vector[I] beta_pred = X * beta_k;        // predicted item difficulties

  for (p in 1:N) {
    vector[Q] lq;
    for (q in 1:Q) {
      real theta_q = gauss_nodes[q] * sigma * sqrt(2.0);
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
