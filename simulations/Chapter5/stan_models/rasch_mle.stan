// rasch_mle.stan
// ============================================================
// Model 1: Rasch MLE via Marginal Likelihood (NB9)
//
// Person ability theta_p is integrated out using
// Gauss-Hermite quadrature nodes/weights supplied as data.
//
// Marginal log-likelihood:
//   log L(beta, sigma) =
//     sum_p log integral P(Y_p | theta, beta) * phi(theta/sigma)/sigma d(theta)
//
// With NO priors on beta and sigma, optimize() finds the
// MLE of the marginal likelihood -- equivalent to NB6 MML.
//
// Parameters: beta_i (I item difficulties), sigma (person SD)
// ============================================================

data {
  int<lower=1> N;                          // persons
  int<lower=1> I;                          // items
  array[N, I] int<lower=0, upper=1> Y;     // response matrix
  int<lower=1> Q;                          // number of quadrature nodes
  vector[Q] gauss_nodes;                   // GH nodes (from numpy hermgauss)
  vector[Q] gauss_weights;                 // GH weights / sqrt(pi), sum = 1
}

parameters {
  vector[I] beta;              // item difficulties
  real<lower=0> sigma;         // person ability SD
}

model {
  // Marginal likelihood: integrate out theta_p via GH quadrature
  for (p in 1:N) {
    vector[Q] lq;
    for (q in 1:Q) {
      real theta_q = gauss_nodes[q] * sigma * sqrt(2.0);
      real ll = 0.0;
      for (i in 1:I) {
        ll += bernoulli_logit_lpmf(Y[p, i] | theta_q - beta[i]);
      }
      lq[q] = log(gauss_weights[q]) + ll;
    }
    target += log_sum_exp(lq);
  }
  // No priors => pure MLE of the marginal likelihood
}
