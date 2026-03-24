// rasch.stan
// ============================================================
// Model 1: The Rasch Model (Doubly Descriptive)
//
//   η_pi = θ_p − β_i
//   P(Y_pi = 1 | θ_p, β_i) = logit⁻¹(θ_p − β_i)
//
//   θ_p ~ Normal(0, σ²)
//   β_i ~ Normal(0, 5²)     [weakly informative prior]
//   σ   ~ Half-Cauchy(0, 2.5)
// ============================================================

data {
  int<lower=1> N;              // number of persons
  int<lower=1> I;              // number of items
  array[N, I] int<lower=0, upper=1> Y;  // response matrix
}

parameters {
  vector[N] theta;             // person abilities (random effects)
  vector[I] beta;              // item difficulties (fixed effects)
  real<lower=0> sigma;         // SD of person ability distribution
}

model {
  // ── Priors ──
  sigma ~ cauchy(0, 2.5);            // half-Cauchy (constrained positive)
  beta  ~ normal(0, 5);              // weakly informative
  theta ~ normal(0, sigma);          // hierarchical prior on persons

  // ── Likelihood ──
  for (p in 1:N) {
    for (i in 1:I) {
      Y[p, i] ~ bernoulli_logit(theta[p] - beta[i]);
    }
  }
}

generated quantities {
  // Log-likelihood for LOO-CV / WAIC
  array[N, I] real log_lik;
  for (p in 1:N) {
    for (i in 1:I) {
      log_lik[p, i] = bernoulli_logit_lpmf(Y[p, i] | theta[p] - beta[i]);
    }
  }
}
