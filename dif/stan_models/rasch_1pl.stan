data {
  int<lower=1> N;          // number of students
  int<lower=1> J;          // number of items
  array[N, J] int<lower=0, upper=1> Y;  // response matrix (0/1)
}
parameters {
  vector[N] theta;          // student ability
  vector[J] beta;           // item difficulty
}
model {
  // priors
  theta ~ normal(0, 1);
  beta  ~ normal(0, 3);

  // likelihood
  for (i in 1:N)
    for (j in 1:J)
      Y[i, j] ~ bernoulli_logit(theta[i] - beta[j]);
}
generated quantities {
  // pointwise log-likelihood for model comparison (WAIC/LOO)
  matrix[N, J] log_lik;
  for (i in 1:N)
    for (j in 1:J)
      log_lik[i, j] = bernoulli_logit_lpmf(Y[i, j] | theta[i] - beta[j]);
}