data {
  int<lower=1> N;                        // number of students
  int<lower=1> J;                        // number of items
  array[N, J] int<lower=0, upper=1> Y;   // response matrix (0/1)
  array[N] int<lower=0, upper=1> group;  // 0 = reference, 1 = focal
}
parameters {
  vector[N] theta;   // student ability
  vector[J] beta;    // baseline item difficulty (reference group)
  vector[J] delta;   // uniform DIF: difficulty shift for focal group
}
model {
  // priors
  theta ~ normal(0, 1);
  beta  ~ normal(0, 3);
  delta ~ normal(0, 1);   // weakly regularising; centred at no DIF

  // likelihood
  for (i in 1:N)
    for (j in 1:J)
      Y[i, j] ~ bernoulli_logit(theta[i] - beta[j] - delta[j] * group[i]);
}
generated quantities {
  matrix[N, J] log_lik;
  for (i in 1:N)
    for (j in 1:J)
      log_lik[i, j] = bernoulli_logit_lpmf(
        Y[i, j] | theta[i] - beta[j] - delta[j] * group[i]);
}