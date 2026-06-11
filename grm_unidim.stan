
data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=2> K;
  array[N, J] int<lower=1, upper=K> y;
}
parameters {
  vector[N] theta;
  vector<lower=0>[J] a;
  array[J] ordered[K-1] b;  // theta-scale thresholds
}
model {
  // Identification (scale)
  theta ~ std_normal();

  // Priors
  a ~ lognormal(0, 0.4);          // discrimination > 0, centered near 1
  for (j in 1:J) b[j] ~ normal(0, 1.5); // thresholds around typical theta range

  // Likelihood (GRM via ordered_logistic)
  for (i in 1:N) {
    for (j in 1:J) {
      y[i, j] ~ ordered_logistic(a[j] * theta[i], a[j] * b[j]);
    }
  }
}
generated quantities {
  array[N, J] int<lower=1, upper=K> y_rep;
  array[N, J] real log_lik;

  for (i in 1:N) {
    for (j in 1:J) {
      log_lik[i, j] = ordered_logistic_lpmf(y[i, j] | a[j] * theta[i], a[j] * b[j]);
      y_rep[i, j] = ordered_logistic_rng(a[j] * theta[i], a[j] * b[j]);
    }
  }
}
