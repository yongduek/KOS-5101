
data {
  int<lower=1> N;
  int<lower=1> J;
  array[N, J] int<lower=0, upper=1> Y;
}
parameters {
  vector[N] theta;
  vector[J] b;
  vector[J] log_a;
}
transformed parameters {
  vector<lower=0>[J] a = exp(log_a);
}
model {
  theta ~ std_normal();
  b ~ normal(0, 1);
  log_a ~ normal(0, 0.5);
  for (i in 1:N)
    for (j in 1:J)
      Y[i, j] ~ bernoulli_logit(a[j] * (theta[i] - b[j]));
}
generated quantities {
  matrix[N, J] log_lik;
  for (i in 1:N)
    for (j in 1:J)
      log_lik[i, j] = bernoulli_logit_lpmf(Y[i, j] | a[j] * (theta[i] - b[j]));
}
