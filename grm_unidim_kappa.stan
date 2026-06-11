
data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=2> K;
  array[N, J] int<lower=1, upper=K> y;
}
parameters {
  vector[N] theta;
  vector[J] alpha;              // log a
  array[J] ordered[K-1] kappa;  // cutpoints on eta-scale
}
transformed parameters {
  vector<lower=0>[J] a = exp(alpha);
  array[J] vector[K-1] b;       // theta-scale thresholds (derived)
  for (j in 1:J) {
    b[j] = to_vector(kappa[j]) / a[j];
  }
}
model {
  // identification
  theta ~ std_normal();

  // priors (좀 더 안정적으로)
  alpha ~ normal(0, 0.5);           // a의 log-scale: median exp(0)=1, 너무 작은 a/큰 a 방지
  for (j in 1:J) kappa[j] ~ normal(0, 2.0);

  // likelihood
  for (i in 1:N) {
    for (j in 1:J) {
      y[i, j] ~ ordered_logistic(a[j] * theta[i], kappa[j]);
    }
  }
}
generated quantities {
  array[N, J] int<lower=1, upper=K> y_rep;
  array[N, J] real log_lik;

  for (i in 1:N) {
    for (j in 1:J) {
      log_lik[i, j] = ordered_logistic_lpmf(y[i, j] | a[j] * theta[i], kappa[j]);
      y_rep[i, j]   = ordered_logistic_rng(a[j] * theta[i], kappa[j]);
    }
  }
}
