
data {
  int<lower=1> N;
  int<lower=1> C;
  vector[N] X;
  vector[N] M;
  vector[N] Y;
  matrix[N, C] covs;
}
parameters {
  real alpha_m;
  real alpha_y;
  real a;
  real b;
  real cp;
  vector[C] beta_m;
  vector[C] beta_y;
  real<lower=0> sigma_m;
  real<lower=0> sigma_y;
}
transformed parameters {
  vector[N] mu_m = alpha_m + a * X + covs * beta_m;
  vector[N] mu_y = alpha_y + cp * X + b * M + covs * beta_y;
}
model {
  alpha_m ~ normal(0, 20);
  alpha_y ~ normal(0, 20);
  a ~ normal(0, 5);
  b ~ normal(0, 5);
  cp ~ normal(0, 5);
  beta_m ~ normal(0, 5);
  beta_y ~ normal(0, 5);
  sigma_m ~ exponential(1);
  sigma_y ~ exponential(1);

  M ~ normal(mu_m, sigma_m);
  Y ~ normal(mu_y, sigma_y);
}
generated quantities {
  real indirect_effect = a * b;
  real total_effect = cp + a * b;
  vector[N] log_lik_m;
  vector[N] log_lik_y;

  for (n in 1:N) {
    log_lik_m[n] = normal_lpdf(M[n] | mu_m[n], sigma_m);
    log_lik_y[n] = normal_lpdf(Y[n] | mu_y[n], sigma_y);
  }
}
