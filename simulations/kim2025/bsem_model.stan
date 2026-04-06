
data {
  int<lower=1> N;
  int<lower=1> P_X; int<lower=1> P_M; int<lower=1> P_Y;
  matrix[N, P_X] X_ind;
  matrix[N, P_M] M_ind;
  matrix[N, P_Y] Y_ind;
}
parameters {
  // 잠재변수
  vector[N] eta_X;
  vector[N] eta_M;
  vector[N] eta_Y;
  
  // 측정모델 로딩 (첫 번째 지표는 1로 고정하여 척도 설정)
  vector[P_X-1] lambda_X_free;
  vector[P_M-1] lambda_M_free;
  vector[P_Y-1] lambda_Y_free;
  
  // 절편 및 오차 분산
  vector[P_X] tau_X; vector[P_M] tau_M; vector[P_Y] tau_Y;
  vector<lower=0>[P_X] theta_X; vector<lower=0>[P_M] theta_M; vector<lower=0>[P_Y] theta_Y;
  
  // 구조모델 경로계수
  real a;
  real b;
  real cp;
  real<lower=0> psi_M;
  real<lower=0> psi_Y;
}
transformed parameters {
  vector[P_X] lambda_X = append_row(1.0, lambda_X_free);
  vector[P_M] lambda_M = append_row(1.0, lambda_M_free);
  vector[P_Y] lambda_Y = append_row(1.0, lambda_Y_free);
}
model {
  // Priors
  eta_X ~ std_normal();
  lambda_X_free ~ normal(0, 5); lambda_M_free ~ normal(0, 5); lambda_Y_free ~ normal(0, 5);
  tau_X ~ normal(3, 3); tau_M ~ normal(3, 3); tau_Y ~ normal(3, 3);
  a ~ normal(0, 5); b ~ normal(0, 5); cp ~ normal(0, 5);
  
  // 구조모델 (Structural Model)
  eta_M ~ normal(a * eta_X, psi_M);
  eta_Y ~ normal(cp * eta_X + b * eta_M, psi_Y);
  
  // 측정모델 (Measurement Model)
  for (i in 1:N) {
    X_ind[i]' ~ normal(tau_X + lambda_X * eta_X[i], theta_X);
    M_ind[i]' ~ normal(tau_M + lambda_M * eta_M[i], theta_M);
    Y_ind[i]' ~ normal(tau_Y + lambda_Y * eta_Y[i], theta_Y);
  }
}
generated quantities {
  real indirect_effect = a * b;
}
