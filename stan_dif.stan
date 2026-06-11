
data {
  int<lower=1> N_ref;
  int<lower=1> N_foc;
  int<lower=1> J;
  array[N_ref, J] int<lower=0, upper=1> Y_ref;
  array[N_foc, J] int<lower=0, upper=1> Y_foc;
}
parameters {
  vector[N_ref] theta_ref;
  vector[N_foc] theta_foc;
  vector[J] b;            // 공통 난이도 (참조집단 기준)
  vector[J] log_a;        // 공통 변별도 (로그 스케일)
  vector[J] delta;        // DIF 효과: b_foc[j] = b[j] + delta[j]
  real mu_gap;            // 집단 능력 평균 차이 (foc - ref)
}
transformed parameters {
  vector<lower=0>[J] a = exp(log_a);
  vector[J] b_ref = b;
  vector[J] b_foc = b + delta;
}
model {
  // ── 능력 모수 ──
  theta_ref ~ std_normal();
  theta_foc ~ normal(mu_gap, 1);

  // ── 집단 능력 차이 사전분포 ──
  mu_gap ~ normal(0, 1);

  // ── 문항 모수 사전분포 ──
  b ~ normal(0, 1);
  log_a ~ normal(0, 0.5);

  // ── DIF 사전분포: 0 중심, 표준편차 0.3 ──
  // 수축이 너무 강하지 않도록 0.3 사용 (논문 beta 범위 0.08~0.36 대응) → DIF 검출이 적어 상향
  delta ~ normal(0, 0.5);   

  // ── 우도 ──
  for (i in 1:N_ref)
    for (j in 1:J)
      Y_ref[i, j] ~ bernoulli_logit(a[j] * (theta_ref[i] - b_ref[j]));
  for (i in 1:N_foc)
    for (j in 1:J)
      Y_foc[i, j] ~ bernoulli_logit(a[j] * (theta_foc[i] - b_foc[j]));
}
