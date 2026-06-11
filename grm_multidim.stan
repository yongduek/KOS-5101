
data {
  int<lower=1> J;           // 피험자 수 (394)
  int<lower=1> I_SE;        // SE 문항 수 (10)
  int<lower=1> I_IS;        // IS 문항 수 (24)
  int<lower=1> I_AS;        // AS 문항 수 (36)
  int<lower=2> K_SE;        // SE 범주 수 (4)
  int<lower=2> K_IS;        // IS 범주 수 (5)
  int<lower=2> K_AS;        // AS 범주 수 (5)

  array[J, I_SE] int<lower=1, upper=4> y_SE;
  array[J, I_IS] int<lower=1, upper=5> y_IS;
  array[J, I_AS] int<lower=1, upper=5> y_AS;
}

parameters {
  // ── 피험자 파라미터 ──────────────────────────────────────
  vector[J] theta_SE;
  vector[J] theta_IS;
  vector[J] theta_AS;

  // ── 문항 변별도 (양수 제약) ──────────────────────────────
  vector<lower=0>[I_SE] a_SE;
  vector<lower=0>[I_IS] a_IS;
  vector<lower=0>[I_AS] a_AS;

  // ── cutpoint: 로짓 척도에서 ordered 파라미터로 직접 선언
  // 주의: ordered_logistic(eta, c)에서 eta = a*theta
  //       c는 로짓 척도의 cutpoint (= a * b_threshold)
  //       alpha * b 형태를 Stan code에 직접 넣지 않음
  array[I_SE] ordered[3] c_SE;  // SE: K=4 → 3개 cutpoint
  array[I_IS] ordered[4] c_IS;  // IS: K=5 → 4개 cutpoint
  array[I_AS] ordered[4] c_AS;  // AS: K=5 → 4개 cutpoint
}

model {
  // ── 사전분포 ────────────────────────────────────────────
  // 피험자 능력치: 척도 불확정성 해결 (표준정규)
  theta_SE ~ std_normal();
  theta_IS ~ std_normal();
  theta_AS ~ std_normal();

  // 변별도: 로그정규 (중앙값 exp(0.3)≈1.35, 양수 보장)
  // 너무 큰 변별도를 억제하여 수치 안정성 향상
  a_SE ~ lognormal(0.3, 0.5);
  a_IS ~ lognormal(0.3, 0.5);
  a_AS ~ lognormal(0.3, 0.5);

  // cutpoint: 로짓 척도에서 정규분포 사전분포
  // ordered 파라미터이므로 a와 곱하지 않음
  // normal(0, 2): 로짓 척도 -6~+6 범위를 95% CI로 포함
  for (i in 1:I_SE) c_SE[i] ~ normal(0, 2);
  for (i in 1:I_IS) c_IS[i] ~ normal(0, 2);
  for (i in 1:I_AS) c_AS[i] ~ normal(0, 2);

  // ── 우도함수 ────────────────────────────────────────────
  // GRM: P(X >= k | theta, a, c) = logistic(a*theta - c_k)
  // ordered_logistic(eta, c): eta = a*theta
  // cutpoints c는 로짓 척도 (c = a * b, b는 능력치 척도 임계값)
  // 즉, ordered_logistic에 a*b를 직접 넣지 않음
  for (j in 1:J) {
    for (i in 1:I_SE) {
      y_SE[j, i] ~ ordered_logistic(a_SE[i] * theta_SE[j], c_SE[i]);
    }
  }
  for (j in 1:J) {
    for (i in 1:I_IS) {
      y_IS[j, i] ~ ordered_logistic(a_IS[i] * theta_IS[j], c_IS[i]);
    }
  }
  for (j in 1:J) {
    for (i in 1:I_AS) {
      y_AS[j, i] ~ ordered_logistic(a_AS[i] * theta_AS[j], c_AS[i]);
    }
  }
}

generated quantities {
  // ── b 파라미터: 능력치 척도 임계값 (CRC 분석용) ──────────
  // b_ik = c_ik / a_i (로짓 척도 cutpoint를 능력치 척도로 변환)
  array[I_SE] vector[3] b_SE;
  array[I_IS] vector[4] b_IS;
  array[I_AS] vector[4] b_AS;

  for (i in 1:I_SE) b_SE[i] = to_vector(c_SE[i]) / a_SE[i];
  for (i in 1:I_IS) b_IS[i] = to_vector(c_IS[i]) / a_IS[i];
  for (i in 1:I_AS) b_AS[i] = to_vector(c_AS[i]) / a_AS[i];

  // ── PPC용 사후예측복제 ───────────────────────────────────
  array[J, I_SE] int y_rep_SE;
  array[J, I_IS] int y_rep_IS;
  array[J, I_AS] int y_rep_AS;

  for (j in 1:J) {
    for (i in 1:I_SE)
      y_rep_SE[j, i] = ordered_logistic_rng(a_SE[i] * theta_SE[j], c_SE[i]);
    for (i in 1:I_IS)
      y_rep_IS[j, i] = ordered_logistic_rng(a_IS[i] * theta_IS[j], c_IS[i]);
    for (i in 1:I_AS)
      y_rep_AS[j, i] = ordered_logistic_rng(a_AS[i] * theta_AS[j], c_AS[i]);
  }
}
