
data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> I;
  int<lower=2> K;
  array[N] int<lower=1, upper=J> jj;
  array[N] int<lower=1, upper=I> ii;
  array[N] int<lower=1, upper=K> y;
}
parameters {
  vector[J] theta;
  vector<lower=0.1>[I] a;    // 0 대신 0.1을 하한으로 설정
  array[I] ordered[K-1] b;   // 원시 임계값 (잠재특성 척도)
}
transformed parameters {
  array[I] ordered[K-1] c;   // 스케일된 임계값 (ordered_logistic에 투입)
  for (i in 1:I) {
    c[i] = a[i] * b[i];     // a > 0, b ordered → c도 ordered 보장
  }
}
model {
  theta ~ std_normal();
  a ~ lognormal(0, 0.5);
  for (i in 1:I) {
    b[i] ~ normal(0, 2);
  }
  for (n in 1:N) {
    y[n] ~ ordered_logistic(a[ii[n]] * theta[jj[n]], c[ii[n]]);
  }
}
generated quantities {
  array[N] int y_rep;
  for (n in 1:N) {
    y_rep[n] = ordered_logistic_rng(a[ii[n]] * theta[jj[n]], c[ii[n]]);
  }
}
