
data {
  int<lower=1> N;         // 총 응답 수
  int<lower=1> J;         // 사람 수
  int<lower=1> I;         // 문항 수
  int<lower=2> K;         // 범주 수 (5)
  array[N] int<lower=1,upper=J> jj; // 응답자의 인덱스
  array[N] int<lower=1,upper=I> ii; // 문항의 인덱스
  array[N] int<lower=1,upper=K> y;  // 실제 응답 데이터
}
parameters {
  vector[J] theta;             // 사람의 잠재 능력 모수
  vector<lower=0>[I] a;        // 문항 변별도 모수 (항상 양수)
  array[I] ordered[K-1] c;     // 문항 임계값 모수 (크기순 정렬 필요)
}
model {
  // 사전분포 설정 (Prior)
  theta ~ std_normal();        // 능력치는 평균 0, 표준편차 1의 정규분포 (척도 고정용)
  a ~ lognormal(0, 0.5);       // 변별도는 음수가 되지 않도록 로그정규분포 적용
  for (i in 1:I) {
    c[i] ~ normal(0, 2);       // 임계값은 평균 0, 표준편차 2의 정규분포 적용
  }

  // 우도함수 (Likelihood)
  for (n in 1:N) {
    y[n] ~ ordered_logistic(a[ii[n]] * theta[jj[n]], c[ii[n]]); // GRM 모델 수식 적용
  }
}
generated quantities {
  array[N] int y_rep;          // PPC(사후예측점검)를 위한 가상 응답 데이터
  for (n in 1:N) {
    y_rep[n] = ordered_logistic_rng(a[ii[n]] * theta[jj[n]], c[ii[n]]); // 예측된 데이터 생성
  }
}
