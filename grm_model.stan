
data {
  int<lower=1> N;         // 전체 응답 수
  int<lower=1> J;         // 피험자 수
  int<lower=1> I;         // 문항 수
  int<lower=2> K;         // 응답 범주 수 (5)
  array[N] int<lower=1,upper=J> jj;  // 피험자 인덱스
  array[N] int<lower=1,upper=I> ii;  // 문항 인덱스
  array[N] int<lower=1,upper=K> y;   // 응답 데이터
}
parameters {
  vector[J] theta;             // 피험자 능력치
  vector<lower=0>[I] a;        // 문항 변별도 (양수 제한)
  array[I] ordered[K-1] c;     // 문항 임계값 (크기 순서대로 정렬됨)
}
model {
  // 사전분포 설정
  theta ~ std_normal();        // 능력치는 평균 0, 표준편차 1
  a ~ lognormal(0, 0.5);       // 변별도는 로그정규분포
  for (i in 1:I) {
    c[i] ~ normal(0, 2);       // 임계값은 넓은 분산의 정규분포
  }

  // 우도 함수 및 모델식 적용
  for (n in 1:N) {
    y[n] ~ ordered_logistic(a[ii[n]] * theta[jj[n]], c[ii[n]]);
  }
}
