
data {
    int<lower=1> J; // 응답자 수
    int<lower=1> I; // 문항 수
    int<lower=1> K; // 응답 범주 수
    int<lower=1> N; // 전체 관측치 수
    array[N] int<lower=1, upper=J> jj; // 응답자 ID (1~J)
    array[N] int<lower=1, upper=I> ii; // 문항 ID (1~I)
    array[N] int<lower=1, upper=K> y;  // 응답 (1~K)
}
parameters {
    vector[J] theta;         // 잠재 능력
    vector<lower=0>[I] a;    // 문항 변별도
    array[I] ordered[K-1] c; // 문항별 범주 경계(cutpoints)
}
model {
    // 사전 분포
    theta ~ std_normal();
    a ~ lognormal(0, 0.5); 
    for (i in 1:I) {
        c[i] ~ normal(0, 2);
    }

    // 우도 함수 (Likelihood)
    for (n in 1:N) {
        y[n] ~ ordered_logistic(a[ii[n]] * theta[jj[n]], c[ii[n]]);
    }
}
