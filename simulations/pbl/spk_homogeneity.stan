
data {
    int<lower=0> n1; int<lower=0> n2;
    vector[n1] y1; vector[n2] y2;
}
parameters {
    real mu1; real mu2;
    real<lower=0> sigma;
}
model {
    mu1 ~ normal(10, 5); // 중급 수준(10점 내외) 반영한 약한 정보적 사전분포
    mu2 ~ normal(10, 5);
    sigma ~ exponential(0.1);
    y1 ~ normal(mu1, sigma);
    y2 ~ normal(mu2, sigma);
}
generated quantities {
    real delta = mu1 - mu2;
}
