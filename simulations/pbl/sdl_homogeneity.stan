
data {
    int<lower=0> n1;
    int<lower=0> n2;
    vector[n1] y1;
    vector[n2] y2;
}
parameters {
    real mu1;
    real mu2;
    real<lower=0> sigma;
}
model {
    mu1 ~ normal(0, 10);
    mu2 ~ normal(0, 10);
    sigma ~ exponential(0.1);
    y1 ~ normal(mu1, sigma);
    y2 ~ normal(mu2, sigma);
}
generated quantities {
    real delta = mu1 - mu2;
}
