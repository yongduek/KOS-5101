
data {
    int<lower=0> N1;
    int<lower=0> N2;
    vector[N1] Y1;
    vector[N2] Y2;
}

parameters {
    real mu1;
    real mu2;
    real<lower=0> sigma;  // shared SD
}

model {
    // priors
    mu1 ~ normal(0, 100);
    mu2 ~ normal(0, 100);
    sigma ~ cauchy(0, 10);

    // likelihood
    Y1 ~ normal(mu1, sigma);
    Y2 ~ normal(mu2, sigma);
}

generated quantities {
    real diff = mu2 - mu1;  // difference of means
}
