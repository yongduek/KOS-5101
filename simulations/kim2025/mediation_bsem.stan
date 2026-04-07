
functions {
    real pcm_item_lpmf(int y, real theta, real item_location, vector step_difficulty) {
        int n_cat = num_elements(step_difficulty) + 1;
        vector[n_cat] eta;
        eta[1] = 0;
        for (c in 2:n_cat) {
            eta[c] = eta[c - 1] + theta - item_location - step_difficulty[c - 1];
        }
        return categorical_logit_lpmf(y | eta);
    }
}
data {
    int<lower=1> N;
    int<lower=1> K_x;
    int<lower=1> K_m;
    int<lower=1> K_y;
    int<lower=1> C;
    array[N, K_x] int<lower=1, upper=4> x_items;
    array[N, K_m] int<lower=1, upper=5> m_items;
    array[N, K_y] int<lower=1, upper=5> y_items;
    matrix[N, C] covs;
}
parameters {
    vector[N] theta_x;
    vector[N] theta_m;
    vector[N] theta_y;

    real alpha_m;
    real alpha_y;
    real a;
    real b;
    real cp;
    vector[C] beta_m;
    vector[C] beta_y;
    real<lower=0> sigma_m;
    real<lower=0> sigma_y;

    vector[K_x] item_loc_x;
    vector[K_m] item_loc_m;
    vector[K_y] item_loc_y;

    array[K_x] ordered[3] raw_steps_x;
    array[K_m] ordered[4] raw_steps_m;
    array[K_y] ordered[4] raw_steps_y;
}
transformed parameters {
    vector[N] mu_m = alpha_m + a * theta_x + covs * beta_m;
    vector[N] mu_y = alpha_y + cp * theta_x + b * theta_m + covs * beta_y;
    array[K_x] vector[3] steps_x;
    array[K_m] vector[4] steps_m;
    array[K_y] vector[4] steps_y;

    for (j in 1:K_x) {
        steps_x[j] = to_vector(raw_steps_x[j]) - mean(to_vector(raw_steps_x[j]));
    }
    for (j in 1:K_m) {
        steps_m[j] = to_vector(raw_steps_m[j]) - mean(to_vector(raw_steps_m[j]));
    }
    for (j in 1:K_y) {
        steps_y[j] = to_vector(raw_steps_y[j]) - mean(to_vector(raw_steps_y[j]));
    }
}
model {
    theta_x ~ std_normal();
    theta_m ~ normal(mu_m, sigma_m);
    theta_y ~ normal(mu_y, sigma_y);

    alpha_m ~ normal(0, 1);
    alpha_y ~ normal(0, 1);
    a ~ normal(0, 1);
    b ~ normal(0, 1);
    cp ~ normal(0, 1);
    beta_m ~ normal(0, 0.75);
    beta_y ~ normal(0, 0.75);
    sigma_m ~ exponential(1);
    sigma_y ~ exponential(1);

    item_loc_x ~ normal(0, 1);
    item_loc_m ~ normal(0, 1);
    item_loc_y ~ normal(0, 1);

    for (j in 1:K_x) {
        raw_steps_x[j] ~ normal(0, 1.5);
    }
    for (j in 1:K_m) {
        raw_steps_m[j] ~ normal(0, 1.5);
    }
    for (j in 1:K_y) {
        raw_steps_y[j] ~ normal(0, 1.5);
    }

    for (n in 1:N) {
        for (j in 1:K_x) {
            target += pcm_item_lpmf(x_items[n, j] | theta_x[n], item_loc_x[j], steps_x[j]);
        }
        for (j in 1:K_m) {
            target += pcm_item_lpmf(m_items[n, j] | theta_m[n], item_loc_m[j], steps_m[j]);
        }
        for (j in 1:K_y) {
            target += pcm_item_lpmf(y_items[n, j] | theta_y[n], item_loc_y[j], steps_y[j]);
        }
    }
}
generated quantities {
    real indirect_effect = a * b;
    real total_effect = cp + a * b;
        vector[N] log_lik;

        for (n in 1:N) {
                real lp = 0;
                for (j in 1:K_x) {
                        lp += pcm_item_lpmf(x_items[n, j] | theta_x[n], item_loc_x[j], steps_x[j]);
                }
                for (j in 1:K_m) {
                        lp += pcm_item_lpmf(m_items[n, j] | theta_m[n], item_loc_m[j], steps_m[j]);
                }
                for (j in 1:K_y) {
                        lp += pcm_item_lpmf(y_items[n, j] | theta_y[n], item_loc_y[j], steps_y[j]);
                }
                log_lik[n] = lp;
        }
}
