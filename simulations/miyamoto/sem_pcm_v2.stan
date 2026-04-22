// sem_pcm_v2.stan  — PCM-SEM 통합 모형 (완전 식별)
data {
  int<lower=1> N; int<lower=1> I; int<lower=1> K;
  int<lower=1, upper=I-1>   I_X;
  int<lower=1, upper=I-I_X> I_M;
  array[N, I] int<lower=1, upper=K> y;
  vector[N] gender;
}
transformed data { int I_Y = I - I_X - I_M; }
parameters {
  array[I] vector[K-1] delta_raw;
  array[N] vector[3]   theta;
  real b1; real b2; real g1;
  real gamma_M; real gamma_Y;
  real alpha_M; real alpha_Y;
}
transformed parameters {
  array[I] vector[K-1] delta;
  for (i in 1:I_X) delta[i] = delta_raw[i];
  { real off=0.0;
    for (i in (I_X+1):(I_X+I_M)) off += sum(delta_raw[i]);
    off /= (I_M*(K-1));
    for (i in (I_X+1):(I_X+I_M)) delta[i] = delta_raw[i]-off; }
  { real off=0.0;
    for (i in (I_X+I_M+1):I) off += sum(delta_raw[i]);
    off /= (I_Y*(K-1));
    for (i in (I_X+I_M+1):I) delta[i] = delta_raw[i]-off; }
}
model {
  for (i in 1:I) delta_raw[i] ~ normal(0,3);
  b1~normal(0,1); b2~normal(0,1); g1~normal(0,1);
  gamma_M~normal(0,1); gamma_Y~normal(0,1);
  alpha_M~normal(0,1); alpha_Y~normal(0,1);
  for (n in 1:N) {
    theta[n][1] ~ normal(0,1);
    theta[n][2] ~ normal(alpha_M+b1*theta[n][1]+gamma_M*gender[n], 1.0);
    theta[n][3] ~ normal(alpha_Y+g1*theta[n][1]+b2*theta[n][2]+gamma_Y*gender[n], 1.0);
    for (i in 1:I) {
      int latent_idx; vector[K] log_probs;
      if      (i<=I_X)       latent_idx=1;
      else if (i<=I_X+I_M)   latent_idx=2;
      else                   latent_idx=3;
      log_probs[1]=0;
      for (k in 2:K)
        log_probs[k]=log_probs[k-1]+(theta[n][latent_idx]-delta[i][k-1]);
      y[n,i] ~ categorical_logit(log_probs);
    }
  }
}
generated quantities {
  real indirect_effect = b1*b2;
  real total_effect    = g1+indirect_effect;
  real prop_mediated   = (fabs(total_effect)>1e-10)
                         ? indirect_effect/total_effect : not_a_number();
}
