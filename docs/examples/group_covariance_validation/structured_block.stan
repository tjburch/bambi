functions {
  real integer_power(real base, int exponent) {
    real result = 1;
    real factor = base;
    int remaining = exponent;
    // Integer multiplication retains signed correlations and derivatives at zero.
    while (remaining > 0) {
      if (remaining % 2 == 1)
        result *= factor;
      remaining = remaining / 2;
      if (remaining > 0)
        factor *= factor;
    }
    return result;
  }

  vector pacf_to_acf(vector partial) {
    int horizon = num_elements(partial);
    vector[horizon + 1] acf = rep_vector(0, horizon + 1);
    vector[horizon] phi = rep_vector(0, horizon);
    real variance = 1;
    acf[1] = 1;
    if (horizon > 0) {
      for (k in 1:horizon) {
        vector[horizon] next_phi = phi;
        acf[k + 1] = partial[k] * variance;
        if (k > 1) {
          for (j in 1:(k - 1)) {
            acf[k + 1] += phi[j] * acf[k - j + 1];
            next_phi[j] = phi[j] - partial[k] * phi[k - j];
          }
        }
        next_phi[k] = partial[k];
        phi = next_phi;
        variance *= 1 - square(partial[k]);
      }
    }
    return acf;
  }
}
data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> G;
  int<lower=2> Q;
  int<lower=1, upper=5> structure;
  matrix[N, P] X;
  array[N] int<lower=0> trials;
  array[N] int<lower=0> y;
  array[N] int<lower=1, upper=G> group_id;
  array[N] int<lower=1, upper=Q> level_id;
  array[Q] int<lower=0, upper=1000000000> time_index;
  vector[Q] time;
  int<lower=0> max_lag;
  int<lower=0, upper=1> use_design;
  matrix[N * use_design, Q] Z;
  int<lower=0, upper=1> prior_only;
}
transformed data {
  int n_sd = structure == 5 ? Q : 1;
  int n_rho = structure == 1 || structure == 3 ? 1 : 0;
  int n_decay = structure == 2 ? 1 : 0;
  int n_partial = structure == 4 ? max_lag : 0;
  int correlation_dim = structure == 5 ? Q : 1;
  real rho_lower = structure == 3 ? -1.0 / (Q - 1) : -1.0;
  array[G] int group_count = rep_array(0, G);

  if (use_design && structure != 5)
    reject("A coefficient design matrix is supported only for unstructured effects");
  if (structure == 4) {
    if (max_lag < 1 || max_lag < time_index[Q] - time_index[1])
      reject("Toeplitz max_lag must cover the entire declared integer time span");
  } else if (max_lag != 0) {
    reject("Set max_lag to zero outside the Toeplitz structure");
  }
  for (q in 1:Q) {
    if (is_nan(time[q]) || is_inf(time[q]))
      reject("Time coordinates must be finite");
    if (q > 1) {
      if ((structure == 1 || structure == 4)
          && time_index[q] <= time_index[q - 1])
        reject("Discrete time coordinates must be strictly increasing");
      if (structure == 2 && time[q] <= time[q - 1])
        reject("Continuous time coordinates must be strictly increasing");
    }
  }
  if (structure == 2 && is_inf(time[Q] - time[1]))
    reject("The continuous time span must be finite");
  for (n in 1:N) {
    group_count[group_id[n]] += 1;
    if (y[n] > trials[n])
      reject("Success counts must not exceed trials");
    for (p in 1:P)
      if (is_nan(X[n, p]) || is_inf(X[n, p]))
        reject("Fixed-effect design entries must be finite");
    if (use_design) {
      if (level_id[n] != 1)
        reject("Set unused level_id entries to one when use_design is enabled");
      for (q in 1:Q)
        if (is_nan(Z[n, q]) || is_inf(Z[n, q]))
          reject("Coefficient design entries must be finite");
    }
  }
  for (g in 1:G)
    if (group_count[g] == 0)
      reject("Every declared fitted group must occur in the observations");
}
parameters {
  vector[P] beta;
  vector<lower=0>[n_sd] sd;
  vector<lower=rho_lower, upper=1>[n_rho] rho;
  vector<lower=0>[n_decay] decay;
  vector<lower=-1, upper=1>[n_partial] partial;
  cholesky_factor_corr[correlation_dim] L_correlation;
  matrix[G, Q] z;
}
transformed parameters {
  matrix[Q, Q] correlation = diag_matrix(rep_vector(1, Q));
  matrix[Q, Q] L;
  matrix[G, Q] coefficient;
  vector[N] eta = X * beta;

  if (structure == 5) {
    correlation = multiply_lower_tri_self_transpose(L_correlation);
    L = diag_pre_multiply(sd, L_correlation);
  } else {
    vector[n_partial + 1] acf = pacf_to_acf(partial);
    for (i in 1:(Q - 1)) {
      for (j in (i + 1):Q) {
        real value;
        if (structure == 1)
          value = integer_power(rho[1], time_index[j] - time_index[i]);
        else if (structure == 2)
          value = exp(-decay[1] * (time[j] - time[i]));
        else if (structure == 3)
          value = rho[1];
        else
          value = acf[time_index[j] - time_index[i] + 1];
        correlation[i, j] = value;
        correlation[j, i] = value;
      }
    }
    // Unit-diagonal correlation makes sd the marginal latent scale.
    L = sd[1] * cholesky_decompose(correlation);
  }
  coefficient = z * L';
  for (n in 1:N) {
    if (use_design)
      eta[n] += dot_product(Z[n], coefficient[group_id[n]]);
    else
      eta[n] += coefficient[group_id[n], level_id[n]];
  }
}
model {
  // Matched reference priors regularize standardized logit-scale effects.
  beta ~ normal(0, 1.5);
  // Positive support gives the half-normal marginal SD prior.
  sd ~ normal(0, 2.5);
  // Bounded support gives truncated normal correlation and PACF priors.
  rho ~ normal(0, 0.5);
  partial ~ normal(0, 0.5);
  // Decay is expressed per unit of the supplied continuous time coordinate.
  decay ~ exponential(1);
  if (structure == 5)
    L_correlation ~ lkj_corr_cholesky(2);
  to_vector(z) ~ std_normal();
  if (!prior_only)
    y ~ binomial_logit(trials, eta);
}
generated quantities {
  vector[N] probability = inv_logit(eta);
  vector[N] log_lik;
  array[N] int y_rep;
  matrix[Q, Q] covariance = multiply_lower_tri_self_transpose(L);
  real log_hyperprior = normal_lpdf(beta | 0, 1.5)
    + normal_lpdf(sd | 0, 2.5) - n_sd * normal_lccdf(0 | 0, 2.5);
  if (n_rho > 0)
    log_hyperprior += normal_lpdf(rho | 0, 0.5)
      - log_diff_exp(normal_lcdf(1 | 0, 0.5), normal_lcdf(rho_lower | 0, 0.5));
  if (n_partial > 0)
    log_hyperprior += normal_lpdf(partial | 0, 0.5)
      - n_partial * log_diff_exp(normal_lcdf(1 | 0, 0.5), normal_lcdf(-1 | 0, 0.5));
  if (n_decay > 0)
    log_hyperprior += exponential_lpdf(decay | 1);
  if (structure == 5)
    log_hyperprior += lkj_corr_lpdf(correlation | 2);
  for (n in 1:N) {
    log_lik[n] = binomial_logit_lpmf(y[n] | trials[n], eta[n]);
    y_rep[n] = binomial_rng(trials[n], probability[n]);
  }
}
