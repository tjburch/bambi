data {
  int<lower=1> N;
  int<lower=1> P;
  matrix[N, P] X;
  array[N] int<lower=0> trials;
  array[N] int<lower=0> y;
  int<lower=4> C;
  int<lower=4, upper=C> G;
  array[C] int<lower=1, upper=4> block_id;
  array[C] int<lower=1, upper=G> group_id;
  array[C] int time;
  array[C] int<lower=0, upper=C> previous;
  array[N, 4] int<lower=1, upper=C> row_cell;
  int<lower=0, upper=1> prior_only;
}
transformed data {
  array[G] int last_cell = rep_array(0, G);
  array[C] int gap;
  for (c in 1:C) {
    int p = previous[c];
    if (p != last_cell[group_id[c]])
      reject("previous must point to the immediately preceding cell in its group");
    if (p == 0) {
      gap[c] = 0;
    } else {
      if (p >= c || block_id[p] != block_id[c])
        reject("Cells must follow their predecessors within the same block");
      if (time[c] <= time[p])
        reject("Each group must have unique, strictly increasing integer times");
      gap[c] = time[c] - time[p];
    }
    last_cell[group_id[c]] = c;
  }
  for (g in 1:G)
    if (last_cell[g] == 0)
      reject("Every declared group must have at least one coefficient cell");
  for (n in 1:N) {
    if (y[n] > trials[n])
      reject("Success counts must not exceed trials");
    for (b in 1:4)
      if (block_id[row_cell[n, b]] != b)
        reject("row_cell column must match the corresponding covariance block");
  }
}
parameters {
  vector[P] beta;
  vector<lower=0>[4] sd;
  vector<lower=-1, upper=1>[4] rho;
  vector[C] z;
}
transformed parameters {
  vector[C] coefficient;
  vector[N] eta = X * beta;
  for (c in 1:C) {
    int b = block_id[c];
    if (previous[c] == 0) {
      // Stationary initialization makes sd the marginal, not innovation, scale.
      coefficient[c] = sd[b] * z[c];
    } else {
      real persistence = pow(rho[b], gap[c]);
      coefficient[c] = persistence * coefficient[previous[c]]
        + sd[b] * sqrt(1 - square(persistence)) * z[c];
    }
  }
  for (n in 1:N)
    for (b in 1:4)
      eta[n] += coefficient[row_cell[n, b]];
}
model {
  // Proper reference priors on the uncentered logit design match the Bambi fit.
  beta ~ normal(0, 1.5);
  // Positive support gives a half-normal marginal scale prior.
  sd ~ normal(0, 2.5);
  // Bounded support gives Normal(0, 0.5) truncated to (-1, 1).
  rho ~ normal(0, 0.5);
  z ~ std_normal();
  if (!prior_only)
    y ~ binomial_logit(trials, eta);
}
generated quantities {
  vector[N] probability = inv_logit(eta);
  vector[N] log_lik;
  array[N] int y_rep;
  real log_hyperprior = normal_lpdf(beta | 0, 1.5)
    + normal_lpdf(sd | 0, 2.5) - 4 * normal_lccdf(0 | 0, 2.5)
    + normal_lpdf(rho | 0, 0.5)
    - 4 * log_diff_exp(normal_lcdf(1 | 0, 0.5), normal_lcdf(-1 | 0, 0.5));
  for (n in 1:N) {
    log_lik[n] = binomial_logit_lpmf(y[n] | trials[n], eta[n]);
    y_rep[n] = binomial_rng(trials[n], probability[n]);
  }
}
