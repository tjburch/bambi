# Single-block Stan reference

`structured_block.stan` specifies one learned group-specific covariance block with
a binomial-logit likelihood. Set every trial count to one for Bernoulli data.
This is an independent reference implementation, not brms residual autocorrelation.

**Status: source supplied; compilation, sampling, and numerical agreement are
unverified.** Do not treat this file as a passed validation gate. No compilation,
sampling, or dependency installation was run while adding it.

## Input schema

All fields are required. Stan indices start at one. Preserve the same row,
coefficient, and group ordering in both implementations.

| Field | Type and shape | Meaning |
|---|---|---|
| `N` | integer ≥ 1 | Observation count |
| `P` | integer ≥ 1 | Fixed-effect coefficient count |
| `G` | integer ≥ 1 | Fitted group count; every group must occur |
| `Q` | integer ≥ 2 | Declared latent levels or unstructured coefficients |
| `structure` | integer 1–5 | 1 AR1; 2 OU; 3 CS; 4 Toeplitz; 5 unstructured |
| `X` | finite `N × P` matrix | Fixed-effect design, including an explicit intercept column if needed |
| `trials` | length-`N` nonnegative integer array | Binomial denominators |
| `y` | length-`N` nonnegative integer array | Successes, no larger than `trials` |
| `group_id` | length-`N` integer array in `1:G` | Group owning each row's coefficient vector |
| `level_id` | length-`N` integer array in `1:Q` | Coefficient selected by each row; all ones when `use_design=1` |
| `time_index` | length-`Q` integer array in `0:1000000000` | AR1/Toeplitz coordinates, strictly increasing; otherwise zeros |
| `time` | finite length-`Q` vector | OU coordinates, strictly increasing; otherwise zeros |
| `max_lag` | nonnegative integer | Toeplitz horizon, at least one and at least the integer coordinate span; zero otherwise |
| `use_design` | integer 0 or 1 | Zero selects visit coefficients; one enables an unstructured coefficient design |
| `Z` | finite `(N × use_design) × Q` matrix | Unstructured coefficient design; a zero-row matrix when disabled |
| `prior_only` | integer 0 or 1 | One omits the likelihood for prior predictive sampling |

Use `matrix(numeric(), nrow=0, ncol=Q)` for disabled `Z` in R. Check that the
chosen JSON writer preserves the intended zero-size array. No implicit predictor
centering occurs: supply exactly the design used in the matched Bambi model and
disable any additional centering there. Standardize predictors upstream, using
the same constants for both fits.

For integer times, subtract a common origin before building `time_index`; retain
actual gaps. The nonnegative bound limits integer subtraction overflow. For OU,
choose and document units before fitting: rescaling time without rescaling the
decay prior changes the model. Use moderate finite coordinates to avoid numerical
overflow or cancellation in time differences.

For CS and unstructured visit effects, `Q` includes all declared levels, including
levels not observed in a particular group. For `us(1 + x | subject)`, set `Q=2`,
`use_design=1`, and use columns `[1, x]` in `Z`. For visit effects, set
`use_design=0`; `level_id` selects the shared coefficient. Duplicate rows do not
create additional latent coefficients. Group interactions are encoded as unique
tuples upstream, not ambiguous concatenated labels.

The reference rejects out-of-range indices, invalid response counts, nonfinite
designs/time, unordered temporal support, insufficient Toeplitz horizon, unused
declared groups, and coefficient designs outside unstructured effects. It does
not reject rank-deficient designs: proper priors define a posterior, but such
fixtures can leave variance components weakly identified. The two-level minimum
is a reference-fixture restriction, not a claim that one-level effects are invalid.

## Model and matched priors

For group `g`, `coefficient[g]` is a zero-mean Gaussian vector with covariance
`D R D`. Groups are independent conditional on common covariance parameters.
The non-centered construction is `coefficient = z L'`, where `L L' = D R D`
and each element of `z` has an independent standard normal prior. No sum-to-zero
constraint or observation-level residual process is added.

| Quantity | Prior or covariance |
|---|---|
| `beta` | Independent Normal(0, 1.5), including the explicit intercept |
| Temporal/CS `sd` | One HalfNormal(2.5), the marginal latent SD |
| Unstructured `sd` | `Q` independent HalfNormal(2.5) scales |
| AR1 `rho` | Normal(0, 0.5), truncated to `(-1, 1)`; `R[i,j] = rho^abs(t[i]-t[j])` |
| OU `decay` | Exponential(rate=1); `R[i,j] = exp(-decay * abs(t[i]-t[j]))` |
| CS `rho` | Normal(0, 0.5), truncated to `(-1/(Q-1), 1)`; constant off-diagonal correlation |
| Toeplitz `partial` | `max_lag` independent Normal(0, 0.5), truncated to `(-1, 1)` |
| Unstructured correlation | LKJ(eta=2) |

Toeplitz uses the inverse Durbin–Levinson recursion to turn partial
autocorrelations into lag correlations. It then selects the covariance for the
declared coordinates from the positive-definite finite Toeplitz matrix. Missing
integer times are not collapsed. `max_lag` is a fixed horizon, not a bandwidth:
correlations beyond that horizon are undefined, not forced to zero. Do not extend
the horizon when matching fits; extra partial autocorrelations change the model.

These are explicit validation priors, not an assertion about every Bambi default.
Set the Bambi priors to match. This file estimates all applicable parameters;
fixed-covariance comparisons use the separate known-covariance reference path.

Only parameters used by the selected structure have free dimensions. Unused
scalar parameter families are zero-length vectors. Outside unstructured effects,
`L_correlation` is a one-dimensional Cholesky correlation factor fixed at `[1]`,
with no free entries. The syntax follows Stan's [optional variable and constrained
type declarations](https://mc-stan.org/docs/reference-manual/types.html) and
[LKJ Cholesky distribution](https://mc-stan.org/docs/functions-reference/correlation_matrix_distributions.html).

## Outputs and pending validation

- `coefficient`: `G × Q` latent effects, in the supplied group/coefficient order.
- `correlation`, `covariance`: `Q × Q` population matrices.
- `eta`, `probability`: observation log odds and probabilities.
- `log_lik`: conditional pointwise binomial log likelihood, including count constants.
- `y_rep`: response replications conditional on the sampled latent effects.
- `log_hyperprior`: normalized prior density for beta, marginal scales, and
  covariance parameters; excludes latent offsets and unconstraining Jacobians.
  For unstructured effects this uses correlation-matrix coordinates, not the
  Cholesky-coordinate density. Compare only densities in matching coordinates.

Run prior predictive checks before posterior fitting. Save raw draws immediately.
On suitable hardware, compile and fit each structure serially with reproducible
seeds; check R-hat ≤ 1.01, zero divergences, bulk/tail ESS ≥ 400, and energy
diagnostics before comparison. Match posterior summaries within four combined
MCSE; inspect 94% intervals and predictive behavior. Retain failed fits. Add
simulation-based calibration and prior sensitivity checks before declaring the
full readiness gate passed.

Required fixtures include negative/zero AR1 correlation, integer gaps, unequal
group observations, repeated group-level cells, irregular OU times, negative CS
correlation within its bound, nontrivial Toeplitz partial correlations, and
unstructured visit plus intercept/slope designs. Compare each matrix against an
independent deterministic calculation before sampling. Use a fully observed
crossed design for recovery, plus deliberately weak designs to document limits.

This single-block source does not implement out-of-sample conditional draws,
multiple additive blocks, other outcome families, or an automated Bambi comparison
runner. Those remain separate validation tasks; the four-block AR1 source covers
the additive reference model.
