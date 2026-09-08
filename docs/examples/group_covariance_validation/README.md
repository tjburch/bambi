# Group-specific covariance reference validation

Status: **posterior validation pending**. The R fixture generator and a two-draw
Bambi AR1 prior-only smoke run passed. R sources parse. Stan compilation, posterior
fits, summary exporters and cross-engine agreement have not been validated.
They do not establish posterior agreement or PR readiness. No dependencies were
installed and no reference fits were run during implementation.

## Models and matching contract

- `reference.R known`: four independent known correlation blocks, each with its
  own learned marginal SD. Each unique group/time cell is one brms grouping level.
  `K1` through `K4` have zero covariance between different groups and the specified
  correlation within each group. Replicated observations reuse their cell ID.
- `reference.R us-slopes`: correlated intercept and `x1` slope within subject.
- `reference.R us-visits`: unstructured visit coefficients within subject.
- `four_block_ar1.stan`: four independent learned AR1 blocks: subject,
  subject:condition, subject:context, subject:condition:context. A stationary
  transition over the actual integer gap integrates out unobserved intermediate
  times. Marginal SD is not innovation SD. Work grows with unique cells, not an
  observation covariance matrix.
- `bambi_reference.py`: matching Bambi learned/fixed AR1 and unstructured fits,
  with explicit priors and serial nutpie. Posterior draws are saved immediately
  to NetCDF before prediction, likelihood calculation, or summary diagnostics.
- `export_summary.R` and `compare_summaries.py`: common posterior-mean summaries
  and a fail-closed four-combined-MCSE check. These scripts remain unrun.

Native brms supports known covariance through `gr(..., cov=K)` and correlated
group coefficients through ordinary group terms. The known matrix must carry
group-level row names. See the official [gr reference](https://paulbuerkner.com/brms/reference/gr.html).
Explicit SD and LKJ priors follow the [brms prior reference](https://paulbuerkner.com/brms/reference/set_prior.html).
brms residual `ar()` is not used as a substitute for these additive latent blocks.

All runners use binomial-logit likelihoods. `trials=1` is exactly Bernoulli.
Use the identical trials, successes, row order, and design in Bambi. Expanded
Bernoulli versus aggregated binomial comparisons must account for combinatorial
likelihood constants and require identical predictors within each aggregated cell.

Matching priors:

| Parameter | Prior |
|---|---|
| Fixed coefficients, including explicit constant | Normal(0, 1.5) |
| Each block SD; each unstructured coefficient SD | HalfNormal(0, 2.5) |
| Each learned AR1 correlation | Normal(0, 0.5), truncated to (-1, 1) |
| Unstructured correlation | LKJ(2) |

brms and Stan use an explicit constant column `one`, with `0 + one + x1 + x2`.
Bambi uses its ordinary `Intercept` with `center_predictors=False` and the same
explicit Normal(0, 1.5) prior; a constant common term is rejected by Bambi.
The exporter calls both intercepts `beta.one`. Do not standardize or center one engine's data
independently. Override Bambi fixed-effect and covariance priors explicitly.
For the known-covariance comparison fix each Bambi correlation parameter to the
corresponding fixture value; do not compare that reference to a learned-rho fit.
For unstructured visits, match visit-level ordering and use full indicator coding.
Inspect saved brms `generated.stan` and `standata.rds` before accepting equivalence.

## Resource limits and execution

Use a separate reference environment with R, brms, cmdstanr, CmdStan, posterior,
and jsonlite already installed. The scripts do not install dependencies.
Compilation and sampling are intended for suitable hardware, not an automatic
laptop test run. Every runner limits numerical threads and concurrent chains to
one. Multiple chains run **serially**. Never overlap these commands with Bambi
sampling, compilation, or another reference run.

From this directory, on the validation runner:

```sh
Rscript prepare_reference.R binomial results/fixture-binomial 240913
Rscript reference.R known results/fixture-binomial/input.rds results/known-prior prior 4 1000 1000 240914
```

Inspect the saved prior draws and prior predictive distribution before proceeding.
The fixed four-block prior is broad on the logit scale; near-zero/one probabilities
can be expected and must be reported, not hidden by changing just one engine.

```sh
Rscript reference.R known results/fixture-binomial/input.rds results/known-posterior posterior 4 1000 1000 240915
Rscript stan_reference.R four_block_ar1.stan results/fixture-binomial/data.json results/ar1-prior prior 4 1000 1000 240916
```

Inspect the AR1 prior run before its posterior run:

```sh
Rscript stan_reference.R four_block_ar1.stan results/fixture-binomial/data.json results/ar1-posterior posterior 4 1000 1000 240917
```

Use `prepare_reference.R bernoulli` for a separate Bernoulli fixture. Native
unstructured modes use the same `input.rds` format and require their own prior
and posterior runs. This fixture was generated under four AR1 blocks: fitting
unstructured models to it tests cross-engine agreement, **not** unstructured
parameter recovery. Separate prior-drawn fixtures are needed for calibration.

Output directories must not already exist. Inputs, generated models, settings,
versions, and raw draws are retained. Fits are saved before post-processing.
For a full analysis, produce a separate report containing diagnostics, 94% HDIs,
predictive checks, sensitivity, failures, and limitations. Do not overwrite a
failed run with a successful rerun.

## Bambi fit and comparison

Use the repository environment with nutpie already installed. Set cache paths
to the external drive before running. The following commands are separate serial
jobs on the validation runner, not permission to start a broad local run:

```sh
uv run --no-sync python bambi_reference.py ar1 results/fixture-binomial/data.csv results/bambi-ar1-prior prior --chains 4 --warmup 1000 --draws 1000 --seed 240918
```

Review its prior predictive output before the posterior job:

```sh
uv run --no-sync python bambi_reference.py ar1 results/fixture-binomial/data.csv results/bambi-ar1-posterior posterior --chains 4 --warmup 1000 --draws 1000 --seed 240919
Rscript export_summary.R stan ar1 results/ar1-posterior/fit.rds results/fixture-binomial/data.csv results/ar1-posterior/summary.json
uv run --no-sync python compare_summaries.py results/bambi-ar1-posterior/summary.json results/ar1-posterior/summary.json
```

For native known-covariance comparison, use Bambi mode `known` with
`--rho 0.6 -0.35 0.2 0.45`, matching this fixture's matrices. Export the R fit
with engine `brms` and mode `known`. Both engines also accept `us-slopes` and
`us-visits`; the exporter requires the exact expected coefficient names and
fails rather than guessing when a brms version uses different names.

The common JSON schema contains `schema_version=2`, `engine`, `mode`,
`phase=posterior`, an MD5 digest of the original fixture CSV (`data_md5`),
`identity`, `diagnostics`, and `metrics`. The identity records source commit and
source/input SHA-256 hashes, explicit designs and priors. Exporters verify their
design mappings. MD5 remains a fixture identifier, not a security guarantee.
Each metric has finite `mean`, nonnegative `mcse_mean`, and 3%, 50%, 97% quantiles
with their MCSE. Exact constants have zero MCSE. Required names include
`beta.one`, `beta.x1`, `beta.x2`, contiguous `probability.1` through
`probability.N`, and covariance parameters (`sd.1` etc.; `rho.1` etc. for
learned AR1; `cor.1.2` etc. for unstructured). Fixed rho values are not random
metrics. Probability means are inverse-link averages, not posterior medians.
The brms exporter uses inverse-link linear predictions to avoid confusing
binomial expected counts with probabilities, including rows with zero trials.

The comparator rejects missing or different metric sets, mismatched fixture
digests or modes, missing/nonfinite diagnostics, fewer than four chains,
R-hat above 1.01, ESS below 400, divergences, E-BFMI below 0.3, or tree-depth
saturation. Exporters also inspect latent coefficient/offset diagnostics.
A missing backend diagnostic field stops summary creation; retained raw draws
allow an explicit backend adapter to be added without refitting. No absent
diagnostic is treated as a pass. Posterior exporter/comparator execution remains
pending; completed smoke checks are listed in [the report](report.md).

Exports also include mapped latent coefficients, conditional pointwise likelihood,
and exact predictive mean, second moment and zero probability. These continuous
predictive moments avoid unstable MCSE for discrete predictive quantiles. The
comparator never certifies overall PR readiness: conditional joint prediction,
held-out refits, calibration, sensitivity and regression gates are separate.

Generate `identity-{mode}.json` beside the fixture CSV with
`validation_identity.py` before fitting either engine. Prefer `run_reference_case.py`
to execute the stages in order. Use [manual CI instructions](CI.md) for bounded
hosted batches and [SBC instructions](SBC.md) for campaign accounting.

## Exact input schemas

The fixture builder creates all inputs with fixed truth and a supplied seed.
It includes unequal trials, year gaps, replicated cells, a missing year within
one subject, and a withheld three-way combination whose lower-order pairs remain.
Six subjects make this a small smoke fixture, not evidence of precise variance
recovery. Larger crossed fixtures are required for the recovery gate.

`reference.R` input is an R list saved with `saveRDS`:

- `data`: data frame containing integer `y`, integer `trials`, finite numeric
  `x1`, `x2`; `subject` for unstructured modes; `visit` for visit mode.
- Known mode also needs `data$cell1` through `data$cell4`, and matrices `K1`
  through `K4` at the top level. Each matrix is symmetric positive definite with
  diagonal one. Its row and column names are the corresponding cell IDs.
  `fixed_rho` must contain the four correlations used to construct those matrices;
  the runner checks the matrices against the grouping and time coordinates.
- The runner adds `one=1`. Unstructured slopes use `x1` as their slope.

The Stan JSON input contains:

| Field | Shape and meaning |
|---|---|
| `N`, `P` | Positive observation and fixed-coefficient counts |
| `X` | N × P fixed design, explicit constant included |
| `trials`, `y` | N nonnegative integers, `y <= trials` |
| `C`, `G` | Total unique coefficient cells and groups across all four blocks |
| `block_id` | C integers in 1..4 |
| `group_id` | C integers in 1..G; group IDs unique across blocks |
| `time` | C integer years; use a modest origin to avoid integer overflow |
| `previous` | C indices; zero for first cell, otherwise preceding same-group cell |
| `row_cell` | N × 4 one-based cell indices, one column per block |
| `prior_only` | 0/1; overwritten by the runner's phase argument |

Cells must precede their successors and increase strictly in time within each
group. A repeated observation maps to an existing cell, not a second latent
coefficient. JSON array dimensions must remain intact. `cells.csv` records the
fixture's coefficient ordering for the Bambi-to-Stan mapping.

## Outstanding acceptance gates

1. Compile and smoke-test the R/Stan scripts on the reference runner.
2. Run the Bambi and R exporters on actual posterior draws; verify their parameter
   mapping, exact designs, priors, cell identity and generated likelihoods. Local
   mapped-latent/quantile tests do not establish posterior agreement.
3. Check each fit: R-hat ≤ 1.01; zero divergences; bulk/tail ESS ≥ 400;
   E-BFMI ≥ 0.3; no unresolved tree-depth or rank-plot warnings.
4. Compare means and selected quantiles of fixed effects, marginal scales,
   correlations, latent coefficients, probabilities, predictive summaries, and
   pointwise likelihoods within four combined Monte Carlo standard errors.
   Diagnose discrepancies before increasing draws.
5. Complete prior/posterior predictive checks, 94% intervals, discrete calibration,
   hyperprior sensitivity, and held-out-subject/future-time refits. The Stan
   `log_hyperprior` supports sensitivity to top-level priors, without power-scaling
   all latent standard normals.
6. Compile and run the connected independent single-block references in
   `structured_block.stan` (see `STRUCTURED_STAN.md`), the joint prediction checks,
   held-out refits and 100 prior-drawn calibration replicates per structure.
   Runners are provided; hosted execution remains pending. Retain failures and
   uncertainty bands; fixed-truth recovery is not simulation-based calibration.
7. Record exact package/compiler versions, hardware, peak RAM, runtime, and actual
   passing results in the validation report. All comparison gates remain pending.
