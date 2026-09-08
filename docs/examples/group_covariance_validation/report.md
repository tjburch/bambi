# Validation report

Date: 2026-09-08. Status: **draft; not ready to merge**.

## Executed checks

| Check | Result |
|---|---|
| Combined covariance, harness, formula and selected ordinary-model tests | 261 passed, two hosted-only sampler tests skipped, 19.67 s |
| Production Pylint | 10.00/10 |
| Black read-only checks, changed Python files | Passed |
| `git diff --check` | Passed |
| Four R reference scripts: syntax parsing | Passed; host locale warnings |
| Small binomial R fixture generation | Passed |
| Bambi four-block AR1 prior-only reference, two draws | Passed; not posterior validation |
| Schema-v2 identity/design entry point, two prior draws | Passed after fixing CSV precision mismatch |
| Real mean/quantile MCSE helper on four small synthetic chains | Passed with JIT disabled |
| Isolated tutorial HTML render, execution disabled | Passed |

Tests used one numerical thread, the Python linker and `cxx=`. No posterior
sampling, Stan compilation, environment installation or broad test matrix was run.
The ordinary regression selection covered shared predictor data, sparse prediction
and log likelihood, group-effect exclusion and pruning.

The targeted tests include five covariance structures, the four-block model,
three-way group novelty, repeated-cell draw identity, signed AR1 gaps, new-time
conditional draws, independent conditioning moments, dense/sparse construction and
prediction, aliases, priors, missing data, omitted-offset restoration, and conditional
log likelihood. These are finite fixtures, not proof of complete API coverage.

## Small graph-build benchmark

Dense predictor, interaction depth three, four subjects, four times, one replicate:
64 observations and 144 latent coefficients. Constructor: 0.038 s. Graph build:
0.183 s. Imports: 2.766 s. Peak process RSS: 237,649,920 bytes (226.6 MiB).
No log-density compilation or sampling. This is a smoke check, not a scaling result
or a prediction of memory requirements for the user's data.

Environment: Python 3.12.3, PyMC 6.3.1, PyTensor 3.3.0, formulae 0.7.0,
NumPy 2.4.6, pandas 3.0.5, SciPy 1.18.1.
Local outputs are under `.scratch/covariance-checks/`; they are not PR artifacts.

## Remaining implementation

The local harness now includes schema-v2 cross-engine exports, independent
single-block runners, SBC orchestration, conditional joint-prediction oracles,
PPC/sensitivity reports, artifact merging and manual hosted workflows.
Edge checks cover transformed US categories, namespace constants, marginal Gaussian
densities and AR1 gradients. These changes still need hosted runtime verification.

Latest combined local run: **261 passed, two opt-in sampler tests skipped**,
19.67 seconds. This includes all covariance suites, formula tests, independent
generator checks, small public prediction draws and mocked posterior-export
pipelines. The mocked pipelines run real likelihood/prior/prediction processing,
but replace fitting with three prior draws; they do not test posterior sampling.

Held-out refit runners and predictive-scoring tests are provided. Complete their
hosted execution, plotting/report coverage and dependency-lock
review on the hosted pilot. Resolve findings before freezing the full campaign.
See [manual CI instructions](CI.md) and [SBC protocol](SBC.md).

## Remaining execution gates

- Native brms known-covariance and learned-unstructured comparisons; independent
  Stan learned-covariance comparisons, including all four additive AR1 blocks.
- Prior/posterior predictive checks, held-out refits, sensitivity and at least 100
  prior-drawn SBC replicates per structure. Record uncertainty and failed runs.
- Supported Python/backend/sampler and ordinary-model regression matrices.
- Scaling benchmarks and full documentation build.

Use suitable hardware for these gates. Require matching designs and priors,
R-hat <= 1.01, bulk/tail ESS >= 400, zero divergences, E-BFMI >= 0.3, no unresolved
tree-depth warnings, and reference agreement within four combined MCSE.
Scripts and successful prior draws do not establish posterior agreement.
