# Structured covariance validation record

Date: October 17, 2026. Status: **draft implementation; acceptance is incomplete**.

The confirmed defects below are fixed. Eight small Gaussian posterior comparisons, six fixed-parameter glmmTMB references, and focused regressions pass. GitHub CI is running. Completed and inspected results are recorded below; the full acceptance matrix is not complete. No useful large-case marginalization benefit is established. Execution smoke checks are not evidence for those criteria.

## Checkouts and environment

- Bambi: `/Volumes/T9/Projects/bambi-cv-redo`, branch `codex/covariance-requirements`, base `ef4097dc61589c74c6e6c3c48a989a9be6c1553b`.
- Formulae: `/Volumes/T9/Projects/formulae-structured`, branch `codex/structured-expressions`, base upstream `90b41e3`. The archived checkout and abandoned covariance branch were not imported.
- Host: Apple M2, 8 GiB RAM, macOS 14, arm64. One test or sampler process at a time; all chains run serially with `cores=1`.
- Isolated task environment: Python 3.12.13, PyMC 6.3.2, PyTensor 3.3.4, nutpie 0.16.11, JAX 0.7.2, NumPyro 0.19.0, BlackJAX 1.2.5, NumPy 2.2.6, SciPy 1.15.3, arviz-stats 1.3.3. R 4.4.3 and glmmTMB 1.1.11 were already installed; no reference toolchain was installed locally.
- Paired editable installation: `uv pip install --python .venv/bin/python --no-deps -e . -e /Volumes/T9/Projects/formulae-structured`. Release metadata has no local override. Package version strings identify the base commits, not these uncommitted source changes.
- Initial inherited-environment failures involved readline, xarray, and h5py. Checks were rerun in the isolated environment, using h5py 3.14.0. Startup failures are not model evidence.

Local commands, logs, configuration, seeds, data, truth, NetCDF draws, diagnostics, and measurements are retained in `.cache/structured-acceptance/` and `.cache/structured-covariance/`. Saved review evidence includes [posterior results](structured-covariance-posterior-results.json), [reference results](structured-covariance-reference-results.json), and [sampler execution results](structured-covariance-sampler-results.json). Publication is authorized. Formulae is published at `2fbeb11d89a67db2700557872896dea0b5c5d9ab`, with [draft PR #1](https://github.com/tjburch/formulae/pull/1). Bambi is published in [draft PR #10](https://github.com/tjburch/bambi/pull/10), initially at `205730c`, with CI/reference corrections at `2eb4c5a`.

## Confirmed defects and repairs

| Gap found in actual code | Repair and evidence |
| --- | --- |
| `auto_scale=False` still divided coefficient SD prior scales by predictor SD. | `StructuredTerm.build_priors` now applies design scaling only when automatic scaling is enabled. Six tests cover `diag`, heterogeneous `cs`, and `us`, with scaling on/off; explicit priors retain their values. |
| An ungrouped process produced no row keys: pandas iteration over a zero-column frame is empty. | `group_keys` emits one empty tuple per observation. Five wrapper tests build latent and marginal graphs, draw priors, and predict repeated/new times for the single process. |
| A vector parameter inside a stochastic correlation prior silently introduced several correlations instead of one shared scalar. | Scalar shape validation rejects vector-valued correlation prior arguments. Per-coefficient SD prior-list entries also reject vector arguments; vector parameters on one broadcast SD prior remain supported. |
| `set_priors` did not resolve structured aliases, and equivalent update spellings could collide or overwrite component updates. | `normalize_prior_updates` resolves canonical, equivalent, qualified, and unambiguous alias names before updating. Tests cover ambiguous aliases, collisions in either order, and parent/auxiliary merges. |
| Installed PyMC 6.3.2 `LKJCorr` did not give the intended LKJ law for dimensions above two. Its Cholesky-coordinate log density omitted the correlation-to-Cholesky Jacobian; its forward onion generator used the wrong successive Beta shape. | US now uses independent Beta canonical partial correlations with `alpha_j = eta + (K-j-2)/2` for zero-based column `j`, followed by a triangular factor. No matrix-valued prior transform or version-specific workaround remains. Independent density/Jacobian tests cover dimensions 2, 3, 4 and eta 0.5, 1, 2, 5. Prior moments use 8,000 tiny draws per dimension, exact fourth moments, and five-MCSE bounds. Corrected 2D/3D US executes on all four samplers; both posterior pairs were rerun. |
| Validation could miss tail-ESS failures, misclassify fixed 0.1 as varying because of floating-point standard deviations, omit nutpie's tree-depth flag, or compare incompatible runs. | Explicit bulk/tail ESS gates, exact range-based constant exclusion, tree-depth name normalization, independent method seeds, and configuration/data/truth/target identity checks are in the focused scripts. Guard tests cover these failures. |

The [PyMC LKJ bug report](pymc-lkj-bug-report.md) explains discovery, independent derivations, the standalone reproducer, and a separate absolute-normalization error confirmed while writing that report. All 12 density/Jacobian cases pass after adding the uniform `eta=1` cases.

For the LKJ discrepancy, at K=3 and eta=2 each correlation must have second moment 1/6. The installed forward generator instead gives 1/6 for the first pair and 1/7 for the others; the installed log density implies 1/5 for the first pair and 1/6 for the others. This was independently derived in read-only mathematical review. Tests compare the replacement with the correlation-matrix density and its Jacobian, not with that affected implementation.

## Requirements audit against code

| Requirement | Implemented behavior and local evidence | Pending acceptance |
| --- | --- | --- |
| Formulae ownership and syntax | Opt-in registered wrappers, both syntaxes, quoted/bare lists, nesting, duplicate identity, variable discovery, ordinary arithmetic isolation, and fitted coefficient designs. 162 focused Formulae regressions pass. | Full Formulae suite passes in PR CI; released interface remains pending. |
| Coordinates and data | Gaps, repeated states, declared grids, datetime elapsed units, typed composite keys, one shared missing-row mask, and useful invalid-input errors have focused tests. | Broader regression matrix. |
| All five covariance definitions and priors | Stationary temporal kernels; coefficient basis; both CS scale variants and negative range; one-dimensional limits; proper truncation; fixed matrix/shape validation; corrected unscaled defaults and LKJ law. Dense likelihood/gradient and prior-law tests pass. | Prior-sensitivity refits and repeated prior-predictive calibration. |
| Inference API and lifecycle | Auto dimension-based selection with qualified-name tie break, explicit selection/errors, latent fallback, build/fit switching, prior/alias invalidation, and metadata have focused tests. | Full Bambi Python/linker CI. |
| Exact one-term partial marginalization | Groupwise temporal filtering and coefficient-space algebra retain other sampled effects and modeled noise. Dense references, recovery checks, and eight converged posterior pairs pass. | Repeated recovery/calibration and large-case scaling/benefit. |
| Additive and auxiliary composition | Fixed, ordinary, HSGP, other structured, and auxiliary terms have construction/execution coverage. Converged AR(1) composition includes an ordinary team intercept, a separate mean diagonal slope, and a diagonal auxiliary scale term. | Full family/integration regression jobs. |
| Four required samplers | PyMC, nutpie, NumPyro, and BlackJAX each execute every wrapper and both inference paths, both CS scale variants, negative AR(1), HSGP composition, and 3D US. | Meaningful AR(1) posterior pairs on the other three backends are configured for PR CI. Nutpie is the primary statistical validator. |
| Supported response axes | Latent Bernoulli/count/ordinal/categorical cases have focused coverage. Ten multinomial/Dirichlet-multinomial cases cover every wrapper, prior draws, response dimensions, predictions, and count totals. | Broader existing supported-family suite. |
| Recovery and prediction | Direct Gaussian conditioning checks full joint covariance, including group resets and other fixed contributions. Observed states, repeats, joint bridges/future states, new groups, numeric slopes, declared categorical domains, reproducibility, empty inputs, and zero-effect controls have focused tests. | Repeated heldout predictive calibration. |
| Results | Aliases, interpretable dimensions, prior/posterior prediction, conditional observation log likelihood, and NetCDF saving have focused coverage. | Full regressions and documentation build. |
| Statistical references | Six matched-design, fixed-parameter glmmTMB comparisons pass. Eight small posterior pairs pass below. | 100-replication SBC/ranks/coverage and actual prior-sensitivity refits. |
| Performance | Cost instrumentation records construction, compilation, sampling, recovery, prediction, memory, common covariance ESS per total second, and log-density evaluation scaling. | Representative larger benchmarks and useful measured benefit. No efficiency acceptance yet. |
| Delivery | User guide, API documentation, examples, changelogs, two separate draft descriptions, PR-triggered acceptance jobs, and shared exact-commit checkout are prepared. | Authorized publication, CI evidence inspection, Formulae release/minimum, documentation CI. |

## Independent numerical and external references

`tests/test_structured_gaussian.py` uses independent SciPy multivariate Gaussian densities and direct conditioning. Coverage includes negative, zero, and near-boundary AR(1); OU gaps; unsorted replicated heteroscedastic measurements; grouped resets with another nonzero mean contribution; coefficient Woodbury likelihoods; both CS scale variants; negative CS near its lower bound; unrestricted covariance; full conditional recovery covariance; and joint interpolation/extrapolation with repeated targets.

Float64 density tolerances are generally 1e-9 to 1e-10. Interior finite-difference steps are 1e-6, relative tolerance 1e-5 and absolute tolerance 1e-7. Tiny conditional simulations use 6,000 draws and fixed-seed tolerances of 0.01–0.025. LKJ density checks use the independently computed correlation-matrix Jacobian. Seeds are descriptive-string sums in the tests.

The R/Python reference scripts compare AR(1), OU, diagonal, heterogeneous CS, homogeneous CS, and US with glmmTMB at fixed covariance and residual parameters. Population parameters are fixed; this is not a comparison of frequentist estimates with Bayesian posteriors. The AR fixture retains a missing level with an NA-response row, avoiding silent AR grid compression. Raw `VarCorr`, design matrices, conditional modes, conditional covariance, predictions, joint log likelihood, R session details, and Python comparisons are saved. The maximum absolute log-density error is 1.07e-14. Maximum conditional mean and covariance Monte Carlo errors are 0.00445 and 0.00255. All six cases pass.

```sh
Rscript --vanilla bambi/examples/structured_covariance_reference.R .cache/structured-acceptance/reference
uv run --no-sync python bambi/examples/structured_covariance_reference.py .cache/structured-acceptance/reference
```

## Meaningful small posterior comparisons

The experiments use independent dense NumPy generating kernels, fixed generating parameters, eight groups and 68 observations, unequal state counts, actual gaps, repeats, correlated intercepts/slopes, and a negative AR correlation. The seed is `sum(map(ord, "structured-covariance-statistical-acceptance")) = 4515`. Latent and marginal sampling seeds are 4615 and 4715, with separate prediction offsets. Each fit runs through `Model.fit`, using nutpie, four serial chains, `target_accept=0.97`, and explicit priors with `auto_scale=False`.

AR(1), OU, US, 3D US, and composition use 2,000 retained draws and 1,500 tuning steps per chain. Diagonal and both CS variants use 1,000 retained draws and 1,000 tuning steps per chain. Data/configuration/truth files match between each pair. All fits pass R-hat <=1.01, bulk and tail ESS >=100 per chain, zero divergences, no maximum tree-depth hits, and E-BFMI >0.3. Diagnostics include varying nuisance parameters and recovered states.

Cells below show **maximum R-hat / minimum bulk ESS / minimum tail ESS**. The last column compares the largest absolute mean-difference/combined-MCSE statistic with a Bonferroni family-wise 1% threshold, accounting for all tested scalar means and second moments in that pair.

| Case | Latent diagnostics | Marginal diagnostics | Max MCSE z / threshold |
| --- | --- | --- | --- |
| ar1 | 1.0027 / 1135 / 1445 | 1.0025 / 2709 / 1862 | 3.29 / 4.01 |
| ou | 1.0009 / 1780 / 2433 | 1.0012 / 5051 / 4227 | 3.19 / 4.01 |
| diag | 1.0067 / 656 / 950 | 1.0023 / 2471 / 2013 | 2.18 / 3.85 |
| cs | 1.0090 / 812 / 1131 | 1.0026 / 2218 / 2171 | 1.49 / 3.86 |
| cs_homogeneous | 1.0067 / 716 / 1069 | 1.0038 / 2024 / 2091 | 1.77 / 3.85 |
| us | 1.0017 / 1703 / 2204 | 1.0013 / 4361 / 4351 | 2.32 / 3.86 |
| us3 | 1.0024 / 2066 / 2928 | 1.0009 / 5052 / 4257 | 2.12 / 3.94 |
| composition | 1.0034 / 1401 / 1877 | 1.0032 / 2870 / 2753 | 2.69 / 4.04 |

Agreement covers fixed effects, residual scale, covariance parameters, every recovered state/coefficient marginal, first and second moments, a designated 4x4 state cross-moment block, joint 4x4 prediction cross moments, and independently reconstructed covariance matrices. Predictions include interpolation/extrapolation, repeated states, numeric slope changes, and a new group. Equality of duplicate-state predictions is checked exactly. These results establish small-case posterior agreement. A single generating dataset per case is not repeated recovery or calibration evidence.

Prior predictive central 99% ranges are finite (roughly ±4.6 for base temporal cases, up to [-8.18, 7.59] for the quadratic US case). In-sample posterior predictive mean/SD tail probabilities are roughly 0.46–0.51. Those are descriptive checks, not calibrated PIT claims. Per-fit elapsed time is about 17–37 seconds, with measured process peak RSS 0.39–0.62 GB. Local fits used no parallel chains. These measurements supported the small local checks; they do not imply that large calibration or benchmarks should run locally.

Example commands, using a separate process for each fit:

```sh
uv run --no-sync python bambi/examples/structured_covariance_validation.py fit ar1 latent --groups 8 --draws 2000 --tune 1500 --chains 4 --output .cache/structured-acceptance/ar1/latent
uv run --no-sync python bambi/examples/structured_covariance_validation.py fit ar1 marginal --groups 8 --draws 2000 --tune 1500 --chains 4 --output .cache/structured-acceptance/ar1/marginal
uv run --no-sync python bambi/examples/structured_covariance_validation.py compare .cache/structured-acceptance/ar1/latent .cache/structured-acceptance/ar1/marginal --output .cache/structured-acceptance/ar1
```

The script captures sampler output in `sampler-raw.nc` before Bambi cleanup/recovery. It saves the recovered posterior before diagnostics/prediction, plus 94% HDI summaries, prior/PPC draws, heldout response draws, joint new-state/group predictions, data/truth, commands, versions, and resources. Early sampler failure cannot produce draws; its command/configuration and CI log remain available.

### Failed checks retained

Initial 1,000-draw AR(1) latent runs failed R-hat (1.012) and bulk/tail ESS (396/371). The initial OU latent run failed R-hat (1.0104). Tuning/draw counts were increased on the same data and seeds; these were not seed searches. Original results are retained under `ar1-1000/`, `ou-1000/`, and their logs.

An earlier composition assessment wrongly included an exact fixed 0.1 because its computed standard deviation was positive from roundoff. The constant-selection test now uses exact max/min, and saved draws were reassessed. The composition experiment was then extended to include a separate sampled mean diagonal slope and an auxiliary diagonal term and both methods were rerun. The prior composition results remain in `composition-before-extra/`. US results from before the Beta replacement remain in `us-before-beta/`; current US pairs use the replacement.

The final tiny benchmark pipeline rerun found a JSON serialization error because the compiler cache path was a `Path` object. Converting that metadata to a string fixed it. The rerun completed in 8.41 seconds wall time with peak RSS 412,336,128 bytes and zero swaps; raw/recovered/prediction files and measurements were inspected. Its one-chain/four-draw diagnostics remain ineligible for performance acceptance. The failure log is retained as `benchmark-protocol-path-failure.log`.

One independent prior-draw pipeline replication for AR(1) and one for 3D US passed fit diagnostics and exercised true heldout new times/slopes and raw-output capture. The calibration aggregator correctly failed with insufficient replications (one versus 100). These are pipeline checks, not passed calibration acceptance; logs/results remain in `protocol/` and `protocol-us3/`.

## Four-sampler execution matrix

There are **72 recorded execution cases**: five wrappers, two methods, four samplers; both CS variants; negative AR(1); ordinary/HSGP composition; and 3D US. Both 2D and 3D US were rerun across all four backends after the Beta correction. Each case uses one chain/core, four tuning steps and four retained draws, seed 3550, and a tiny six-observation dataset. Prior prediction, fit, recovered effects, existing/new-group prediction, conditional observation log likelihood, prior prediction after fitting, and NetCDF saving execute.

The matrix has **30 divergences across 288 retained draws in 14 cases**. Four tuning steps do not establish convergence. Those draws are not used for scientific estimates, calibration, or ESS comparisons. The workflow separates this compatibility matrix from meaningful posterior checks.

```sh
uv run --no-sync python bambi/examples/structured_covariance_smoke.py ar1 marginal nutpie
uv run --no-sync python bambi/examples/structured_covariance_smoke.py cs latent blackjax homogeneous
uv run --no-sync python bambi/examples/structured_covariance_smoke.py ar1 marginal numpyro negative
uv run --no-sync python bambi/examples/structured_covariance_smoke.py ar1 marginal nutpie hsgp
uv run --no-sync python bambi/examples/structured_covariance_smoke.py us latent nutpie three
```

## Focused regressions and independent review

**163 distinct focused Bambi tests pass**: a full 150-test run, ten additional multinomial/Dirichlet-multinomial cases, and three additional `eta=1` density cases. The final LKJ density run passes all 12 cases in 1.34 seconds. The later 15-case run includes five previously counted ungrouped tests, expanded to check predictions and marginal construction. The grouped full-covariance recovery test also passed after extension. The 150-test run took 49.31 seconds in pytest (54.95 seconds wall), with process peak RSS 680,361,984 bytes and peak footprint 800,329,088 bytes. The final **162 Formulae tests pass** in 1.23 seconds (1.58 seconds wall), peak RSS 141,180,928 bytes.

```sh
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/time -l .venv/bin/python -m pytest -q tests/test_structured.py tests/test_structured_gaussian.py tests/test_structured_validation.py tests/test_formula.py tests/test_noncentered.py --tb=short
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_structured.py -k 'multivariate_count or ungrouped' --tb=short
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_structured_gaussian.py::test_grouped_filter_replicates_with_another_effect --tb=short
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/time -l .venv/bin/python -m pytest -q /Volumes/T9/Projects/formulae-structured/tests/test_formula_wrappers.py /Volumes/T9/Projects/formulae-structured/tests/test_parser.py /Volumes/T9/Projects/formulae-structured/tests/test_resolver.py /Volumes/T9/Projects/formulae-structured/tests/test_scanner.py /Volumes/T9/Projects/formulae-structured/tests/test_design_matrices.py /Volumes/T9/Projects/formulae-structured/tests/test_eval_new_data.py /Volumes/T9/Projects/formulae-structured/tests/test_terms_call.py --tb=short
```

Black 25.9.0 checks all 19 relevant Bambi Python files. Formulae's Black 25.12.0 check passed its source. Serial pylint 3.3.9 rates Bambi 10.00/10; the changed Formulae modules previously passed at 10.00/10. Actionlint 1.7.7 checks the changed CI workflows. `git diff --check` passes both checkouts. No full regression suite runs locally.

Read-only Sol High reviewers received the requirements and source and independently checked Formulae, likelihood/gradient/recovery mathematics, prediction, integration, acceptance gates, CI, and the LKJ measure. Review findings produced the focused fixes above. The final LKJ reviewer independently verified the actual Beta implementation and density/moment tests and found no remaining mathematical error. Reviewers ran no tests or samplers; local execution stayed with the lead and remained sequential.

## CI acceptance and artifact contract

`.github/workflows/structured-covariance.yml` now has PR triggers. Bounded jobs run:

- The API documentation and structured covariance guide build, using the repository-pinned Quartodoc commit. This runs on forks; the existing publication workflow is upstream-only.
- The full paired Formulae regression suite.
- Eleven meaningful latent/marginal posterior pairs: all structures including both CS variants and 3D US, full composition on nutpie, and AR(1) on PyMC/NumPyro/BlackJAX; 4 serial chains, 2,000 draws, 1,500 tuning steps, diagnostic and MCSE agreement failure gates.
- The 72-case execution matrix across all four samplers.
- Fixed-parameter glmmTMB reference comparisons.

The broader Bambi `ci.yml` matrix (Python 3.12/3.13, CVM/Numba linkers) uses the same exact paired Formulae checkout. It captures the Pixi interpreter before overriding Formulae, then invokes that interpreter directly for pre-commit, pylint, tests and coverage, preventing a later Pixi sync from replacing the paired dependency.

Larger jobs run with `workflow_dispatch` extended=true or, on PR opening/description edits, the PR body marker `<!-- structured-full-acceptance -->`. Ordinary code pushes rerun bounded checks without restarting already running extended jobs. The marker allows PR validation before the workflow is present on the repository default branch, as GitHub requires for manual dispatch. Extended jobs run:

- 100 independent prior-draw replications for each of seven structures, 4 serial chains and 1,500 draws/tuning. Generating kernels use dense NumPy algebra and US3 uses normalized Wishart draws independent of the backend Beta parameterization. Exact proper priors match the fitting model. Every fit must converge. Scalar parameter/state SBC ranks are conservatively thinned to at most 100 draws at roughly four estimated autocorrelation times; randomized ranks are tested for uniformity. True heldout responses at new times or numeric slopes have predictive ranks/PIT and 94% HDI coverage. Multiplicity-adjusted family-wise 99% exact binomial intervals quantify coverage uncertainty; rank tests use matching correction. MCMC ranks remain approximate and their limits must be considered during review.
- Actual prior-scale 0.5/1/2 refits for seven structures, convergence checks, mean shifts in baseline posterior SDs, and 94% HDIs. Shifts above 0.3 SD flag prior dependence for scientific review; they are not silently called robustness or implementation failures.
- AR(1)/OU benchmarks at 50, 250 and 1,000 groups, and 50-group coefficient benchmarks including both CS variants. Unequal grids, gaps, repeats, an ordinary sampled team effect, and modeled noise are retained. Each case uses a separate cold PyTensor/Numba cache. Reports include model/data/priors, all cost phases, RSS, diagnostics and covariance-parameter ESS per total second. The assessment fails on missing sizes or failed convergence, temporal log-density scaling exponent above 1.5, or failure to show at least 1.1x AR(1) total speed and covariance ESS/total-second benefit at the representative large size. These ratios are comparative acceptance gates, not machine-specific wall-time thresholds.

CI uploads logs, JUnit, commands, exact versions/commits, host/memory information, data/truth, raw and recovered draws, predictions, diagnostics, reference matrices and measurements **also on failure**. Uploads explicitly include hidden `.cache` paths; run-attempt suffixes retain evidence from earlier attempts. A green execution-only job is insufficient. The scientific JSON, diagnostics, coverage uncertainty, sensitivity and cost results must be inspected after the runs, with focused fixes and reruns for failures.

## Pending external work and dependency blockers

Publication is authorized. Formulae commit `2fbeb11d89a67db2700557872896dea0b5c5d9ab` is pushed to the user fork, with [draft PR #1](https://github.com/tjburch/formulae/pull/1) against the focused validation base. Bambi draft PR #10 and the GitHub runs below are published. Only inspected completed artifacts count as verified evidence.

`.ci/formulae.json` now contains that exact 40-character Formulae feature commit. The checkout action rejects null, branch names and non-40-character pins. Both the broader and structured Bambi jobs use this same immutable dependency. Testing the old released Formulae or a moving branch would not satisfy paired CI.

For fork-local draft validation, Bambi's `main` matches the intended base. Its default branch is the abandoned covariance branch and must remain untouched. Formulae's fork `master` predates upstream `90b41e3`; a focused validation base branch at that upstream commit permits a draft PR without unrelated history. No default branch needs changing. The PR body marker starts extended checks without relying on manual dispatch registration.

The release manifest still has `formulae>=0.6,<0.7`, which does not supply the required interface. A Formulae release containing the paired extension is needed; its assigned version must determine Bambi's released minimum/range. No version has been invented. The temporary CI override must be removed when that release dependency is coordinated. This remains a merge/release blocker even if paired development CI passes.

Acceptance remains pending for completion of the full Bambi regression matrix, meaningful non-nutpie posterior pairs, repeated SBC/PIT/coverage, remaining prior-sensitivity interpretation, representative performance and useful benefit, and the remaining CI artifact review. The full Formulae matrix and corrected documentation/reference jobs pass as recorded below. There is no known unresolved confirmed implementation defect after the repairs above; these pending checks can still reveal defects and must not be waived.

## GitHub follow-through: partial results

The [CI evidence summary](structured-covariance-ci-results.json) records inspected artifacts. Runs [39076990474](https://github.com/tjburch/bambi/actions/runs/39076990474) (initial extended acceptance), [39077354869](https://github.com/tjburch/bambi/actions/runs/39077354869) (bounded rerun after fixture/environment repairs), and the Python/linker regression runs remain in progress. The production implementation is unchanged between `205730c` and `2eb4c5a`; the latter fixes validation environments and the reference script.

- Formulae [CI 39076908729](https://github.com/tjburch/formulae/actions/runs/39076908729) passes Python 3.11–3.14: 288 passed, one skipped per job, with pre-commit and pylint passing. Logs were inspected. The separate paired full-suite artifact also records 288 passed/one skipped and the exact expected Formulae commit.
- The first full Bambi Python 3.12/CVM artifact has 530 passed, one skipped, zero failures/errors (531 collected), pytest time 499.672 seconds, pylint 10.00/10, and the expected Formulae pin. Other Python/linker jobs remain pending.
- AR(1), OU, diagonal, both CS variants and 2D US posterior artifacts pass both convergence and MCSE agreement. These CI datasets use 12 groups, four serial chains and 2,000 draws/1,500 tuning steps. The inspected maximum pairwise z statistics are 2.58, 3.03, 1.84, 2.12, 1.53 and 2.09, below their multiplicity-adjusted thresholds. Raw draws and 94% HDI summaries were retained and selected parameter summaries inspected.
- AR(1) prior-scale 0.5/1/2 refits converge with no shifts above 0.3 baseline posterior SD. OU refits converge; halving its SD prior scale shifts the SD mean by 0.308 baseline posterior SD, triggering the prespecified prior-dependence flag. The baseline 94% HDI [0.478, 0.844] changes to [0.473, 0.788]; doubling gives [0.492, 0.855]. This is reported as prior dependence, not an implementation failure or blanket robustness claim.
- The initial documentation job failed because the pip-only job installed Griffe 2 while the repository constrains Griffe to 1. Applying that existing constraint fixes the rerun. The API build and guide render pass; the rendered 57,748-byte guide artifact was inspected for the new API and scaling sections.
- The initial reference job failed before fitting: Ubuntu supplies glmmTMB 1.1.8 with `betad`/log variance, while local 1.1.11 uses `betadisp`/log SD. The fixture detects the parameter name, fixes the requested SD under the appropriate convention, and asserts SD 0.3 in Python. All six local references pass again. The corrected CI artifact on R 4.3.3/glmmTMB 1.1.8 also passes all six cases, with residual SD 0.3 and maximum absolute log-density error 1.25e-14. Failure logs and artifacts remain available.

Repeated calibration, large benchmarks, remaining sampler/posterior jobs, remaining sensitivity interpretation, and full regression completion are still pending. No scientific pass is inferred from an execution-only green job.

## Earlier small timing records

[Earlier small benchmark measurements](structured-covariance-benchmark-results.json) retain twelve 8-group nutpie runs with 10 tuning/10 retained draws and shared compiler caches. They measure execution costs only. Their apparently quicker temporal sampling was offset by recovery, and coefficient marginal construction cost more at that size. They establish no convergence, efficiency, linear scaling, or useful total benefit. Current acceptance uses the explicit converged, cold-cache CI protocol above. The final 2-group benchmark pipeline check also has one chain/four draws and explicitly reports failed diagnostic eligibility; it is not benchmark acceptance.
