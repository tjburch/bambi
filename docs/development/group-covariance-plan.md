# Group-specific covariance: implementation and readiness

## Objective

Add independent latent Gaussian coefficient blocks with AR1, continuous-time AR1
(`ou`), exchangeable (`cs`), finite Toeplitz (`toep`), and unstructured (`us`)
covariance. Keep ordinary group-effect defaults and observation likelihoods.

Acceptance model: four additive AR1 blocks grouped by `subject`,
`subject:condition`, `subject:context`, and `subject:condition:context`.
Test Bernoulli and binomial outcomes; cover Gaussian, Poisson and negative-binomial
construction too. Support dense/sparse predictors, centered/non-centered coefficients,
aliases, prior updates, repeated cells, actual time gaps, and joint prediction.

## Preservation

Residual work was archived before source changes: 521 files, verified against
SHA-256 hashes. Tracked/untracked changes were stashed with name
`residual-correlation-preserved-20260907` and commit
`3593732b6d42de67b02225f7d6845821853c12c3`.

Archive, manifest, original status, patch and recovery instructions:
`.scratch/residual-validation/preservation-20260907/`.
Ignored results and personal HTML are preserved separately from the stash.
Environments and compiler caches remain in place and are not archived.
Local Git exclusions protect these artifacts after restoring the baseline ignore file.

Current branch: `structured-group-covariance`, based on
`1420a967f6de4fa783a27712f303505ee48399a1`. Publication is isolated from upstream;
hosted validation results are recorded separately from local checks.
For recovery, use `git stash apply` with the recorded commit on a separate branch
at that baseline. Do not pop the stash or extract the archive over active work.

## Statistical contract

- Each block has its own covariance parameters, shared by its independent groups.
- AR1 uses stationary marginal variance and signed correlation at actual integer gaps.
- OU uses finite numeric time and positive decay; time units affect its prior.
- Toeplitz uses partial autocorrelations for positive definiteness; the union of
  fitted and predicted times must fit within `max_lag`.
- Exchangeable bounds depend on declared coefficient levels, not observation count.
- Temporal/exchangeable structures have one marginal SD per block. Unstructured
  blocks have one SD per coefficient and an LKJ correlation prior.
- Heterogeneous temporal scales and scale regression are deferred. No speculative
  framework or default scale extrapolation is included.
- Existing cells retain fitted coefficients. New times are drawn jointly conditional
  on fitted coefficients; new groups receive joint population trajectories.
- Group novelty is determined separately per block. Replicate rows share a draw.
- Pointwise log likelihood remains conditional on latent effects. Future-time and
  held-out-subject validation require appropriate refits, not ordinary row-wise LOO.

## Readiness gates

| Gate | Status |
|---|---|
| Archive verification, named stash, clean baseline | Passed |
| Wrapper parsing, independent block construction, five structures | Implemented; small checks passed |
| Four-block Bernoulli/binomial dense/sparse construction | Passed |
| Joint new-time/new-group prediction, repeated cells, novelty by block | Small checks passed |
| Aliases, prior updates, exclusion, shared missing-data mask | Small checks passed |
| Independent NumPy covariance and conditioning checks | Passed |
| Density/gradient/moment and invalid-input coverage | Expanded deterministic checks passed; backend matrix pending |
| Native brms known covariance and learned unstructured | Scripts provided; fits/comparison pending |
| Independent Stan learned four-block AR1 | Scripts provided; fits/comparison pending |
| Independent learned OU/CS/Toeplitz references | Runners/exporters connected; hosted compilation/comparison pending |
| Prior/PPC, held-out prediction, sensitivity, discrete calibration | Runners provided; hosted execution/report review pending |
| 100 prior-drawn SBC replicates per structure | Frozen 600-case runner provided, including four-block AR1; execution pending |
| Python 3.12–3.14 × CVM/NUMBA, sampler matrix | Pending |
| RAM/time benchmarks over groups, visits, replication, interaction depth | Bounded serial campaign provided; hosted scaling runs pending |
| Lint/format checks and documentation render | Passed for changed Python and isolated tutorial; full docs build pending |

Executed checks and remaining implementation work are recorded in the
[validation report](../examples/group_covariance_validation/report.md).

Deterministic tolerances: `rtol=1e-7`, `atol=1e-8` on well-conditioned fixtures.
Reference fits must match design matrices, priors, scale meanings and prediction
targets. Require R-hat <= 1.01, zero divergences, bulk/tail ESS >= 400, acceptable
energy diagnostics, and selected summaries within four combined MCSE. Diagnose
disagreement before increasing draws. Report 94% intervals and uncertainty in SBC
coverage/ranks. Retain failures. Scripts alone do not satisfy a validation gate.

## Resource policy

One local process at a time; one numerical thread. Small graph and deterministic
checks use the Python linker with C compilation disabled. No broad local test run,
parallel sampling, or heavy reference compilation. Save results on the external drive.
Use suitable hardware for reference fits, SBC and the full regression matrix.
Do not mark unavailable checks as passed. Do not commit, push or publish a PR
without explicit approval.

Unresolved questions: No interface decisions. Suitable hardware is still needed
for the heavy validation gates. PR readiness is not established.

1. Preserve residual source, documentation and results; verify stash and archive.
2. Implement formula lowering, coefficient blocks, covariance priors and indexing.
3. Integrate likelihoods, aliases, missing data and joint prediction.
4. Complete deterministic edge checks and cross-engine comparison harnesses.
5. Run resource-controlled regression, reference fits, SBC and benchmarks.
6. Render documentation; update `PR.md` with actual results and remaining limits.
