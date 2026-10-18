# Add native structured Gaussian effects and exact partial marginal inference

Repeated trajectories and correlated coefficient vectors require custom covariance code. This change adds `ar1`, `ou`, `diag`, `cs`, and `us` formula wrappers, with equivalent bar and `by=` forms. The paired Formulae interface retains coordinate or ordinary coefficient-design semantics, composite groups, and standard nesting expansion.

Gaussian identity-link mean models can integrate one eligible structured term while sampling the remaining fixed, ordinary random, HSGP, and structured effects. Automatic selection uses scalar latent dimension, then component-qualified name. Temporal effects use grouped transitions and exact filtering; coefficient effects use coefficient-sized Gaussian algebra. Joint conditional recovery supplies public effects for prediction. Other supported families and auxiliary predictors use latent inference. PyMC, nutpie, NumPyro, and BlackJAX remain required for both paths.

Prediction reuses fitted states/designs and supports joint interpolation/extrapolation, declared coefficient domains, numeric slope values, and fresh population draws for new groups. Observation log likelihood stays conditional on sampled or recovered effects. Metadata records the inference plan. Prior prediction uses the full latent generative graph after either fit method.

## Correctness

The implementation includes corrected `auto_scale=False` coefficient SD defaults, one-process ungrouped keys, scalar correlation prior validation, and alias-aware prior updates. US uses independent Beta canonical partial correlations to implement LKJ: independent density/Jacobian and prior-moment tests cover dimensions above two, avoiding the affected matrix-valued LKJ transform/generator in the tested PyMC version.

## Validation

See [the acceptance record](structured-covariance-validation.md) for commands, seeds, versions, diagnostics, resources, failed runs, and precise pending criteria.

- 189 focused Bambi tests and 162 focused Formulae regressions pass; Black, serial pylint, actionlint, and diff whitespace checks pass.
- Eight small Gaussian latent/marginal posterior pairs pass four-chain convergence and Monte Carlo agreement checks for covariance parameters, recovered states, full designated joint moments, and predictions. Six fixed-parameter glmmTMB references pass.
- 72 short execution cases cover both inference paths and all four samplers, including both CS variants, negative AR(1), HSGP composition, and 3D US. They contain divergences and establish execution only.
- Full Bambi regression CI passes Python 3.12/3.13 with both linkers (543 passed, one skipped per job before the latest 13 diagnostic/summary tests); Formulae passes Python 3.11–3.14 (288 passed, one skipped each). All eleven meaningful posterior pairs, including AR(1) on all four samplers and the repaired composition rerun, pass convergence and MCSE agreement. Documentation and six glmmTMB reference checks pass.
- All seven prior-scale sensitivity refits converge; halving the scale produces moderate amplitude-prior dependence for six structures, so blanket robustness is not claimed. AR(1), OU and diagonal structures pass the bounded 100-replication SBC/heldout coverage checks after independent artifact review and stricter reassessment of every saved fit. Four calibration cases remain pending.
- All initial benchmark families fail convergence. Their failed records are preserved and provide no accepted efficiency evidence. A controlled adaptation-only rerun uses the same data, priors, seeds, retained draws and four serial chains, with target acceptance 0.99 and 2,500 tuning steps. Benchmark diagnostics now include all structured states and require four converged chains. Logs and raw results are retained on failure and on reruns.

This remains a draft: four calibration cases, representative converged performance/benefit, and the latest CI reruns remain pending. The initial extended run continues. The marker below starts a benchmark-only rerun before manual dispatch is registered on the default branch; it does not restart calibration or sensitivity.

<!-- structured-benchmark-acceptance -->

## Dependency coordination

The separate Formulae change is on `codex/structured-expressions`, based on upstream `90b41e3`. Formulae [draft PR #1](https://github.com/tjburch/formulae/pull/1) publishes commit `2fbeb11d89a67db2700557872896dea0b5c5d9ab`, pinned exactly in `.ci/formulae.json`. Both the broad Bambi regression matrix and structured jobs use that same commit and assert the imported module path/interface. The action rejects missing pins and moving refs.

The release manifest retains the original Formulae release range. A release containing the paired interface must be assigned before setting the released dependency minimum/range and removing the temporary development override. This is a merge/release blocker; no nonexistent release version or local path is inserted into release metadata.
