# Manual validation on a public fork

Status: workflows and runners implemented; hosted execution not yet verified.
Do not interpret a workflow file, prior-only run or sampling smoke check as a
completed posterior-comparison or calibration gate.

## Resource and publication rules

Keep the laptop on small serial checks. Run reference compilation and posterior
sampling only on standard hosted runners. The manual workflow rejects private
repositories, permits two concurrent jobs, and serializes workflow dispatches.
Each job has a 120-minute limit and one numerical thread. Fits use four sequential
chains with 1,000 warmup and 1,000 retained draws initially.

Creating a fork, committing, pushing, or dispatching jobs needs explicit approval.
The existing origin points to upstream; do not push this branch there.
No paid runner or self-hosted laptop runner is configured.
GitHub must also see the manual workflows on the fork's default branch. For a new
validation fork, make the covariance branch its default after pushing it. Do not
change an existing fork's default branch without approval; instead agree on a
workflow-only bootstrap commit to that fork's default branch.

## Workflow stages

`Manual covariance validation` takes `suite`, zero-based `start`, and `count`
(1–10, default 2). The case order is deterministic and printable without loading
numerical libraries:

```sh
uv run --no-sync python docs/examples/group_covariance_validation/workflow_cases.py references --start 0 --count 8
```

| Suite | Cases | Meaning |
|---|---:|---|
| `compile-smoke` | 8 | Prior-only Bambi and R/Stan pipeline smoke; no posterior certification |
| `references` | 8 | Four existing reference modes × Bernoulli/binomial; full summary comparison and Bambi PPC/sensitivity |
| `single-block` | 10 | Five learned structures × Bernoulli/binomial against independent Stan |
| `heldout` | 6 | Four-block AR1 refits: subject, future-time and three-way holdouts × both families |
| `sbc-pilot` | 60 | Ten prior-drawn replicates per structure, including four-block AR1 |
| `sbc-full` | 600 | 100 replicates per structure, including four-block AR1 |
| `samplers` | 4 | PyMC, nutpie, NumPyro, BlackJAX short structured-model integration |
| `benchmarks` | 1 | Serial dense/sparse build, compile and future-prediction benchmarks |

Pilot and full SBC use disjoint deterministic seeds. Each case is a separate job
and its completed outputs survive failures in other jobs. Dispatch the next batch
only after inspecting the previous one. Do not dispatch 600 cases before pilots
confirm runtime, memory use, rank thinning and sampler diagnostics.

Use `Run CI` manually for the existing Python/backend regression matrix.
Use `Manual covariance documentation and package checks` for style, lint, wheel
building and a full documentation render with model execution disabled.

## Dependencies and evidence

The reference requirements pin primary numerical packages. R pins brms, posterior,
jsonlite and CmdStanR; CmdStan is 2.37.0. Each job records the full resolved Python
environment and installed R package versions. Transitive dependency resolution is
not yet a committed cross-platform lock: retain the successful pilot environment
records, and reject mixed SBC environments. Any toolchain change starts a separate
campaign; never merge its results with the previous one.

Input manifests identify the commit, actual source bytes (including dirty edits),
CSV bytes, design and priors. Each exporter verifies the inputs and coefficient
mapping. Reference summaries require all expected metrics; omitted parameters are
not silently removed from comparisons. Four-combined-MCSE discrepancies must be
investigated, including the effect of comparing many correlated summaries.
Fixtures place `x1` on an exactly representable binary grid, and Python readers use
round-trip float parsing. This prevents CSV/R/JSON rounding from creating different
design matrices. The exact-design checks are not replaced with loose tolerances.

Artifacts expire after seven days. Download each batch to the external drive before
expiry. Preserve raw fits and logs outside Git; include compact versioned summaries
and the final report in the PR. Do not upload personal datasets or residual-work
archives. Compression reduces storage; it does not remove storage limits.

Merge downloaded SBC artifacts into a separate directory, then check completeness:

```sh
uv run --no-sync python docs/examples/group_covariance_validation/merge_sbc.py /path/on/external-drive/merged /path/on/external-drive/downloads
uv run --no-sync python docs/examples/group_covariance_validation/sbc.py check /path/on/external-drive/merged/manifest.json
```

The merge rejects differing manifests and duplicate case IDs before copying.
Choose explicitly between duplicate attempts after investigating them; keep both
original artifacts. The aggregate checker rejects missing, failed, interrupted,
changed or stale cases. Failed sampling is not a reason to remove a replicate from
the calibration denominator. See [SBC details](SBC.md).

## Required follow-through

Hosted smoke runs must confirm the R export names, quantile MCSE APIs, compilation,
sampler diagnostics and environment installation. Fix failures before posterior
comparisons. Re-run affected cases after code changes, with matching identities.

The independent joint-prediction check validates conditional means and covariances
at shared latent/covariance values. It does not replace held-out posterior refits or
predictive calibration. The PPC/sensitivity report handles the four original
reference modes; other modes, report plots and unreliable-importance-weight refits
still need review or extension. Benchmarks separate graph/compile and future-prediction
cost. Prediction timings use four prior-derived draws, not a fitted-posterior
performance study. These limitations remain explicit in the validation report.
