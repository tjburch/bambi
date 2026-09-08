# Simulation-based calibration

These commands are intended for the manually triggered reference runner, not a
compute-limited laptop. They have no sampling side effects until `fit` or `run`.
Four chains run sequentially, with one numerical thread. No automatic fit retries.

## Independent fixtures and single-block comparisons

```sh
python docs/examples/group_covariance_validation/single_block_reference.py generate ou binomial results/ou-fixture --seed 20260908
python docs/examples/group_covariance_validation/single_block_reference.py fit results/ou-fixture results/ou-bambi --engine bambi --seed 20260909
python docs/examples/group_covariance_validation/single_block_reference.py fit results/ou-fixture results/ou-stan --engine stan --seed 20260910
python docs/examples/group_covariance_validation/compare_summaries.py results/ou-bambi/summary.json results/ou-stan/summary.json
```

Structures: `ar1`, `ou`, `cs`, `toep`, `us`. The fixture generator also supports
`four-block-ar1` for Bambi SBC; the existing four-block Stan runner serves the
separate learned four-block posterior comparison. The single-block Stan runner
explicitly rejects this mode.

Each fixture draws fixed coefficients, marginal scales, covariance parameters,
latent effects and responses from the **same proper priors fitted by both engines**.
Generation does not import Bambi covariance helpers. Toeplitz uses independent
Yule–Walker solves. Unstructured correlation uses the LKJ C-vine beta construction.
AR1 uses actual integer time gaps; OU uses irregular continuous coordinates.
Bernoulli uses the Bambi Bernoulli family; Stan's binomial with one trial is the
same likelihood. Unequal binomial trial counts range from two to eight.

Fixtures have replicated cells, a missing intermediate visit, and an absent
three-way group while retaining its lower-order groups. Selected SBC latent
quantities include an unobserved fitted visit; no generating values are estimated
from observations. Prior draws can produce weakly identified data. Such fits must
not be dropped or replaced by more convenient simulations.

Both engines export fixed effects, marginal scales, covariance parameters, all
group-by-level coefficients, probabilities, conditional pointwise likelihood and
exact conditional predictive moments. Draws retain separate chain and iteration
dimensions. Mean and quantile errors use autocorrelation-aware estimates.

## Frozen campaigns

```sh
python docs/examples/group_covariance_validation/sbc.py init results/pilot --phase pilot --commit COMMIT_SHA
python docs/examples/group_covariance_validation/sbc.py run results/pilot/manifest.json --case ar1-bernoulli-000
python docs/examples/group_covariance_validation/sbc.py check results/pilot/manifest.json
```

Replace `COMMIT_SHA` with the exact tested commit. Each campaign also records a
source-byte hash, including its reference scripts. The pilot contains ten cases
per structure, five per response family: 60 total, including four-block AR1.
After reviewing the pilot, create a separate `--phase full` campaign: 100 cases
per structure, 50 per family, for 600 total. Pilot and full seeds are disjoint.

Defaults are 1,000 warmup and 1,000 retained draws per chain. `init --warmup` and
`--draws` permit a documented increase in a **separate campaign**. Settings, cases
and seeds are part of the campaign hash. Do not edit a manifest after starting.
Changing code requires another campaign; previous output remains evidence.

`run` runs one case and writes `status.json` before starting. Complete results
can be resumed only with the same case and campaign identity and unchanged rank
artifact. Failed/interrupted cases remain failed; output is never overwritten.
Sampling output is saved before diagnostics and rank processing. Retain the case
log and completed draws even if later diagnostics fail. `check` exits nonzero for
missing, failed, stale, corrupted or non-calibrated results. It prints a JSON
report, including the full expected denominator and each incomplete case.

## Statistical checks and limits

Convergence gates: four chains; rank-normalized R-hat at most 1.01; bulk and tail
ESS at least 400; zero divergences; E-BFMI at least 0.3; no tree-depth saturation.
Missing or nonfinite diagnostics fail closed.

Ranks use exactly 100 retained posterior draws. The spacing is twice the ratio
of raw draws to the worst bulk/tail ESS among the selected calibration quantities,
rounded upward. A random starting index avoids always using the beginning of a
chain. The retained sample must also have bulk and tail ESS of at least 80.
Insufficient effective draws or residual dependence fails the case. This finite
ESS check is conservative and noisy; review pilot failures before fixing the full
campaign settings. It cannot prove independence of finite MCMC draws.

Exact ties are randomized, including numerical ties in near-saturated
probabilities. Calibration checks include fixed effects, each covariance
hyperparameter, selected latent effects and three fitted probabilities. The
checker reports a discrete-uniform rank CDF with a conservative DKW envelope,
adjusted across all reported structure/family/quantity tests. It also reports
94% equal-tail coverage with a 95% Wilson interval. This coverage interval is not
an HDI or a separate unadjusted significance test.

Each family has only 50 full-campaign replicates. Simultaneous envelopes are
therefore broad; passing is limited evidence, not proof. The unit tests verify
that deliberately biased rank sequences fail. Review rank plots and failure
patterns alongside the machine report; model criticism and cross-engine
agreement remain separate gates.

No SBC sampling or Stan compilation has been certified locally. Successful CLI
and unit checks must not be reported as a successful calibration campaign.
