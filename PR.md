# Add structured covariance for group-specific effects

Draft: **not ready to merge**. Core implementation and targeted checks exist;
cross-engine posterior validation and the complete readiness matrix are pending.

## Summary

Add additive formula wrappers for joint Gaussian group-specific coefficients:
`ar1`, `ou`, `cs`, `toep`, and `us`. Effects enter the linear predictor before the
inverse link. They are not residual-correlation structures, and do not change the
observation likelihood or ordinary group-effect defaults.

```python
model = bmb.Model(
    "outcome ~ x1 + x2"
    " + ar1(0 + year | subject)"
    " + ar1(0 + year | subject:condition)"
    " + ar1(0 + year | subject:context)"
    " + ar1(0 + year | subject:condition:context)",
    data,
    family="bernoulli",
)
```

Each block has independent latent coefficients and its own covariance parameters.
Grouping interactions use tuple-safe identifiers. Repeated group/time cells share
coefficients. Dense and sparse predictors use the same coefficient prior.

## Scale restriction

**AR1, OU, exchangeable and Toeplitz blocks have one marginal latent SD per term.**
Different terms have separate SDs; each term shares its SD across groups and times.
The AR1 SD is stationary marginal scale, not innovation scale.

**Unstructured blocks have one SD per coefficient. Heterogeneous temporal scales
and scale regression are deferred.** A later scale extension must define how scales
are assigned at unseen times before adding prediction support. This PR does not
choose an implicit scale-extrapolation rule.

## Behavior

- AR1 permits negative correlation and respects actual integer time gaps.
- OU supports continuous time; its positive decay prior depends on time units.
- Exchangeable correlation has a level-count-dependent valid range.
- Toeplitz uses partial autocorrelations on a declared finite horizon.
- Unstructured covariance supports visit coefficients and correlated intercept/slopes.
- New temporal coordinates are drawn jointly conditional on fitted coefficients.
  New groups receive joint population trajectories. Novelty is checked per block.
- CS/unstructured prediction is limited to declared coefficient levels; Toeplitz
  prediction must remain within its horizon.
- `include_group_specific=False` removes all group-specific contributions.
- Pointwise likelihood remains conditional on latent coefficients. Appropriate
  future-time/subject-held-out validation is separate from ordinary row-wise LOO.

Scalar-family construction checks cover Gaussian, Bernoulli, binomial, Poisson and
negative binomial. Vector-response coefficient covariance and covariance shared
across distributional parameters are not included.

## Validation status

Targeted serial checks cover independent numerical covariance/conditioning,
four-block construction, signed time gaps, joint prediction, repeated cells,
three-way group novelty, prior updates, aliases, and selected existing prediction
behavior. These checks include prior draws for graph testing, not converged
posterior fits or simulation calibration.

Executed: 261 targeted tests, including selected ordinary-model regressions, passed;
two hosted-only sampler checks were skipped locally. No posterior fits ran locally.
Production Pylint scored 10/10; changed Python files passed Black checks. The
nonexecuting tutorial rendered. A tiny graph-build benchmark and two-draw AR1
prior-only reference smoke run passed. See the [validation report](docs/examples/group_covariance_validation/report.md).

Reference scripts distinguish native brms known covariance and learned unstructured
models from independent Stan learned AR1 blocks. They do not treat brms residual
`ar()` as equivalent to several latent group-specific AR1 terms.

Still required: completed reference comparisons (including learned OU/CS/Toeplitz),
full density/gradient/prediction coverage, predictive criticism and sensitivity,
SBC, sampler/backend/Python matrix, scaling benchmarks, and full documentation build.
The comparison harness includes latent-effect, quantile, predictive-moment and
pointwise-likelihood exports. Independent joint prediction checks and a batched
SBC runner are provided. Hosted posterior execution and campaign completion remain
required; runnable scripts alone are not validation evidence.

Manual workflows separate reference comparisons, SBC, sampler smoke checks,
benchmarks and documentation builds from ordinary PR regression tests. They use
standard runners, bounded concurrency and short artifact retention. No hosted jobs
have been run for this change yet.

Unstructured prediction follows formulae's categorical rules. For subset prediction
with explicitly ordered or unobserved visits, use an ordered pandas categorical
column in `us(0 + visit | subject)`. Formulae's explicit `C(visit, levels=...)`
transform rejects prediction subsets that omit declared levels; Bambi reports this
limitation with a clear error.
See `docs/development/group-covariance-plan.md` for acceptance criteria.
