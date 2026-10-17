# Structured covariance terms implementation requirements

Implement five native Bambi covariance wrappers with two equivalent formula syntaxes, broad additive composition, exact partial Gaussian marginalization, and prediction. Deliver a coordinated Formulae PR and Bambi PR.

## Scope and delivery

The required wrappers are `ar1`, `ou`, `cs`, `diag`, and `us`. They add zero-mean Gaussian latent effects to a distributional component's linear predictor, before its inverse link.

Required capabilities:

- Both bar syntax and `by=` syntax, with equivalent designs, priors, canonical term identities, and posterior names.
- Ungrouped processes, simple groups, composite groups, and standard nested expansion.
- Multiple additive structured terms alongside fixed effects, offsets, ordinary group-specific effects, and HSGPs.
- Structured terms in the parent and auxiliary parameter formulas.
- Latent inference across Bambi's supported families and distributional components, preserving their existing response dimensions and identifying constraints.
- Exact Gaussian marginalization of at most one selected structured term, conditional on all remaining sampled effects.
- PyMC, nutpie, NumPyro, and BlackJAX support for both inference paths. These are release requirements.
- Posterior state recovery, prior and posterior prediction, priors, aliases, coordinates, model summaries, validation, and documentation.

The Bambi feature PR depends on one focused Formulae PR. Prepare separate implementation changes and review summaries for the two repositories. Bambi's released dependency range must include the released Formulae support it uses. Temporary local dependency overrides are acceptable for development and CI against the paired change, but must be identified and removed from release configuration.

The Bambi specification branch starts from upstream main at `ef4097dc61589c74c6e6c3c48a989a9be6c1553b`. The older `structured-group-covariance` branch was abandoned because its scope grew too large. It is optional reference material for specific questions, not an implementation to resume or reproduce. Do not merge or cherry-pick that branch wholesale or inherit its unrelated refactors and validation infrastructure.

## Formulae dependency

Formulae owns parsing and formula interpretation. Implement the supported syntax there rather than bypassing its decisions through Bambi-specific parsing or source rewriting.

A probe of Formulae 0.6.2 and 0.7.0 found:

| Input | Result from `model_description` |
| --- | --- |
| `ar1(season, by=player)` | Recognized as an ordinary function call; no structured semantics yet |
| `ar1(season, by=["player", "pitcher_hand"])` | Parse error at `[` |
| `ar1(0 + season \| player)` | Function-call resolution rejects the inner pipe expression |
| `ar1(0 + season \| team/player)` | Same pipe-resolution limitation |
| `(0 + season \| team/player)` | Ordinary nested group expansion succeeds |

The tandem Formulae change must provide a supported way to:

- Represent a formula expression inside a registered structured wrapper, including its intercept, interactions, transformations, grouping, and options.
- Preserve the inner formula semantics instead of evaluating `1 + x` as ordinary elementwise arithmetic for coefficient wrappers.
- Resolve both wrapper forms into metadata Bambi can consume.
- Support grouping lists containing column names, including the agreed quoted list syntax.
- Expose all referenced data columns, including quoted grouping names, to missing-data handling and new-data evaluation.
- Preserve ordinary calls, arithmetic, namespace lookup, and existing group-specific formula semantics.
- Apply standard `:` interaction and `/` expansion rules inside the supported wrapper context.

Use a focused, explicit extension point consistent with Formulae's architecture. It should activate structured formula semantics only for functions that register for them. Do not hardcode Bambi's covariance algorithms into Formulae, globally reinterpret ordinary function arguments, or maintain a second parser in Bambi.

Settle the internal Formulae interface with a small prototype before building the Bambi backend. Changes to public syntax or the agreed semantics require a design decision; implementation difficulty alone does not make a requirement optional.

## Public formulas

The formula fragments below belong inside Bambi formula strings.

### Temporal coordinate wrappers

These pairs describe the same process:

~~~python
ar1(season, by=player)
ar1(0 + season | player)

ou(time, by=player)
ou(0 + time | player)

ou(timestamp, by=player, time_unit="hours")
ou(0 + timestamp | player, time_unit="hours")

ar1(season, by=["team", "player"])
ar1(0 + season | team:player)
~~~

For `ar1` and `ou`, the input is a coordinate, including when it is numeric. It is not a numeric random slope. The bar form requires an explicit zero intercept and exactly one coordinate expression. The `by=` form has coordinate semantics without adding an intercept.

Require scalar grouping via `by=player` and `by="player"`. Require composite grouping via `by=["team", "player"]` and `by=[team, player]`. In this registered context, quoted strings identify data columns. Preserve quoted column names and report missing columns clearly.

Omitting `by` creates one ungrouped process. A precomputed composite grouping column is also valid. Do not interpret a list as several additive terms.

### Coefficient covariance wrappers

`us`, `diag`, and `cs` use ordinary Formulae coefficient-design semantics. Numeric predictors create slopes. Categorical terms use ordinary contrasts unless an explicit zero intercept requests a full set of indicators.

~~~python
us(1 + x | player)
us(1 + x, by=player)

us(x | player)
us(x, by=player)  # Both include the ordinary implicit intercept.

us(0 + C(visit) | patient)
us(0 + C(visit), by=patient)

diag(1 + x | player)
diag(1 + x, by=player)

cs(0 + C(visit) | patient)
cs(0 + C(visit), by=patient)
~~~

For covariance across numeric visit labels, users must make visits categorical. A prediction at a new numeric slope value uses the existing coefficient basis and is allowed; it does not add a covariance dimension.

Support ordinary valid predictor designs, including multiple numeric slopes, categorical predictors, and their supported interactions and transformations. Reuse Formulae's fitted design when evaluating prediction data.

Default CS is heterogeneous, with one SD per design coefficient. Also require:

~~~python
cs(1 + x | player, homogeneous=True)
cs(1 + x, by=player, homogeneous=True)
~~~

`homogeneous=True` means one shared SD in the coefficient basis. It does not change the correlation rule. A homogeneous option for other wrappers is outside the required API.

Omitting `by` in a coefficient wrapper creates one Gaussian coefficient vector with the specified covariance prior. This may be weakly identified; it remains a valid specification.

### Nested expansion and identity

~~~python
ar1(0 + season | team/player)
~~~

must expand into:

~~~python
ar1(season, by=team) + ar1(season, by=["team", "player"])
~~~

Apply standard recursive Formulae nesting semantics for deeper expressions. Each expanded term has its own covariance parameters. A composite group `team:player` is only the team-player process.

A player trajectory grouped by `player` continues across team changes. A trajectory grouped by `team:player` is independent across distinct team-player keys, conditional on covariance parameters. Returning to a previous team uses that existing key; users need a stint identifier to model a fresh process for each stint.

Reject a term that supplies grouping through both an inner bar and `by=`. Follow Formulae's duplicate-term semantics so equivalent syntax cannot accidentally add the same contribution twice. Distinct specifications that happen to produce identical partitions in a dataset remain distinct effects; flag exact confounding where it can be established.

Normalize syntax before prior lookup and backend construction. Equivalent terms must expose the same canonical identity. User-supplied spelling and unambiguous aliases must resolve to that identity. Ambiguous selectors must raise an error listing the component-qualified choices.

## Covariance definitions

Covariance parameters are shared across groups within a term. Different terms, including terms in different distributional components, have separate parameters. For a family with free response-coefficient axes, use independent structured vectors and separate covariance parameters per free response axis, following Bambi's ordinary group-specific dimension and reference-category rules. Do not introduce covariance between response axes. Group-specific covariance parameters and hierarchical pooling of those parameters are outside this PR.

| Wrapper | Covariance |
| --- | --- |
| `ar1` | `K[i,j] = sd**2 * rho**abs(t[i] - t[j])` |
| `ou` | `K[i,j] = sd**2 * exp(-decay * abs(t[i] - t[j]))` |
| `diag` | `K = diag(sd**2)` |
| `cs` | `K = D @ R @ D`, with all diagonal correlations 1 and all off-diagonal correlations `rho` |
| `us` | `K = D @ R @ D`, with an unrestricted positive-definite correlation matrix |

For `ar1`, require `-1 < rho < 1`. Its SD is the stationary marginal SD, not the innovation SD. For `ou`, require positive decay and use the stationary marginal SD.

For CS dimension `K > 1`, require `-1/(K-1) < rho < 1`. Use the complete fitted coefficient dimension, even when individual groups observe only subsets. Support the full negative correlation range in both CS scale variants. A random-intercept shortcut alone does not cover that range.

Use one scale per coefficient for `diag`, `us`, and heterogeneous `cs`; use a scalar scale for homogeneous CS. Handle one-dimensional designs explicitly: CS and US correlations reduce to an identity matrix, with no division by zero or invalid LKJ construction. AR(1)/OU parameters with no observed transitions may remain informed only by their priors.

## Coordinates and data

### AR1 ordering

- Finite integer numeric coordinates use actual integer differences. Accept integer-valued floats without silently rounding other values.
- Seasons 2024 and 2026 are two steps apart, even if 2025 is absent.
- Ordered pandas categoricals use declared category positions, including declared but unobserved categories.
- An explicit categorical level order supplied through supported Formulae categorical metadata is also valid.
- Unordered non-numeric coordinates must raise an actionable error. Never infer order from rows or lexical sorting.
- A prediction using categorical coordinates must use the fitted declared grid. An unknown label does not define its own position.
- New integer numeric coordinates are allowed for prediction without refitting.

Do not allocate every missing integer step. Use transition powers across observed gaps.

### OU time

Accept finite real numeric coordinates and datetime columns.

- Numeric coordinates retain their supplied units.
- Datetimes use elapsed days by default.
- Accept `time_unit="seconds"`, `"minutes"`, `"hours"`, `"days"`, and `"weeks"` for datetime coordinates. These are fixed elapsed durations.
- For numeric coordinates, an explicit `time_unit` labels the supplied units without converting values. Without a label, report native numeric coordinate units.
- For datetime conversion, reject ambiguous calendar units such as months and years; users can supply an explicit numeric coordinate instead.
- Recenter datetime arithmetic before float conversion. Handle timezone-aware values as elapsed UTC time and reject incompatible naive/aware prediction inputs clearly.
- The decay parameter is in inverse coordinate units. Expose the chosen unit in term metadata and summaries.
- Do not silently standardize coordinates. Default prior scaling must be documented separately from coordinate conversion.

### Repeated states and missing values

Repeated observations at the same group-coordinate pair share one latent state. They still have separate observation means, variances, and likelihood contributions. Unsorted rows, unequal group lengths, and different time grids across groups are supported.

Encode composite keys as tuples or another collision-safe representation. Preserve identifiers that contain separators and distinguish values with different types. Do not allocate unobserved Cartesian combinations of grouping levels.

Preserve Bambi's `dropna` policy across every referenced column in the main and auxiliary formulas. Apply one consistent row mask and retain alignment among responses, designs, groups, and coordinates. Errors must identify missing or non-finite coordinates and grouping values. If no observations remain, raise a clear error.

Categorical coefficient domains are fixed when constructing the model. Preserve explicitly declared unobserved categories. Covariance parameters associated only with unobserved levels remain governed by their priors; documentation must explain this.

## Inference API

Add these keyword arguments to `Model.fit`:

~~~python
model.fit(structured_method="auto")
model.fit(structured_method="latent")
model.fit(structured_method="marginal")

model.fit(
    structured_method="marginal",
    marginalize="ar1(season, by=player)",
)
~~~

The default is `structured_method="auto"`. The argument is independent of `inference_method`, which continues to select the sampler.

| Request | Required behavior |
| --- | --- |
| `auto` without a selector | Select one eligible term; use all-latent inference if no term is eligible |
| `latent` without a selector | Sample every structured term |
| `marginal` without a selector | Select one eligible term; error if none is eligible |
| `auto` or `marginal` with a selector | Honor that exact selection or raise an error |
| `latent` with a selector | Raise an incompatible-options error |

`marginalize` accepts one canonical term name, a supported equivalent source spelling, or an unambiguous term alias. Use a component-qualified name when required for disambiguation. Multiple selected terms are outside this PR and must produce a clear error.

A model with no structured terms behaves as before under `auto` and `latent`. An explicit selector or `marginal` request with no eligible term raises an error.

Use a deterministic automatic selection rule: among eligible terms, select the one with the largest number of scalar latent coefficients that would otherwise be sampled. For temporal terms count observed unique group-state pairs; for coefficient designs count the fitted coefficient dimension per observed group. Break ties by component-qualified canonical name. This is a dimension-reduction heuristic, not a guarantee of minimum runtime. Publish the rule and benchmark its behavior.

Report the requested method, effective method, selected term, remaining latent terms, and any automatic fallback reason in the fit log and result metadata. Do not emit a warning for an ordinary supported automatic fallback.

Resolve the plan before compilation. Calling `build()` before `fit()` must not lock the model into a different plan. Switching methods or selectors on repeated fits must invalidate the relevant backend state. Prior updates and aliases must retain their existing invalidation behavior. Prior predictive draws must remain available before and after either kind of fit.

## Latent inference

Use one shared term representation and covariance implementation behind both public syntaxes. The representation must retain the structure, component, design or coordinates, grouping, observation indexing, prior configuration, and canonical identity.

Default to a noncentered Gaussian representation. Respect Bambi's existing global and per-component parameterization controls, including supported prior overrides. A centered alternative may use the same transition or covariance factorization.

AR(1) and OU must use Markov transitions rather than a dense covariance over all temporal states. Initialize each group from its stationary distribution. Across gap `d`, use transition `a = rho**d` for AR(1) or `a = exp(-decay*d)` for OU, with innovation variance `sd**2 * (1-a**2)`. Use stable calculations near zero gaps and correlation boundaries.

Coefficient covariance wrappers use the fitted design to map Gaussian coefficient vectors into the linear predictor. Factor shared covariance matrices once per parameter evaluation where possible. Do not create a dense covariance matrix over all observations or groups.

Allow the agreed additive compositions, including:

~~~python
"y ~ x + (1 | team) + ar1(season, by=player)"

formula = (
    "y ~ x + ar1(season, by=player)"
    " + ar1(season, by=['player', 'pitcher_hand'])"
)

bmb.Formula(
    "y ~ x + ar1(season, by=player)",
    "sigma ~ z + ar1(season, by=team)",
)
~~~

In the last example, the auxiliary process enters the scale predictor through the existing scale link. Latent inference must preserve the existing family-specific behavior rather than assuming a scalar Gaussian response.

## Exact partial Gaussian marginalization

### Eligibility

Require exact marginalization for all five wrappers when the selected term:

- Contributes linearly to the mean of a univariate Gaussian family with identity mean link.
- Has a Gaussian conditional prior with the supported covariance.
- Does not also enter observation variance, another parameter, or an unsupported potential through its latent values.

Other fixed, group-specific, HSGP, and structured effects may remain sampled. Crossed or nested group membership in those remaining effects does not disqualify the selected term.

Support constant, known, and modeled heteroscedastic observation SDs, including auxiliary formulas whose latent effects remain sampled. Require finite positive conditional observation variances.

A structured effect in an auxiliary scale predictor uses latent inference. A non-Gaussian response, non-identity Gaussian mean link, or modified likelihood such as censoring, truncation, constraints, or power weights is outside required exact marginalization. The complete model remains supported through the existing latent path. `auto` may fall back; an explicit unsupported request must error with its reason.

Potentials or custom dependencies involving the selected latent vector need an implemented exact treatment to be eligible. Do not silently drop those dependencies.

### Likelihood

For selected effect `u` and all remaining sampled parameters and effects `theta`, target:

`p(y | theta) = integral p(y | u, theta) p(u | theta) du`.

Integrate out exactly one term. Evaluate the remaining effects at their current sampled values as part of each observation's conditional mean and variance. This permits a player process to be marginalized while team effects or another player-hand process are sampled.

Never integrate several overlapping effects separately against the same observations and add those likelihoods. Each observation must enter the model once.

Use groupwise scalar Kalman filtering or an equivalent linear-time exact method for AR(1)/OU. Process repeated-time measurements as repeated observations of the same state, or use exact sufficient statistics with all normalizing constants retained. Replicate covariates and observation SDs may differ.

For `diag`, `cs`, and `us`, use exact algebra on each selected group and its coefficient/state dimension. A small dense coefficient covariance is acceptable. Use Gaussian identities or sufficient statistics to avoid an observation-sized dense covariance, especially with many replicates.

Keep AR(1)/OU likelihood evaluation linear in observations and observed states after indexing. Avoid global dense fallback and arbitrary crossed-effect joint marginalization.

### Recovery

After fitting, draw the selected latent vector from its exact conditional posterior for each sampled parameter/effect draw. Preserve joint covariance across times or coefficients. Independent draws from per-state marginal distributions are insufficient.

For temporal terms, use a simulation smoother or equivalent exact joint recovery. For coefficient terms, use the exact conditional Gaussian. Include uncertainty in the recovered states; conditional means alone are insufficient.

Expose recovered final effects under the same public names and coordinates as latent fits. Internal noncentered auxiliaries may differ and must be identified as such. `omit_offsets` must not remove the final effects needed for prediction.

All four required samplers must use the same statistical model, with working differentiation and state recovery. Do not change the requested method or substitute another sampler to hide a backend limitation.

## Priors

Use parameter dictionaries under each structured term, following the existing HSGP-style prior interface and Bambi's component nesting and precedence rules:

~~~python
term = "ar1(season, by=player)"
priors = {
    term: {
        "sd": bmb.Prior("HalfNormal", sigma=0.7),
        "rho": bmb.Prior("Normal", mu=0, sigma=0.5),
    },
}
~~~

Required keys:

| Wrapper | Keys |
| --- | --- |
| `ar1` | `sd`, `rho` |
| `ou` | `sd`, `decay` |
| `diag` | `sd` |
| `cs` | `sd`, `rho` when dimension exceeds one |
| `us` | `sd`, and either `eta` for LKJ or a fixed `correlation` matrix |

Support proper `bmb.Prior` specifications and valid numeric constants. Support scalar SD priors broadcast independently over heterogeneous coefficient scales, and correctly shaped per-coefficient specifications. A scalar fixed SD may set heterogeneous scales to equal constants without changing the wrapper's definition.

Use these implementation defaults:

- SD: HalfNormal with scale `2.5 * s`. For Gaussian parent coordinate effects with `auto_scale=True`, `s` is the response SD, falling back to 1 for constant responses. For other component/link scales or `auto_scale=False`, use `s=1`. For coefficient designs, reuse existing Bambi group-specific scaling rules when they apply, with finite positive scaling for constant design columns. Homogeneous CS must retain one shared SD in its raw coefficient basis.
- AR(1) correlation: Normal(0, 0.5), normalized and truncated to `(-1, 1)`.
- CS correlation: Normal(0, 0.5), normalized and truncated to its admissible interval.
- US correlation: LKJ with `eta=2`; `eta` is a positive fixed tuning parameter.
- OU decay: Exponential with rate equal to a reference time gap when `auto_scale=True`. Use the median positive within-group adjacent gap, falling back to the global positive adjacent gap and then one native time unit if no gap exists. With `auto_scale=False`, use rate 1 in the documented units.

These numerical defaults are implementation choices supplied to complete the specification. Validate their prior predictive behavior and document their exact definitions. If evidence requires a material modeling change, surface it before changing the specification.

Correlation priors refer to the actual correlation, not an undocumented unconstrained coefficient. Apply normalized truncation to the admissible interval for user correlation priors too; document this behavior. Reject fixed correlations at invalid boundaries, invalid matrices, non-positive scales/decay, and incompatible prior shapes. A fixed US matrix must be symmetric, positive definite, and have unit diagonal. Reject simultaneous `eta` and `correlation` settings.

User priors take precedence and must not be rescaled silently. Unknown parameter keys and incompatible shared `common`/`group_specific` scalar prior specifications must follow clear validation rather than being applied to an unrelated covariance parameter.

Document weak identification of independent latent variance versus residual variance, unrestricted covariance versus residual noise, and overlapping terms with identical grouping. Proper priors do not make unidentified components data-identified. Do not impose an arbitrary sum-to-zero trajectory constraint or remove residual noise automatically.

## Prediction and results

### Required prediction cases

| Request | Support |
| --- | --- |
| Observed group and observed state/design | Reuse posterior effects |
| Another observation at an existing state | Reuse that state; generate fresh observation noise when requested |
| Existing group, state in the fitted domain but not observed in that group | Draw conditionally from the fitted process |
| New group, fitted coordinate domain or coefficient basis | Draw a fresh joint Gaussian process/vector using each posterior covariance draw |
| New integer AR(1) time or new OU time | Joint interpolation/extrapolation |
| New categorical coordinate/coefficient level for `cs`, `diag`, or `us` | Error unless the level was declared when constructing the model |
| New numeric predictor values in a fitted coefficient design | Evaluate the existing basis; allowed |

For new groups, draw from the fitted population distribution rather than copying an existing group's trajectory. Different new groups are conditionally independent. In additive models, the same new player draw must be shared across every relevant row and player-hand combination; the composite effect has its own independent draws.

Honor Bambi's current `sample_new_groups` control and its existing deprecation behavior. New-group prediction is required when enabled through that interface; a separate policy change for ordinary random effects is not part of this PR.

Predict requested states jointly. Repeated target rows at the same state share one latent draw. Preserve dependence between interpolation and extrapolation targets, and between new and existing times. Both latent and marginalized fits must provide the same prediction contract and uncertainty.

Declared future categorical levels are permitted. Parameters identified only by their priors must remain visible as such. A new category in fixed-effect predictors still follows Bambi's existing validation.

`include_group_specific=False` must set structured random effects to zero along with ordinary group-specific effects, including ungrouped structured Gaussian effects. This is a conditional zero-effect prediction, not an integration over new effects. Existing HSGP behavior is unchanged.

### Results and integration

Expose interpretable group, coordinate or coefficient, component, and covariance-parameter dimensions. Ragged flattened state storage is acceptable if it retains an unambiguous mapping and serializes cleanly. Do not allocate a large Cartesian state array solely for display.

Require:

- Equivalent syntax produces the same final variable names and coordinates.
- Aliases affect parameters, final effects, summaries, and selectors consistently.
- Prior predictive, `predict(kind="response_params")`, `predict(kind="response")`, prior plotting, model representations, and serialization work.
- Prediction with a fixed seed is reproducible; row reordering or batching within one prediction call does not create independent copies of the same state.
- No stale graph or selector metadata after changing priors or fitting with another method.

Preserve Bambi's observation-level `compute_log_likelihood` contract by evaluating the conditional observation likelihood using sampled or recovered latent effects. Keep observations in their original order. Do not put Kalman one-step predictive increments into the pointwise `log_likelihood` group: that factorization has a different conditioning interpretation.

The collapsed joint/group likelihood used for fitting must be testable separately. Any optional export must label its unit and conditioning clearly. Document that observation-level LOO, leaving out a whole group, and forecasting a future period answer different questions. A separate cross-validation framework is outside this PR.

## Validation and acceptance

The user's local machine has very limited RAM. Local checks must use tiny datasets, small retained arrays, and focused tests. Run one test process or sampler at a time, with no parallel pytest workers or parallel chains. Local sampling smoke checks use one chain and one core with short runs; run backends in separate processes so compiled graphs can be released. Dense numerical references must remain tiny.

Keep full regression runs, repeated statistical calibration, larger benchmarks, and reference-toolchain installation outside the default local loop. Use CI or a suitably provisioned environment for heavier checks and report any pending validation explicitly. Short smoke runs do not establish convergence or posterior recovery. Reuse existing tests and focused scripts; avoid growing a general validation framework.

Fast deterministic tests belong in normal CI. Sampling recovery, calibration, and performance checks may run in a focused slower workflow, with seeds, versions, commands, and retained results. Follow the existing supported Python and dependency matrix; extend CI filters so relevant dependency and feature changes trigger their checks.

### Formulae and design checks

Verify both syntaxes, quoted and bare grouping names/lists, composite identifiers, nested expansion, auxiliary formulas, implicit and explicit intercepts, and numeric-versus-categorical coefficient designs. Test existing formula behavior and ordinary arithmetic as regressions. Assert canonical identity, design equivalence, variable discovery, duplicate handling, namespace behavior, and useful errors.

### Covariance and likelihood checks

For every structure, compare covariance values, log densities, exact Gaussian marginal likelihoods, and parameter gradients against an independent small dense reference. Cover heterogeneous and homogeneous CS, negative correlations, boundary-near values, missing periods, unsorted data, replicates, group-specific time grids, and heteroscedastic observation SDs.

Use float64 checks with justified tolerances and independent finite-difference gradient checks at interior parameter values. Include additive crossed/nested effects held at fixed values to verify partial marginalization rather than accidentally dropping the remaining effects.

Check limiting cases: AR(1) zero correlation, zero gap for repeated states, OU short/long gaps, single-state groups, one-dimensional coefficient designs, empty prediction data, and all-invalid training rows.

For Gaussian recovery, compare conditional means, full covariance, and joint simulated draws against analytic conditioning. Validate covariance between distinct predicted times; matching each marginal variance is insufficient.

### Sampling and family checks

Require end-to-end latent and eligible marginal smoke tests for each wrapper on all four samplers. Include a negative-rho temporal case and both CS scale variants across the matrix. Compare posterior quantities within Monte Carlo error rather than demanding identical draws.

Run meaningful recovery cases for AR(1)/OU and all coefficient covariance families, including correlated intercepts/slopes. Compare latent and marginal posteriors for covariance parameters, fixed effects, residual scale, latent states, and predictions.

Exercise Bernoulli and count likelihoods, auxiliary structured predictors, and representative categorical/ordinal/multivariate response dimensions where Bambi currently allows group-specific effects. Verify automatic fallback and explicit errors for ineligible marginal requests.

Simulation-based calibration and predictive coverage must use independently generated data, varying parameter draws, reproducible seeds, and enough repetitions to assess uncertainty in the check. Use prior predictive checks before recovery runs, convergence diagnostics before interpretation, posterior predictive checks, and suitable PIT/coverage checks. Use 94 percent HDIs when reporting interval recovery. Keep slow statistical acceptance checks out of fragile per-test single-dataset truth-coverage assertions.

Use nutpie for the primary Bayesian validation and prior-sensitivity runs; the sampler compatibility matrix still requires all four backends. Preserve results before post-processing. Record divergences, R-hat, ESS, and diagnostic limitations. Compute the likelihood/prior quantities needed for any reported sensitivity checks coherently for the chosen representation.

### Reference checks and performance

Compare covariance matrices and Gaussian conditional predictions with glmmTMB under matched designs and fixed parameter values. Frequentist glmmTMB estimates are not a Bayesian posterior-equivalence oracle. Use independent Gaussian algebra or a matched-prior implementation for posterior checks.

Benchmark latent and marginal AR(1)/OU with many groups, unequal lengths, missing steps, repeated measurements, and an additional sampled effect. Include a small coefficient-block benchmark for `diag`, both CS variants, and US. Record compilation, sampling, recovery, prediction, peak memory, and effective samples per second for common covariance parameters. Report hardware, versions, model size, priors, and diagnostics.

Demonstrate the expected temporal scaling and a useful measured marginalization benefit on a representative large Gaussian case. Do not assume marginalization is faster for every dataset or hide recovery cost. No machine-specific wall-time threshold belongs in normal CI. Explain any failure to obtain the intended computational benefit before calling the feature complete.

### Completion checklist

The feature is complete only when:

- The paired Formulae interface is implemented, documented, and covered by regression tests.
- All five wrappers and both syntaxes satisfy the formulas and covariance definitions.
- Auxiliary formulas, coefficient designs, nested expansion, and additive composition work.
- Automatic, latent, and explicit marginal modes follow the selection/error contract.
- All four samplers work for latent and eligible marginal inference.
- Conditional state recovery and every required prediction case preserve joint uncertainty.
- Priors, aliases, missing-data handling, summaries, log likelihood, and serialization integrate with Bambi.
- Deterministic tests, sampler checks, statistical validation, and benchmarks have recorded results.
- Relevant Bambi and Formulae regression suites and repository-required checks pass.
- User documentation, examples, API documentation, changelogs, and dependency changes are ready for review.

## Documentation and implementation boundaries

Document both formula syntaxes together. Include examples for player trajectories, player-by-pitcher-hand deviations, player team changes, nested expansion, irregular OU datetimes, categorical covariance, correlated slopes, auxiliary effects, marginal selection, and new-group/new-time prediction.

Explain stationary versus innovation SD, elapsed versus discrete time, categorical contrast bases, homogeneous CS, shared covariance parameters, negative CS bounds, weak identification, and conditional versus population-integrated predictions. Document exact automatic selection and fallback reasons.

Use current Bambi extension points where practical:

| Concern | Starting locations |
| --- | --- |
| Formula construction and component lifecycle | `bambi/formula.py`, `bambi/models.py`, `bambi/model_components.py` |
| Specialized transforms and terms | `bambi/transformations.py`, `bambi/terms/hsgp.py`, `bambi/terms/group_specific.py` |
| Backend assembly and fitting | `bambi/backend/model_components.py`, `bambi/backend/terms.py`, `bambi/backend/pymc.py` |
| Priors and scaling | `bambi/priors`, `bambi/defaults` |
| Existing integration tests | `tests/test_hsgp.py`, `tests/test_glmlss.py`, `tests/test_noncentered.py`, `tests/test_models.py` |
| Test environment and CI | `pyproject.toml`, `CONTRIBUTING.md`, `.github/workflows/ci.yml` |

The internal class/module layout is an implementation choice. Keep covariance math, term construction, and inference responsibilities clear without a general backend redesign. Avoid unrelated family changes, dependency upgrades, workflow rewrites, or notebook reformatting.

Excluded capabilities: AR(2), Toeplitz, reduced rank, general spatial kernels, arbitrary joint marginalization of overlapping effects, group-specific covariance parameters, Laplace marginalization for non-Gaussian responses, and automatic addition of undefined coefficient levels. These exclusions do not remove existing Bambi features.

The four-NUTS-sampler requirement does not require a separate feature project for VI or Bambi's pedagogical Laplace fitter. Preserve existing behavior where the resulting graph supports it, and validate unsupported requests explicitly.

## Sources

The statistical and public-API choices above are the requirements for this implementation. These primary references explain the comparison behavior and available building blocks:

- [glmmTMB covariance design and coefficient bases](https://glmmtmb.github.io/glmmTMB/articles/covstruct.html#construction-of-structured-covariance-matrices)
- [glmmTMB covariance structures](https://glmmtmb.github.io/glmmTMB/reference/glmmTMB.html)
- [glmmTMB prediction and unchanged covariance parameter counts](https://github.com/glmmTMB/glmmTMB/blob/master/glmmTMB/R/predict.R)
- [PyMC Extras state-space filtering and conditional joint trajectory recovery](https://www.pymc.io/projects/extras/en/stable/statespace/generated/pymc_extras.statespace.core.PyMCStateSpace.html)

Existing PyMC Extras infrastructure is a candidate building block, not proof that grouped filtering, Bambi prediction, or all sampler backends work automatically. Prototype compatibility before choosing a dependency.

## Open decisions

No product-scope questions remain from the interview. The Formulae extension interface, backend organization, and numerical implementation remain engineering decisions within this specification.

The first prototypes must establish parser integration, exact likelihood and gradients, recovery, all four sampler translations, and Bambi lifecycle compatibility. If a prototype fails, report the concrete blocker and a proposed adjustment. Do not silently reduce syntax, sampler, prediction, or inference requirements.

## Implementation steps

1. Implement and test the focused Formulae extension for both syntaxes, grouping lists, coefficient designs, and nesting.
2. Build a grouped Gaussian AR(1) prototype with replicates, another sampled effect, exact gradients, joint recovery, and all four samplers.
3. Integrate normalized structured terms, priors, coordinate validation, component lifecycle, and latent inference for all five wrappers.
4. Implement exact one-term Gaussian marginalization, automatic selection, explicit selectors, and method-switching behavior.
5. Complete auxiliary formulas, prediction, aliases, result coordinates, serialization, and log-likelihood integration.
6. Run the regression suites, sampler matrix, recovery/calibration checks, and performance benchmarks; resolve failures.
7. Finish documentation, examples, changelogs, dependency coordination, and separate review summaries for Formulae and Bambi.
