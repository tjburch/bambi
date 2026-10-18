# PyMC 6.3.2 LKJ discrepancies found during structured covariance work

Date: October 17, 2026. This is a local, reproducible bug analysis, not an upstream maintainer decision. No PyMC issue or patch has been published as part of this work.

Three separate errors are confirmed in the inspected PyMC **6.3.2** source: a missing Jacobian in `LKJCorr.logp`, an incorrect forward-draw Beta schedule shared by `LKJCorr` and `LKJCholeskyCov`, and incorrect absolute density normalization. Bambi's structured `us` implementation now constructs the LKJ prior from independent Beta partial correlations and does not use either affected matrix-valued API.

Only version 6.3.2 was inspected. This report does not establish which earlier or later releases are affected.

## How the issue was found

The Bambi requirements specify an LKJ prior for unrestricted coefficient correlations, including vectors with three or more coefficients. The first implementation used `pm.LKJCorr`. Reading its actual output showed that this version returns a **lower Cholesky factor** `L`, despite some correlation-matrix descriptions and examples. The initial Bambi consumer had to be corrected to expose `R = L @ L.T`; interpreting `L` itself as `R` was a Bambi integration error, distinct from the PyMC defects below.

After that correction, a fresh read-only mathematical review checked the probability measure for three-dimensional US. The reviewer derived the density in the free entries of `L`, compared it with the actual `LKJCorr.logp`, and identified a missing Jacobian. Reviewing the forward generator then exposed a separate off-by-one-step Beta schedule.

This was not detected by matching latent and marginal Bambi posteriors: both paths can agree while sharing the same incorrect prior. It also does not appear in two-dimensional prior-shape checks, where the missing Jacobian is zero and the erroneous onion loop does not run. Independent checks in dimensions 2, 3 and 4 were therefore added against the correlation-matrix density and a separately derived prior-moment identity.

While preparing this report, the isolated reproducer confirmed both defects directly. A further independent check of the absolute normalizer found the third error, using the uniform two-dimensional case. The replacement Beta construction already avoids this normalization error; its regression test now also includes `eta=1`.

## Exact source and environment

The installed `pymc/distributions/multivariate.py` is byte-for-byte identical to the file at upstream tag `v6.3.2`, commit `9c2c66a4414bf2f28940484464031fb1ccbebf3d`. Its SHA-256 is:

```text
0f5fe7209490a94d1a0f4f75dbaa71956029db4839bb43b7021c050d2dfc8d1d
```

The reproducer uses Python 3.12.13, PyMC 6.3.2, PyTensor 3.3.4, NumPy 2.2.6 and SciPy 1.15.3, on an Apple M2 with 8 GiB RAM. It uses no MCMC, no Bambi import, and no monkeypatch. It draws 20,000 small three-dimensional prior matrices per implementation, sequentially. Commands, source fingerprints, installed source excerpts, raw draws, results and resource measurements are saved.

Primary implementation references are the immutable [LKJCorr generator and density source](https://github.com/pymc-devs/pymc/blob/9c2c66a4414bf2f28940484464031fb1ccbebf3d/pymc/distributions/multivariate.py#L1487), [shared covariance generator](https://github.com/pymc-devs/pymc/blob/9c2c66a4414bf2f28940484464031fb1ccbebf3d/pymc/distributions/multivariate.py#L1202), and [Cholesky correlation transform](https://github.com/pymc-devs/pymc/blob/9c2c66a4414bf2f28940484464031fb1ccbebf3d/pymc/distributions/transforms.py#L145). The independent target density is described in the [Stan correlation-matrix and Cholesky LKJ reference](https://mc-stan.org/docs/functions-reference/correlation_matrix_distributions.html).

## 1. Missing Jacobian changes the LKJCorr target distribution

The intended LKJ density, relative to the free correlations in `R`, is

```text
p_R(R | eta) = c(K, eta) * det(R)**(eta - 1).
```

PyMC 6.3.2's `LKJCorr` instead represents a lower Cholesky factor `L`, with `R = L @ L.T`. Its density must therefore include the change of coordinates from the free entries of `L` to the free entries of `R`. Using one-based diagonal indices, the required log Jacobian is

```text
log |d R_free / d L_free| = sum(i=2..K) (K-i) * log(L[i,i]).
```

Consequently the Cholesky density has diagonal exponents `2*eta - 2 + K-i`. The inspected base log density supplies only the `2*eta - 2` part. The separate `CholeskyCorrTransform` does not repair this: it correctly maps unconstrained real coordinates to the free entries of `L`, rather than directly to `R`.

The reproducer independently checks that transform with finite differences. Its numerical log Jacobian is `-1.18833146535837`, versus PyMC's `-1.18833146538258`. The transform agrees; the missing term belongs in the base factor density.

For a three-dimensional example, use

```python
L = np.array([
    [1.0, 0.0, 0.0],
    [0.6, 0.8, 0.0],
    [0.2, -0.3, np.sqrt(0.87)],
])
```

At `eta=2`, the log-density difference from the identity should be `log(det(L @ L.T)) + log(0.8) = -0.808692721276137`. PyMC gives `-0.585549169961927`. The discrepancy is exactly `-log(0.8) = 0.223143551314210`. Taking a difference cancels normalizing constants, so this is a shape error independent of the third issue below.

For `K=3, eta=2`, the normalized distribution targeted by the current base density plus default transform implies squared-correlation expectations `(1/5, 1/6, 1/6)`, rather than the intended exchangeable `(1/6, 1/6, 1/6)`. This is an analytic implication, not an estimate from an MCMC run.

## 2. The forward onion generator uses the wrong Beta shape

In `LKJCorrRV._random_corr_matrix`, the code computes `beta_next = beta - 0.5` but draws the new radial variable using the old `beta`. The correct radial second parameter for added row `i` (zero-based, `i >= 2`) is

```text
eta + (K-i-1)/2.
```

The inspected generator uses a value larger by `0.5`. For `K=3, eta=2`, the first added squared radius should use `Beta(1, 2)`; the implementation uses `Beta(1, 2.5)`. Its radius squared has mean `2/7` instead of `1/3`, so each of its two free spherical coordinates has second moment `1/7` instead of `1/6`. This causes an artificial dependence on coefficient order.

`LKJCholeskyCov` calls the same helper. Removing its drawn marginal scales does not remove the incorrect correlation law. Its forward/prior-predictive draws are affected too; its correlation-dependent log-density exponents and Jacobians are different from `LKJCorr` and already have the correct shape.

The intended LKJ identity is `E[R[i,j]**2] = 1 / (2*eta + K - 1)`, for every off-diagonal pair. The independent reference normalizes a Wishart matrix with `df = 2*eta + K - 1`, generated from standard Normal draws. It does not use PyMC's LKJ code or Bambi's Beta construction.

Observed results, with Monte Carlo standard errors in parentheses:

| Generator | R12 squared | R13 squared | R23 squared |
| --- | ---: | ---: | ---: |
| Intended LKJ expectation | 0.166667 | 0.166667 | 0.166667 |
| `pm.LKJCorr` | 0.164508 (0.001308) | 0.144403 (0.001172) | 0.140671 (0.001160) |
| `pm.LKJCholeskyCov`, after scale removal | 0.168489 (0.001343) | 0.141937 (0.001160) | 0.143275 (0.001164) |
| Independent normalized-Wishart reference | 0.166486 (0.001311) | 0.167677 (0.001323) | 0.167454 (0.001321) |

The affected pairs differ from the intended expectation by roughly 19–22 estimated Monte Carlo standard errors. They agree with the analytically implied `1/7` law. This is not evidence of poor MCMC convergence: these are direct independent prior draws.

## 3. Absolute normalizing constants are incorrect

For `K=2, eta=1`, the only free correlation is uniform over `(-1, 1)`. Its density is `1/2`, and its log density is `-log(2)`.

```python
import numpy as np
import pymc as pm

print(pm.logp(pm.LKJCorr.dist(n=2, eta=1), np.eye(2)).eval())
# Observed: +0.6931471824645996
# Required: -0.6931471805599453
```

The inspected implementation adds the helper's `+log(2)`. Its integral over the free correlation is therefore approximately **4**, rather than 1. The reproducer checks that `LKJCholeskyCov` shares this problem, subtracting the independent scale density at unit scales to isolate the correlation contribution.

With fixed `eta`, this constant error alone does not change a posterior's shape. It does invalidate absolute log densities and can matter when comparing marginal likelihoods or inferring `eta` in an API that permits it. `LKJCorr` supports symbolic `eta` in this version; `LKJCholeskyCov.logp` restricts it to a constant. This must not be confused with the missing Jacobian, which changes posterior shape even with fixed `eta`.

This report is not a complete repair of PyMC's normalization helper. Its special-case branches require their own audit; simply negating it everywhere is not an established fix.

## Bambi repair and verification

The structured `us` term uses independent canonical partial correlations. For each lower-triangular entry in zero-based column `j`:

```text
u_ij ~ Beta(alpha_j, alpha_j), independently
alpha_j = eta + (K-j-2)/2
p_ij = 2*u_ij - 1
L_ij = p_ij * product(k<j) sqrt(1-p_ik**2)
L_ii = product(k<i) sqrt(1-p_ik**2)
R = L @ L.T
```

Primitive Beta variables include their own normalized density. The deterministic factor induces the intended LKJ law, so Bambi needs no matrix-density Jacobian correction and no version-dependent PyMC patch. Marginal SDs remain separate, with `K = D @ R @ D` as required by the feature contract.

The actual [backend implementation](../../bambi/structured/backend.py) has independent density/Jacobian regression checks in [test_structured_gaussian.py](../../tests/test_structured_gaussian.py) for dimensions 2, 3 and 4 and `eta` 0.5, 1, 2 and 5. [Prior-moment tests](../../tests/test_structured.py) compare every pair using 8,000 tiny draws per dimension and bounds derived from exact second/fourth moments. The corrected 2D/3D US graph executes with PyMC, nutpie, NumPyro and BlackJAX for both latent and marginal inference. Four-chain small posterior comparisons were rerun for both dimensions and pass convergence/agreement gates.

Those Bambi checks establish the replacement's behavior; they do not establish a fix in PyMC itself. Remaining feature acceptance, calibration and performance work is tracked in the [structured covariance acceptance report](structured-covariance-validation.md).

## Reproduction and retained evidence

From the Bambi checkout, using the existing environment:

```sh
PYTHONNOUSERSITE=1 /usr/bin/time -l .venv/bin/python bambi/examples/pymc_lkj_reproduction.py --output .cache/pymc-lkj-report
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_structured_gaussian.py -k lkj_density --tb=short
```

The [standalone reproducer](../../bambi/examples/pymc_lkj_reproduction.py) uses seed `sum(map(ord, "pymc-6.3.2-lkj-reproduction")) = 2454`, with offsets 1 and 2 for the other generators. [Saved JSON evidence](pymc-lkj-reproduction-results.json) includes the script arguments, versions, hashes and numerical results. Local raw arrays are in `.cache/pymc-lkj-report/raw-priors.npz`; the same directory retains `command.json`, `installed-source.txt`, `results.json`, `run.log`, and the Bambi density-test log.

The reproducer checks numerical facts without asserting that a later PyMC version must remain broken. The standalone reproducer runs no sampler. The measured resource use is recorded in `run.log`; the check remains small and sequential on the user's machine.
