"""Check glmmTMB fixed-parameter fixtures against Gaussian likelihood and recovery."""

import json
import sys
from pathlib import Path

import numpy as np
import pytensor.tensor as pt
from scipy.stats import multivariate_normal

from bambi.structured.gaussian import (
    coefficient_conditional,
    coefficient_logp,
    temporal_conditional,
    temporal_logp,
    temporal_predict,
    transitions,
)

output = Path(sys.argv[1])
reference = json.loads((output / "glmmTMB.json").read_text())
seed = sum(map(ord, "structured-covariance-glmmTMB-reference"))
rng = np.random.default_rng(seed)
results = {}
for structure, case in reference["cases"].items():
    np.testing.assert_allclose(case["sigma"], case["requested_sigma"], atol=1e-12)
    covariance, design = np.array(case["covariance"]), np.array(case["design"])
    y, mean = np.array(case["y"]), np.array(case["mean"])
    residual, variance = y - mean, np.full(len(y), case["sigma"] ** 2)
    if structure in ("ar1", "ou"):
        times = np.arange(5)
        distance = abs(times[:, None] - times)
        expected = 0.7**2 * ((-0.35) ** distance if structure == "ar1" else np.exp(-0.4 * distance))
    else:
        scales = np.array([0.7, 0.4]) if structure in ("diag", "us") else np.array([0.7, 0.5, 0.4])
        if structure == "cs_homogeneous":
            scales[:] = 0.7
        correlation = (
            np.eye(len(scales))
            if structure == "diag"
            else np.full((len(scales), len(scales)), -0.35)
        )
        np.fill_diagonal(correlation, 1)
        expected = scales[:, None] * correlation * scales
    np.testing.assert_allclose(covariance, expected, atol=1e-10)
    observed_covariance = design @ covariance @ design.T + np.diag(variance)
    dense_logp = multivariate_normal.logpdf(y, mean=mean, cov=observed_covariance)
    np.testing.assert_allclose(case["log_likelihood"], dense_logp, atol=1e-7)
    expected_mean = covariance @ design.T @ np.linalg.solve(observed_covariance, residual)
    expected_covariance = covariance - covariance @ design.T @ np.linalg.solve(
        observed_covariance, design @ covariance
    )
    np.testing.assert_allclose(case["conditional_mean"], expected_mean, atol=1e-7)
    np.testing.assert_allclose(case["conditional_covariance"], expected_covariance, atol=1e-7)
    np.testing.assert_allclose(case["predicted_mean"], mean + design @ expected_mean, atol=1e-7)
    if structure in ("ar1", "ou"):
        coordinates = np.array(case["time"])
        order = np.argsort(coordinates, kind="stable")
        observed = coordinates[order]
        starts = np.r_[True, np.zeros(len(y) - 1, dtype=bool)]
        a, q = transitions(
            structure, np.r_[0, np.diff(observed)], -0.35 if structure == "ar1" else 0.4, starts
        )
        value = temporal_logp(residual[order], variance[order], 0.7, a, q).eval()
        unique, indices = np.unique(observed, return_index=True)
        draws = []
        for _ in range(6000):
            fitted = temporal_conditional(residual[order], variance[order], 0.7, a, q, rng)[indices]
            draws.append(
                temporal_predict(
                    times, unique, fitted, 0.7, structure, -0.35 if structure == "ar1" else 0.4, rng
                )
            )
        draws = np.asarray(draws)
    else:
        factor = np.linalg.cholesky(covariance)
        value = coefficient_logp(pt.as_tensor_variable(residual), variance, design, factor).eval()
        draws = np.array(
            [coefficient_conditional(residual, variance, design, factor, rng) for _ in range(6000)]
        )
    np.testing.assert_allclose(value, dense_logp, atol=1e-9)
    np.testing.assert_allclose(draws.mean(0), expected_mean, atol=0.025)
    np.testing.assert_allclose(np.cov(draws.T), expected_covariance, atol=0.02)
    results[structure] = {
        "passed": True,
        "logp_error": float(value - dense_logp),
        "conditional_mean_max_error": float(abs(draws.mean(0) - expected_mean).max()),
        "conditional_covariance_max_error": float(abs(np.cov(draws.T) - expected_covariance).max()),
    }
(output / "comparison.json").write_text(
    json.dumps({"seed": seed, "cases": results}, indent=2) + "\n"
)
print(json.dumps(results, indent=2))
