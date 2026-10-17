"""Reproduce PyMC 6.3.2 LKJ density and forward-draw discrepancies without MCMC.

Run from the Bambi checkout with its existing environment. The reproducer does
not import Bambi or change PyMC. It saves the installed source, numerical checks,
and the small raw prior arrays for independent inspection.
"""

import argparse
import hashlib
import inspect
import json
import platform
import resource
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pymc.distributions.multivariate import LKJCorr, LKJCorrRV
from pymc.distributions.transforms import CholeskyCorrTransform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed = sum(map(ord, "pymc-6.3.2-lkj-reproduction"))
    draws, dimension, eta = 20000, 3, 2
    configuration = {
        "argv": sys.argv,
        "seed": seed,
        "draws": draws,
        "dimension": dimension,
        "eta": eta,
        "platform": platform.platform(),
        "versions": {name: version(name) for name in ("pymc", "pytensor", "numpy", "scipy")},
        "source_sha256": {},
    }
    for name, obj in (("multivariate", LKJCorr), ("transforms", CholeskyCorrTransform)):
        source = Path(inspect.getsourcefile(obj))
        configuration["source_sha256"][name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (args.output / "command.json").write_text(json.dumps(configuration, indent=2) + "\n")
    (args.output / "installed-source.txt").write_text(
        inspect.getsource(LKJCorrRV)
        + "\n"
        + inspect.getsource(LKJCorr)
        + "\n"
        + inspect.getsource(CholeskyCorrTransform)
    )

    distribution = pm.LKJCorr.dist(n=dimension, eta=eta)
    lower = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.2, -0.3, np.sqrt(0.87)]])
    identity = np.eye(dimension)
    actual_delta = float((pm.logp(distribution, lower) - pm.logp(distribution, identity)).eval())
    determinant_part = (eta - 1) * np.linalg.slogdet(lower @ lower.T)[1]
    # d(R21,R31,R32)/d(L21,L31,L32) has determinant L22 for K=3.
    factor_to_matrix_log_jacobian = np.log(lower[1, 1])
    expected_delta = determinant_part + factor_to_matrix_log_jacobian
    uniform_logp = float(pm.logp(pm.LKJCorr.dist(n=2, eta=1), np.eye(2)).eval())
    uniform_covariance = pm.LKJCholeskyCov.dist(
        n=2, eta=1, sd_dist=pm.HalfNormal.dist(sigma=1, shape=2), compute_corr=False
    )
    covariance_correlation_logp = float(
        (
            pm.logp(uniform_covariance, np.array([1.0, 0.0, 1.0]))
            - pm.logp(pm.HalfNormal.dist(sigma=1), np.ones(2)).sum()
        ).eval()
    )

    transform = CholeskyCorrTransform(dimension)
    point = np.array([0.3, -0.4, 0.8])
    unconstrained = pt.dvector("unconstrained")
    indices = np.tril_indices(dimension, -1)
    evaluate = pytensor.function([unconstrained], transform.backward(unconstrained)[indices])
    jacobian = np.column_stack(
        [
            (evaluate(point + step * 1e-6) - evaluate(point - step * 1e-6)) / 2e-6
            for step in np.eye(3)
        ]
    )
    numeric_transform_log_jacobian = np.linalg.slogdet(jacobian)[1]
    supplied_transform_log_jacobian = float(transform.log_jac_det(point).eval())
    np.testing.assert_allclose(
        numeric_transform_log_jacobian, supplied_transform_log_jacobian, atol=1e-8
    )

    native = pm.draw(pm.LKJCorr.dist(n=dimension, eta=eta, size=draws), random_seed=seed)
    native_correlation = native @ np.swapaxes(native, -1, -2)
    packed = pm.draw(
        pm.LKJCholeskyCov.dist(
            n=dimension,
            eta=eta,
            sd_dist=pm.HalfNormal.dist(sigma=1, shape=dimension),
            size=draws,
            compute_corr=False,
        ),
        random_seed=seed + 1,
    )
    covariance_factor = np.zeros((draws, dimension, dimension))
    covariance_factor[:, *np.tril_indices(dimension)] = packed
    covariance = covariance_factor @ np.swapaxes(covariance_factor, -1, -2)
    scales = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    covariance_correlation = covariance / scales[..., :, None] / scales[..., None, :]

    # Normalized Wishart with df=2*eta+K-1 independently yields LKJ(eta).
    normal = np.random.default_rng(seed + 2).normal(
        size=(draws, 2 * eta + dimension - 1, dimension)
    )
    wishart = np.swapaxes(normal, -1, -2) @ normal
    scales = np.sqrt(np.diagonal(wishart, axis1=-2, axis2=-1))
    reference = wishart / scales[..., :, None] / scales[..., None, :]
    np.savez_compressed(
        args.output / "raw-priors.npz",
        lkjcorr_factor=native,
        lkjcholeskycov_packed=packed,
        wishart_correlation=reference,
    )
    moments = {}
    expected_moment = 1 / (2 * eta + dimension - 1)
    for name, correlation in (
        ("LKJCorr", native_correlation),
        ("LKJCholeskyCov", covariance_correlation),
        ("Wishart_reference", reference),
    ):
        squared = correlation[:, indices[0], indices[1]] ** 2
        mean = squared.mean(axis=0)
        error = squared.std(axis=0, ddof=1) / np.sqrt(draws)
        moments[name] = {
            "squared_correlation_means": mean.tolist(),
            "mcse": error.tolist(),
            "z_from_LKJ_expectation": ((mean - expected_moment) / error).tolist(),
        }
    record = {
        "pair_order": [[int(i + 1), int(j + 1)] for i, j in zip(*indices)],
        "expected_squared_correlation": expected_moment,
        "density_difference_from_identity": {
            "pymc": actual_delta,
            "expected_factor_density": expected_delta,
            "missing_log_jacobian": factor_to_matrix_log_jacobian,
            "pymc_minus_expected": actual_delta - expected_delta,
        },
        "two_dimensional_uniform_density": {
            "LKJCorr_logp": uniform_logp,
            "LKJCholeskyCov_logp_minus_scale_logp_at_unit_scales": covariance_correlation_logp,
            "expected_logp": -np.log(2),
            "LKJCorr_integrated_mass": 2 * np.exp(uniform_logp),
        },
        "transform_log_jacobian": {
            "finite_difference": numeric_transform_log_jacobian,
            "pymc": supplied_transform_log_jacobian,
        },
        "prior_moments": moments,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if platform.system() == "Darwin" else 1024),
    }
    (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
