"""Check joint latent predictions against an independent Gaussian conditional oracle."""

import argparse
from itertools import product
import json
import os
from pathlib import Path

for thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[thread_variable] = "1"


def _correlation(kind, times, parameters, np):
    distances = np.abs(times[:, None] - times[None, :])
    if kind == "ar1":
        return parameters["rho"] ** distances.astype(int)
    if kind == "ou":
        return np.exp(-parameters["decay"] * distances)
    if kind != "toep":
        raise ValueError("The joint oracle supports AR1, OU and Toeplitz blocks")
    # Partial correlation is endpoint correlation conditional on intermediate visits.
    correlations = [1.0]
    for partial in np.atleast_1d(parameters["partial"]):
        width = len(correlations) - 1
        if width == 0:
            correlations.append(float(partial))
            continue
        previous = np.asarray(correlations[1:])
        middle = np.asarray(correlations)[
            np.abs(np.arange(width)[:, None] - np.arange(width)[None, :])
        ]
        weights = np.linalg.solve(middle, previous)
        correlations.append(previous[::-1] @ weights + partial * (1 - previous @ weights))
    return np.asarray(correlations)[distances.astype(int)]


def conditional_moments(model, point, target):
    """Return predictor moments conditional on one fitted posterior draw.

    ``point`` is a posterior Dataset with chain/draw dimensions removed. Shared fitted
    coefficients are held fixed; uncertainty is only in new groups and new times.
    """
    import numpy as np
    from bambi.priors import Prior

    parameter = model.parameters[model.family.likelihood.parent]
    if parameter.offset_terms or parameter.hsgp_terms:
        raise ValueError("The joint oracle does not support offsets or HSGP terms")
    mean = np.zeros(len(target))
    covariance = np.zeros((len(target), len(target)))
    if parameter.design.common is not None:
        common = parameter.design.common.evaluate_new_data(target)
        for name, term in parameter.common_terms.items():
            mean += common[name] @ np.atleast_1d(np.asarray(point[term.label]))
        if parameter.intercept_term is not None:
            mean += float(point[parameter.intercept_term.label])
    for term in parameter.group_specific_terms.values():
        block = term.block
        if block.kind not in {"ar1", "ou", "toep"}:
            raise ValueError("The joint oracle supports temporal structured terms only")
        parameters = {}
        for key, value in term.prior.items():
            parameters[key] = (
                np.asarray(point[f"{term.label}_{term.hyperprior_alias.get(key, key)}"])
                if isinstance(value, Prior)
                else np.asarray(value)
            )
        if block.kind == "toep":
            parameters["partial"] = np.broadcast_to(parameters["partial"], (block.max_lag,))
        observed = np.asarray(block.coordinates, dtype=float)
        times = target[block.variables[0].strip("`")].to_numpy(dtype=float)
        coordinates = np.unique(np.r_[observed, times])
        matrix = float(parameters["sd"]) ** 2 * _correlation(
            block.kind, coordinates, parameters, np
        )
        fitted_indices = np.searchsorted(coordinates, observed)
        target_indices = np.searchsorted(coordinates, times)
        names = [name.strip("`") for name in block.group_variables]
        fitted_groups = {}
        for row, index in zip(
            model.data[names].itertuples(index=False, name=None), term.group_index
        ):
            fitted_groups[row] = index
        target_groups = list(target[names].itertuples(index=False, name=None))
        for group in dict.fromkeys(target_groups):
            rows = np.asarray([i for i, value in enumerate(target_groups) if value == group])
            indices = target_indices[rows]
            conditional = matrix[np.ix_(indices, indices)]
            if group in fitted_groups:
                fitted = np.asarray(point[term.label])[fitted_groups[group]]
                cross = matrix[np.ix_(fitted_indices, indices)]
                weights = np.linalg.solve(matrix[np.ix_(fitted_indices, fitted_indices)], cross)
                mean[rows] += fitted @ weights
                conditional = conditional - cross.T @ weights
            covariance[np.ix_(rows, rows)] += conditional
    return mean, (covariance + covariance.T) / 2


def check_joint_predictions(model, inference, target, draws=2048, seed=2026090801, sparse=False):
    """Compare public predictions with conditional moments at one shared posterior draw."""
    import numpy as np
    import xarray as xr
    import bambi as bmb

    if draws < 100:
        raise ValueError("At least 100 independent forward draws are required")
    posterior = inference["posterior"].to_dataset()
    point = posterior.isel(chain=0, draw=0, drop=True)
    expected_mean, expected_covariance = conditional_moments(model, point, target)
    repeated = posterior.isel(chain=[0], draw=np.zeros(draws, dtype=int)).assign_coords(
        draw=np.arange(draws)
    )
    trace = xr.DataTree.from_dict({"posterior": repeated})
    previous = bmb.config["SPARSE_DOT"]
    try:
        bmb.config["SPARSE_DOT"] = sparse
        model.predict(trace, data=target, random_seed=seed)
    finally:
        bmb.config["SPARSE_DOT"] = previous
    parent = model.family.likelihood.parent
    samples = np.asarray(trace["predictions"][parent]).reshape(draws, len(target))
    if model.family.name in {"bernoulli", "binomial"}:
        if np.any((samples <= 0) | (samples >= 1)):
            raise ValueError("Saturated probabilities prevent a stable latent-scale comparison")
        samples = np.log(samples) - np.log1p(-samples)
    elif model.family.name != "gaussian":
        raise ValueError("Use Gaussian, Bernoulli or binomial for the joint prediction check")
    variance = np.maximum(np.diag(expected_covariance), 0)
    mean_se = np.sqrt(variance / draws)
    covariance_se = np.sqrt((expected_covariance**2 + np.outer(variance, variance)) / (draws - 1))
    mean_error = np.abs(samples.mean(axis=0) - expected_mean)
    covariance_error = np.abs(np.cov(samples, rowvar=False) - expected_covariance)
    if np.any(mean_error > 6 * mean_se + 1e-8):
        raise AssertionError("Joint predictive means disagree with the conditional oracle")
    if np.any(covariance_error > 6 * covariance_se + 1e-8):
        raise AssertionError("Joint predictive covariance disagrees with the conditional oracle")
    return {
        "passed": True,
        "forward_draws": draws,
        "seed": seed,
        "target_rows": len(target),
        "max_mean_error": float(mean_error.max()),
        "max_covariance_error": float(covariance_error.max()),
        "expected_covariance": expected_covariance.tolist(),
    }


def fixture(kind="ar1", sparse=False):
    """Small four-block panel with one withheld three-way group and repeated targets."""
    import bambi as bmb
    import numpy as np
    import pandas as pd
    import xarray as xr

    data = pd.DataFrame(
        product(["s0", "s1"], ["a", "b"], ["u", "v"], [0, 1, 3]),
        columns=["subject", "condition", "context", "year"],
    )
    data["y"] = np.arange(len(data)) % 2
    if kind == "ou":
        data["year"] = data.year.astype(float).replace({1.0: 1.5})
    heldout = (data.subject == "s0") & (data.condition == "a") & (data.context == "u")
    factors = ["subject", "subject:condition", "subject:context", "subject:condition:context"]
    suffix = ", max_lag=5" if kind == "toep" else ""
    wrappers = [f"{kind}(0 + year | {factor}{suffix})" for factor in factors]
    priors = {}
    for index, wrapper in enumerate(wrappers):
        settings = {"sd": 0.2 + 0.05 * index}
        if kind == "ar1":
            settings["rho"] = [-0.4, 0.2, 0.6, -0.2][index]
        elif kind == "ou":
            settings["decay"] = 0.3 + 0.1 * index
        else:
            settings["partial"] = [0.2, -0.1, 0.15, 0.05, -0.05]
        priors[wrapper] = settings
    previous = bmb.config["SPARSE_DOT"]
    try:
        bmb.config["SPARSE_DOT"] = sparse
        model = bmb.Model(
            "y ~ 0 + " + " + ".join(wrappers),
            data.loc[~heldout],
            family="bernoulli",
            priors=priors,
        )
        model.build()
    finally:
        bmb.config["SPARSE_DOT"] = previous
    prior = model.prior_predictive(draws=1, random_seed=2026090802, omit_offsets=False)
    inference = xr.DataTree.from_dict({"posterior": prior["prior"].to_dataset()})
    target = data.loc[heldout].iloc[[0] * 8].copy().reset_index(drop=True)
    target["year"] = [0, 2, 2, 4, 2, 4, 2, 2]
    if kind == "ou":
        target["year"] = target.year.astype(float).replace({2.0: 2.25, 4.0: 4.5})
    target.loc[[4, 5], "subject"] = "new"
    target.loc[[6, 7], "condition"] = "b"
    return model, inference, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["ar1", "ou", "toep"], default="ar1")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--draws", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose another path")
    model, inference, target = fixture(args.kind, args.sparse)
    result = check_joint_predictions(model, inference, target, args.draws, sparse=args.sparse)
    result.update(kind=args.kind, sparse=args.sparse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
