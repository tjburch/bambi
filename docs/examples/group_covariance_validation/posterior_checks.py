"""Report saved Bambi reference predictions and hyperprior sensitivity without refitting."""

import argparse
import hashlib
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[variable] = "1"


def hyperprior_terms(mode, wrappers):
    """Select covariance hyperparameters only, never standardized latent offsets."""
    if mode not in {"ar1", "known", "us-slopes", "us-visits"}:
        raise ValueError("No posterior-check adapter for this model mode")
    terms = [(f"{wrapper}_sd", "sd") for wrapper in wrappers]
    if mode == "ar1":
        terms.extend((f"{wrapper}_rho", "rho") for wrapper in wrappers)
    elif mode.startswith("us-"):
        terms.extend((f"{wrapper}_corr_cholesky", "correlation") for wrapper in wrappers)
    return terms


def hyperprior_log_density(posterior, mode, wrappers):
    """Natural-coordinate hyperprior density, up to parameter-independent constants."""
    import numpy as np
    import xarray as xr

    total = None
    for name, kind in hyperprior_terms(mode, wrappers):
        value = posterior[name].transpose("chain", "draw", ...)
        if kind == "correlation":
            # LKJ(2) has density proportional to det(R) in correlation coordinates.
            diagonal = np.diagonal(value.values, axis1=-2, axis2=-1)
            if np.any(diagonal <= 0):
                raise ValueError("Invalid correlation Cholesky factor")
            contribution = xr.DataArray(2 * np.log(diagonal).sum(axis=-1), dims=("chain", "draw"))
        else:
            if kind == "sd" and bool((value <= 0).any()):
                raise ValueError("Invalid marginal scale")
            if kind == "rho" and bool((abs(value) >= 1).any()):
                raise ValueError("Invalid AR1 correlation")
            contribution = -0.5 * (value / (2.5 if kind == "sd" else 0.5)) ** 2
            contribution = contribution.sum(
                dim=[dim for dim in contribution.dims if dim not in ("chain", "draw")]
            )
        total = contribution if total is None else total + contribution
    if total is None or not bool(np.isfinite(total).all()):
        raise ValueError("Missing or nonfinite hyperprior density")
    return total


def predictive_statistics(prediction, data):
    """Aggregate counts with trial weighting and retain replicated-data uncertainty."""
    import numpy as np
    import xarray as xr

    dimensions = [dim for dim in prediction.dims if dim not in ("chain", "draw")]
    if len(dimensions) != 1 or prediction.sizes[dimensions[0]] != len(data):
        raise ValueError("Predictive response must have one matching observation dimension")
    dimension = dimensions[0]
    draws = prediction.transpose("chain", "draw", dimension)
    values = draws.values
    trials = data["trials"].to_numpy()
    if not np.isfinite(values).all() or np.any(values != np.floor(values)):
        raise ValueError("Predictive response must contain finite binomial counts")
    if np.any(values < 0) or np.any(values > trials):
        raise ValueError("Predictive counts exceed binomial support")
    groups = {"overall": np.arange(len(data))}
    for columns in (["subject"], ["year"], ["subject", "condition", "context"]):
        for key, indices in data.groupby(columns, sort=True).indices.items():
            label = json.dumps(key if isinstance(key, tuple) else (key,), default=str)
            groups[f"{':'.join(columns)}:{label}"] = indices
    results = {}
    observed = {}
    for label, indices in groups.items():
        denominator = trials[indices].sum()
        selected = draws.isel({dimension: indices})
        statistics = [
            (
                "zero_fraction",
                (selected == 0).mean(dimension),
                (data["y"].iloc[indices] == 0).mean(),
            ),
        ]
        if denominator > 0:
            statistics.append(
                (
                    "rate",
                    selected.sum(dimension) / denominator,
                    data["y"].iloc[indices].sum() / denominator,
                )
            )
        for statistic, sample, actual in statistics:
            name = f"{statistic}:{label}"
            results[name] = sample
            observed[name] = float(actual)
    return xr.Dataset(results), observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("posterior", type=Path, help="Completed Bambi reference run directory")
    parser.add_argument("prior", type=Path, help="Matching prior-predictive run directory")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; retain the previous report and use another directory")
    posterior_settings = json.loads((args.posterior / "settings.json").read_text())
    prior_settings = json.loads((args.prior / "settings.json").read_text())
    if posterior_settings["identity"] != prior_settings["identity"]:
        parser.error("Prior and posterior run identities differ")
    if posterior_settings["phase"] != "posterior" or prior_settings["phase"] != "prior":
        parser.error("Incorrect prior/posterior phases")
    hyperprior_terms(posterior_settings["mode"], posterior_settings["wrappers"])

    import arviz_stats as azs
    import numpy as np
    import pandas as pd
    import xarray as xr

    from compare_summaries import check_summary

    summary = json.loads((args.posterior / "summary.json").read_text())
    check_summary(summary)
    if summary["identity"] != posterior_settings["identity"]:
        raise ValueError("Saved posterior summary differs from run identity")
    if hashlib.md5((args.posterior / "data.csv").read_bytes()).hexdigest() != summary["data_md5"]:
        raise ValueError("Saved data differ from fitted data")
    inference = xr.open_datatree(args.posterior / "inference-predictive.nc")
    prior = xr.open_datatree(args.prior / "inference.nc")
    data = pd.read_csv(args.posterior / "data.csv", float_precision="round_trip")
    args.output.mkdir(parents=True)
    for phase, tree, group in (
        ("prior", prior, "prior_predictive"),
        ("posterior", inference, "posterior_predictive"),
    ):
        predictive = tree[group].to_dataset()
        if len(predictive.data_vars) != 1:
            raise ValueError("Predictive adapter requires one response variable")
        statistics, observed = predictive_statistics(predictive[next(iter(predictive))], data)
        table = azs.summary(statistics, kind="stats", ci_prob=0.94, ci_kind="hdi", round_to="none")
        table["observed"] = pd.Series(observed)
        table.to_csv(args.output / f"{phase}-predictive.csv")

    posterior = inference["posterior"].to_dataset()
    log_hyperprior = hyperprior_log_density(
        posterior, posterior_settings["mode"], posterior_settings["wrappers"]
    )
    sensitivity_data = xr.DataTree.from_dict(
        {
            "posterior": posterior,
            "log_prior": xr.Dataset({"covariance_hyperprior": log_hyperprior}),
            "log_likelihood": inference["log_likelihood"].to_dataset(),
        }
    )
    names = ["Intercept", "x1", "x2"] + [
        name
        for name, kind in hyperprior_terms(
            posterior_settings["mode"], posterior_settings["wrappers"]
        )
        if kind != "correlation"
    ]
    sensitivity = azs.psense_summary(
        sensitivity_data, var_names=names, prior_var_names=["covariance_hyperprior"], round_to=8
    )
    if not np.isfinite(sensitivity[["prior", "likelihood"]].to_numpy()).all():
        raise ValueError("Nonfinite sensitivity result; adapter cannot certify this run")
    sensitivity.to_csv(args.output / "sensitivity.csv")
    importance = {}
    for group, density in (
        ("hyperprior", log_hyperprior),
        (
            "likelihood",
            inference["log_likelihood"]
            .to_dataset()
            .to_array()
            .sum(
                dim=[
                    dim
                    for dim in inference["log_likelihood"].to_dataset().to_array().dims
                    if dim not in ("chain", "draw")
                ]
            ),
        ),
    ):
        effective = float(azs.ess(density, method="mean")) / density.size
        if not np.isfinite(effective) or effective <= 0:
            raise ValueError("Cannot estimate sensitivity importance-weight efficiency")
        for alpha in (0.8, 0.99, 1.01, 1.25):
            _, k = ((alpha - 1) * density).azstats.psislw(
                dim=("chain", "draw"), r_eff=min(1.0, effective)
            )
            importance[f"{group}:{alpha}"] = float(k)
    refit = any(not np.isfinite(k) or k > 0.7 for k in importance.values())
    (args.output / "importance-diagnostics.json").write_text(
        json.dumps(
            {
                "pareto_k": {name: k if np.isfinite(k) else None for name, k in importance.items()},
                "targeted_refits_required": refit,
            },
            indent=2,
            allow_nan=False,
        )
    )
    status = (
        "Targeted prior/likelihood refits required: importance weights are unreliable."
        if refit
        else (
            "Importance-weight diagnostics pass; inspect sensitivity.csv for substantive sensitivity."
        )
    )
    (args.output / "report.md").write_text(
        "# Covariance posterior checks\n\n"
        f"Source: {summary['identity']['source_commit']}.\n\n"
        "## Predictive checks\n\n"
        "prior-predictive.csv and posterior-predictive.csv report predictive means and 94% HDIs "
        "for trial-weighted rates and zero fractions, overall, by subject, time and three-way group. "
        "Observed statistics appear alongside them. These are in-sample criticism, not held-out calibration.\n\n"
        "Rates are omitted for groups with zero total trials; zero fractions remain defined.\n\n"
        "## Sensitivity\n\n"
        "sensitivity.csv perturbs covariance hyperpriors only; fixed-effect and standardized latent "
        "priors remain unchanged. Correlation densities use natural correlation coordinates. "
        "Likelihood sensitivity is reported separately. Sensitivity is not an automatic implementation failure.\n\n"
        f"{status}\n\n"
        "## Outstanding validation\n\n"
        "Future-time and held-out-subject refits, empirical held-out calibration, joint out-of-sample "
        "reference comparisons and human assessment of predictive discrepancies are not performed by this runner. "
        "This report does not certify PR readiness.\n"
    )
    inference.close()
    prior.close()


if __name__ == "__main__":
    main()
