"""Run one explicitly matched reference model with serial nutpie sampling."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

for thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
):
    os.environ[thread_variable] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["ar1", "known", "us-slopes", "us-visits"])
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("phase", choices=["prior", "posterior"])
    parser.add_argument("--chains", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--rho", nargs=4, type=float, help="Required fixed correlations for known mode"
    )
    args = parser.parse_args()
    if min(args.chains, args.warmup, args.draws, args.seed) < 1:
        parser.error("Sampling settings must be positive")
    if (args.mode == "known") != (args.rho is not None):
        parser.error("Supply --rho only for known mode")
    if args.output.exists():
        parser.error("Output already exists; use a separate run directory")

    # Heavy libraries load only after input settings pass validation.
    import bambi as bmb
    import numpy as np
    import pandas as pd
    from validation_identity import create_identity

    data = pd.read_csv(args.data, float_precision="round_trip")
    identity_path = args.data.parent / f"identity-{args.mode}.json"
    identity = json.loads(identity_path.read_text())
    if identity != create_identity(args.data, args.mode):
        raise ValueError("Fixture identity does not match source, data, or model contract")
    if args.mode == "known" and args.rho != identity["priors"]["fixed_rho"]:
        raise ValueError("Fixed correlations differ from the identity contract")
    data["visit"] = pd.Categorical(data["year"], categories=sorted(data["year"].unique()))
    fixed = bmb.Prior("Normal", mu=0, sigma=1.5, auto_scale=False)
    priors = {name: fixed for name in ("Intercept", "x1", "x2")}
    if args.mode in {"ar1", "known"}:
        groups = ["subject", "subject:condition", "subject:context", "subject:condition:context"]
        wrappers = [f"ar1(0 + year | {group})" for group in groups]
        for index, wrapper in enumerate(wrappers):
            priors[wrapper] = {
                "sd": bmb.Prior("HalfNormal", sigma=2.5, auto_scale=False),
                "rho": (
                    args.rho[index]
                    if args.rho is not None
                    else bmb.Prior("Normal", mu=0, sigma=0.5, auto_scale=False)
                ),
            }
    else:
        wrappers = [
            "us(1 + x1 | subject)" if args.mode == "us-slopes" else "us(0 + C(visit) | subject)"
        ]
        priors[wrappers[0]] = {
            "sd": bmb.Prior("HalfNormal", sigma=2.5, auto_scale=False),
            "eta": 2,
        }
    formula = "proportion(y, trials) ~ 1 + x1 + x2 + " + " + ".join(wrappers)
    model = bmb.Model(
        formula,
        data,
        family="binomial",
        priors=priors,
        center_predictors=False,
        noncentered=True,
    )
    parameter = model.parameters[model.family.likelihood.parent]
    if not np.array_equal(np.asarray(parameter.design.common), identity["design"]["fixed_matrix"]):
        raise ValueError("Built fixed-effects design differs from reference contract")
    for index, wrapper in enumerate(wrappers):
        term = parameter.group_specific_terms[wrapper]
        expected_groups = [tuple(row[index]) for row in identity["design"]["row_groups"]]
        seen = {}
        for expected, actual in zip(expected_groups, term.group_index):
            if expected in seen and seen[expected] != actual:
                raise ValueError("Built grouping partition differs from reference contract")
            seen[expected] = actual
        if len(set(seen.values())) != len(seen):
            raise ValueError("Distinct reference groups share a coefficient")
        if args.mode in {"ar1", "known", "us-visits"}:
            coordinates = (
                identity["design"]["time_levels"]
                if args.mode == "us-visits"
                else term.block.coordinates
            )
            actual_times = np.asarray(coordinates)[np.argmax(term.predictor, axis=1)]
            if not np.array_equal(actual_times.astype(float), identity["design"]["row_times"]):
                raise ValueError("Built coordinate design differs from reference contract")
    args.output.mkdir(parents=True)
    (args.output / "data.csv").write_bytes(args.data.read_bytes())
    metadata = {
        "engine": "bambi",
        "identity": identity,
        "mode": args.mode,
        "phase": args.phase,
        "data_md5": hashlib.md5(args.data.read_bytes()).hexdigest(),
        "formula": formula,
        "wrappers": wrappers,
        "fixed_rho": args.rho,
        "chains": args.chains,
        "warmup": args.warmup,
        "draws": args.draws,
        "seed": args.seed,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "bambi",
                "pymc",
                "nutpie",
                "arviz-stats",
                "pytensor",
            )
        },
    }
    (args.output / "settings.json").write_text(json.dumps(metadata, indent=2))
    model.build()
    if args.phase == "prior":
        inference = model.prior_predictive(
            draws=args.draws, random_seed=args.seed, omit_offsets=False
        )
        inference.to_netcdf(args.output / "inference.nc")
        return
    inference = model.fit(
        draws=args.draws,
        tune=args.warmup,
        chains=args.chains,
        cores=1,
        inference_method="nutpie",
        random_seed=args.seed,
        target_accept=0.95,
        omit_offsets=False,
        include_response_params=True,
    )
    inference.to_netcdf(args.output / "inference.nc")
    model.compute_log_likelihood(inference)
    model.compute_log_prior(inference)
    model.predict(inference, kind="response", random_seed=args.seed + 1)
    inference.to_netcdf(args.output / "inference-predictive.nc")
    export_summary(inference, model, metadata, args.output, np, data)


def export_summary(inference, model, metadata, output, np, data):
    """Export matched posterior summaries; missing sampler fields are errors."""
    import arviz_stats as azs
    import xarray as xr
    from reference_metrics import add_binomial_metrics, diagnose_scalar_metrics, summarize_metrics

    posterior = inference["posterior"].to_dataset()
    metrics = {f"beta.{name}": posterior[name] for name in ("x1", "x2")}
    metrics["beta.one"] = posterior["Intercept"]
    probability = posterior[model.family.likelihood.parent]
    likelihood = inference["log_likelihood"].to_dataset()
    if len(likelihood.data_vars) != 1:
        raise ValueError("Expected one response likelihood")
    add_binomial_metrics(
        metrics, probability, data["trials"], likelihood[next(iter(likelihood.data_vars))]
    )
    terms = model.parameters[model.family.likelihood.parent].group_specific_terms
    if metadata["mode"] in {"ar1", "known"}:
        for index, wrapper in enumerate(metadata["wrappers"], start=1):
            metrics[f"sd.{index}"] = posterior[f"{wrapper}_sd"]
            if metadata["mode"] == "ar1":
                metrics[f"rho.{index}"] = posterior[f"{wrapper}_rho"]
            term = terms[wrapper]
            for cell, rows in data.groupby(f"cell{index}", sort=True).groups.items():
                groups = np.unique(term.group_index[rows])
                coordinates = np.unique(np.argmax(term.predictor[rows], axis=1))
                if len(groups) != 1 or len(coordinates) != 1:
                    raise ValueError("Fixture cell maps to multiple latent coefficients")
                metrics[f"latent.{cell}"] = posterior[wrapper].isel(
                    {
                        f"{term.factor_name}_dim": int(groups[0]),
                        f"{term.expr_name}_dim": int(coordinates[0]),
                    },
                    drop=True,
                )
    else:
        wrapper = metadata["wrappers"][0]
        term = terms[wrapper]
        for subject, rows in data.groupby("subject", sort=True).groups.items():
            groups = np.unique(term.group_index[rows])
            if len(groups) != 1:
                raise ValueError("Subject maps to multiple latent groups")
            for index in range(len(term.block.coordinates)):
                metrics[f"latent.subject.{subject}.{index + 1}"] = posterior[wrapper].isel(
                    {f"{term.factor_name}_dim": int(groups[0]), f"{term.expr_name}_dim": index},
                    drop=True,
                )
        chol = posterior[f"{wrapper}_corr_cholesky"].transpose("chain", "draw", ...)
        sd = posterior[f"{wrapper}_sd"].transpose("chain", "draw", ...)
        matrix = np.asarray(chol) * np.asarray(sd)[..., None]
        covariance = matrix @ np.swapaxes(matrix, -1, -2)
        scales = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
        for index in range(scales.shape[-1]):
            metrics[f"sd.{index + 1}"] = xr.DataArray(scales[..., index], dims=("chain", "draw"))
        for i in range(scales.shape[-1]):
            for j in range(i + 1, scales.shape[-1]):
                metrics[f"cor.{i + 1}.{j + 1}"] = xr.DataArray(
                    covariance[..., i, j] / (scales[..., i] * scales[..., j]),
                    dims=("chain", "draw"),
                )
    dataset = xr.Dataset(metrics)
    dataset.to_netcdf(output / "metrics.nc")
    azs.summary(dataset, kind="all", round_to="none", ci_prob=0.94, ci_kind="hdi").to_csv(
        output / "metrics-summary.csv"
    )
    result = {
        "schema_version": 2,
        "identity": metadata["identity"],
        "engine": "bambi",
        "fixed_rho": metadata["fixed_rho"],
        "mode": metadata["mode"],
        "phase": metadata["phase"],
        "data_md5": metadata["data_md5"],
        "diagnostics": diagnose_scalar_metrics(inference, dataset),
        "metrics": summarize_metrics(dataset),
        "coverage": "fixed effects, latent effects, covariance, probability, log likelihood, predictive moments",
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
