"""Focused statistical acceptance checks for structured Gaussian effects.

Run each fit in a separate process, then compare the saved results. Local runs
must use serial chains. Calibration replications and large cases belong in CI.
"""

import argparse
import json
import platform
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import arviz_stats as azs
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from scipy import stats

import bambi as bmb

SEED = sum(map(ord, "structured-covariance-statistical-acceptance"))


def write_json(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, (xr.Dataset, xr.DataArray)):
            return item.to_dict()
        raise TypeError(type(item).__name__)

    path.write_text(json.dumps(value, indent=2, default=convert) + "\n")


def simulate(structure, groups, seed, prior_draw=False, composition=False):
    """Generate with independent dense NumPy kernels, without Bambi covariance code."""
    rng = np.random.default_rng(seed)
    temporal = structure in ("ar1", "ou")
    dimension = 3 if structure == "us3" else 2
    homogeneous = structure == "cs_homogeneous"
    intercept, slope, sigma = 0.2, 0.5, 0.35
    sd = np.array([0.6]) if temporal or homogeneous else np.array([0.6, 0.3, 0.2])[:dimension]
    parameter = 0.4 if structure == "ou" else -0.45
    if prior_draw:
        # These are exactly the proper, unscaled priors used in make_model.
        intercept, slope = rng.normal(0, 1, 2)
        sigma = abs(rng.normal(0, 0.6))
        sd = abs(rng.normal(0, 0.8, len(sd)))
        parameter = (
            rng.exponential(1 / 2)
            if structure == "ou"
            else float(stats.truncnorm.rvs(-2, 2, scale=0.5, random_state=rng))
        )
        if structure == "us":
            parameter = 2 * rng.beta(2, 2) - 1
    coefficient_sd = np.broadcast_to(sd, (dimension,))
    correlation = (
        np.eye(dimension) if structure == "diag" else np.full((dimension, dimension), parameter)
    )
    np.fill_diagonal(correlation, 1)
    if structure == "us3":
        if prior_draw:
            # Normalized Wishart with df=2*eta+K-1 independently generates LKJ(eta).
            normal = rng.normal(size=(4 + dimension - 1, dimension))
            wishart = normal.T @ normal
            scales = np.sqrt(np.diag(wishart))
            correlation = wishart / scales[:, None] / scales
        else:
            correlation = 1.3 * np.eye(dimension) - 0.3
    rows, effects = [], []
    holdout_effect = None
    holdout_time = 2 if seed % 2 == 0 else 8
    team_effect = rng.normal(0, 0.15, 3)
    team_slopes = rng.normal(0, 0.15, 3) if composition else np.zeros(3)
    auxiliary = rng.normal(0, 0.1, 3)
    for group in range(groups):
        times = np.array([0, 1, 3, 4, 6, 7])[: 5 + group % 2]
        if temporal:
            process_grid = np.union1d(times, [2, 8]) if prior_draw else times
            distance = np.abs(process_grid[:, None] - process_grid)
            kernel = parameter**distance if structure == "ar1" else np.exp(-parameter * distance)
            process = rng.multivariate_normal(np.zeros(len(process_grid)), sd[0] ** 2 * kernel)
            effect = process[np.searchsorted(process_grid, times)]
            if group == 0 and prior_draw:
                holdout_effect = process[np.searchsorted(process_grid, holdout_time)]
        else:
            covariance = coefficient_sd[:, None] * correlation * coefficient_sd
            effect = rng.multivariate_normal(np.zeros(dimension), covariance)
            if group == 0 and prior_draw:
                holdout_effect = effect @ (1.5 ** np.arange(dimension))
        effects.extend(effect)
        for index, coordinate in enumerate(times):
            for _ in range(2 if index % 2 == 0 else 1):
                x, z = rng.normal(size=2)
                team = int(rng.integers(0, 3))
                contribution = effect[index] if temporal else effect @ (x ** np.arange(dimension))
                extra = team_effect[team] + team_slopes[team] * x if composition else 0
                noise = sigma * np.exp(0.1 * z + auxiliary[team]) if composition else sigma
                y = intercept + slope * x + contribution + extra + noise * rng.normal()
                rows.append((y, x, z, coordinate, f"g{group:03}", f"t{team}", contribution))
    data = pd.DataFrame(rows, columns=["y", "x", "z", "time", "g", "team", "true_effect"])
    data = data.iloc[rng.permutation(len(data))].reset_index(drop=True)
    truth = {"Intercept": intercept, "x": slope, "sigma": sigma, "effect_sd": sd.tolist()}
    if structure != "diag":
        truth["effect_decay" if structure == "ou" else "correlation"] = parameter
    if structure == "us3":
        truth["correlation"] = correlation[np.tril_indices(dimension, -1)].tolist()
    holdout = data.iloc[[0]].copy()
    if prior_draw:
        holdout["g"], holdout["time"], holdout["x"] = "g000", holdout_time, 1.5
        holdout["true_effect"] = holdout_effect
    row = holdout.iloc[0]
    team = int(row.team[1:])
    noise = sigma * np.exp(0.1 * row.z + auxiliary[team]) if composition else sigma
    holdout["y"] = (
        intercept
        + slope * row.x
        + row.true_effect
        + (team_effect[team] + team_slopes[team] * row.x if composition else 0)
        + noise * rng.normal()
    )
    return data, truth, effects, holdout


def make_model(data, structure, composition=False, prior_scale=1):
    wrapper = "cs" if structure == "cs_homogeneous" else ("us" if structure == "us3" else structure)
    inner = "time" if wrapper in ("ar1", "ou") else "1+x"
    if structure == "us3":
        inner += "+I(x**2)"
    option = ", homogeneous=True" if structure == "cs_homogeneous" else ""
    term = f"{wrapper}({inner}, by=g{option})"
    # Moderate proper priors make the acceptance experiment identifiable at small sizes.
    covariance = {"sd": bmb.Prior("HalfNormal", sigma=0.8 * prior_scale)}
    if wrapper in ("ar1", "cs"):
        covariance["rho"] = bmb.Prior("Normal", mu=0, sigma=0.5)
    elif wrapper == "ou":
        covariance["decay"] = bmb.Prior("Exponential", lam=2)
    elif wrapper == "us":
        covariance["eta"] = 2
    priors = {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
        "x": bmb.Prior("Normal", mu=0, sigma=1),
        term: covariance,
    }
    formula = f"y ~ x + {term}"
    if composition:
        formula = bmb.Formula(
            formula + " + (1|team) + diag(0+x, by=team)", "sigma ~ z + diag(1, by=team)"
        )
        priors["mu"] = {
            "Intercept": priors.pop("Intercept"),
            "x": priors.pop("x"),
            term: priors.pop(term),
            "1|team": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=0.3)),
            "diag(0+x, by=team)": {"sd": 0.15},
        }
        priors["sigma"] = {
            "Intercept": bmb.Prior("Normal", mu=-1, sigma=0.5),
            "z": bmb.Prior("Normal", mu=0, sigma=0.2),
            "diag(1, by=team)": {"sd": 0.1},
        }
    else:
        priors["sigma"] = bmb.Prior("HalfNormal", sigma=0.6)
    model = bmb.Model(formula, data, priors=priors, auto_scale=False, center_predictors=False)
    model.set_alias({"mu": {term: "effect"}} if composition else {term: "effect"})
    return model


def monitored(idata):
    posterior = idata.posterior.to_dataset()
    names = [
        name
        for name, value in posterior.data_vars.items()
        if not name.endswith(("_correlation", "_correlation_cholesky", "_offset"))
        and name != "mu"
        and bool((value.max(("chain", "draw")) != value.min(("chain", "draw"))).any())
    ]
    result = posterior[names]
    if "effect_correlation" in posterior:
        correlation = posterior.effect_correlation
        rows, columns = np.tril_indices(correlation.shape[-1], -1)
        result["correlation"] = xr.DataArray(
            correlation.values[..., rows, columns], dims=("chain", "draw", "correlation_pair")
        )
    return result


def diagnostic_statistics(idata):
    statistics = idata.sample_stats.to_dataset()
    if "maxdepth_reached" in statistics and "reached_max_treedepth" not in statistics:
        statistics = statistics.assign(reached_max_treedepth=statistics.maxdepth_reached)
    return statistics


def diagnose(idata, output):
    variables = monitored(idata)
    diagnostic_data = xr.DataTree.from_dict(
        {"posterior": variables, "sample_stats": diagnostic_statistics(idata)}
    )
    failed, details = azs.diagnose(diagnostic_data, return_diagnostics=True)
    summary = azs.summary(variables, ci_prob=0.94, ci_kind="hdi")
    summary.to_csv(output / "summary.csv")
    rhat, bulk = azs.rhat(variables), azs.ess(variables)
    tail = azs.ess(variables, method="tail")
    rhat_max = max(float(value.max()) for value in rhat.data_vars.values())
    ess_min = min(float(value.min()) for value in bulk.data_vars.values())
    tail_min = min(float(value.min()) for value in tail.data_vars.values())
    finite = all(
        np.isfinite(value).all()
        for dataset in (rhat, bulk, tail)
        for value in dataset.data_vars.values()
    )
    passed = (
        not failed
        and finite
        and ess_min >= 100 * variables.sizes["chain"]
        and tail_min >= 100 * variables.sizes["chain"]
    )
    record = {
        "passed": bool(passed),
        "rhat_max": rhat_max,
        "ess_bulk_min": ess_min,
        "ess_tail_min": tail_min,
        "details": details,
    }
    write_json(output / "diagnostics.json", record)
    return record


def fit(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    configuration = {
        name: getattr(args, name)
        for name in (
            "seed",
            "structure",
            "method",
            "groups",
            "draws",
            "tune",
            "chains",
            "sampler",
            "prior_draw",
            "prior_scale",
            "composition",
        )
    }
    configuration["target_accept"] = 0.99 if args.composition else 0.97
    write_json(
        output / "command.json",
        {
            "argv": sys.argv,
            "configuration": configuration,
            "platform": platform.platform(),
            "python": sys.version,
            "versions": {
                name: version(name)
                for name in (
                    "bambi",
                    "formulae",
                    "pymc",
                    "pytensor",
                    "nutpie",
                    "jax",
                    "numpyro",
                    "blackjax",
                    "arviz_stats",
                    "numpy",
                    "scipy",
                    "pandas",
                    "xarray",
                )
            },
        },
    )
    if args.prior_draw and (args.composition or args.prior_scale != 1):
        raise ValueError("Prior-draw calibration uses the base single-effect prior specification.")
    data, truth, effects, holdout = simulate(
        args.structure, args.groups, args.seed, args.prior_draw, args.composition
    )
    data.to_csv(output / "data.csv", index=False)
    write_json(output / "truth.json", {"parameters": truth, "effects": effects})
    model = make_model(data, args.structure, args.composition, args.prior_scale)
    model.build()
    (output / "model.txt").write_text(str(model))
    prior = model.prior_predictive(draws=100, random_seed=args.seed + 1)
    prior.to_netcdf(output / "prior.nc")
    prior_y = prior.prior_predictive.y.values
    write_json(
        output / "prior_predictive.json",
        {
            "finite": bool(np.isfinite(prior_y).all()),
            "quantiles": np.quantile(prior_y, [0.005, 0.5, 0.995]),
        },
    )
    run_seed = args.seed + (100 if args.method == "latent" else 200)
    sampler = pm.sample

    def preserve_sampler_output(*positional, **keywords):
        raw = sampler(*positional, **keywords)
        raw.to_netcdf(output / "sampler-raw.nc")
        return raw

    # Capture the sampler result before Bambi cleans dimensions or recovers effects.
    with patch("bambi.backend.pymc.pm.sample", side_effect=preserve_sampler_output):
        result = model.fit(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=1,
            random_seed=run_seed,
            inference_method=args.sampler,
            structured_method=args.method,
            target_accept=configuration["target_accept"],
            progressbar=False,
            compute_convergence_checks=False,
        )
    sampled = time.perf_counter()
    result.to_netcdf(output / "posterior.nc")
    diagnostics = diagnose(result, output)
    model.compute_log_likelihood(result)
    model.predict(result, kind="response", random_seed=run_seed + 1)
    result.to_netcdf(output / "posterior_predictive.nc")
    observed = data.y.to_numpy()
    predicted = result.posterior_predictive.y.values.reshape(-1, len(data))
    # In-sample PPC is descriptive, not a calibrated out-of-sample PIT claim.
    write_json(
        output / "posterior_predictive.json",
        {
            "observed_mean": observed.mean(),
            "observed_sd": observed.std(),
            "mean_tail_probability": (predicted.mean(axis=1) <= observed.mean()).mean(),
            "sd_tail_probability": (predicted.std(axis=1) <= observed.std()).mean(),
        },
    )
    model.predict(result, data=holdout, kind="response", random_seed=run_seed + 2)
    holdout.to_csv(output / "holdout_data.csv", index=False)
    result.to_netcdf(output / "holdout.nc")
    target = data.iloc[[0, 0, 0, 0]].copy()
    target["g"] = ["g000", "g000", "g000", "new"]
    target["time"] = [2, 8, 2, 2]
    target["x"] = [-0.5, 1.5, -0.5, -0.5]
    model.predict(
        result, data=target, kind="response", sample_new_groups=True, random_seed=run_seed + 3
    )
    target.to_csv(output / "prediction_data.csv", index=False)
    result.to_netcdf(output / "predictions.nc")
    np.testing.assert_array_equal(
        result.posterior.mu.values[:, :, 0], result.posterior.mu.values[:, :, 2]
    )
    record = {
        "passed": diagnostics["passed"],
        "fit_and_recovery_seconds": sampled - started,
        "total_seconds": time.perf_counter() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if platform.system() == "Darwin" else 1024),
        "diagnostics": diagnostics,
    }
    write_json(output / "result.json", record)
    print(
        json.dumps({key: value for key, value in record.items() if key != "diagnostics"}, indent=2)
    )
    if not record["passed"]:
        raise SystemExit("Convergence criteria failed; raw posterior and diagnostics retained.")


def comparison_quantities(path):
    idata = xr.open_datatree(path / "posterior.nc").load()
    variables = monitored(idata)
    quantities = variables.copy()
    for name, value in variables.data_vars.items():
        quantities[name + "_second_moment"] = value**2
    effect = variables.effect
    flat = effect.stack(coefficient=[dim for dim in effect.dims if dim not in ("chain", "draw")])
    selected = flat.values[..., :4]
    quantities["state_cross_moments"] = xr.DataArray(
        selected[..., :, None] * selected[..., None, :],
        dims=("chain", "draw", "state_row", "state_column"),
    )
    predicted = xr.open_datatree(path / "predictions.nc").load().posterior.mu
    quantities["prediction"] = predicted
    quantities["prediction_second_moment"] = predicted**2
    values = predicted.values
    quantities["prediction_cross_moments"] = xr.DataArray(
        values[..., :, None] * values[..., None, :],
        dims=("chain", "draw", "prediction_row", "prediction_column"),
    )
    configuration = json.loads((path / "command.json").read_text())["configuration"]
    structure = configuration["structure"]
    sd = idata.posterior.effect_sd.values
    if structure in ("ar1", "ou"):
        grid = np.array([0, 1, 3, 4, 6])
        distance = abs(grid[:, None] - grid)
        if structure == "ar1":
            kernel = idata.posterior.effect_rho.values[..., None, None] ** distance
        else:
            kernel = np.exp(-idata.posterior.effect_decay.values[..., None, None] * distance)
        covariance = sd[..., None, None] ** 2 * kernel
    else:
        if structure == "cs_homogeneous":
            sd = np.repeat(sd[..., None], 2, axis=-1)
        if structure in ("cs", "cs_homogeneous"):
            rho = idata.posterior.effect_rho.values[..., None, None]
            correlation = (1 - rho) * np.eye(2) + rho
        elif structure in ("us", "us3"):
            correlation = idata.posterior.effect_correlation.values
        else:
            correlation = np.eye(2)
        covariance = sd[..., :, None] * correlation * sd[..., None, :]
    quantities["covariance"] = xr.DataArray(
        covariance, dims=("chain", "draw", "covariance_row", "covariance_column")
    )
    return quantities


def compare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    configurations = [
        json.loads((path / "command.json").read_text())["configuration"]
        for path in (args.latent, args.marginal)
    ]
    for key in ("seed", "structure", "groups", "prior_draw", "prior_scale", "composition"):
        if configurations[0][key] != configurations[1][key]:
            raise ValueError(f"Comparison has incompatible {key}.")
    if [item["method"] for item in configurations] != ["latent", "marginal"]:
        raise ValueError("Comparison requires latent and marginal fits in that order.")
    for filename in ("data.csv", "truth.json", "prediction_data.csv"):
        if (args.latent / filename).read_bytes() != (args.marginal / filename).read_bytes():
            raise ValueError(f"Comparison inputs differ: {filename}")
    left, right = comparison_quantities(args.latent), comparison_quantities(args.marginal)
    if set(left) != set(right):
        raise ValueError("Compared fits expose different variables.")
    n = sum(
        value.size // (value.sizes["chain"] * value.sizes["draw"])
        for value in left.data_vars.values()
    )
    # Family-wise 1% threshold for mean and second-moment comparisons using MCMC MCSE.
    threshold = float(stats.norm.isf(0.01 / (2 * n)))
    comparisons = []
    for name in left:
        a, b = left[name], right[name]
        difference = a.mean(("chain", "draw")) - b.mean(("chain", "draw"))
        error = np.sqrt(azs.mcse(a) ** 2 + azs.mcse(b) ** 2)
        z = np.asarray(xr.where((error == 0) & (difference == 0), 0, abs(difference) / error))
        comparisons.append(
            {
                "quantity": name,
                "max_mcse_z": float(np.nanmax(z)),
                "passed": bool(np.isfinite(z).all() and np.all(z < threshold)),
            }
        )
    convergence = [
        json.loads((path / "diagnostics.json").read_text())["passed"]
        for path in (args.latent, args.marginal)
    ]
    record = {
        "passed": all(convergence) and all(item["passed"] for item in comparisons),
        "convergence_passed": convergence,
        "family_wise_alpha": 0.01,
        "threshold": threshold,
        "comparisons": comparisons,
    }
    write_json(args.output / "comparison.json", record)
    print(json.dumps(record, indent=2))
    if not record["passed"]:
        raise SystemExit(
            "Posterior comparison failed; inspect convergence and comparison artifacts."
        )


def sensitivity(args):
    """Report sensitivity from actual prior refits, without representation-dependent scores."""
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = monitored(xr.open_datatree(args.baseline / "posterior.nc").load())
    baseline_config = json.loads((args.baseline / "command.json").read_text())["configuration"]
    records = []
    passed = json.loads((args.baseline / "diagnostics.json").read_text())["passed"]
    for path in args.refits:
        configuration = json.loads((path / "command.json").read_text())["configuration"]
        if configuration["prior_scale"] == baseline_config["prior_scale"]:
            raise ValueError("Sensitivity requires a changed prior scale.")
        for key in ("structure", "seed", "groups", "method", "composition", "prior_draw"):
            if configuration[key] != baseline_config[key]:
                raise ValueError(f"Sensitivity refits differ in {key}.")
        if (path / "data.csv").read_bytes() != (args.baseline / "data.csv").read_bytes():
            raise ValueError("Sensitivity refits must use the same data.")
        shifted = monitored(xr.open_datatree(path / "posterior.nc").load())
        diagnostics = json.loads((path / "diagnostics.json").read_text())
        passed = passed and diagnostics["passed"]
        changes = {}
        for name in baseline:
            shift = (
                shifted[name].mean(("chain", "draw")) - baseline[name].mean(("chain", "draw"))
            ) / baseline[name].std(("chain", "draw"))
            changes[name] = {
                "max_mean_shift_in_baseline_sd": float(abs(shift).max()),
                "baseline_94_hdi": np.asarray(azs.hdi(baseline[name], prob=0.94)).tolist(),
                "refit_94_hdi": np.asarray(azs.hdi(shifted[name], prob=0.94)).tolist(),
            }
        records.append(
            {
                "prior_scale": configuration["prior_scale"],
                "diagnostics_passed": diagnostics["passed"],
                "quantities": changes,
            }
        )
    write_json(
        args.output / "sensitivity.json",
        {
            "convergence_passed": bool(passed),
            "note": (
                "Shifts above 0.3 posterior SD flag prior dependence for scientific review; "
                "they are not implementation failures."
            ),
            "refits": records,
        },
    )
    if not passed:
        raise SystemExit("Sensitivity refit convergence failed; interpretation is pending.")


def calibration(args):
    """Assess independent prior-draw replications; one scalar rank per fitted quantity."""
    args.output.mkdir(parents=True, exist_ok=True)
    ranks, coverage = {}, {}
    failed, seeds, specifications = [], set(), set()
    for path in sorted(args.input.glob("replicate-*/marginal")):
        if not (path / "result.json").exists():
            failed.append(str(path))
            continue
        result = json.loads((path / "result.json").read_text())
        configuration = json.loads((path / "command.json").read_text())["configuration"]
        if (
            not configuration["prior_draw"]
            or configuration["composition"]
            or configuration["prior_scale"] != 1
        ):
            raise ValueError(f"Calibration requires compatible independent prior draws: {path}")
        if configuration["seed"] in seeds:
            raise ValueError(f"Duplicate simulation seed: {path}")
        seeds.add(configuration["seed"])
        specifications.add(
            (configuration["structure"], configuration["groups"], configuration["method"])
        )
        if len(specifications) > 1 or configuration["method"] != "marginal":
            raise ValueError("Calibration inputs have incompatible specifications.")
        rank_rng = np.random.default_rng(configuration["seed"] + 1000)
        if not result["passed"]:
            failed.append(str(path))
        simulation = json.loads((path / "truth.json").read_text())
        truth = simulation["parameters"]
        idata = xr.open_datatree(path / "posterior.nc").load()
        variables = monitored(idata)
        for name in (
            "Intercept",
            "x",
            "sigma",
            "effect_sd",
            "effect_rho",
            "effect_decay",
            "correlation",
            "effect",
        ):
            if name not in variables:
                continue
            key = "correlation" if name == "effect_rho" else name
            values = variables[name].values.reshape(
                variables.sizes["chain"], variables.sizes["draw"], -1
            )
            actual = np.asarray(simulation["effects"] if name == "effect" else truth[key]).reshape(
                -1
            )
            for index in range(values.shape[-1]):
                label = f"{name}[{index}]" if len(actual) > 1 else name
                draws = xr.DataArray(values[..., index], dims=("chain", "draw"))
                ess = float(azs.ess(draws))
                flat = draws.values.reshape(-1)
                # Conservative thinning reduces residual MCMC dependence; ranks are approximate.
                count = min(100, max(1, int(ess / 4)))
                samples = flat[np.linspace(0, flat.size - 1, count, dtype=int)]
                ranks.setdefault(label, []).append(
                    float(
                        ((samples < actual[index]).sum() + rank_rng.uniform()) / (samples.size + 1)
                    )
                )
                bounds = np.asarray(azs.hdi(draws, prob=0.94)).reshape(-1)
                coverage.setdefault(label, []).append(
                    bool(bounds.min() <= actual[index] <= bounds.max())
                )
        prediction = (
            xr.open_datatree(path / "holdout.nc").load().posterior_predictive.y.values.reshape(-1)
        )
        actual = pd.read_csv(path / "holdout_data.csv").y.iloc[0]
        samples = prediction[np.linspace(0, prediction.size - 1, 100, dtype=int)]
        ranks.setdefault("heldout_response", []).append(
            float(((samples < actual).sum() + rank_rng.uniform()) / (samples.size + 1))
        )
        interval = np.asarray(azs.hdi(xr.DataArray(prediction, dims="draw"), dim="draw", prob=0.94))
        coverage.setdefault("heldout_response", []).append(
            bool(interval[0] <= actual <= interval[1])
        )
    minimum = args.replicates
    results = {}
    for name, values in ranks.items():
        hits = int(sum(coverage[name]))
        interval = stats.binomtest(hits, len(values), 0.94).proportion_ci(
            confidence_level=1 - 0.01 / len(ranks)
        )
        uniform = stats.kstest(values, "uniform")
        results[name] = {
            "replicates": len(values),
            "ranks": values,
            "coverage_94": hits / len(values),
            "coverage_familywise_99_interval": [interval.low, interval.high],
            "rank_uniformity_p": uniform.pvalue,
            "passed": bool(
                len(values) >= minimum
                and uniform.pvalue > 0.01 / len(ranks)
                and interval.low <= 0.94 <= interval.high
            ),
        }
    passed = bool(results) and not failed and all(item["passed"] for item in results.values())
    write_json(
        args.output / "calibration.json",
        {"passed": passed, "failed_or_missing_fits": failed, "quantities": results},
    )
    if not passed:
        raise SystemExit("Calibration failed or lacks enough independent converged replications.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("fit")
    fitting.add_argument(
        "structure", choices=["ar1", "ou", "diag", "cs", "cs_homogeneous", "us", "us3"]
    )
    fitting.add_argument("method", choices=["latent", "marginal"])
    fitting.add_argument("--groups", type=int, default=12)
    fitting.add_argument("--draws", type=int, default=1000)
    fitting.add_argument("--tune", type=int, default=1000)
    fitting.add_argument("--chains", type=int, default=4)
    fitting.add_argument(
        "--sampler", choices=["nutpie", "pymc", "numpyro", "blackjax"], default="nutpie"
    )
    fitting.add_argument("--seed", type=int, default=SEED)
    fitting.add_argument("--prior-draw", action="store_true")
    fitting.add_argument("--composition", action="store_true")
    fitting.add_argument("--prior-scale", type=float, default=1)
    fitting.add_argument("--output", type=Path, required=True)
    fitting.set_defaults(run=fit)
    comparing = commands.add_parser("compare")
    comparing.add_argument("latent", type=Path)
    comparing.add_argument("marginal", type=Path)
    comparing.add_argument("--output", type=Path, required=True)
    comparing.set_defaults(run=compare)
    sensitivity_parser = commands.add_parser("sensitivity")
    sensitivity_parser.add_argument("baseline", type=Path)
    sensitivity_parser.add_argument("refits", type=Path, nargs="+")
    sensitivity_parser.add_argument("--output", type=Path, required=True)
    sensitivity_parser.set_defaults(run=sensitivity)
    calibrating = commands.add_parser("calibration")
    calibrating.add_argument("input", type=Path)
    calibrating.add_argument("--replicates", type=int, default=100)
    calibrating.add_argument("--output", type=Path, required=True)
    calibrating.set_defaults(run=calibration)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
