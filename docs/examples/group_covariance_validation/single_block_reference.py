"""Independent prior fixtures and serial Bambi fits for covariance validation."""

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess

KINDS = ("ar1", "ou", "cs", "toep", "us", "four-block-ar1")
GROUPS = (
    ("subject",),
    ("subject", "condition"),
    ("subject", "context"),
    ("subject", "condition", "context"),
)
PRIORS = {
    "beta": "Normal(0,1.5)",
    "sd": "HalfNormal(2.5)",
    "rho": "TruncatedNormal(0,0.5;structure bounds)",
    "decay": "Exponential(1)",
    "partial": "TruncatedNormal(0,0.5;-1,1)",
    "correlation": "LKJ(2)",
}


def bounded_normal(rng, lower=-1, upper=1):
    """Draw the normalized, truncated Normal(0, .5) by rejection."""
    while True:
        value = float(rng.normal(0, 0.5))
        if lower < value < upper:
            return value


def prior_correlation(kind, times, rng):
    """Generate correlations independently of Bambi and the Stan implementation."""
    import numpy as np

    size = len(times)
    gap = np.abs(np.subtract.outer(times, times))
    if kind in {"ar1", "cs"}:
        rho = bounded_normal(rng, -1 if kind == "ar1" else -1 / (size - 1))
        matrix = rho ** gap.astype(int) if kind == "ar1" else np.full((size, size), rho)
        np.fill_diagonal(matrix, 1)
        return matrix, {"rho.1": rho}
    if kind == "ou":
        decay = float(rng.exponential())
        return np.exp(-decay * gap), {"decay.1": decay}
    if kind == "us":
        # The LKJ(eta=2) C-vine uses independent symmetric beta partial correlations.
        lower = np.zeros((size, size))
        lower[0, 0] = 1
        for row in range(1, size):
            remaining = 1.0
            for column in range(row):
                shape = 2 + (size - column - 2) / 2
                partial = 2 * rng.beta(shape, shape) - 1
                lower[row, column] = partial * remaining
                remaining *= np.sqrt(1 - partial**2)
            lower[row, row] = remaining
        matrix = lower @ lower.T
        return matrix, {
            f"cor.{i + 1}.{j + 1}": float(matrix[i, j])
            for i in range(size)
            for j in range(i + 1, size)
        }
    horizon = int(max(times) - min(times))
    partial = [bounded_normal(rng) for _ in range(horizon)]
    acf = [1.0]
    for lag, value in enumerate(partial, start=1):
        if lag == 1:
            acf.append(value)
        else:
            previous = np.asarray(acf[1:])
            covariance = np.asarray(acf)[
                np.abs(np.subtract.outer(np.arange(lag - 1), np.arange(lag - 1)))
            ]
            weights = np.linalg.solve(covariance, previous)
            variance = 1 - previous @ weights
            acf.append(float(previous[::-1] @ weights + value * variance))
    return np.asarray(acf)[gap.astype(int)], {
        f"partial.{i + 1}": value for i, value in enumerate(partial)
    }


def generate_fixture(kind, family, seed):
    """Return a small incomplete-panel fixture, with parameters drawn from the fitted prior."""
    import numpy as np

    if kind not in KINDS or family not in {"bernoulli", "binomial"} or seed < 1:
        raise ValueError("Invalid structure, family, or seed")
    rng = np.random.default_rng(seed)
    times = [0.0, 0.5, 2.25] if kind == "ou" else [0, 1, 3]
    columns = GROUPS if kind == "four-block-ar1" else GROUPS[:1]
    rows = []
    for subject, condition, context, time, replicate in itertools.product(
        range(1, 7), ("a", "b"), ("c", "d"), times, range(2)
    ):
        if subject == 1 and condition == "a" and context == "c":
            continue
        if subject == 2 and time == times[1]:
            continue
        rows.append(
            dict(
                subject=f"s{subject}",
                condition=condition,
                context=context,
                year=time,
                replicate=replicate,
                one=1,
                x1=float(np.round(rng.normal() * 8) / 8),
                x2=int(rng.binomial(1, 0.5)),
                trials=1 if family == "bernoulli" else int(rng.integers(2, 9)),
            )
        )
    beta = rng.normal(0, 1.5, 3)
    truth = {f"beta.{name}": float(value) for name, value in zip(("one", "x1", "x2"), beta)}
    eta = np.asarray([[1, row["x1"], row["x2"]] for row in rows]) @ beta
    blocks = []
    for index, names in enumerate(columns, start=1):
        block_kind = "ar1" if kind == "four-block-ar1" else kind
        correlation, parameters = prior_correlation(block_kind, times, rng)
        scales = np.abs(rng.normal(0, 2.5, len(times) if kind == "us" else 1))
        covariance = correlation * scales[:, None] * scales[None, :]
        groups = sorted({tuple(row[name] for name in names) for row in rows})
        effects = rng.multivariate_normal(np.zeros(len(times)), covariance, len(groups))
        group_ids = [groups.index(tuple(row[name] for name in names)) for row in rows]
        level_ids = [times.index(row["year"]) for row in rows]
        eta += effects[group_ids, level_ids]
        prefix = f"block.{index}." if len(columns) > 1 else ""
        truth.update({prefix + name: value for name, value in parameters.items()})
        truth.update({f"{prefix}sd.{i + 1}": float(value) for i, value in enumerate(scales)})
        # Selected latent effects include a data-informed and an unobserved fitted cell.
        truth.update(
            {
                f"{prefix}coefficient.{g + 1}.{q + 1}": float(effects[g, q])
                for g, q in ((0, 0), (1, 1))
            }
        )
        blocks.append(
            dict(
                kind=block_kind,
                group_columns=list(names),
                groups=groups,
                group_id=[i + 1 for i in group_ids],
                times=times,
                level_id=[i + 1 for i in level_ids],
                max_lag=3 if kind == "toep" else 0,
            )
        )
    probability = np.exp(-np.logaddexp(0, -eta))
    for row, value in zip(rows, probability):
        row["y"] = int(rng.binomial(row["trials"], value))
    for index in (0, len(rows) // 2, len(rows) - 1):
        truth[f"probability.{index + 1}"] = float(probability[index])
    fixture = dict(
        schema_version=1,
        mode=kind,
        family=family,
        seed=seed,
        blocks=blocks,
        prior_draw=True,
        priors=PRIORS.copy(),
    )
    return rows, fixture, truth


def write_fixture(kind, family, seed, output):
    from validation_identity import write_identity

    rows, fixture, truth = generate_fixture(kind, family, seed)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, value in (("fixture.json", fixture), ("truth.json", truth)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False))
    write_identity(output / "data.csv", kind, design=fixture["blocks"], priors=fixture["priors"])
    if kind != "four-block-ar1":
        block = fixture["blocks"][0]
        stan = dict(
            N=len(rows),
            P=3,
            G=len(block["groups"]),
            Q=len(block["times"]),
            structure=KINDS.index(kind) + 1,
            X=[[1, row["x1"], row["x2"]] for row in rows],
            trials=[row["trials"] for row in rows],
            y=[row["y"] for row in rows],
            group_id=block["group_id"],
            level_id=block["level_id"],
            time_index=[int(time) for time in block["times"]],
            time=block["times"],
            max_lag=block["max_lag"],
            use_design=0,
            Z=[],
            prior_only=0,
        )
        (output / "data.json").write_text(json.dumps(stan, indent=2, allow_nan=False))


def validate_fixture_identity(fixture_path):
    from validation_identity import create_identity

    fixture = json.loads((fixture_path / "fixture.json").read_text())
    if (
        fixture.get("schema_version") != 1
        or fixture.get("mode") not in KINDS
        or fixture.get("family") not in {"bernoulli", "binomial"}
        or fixture.get("priors") != PRIORS
        or fixture.get("prior_draw") is not True
    ):
        raise ValueError("Unsupported fixture schema, family, mode, or prior contract")
    with (fixture_path / "data.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    mode = fixture["mode"]
    expected_columns = GROUPS if mode == "four-block-ar1" else GROUPS[:1]
    times = [0.0, 0.5, 2.25] if mode == "ou" else [0, 1, 3]
    if not rows or sorted({float(row["year"]) for row in rows}) != times:
        raise ValueError("Fixture time support changed")
    expected_blocks = []
    for names in expected_columns:
        groups = sorted({tuple(row[name] for name in names) for row in rows})
        expected_blocks.append(
            dict(
                kind="ar1" if mode == "four-block-ar1" else mode,
                group_columns=list(names),
                groups=[list(group) for group in groups],
                group_id=[groups.index(tuple(row[name] for name in names)) + 1 for row in rows],
                times=times,
                level_id=[times.index(float(row["year"])) + 1 for row in rows],
                max_lag=3 if mode == "toep" else 0,
            )
        )
    if fixture["blocks"] != expected_blocks:
        raise ValueError("Fixture block ordering, grouping, or coordinate mapping changed")
    if fixture["family"] == "bernoulli" and any(int(row["trials"]) != 1 for row in rows):
        raise ValueError("Bernoulli fixtures must use one trial")
    identity = json.loads((fixture_path / f"identity-{fixture['mode']}.json").read_text())
    if identity != create_identity(
        fixture_path / "data.csv",
        fixture["mode"],
        design=fixture["blocks"],
        priors=fixture["priors"],
    ):
        raise ValueError("Fixture source, data, design, or priors changed")
    return fixture, identity


def build_model(fixture_path):
    """Build the public Bambi formula without sampling or compiling a log density."""
    fixture, _ = validate_fixture_identity(fixture_path)
    import bambi as bmb
    import pandas as pd

    data = pd.read_csv(fixture_path / "data.csv", float_precision="round_trip")
    data["visit"] = pd.Categorical(data["year"], categories=fixture["blocks"][0]["times"])
    fixed = bmb.Prior("Normal", mu=0, sigma=1.5, auto_scale=False)
    priors = {name: fixed for name in ("Intercept", "x1", "x2")}
    wrappers = []
    for block in fixture["blocks"]:
        group = ":".join(block["group_columns"])
        kind = block["kind"]
        expression = "C(visit)" if kind == "us" else "year"
        suffix = ", max_lag=3" if kind == "toep" else ""
        wrapper = f"{kind}(0 + {expression} | {group}{suffix})"
        wrappers.append(wrapper)
        prior = {"sd": bmb.Prior("HalfNormal", sigma=2.5, auto_scale=False)}
        if kind in {"ar1", "cs"}:
            prior["rho"] = bmb.Prior("Normal", mu=0, sigma=0.5, auto_scale=False)
        elif kind == "ou":
            prior["decay"] = bmb.Prior("Exponential", lam=1, auto_scale=False)
        elif kind == "toep":
            prior["partial"] = bmb.Prior("Normal", mu=0, sigma=0.5, auto_scale=False)
        else:
            prior["eta"] = 2
        priors[wrapper] = prior
    response = "y" if fixture["family"] == "bernoulli" else "proportion(y, trials)"
    model = bmb.Model(
        response + " ~ 1 + x1 + x2 + " + " + ".join(wrappers),
        data,
        family=fixture["family"],
        priors=priors,
        center_predictors=False,
    )
    model.build()
    return model, data, fixture


def bambi_fit(fixture_path, output, chains, warmup, draws, seed):
    import numpy as np
    import xarray as xr
    from reference_metrics import add_binomial_metrics, diagnose_scalar_metrics, summarize_metrics

    model, data, fixture = build_model(fixture_path)
    _, identity = validate_fixture_identity(fixture_path)
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(
        chains=chains, warmup=warmup, draws=draws, seed=seed, fixture=fixture, identity=identity
    )
    (output / "settings.json").write_text(json.dumps(settings, indent=2))
    prior = model.prior_predictive(draws=100, random_seed=seed, omit_offsets=False)
    prior.to_netcdf(output / "prior.nc")
    inference = model.fit(
        draws=draws,
        tune=warmup,
        chains=chains,
        cores=1,
        inference_method="nutpie",
        random_seed=seed,
        target_accept=0.95,
        omit_offsets=False,
        include_response_params=True,
    )
    inference.to_netcdf(output / "inference.nc")
    model.compute_log_likelihood(inference)
    model.compute_log_prior(inference)
    model.predict(inference, kind="response", random_seed=seed + 1)
    inference.to_netcdf(output / "inference-predictive.nc")
    posterior = inference["posterior"].to_dataset()
    metrics = {
        "beta.one": posterior["Intercept"],
        "beta.x1": posterior["x1"],
        "beta.x2": posterior["x2"],
    }
    terms = list(model.parameters[model.family.likelihood.parent].group_specific_terms.values())
    for index, (term, block) in enumerate(zip(terms, fixture["blocks"]), start=1):
        prefix = f"block.{index}." if len(terms) > 1 else ""
        for key in ("sd", "rho", "decay", "partial"):
            name = f"{term.label}_{key}"
            if name not in posterior:
                continue
            value = posterior[name].transpose("chain", "draw", ...)
            if value.ndim == 2:
                metrics[f"{prefix}{key}.1"] = value
            else:
                for i in range(value.shape[-1]):
                    metrics[f"{prefix}{key}.{i + 1}"] = value.isel({value.dims[-1]: i}, drop=True)
        if block["kind"] == "us":
            lower = np.asarray(
                posterior[f"{term.label}_corr_cholesky"].transpose("chain", "draw", ...)
            )
            correlation = lower @ np.swapaxes(lower, -1, -2)
            for i in range(len(block["times"])):
                for j in range(i + 1, len(block["times"])):
                    metrics[f"{prefix}cor.{i + 1}.{j + 1}"] = xr.DataArray(
                        correlation[..., i, j], dims=("chain", "draw")
                    )
        coefficients = posterior[term.label]
        factor_dim = f"{term.factor_name}_dim"
        expression_dim = f"{term.expr_name}_dim"
        fitted_groups = [
            tuple(part[1] for part in json.loads(str(label)))
            for label in coefficients.coords[factor_dim].values
        ]
        for group, label in enumerate(block["groups"]):
            location = fitted_groups.index(tuple(label))
            for level in range(len(block["times"])):
                metrics[f"{prefix}coefficient.{group + 1}.{level + 1}"] = coefficients.isel(
                    {factor_dim: location, expression_dim: level}, drop=True
                )
    probability = posterior[model.family.likelihood.parent].transpose("chain", "draw", ...)
    likelihood = inference["log_likelihood"].to_dataset()
    if len(likelihood.data_vars) != 1:
        raise ValueError("Expected one response log likelihood")
    add_binomial_metrics(metrics, probability, data["trials"], next(iter(likelihood.values())))
    dataset = xr.Dataset(metrics)
    dataset.to_netcdf(output / "metrics.nc")
    diagnostics = diagnose_scalar_metrics(inference, dataset)
    result = dict(
        schema_version=2,
        engine="bambi",
        mode=fixture["mode"],
        phase="posterior",
        data_md5=hashlib.md5((fixture_path / "data.csv").read_bytes()).hexdigest(),
        identity=identity,
        diagnostics=diagnostics,
        metrics=summarize_metrics(dataset),
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("kind", choices=KINDS)
    generate.add_argument("family", choices=("bernoulli", "binomial"))
    generate.add_argument("output", type=Path)
    generate.add_argument("--seed", type=int, required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("fixture", type=Path)
    fit.add_argument("output", type=Path)
    fit.add_argument("--engine", choices=("bambi", "stan"), default="bambi")
    for name, default in (("chains", 4), ("warmup", 1000), ("draws", 1000)):
        fit.add_argument(f"--{name}", type=int, default=default)
    fit.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.seed < 1:
        parser.error("seed must be positive")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ[name] = "1"
    if args.command == "generate":
        write_fixture(args.kind, args.family, args.seed, args.output)
    elif min(args.chains, args.warmup, args.draws) < 1:
        parser.error("Sampling settings must be positive")
    elif args.engine == "bambi":
        bambi_fit(args.fixture, args.output, args.chains, args.warmup, args.draws, args.seed)
    else:
        validate_fixture_identity(args.fixture)
        subprocess.run(
            [
                "Rscript",
                str(Path(__file__).with_name("single_block_stan.R")),
                str(args.fixture),
                str(args.output),
                str(args.chains),
                str(args.warmup),
                str(args.draws),
                str(args.seed),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
