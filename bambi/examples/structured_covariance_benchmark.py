"""Measure compilation, sampling, recovery, prediction, and process peak memory.

Defaults are deliberately small. Larger acceptance runs belong on provisioned CI:
    uv run --no-sync python bambi/examples/structured_covariance_benchmark.py \
        ar1 marginal --groups 500 --draws 1000 --tune 1000 --chains 4
Run each structure and inference method in a separate process. No parallel chains
are used. Short default runs do not establish posterior accuracy or efficiency.
"""

# pylint: disable=invalid-name

import argparse
import json
import os
import platform
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path

import arviz_stats as azs
import numpy as np
import nutpie
import pandas as pd
import pytensor
import xarray as xr
from pymc.backends.arviz import patch_nutpie_idata
from pymc.util import get_default_varnames

from structured_covariance_validation import diagnostic_statistics, write_json

import bambi as bmb
from bambi.structured.inference import resolve_plan, recover_effect

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("structure", choices=["ar1", "ou", "diag", "cs", "us"])
parser.add_argument("method", choices=["latent", "marginal", "assess"])
parser.add_argument("--groups", type=int, default=8)
parser.add_argument("--draws", type=int, default=10)
parser.add_argument("--tune", type=int, default=10)
parser.add_argument("--chains", type=int, default=1)
parser.add_argument("--homogeneous", action="store_true")
args = parser.parse_args()
output = Path(".cache/structured-covariance")
output.mkdir(parents=True, exist_ok=True)
if args.method == "assess":
    records = [
        json.loads(path.read_text())
        for path in output.glob(f"benchmark-{args.structure}-*-g*.json")
    ]
    records = [record for record in records if "ess_per_total_second" in record]
    pairs = {}
    for record in records:
        pairs.setdefault((record["groups"], record["homogeneous"]), {})[record["method"]] = record
    comparisons = []
    for (groups, homogeneous), methods in sorted(pairs.items()):
        if set(methods) != {"latent", "marginal"}:
            continue
        latent, marginal = methods["latent"], methods["marginal"]
        comparisons.append(
            {
                "groups": groups,
                "homogeneous": homogeneous,
                "total_speedup": latent["total_inference_seconds"]
                / marginal["total_inference_seconds"],
                "ess_per_total_second_ratio": marginal["ess_per_total_second"]
                / latent["ess_per_total_second"],
                "diagnostics_passed": latent["diagnostics_passed"]
                and marginal["diagnostics_passed"],
            }
        )
    marginal = sorted(
        [
            record
            for record in records
            if record["method"] == "marginal" and not record["homogeneous"]
        ],
        key=lambda record: record["groups"],
    )
    slope = (
        float(
            np.polyfit(
                np.log([record["observations"] for record in marginal]),
                np.log([record["median_logp_seconds"] for record in marginal]),
                1,
            )[0]
        )
        if len(marginal) >= 3
        else None
    )
    required_sizes = {50, 250, 1000} if args.structure in ("ar1", "ou") else {50}
    complete = {item["groups"] for item in comparisons if not item["homogeneous"]} >= required_sizes
    if args.structure == "cs":
        complete = complete and any(
            item["homogeneous"] and item["groups"] == 50 for item in comparisons
        )
    diagnostics_ok = all(item["diagnostics_passed"] for item in comparisons)
    benefit = any(
        item["groups"] >= 1000
        and item["total_speedup"] >= 1.1
        and item["ess_per_total_second_ratio"] >= 1.1
        for item in comparisons
    )
    scaling_ok = slope is not None and slope <= 1.5
    passed = (
        complete
        and diagnostics_ok
        and (args.structure not in ("ar1", "ou") or scaling_ok)
        and (args.structure != "ar1" or benefit)
    )
    report = {
        "passed": passed,
        "complete_sizes": complete,
        "diagnostics_passed": diagnostics_ok,
        "marginal_logp_scaling_exponent": slope,
        "useful_total_benefit": benefit,
        "comparisons": comparisons,
    }
    (output / f"benchmark-{args.structure}-assessment.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    sys.exit(0 if passed else 1)
if min(args.groups, args.draws, args.tune, args.chains) < 1:
    parser.error("Group and sampler sizes must be positive.")

seed = sum(map(ord, "structured-covariance-phase-benchmark"))
rng = np.random.default_rng(seed)
rows = []
team_effects = rng.normal(0, 0.2, 3)
for group in range(args.groups):
    n = int(rng.integers(4, 9))
    coefficients = np.empty(2)
    times = np.r_[0, np.cumsum(rng.integers(1, 4, n - 1))]
    if args.structure in ("ar1", "ou"):
        states = np.empty(n)
        states[0] = rng.normal(0, 0.6)
        for index in range(1, n):
            gap = times[index] - times[index - 1]
            a = (-0.55) ** gap if args.structure == "ar1" else np.exp(-0.4 * gap)
            states[index] = a * states[index - 1] + 0.6 * np.sqrt(1 - a * a) * rng.normal()
    else:
        scales = np.array([0.5, 0.5 if args.homogeneous else 0.25])
        rho = 0.0 if args.structure == "diag" else -0.3
        correlation = np.array([[1.0, rho], [rho, 1.0]])
        coefficients = scales * (np.linalg.cholesky(correlation) @ rng.normal(size=2))
    for index, coordinate in enumerate(times):
        for replicate in range(2 if index % 3 == 0 else 1):
            x, z = rng.normal(size=2)
            team = int(rng.integers(0, 3))
            effect = (
                states[index]
                if args.structure in ("ar1", "ou")
                else np.array([1.0, x]) @ coefficients
            )
            y = 0.3 + 0.4 * x + team_effects[team] + effect + np.exp(-0.8 + 0.15 * z) * rng.normal()
            rows.append((y, x, z, coordinate, str(group), str(team)))
data = pd.DataFrame(rows, columns=["y", "x", "z", "time", "group", "team"])
inner = "time" if args.structure in ("ar1", "ou") else "1+x"
option = ", homogeneous=True" if args.homogeneous else ""
formula = bmb.Formula(
    f"y ~ x + (1|team) + {args.structure}({inner}, by=group{option})", "sigma ~ z"
)
name = (
    f"benchmark-{args.structure}-{args.method}"
    + ("-homogeneous" if args.homogeneous else "")
    + f"-g{args.groups}"
)
(output / f"{name}-command.json").write_text(
    json.dumps({"argv": sys.argv, "seed": seed, "configuration": vars(args)}, indent=2) + "\n"
)
data.to_csv(output / f"{name}-data.csv", index=False)
start = time.perf_counter()
model = bmb.Model(formula, data)
model.structured_plan = resolve_plan(model, args.method, None)
model.build()
# Check the complete generative model before compiling the selected likelihood.
model.prior_predictive(draws=2, random_seed=seed)
built = time.perf_counter()
variables = get_default_varnames(
    model.backend.model.unobserved_value_vars, include_transformed=False
)
names = [
    variable.name for variable in variables if variable.name not in model.family.likelihood.params
]
compiled = nutpie.compile_pymc_model(model.backend.model, var_names=names)
compiled_at = time.perf_counter()
result = nutpie.sample(
    compiled,
    draws=args.draws,
    tune=args.tune,
    chains=args.chains,
    cores=1,
    seed=seed,
    progress_bar=False,
    save_warmup=False,
)
sampled_at = time.perf_counter()
patch_nutpie_idata(result, model.backend.model, sampling_time=sampled_at - compiled_at)
result.to_netcdf(output / f"{name}-raw.nc")
saved_raw_at = time.perf_counter()
result = model.backend._clean_results(result, True, False)  # pylint: disable=protected-access
recover_effect(model, result, seed)
recovered_at = time.perf_counter()
result.to_netcdf(output / f"{name}.nc")
model.predict(result, data=data.iloc[: min(8, len(data))], kind="response", random_seed=seed)
xr.DataTree.from_dict(
    {
        "response_parameters": result.posterior.to_dataset()[["mu", "sigma"]],
        "posterior_predictive": result.posterior_predictive.to_dataset(),
    }
).to_netcdf(output / f"{name}-predictions.nc")
predicted_at = time.perf_counter()
term = next(iter(model.parameters["mu"].structured_terms.values()))
common_names = [
    name
    for name in result.posterior.data_vars
    if name != term.label and not name.endswith(("_correlation", "_correlation_cholesky"))
]
common = result.posterior.to_dataset()[common_names]
if term.structure == "us":
    correlation = result.posterior[term.label + "_correlation"]
    common["correlation"] = correlation.isel(
        {correlation.dims[-2]: 0, correlation.dims[-1]: 1}, drop=True
    )
diagnostic_data = xr.DataTree.from_dict(
    {"posterior": common, "sample_stats": diagnostic_statistics(result)}
)
diagnostic_error, diagnostics = azs.diagnose(diagnostic_data, return_diagnostics=True)
write_json(output / f"{name}-diagnostics.json", diagnostics)
bulk_ess = min(float(value.min()) for value in azs.ess(common).data_vars.values())
tail_ess = min(float(value.min()) for value in azs.ess(common, method="tail").data_vars.values())
covariance_names = [name for name in common if name.startswith(term.label + "_")]
if "correlation" in common:
    covariance_names.append("correlation")
covariance_ess = {
    name: float(value.min()) for name, value in azs.ess(common[covariance_names]).data_vars.items()
}
covariance_ess_min = min(covariance_ess.values())
diagnostics_passed = (
    args.chains >= 2
    and not diagnostic_error
    and bulk_ess >= 100 * args.chains
    and tail_ess >= 100 * args.chains
)
total_inference = predicted_at - start - (saved_raw_at - sampled_at)
compile_logp_started = time.perf_counter()
logp = model.backend.model.compile_logp()
initial = model.backend.model.initial_point()
logp(initial)
logp_compiled_at = time.perf_counter()
evaluation_times = []
for _ in range(10):
    evaluation_started = time.perf_counter()
    logp(initial)
    evaluation_times.append(time.perf_counter() - evaluation_started)
record = dict(
    vars(args),
    seed=seed,
    observations=len(data),
    build_and_prior_seconds=built - start,
    compile_seconds=compiled_at - built,
    sampling_seconds=sampled_at - compiled_at,
    cleaning_and_recovery_seconds=recovered_at - saved_raw_at,
    raw_save_seconds=saved_raw_at - sampled_at,
    save_and_prediction_seconds=predicted_at - recovered_at,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    * (1 if platform.system() == "Darwin" else 1024),
    divergences=int(result.sample_stats["diverging"].sum()),
    diagnostics_passed=diagnostics_passed,
    ess_bulk_min=bulk_ess,
    ess_tail_min=tail_ess,
    total_inference_seconds=total_inference,
    covariance_ess_bulk=covariance_ess,
    covariance_ess_per_total_second={
        key: value / total_inference for key, value in covariance_ess.items()
    },
    ess_per_total_second=covariance_ess_min / total_inference,
    logp_evaluation_compile_seconds=logp_compiled_at - compile_logp_started,
    median_logp_seconds=float(np.median(evaluation_times)),
    platform=platform.platform(),
    compiler_cache={
        "pytensor": str(pytensor.config.compiledir),
        "numba": os.environ.get("NUMBA_CACHE_DIR", "default"),
    },
    versions={p: version(p) for p in ["bambi", "formulae", "pymc", "pytensor", "nutpie"]},
    diagnostic_limit=(
        "Default short runs cannot establish convergence or efficiency; "
        "acceptance requires four converged chains."
    ),
)
(output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
if args.chains >= 2 and not diagnostics_passed:
    raise SystemExit(
        "Benchmark failed convergence criteria; measurements are not efficiency evidence."
    )
