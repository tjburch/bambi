"""Measure structured model construction and optional prediction without MCMC."""

import argparse
from importlib import metadata
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def positive_integer(value):
    """Parse positive workload dimensions before loading numerical libraries."""
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=positive_integer, default=4)
    parser.add_argument("--times", type=positive_integer, default=4)
    parser.add_argument("--replicates", type=positive_integer, default=1)
    parser.add_argument("--conditions", type=positive_integer, default=2)
    parser.add_argument("--contexts", type=positive_integer, default=2)
    parser.add_argument("--interaction-depth", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--compile", action="store_true", dest="compile_logp")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.times < 2:
        parser.error("--times must be at least two for the AR1 benchmark")
    if args.output.exists() or args.output.is_symlink():
        parser.error("--output already exists; choose a different file")
    if not args.output.parent.is_dir():
        parser.error("--output parent directory must already exist")
    if platform.system() not in {"Darwin", "Linux"}:
        parser.error("peak RSS reporting currently supports macOS and Linux only")
    return args


def peak_rss():
    """Normalize the process high-water mark to bytes on supported platforms."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    unit = "bytes" if platform.system() == "Darwin" else "KiB"
    return {"bytes": int(raw if unit == "bytes" else raw * 1024), "raw": raw, "unit": unit}


def package_versions():
    versions = {"python": platform.python_version()}
    for name in ("bambi", "pymc", "pytensor", "formulae", "numpy", "pandas", "scipy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main():
    args = parse_args()
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"

    started = perf_counter()
    # Imports must follow the thread limits so BLAS pools start with one worker.
    import bambi as bmb  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel
    import pandas as pd  # pylint: disable=import-outside-toplevel

    import_seconds = perf_counter() - started
    bmb.config.SPARSE_DOT = args.sparse
    conditions = args.conditions if args.interaction_depth >= 2 else 1
    contexts = args.contexts if args.interaction_depth == 3 else 1
    started = perf_counter()
    rows = itertools.product(
        range(args.groups),
        range(conditions),
        range(contexts),
        range(args.times),
        range(args.replicates),
    )
    data = pd.DataFrame(rows, columns=["subject", "condition", "context", "year", "replicate"])
    rng = np.random.default_rng(sum(map(ord, "group-covariance-construction-benchmark")))
    data["x"] = rng.normal(size=len(data))
    data["outcome"] = np.arange(len(data)) % 2
    for name in ("subject", "condition", "context"):
        data[name] = pd.Categorical(data[name])
    grouping_factors = ["subject"]
    block_group_counts = [args.groups]
    if args.interaction_depth >= 2:
        grouping_factors.append("subject:condition")
        block_group_counts.append(args.groups * conditions)
    if args.interaction_depth == 3:
        grouping_factors.extend(["subject:context", "subject:condition:context"])
        block_group_counts.extend([args.groups * contexts, args.groups * conditions * contexts])
    formula = "outcome ~ x + " + " + ".join(
        f"ar1(0 + year | {factor})" for factor in grouping_factors
    )
    data_seconds = perf_counter() - started

    started = perf_counter()
    model = bmb.Model(formula, data, family="bernoulli")
    construction_seconds = perf_counter() - started
    rss_after_construction = peak_rss()
    started = perf_counter()
    model.build()
    build_seconds = perf_counter() - started
    rss_after_build = peak_rss()
    compile_seconds = None
    if args.compile_logp:
        started = perf_counter()
        model.backend.model.compile_logp()
        compile_seconds = perf_counter() - started

    preparation_seconds = None
    prediction_seconds = None
    prediction_rows = None
    if args.predict:
        started = perf_counter()
        import xarray as xr  # pylint: disable=import-outside-toplevel

        prior = model.prior_predictive(draws=4, random_seed=2026090811, omit_offsets=False)
        inference = xr.DataTree.from_dict({"posterior": prior["prior"].to_dataset()})
        target = data.copy()
        target["year"] += args.times
        preparation_seconds = perf_counter() - started
        started = perf_counter()
        model.predict(inference, data=target, random_seed=2026090812)
        prediction_seconds = perf_counter() - started
        prediction_rows = len(target)

    report = {
        "schema_version": 1,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "formula": formula,
        "row_count": len(data),
        "coefficient_count": args.times * sum(block_group_counts),
        "blocks": [
            {"grouping": factor, "group_count": count, "coefficient_count": count * args.times}
            for factor, count in zip(grouping_factors, block_group_counts)
        ],
        "wall_seconds": {
            "imports": import_seconds,
            "data": data_seconds,
            "construction": construction_seconds,
            "build": build_seconds,
            "compile_logp": compile_seconds,
            "prior_prediction_preparation": preparation_seconds,
            "future_prediction": prediction_seconds,
        },
        "peak_rss": peak_rss(),
        "peak_rss_after_construction": rss_after_construction,
        "peak_rss_after_build": rss_after_build,
        "versions": package_versions(),
        "platform": platform.platform(),
        "bambi_source": bmb.__file__,
        "thread_environment": {name: os.environ[name] for name in THREAD_VARIABLES},
        "pytensor_flags": os.environ.get("PYTENSOR_FLAGS", ""),
        "sampling_performed": args.predict,
        "mcmc_performed": False,
        "prior_forward_draws": 4 if args.predict else 0,
        "prediction_row_count": prediction_rows,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        sys.exit(1)
