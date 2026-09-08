"""Restartable, fail-closed simulation-based calibration for covariance models.

Each case runs in a separate process. A failed case is never overwritten or silently
removed from the campaign denominator. Rank selection is conservative thinning,
not a claim that finite MCMC output is exactly independent.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import traceback

from single_block_reference import KINDS


def runtime_identity():
    """Record installed distributions without importing numerical libraries."""
    return dict(
        python=platform.python_version(),
        packages=dict(
            sorted(
                (distribution.metadata["Name"].lower().replace("_", "-"), distribution.version)
                for distribution in importlib.metadata.distributions()
            )
        ),
    )


def case_specifications(phase):
    if phase not in {"pilot", "full"}:
        raise ValueError("phase must be pilot or full")
    per_family = 5 if phase == "pilot" else 50
    cases = []
    for kind in KINDS:
        for family in ("bernoulli", "binomial"):
            for replicate in range(per_family):
                name = f"{kind}-{family}-{replicate:03d}"
                seed = (
                    int.from_bytes(
                        hashlib.sha256(f"covariance-sbc-v1:{phase}:{name}".encode()).digest()[:4],
                        "big",
                    )
                    % (2**30 - 1)
                    + 1
                )
                cases.append(dict(id=name, kind=kind, family=family, seed=seed))
    return cases


def campaign_digest(manifest):
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def initialize(output, phase, commit, warmup=1000, draws=1000):
    from validation_identity import source_identity

    source = source_identity()
    if source["source_commit"] != commit:
        raise ValueError("Requested commit does not match the checkout")
    if min(warmup, draws) < 1000:
        raise ValueError("Campaign warmup and retained draws must each be at least 1000")
    manifest = dict(
        schema_version=1,
        phase=phase,
        source=source,
        runtime=runtime_identity(),
        settings=dict(
            chains=4,
            warmup=warmup,
            draws=draws,
            rank_draws=100,
            target_accept=0.95,
            min_thinned_ess_ratio=0.8,
        ),
        cases=case_specifications(phase),
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


def validate_diagnostics(values):
    checks = {
        "chains": lambda x: x >= 4,
        "rhat_max": lambda x: x <= 1.01,
        "ess_bulk_min": lambda x: x >= 400,
        "ess_tail_min": lambda x: x >= 400,
        "divergences": lambda x: x == 0,
        "bfmi_min": lambda x: x >= 0.3,
        "treedepth_hits": lambda x: x == 0,
    }
    for name, check in checks.items():
        value = values.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or not check(value):
            raise ValueError(f"Uncertified sampler diagnostic: {name}")
        if name in {"chains", "divergences", "treedepth_hits"} and value != int(value):
            raise ValueError(f"Noninteger sampler diagnostic: {name}")


def randomized_rank(truth, draws, rng):
    """Randomize exact ties to preserve discrete ranks, including saturated probabilities."""
    if not math.isfinite(truth) or not draws or not all(math.isfinite(x) for x in draws):
        raise ValueError("Ranks require finite truth and nonempty finite draws")
    below = sum(value < truth for value in draws)
    ties = sum(value == truth for value in draws)
    return below + int(rng.integers(ties + 1))


def select_rank_draws(dataset, count, seed, minimum_ratio=0.8):
    """Space draws within chains using the worst bulk/tail ESS, then check retained ESS."""
    import arviz_stats as azs
    import numpy as np

    if count < 4 or count % dataset.sizes["chain"]:
        raise ValueError("Rank draw count must be divisible by the number of chains")
    summary = azs.summary(dataset, kind="diagnostics", round_to="none")
    ess = min(float(summary.ess_bulk.min(skipna=False)), float(summary.ess_tail.min(skipna=False)))
    total = dataset.sizes["chain"] * dataset.sizes["draw"]
    if not math.isfinite(ess) or ess <= 0:
        raise ValueError("Nonfinite or zero ESS")
    stride = max(1, math.ceil(2 * total / ess))
    per_chain = count // dataset.sizes["chain"]
    if stride * per_chain > dataset.sizes["draw"]:
        raise ValueError(
            "Too few effective draws for the frozen rank count; keep failure and revise campaign"
        )
    rng = np.random.default_rng(seed)
    start = int(rng.integers(dataset.sizes["draw"] - stride * (per_chain - 1)))
    selected = dataset.isel(draw=start + stride * np.arange(per_chain))
    check = azs.summary(selected, kind="diagnostics", round_to="none")
    retained_ess = min(
        float(check.ess_bulk.min(skipna=False)), float(check.ess_tail.min(skipna=False))
    )
    if not math.isfinite(retained_ess) or retained_ess < minimum_ratio * count:
        raise ValueError("Retained rank draws remain too autocorrelated")
    return selected, dict(
        stride=stride,
        start=start,
        retained_ess_min=retained_ess,
        original_ess_min=ess,
        rank_draws=count,
    )


def export_ranks(fixture, fit, settings, seed):
    import numpy as np
    import xarray as xr

    truth = json.loads((fixture / "truth.json").read_text())
    summary = json.loads((fit / "summary.json").read_text())
    validate_diagnostics(summary["diagnostics"])
    with xr.open_dataset(fit / "metrics.nc") as dataset:
        if not set(truth) <= set(dataset.data_vars):
            raise ValueError("Truth metrics missing from posterior draws")
        selected, thinning = select_rank_draws(
            dataset[list(truth)], settings["rank_draws"], seed, settings["min_thinned_ess_ratio"]
        )
        rng = np.random.default_rng(seed)
        ranks = {}
        for name, value in truth.items():
            draws = np.asarray(selected[name]).reshape(-1).tolist()
            all_draws = np.asarray(dataset[name]).reshape(-1)
            lower, upper = np.quantile(all_draws, [0.03, 0.97])
            ranks[name] = dict(
                rank=randomized_rank(value, draws, rng),
                covered=bool(lower <= value <= upper),
                truth=value,
            )
    return dict(metrics=ranks, thinning=thinning, diagnostics=summary["diagnostics"])


def run_case(manifest_path, case_id):
    from validation_identity import source_identity

    manifest = json.loads(manifest_path.read_text())
    if source_identity() != manifest["source"]:
        raise ValueError("Campaign source differs from this checkout")
    if runtime_identity() != manifest["runtime"]:
        raise ValueError("Campaign Python distribution versions differ from this environment")
    expected = case_specifications(manifest["phase"])
    if manifest["cases"] != expected:
        raise ValueError("Campaign cases differ from the frozen specification")
    cases = [case for case in expected if case["id"] == case_id]
    if len(cases) != 1:
        raise ValueError("Unknown or duplicate case")
    case = cases[0]
    output = manifest_path.parent / "cases" / case_id
    digest = campaign_digest(manifest)
    status_path = output / "status.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous["campaign_sha256"] != digest or previous["case"] != case:
            raise ValueError("Stale case cannot be resumed")
        if previous["status"] == "complete":
            if (
                previous.get("ranks_sha256")
                != hashlib.sha256((output / "ranks.json").read_bytes()).hexdigest()
            ):
                raise ValueError("Completed rank artifact is missing or changed")
            return
        raise ValueError("Case previously failed or was interrupted; retained for diagnosis")
    output.mkdir(parents=True, exist_ok=False)
    status = dict(case=case, campaign_sha256=digest, status="running")
    status_path.write_text(json.dumps(status, indent=2))
    script = str(Path(__file__).with_name("single_block_reference.py"))
    settings = manifest["settings"]
    try:
        with (output / "run.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    script,
                    "generate",
                    case["kind"],
                    case["family"],
                    str(output / "fixture"),
                    "--seed",
                    str(case["seed"]),
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            subprocess.run(
                [
                    sys.executable,
                    script,
                    "fit",
                    str(output / "fixture"),
                    str(output / "fit"),
                    "--engine",
                    "bambi",
                    "--seed",
                    str(case["seed"] + 1),
                    "--chains",
                    str(settings["chains"]),
                    "--warmup",
                    str(settings["warmup"]),
                    "--draws",
                    str(settings["draws"]),
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        result = export_ranks(output / "fixture", output / "fit", settings, case["seed"] + 2)
        (output / "ranks.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        status["ranks_sha256"] = hashlib.sha256((output / "ranks.json").read_bytes()).hexdigest()
        status["status"] = "complete"
    except Exception:
        status["status"] = "failed"
        status["error"] = traceback.format_exc()
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2, allow_nan=False))


def wilson_interval(successes, total, z=1.959963984540054):
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Invalid coverage counts")
    rate = successes / total
    center = (rate + z * z / (2 * total)) / (1 + z * z / total)
    half = (
        z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    )
    return [max(0, center - half), min(1, center + half)]


def rank_check(ranks, count, tests, alpha=0.05):
    """A conservative simultaneous DKW envelope for the discrete-uniform rank CDF."""
    if not ranks or count < 1 or tests < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid rank-check settings")
    if any(type(rank) is not int or not 0 <= rank <= count for rank in ranks):
        raise ValueError("Invalid rank")
    epsilon = math.sqrt(math.log(2 * tests / alpha) / (2 * len(ranks)))
    deviation = max(
        abs(sum(rank <= k for rank in ranks) / len(ranks) - (k + 1) / (count + 1))
        for k in range(count + 1)
    )
    return dict(
        n=len(ranks),
        max_cdf_deviation=deviation,
        simultaneous_envelope=epsilon,
        passed=deviation <= epsilon,
    )


def aggregate(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if manifest["cases"] != case_specifications(manifest["phase"]):
        raise ValueError("Campaign case specification was changed")
    digest = campaign_digest(manifest)
    results = {}
    failures = []
    for case in manifest["cases"]:
        output = manifest_path.parent / "cases" / case["id"]
        try:
            status = json.loads((output / "status.json").read_text())
            if (
                status["campaign_sha256"] != digest
                or status["case"] != case
                or status["status"] != "complete"
            ):
                raise ValueError("Missing, failed, interrupted or stale case")
            result = json.loads((output / "ranks.json").read_text())
            if (
                status.get("ranks_sha256")
                != hashlib.sha256((output / "ranks.json").read_bytes()).hexdigest()
            ):
                raise ValueError("Rank artifact changed after completion")
            validate_diagnostics(result["diagnostics"])
            if result["thinning"]["rank_draws"] != manifest["settings"]["rank_draws"]:
                raise ValueError("Rank count differs from campaign")
            truth = json.loads((output / "fixture" / "truth.json").read_text())
            if set(result["metrics"]) != set(truth):
                raise ValueError("Missing or extra calibration metrics")
            for metric, entry in result["metrics"].items():
                if entry["truth"] != truth[metric] or type(entry["covered"]) is not bool:
                    raise ValueError("Invalid truth or coverage record")
                if (
                    type(entry["rank"]) is not int
                    or not 0 <= entry["rank"] <= manifest["settings"]["rank_draws"]
                ):
                    raise ValueError("Invalid calibration rank")
            group = (case["kind"], case["family"])
            if group in results and set(results[group][0]["metrics"]) != set(result["metrics"]):
                raise ValueError("Metric set changed within campaign")
            results.setdefault(group, []).append(result)
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(dict(case=case["id"], error=str(error)))
    tests = sum(len(values[0]["metrics"]) for values in results.values())
    checks = {}
    for group, values in results.items():
        for metric in values[0]["metrics"]:
            entries = [value["metrics"][metric] for value in values]
            check = rank_check(
                [entry["rank"] for entry in entries], manifest["settings"]["rank_draws"], tests
            )
            successes = sum(entry["covered"] for entry in entries)
            check.update(
                coverage_94=successes / len(entries),
                coverage_95_interval=wilson_interval(successes, len(entries)),
            )
            checks["/".join((*group, metric))] = check
    return dict(
        schema_version=1,
        campaign_sha256=digest,
        phase=manifest["phase"],
        expected=len(manifest["cases"]),
        completed=sum(len(values) for values in results.values()),
        failures=failures,
        checks=checks,
        passed=not failures and bool(checks) and all(value["passed"] for value in checks.values()),
        limitation="Finite thinned MCMC and 100 replicates per structure give limited calibration evidence; pilot is not a merge gate.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("output", type=Path)
    init.add_argument("--phase", choices=("pilot", "full"), required=True)
    init.add_argument("--commit", required=True)
    init.add_argument("--warmup", type=int, default=1000)
    init.add_argument("--draws", type=int, default=1000)
    run = sub.add_parser("run")
    run.add_argument("manifest", type=Path)
    run.add_argument("--case", required=True)
    check = sub.add_parser("check")
    check.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.command == "init":
        initialize(args.output, args.phase, args.commit, args.warmup, args.draws)
    elif args.command == "run":
        run_case(args.manifest, args.case)
    else:
        result = aggregate(args.manifest)
        print(json.dumps(result, indent=2, allow_nan=False))
        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
