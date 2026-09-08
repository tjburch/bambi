"""Compare matched posterior summaries after checking identity and diagnostics."""

import argparse
import json
import math
from pathlib import Path
import re

QUANTILES = {"0.03", "0.5", "0.97"}


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def block_metrics(blocks, mode):
    """Expected metrics from the independent fixture, not the exported intersection."""
    kinds = {"ar1", "ou", "cs", "toep", "us"}
    if mode == "four-block-ar1":
        if len(blocks) != 4 or any(block["kind"] != "ar1" for block in blocks):
            raise ValueError("Expected four AR1 blocks")
    elif mode not in kinds or len(blocks) != 1 or blocks[0]["kind"] != mode:
        raise ValueError("Invalid single-block design")
    expected = set()
    for number, block in enumerate(blocks, start=1):
        prefix = f"block.{number}." if len(blocks) > 1 else ""
        kind, size = block["kind"], len(block["times"])
        expected |= {f"{prefix}sd.{i}" for i in range(1, (size if kind == "us" else 1) + 1)}
        if kind in {"ar1", "cs"}:
            expected.add(f"{prefix}rho.1")
        elif kind == "ou":
            expected.add(f"{prefix}decay.1")
        elif kind == "toep":
            expected |= {f"{prefix}partial.{i}" for i in range(1, block["max_lag"] + 1)}
        else:
            expected |= {
                f"{prefix}cor.{i}.{j}" for i in range(1, size + 1) for j in range(i + 1, size + 1)
            }
        expected |= {
            f"{prefix}coefficient.{g}.{q}"
            for g in range(1, len(block["groups"]) + 1)
            for q in range(1, size + 1)
        }
    return expected


def check_summary(summary):
    if summary["schema_version"] != 2 or summary["phase"] != "posterior":
        raise ValueError("Only version-2 posterior summaries can be compared")
    identity = summary["identity"]
    for key, length in (("source_commit", 40), ("source_sha256", 64), ("data_sha256", 64)):
        if not isinstance(identity.get(key), str) or not re.fullmatch(
            f"[0-9a-f]{{{length}}}", identity[key]
        ):
            raise ValueError(f"Missing or invalid identity: {key}")
    if identity["mode"] != summary["mode"] or identity["data_md5"] != summary["data_md5"]:
        raise ValueError("Summary disagrees with its input identity")
    if not identity["design"] or not identity["priors"]:
        raise ValueError("Missing design or prior contract")
    diagnostics = summary["diagnostics"]
    limits = {
        "chains": lambda value: value >= 4,
        "rhat_max": lambda value: value <= 1.01,
        "ess_bulk_min": lambda value: value >= 400,
        "ess_tail_min": lambda value: value >= 400,
        "divergences": lambda value: value == 0,
        "bfmi_min": lambda value: value >= 0.3,
        "treedepth_hits": lambda value: value == 0,
    }
    for name, acceptable in limits.items():
        value = diagnostics[name]
        if not finite_number(value) or not acceptable(value):
            raise ValueError(f"Invalid or failing diagnostic: {name}={value}")
        if name in {"chains", "divergences", "treedepth_hits"} and value != int(value):
            raise ValueError(f"Noninteger diagnostic count: {name}")
    metrics = summary["metrics"]
    if not metrics or not {"beta.one", "beta.x1", "beta.x2"}.issubset(metrics):
        raise ValueError("Missing fixed-effect metrics")
    if not any(name.startswith("probability.") for name in metrics):
        raise ValueError("Missing fitted-probability metrics")
    probabilities = {name for name in metrics if name.startswith("probability.")}
    if probabilities != {f"probability.{index}" for index in range(1, len(probabilities) + 1)}:
        raise ValueError("Probability indices must be contiguous and one-based")
    design = identity["design"]
    if isinstance(design, list):
        expected = block_metrics(design, summary["mode"])
    elif summary["mode"] in {"ar1", "known"}:
        if summary["mode"] == "known":
            fixed = summary["fixed_rho"]
            if not isinstance(fixed, list) or len(fixed) != 4:
                raise ValueError("Known-covariance comparison requires four fixed correlations")
            if any(isinstance(value, bool) or not -1 < value < 1 for value in fixed):
                raise ValueError("Invalid fixed correlations")
            if fixed != identity["priors"].get("fixed_rho"):
                raise ValueError("Fixed correlations differ from prior contract")
        expected = {f"sd.{index}" for index in range(1, 5)}
        if summary["mode"] == "ar1":
            expected |= {f"rho.{index}" for index in range(1, 5)}
    elif summary["mode"] in {"us-slopes", "us-visits"}:
        count = len(design["coefficient_columns"])
        if count < 2 or (summary["mode"] == "us-slopes" and count != 2):
            raise ValueError("Missing unstructured coefficient scales")
        expected = {f"sd.{index}" for index in range(1, count + 1)}
        expected |= {f"cor.{i}.{j}" for i in range(1, count + 1) for j in range(i + 1, count + 1)}
    else:
        raise ValueError("Unsupported comparison mode")
    expected |= {"beta.one", "beta.x1", "beta.x2"} | probabilities
    row_count = len(design[0]["group_id"]) if isinstance(design, list) else len(design["response"])
    if len(probabilities) != row_count:
        raise ValueError("Fitted probability count differs from design")
    for prefix in (
        "log_likelihood",
        "predictive_mean",
        "predictive_second_moment",
        "predictive_zero_probability",
    ):
        expected |= {f"{prefix}.{index}" for index in range(1, row_count + 1)}
    if isinstance(design, list):
        pass
    elif summary["mode"] in {"ar1", "known"}:
        for block in range(4):
            cells = {
                (tuple(groups[block]), time)
                for groups, time in zip(design["row_groups"], design["row_times"], strict=True)
            }
            expected |= {f"latent.b{block + 1}cell{index}" for index in range(1, len(cells) + 1)}
    else:
        subjects = {groups[0][0] for groups in design["row_groups"]}
        expected |= {
            f"latent.subject.{subject}.{index}"
            for subject in subjects
            for index in range(1, count + 1)
        }
    if set(metrics) != expected:
        raise ValueError("Incomplete or unexpected parameter metric set")
    for name, metric in metrics.items():
        if not finite_number(metric["mean"]) or not finite_number(metric["mcse_mean"]):
            raise ValueError(f"Nonfinite metric: {name}")
        if metric["mcse_mean"] < 0:
            raise ValueError(f"Negative MCSE: {name}")
        if set(metric["quantiles"]) != QUANTILES:
            raise ValueError(f"Missing quantiles: {name}")
        previous = -math.inf
        for probability in sorted(QUANTILES, key=float):
            quantile = metric["quantiles"][probability]
            if not finite_number(quantile["value"]) or not finite_number(quantile["mcse"]):
                raise ValueError(f"Nonfinite quantile: {name}")
            if quantile["mcse"] < 0 or quantile["value"] < previous:
                raise ValueError(f"Invalid quantile: {name}")
            previous = quantile["value"]


def compare(left, right):
    """Return every discrepancy; never silently compare only shared metrics."""
    for summary in (left, right):
        check_summary(summary)
    for key in ("mode", "data_md5", "identity"):
        if not left[key] or left[key] != right[key]:
            raise ValueError(f"Reference inputs do not match: {key}")
    if left["mode"] == "known" and left["fixed_rho"] != right["fixed_rho"]:
        raise ValueError("Fixed correlation parameters differ between engines")
    if left["engine"] == right["engine"]:
        raise ValueError("Cross-engine validation requires different engines")
    if set(left["metrics"]) != set(right["metrics"]):
        raise ValueError("Metric sets differ; refusing an intersection-only comparison")
    failures = []
    for name in sorted(left["metrics"]):
        first, second = left["metrics"][name], right["metrics"][name]
        pairs = [("mean", first["mean"], second["mean"], first["mcse_mean"], second["mcse_mean"])]
        pairs.extend(
            (
                f"quantile.{q}",
                first["quantiles"][q]["value"],
                second["quantiles"][q]["value"],
                first["quantiles"][q]["mcse"],
                second["quantiles"][q]["mcse"],
            )
            for q in sorted(QUANTILES, key=float)
        )
        for statistic, value1, value2, error1, error2 in pairs:
            threshold = 4 * math.hypot(error1, error2)
            # Exact constants (for example zero trials) have no Monte Carlo error.
            if error1 == error2 == 0:
                threshold = 1e-12
            difference = abs(value1 - value2)
            if difference > threshold:
                failures.append(
                    {
                        "metric": name,
                        "statistic": statistic,
                        "difference": difference,
                        "threshold": threshold,
                    }
                )
    return {
        "compared_metrics": len(left["metrics"]),
        "statistics_per_metric": 4,
        "failures": failures,
        "scope": "Matched marginal summaries; joint prediction and SBC are separate gates",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    left, right = (json.loads(path.read_text()) for path in (args.left, args.right))
    result = compare(left, right)
    encoded = json.dumps(result, indent=2, allow_nan=False)
    print(encoded)
    if args.output:
        with args.output.open("x") as stream:
            stream.write(encoded + "\n")
    raise SystemExit(bool(result["failures"]))


if __name__ == "__main__":
    main()
