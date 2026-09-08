"""The reference comparator rejects incomplete and statistically mismatched evidence."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "docs/examples/group_covariance_validation"
SPEC = importlib.util.spec_from_file_location(
    "compare_summaries", DIRECTORY / "compare_summaries.py"
)
COMPARATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARATOR)


def summary(engine="bambi"):
    names = {"beta.one", "beta.x1", "beta.x2", "probability.1"}
    names |= {f"{parameter}.{block}" for parameter in ("sd", "rho") for block in range(1, 5)}
    names |= {f"latent.b{block}cell1" for block in range(1, 5)}
    names |= {
        f"{prefix}.1"
        for prefix in (
            "log_likelihood",
            "predictive_mean",
            "predictive_second_moment",
            "predictive_zero_probability",
        )
    }
    return {
        "schema_version": 2,
        "phase": "posterior",
        "engine": engine,
        "mode": "ar1",
        "data_md5": "a" * 32,
        "identity": {
            "source_commit": "a" * 40,
            "source_sha256": "a" * 64,
            "data_sha256": "b" * 64,
            "data_md5": "a" * 32,
            "mode": "ar1",
            "design": {"response": [1], "row_groups": [[["s1"]] * 4], "row_times": [0]},
            "priors": {"scale_meaning": "marginal"},
        },
        "diagnostics": {
            "chains": 4,
            "rhat_max": 1.001,
            "ess_bulk_min": 800,
            "ess_tail_min": 700,
            "divergences": 0,
            "bfmi_min": 0.8,
            "treedepth_hits": 0,
        },
        "metrics": {
            name: {
                "mean": 0.5,
                "mcse_mean": 0.01,
                "quantiles": {q: {"value": float(q), "mcse": 0.01} for q in COMPARATOR.QUANTILES},
            }
            for name in names
        },
    }


def test_complete_reference_comparison():
    assert not COMPARATOR.compare(summary(), summary("stan"))["failures"]


def test_known_correlations_must_match_prior_contract():
    candidate = summary()
    candidate["mode"] = candidate["identity"]["mode"] = "known"
    candidate["fixed_rho"] = [0.2] * 4
    candidate["identity"]["priors"]["fixed_rho"] = [0.3] * 4
    with pytest.raises(ValueError, match="prior contract"):
        COMPARATOR.check_summary(candidate)


def test_us_visits_cannot_omit_a_whole_coefficient():
    candidate = summary()
    candidate["mode"] = candidate["identity"]["mode"] = "us-visits"
    candidate["identity"]["design"]["coefficient_columns"] = [0, 1, 3]
    template = deepcopy(candidate["metrics"]["sd.1"])
    candidate["metrics"] = {
        name: metric
        for name, metric in candidate["metrics"].items()
        if not name.startswith(("sd.", "rho.", "latent."))
    }
    for name in ("sd.1", "sd.2", "cor.1.2", "latent.subject.s1.1", "latent.subject.s1.2"):
        candidate["metrics"][name] = deepcopy(template)
    with pytest.raises(ValueError, match="metric set"):
        COMPARATOR.check_summary(candidate)


@pytest.mark.parametrize(
    "diagnostic,value",
    [
        ("chains", 3),
        ("chains", 4.5),
        ("rhat_max", 1.02),
        ("ess_bulk_min", 399),
        ("ess_tail_min", 399),
        ("divergences", 1),
        ("bfmi_min", 0.29),
        ("treedepth_hits", 1),
        ("rhat_max", float("nan")),
        ("chains", True),
    ],
)
def test_failing_diagnostic(diagnostic, value):
    candidate = summary()
    candidate["diagnostics"][diagnostic] = value
    with pytest.raises(ValueError):
        COMPARATOR.check_summary(candidate)


@pytest.mark.parametrize(
    "key", ["source_commit", "source_sha256", "data_sha256", "priors", "design"]
)
def test_identity_mismatch(key):
    first, second = summary(), summary("stan")
    if isinstance(second["identity"][key], str):
        second["identity"][key] = "f" * len(second["identity"][key])
    else:
        second["identity"][key]["changed"] = True
    with pytest.raises(ValueError):
        COMPARATOR.compare(first, second)


@pytest.mark.parametrize(
    "name", ["latent.b1cell1", "rho.4", "log_likelihood.1", "predictive_mean.1"]
)
def test_shared_missing_metric_is_not_a_pass(name):
    first, second = summary(), summary("stan")
    del first["metrics"][name]
    del second["metrics"][name]
    with pytest.raises(ValueError):
        COMPARATOR.compare(first, second)


def test_quantile_disagreement_detected_without_mean_difference():
    first, second = summary(), summary("stan")
    second["metrics"]["sd.1"]["quantiles"]["0.97"]["value"] = 1.3
    failures = COMPARATOR.compare(first, second)["failures"]
    assert [failure["statistic"] for failure in failures] == ["quantile.0.97"]


def test_negative_mcse_and_missing_quantile_rejected():
    candidate = summary()
    candidate["metrics"]["sd.1"]["mcse_mean"] = -0.1
    with pytest.raises(ValueError):
        COMPARATOR.check_summary(candidate)
    candidate = summary()
    del candidate["metrics"]["sd.1"]["quantiles"]["0.03"]
    with pytest.raises(ValueError):
        COMPARATOR.check_summary(candidate)


def test_deterministic_metrics_allow_zero_mcse():
    first, second = summary(), summary("stan")
    metric = {
        "mean": 0,
        "mcse_mean": 0,
        "quantiles": {q: {"value": 0, "mcse": 0} for q in COMPARATOR.QUANTILES},
    }
    first["metrics"]["predictive_mean.1"] = deepcopy(metric)
    second["metrics"]["predictive_mean.1"] = deepcopy(metric)
    assert not COMPARATOR.compare(first, second)["failures"]
    second["metrics"]["predictive_mean.1"]["mean"] = 0.1
    assert COMPARATOR.compare(first, second)["failures"]


def test_legacy_and_same_engine_results_rejected():
    with pytest.raises(ValueError):
        COMPARATOR.compare(summary(), summary())
    candidate = summary()
    candidate["schema_version"] = 1
    with pytest.raises(ValueError):
        COMPARATOR.check_summary(candidate)
