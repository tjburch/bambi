"""Small post-processing checks; no model fitting or compilation."""

import importlib.util
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def load_helper(name):
    path = Path(__file__).parents[1] / "docs/examples/group_covariance_validation" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checks = load_helper("posterior_checks")
metrics = load_helper("reference_metrics")


def test_reference_csv_requires_roundtrip_float_parser():
    value = "-0.00352339790389752"
    parsed = pd.read_csv(StringIO(f"x1\n{value}\n"), float_precision="round_trip")
    assert parsed.x1.iloc[0] == float(value)


def test_hyperprior_selection_excludes_latent_offsets():
    assert checks.hyperprior_terms("ar1", ["block"]) == [("block_sd", "sd"), ("block_rho", "rho")]
    with pytest.raises(ValueError, match="adapter"):
        checks.hyperprior_terms("unsupported", ["block"])


def test_hyperprior_density_uses_natural_correlation_coordinates():
    rho = 0.6
    chol = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho**2)]])
    posterior = xr.Dataset(
        {
            "block_sd": (("chain", "draw", "coefficient"), [[[1.0, 2.0]]]),
            "block_corr_cholesky": (("chain", "draw", "row", "column"), chol[None, None]),
            "block_offset": (("chain", "draw"), [[1e6]]),
        }
    )
    actual = checks.hyperprior_log_density(posterior, "us-slopes", ["block"])
    expected = -0.5 * (1 + 4) / 2.5**2 + np.log(1 - rho**2)
    np.testing.assert_allclose(actual, expected)


def test_predictive_statistics_weight_trials_and_reject_probabilities():
    data = pd.DataFrame(
        {
            "subject": ["s1", "s1"],
            "condition": ["a", "a"],
            "context": ["c", "c"],
            "year": [0, 1],
            "trials": [1, 3],
            "y": [1, 0],
        }
    )
    predictive = xr.DataArray([[[1, 3], [0, 1]]], dims=("chain", "draw", "obs"))
    result, observed = checks.predictive_statistics(predictive, data)
    np.testing.assert_allclose(result["rate:overall"], [[1.0, 0.25]])
    assert observed["rate:overall"] == 0.25
    with pytest.raises(ValueError, match="counts"):
        checks.predictive_statistics(predictive / 2, data)


def test_constant_predictive_metrics_have_zero_mcse():
    dataset = xr.Dataset({"predictive_mean.1": (("chain", "draw"), np.zeros((4, 20)))})
    summary = metrics.summarize_metrics(dataset)["predictive_mean.1"]
    assert summary["mcse_mean"] == 0
    assert all(value == {"value": 0.0, "mcse": 0.0} for value in summary["quantiles"].values())


def test_nonconstant_metrics_use_chain_aware_quantile_mcse():
    values = np.random.default_rng(20260908).normal(size=(4, 100))
    dataset = xr.Dataset({"beta.one": (("chain", "draw"), values)})
    summary = metrics.summarize_metrics(dataset)["beta.one"]
    assert summary["mcse_mean"] > 0
    np.testing.assert_allclose(summary["mean"], values.mean())
    for probability, result in summary["quantiles"].items():
        np.testing.assert_allclose(result["value"], np.quantile(values, float(probability)))
        assert np.isfinite(result["mcse"]) and result["mcse"] > 0


def test_diagnostics_exclude_only_constant_derived_metrics(monkeypatch):
    import arviz_stats as azs

    captured = {}

    def summary(dataset, **kwargs):
        captured.update(dataset.data_vars)
        return pd.DataFrame({"r_hat": [1.0], "ess_bulk": [500], "ess_tail": [500]})

    monkeypatch.setattr(azs, "summary", summary)
    zeros = xr.DataArray(np.zeros((4, 20)), dims=("chain", "draw"))
    posterior = xr.Dataset({"block_offset": zeros})
    samples = xr.Dataset(
        {
            "energy": (("chain", "draw"), np.tile(np.arange(20), (4, 1))),
            "diverging": zeros,
            "reached_max_treedepth": zeros,
        }
    )
    inference = xr.DataTree.from_dict({"posterior": posterior, "sample_stats": samples})
    metrics.diagnose_scalar_metrics(
        inference, xr.Dataset({"sd.1": zeros, "log_likelihood.1": zeros})
    )
    assert "log_likelihood.1" not in captured
    assert "sd.1" in captured
    assert "block_offset" in captured
