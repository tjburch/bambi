"""Guard against false acceptance from mismatched or insufficient evidence."""

import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from bambi.examples import structured_covariance_validation as validation


def test_comparison_rejects_different_data(tmp_path):
    configuration = dict(
        seed=10, structure="ar1", groups=2, prior_draw=False, prior_scale=1, composition=False
    )
    for method in ("latent", "marginal"):
        directory = tmp_path / method
        directory.mkdir()
        (directory / "command.json").write_text(
            json.dumps({"configuration": {**configuration, "method": method}})
        )
        (directory / "data.csv").write_text(method)
    with pytest.raises(ValueError, match="inputs differ: data.csv"):
        validation.compare(
            Namespace(
                latent=tmp_path / "latent",
                marginal=tmp_path / "marginal",
                output=tmp_path / "comparison",
            )
        )


def test_calibration_refuses_fixed_truth_runs(tmp_path):
    directory = tmp_path / "replicate-1" / "marginal"
    directory.mkdir(parents=True)
    (directory / "result.json").write_text('{"passed": true}')
    (directory / "command.json").write_text(json.dumps({"configuration": {"prior_draw": False}}))
    with pytest.raises(ValueError, match="independent prior draws"):
        validation.calibration(
            Namespace(input=tmp_path, output=tmp_path / "assessment", replicates=100)
        )


def test_calibration_cannot_pass_without_replications(tmp_path):
    with pytest.raises(SystemExit, match="lacks enough"):
        validation.calibration(
            Namespace(input=tmp_path, output=tmp_path / "assessment", replicates=100)
        )
    assert not json.loads((tmp_path / "assessment" / "calibration.json").read_text())["passed"]


def test_convergence_requires_tail_ess(tmp_path, monkeypatch):
    seed = sum(map(ord, "structured-validation-tail-gate"))
    variables = xr.Dataset(
        {"effect": (("chain", "draw"), np.random.default_rng(seed).normal(size=(4, 100)))}
    )
    statistics = xr.Dataset(
        {
            "energy": variables.effect,
            "diverging": xr.zeros_like(variables.effect, dtype=bool),
            "reached_max_treedepth": xr.zeros_like(variables.effect, dtype=bool),
        }
    )
    idata = xr.DataTree.from_dict({"posterior": variables, "sample_stats": statistics})
    monkeypatch.setattr(
        validation.azs,
        "diagnose",
        lambda *args, **kwargs: (False, {"bfmi": {"bfmi_values": np.ones(4)}}),
    )
    monkeypatch.setattr(validation.azs, "summary", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(validation.azs, "rhat", lambda *args, **kwargs: xr.Dataset({"effect": 1.0}))
    monkeypatch.setattr(
        validation.azs,
        "ess",
        lambda *args, method="bulk", **kwargs: xr.Dataset(
            {"effect": 1000 if method == "bulk" else 20}
        ),
    )
    assert not validation.diagnose(idata, tmp_path)["passed"]


def test_diagnostic_statistics_checks_jax_depth_limit():
    statistics = xr.Dataset(
        {
            "energy": ("draw", [1.0, 2.0, 3.0]),
            "diverging": ("draw", [False, False, False]),
            "tree_depth": ("draw", [9, 10, 11]),
        }
    )
    result = validation.diagnostic_statistics(xr.DataTree.from_dict({"sample_stats": statistics}))
    np.testing.assert_array_equal(result.reached_max_treedepth, [False, True, True])


@pytest.mark.parametrize("flag", ["reached_max_treedepth", "maxdepth_reached"])
def test_diagnostic_statistics_preserves_native_flags(flag):
    statistics = xr.Dataset(
        {
            "energy": ("draw", [1.0]),
            "diverging": ("draw", [False]),
            "tree_depth": ("draw", [10]),
            flag: ("draw", [False]),
        }
    )
    result = validation.diagnostic_statistics(xr.DataTree.from_dict({"sample_stats": statistics}))
    assert not bool(result.reached_max_treedepth.any())


@pytest.mark.parametrize("missing", ["energy", "diverging", "tree_depth"])
def test_diagnostic_statistics_requires_sampler_evidence(missing):
    statistics = xr.Dataset(
        {"energy": ("draw", [1.0]), "diverging": ("draw", [False]), "tree_depth": ("draw", [2])}
    ).drop_vars(missing)
    with pytest.raises(ValueError, match="lacks required diagnostics"):
        validation.diagnostic_statistics(xr.DataTree.from_dict({"sample_stats": statistics}))


@pytest.mark.parametrize("invalid", ["energy", "diverging", "tree_depth", "reached_max_treedepth"])
def test_diagnostic_statistics_rejects_nonfinite_evidence(invalid):
    statistics = xr.Dataset(
        {"energy": ("draw", [1.0]), "diverging": ("draw", [False]), "tree_depth": ("draw", [2])}
    )
    statistics[invalid] = xr.DataArray([np.nan], dims="draw")
    with pytest.raises(ValueError, match="nonfinite"):
        validation.diagnostic_statistics(xr.DataTree.from_dict({"sample_stats": statistics}))


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_convergence_requires_finite_bfmi(tmp_path, monkeypatch, invalid):
    variables = xr.Dataset({"effect": (("chain", "draw"), np.arange(400).reshape(4, 100))})
    statistics = xr.Dataset(
        {
            "energy": variables.effect,
            "diverging": xr.zeros_like(variables.effect, dtype=bool),
            "reached_max_treedepth": xr.zeros_like(variables.effect, dtype=bool),
        }
    )
    idata = xr.DataTree.from_dict({"posterior": variables, "sample_stats": statistics})
    monkeypatch.setattr(
        validation.azs,
        "diagnose",
        lambda *args, **kwargs: (False, {"bfmi": {"bfmi_values": [invalid, 1, 1, 1]}}),
    )
    monkeypatch.setattr(validation.azs, "summary", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(validation.azs, "rhat", lambda *args, **kwargs: xr.Dataset({"effect": 1.0}))
    monkeypatch.setattr(validation.azs, "ess", lambda *args, **kwargs: xr.Dataset({"effect": 1000}))
    assert not validation.diagnose(idata, tmp_path)["passed"]


def test_monitored_excludes_fixed_values_without_roundoff():
    posterior = xr.Dataset(
        {
            "effect": (("chain", "draw"), np.arange(12).reshape(2, 6)),
            "fixed_sd": (("chain", "draw"), np.full((2, 6), 0.1)),
        }
    )
    idata = xr.DataTree.from_dict({"posterior": posterior})
    assert set(validation.monitored(idata)) == {"effect"}


@pytest.mark.parametrize(
    ("composition", "missing"),
    [(False, "Intercept"), (False, "x"), (False, "sigma"), (True, "sigma_z")],
)
def test_comparison_requires_population_parameters(tmp_path, monkeypatch, composition, missing):
    names = {
        "Intercept",
        "x",
        "sigma",
        "effect",
        "effect_sd",
        "sigma_Intercept",
        "sigma_z",
        "1|team_sigma",
        "1|team",
        "diag(0 + x, by=team)",
        "sigma_diag(1, by=team)",
    }
    values = np.arange(20).reshape(2, 10)
    posterior = xr.Dataset({name: (("chain", "draw"), values) for name in names - {missing}})
    idata = xr.DataTree.from_dict({"posterior": posterior})
    monkeypatch.setattr(validation.xr, "open_datatree", lambda *args, **kwargs: idata)
    (tmp_path / "command.json").write_text(
        json.dumps({"configuration": {"composition": composition, "structure": "diag"}})
    )
    with pytest.raises(ValueError, match="lacks expected stochastic variables"):
        validation.comparison_quantities(tmp_path)


@pytest.mark.parametrize(
    ("structure", "name"),
    [
        ("ar1", "effect_rho"),
        ("ou", "effect_decay"),
        ("cs", "effect_rho"),
        ("cs_homogeneous", "effect_rho"),
        ("us", "correlation"),
        ("us3", "correlation"),
    ],
)
def test_comparison_rejects_constant_covariance_parameter(tmp_path, monkeypatch, structure, name):
    values = np.arange(20).reshape(2, 10)
    posterior = xr.Dataset(
        {
            key: (("chain", "draw"), values)
            for key in ("Intercept", "x", "sigma", "effect", "effect_sd")
        }
    )
    if name == "correlation":
        size = 3 if structure == "us3" else 2
        posterior["effect_correlation"] = xr.DataArray(
            np.broadcast_to(np.eye(size), (2, 10, size, size)),
            dims=("chain", "draw", "row", "column"),
        )
    else:
        posterior[name] = xr.zeros_like(posterior.effect_sd)
    idata = xr.DataTree.from_dict({"posterior": posterior})
    monkeypatch.setattr(validation.xr, "open_datatree", lambda *args, **kwargs: idata)
    (tmp_path / "command.json").write_text(
        json.dumps({"configuration": {"composition": False, "structure": structure}})
    )
    with pytest.raises(ValueError, match=f"lacks expected stochastic variables: .*{name}"):
        validation.comparison_quantities(tmp_path)


@pytest.mark.parametrize("defect", ["missing", "constant", "scalar_count"])
def test_calibration_requires_all_simulated_quantities(tmp_path, monkeypatch, defect):
    directory = tmp_path / "replicate-1" / "marginal"
    directory.mkdir(parents=True)
    configuration = dict(
        seed=100,
        structure="ou",
        groups=1,
        prior_draw=True,
        prior_scale=1,
        composition=False,
        method="marginal",
    )
    truth = dict(Intercept=0, x=0.5, sigma=0.3, effect_sd=[0.6], effect_decay=0.4)
    (directory / "result.json").write_text('{"passed": true}')
    (directory / "command.json").write_text(json.dumps({"configuration": configuration}))
    (directory / "truth.json").write_text(json.dumps({"parameters": truth, "effects": [0, 1]}))
    values = np.arange(40).reshape(2, 20)
    posterior = xr.Dataset({name: (("chain", "draw"), values) for name in truth})
    posterior["effect"] = xr.DataArray(
        np.stack([values, values], axis=-1), dims=("chain", "draw", "state")
    )
    if defect == "missing":
        posterior = posterior.drop_vars("effect_decay")
    elif defect == "constant":
        posterior["effect_decay"] = xr.zeros_like(posterior.effect_decay)
    else:
        posterior["effect_sd"] = xr.DataArray(
            np.stack([values, values], axis=-1), dims=("chain", "draw", "coefficient")
        )
    idata = xr.DataTree.from_dict({"posterior": posterior})
    monkeypatch.setattr(validation.xr, "open_datatree", lambda *args, **kwargs: idata)
    monkeypatch.setattr(validation.azs, "ess", lambda *args, **kwargs: 1000)
    monkeypatch.setattr(validation.azs, "hdi", lambda *args, **kwargs: [-1, 1])
    with pytest.raises(ValueError, match="Calibration (lacks expected|has incorrect scalar count)"):
        validation.calibration(
            Namespace(input=tmp_path, output=tmp_path / "assessment", replicates=100)
        )


@pytest.mark.parametrize("structure", ["ar1", "ou", "diag", "cs", "cs_homogeneous", "us"])
def test_prior_simulations_hold_out_new_time_or_slope(structure):
    data, _, _, holdout = validation.simulate(structure, 2, validation.SEED, prior_draw=True)
    assert holdout.g.iloc[0] == "g000"
    assert holdout.x.iloc[0] == 1.5
    if structure in ("ar1", "ou"):
        assert holdout.time.iloc[0] not in data.time.to_numpy()
