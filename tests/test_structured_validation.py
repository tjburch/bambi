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
    idata = xr.DataTree.from_dict({"posterior": variables, "sample_stats": variables})
    monkeypatch.setattr(validation.azs, "diagnose", lambda *args, **kwargs: (False, {}))
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


def test_monitored_excludes_fixed_values_without_roundoff():
    posterior = xr.Dataset(
        {
            "effect": (("chain", "draw"), np.arange(12).reshape(2, 6)),
            "fixed_sd": (("chain", "draw"), np.full((2, 6), 0.1)),
        }
    )
    idata = xr.DataTree.from_dict({"posterior": posterior})
    assert set(validation.monitored(idata)) == {"effect"}


@pytest.mark.parametrize("structure", ["ar1", "ou", "diag", "cs", "cs_homogeneous", "us"])
def test_prior_simulations_hold_out_new_time_or_slope(structure):
    data, _, _, holdout = validation.simulate(structure, 2, validation.SEED, prior_draw=True)
    assert holdout.g.iloc[0] == "g000"
    assert holdout.x.iloc[0] == 1.5
    if structure in ("ar1", "ou"):
        assert holdout.time.iloc[0] not in data.time.to_numpy()
