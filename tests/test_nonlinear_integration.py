import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

import bambi as bmb


@pytest.fixture
def linked_auxiliary_model():
    data = pd.DataFrame(
        {"x": [0.0, 0.5, 1.5, 2.0], "z": [-1.0, 0.0, 0.5, 1.0], "y": [0.2, 0.4, 0.7, 0.6]}
    )
    formula = bmb.Formula("y ~ a + b * exp(-x)", "a ~ 1", "b ~ 1", "kappa ~ 1 + z", nonlinear=True)
    model = bmb.Model(
        formula,
        data,
        family="beta",
        center_predictors=False,
        priors={
            name: {term: bmb.Prior("Normal", mu=0, sigma=1) for term in terms}
            for name, terms in {
                "a": ["Intercept"],
                "b": ["Intercept"],
                "kappa": ["Intercept", "z"],
            }.items()
        },
    )
    model.set_alias(
        {
            "y": "outcome",
            "mu": {"mu": "probability"},
            "a": {"a": "baseline", "Intercept": "a0"},
            "b": {"b": "amplitude", "Intercept": "b0"},
            "kappa": {"kappa": "precision", "Intercept": "p0", "z": "pz"},
        }
    )
    model.build()
    return model


@pytest.mark.parametrize("out_of_sample", [False, True])
def test_link_auxiliary_and_aliases_match_direct_pymc(linked_auxiliary_model, out_of_sample):
    model = linked_auxiliary_model
    data = model.data
    posterior = xr.Dataset(
        {
            name: (("chain", "draw"), [values])
            for name, values in {
                "a0": [-0.4, 0.2],
                "b0": [0.8, -0.3],
                "p0": [1.0, 1.5],
                "pz": [0.2, -0.1],
            }.items()
        }
    )
    with pm.Model(coords={"__obs__": range(len(data))}) as reference:
        x = pm.Data("x", data.x, dims="__obs__")
        z = pm.Data("z", data.z, dims="__obs__")
        y = pm.Data("y", data.y, dims="__obs__")
        a0 = pm.Normal("a0", mu=0, sigma=1)
        b0 = pm.Normal("b0", mu=0, sigma=1)
        p0 = pm.Normal("p0", mu=0, sigma=1)
        pz = pm.Normal("pz", mu=0, sigma=1)
        probability = pm.Deterministic(
            "probability", pm.math.sigmoid(a0 + b0 * pm.math.exp(-x)), dims="__obs__"
        )
        precision = pm.Deterministic("precision", pm.math.exp(p0 + pz * z), dims="__obs__")
        pm.Beta(
            "outcome",
            alpha=probability * precision,
            beta=(1 - probability) * precision,
            observed=y,
            dims="__obs__",
        )

    actual_logp = model.backend.model.compile_logp()
    expected_logp = reference.compile_logp()
    for draw in range(2):
        point = {name: value.values[0, draw] for name, value in posterior.items()}
        np.testing.assert_allclose(actual_logp(point), expected_logp(point))

    if out_of_sample:
        data = pd.DataFrame({"x": [0.2, 2.5], "z": [0.3, -0.7], "y": [0.4, 0.8]})
        pm.set_data(
            {"x": data.x, "z": data.z, "y": data.y},
            coords={"__obs__": range(len(data))},
            model=reference,
        )
    with reference:
        expected = pm.compute_deterministics(
            posterior, var_names=["probability", "precision"], progressbar=False
        )
        expected_likelihood = pm.compute_log_likelihood(
            xr.DataTree.from_dict({"posterior": posterior}), progressbar=False
        )

    idata = xr.DataTree.from_dict({"posterior": posterior})
    prediction_data = data.drop(columns="y") if out_of_sample else None
    predicted = model.predict(idata, data=prediction_data, inplace=False)
    actual = predicted.predictions if out_of_sample else predicted.posterior
    for name in ["probability", "precision"]:
        xr.testing.assert_allclose(actual[name], expected[name])
    likelihood = model.compute_log_likelihood(
        idata, data=data if out_of_sample else None, inplace=False
    )
    xr.testing.assert_allclose(
        likelihood.log_likelihood["outcome"], expected_likelihood.log_likelihood["outcome"]
    )
    assert {"baseline", "amplitude", "mu", "kappa"}.isdisjoint(actual.data_vars)


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("sparse", [False, True])
def test_linked_auxiliary_aliases_with_groups(monkeypatch, sparse):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse)
    data = pd.DataFrame(
        {
            "x": [0.0, 0.5, 1.5, 2.0],
            "z": [-1.0, 0.0, 0.5, 1.0],
            "y": [0.2, 0.4, 0.7, 0.6],
            "g": ["a", "a", "b", "b"],
        }
    )
    formula = bmb.Formula("y ~ a * exp(-x)", "a ~ 1 + (1 | g)", "kappa ~ 1 + z", nonlinear=True)
    model = bmb.Model(formula, data, family="beta")
    model.set_alias(
        {
            "y": "outcome",
            "mu": {"mu": "probability"},
            "a": {"a": "baseline", "1|g": "group_effect"},
            "kappa": {"kappa": "precision"},
        }
    )
    idata = model.fit(draws=3, chains=1, random_seed=123)
    new_data = pd.DataFrame({"x": [0.2, 1.0], "z": [-0.5, 0.5], "g": ["a", "new"]})
    for include_groups in [False, True]:
        predicted = model.predict(
            idata,
            data=new_data,
            kind="response",
            include_group_specific=include_groups,
            random_seed=123,
            inplace=False,
        )
        assert predicted.predictions["outcome"].shape == (1, 3, 2)
        assert np.isfinite(predicted.predictions["probability"]).all()
        assert (
            (predicted.predictions["probability"] > 0) & (predicted.predictions["probability"] < 1)
        ).all()
        assert (predicted.predictions["precision"] > 0).all()
        baseline = idata.posterior["a_Intercept"]
        if include_groups:
            baseline = baseline + idata.posterior["group_effect"].sel(g_dim="a")
        expected = 1 / (1 + np.exp(-baseline * np.exp(-0.2)))
        np.testing.assert_allclose(predicted.predictions["probability"].isel(__obs__=0), expected)
