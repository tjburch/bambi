import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from scipy.stats import norm

import bambi as bmb


@pytest.fixture
def exponential_model():
    data = pd.DataFrame(
        {"x": [0.0, 0.5, 1.5, 3.0], "z": [-1.0, 0.0, 0.5, 2.0], "y": [2.0, 1.5, 1.0, 0.8]}
    )
    formula = bmb.Formula("y ~ a + b * exp(-k * x)", "a ~ 1 + z", "b ~ 1", "k ~ 1", nonlinear=True)
    model = bmb.Model(
        formula,
        data,
        priors={
            "a": {
                "Intercept": bmb.Prior("Normal", mu=0, sigma=2),
                "z": bmb.Prior("Normal", mu=0, sigma=1),
            },
            "b": {"Intercept": bmb.Prior("Normal", mu=1, sigma=2)},
            "k": {"Intercept": bmb.Prior("LogNormal", mu=0, sigma=0.5)},
            "sigma": bmb.Prior("HalfNormal", sigma=1),
        },
        center_predictors=False,
    )
    model.build()
    return model


def test_exponential_log_density_matches_direct_pymc(exponential_model):
    data = exponential_model.data
    with pm.Model(coords={"__obs__": np.arange(len(data))}) as reference:
        x = pm.Data("x", data["x"], dims="__obs__")
        z = pm.Data("z", data["z"], dims="__obs__")
        a_intercept = pm.Normal("a_Intercept", mu=0, sigma=2)
        a_z = pm.Normal("a_z", mu=0, sigma=1)
        b = pm.Normal("b_Intercept", mu=1, sigma=2)
        k = pm.LogNormal("k_Intercept", mu=0, sigma=0.5)
        sigma = pm.HalfNormal("sigma", sigma=1)
        mu = a_intercept + a_z * z + b * pm.math.exp(-k * x)
        pm.Normal("y", mu=mu, sigma=sigma, observed=data["y"], dims="__obs__")

    actual_logp = exponential_model.backend.model.compile_logp()
    expected_logp = reference.compile_logp()
    for a_intercept, a_z, b, k, sigma in [(0.4, 0.2, 1.5, 0.8, 0.3), (-0.2, 0.5, 2, 1.2, 0.7)]:
        point = {
            "a_Intercept": np.array(a_intercept),
            "a_z": np.array(a_z),
            "b_Intercept": np.array(b, dtype=float),
            "k_Intercept_log__": np.log(k),
            "sigma_log__": np.log(sigma),
        }
        np.testing.assert_allclose(actual_logp(point), expected_logp(point))


@pytest.mark.parametrize("out_of_sample", [False, True])
def test_exponential_log_likelihood_matches_normal(exponential_model, out_of_sample):
    posterior = xr.Dataset(
        {
            "a_Intercept": (("chain", "draw"), [[0.4, -0.2]]),
            "a_z": (("chain", "draw"), [[0.2, 0.5]]),
            "b_Intercept": (("chain", "draw"), [[1.5, 2.0]]),
            "k_Intercept": (("chain", "draw"), [[0.8, 1.2]]),
            "sigma": (("chain", "draw"), [[0.3, 0.7]]),
        }
    )
    idata = xr.DataTree.from_dict({"posterior": posterior})
    data = (
        pd.DataFrame({"x": [0.2, 2.5], "z": [1.5, -0.5], "y": [2.2, 0.3]})
        if out_of_sample
        else exponential_model.data
    )
    result = exponential_model.compute_log_likelihood(
        idata, data=data if out_of_sample else None, inplace=False
    )
    x = xr.DataArray(data["x"].to_numpy(), dims="__obs__")
    z = xr.DataArray(data["z"].to_numpy(), dims="__obs__")
    mu = posterior["a_Intercept"] + posterior["a_z"] * z
    mu += posterior["b_Intercept"] * np.exp(-posterior["k_Intercept"] * x)
    expected = norm.logpdf(data["y"].to_numpy(), loc=mu, scale=posterior["sigma"].values[..., None])

    np.testing.assert_allclose(result.log_likelihood["y"], expected)
    assert "log_likelihood" not in idata


def test_zero_predictor_broadcasts_for_new_observations():
    data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.1, 1.2, 2.1]})
    model = bmb.Model(bmb.Formula("y ~ a + x", "a ~ 0", nonlinear=True), data)
    model.build()
    idata = xr.DataTree.from_dict(
        {"posterior": xr.Dataset({"sigma": (("chain", "draw"), [[0.2]])})}
    )

    fitted = model.predict(idata, inplace=False)
    predicted = model.predict(idata, data=pd.DataFrame({"x": [3.0, 4.0]}), inplace=False)

    np.testing.assert_allclose(fitted.posterior["mu"], [[[0.0, 1.0, 2.0]]])
    np.testing.assert_allclose(predicted.predictions["mu"], [[[3.0, 4.0]]])


@pytest.mark.parametrize("out_of_sample", [False, True])
def test_multiple_nonlinear_summands_share_parameter(out_of_sample):
    data = pd.DataFrame({"x": [0.0, 0.5, 1.5, 3.0], "y": [2.0, 1.5, 1.0, 0.8]})
    formula = bmb.Formula(
        "y ~ a * exp(-k * x) + b * exp(-2 * k * x)",
        "a ~ 1",
        "b ~ 1",
        "k ~ 1",
        nonlinear=True,
    )
    model = bmb.Model(formula, data)
    model.build()
    posterior = xr.Dataset(
        {
            "a_Intercept": (("chain", "draw"), [[0.4, 1.2]]),
            "b_Intercept": (("chain", "draw"), [[1.5, 2.0]]),
            "k_Intercept": (("chain", "draw"), [[0.8, 1.2]]),
            "sigma": (("chain", "draw"), [[0.3, 0.7]]),
        }
    )
    idata = xr.DataTree.from_dict({"posterior": posterior})
    prediction_data = pd.DataFrame({"x": [0.2, 2.5]}) if out_of_sample else data
    result = model.predict(idata, data=prediction_data if out_of_sample else None, inplace=False)
    x = xr.DataArray(prediction_data["x"].to_numpy(), dims="__obs__")
    a = posterior["a_Intercept"]
    b = posterior["b_Intercept"]
    k = posterior["k_Intercept"]
    expected = a * np.exp(-k * x) + b * np.exp(-2 * k * x)
    actual = result.predictions["mu"] if out_of_sample else result.posterior["mu"]

    assert actual.dims == ("chain", "draw", "__obs__")
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("out_of_sample", [False, True])
def test_nonlinear_parameter_offset_prediction(out_of_sample):
    data = pd.DataFrame(
        {
            "x": [0.0, 0.5, 1.5, 3.0],
            "z": [-1.0, 0.0, 0.5, 2.0],
            "exposure": [0.2, 0.5, 1.0, 1.5],
            "y": [0.1, 1.5, 3.0, 5.8],
        }
    )
    formula = bmb.Formula("y ~ exp(a) * x", "a ~ 1 + z + offset(exposure)", nonlinear=True)
    model = bmb.Model(formula, data, center_predictors=False)
    model.build()
    posterior = xr.Dataset(
        {
            "a_Intercept": (("chain", "draw"), [[0.4, -0.2]]),
            "a_z": (("chain", "draw"), [[0.2, 0.5]]),
            "sigma": (("chain", "draw"), [[0.3, 0.7]]),
        }
    )
    idata = xr.DataTree.from_dict({"posterior": posterior})
    prediction_data = (
        pd.DataFrame({"x": [0.2, 2.5], "z": [1.5, -0.5], "exposure": [0.7, 2.0]})
        if out_of_sample
        else data
    )
    result = model.predict(idata, data=prediction_data if out_of_sample else None, inplace=False)
    x = xr.DataArray(prediction_data["x"].to_numpy(), dims="__obs__")
    z = xr.DataArray(prediction_data["z"].to_numpy(), dims="__obs__")
    exposure = xr.DataArray(prediction_data["exposure"].to_numpy(), dims="__obs__")
    a = posterior["a_Intercept"] + posterior["a_z"] * z + exposure
    expected = np.exp(a) * x
    actual = result.predictions["mu"] if out_of_sample else result.posterior["mu"]

    assert actual.dims == ("chain", "draw", "__obs__")
    np.testing.assert_allclose(actual, expected)
