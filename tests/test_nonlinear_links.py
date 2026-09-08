import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from scipy.special import expit, ndtr  # pylint: disable=no-name-in-module

import bambi as bmb

from bambi.backend.pymc.transform import transforms_registry


@pytest.mark.parametrize(
    "family, link, inverse_link, distribution, parent, auxiliary",
    [
        ("poisson", "log", np.exp, pm.Poisson, "mu", {}),
        ("bernoulli", "logit", expit, pm.Bernoulli, "p", {}),
        ("bernoulli", "probit", ndtr, pm.Bernoulli, "p", {}),
        ("bernoulli", "cloglog", lambda x: -np.expm1(-np.exp(x)), pm.Bernoulli, "p", {}),
        ("gaussian", "identity", lambda x: x, pm.Normal, "mu", {"sigma": 1.0}),
        (
            "poisson",
            bmb.Link("scaled_log", inverse_link=lambda x: pm.math.exp(x / 2)),
            lambda x: np.exp(x / 2),
            pm.Poisson,
            "mu",
            {},
        ),
    ],
)
def test_parent_link_matches_pymc_and_prediction(
    family, link, inverse_link, distribution, parent, auxiliary
):
    data = pd.DataFrame({"y": [0, 1, 0, 1], "x": [-1.0, -0.3, 0.2, 0.7]})
    model = bmb.Model(
        bmb.Formula("y ~ a + b * x ** 2", "a ~ 1", "b ~ 1", nonlinear=True),
        data,
        family=family,
        link={parent: link},
        priors=auxiliary,
    )
    model.build()
    draws = xr.Dataset(
        {
            "a_Intercept": (("chain", "draw"), [[-0.4, 0.3]]),
            "b_Intercept": (("chain", "draw"), [[0.8, -0.2]]),
        }
    )
    idata = xr.DataTree.from_dict({"posterior": draws})
    with model.backend.model:
        actual = pm.compute_deterministics(draws, progressbar=False)
    np.testing.assert_allclose(actual.a.values, np.broadcast_to([[[-0.4], [0.3]]], (1, 2, 4)))
    for new_data in (None, pd.DataFrame({"x": [0.1, 1.4], "y": [1, 0]})):
        prediction_data = data if new_data is None else new_data
        eta = np.array([-0.4, 0.3])[:, None] + np.array([0.8, -0.2])[:, None] * (
            prediction_data.x.to_numpy() ** 2
        )
        expected = inverse_link(eta)
        result = model.predict(idata, data=new_data, inplace=False)
        group = result.posterior if new_data is None else result.predictions
        np.testing.assert_allclose(group[parent].values, expected[None])
        likelihood = model.compute_log_likelihood(idata, data=new_data, inplace=False)
        direct = pm.logp(
            distribution.dist(**{parent: expected}, **auxiliary), prediction_data.y.to_numpy()
        ).eval()
        np.testing.assert_allclose(likelihood.log_likelihood.y.values, direct[None])


def test_link_preserves_likelihood_parameter_transform():
    data = pd.DataFrame({"y": [0.2, 0.7], "x": [-0.5, 0.5]})
    model = bmb.Model(
        bmb.Formula("y ~ a * x", "a ~ 1", nonlinear=True),
        data,
        family="beta",
        priors={"kappa": 4.0},
    )
    model.build()
    draws = xr.Dataset({"a_Intercept": (("chain", "draw"), [[1.2]])})
    result = model.compute_log_likelihood(
        xr.DataTree.from_dict({"posterior": draws}), inplace=False
    )
    mu = expit(1.2 * data.x.to_numpy())
    expected = pm.logp(pm.Beta.dist(alpha=mu * 4, beta=(1 - mu) * 4), data.y).eval()
    np.testing.assert_allclose(result.log_likelihood.y.values, [[expected]])


def test_scalar_predictor_transform_receives_auxiliary_parameters(monkeypatch):
    data = pd.DataFrame({"y": [0.0, 1.0], "x": [-0.5, 0.5]})
    model = bmb.Model(
        bmb.Formula("y ~ a * x", "a ~ 1", nonlinear=True), data, priors={"sigma": 2.0}
    )
    monkeypatch.setitem(
        transforms_registry.additive_predictors,
        (type(model.family), "mu"),
        lambda value, parameters, inverse_link: inverse_link(value + parameters["sigma"]),
    )
    model.build()
    draws = xr.Dataset({"a_Intercept": (("chain", "draw"), [[1.2]])})
    result = model.predict(xr.DataTree.from_dict({"posterior": draws}), inplace=False)
    np.testing.assert_allclose(result.posterior.mu.values, [[[1.4, 2.6]]])


@pytest.mark.parametrize("transform", ["inverse_link", "predictor"])
def test_scalar_transform_result_broadcasts_for_prediction(monkeypatch, transform):
    data = pd.DataFrame({"y": [0.0, 1.0], "x": [-0.5, 0.5]})
    link = {"mu": bmb.Link("constant", inverse_link=lambda value: 1.0)}
    model = bmb.Model(
        bmb.Formula("y ~ a * x", "a ~ 1", nonlinear=True),
        data,
        priors={"sigma": 1.0},
        link=link if transform == "inverse_link" else None,
    )
    if transform == "predictor":
        monkeypatch.setitem(
            transforms_registry.additive_predictors,
            (type(model.family), "mu"),
            lambda value, parameters, inverse_link: 1.0,
        )
    model.build()
    draws = xr.Dataset({"a_Intercept": (("chain", "draw"), [[1.2]])})
    idata = xr.DataTree.from_dict({"posterior": draws})
    for new_data in (None, pd.DataFrame({"x": [-1.0, 0.0, 1.0]})):
        result = model.predict(idata, data=new_data, inplace=False)
        group = result.posterior if new_data is None else result.predictions
        size = len(data) if new_data is None else len(new_data)
        np.testing.assert_array_equal(group.mu.values, np.ones((1, 1, size)))


@pytest.mark.parametrize("family", ["categorical", "cumulative", "sratio"])
def test_vector_parent_links_remain_rejected(family):
    with pytest.raises(ValueError, match="scalar parent parameter"):
        bmb.Model(
            bmb.Formula("y ~ a * x", "a ~ 1", nonlinear=True),
            pd.DataFrame({"y": [0, 1, 2], "x": [0, 1, 2]}),
            family=family,
        )


@pytest.mark.parametrize("family, inverse_link", [("gaussian", lambda x: x), ("poisson", np.exp)])
@pytest.mark.parametrize("predictor", ["a ~ 0", "a ~ 1"])
def test_intercept_only_prediction_accepts_row_only_data(family, inverse_link, predictor):
    model = bmb.Model(
        bmb.Formula("y ~ a + b", predictor, "b ~ 1", nonlinear=True),
        pd.DataFrame({"y": [0, 1]}),
        family=family,
        priors={"sigma": 1.0} if family == "gaussian" else None,
    )
    model.build()
    value = 0.5 if predictor == "a ~ 1" else 0.0
    draws = xr.Dataset({"b_Intercept": (("chain", "draw"), [[0.2]])})
    if predictor == "a ~ 1":
        draws["a_Intercept"] = (("chain", "draw"), [[value]])
    idata = xr.DataTree.from_dict({"posterior": draws})
    result = model.predict(idata, data=pd.DataFrame(index=range(3)), inplace=False)
    np.testing.assert_allclose(
        result.predictions.mu.values, np.full((1, 1, 3), inverse_link(value + 0.2))
    )
    with pytest.raises(ValueError, match="does not contain any complete observation"):
        model.predict(idata, data=pd.DataFrame(), inplace=False)


def test_invalid_family_link_remains_rejected():
    with pytest.raises(ValueError, match="cannot be used"):
        bmb.Model(
            bmb.Formula("y ~ a", "a ~ 1", nonlinear=True),
            pd.DataFrame({"y": [0, 1]}),
            family="poisson",
            link="logit",
        )


def test_bare_predictor_preserves_parent_name_in_new_data():
    model = bmb.Model(
        bmb.Formula("y ~ a", "a ~ 1", nonlinear=True),
        pd.DataFrame({"y": [0, 1]}),
        priors={"sigma": 1.0},
    )
    model.build()
    draws = xr.Dataset({"a_Intercept": (("chain", "draw"), [[0.5]])})
    idata = xr.DataTree.from_dict({"posterior": draws})
    for new_data in (None, pd.DataFrame(index=range(3))):
        result = model.predict(idata, data=new_data, inplace=False)
        group = result.posterior if new_data is None else result.predictions
        size = 2 if new_data is None else 3
        np.testing.assert_allclose(group.mu.values, np.full((1, 1, size), 0.5))
        assert "a" not in group
