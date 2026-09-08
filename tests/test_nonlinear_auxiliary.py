import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

import bambi as bmb


def make_data():
    x = np.linspace(-1, 1, 12)
    return pd.DataFrame({"y": 1 + x, "x": x, "z": x**2, "group": np.repeat(["a", "b", "c"], 4)})


def make_formula(groups=False):
    suffix = " + (1 | group)" if groups else ""
    return bmb.Formula("y ~ a * x", "a ~ 1" + suffix, "sigma ~ z" + suffix, nonlinear=True)


def test_auxiliary_graph_matches_direct_pymc():
    data = make_data()
    model = bmb.Model(make_formula(), data, center_predictors=False)
    model.build()
    draws = xr.Dataset(
        {
            "a_Intercept": (("chain", "draw"), [[2.0]]),
            "sigma_Intercept": (("chain", "draw"), [[-0.5]]),
            "sigma_z": (("chain", "draw"), [[0.3]]),
        }
    )
    with model.backend.model:
        actual = pm.compute_deterministics(draws, progressbar=False)
    np.testing.assert_allclose(actual.mu.values, [[2 * data.x]])
    np.testing.assert_allclose(actual.sigma.values, [[np.exp(-0.5 + 0.3 * data.z)]])
    likelihood = model.compute_log_likelihood(
        xr.DataTree.from_dict({"posterior": draws}), inplace=False
    )
    direct = pm.logp(
        pm.Normal.dist(mu=2 * data.x, sigma=np.exp(-0.5 + 0.3 * data.z)), data.y
    ).eval()
    np.testing.assert_allclose(likelihood.log_likelihood.y.values, [[direct]])
    assert set(model.additive_parameters) == {"a", "sigma"}
    assert set(model.nonlinear_predictors) == {"a"}
    assert not model.marginal_parameters
    assert "target = sigma" in str(model)


def test_auxiliary_priors_match_ordinary_model_and_update():
    data = make_data()
    nonlinear = bmb.Model(make_formula(), data)
    ordinary = bmb.Model(bmb.Formula("y ~ x", "sigma ~ z"), data)
    for name, term in nonlinear.parameters["sigma"].terms.items():
        assert term.prior == ordinary.parameters["sigma"].terms[name].prior
    nonlinear.set_priors({"sigma": {"z": bmb.Prior("Normal", mu=0.2, sigma=0.4)}})
    assert nonlinear.parameters["sigma"].terms["z"].prior.args["mu"] == 0.2
    nonlinear.build()
    with pytest.warns(UserWarning, match="sigma.unknown"):
        nonlinear.set_priors({"sigma": {"unknown": bmb.Prior("Normal", mu=0, sigma=1)}})
    with pytest.raises(ValueError, match="must be a dictionary"):
        bmb.Model(make_formula(), data, priors={"sigma": bmb.Prior("HalfNormal", sigma=1)})


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("sparse_dot", [False, True])
def test_auxiliary_group_prediction_and_likelihood(monkeypatch, sparse_dot):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    model = bmb.Model(
        make_formula(groups=True), make_data(), noncentered={"a": False, "sigma": True}
    )
    idata = model.fit(draws=3, chains=1, include_response_params=True)
    assert {"mu", "sigma"} <= set(idata.posterior)
    assert "a" not in idata.posterior
    new_data = make_data().iloc[:3].copy()
    new_data["z"] += 1
    result = model.predict(idata, data=new_data, kind="response", inplace=False)
    assert result.predictions.sigma.shape == (1, 3, 3)
    assert result.predictions.y.shape == (1, 3, 3)
    likelihood = model.compute_log_likelihood(idata, data=new_data, inplace=False)
    assert likelihood.log_likelihood.y.shape == (1, 3, 3)
    new_data.loc[new_data.index[0], "z"] = np.nan
    with pytest.raises(ValueError, match="incomplete rows"):
        model.predict(idata, data=new_data, inplace=False)


def test_auxiliary_missing_rows_share_one_mask():
    data = make_data()
    for index, column in enumerate(["y", "x", "z", "group"]):
        data.loc[index, column] = np.nan
    model = bmb.Model(make_formula(groups=True), data, dropna=True)
    assert len(model.data) == 8
    np.testing.assert_array_equal(model.response_term.data, data.y.iloc[4:])
    np.testing.assert_array_equal(model.parameters["sigma"].terms["z"].data, data.z.iloc[4:])
    model.build()
    with pytest.raises(ValueError, match="incomplete rows"):
        bmb.Model(make_formula(groups=True), data)


@pytest.mark.parametrize(
    "main, additionals",
    [
        ("y ~ a * sigma", ["a ~ 1", "sigma ~ z"]),
        ("y ~ a * x", ["a ~ sigma", "sigma ~ z"]),
        ("y ~ a * x", ["a ~ 1", "sigma ~ a"]),
        ("y ~ a * x", ["a ~ 1", "sigma ~ sigma"]),
        ("y ~ a * x", ["a ~ sigma"]),
    ],
)
def test_auxiliary_dependencies_rejected(main, additionals):
    with pytest.raises(ValueError, match="names must not|cannot depend"):
        bmb.Model(bmb.Formula(main, *additionals, nonlinear=True), make_data())


def test_duplicate_auxiliary_formula_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        bmb.Formula("y ~ a * x", "a ~ 1", "sigma ~ z", "sigma ~ 1", nonlinear=True)


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("auxiliary", [False, True])
def test_undeclared_likelihood_names_can_reference_data(auxiliary):
    data = make_data().rename(columns={"x": "sigma", "z": "mu"})
    additionals = ["a ~ mu"]
    if auxiliary:
        data["z"] = data["sigma"]
        additionals.append("sigma ~ z")
        expression = "a * z"
    else:
        expression = "a * sigma"
    formula = bmb.Formula(f"y ~ {expression}", *additionals, nonlinear=True)
    model = bmb.Model(formula, data, center_predictors=False)
    idata = model.fit(draws=3, chains=1)
    predicted = model.predict(idata, data=data.iloc[:2], inplace=False)
    assert predicted.predictions.mu.shape == (1, 3, 2)
    np.testing.assert_array_equal(model.nonlinear_predictors["a"].terms["mu"].data, data.mu)
