import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

import bambi as bmb

from helpers import assert_ip_dlogp


def normal_prior(sigma=2):
    return bmb.Prior("Normal", mu=0, sigma=sigma)


def linear_data(size=30):
    x = np.linspace(-1, 1, size)
    return pd.DataFrame({"x": x, "z": x**2, "y": 1 + 2 * x})


def exponential_formula(group_specific=False):
    a_formula = "a ~ 1 + (1 | group)" if group_specific else "a ~ 1 + z"
    return bmb.Formula(
        "y ~ a + b * exp(-k * x)",
        a_formula,
        "b ~ 1",
        "k ~ 1",
        nonlinear=True,
    )


def exponential_priors(group_specific=False):
    a_priors = {"Intercept": normal_prior(), "z": normal_prior()}
    if group_specific:
        a_priors = {
            "Intercept": normal_prior(),
            "1|group": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)),
        }
    return {
        "a": a_priors,
        "b": {"Intercept": normal_prior()},
        "k": {"Intercept": normal_prior()},
    }


def test_constant_parameters_match_linear_regression():
    data = linear_data()
    nonlinear = bmb.Model(
        bmb.Formula("y ~ a + b * x", "a ~ 1", "b ~ 1", nonlinear=True),
        data,
        priors={
            "a": {"Intercept": normal_prior()},
            "b": {"Intercept": normal_prior()},
        },
        center_predictors=False,
    )
    linear = bmb.Model(
        "y ~ x",
        data,
        priors={"Intercept": normal_prior(), "x": normal_prior()},
        center_predictors=False,
    )
    nonlinear.build()
    linear.build()

    nonlinear_draws = xr.Dataset(
        {
            "a_Intercept": (("chain", "draw"), [[1.25]]),
            "b_Intercept": (("chain", "draw"), [[-0.75]]),
            "sigma": (("chain", "draw"), [[1.0]]),
        }
    )
    linear_draws = xr.Dataset(
        {
            "Intercept": (("chain", "draw"), [[1.25]]),
            "x": (("chain", "draw"), [[-0.75]]),
            "sigma": (("chain", "draw"), [[1.0]]),
        }
    )

    with nonlinear.backend.model:
        nonlinear_mu = pm.compute_deterministics(
            nonlinear_draws, var_names=["mu"], progressbar=False
        )["mu"]
    with linear.backend.model:
        linear_mu = pm.compute_deterministics(linear_draws, var_names=["mu"], progressbar=False)[
            "mu"
        ]

    xr.testing.assert_allclose(nonlinear_mu, linear_mu)


def test_supported_expression_operations():
    data = pd.DataFrame({"x": [1.0, 2.0], "y": [0.0, 0.0]})
    formula = bmb.Formula(
        "y ~ sqrt(a ** 2) + log(b) / x",
        "a ~ 1",
        "b ~ 1",
        nonlinear=True,
    )
    model = bmb.Model(
        formula,
        data,
        priors={
            "a": {"Intercept": normal_prior()},
            "b": {"Intercept": bmb.Prior("LogNormal", mu=0, sigma=1)},
        },
    )
    model.build()
    draws = xr.Dataset(
        {
            "a_Intercept": (("chain", "draw"), [[-3.0]]),
            "b_Intercept": (("chain", "draw"), [[np.exp(2.0)]]),
            "sigma": (("chain", "draw"), [[1.0]]),
        }
    )

    with model.backend.model:
        result = pm.compute_deterministics(draws, var_names=["mu"], progressbar=False)["mu"]

    np.testing.assert_allclose(result, [[[5.0, 4.0]]])


def test_predictor_dependent_parameter_builds_expected_graph():
    data = linear_data()
    model = bmb.Model(exponential_formula(), data, priors=exponential_priors())
    model.build()

    assert_ip_dlogp(model)
    assert set(model.nonlinear_predictors) == {"a", "b", "k"}
    assert model.backend.model.named_vars_to_dims["mu"] == ("__obs__",)
    assert model.backend.model.named_vars_to_dims["a"] == ("__obs__",)
    assert model.backend.model.named_vars_to_dims["mu__x_data"] == ("__obs__",)
    assert {"a_Intercept", "a_z", "b_Intercept", "k_Intercept"} <= set(
        model.backend.model.named_vars
    )


def test_predictor_dependent_parameter_recovers_simulated_values():
    rng = np.random.default_rng(8)
    size = 120
    x = rng.uniform(0, 3, size)
    z = rng.normal(size=size)
    a = 0.7 + 0.4 * z
    y = a + 1.5 * np.exp(-0.8 * x) + rng.normal(0, 0.15, size)
    data = pd.DataFrame({"x": x, "z": z, "y": y})
    priors = exponential_priors()
    priors["b"]["Intercept"] = bmb.Prior("Normal", mu=1.5, sigma=0.5)
    priors["k"]["Intercept"] = bmb.Prior("LogNormal", mu=np.log(0.8), sigma=0.35)
    priors["sigma"] = bmb.Prior("HalfNormal", sigma=0.5)
    model = bmb.Model(exponential_formula(), data, priors=priors)

    idata = model.fit(
        draws=200,
        tune=200,
        chains=2,
        cores=1,
        random_seed=11,
        inference_method="pymc",
        target_accept=0.9,
    )

    expected = {
        "a_Intercept": 0.7,
        "a_z": 0.4,
        "b_Intercept": 1.5,
        "k_Intercept": 0.8,
    }
    for name, value in expected.items():
        assert float(idata.posterior[name].mean()) == pytest.approx(value, abs=0.15)

    predicted = model.predict(idata, kind="response", random_seed=4, inplace=False)
    assert predicted.posterior_predictive["y"].shape == (2, 200, size)


@pytest.mark.usefixtures("mock_pymc_sample")
def test_posterior_predictive_and_new_data():
    data = linear_data()
    model = bmb.Model(exponential_formula(), data, priors=exponential_priors())
    idata = model.fit(draws=5, chains=2)
    assert {"a", "b", "k", "mu"}.isdisjoint(idata.posterior.data_vars)

    predicted = model.predict(idata, kind="response", inplace=False)
    assert predicted.posterior["mu"].shape == (2, 5, len(data))
    assert predicted.posterior_predictive["y"].shape == (2, 5, len(data))

    new_data = pd.DataFrame({"x": np.linspace(1, 2, 7), "z": np.linspace(-0.5, 0.5, 7)})
    predicted = model.predict(idata, kind="response", data=new_data, inplace=False)
    assert predicted.predictions["mu"].shape == (2, 5, len(new_data))
    assert predicted.predictions["y"].shape == (2, 5, len(new_data))

    with pytest.raises(ValueError, match="missing nonlinear expression column"):
        model.predict(idata, data=new_data.drop(columns="x"), inplace=False)


@pytest.mark.usefixtures("mock_pymc_sample")
def test_include_response_params_only_keeps_likelihood_parameter():
    model = bmb.Model(exponential_formula(), linear_data(), priors=exponential_priors())

    idata = model.fit(draws=5, chains=2, include_response_params=True)

    assert "mu" in idata.posterior
    assert {"a", "b", "k"}.isdisjoint(idata.posterior.data_vars)


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("sparse_dot", [False, True])
def test_group_specific_parameter_predicts_new_data(monkeypatch, sparse_dot):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    data = linear_data(24)
    data["group"] = np.repeat(["a", "b", "c"], 8)
    formula = exponential_formula(group_specific=True)
    model = bmb.Model(formula, data, priors=exponential_priors(group_specific=True))
    idata = model.fit(draws=5, chains=2)

    assert model.backend.model.named_vars_to_dims["a_1|group"] == ("group_dim",)
    assert idata.posterior["a_1|group"].shape == (2, 5, 3)
    new_data = pd.DataFrame(
        {
            "x": [0.1, 0.2, 0.3],
            "group": ["a", "new", "new"],
        }
    )
    predicted = model.predict(idata, data=new_data, random_seed=123, inplace=False)
    assert predicted.predictions["mu"].shape == (2, 5, 3)


@pytest.mark.parametrize(
    "formula, error",
    [
        (
            bmb.Formula("y ~ a + b * x", "b ~ 1", nonlinear=True),
            "No nonlinear parameter formula or data column",
        ),
        (
            bmb.Formula("y ~ a + x", "a ~ 1", "b ~ 1", nonlinear=True),
            "not used by the expression",
        ),
        (
            bmb.Formula("y ~ a + unknown", "a ~ 1", nonlinear=True),
            "No nonlinear parameter formula or data column",
        ),
        (
            bmb.Formula("y ~ a + sin(x)", "a ~ 1", nonlinear=True),
            "Unsupported nonlinear function 'sin'",
        ),
        (
            bmb.Formula("y ~ a + b * x", "a ~ 1 + b", "b ~ 1", nonlinear=True),
            "cannot depend on one another",
        ),
    ],
)
def test_validation_errors(formula, error):
    with pytest.raises(ValueError, match=error):
        bmb.Model(formula, linear_data())


def test_malformed_expression():
    formula = bmb.Formula("y ~ a +", "a ~ 1", nonlinear=True)
    with pytest.raises(ValueError, match="Malformed nonlinear expression"):
        bmb.Model(formula, linear_data())


def test_nonnumeric_expression_data():
    data = linear_data()
    data["label"] = "a"
    formula = bmb.Formula("y ~ a + label", "a ~ 1", nonlinear=True)

    with pytest.raises(ValueError, match="Nonlinear expression data must be numeric"):
        bmb.Model(formula, data)


def test_nested_priors_are_assigned():
    model = bmb.Model(exponential_formula(), linear_data(), priors=exponential_priors())

    prior = model.nonlinear_predictors["a"].terms["z"].prior
    assert prior.name == "Normal"
    assert prior.args == {"mu": 0, "sigma": 2}


def test_set_priors_uses_nested_parameter_names():
    model = bmb.Model(exponential_formula(), linear_data(), priors=exponential_priors())
    updated = bmb.Prior("Normal", mu=1, sigma=0.25)

    model.set_priors({"a": {"z": updated}})

    prior = model.nonlinear_predictors["a"].terms["z"].prior
    assert prior.name == "Normal"
    assert prior.args == {"mu": 1, "sigma": 0.25}


def test_nonlinear_coefficient_alias():
    model = bmb.Model(exponential_formula(), linear_data(), priors=exponential_priors())

    model.set_alias({"a": {"Intercept": "a0"}})
    model.build()
    assert "a0" in model.backend.model.named_vars
    assert "a_Intercept" not in model.backend.model.named_vars


def test_vector_parent_is_rejected():
    formula = bmb.Formula("y ~ rate * x", "rate ~ 1", nonlinear=True)

    with pytest.raises(ValueError, match="scalar parent parameter"):
        bmb.Model(formula, linear_data(), family="categorical")


def test_dropna_aligns_all_model_inputs():
    data = linear_data(8)
    data.index = [4, 4, 2, 2, 9, 9, 1, 1]
    data["group"] = ["g", "h"] * 4
    data["unused"] = np.nan
    for row, column in enumerate(["y", "x", "z", "group"]):
        data.iloc[row, data.columns.get_loc(column)] = np.nan
    original = data.copy(deep=True)
    formula = bmb.Formula("y ~ a + b * x", "a ~ 1 + z", "b ~ 1 + (1 | group)", nonlinear=True)
    model = bmb.Model(formula, data, dropna=True)
    model.build()

    expected = data.iloc[4:].copy()
    expected["group"] = expected["group"].astype("category")
    pd.testing.assert_frame_equal(model.data, expected)
    pd.testing.assert_frame_equal(data, original)
    np.testing.assert_array_equal(model.response_term.data, data.y.iloc[4:])
    np.testing.assert_array_equal(model.backend.model["mu__x_data"].get_value(), data.x.iloc[4:])
    np.testing.assert_array_equal(model.nonlinear_predictors["a"].terms["z"].data, data.z.iloc[4:])
    assert_ip_dlogp(model)


@pytest.mark.parametrize("column", ["y", "x", "z"])
def test_missing_model_inputs_raise_without_dropna(column):
    data = linear_data()
    data.loc[0, column] = np.nan
    with pytest.raises(ValueError, match="incomplete rows"):
        bmb.Model(exponential_formula(), data)


def test_dropna_rejects_no_complete_observations():
    data = linear_data()
    data["x"] = np.nan
    with pytest.raises(ValueError, match="complete observation"):
        bmb.Model(exponential_formula(), data, dropna=True)


def test_dependency_check_ignores_string_literals():
    data = linear_data(4)
    data["category"] = ["a", "b", "a", "b"]
    formula = bmb.Formula("y ~ a * x", "a ~ C(category, Treatment(reference='a'))", nonlinear=True)
    model = bmb.Model(formula, data)
    model.build()
    assert_ip_dlogp(model)


@pytest.mark.parametrize("rhs", ["b", "I(b ** 2)", "(1 | b)"])
def test_dependency_check_uses_formula_variables(rhs):
    formula = bmb.Formula("y ~ a + b * x", f"a ~ {rhs}", "b ~ 1", nonlinear=True)
    with pytest.raises(ValueError, match="cannot depend on one another"):
        bmb.Model(formula, linear_data())


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("column", ["x", "z"])
def test_prediction_rejects_incomplete_inputs(column):
    model = bmb.Model(
        exponential_formula(), linear_data(), priors=exponential_priors(), dropna=True
    )
    idata = model.fit(draws=2, chains=1)
    data = linear_data(3).drop(columns="y")
    data.loc[1, column] = np.nan
    with pytest.raises(ValueError, match="incomplete rows"):
        model.predict(idata, data=data, inplace=False)


@pytest.mark.parametrize(
    "expression",
    [
        "a + np.exp(x)",
        "a + x[0]",
        "a + (x > 0)",
        "a + [x]",
        "a + True",
        "a + 'x'",
        "a % x",
        "a // x",
        "a + (lambda: x)()",
        "a + exp(x, x)",
        "a + exp(value=x)",
    ],
)
def test_unsupported_expression_syntax_is_rejected(expression):
    formula = bmb.Formula(f"y ~ {expression}", "a ~ 1", nonlinear=True)
    with pytest.raises(ValueError, match="Nonlinear|nonlinear|Unsupported"):
        bmb.Model(formula, linear_data())


@pytest.mark.parametrize("name", ["mu", "sigma", "exp", "x"])
def test_reserved_parameter_names_are_rejected(name):
    formula = bmb.Formula(f"y ~ {name}", f"{name} ~ 1", nonlinear=True)
    with pytest.raises(ValueError, match="names must not"):
        bmb.Model(formula, linear_data())
