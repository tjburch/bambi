import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr
from scipy.special import erf, erfc, expit, logit, ndtr, ndtri  # pylint: disable=no-name-in-module

import bambi as bmb
from bambi.nonlinear import (
    NonlinearExpression,
    parameter_dependency_order,
    resolve_nonlinear_symbols,
    split_predictor,
)

from bambi.backend.pymc.nonlinear import evaluate_expression
from bambi.parameters import MarginalCoefficient

from helpers import assert_ip_dlogp


def normal_prior(sigma=2):
    return bmb.Prior("Normal", mu=0, sigma=sigma)


def linear_data(size=30):
    x = np.linspace(-1, 1, size)
    return pd.DataFrame({"x": x, "z": x**2, "y": 1 + 2 * x})


def test_parameter_dependency_order_is_stable_and_topological():
    dependencies = {
        "sigma": ("mu", "sigma_y"),
        "mu": ("p_angle", "p_distance"),
        "sigma_y": (),
        "p_distance": (),
        "p_angle": (),
    }

    result = parameter_dependency_order(
        dependencies, ("sigma", "mu", "sigma_y", "p_distance", "p_angle")
    )

    assert result == ("p_distance", "p_angle", "mu", "sigma_y", "sigma")


@pytest.mark.parametrize(
    ("dependencies", "message"),
    [
        ({"mu": ("mu",)}, "cannot depend on themselves: \\['mu'\\]"),
        ({"mu": ("sigma",), "sigma": ("mu",)}, "mu -> sigma -> mu"),
        (
            {"mu": ("scale",), "scale": ("sigma",), "sigma": ("mu",)},
            "mu -> scale -> sigma -> mu",
        ),
        ({"mu": ("missing",)}, "Unknown nonlinear parameter reference.*missing"),
    ],
)
def test_parameter_dependency_order_errors(dependencies, message):
    with pytest.raises(ValueError, match=message):
        parameter_dependency_order(dependencies)


def test_resolve_nonlinear_symbols_separates_parameters_and_data():
    expression = NonlinearExpression.parse("sqrt(mu * (1 - mu) / attempts + sigma_y ** 2)")
    data = pd.DataFrame({"attempts": [10, 20]})

    dependencies, data_names = resolve_nonlinear_symbols(
        expression, ("mu", "sigma", "sigma_y"), data
    )

    assert dependencies == ("mu", "sigma_y")
    assert data_names == ("attempts",)


def test_resolve_nonlinear_symbols_rejects_ambiguous_name():
    expression = NonlinearExpression.parse("mu + x")
    data = pd.DataFrame({"mu": [0.2], "x": [1.0]})

    with pytest.raises(ValueError, match="both modeled parameters and data columns.*mu"):
        resolve_nonlinear_symbols(expression, ("mu",), data)


@pytest.mark.parametrize(
    ("main", "additionals", "message"),
    [
        ("y ~ mu + a * x", (), "cannot depend on themselves.*mu"),
        ("y ~ sigma * a", ("sigma ~ mu + 1",), "mu -> sigma -> mu"),
        (
            "y ~ a * x",
            ("a ~ sigma", "sigma ~ mu + 1"),
            "a -> sigma -> mu -> a",
        ),
        (
            "y ~ a * x",
            ("sigma ~ 0 + nl(mu + unknown)",),
            "No nonlinear parameter formula or data column.*unknown",
        ),
    ],
)
def test_parameter_dependency_validation(main, additionals, message):
    formula = bmb.Formula(main, *additionals, nlpars=("a",))
    with pytest.raises(ValueError, match=message):
        bmb.Model(formula, linear_data())


def test_likelihood_parameter_and_expression_data_collision_is_rejected():
    data = linear_data().assign(mu=0.5)
    formula = bmb.Formula("y ~ a * x", "sigma ~ mu + z", nlpars=("a",))

    with pytest.raises(ValueError, match="both modeled parameters and data columns.*mu"):
        bmb.Model(formula, data)


@pytest.mark.parametrize("response", ["mu", "sigma"])
def test_likelihood_parameter_and_response_data_collision_is_rejected(response):
    data = linear_data().rename(columns={"y": response})
    formula = bmb.Formula(f"{response} ~ a * x", nlpars=("a",))

    with pytest.raises(ValueError, match=f"modeled likelihood parameters.*{response}"):
        bmb.Model(formula, data)


def exponential_formula(group_specific=False):
    a_formula = "a ~ 1 + (1 | group)" if group_specific else "a ~ 1 + z"
    return bmb.Formula(
        "y ~ a + b * exp(-k * x)",
        a_formula,
        nlpars=("a", "b", "k"),
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
        "b": normal_prior(),
        "k": normal_prior(),
    }


def evaluate_nonlinear_expression(expression, data):
    data = data.assign(y=0.0)
    model = bmb.Model(
        bmb.Formula(f"y ~ 0 + nl({expression} + a)", nlpars=("a",)),
        data,
        priors={"a": normal_prior()},
    )
    model.build()
    draws = xr.Dataset(
        {
            "a": (("chain", "draw"), [[0.0]]),
            "sigma": (("chain", "draw"), [[1.0]]),
        }
    )

    with model.backend.model:
        return pm.compute_deterministics(draws, var_names=["mu"], progressbar=False)["mu"]


def test_constant_parameters_match_linear_regression():
    data = linear_data()
    nonlinear = bmb.Model(
        bmb.Formula("y ~ a + b * x", nlpars=("a", "b")),
        data,
        priors={
            "a": normal_prior(),
            "b": normal_prior(),
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
            "a": (("chain", "draw"), [[1.25]]),
            "b": (("chain", "draw"), [[-0.75]]),
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


def test_omitted_constant_formulas_match_explicit_intercepts():
    data = linear_data()
    formulas = [
        bmb.Formula("y ~ a + b * x", nlpars=("a", "b")),
        bmb.Formula("y ~ a + b * x", "a ~ 1", "b ~ 1", nlpars=("a", "b")),
    ]
    draws = xr.Dataset(
        {
            "a": (("chain", "draw"), [[1.25]]),
            "b": (("chain", "draw"), [[-0.75]]),
            "sigma": (("chain", "draw"), [[1.0]]),
        }
    )
    predictions = []

    for formula in formulas:
        model = bmb.Model(formula, data, center_predictors=False)
        model.build()
        with model.backend.model:
            predictions.append(
                pm.compute_deterministics(
                    draws,
                    var_names=["mu"],
                    progressbar=False,
                )["mu"]
            )

    xr.testing.assert_allclose(predictions[0], predictions[1])


def test_supported_expression_operations():
    data = pd.DataFrame({"x": [1.0, 2.0], "y": [0.0, 0.0]})
    formula = bmb.Formula(
        "y ~ sqrt(a ** 2) + log(b) / x",
        nlpars=("a", "b"),
    )
    model = bmb.Model(
        formula,
        data,
        priors={
            "a": normal_prior(),
            "b": bmb.Prior("LogNormal", mu=0, sigma=1),
        },
    )
    model.build()
    draws = xr.Dataset(
        {
            "a": (("chain", "draw"), [[-3.0]]),
            "b": (("chain", "draw"), [[np.exp(2.0)]]),
            "sigma": (("chain", "draw"), [[1.0]]),
        }
    )

    with model.backend.model:
        result = pm.compute_deterministics(draws, var_names=["mu"], progressbar=False)["mu"]

    np.testing.assert_allclose(result, [[[5.0, 4.0]]])


@pytest.mark.parametrize(
    "function",
    [
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "arcsinh",
        "arccosh",
        "arctanh",
        "log1p",
        "expm1",
        "softplus",
        "erf",
        "erfc",
    ],
)
def test_supported_unary_math_functions(function):
    x = np.array([0.2, 0.5])
    z = np.array([1.2, 1.5])
    references = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
        "arctan": np.arctan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "asinh": np.arcsinh,
        "acosh": np.arccosh,
        "atanh": np.arctanh,
        "arcsinh": np.arcsinh,
        "arccosh": np.arccosh,
        "arctanh": np.arctanh,
        "log1p": np.log1p,
        "expm1": np.expm1,
        "softplus": lambda value: np.logaddexp(0, value),
        "erf": erf,
        "erfc": erfc,
    }
    argument_name = "z" if function in {"acosh", "arccosh"} else "x"
    argument = z if argument_name == "z" else x

    result = evaluate_nonlinear_expression(
        f"{function}({argument_name})", pd.DataFrame({"x": x, "z": z})
    )

    expected = references[function](argument)
    np.testing.assert_allclose(result, expected[None, None, :])


@pytest.mark.parametrize("function", ["atan2", "arctan2"])
def test_supported_two_argument_functions(function):
    x = np.array([0.2, 0.5])
    z = np.array([1.2, 1.5])

    result = evaluate_nonlinear_expression(f"{function}(x, z)", pd.DataFrame({"x": x, "z": z}))

    np.testing.assert_allclose(result, np.arctan2(x, z)[None, None, :])


@pytest.mark.parametrize(
    "function",
    [
        "logit",
        "invlogit",
        "expit",
        "normal_cdf",
        "norm_cdf",
        "normal_ppf",
        "norm_ppf",
        "probit",
        "invprobit",
        "cloglog",
        "invcloglog",
    ],
)
def test_supported_probability_transforms(function):
    x = np.array([0.2, 0.5])
    references = {
        "logit": logit,
        "invlogit": expit,
        "expit": expit,
        "normal_cdf": ndtr,
        "norm_cdf": ndtr,
        "normal_ppf": ndtri,
        "norm_ppf": ndtri,
        "probit": ndtri,
        "invprobit": ndtr,
        "cloglog": lambda value: np.log(-np.log1p(-value)),
        "invcloglog": lambda value: -np.expm1(-np.exp(value)),
    }

    result = evaluate_nonlinear_expression(f"{function}(x)", pd.DataFrame({"x": x}))

    np.testing.assert_allclose(result, references[function](x)[None, None, :])


@pytest.mark.parametrize(
    "expression, message",
    [
        ("atan2(x)", "requires exactly 2 positional arguments"),
        ("atan2(x, x, x)", "requires exactly 2 positional arguments"),
        ("atan2(x=x, y=x)", "requires exactly 2 positional arguments"),
        ("sin(x, x)", "requires exactly 1 positional argument"),
    ],
)
def test_nonlinear_function_arity(expression, message):
    formula = bmb.Formula(f"y ~ 0 + nl(a + {expression})", nlpars=("a",))

    with pytest.raises(ValueError, match=message):
        bmb.Model(formula, linear_data())


def test_predictor_dependent_parameter_builds_expected_graph():
    data = linear_data()
    model = bmb.Model(exponential_formula(), data, priors=exponential_priors())
    model.build()

    assert_ip_dlogp(model)
    coefficients = model.parameters["mu"].nonlinear_coefficients
    assert set(coefficients) == {"a", "b", "k"}
    assert set(coefficients["a"].terms) == {"Intercept", "z"}
    assert isinstance(coefficients["b"], MarginalCoefficient)
    assert isinstance(coefficients["k"], MarginalCoefficient)
    assert model.backend.model.named_vars_to_dims["mu"] == ("__obs__",)
    assert model.backend.model.named_vars_to_dims["a"] == ("__obs__",)
    assert model.backend.model.named_vars_to_dims["mu__x_data"] == ("__obs__",)
    assert {"a_Intercept", "a_z", "b", "k"} <= set(model.backend.model.named_vars)


def test_predictor_dependent_parameter_recovers_simulated_values():
    rng = np.random.default_rng(8)
    size = 120
    x = rng.uniform(0, 3, size)
    z = rng.normal(size=size)
    a = 0.7 + 0.4 * z
    y = a + 1.5 * np.exp(-0.8 * x) + rng.normal(0, 0.15, size)
    data = pd.DataFrame({"x": x, "z": z, "y": y})
    priors = exponential_priors()
    priors["b"] = bmb.Prior("Normal", mu=1.5, sigma=0.5)
    priors["k"] = bmb.Prior("LogNormal", mu=np.log(0.8), sigma=0.35)
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
        "b": 1.5,
        "k": 0.8,
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
    assert {"b", "k"} <= set(idata.posterior.data_vars)
    assert {"a", "mu"}.isdisjoint(idata.posterior.data_vars)

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
    assert {"b", "k"} <= set(idata.posterior.data_vars)
    assert {"a"}.isdisjoint(idata.posterior.data_vars)


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
            bmb.Formula("y ~ 0 + nl(a + b * x)", nlpars=("b",)),
            "No nonlinear parameter formula or data column",
        ),
        (
            bmb.Formula("y ~ a + x", nlpars=("a", "b")),
            "not used by the expression",
        ),
        (
            bmb.Formula("y ~ 0 + nl(a + unknown)", nlpars=("a",)),
            "No nonlinear parameter formula or data column",
        ),
    ],
)
def test_validation_errors(formula, error):
    with pytest.raises(ValueError, match=error):
        bmb.Model(formula, linear_data())


@pytest.mark.parametrize(
    "additionals",
    [("a ~ 1 + b",), ("a ~ 1", "sigma ~ 1 + b")],
)
def test_data_name_collision_is_reported_before_parameter_dependency(additionals):
    data = linear_data()
    data["b"] = 1.0
    formula = bmb.Formula("y ~ a + b * x", *additionals, nlpars=("a", "b"))

    with pytest.raises(ValueError, match=r"must not also be data columns: \['b'\]"):
        bmb.Model(formula, data)


def test_malformed_expression():
    formula = bmb.Formula("y ~ a +", nlpars=("a",))
    with pytest.raises(ValueError, match="Malformed nonlinear expression"):
        bmb.Model(formula, linear_data())


def test_nonnumeric_expression_data():
    data = linear_data()
    data["label"] = "a"
    formula = bmb.Formula("y ~ 0 + nl(a + label)", nlpars=("a",))

    with pytest.raises(ValueError, match="Nonlinear expression data must be numeric"):
        bmb.Model(formula, data)


def test_nested_priors_are_assigned():
    model = bmb.Model(exponential_formula(), linear_data(), priors=exponential_priors())

    prior = model.parameters["mu"].nonlinear_coefficients["a"].terms["z"].prior
    assert prior.name == "Normal"
    assert prior.args == {"mu": 0, "sigma": 2}


def test_set_priors_uses_nested_parameter_names():
    model = bmb.Model(exponential_formula(), linear_data(), priors=exponential_priors())
    updated = bmb.Prior("Normal", mu=1, sigma=0.25)

    model.set_priors({"a": {"z": updated}})

    prior = model.parameters["mu"].nonlinear_coefficients["a"].terms["z"].prior
    assert prior.name == "Normal"
    assert prior.args == {"mu": 1, "sigma": 0.25}


def test_nonlinear_coefficient_alias():
    model = bmb.Model(exponential_formula(), linear_data(), priors=exponential_priors())

    model.set_alias({"a": {"Intercept": "a0"}})
    model.build()
    assert "a0" in model.backend.model.named_vars
    assert "a_Intercept" not in model.backend.model.named_vars


@pytest.mark.parametrize("family", ["cumulative", "sratio", "cratio", "acat", "osm"])
def test_nonlinear_ordinal_parent_is_rejected(family):
    formula = bmb.Formula("y ~ rate * x", nlpars=("rate",))

    with pytest.raises(ValueError, match="ordinal predictors remain unsupported"):
        bmb.Model(formula, linear_data(), family=family)


def test_non_proportion_multicolumn_response_is_rejected():
    data = linear_data()
    data["status"] = "none"
    formula = bmb.Formula("censored(y, status) ~ rate * x", nlpars=("rate",))

    with pytest.raises(ValueError, match="one observed response or a proportion response"):
        bmb.Model(formula, data)


def test_dropna_aligns_all_model_inputs():
    data = linear_data(8)
    data.index = [4, 4, 2, 2, 9, 9, 1, 1]
    data["group"] = ["g", "h"] * 4
    data["unused"] = np.nan
    for row, column in enumerate(["y", "x", "z", "group"]):
        data.iloc[row, data.columns.get_loc(column)] = np.nan
    original = data.copy(deep=True)
    formula = bmb.Formula("y ~ a + b * x", "a ~ 1 + z", "b ~ 1 + (1 | group)", nlpars=("a", "b"))
    model = bmb.Model(formula, data, dropna=True)
    model.build()

    expected = data.iloc[4:].copy()
    expected["group"] = expected["group"].astype("category")
    pd.testing.assert_frame_equal(model.data, expected)
    pd.testing.assert_frame_equal(data, original)
    np.testing.assert_array_equal(model.response_term.data, data.y.iloc[4:])
    np.testing.assert_array_equal(model.backend.model["mu__x_data"].get_value(), data.x.iloc[4:])
    coefficient = model.parameters["mu"].nonlinear_coefficients["a"]
    np.testing.assert_array_equal(coefficient.terms["z"].data, data.z.iloc[4:])
    assert_ip_dlogp(model)


@pytest.mark.parametrize("column", ["successes", "attempts", "x"])
def test_dropna_aligns_nonlinear_proportion_inputs(column):
    data = pd.DataFrame(
        {
            "successes": [6.0, 13.0, 18.0],
            "attempts": [59.0, 60.0, 62.0],
            "x": [0.1, 0.2, 0.3],
        }
    )
    data.loc[0, column] = np.nan
    formula = bmb.Formula("prop(successes, attempts) ~ normal_cdf(a + b * x)", nlpars=("a", "b"))

    model = bmb.Model(formula, data, family="binomial", link="identity", dropna=True)
    model.build()

    pd.testing.assert_frame_equal(model.data, data.iloc[1:])
    np.testing.assert_array_equal(model.response_term.data, data.loc[1:, ["successes", "attempts"]])
    np.testing.assert_array_equal(model.backend.model["p__x_data"].get_value(), data.x.iloc[1:])


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
    formula = bmb.Formula("y ~ a * x", "a ~ C(category, Treatment(reference='a'))", nlpars=("a",))
    model = bmb.Model(formula, data)
    model.build()
    assert_ip_dlogp(model)


def test_nonlinear_coefficient_can_depend_on_another_coefficient():
    formula = bmb.Formula("y ~ a + b * x", "a ~ 1 + b", nlpars=("a", "b"))
    model = bmb.Model(formula, linear_data())
    model.build()
    draws = xr.Dataset(
        {
            "b": (("chain", "draw"), [[0.25, -0.5]]),
            "sigma": (("chain", "draw"), [[1.0, 1.0]]),
        }
    )

    with model.backend.model:
        actual = pm.compute_deterministics(draws, var_names=["mu"], progressbar=False)

    x = xr.DataArray(linear_data().x.to_numpy(), dims="__obs__")
    expected = 1 + draws.b + draws.b * x
    np.testing.assert_allclose(actual.mu, expected)


@pytest.mark.parametrize(("rhs", "error"), [("I(b ** 2)", "function 'I'"), ("(1 | b)", "BitOr")])
def test_parameter_dependent_formula_requires_nonlinear_expression_syntax(rhs, error):
    formula = bmb.Formula("y ~ a + b * x", f"a ~ {rhs}", nlpars=("a", "b"))
    with pytest.raises(ValueError, match=error):
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
    formula = bmb.Formula(f"y ~ 0 + nl({expression})", nlpars=("a",))
    with pytest.raises(ValueError, match="Nonlinear|nonlinear|Unsupported"):
        bmb.Model(formula, linear_data())


@pytest.mark.parametrize("name", ["mu", "sigma", "exp", "sin", "normal_cdf", "x"])
def test_reserved_parameter_names_are_rejected(name):
    formula = bmb.Formula(f"y ~ {name}", nlpars=(name,))
    with pytest.raises(ValueError, match="names must not"):
        bmb.Model(formula, linear_data())


@pytest.mark.parametrize("auto_scale", [False, True])
@pytest.mark.parametrize("prior", [None, bmb.Prior("Normal", mu=0.3, sigma=1.2)])
def test_marginal_coefficient_matches_explicit_intercept_density(auto_scale, prior):
    models = [
        bmb.Model(
            bmb.Formula("y ~ a * x", *additionals, nlpars=("a",)),
            linear_data(),
            priors={"a": prior, "sigma": 1.0},
            auto_scale=auto_scale,
        )
        for additionals, prior in [
            ((), prior),
            (("a ~ 1",), prior),
        ]
    ]
    for model in models:
        model.build()
    for value in (-0.5, 0.0, 1.3):
        points = [{"a": value}, {"a": value}]
        logps = [model.backend.model.compile_logp()(point) for model, point in zip(models, points)]
        gradients = [
            model.backend.model.compile_dlogp()(point) for model, point in zip(models, points)
        ]
        np.testing.assert_allclose(*logps)
        np.testing.assert_allclose(*gradients)


@pytest.mark.parametrize("prior", [normal_prior(), 0.5])
@pytest.mark.parametrize("additionals", [(), ("a ~ 1",), ("a ~ 0 + 1",)])
def test_marginal_coefficient_prior_alias_and_output(prior, additionals, mock_pymc_sample):
    model = bmb.Model(
        bmb.Formula("y ~ a * x", *additionals, nlpars=("a",)), linear_data(), priors={"a": prior}
    )
    coefficient = model.parameters["mu"].nonlinear_coefficients["a"]
    assert isinstance(coefficient, MarginalCoefficient)
    model.set_alias({"a": "slope"})
    model.build()
    assert "slope" in str(model)
    assert "a_Intercept" not in model.backend.model.named_vars
    assert model.backend.model["slope"].ndim == 0
    prior_draws = model.backend.prior_predictive(draws=3, prior_only=True, random_seed=123)
    assert "slope" in prior_draws.prior
    posterior = model.fit(draws=3, chains=1, random_seed=123)
    assert "slope" in posterior.posterior
    model.set_priors({"a": bmb.Prior("Normal", mu=0.7, sigma=0.2)})
    assert coefficient.prior.args == {"mu": 0.7, "sigma": 0.2}
    model.build()
    assert "slope" in model.backend.model.named_vars


def test_marginal_coefficient_rejects_term_prior_dictionary():
    with pytest.raises(ValueError, match="not a valid prior"):
        bmb.Model(
            bmb.Formula("y ~ a * x", nlpars=("a",)),
            linear_data(),
            priors={"a": {"Intercept": normal_prior()}},
        )


@pytest.mark.parametrize(
    "source, additive, expression",
    [
        ("nl(exp(b * x) + x) + z", "z", "exp(b * x) + x"),
        ("z - exp(b * x) + (1 + x | group)", "z + (1 + x | group)", "-exp(b * x)"),
        ("nl(exp(b * x) - 1) + z", "z", "exp(b * x) - 1"),
        ("nl(exp(b * x) + x) - log1p(b ** 2) + z", "z", "exp(b * x) + x - log1p(b**2)"),
        ("(exp(b * x) + x) + z", "x + z", "exp(b * x)"),
        ("0 + nl(b * x) + x:z", "0 + x:z", "b * x"),
        ("nl(b * x) - 1", "-1", "b * x"),
        ("nl(x + 1)", "1", "x + 1"),
        ("a + b * x", None, "a + b * x"),
        ("a + b * x + 2", None, "a + b * x + 2"),
        ("a + b * x - 0.5", None, "a + b * x - 0.5"),
        ("a * x + 1e-3", None, "a * x + 1e-3"),
        ("nl(a * x + 1e-3) + z", "z", "a * x + 1e-3"),
        ("(nl(b * x) + (1 | group))", "(1 | group)", "b * x"),
    ],
)
def test_split_additive_and_nonlinear_predictor(source, additive, expression):
    actual_additive, actual_expression = split_predictor(source, ("a", "b"))
    if additive is None:
        assert actual_additive is None
    else:
        assert actual_additive.replace(" ", "") == additive.replace(" ", "")
    values = {"a": 0.5, "b": 0.3, "x": pt.as_tensor_variable([0.2, 0.8])}
    actual = evaluate_expression(actual_expression.root, values).eval()
    expected = evaluate_expression(NonlinearExpression.parse(expression).root, values).eval()
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "source", ["nl()", "nl(a, x)", "nl(a=x)", "2 * nl(a*x)", "nl(a*x):z", "nl(nl(a*x))"]
)
def test_invalid_nl_wrapper(source):
    with pytest.raises(ValueError):
        split_predictor(source, ("a",))


def test_nl_wrapper_accepts_underscore_marginal_coefficient():
    model = bmb.Model(bmb.Formula("y ~ 0 + nl(_a * x)", nlpars=("_a",)), linear_data())
    model.build()
    assert "_a" in model.backend.model.named_vars


def vector_parent_data():
    data = linear_data(6)
    data["y"] = pd.Categorical(
        ["middle", "last", "first"] * 2, categories=["middle", "first", "last"], ordered=True
    )
    data["c1"] = [1, 2, 0, 3, 1, 2]
    data["c2"] = [2, 0, 3, 1, 2, 1]
    data["c3"] = [1, 2, 1, 0, 1, 1]
    data["n"] = data[["c1", "c2", "c3"]].sum(axis=1)
    data["group"] = ["g", "h"] * 3
    return data


@pytest.mark.parametrize("family", ["categorical", "multinomial", "dirichlet_multinomial"])
@pytest.mark.parametrize("mixed", [False, True])
def test_nonlinear_vector_parent_matches_pymc(family, mixed):
    data = vector_parent_data()
    response = "y" if family == "categorical" else "counts(c1, c2, c3, n=n)"
    parent = "a" if family == "dirichlet_multinomial" else "p"
    size = 3 if family == "dirichlet_multinomial" else 2
    rhs = "z + nl(shared + rate * x)" if mixed else "shared + rate * x"
    formula = bmb.Formula(
        f"{response} ~ {rhs}", "rate ~ 1 + z", "shared ~ 1", nlpars=("rate", "shared")
    )
    priors = {"shared": normal_prior(), "rate": {"Intercept": normal_prior(), "z": normal_prior()}}
    if mixed:
        priors[parent] = {"Intercept": normal_prior(), "z": normal_prior()}
    model = bmb.Model(formula, data, family=family, priors=priors, center_predictors=False)
    model.build()
    backend = model.backend.model
    dim = next(iter(backend.__bambi_attrs__["response_coords"]))
    coord_key = "response_coords" if size == 3 else "response_coords_reduced"
    coef_dim = next(iter(backend.__bambi_attrs__[coord_key]))
    assert backend.named_vars_to_dims["rate"] == ("__obs__", coef_dim)
    assert backend.named_vars_to_dims[parent] == ("__obs__", dim)
    assert backend.named_vars_to_dims["shared"] == ()
    levels = ["middle", "first", "last"] if family == "categorical" else ["c1", "c2", "c3"]
    assert list(backend.coords[dim]) == levels
    if size == 2:
        assert list(backend.coords[coef_dim]) == levels[1:]
    point = {
        "shared": np.array(0.4),
        "rate_Intercept": np.linspace(-0.3, 0.5, size),
        "rate_z": np.linspace(0.2, -0.1, size),
    }
    with pm.Model() as direct:
        shared = pm.Normal("shared", 0, 2)
        intercept = pm.Normal("rate_Intercept", 0, 2, shape=size)
        slope = pm.Normal("rate_z", 0, 2, shape=size)
        eta = shared + (intercept + data.z.to_numpy()[:, None] * slope) * data.x.to_numpy()[:, None]
        if mixed:
            base = pm.Normal("Intercept", 0, 2, shape=size)
            linear = pm.Normal("z", 0, 2, shape=size)
            eta += base + data.z.to_numpy()[:, None] * linear
            point.update(Intercept=np.linspace(0.1, 0.3, size), z=np.linspace(-0.2, 0.1, size))
        if family == "dirichlet_multinomial":
            value = pt.exp(eta)
            pm.DirichletMultinomial(
                "observed", n=data.n, a=value, observed=data[["c1", "c2", "c3"]]
            )
        else:
            value = pt.special.softmax(
                pt.concatenate([pt.zeros((len(data), 1)), eta], axis=1), axis=1
            )
            if family == "categorical":
                pm.Categorical("observed", p=value, observed=data.y.cat.codes)
            else:
                pm.Multinomial("observed", n=data.n, p=value, observed=data[["c1", "c2", "c3"]])
        pm.Deterministic("value", value)
    np.testing.assert_allclose(backend.compile_logp()(point), direct.compile_logp()(point))
    actual = backend.compile_fn(backend[parent], inputs=backend.free_RVs)(point)
    expected = direct.compile_fn(direct["value"], inputs=direct.free_RVs)(point)
    np.testing.assert_allclose(actual, expected)
    if parent == "p":
        np.testing.assert_allclose(actual.sum(axis=-1), 1)


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("family", ["categorical", "multinomial", "dirichlet_multinomial"])
@pytest.mark.parametrize("expression", ["z + nl(0.7)", "shared * x", "rate * x + shared"])
def test_nonlinear_vector_prediction(family, expression):
    data = vector_parent_data()
    response = "y" if family == "categorical" else "counts(c1, c2, c3, n=n)"
    names = tuple(name for name in ("shared", "rate") if name in expression)
    additionals = ("rate ~ 1 + z",) if "rate" in names else ()
    model = bmb.Model(
        bmb.Formula(f"{response} ~ {expression}", *additionals, nlpars=names), data, family=family
    )
    model.set_alias({model.family.likelihood.parent: {model.family.likelihood.parent: "parameter"}})
    idata = model.fit(draws=3, chains=1, random_seed=101)
    for count in (len(data), 2, 1):
        new_data = None if count == len(data) else data.iloc[:count].copy()
        result = model.predict(
            idata, data=new_data, kind="response", inplace=False, random_seed=102
        )
        group = result.posterior if new_data is None else result.predictions
        values = group["parameter"].values
        assert values.shape == (1, 3, count, 3)
        if family != "dirichlet_multinomial":
            np.testing.assert_allclose(values.sum(axis=-1), 1)
        observed = result.posterior_predictive if new_data is None else result.predictions
        draws = observed[model.response_term.label].values
        if family == "categorical":
            assert draws.shape == (1, 3, count)
            assert np.isin(draws, [0, 1, 2]).all()
        else:
            np.testing.assert_array_equal(draws.sum(axis=-1), np.full((1, 3, count), 4))
        likelihood = model.compute_log_likelihood(idata, data=new_data, inplace=False)
        assert likelihood.log_likelihood[model.response_term.label].shape == (1, 3, count)
        assert np.isfinite(likelihood.log_likelihood[model.response_term.label]).all()


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("sparse_dot", [False, True])
@pytest.mark.parametrize("family", ["categorical", "multinomial", "dirichlet_multinomial"])
def test_nonlinear_vector_group_prediction(monkeypatch, sparse_dot, family):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    data = vector_parent_data()
    response = "y" if family == "categorical" else "counts(c1, c2, c3, n=n)"
    parent = "a" if family == "dirichlet_multinomial" else "p"
    dim = "y_dim_reduced" if family == "categorical" else "c1_c2_c3_dim_reduced"
    if family == "dirichlet_multinomial":
        dim = "c1_c2_c3_dim"
    model = bmb.Model(
        bmb.Formula(f"{response} ~ rate * x", "rate ~ 0 + z + (1 | group)", nlpars=("rate",)),
        data,
        family=family,
        center_predictors=False,
    )
    model.set_alias({"rate": {"rate": "r", "1|group": "group_effect", "z": "slope"}})
    idata = model.fit(draws=3, chains=1, random_seed=103)
    assert idata.posterior["group_effect"].dims == ("chain", "draw", "group_dim", dim)
    for groups in (["g", "h"], ["g"], ["new", "new"]):
        new_data = pd.DataFrame(
            {"x": [0.3] * len(groups), "z": [0.2] * len(groups), "group": groups, "n": 4}
        )
        result = model.predict(idata, data=new_data, inplace=False, random_seed=104)
        values = result.predictions[parent].values
        assert values.shape == (1, 3, len(groups), 3)
        if parent == "p":
            np.testing.assert_allclose(values.sum(axis=-1), 1)
        if groups[0] == "new":
            np.testing.assert_allclose(values[:, :, 0], values[:, :, 1])
        else:
            effects = idata.posterior.group_effect.sel(group_dim=groups).values
            slope = idata.posterior.slope.values[:, :, None, :]
            eta = (effects + 0.2 * slope) * 0.3
            weights = np.exp(np.concatenate([np.zeros((*eta.shape[:-1], 1)), eta], axis=-1))
            expected = weights / weights.sum(axis=-1, keepdims=True)
            if parent == "a":
                expected = np.exp(eta)
            np.testing.assert_allclose(values, expected)
        population = model.predict(
            idata, data=new_data, include_group_specific=False, inplace=False
        )
        eta = idata.posterior.slope.values[:, :, None, :] * 0.2 * 0.3
        weights = np.exp(np.concatenate([np.zeros((*eta.shape[:-1], 1)), eta], axis=-1))
        expected = weights / weights.sum(axis=-1, keepdims=True)
        if parent == "a":
            expected = np.exp(eta)
        np.testing.assert_allclose(
            population.predictions[parent], np.broadcast_to(expected, values.shape)
        )


@pytest.mark.usefixtures("mock_pymc_sample")
def test_nonlinear_categorical_interpret():
    from bambi.interpret.effects import comparisons, predictions, slopes

    model = bmb.Model(
        bmb.Formula("y ~ shared * x + rate", "rate ~ 0 + z", nlpars=("shared", "rate")),
        vector_parent_data(),
        family="categorical",
    )
    idata = model.fit(draws=5, chains=2, random_seed=105)
    conditional = {"x": [-0.2, 0.3], "z": [0.1]}
    result = predictions(model, idata, conditional=conditional).summary
    assert len(result) == 6
    assert set(result.y_dim) == {"middle", "first", "last"}
    np.testing.assert_allclose(result.groupby("x").estimate.sum(), 1)
    for function, kwargs in (
        (comparisons, {"contrast": {"x": [-0.2, 0.3]}}),
        (slopes, {"wrt": "x"}),
    ):
        result = function(model, idata, conditional={"z": [0.1]}, **kwargs).summary
        assert set(result.y_dim) == {"middle", "first", "last"}
        np.testing.assert_allclose(result.estimate.sum(), 0, atol=1e-8)


@pytest.mark.parametrize("family", ["categorical", "multinomial", "dirichlet_multinomial"])
def test_nonlinear_vector_constant_expression(family):
    data = vector_parent_data()
    response = "y" if family == "categorical" else "counts(c1, c2, c3, n=n)"
    model = bmb.Model(bmb.Formula(f"{response} ~ 0 + nl(0.7)"), data, family=family)
    model.build()
    backend = model.backend.model
    parent = model.family.likelihood.parent
    expected = np.exp([0, 0.7, 0.7])
    expected /= expected.sum()
    if family == "dirichlet_multinomial":
        expected = np.full(3, np.exp(0.7))
    for count in (len(data), 2, 1):
        if count != len(data):
            replacements, coords, _ = model.backend._build_new_data(data.iloc[:count], "prediction")
            pm.set_data(replacements, coords=coords, model=backend)
        np.testing.assert_allclose(backend[parent].eval(), np.broadcast_to(expected, (count, 3)))


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("categories", [2, 3])
def test_nonlinear_vector_dependencies_and_offsets(categories):
    data = vector_parent_data()
    if categories == 2:
        data = data[data.y != "last"].copy()
        data["y"] = data.y.cat.remove_unused_categories()
    formula = bmb.Formula(
        "y ~ offset(x) + nl(curve)",
        "curve ~ rate * x + shared",
        "rate ~ 0 + z",
        nlpars=("curve", "rate", "shared"),
    )
    model = bmb.Model(formula, data, family="categorical", center_predictors=False)
    idata = model.fit(draws=3, chains=1, random_seed=106)
    for count in (len(data), 1):
        new_data = data.iloc[:count]
        result = model.predict(idata, data=new_data, inplace=False)
        eta = (
            idata.posterior.rate_z.values[:, :, None, :]
            * new_data.z.to_numpy()[None, None, :, None]
            * new_data.x.to_numpy()[None, None, :, None]
            + idata.posterior.shared.values[:, :, None, None]
            + idata.posterior.Intercept.values[:, :, None, :]
            + new_data.x.to_numpy()[None, None, :, None]
        )
        weights = np.exp(np.concatenate([np.zeros((*eta.shape[:-1], 1)), eta], axis=-1))
        np.testing.assert_allclose(
            result.predictions.p, weights / weights.sum(axis=-1, keepdims=True)
        )


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("family", ["multinomial", "dirichlet_multinomial"])
def test_nonlinear_count_interpret(family):
    from bambi.interpret.effects import predictions

    model = bmb.Model(
        bmb.Formula(
            "counts(c1, c2, c3, n=n) ~ shared * x + rate",
            "rate ~ 0 + z",
            nlpars=("shared", "rate"),
        ),
        vector_parent_data(),
        family=family,
    )
    idata = model.fit(draws=5, chains=2, random_seed=107)
    conditional = {"x": [-0.2, 0.3], "z": [0.1], "n": [4]}
    summary = predictions(model, idata, conditional=conditional).summary
    assert len(summary) == 6
    assert set(summary.c1_c2_c3_dim) == {"c1", "c2", "c3"}
    data = pd.DataFrame({"x": [-0.2, 0.3], "z": [0.1, 0.1], "n": [4, 4]})
    result = model.predict(idata, data=data, inplace=False)
    expected = result.predictions[model.family.likelihood.parent].mean(("chain", "draw"))
    actual = summary.pivot(index="x", columns="c1_c2_c3_dim", values="estimate")
    np.testing.assert_allclose(actual[["c1", "c2", "c3"]], expected)
