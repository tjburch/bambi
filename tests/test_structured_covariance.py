"""Small construction and graph checks for structured group effects."""

from itertools import product

import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from scipy.special import expit

from bambi.covariance import encode_groups, lower_covariance_formula
from bambi.covariance_math import correlation_matrix, conditional_gaussian
from bambi.backend.pymc.terms.structured import prediction_coefficients
import pytensor.tensor as pt


@pytest.fixture
def crossed():
    rows = list(product(["s0", "s1"], ["a", "b"], ["u", "v"], [0, 1, 3]))
    data = pd.DataFrame(rows, columns=["subject", "condition", "context", "year"])
    data["x"] = np.linspace(-1, 1, len(data))
    data["y"] = np.arange(len(data)) % 2
    data["n"] = 2 + np.arange(len(data)) % 3
    return data


def test_group_labels_do_not_collide():
    labels = encode_groups(pd.Series(["a:b", "a"]), pd.Series(["c", "b:c"]))
    assert labels[0] != labels[1]


@pytest.mark.parametrize(
    "wrapper",
    [
        "ar1(year | subject)",
        "ar1(0 + year)",
        "ar1(0 + year | subject, bad=1)",
        "toep(0 + year | subject)",
        "toep(0 + year | subject, max_lag=-1)",
    ],
)
def test_invalid_wrapper(wrapper):
    with pytest.raises(ValueError):
        lower_covariance_formula(f"y ~ {wrapper}", {})


@pytest.mark.parametrize("family,response", [("bernoulli", "y"), ("binomial", "proportion(y, n)")])
@pytest.mark.parametrize("sparse", [False, True])
def test_four_blocks(crossed, family, response, sparse):
    factors = ["subject", "subject:condition", "subject:context", "subject:condition:context"]
    wrappers = [f"ar1(0 + year | {factor})" for factor in factors]
    previous = bmb.config["SPARSE_DOT"]
    try:
        bmb.config["SPARSE_DOT"] = sparse
        model = bmb.Model(
            f"{response} ~ x + " + " + ".join(wrappers),
            crossed,
            family=family,
            priors={wrapper: {"sd": 0.7, "rho": -0.4} for wrapper in wrappers},
        )
        assert set(model.parameters["p"].group_specific_terms) == set(wrappers)
        assert [
            len(term.groups) for term in model.parameters["p"].group_specific_terms.values()
        ] == [2, 4, 4, 8]
        model.build()
        backend = model.backend.model
        assert np.isfinite(backend.compile_logp()(backend.initial_point()))
        for wrapper in wrappers:
            chol = backend[f"{wrapper}_cholesky"].eval()
            expected = 0.7**2 * (-0.4) ** np.abs(np.array([0, 1, 3])[:, None] - [0, 1, 3])
            np.testing.assert_allclose(chol @ chol.T, expected)
    finally:
        bmb.config["SPARSE_DOT"] = previous


@pytest.mark.parametrize(
    "wrapper,prior",
    [
        ("ou(0 + year | subject)", {"sd": 0.6, "decay": 0.3}),
        ("cs(0 + year | subject)", {"sd": 0.6, "rho": -0.2}),
        ("toep(0 + year | subject, max_lag=3)", {"sd": 0.6, "partial": [0.2, -0.1, 0.1]}),
        ("us(1 + x | subject)", {"sd": [0.6, 0.3], "correlation": [[1, 0.2], [0.2, 1]]}),
        ("us(0 + C(year) | subject)", {"sd": 0.6}),
    ],
)
def test_structure_builds(crossed, wrapper, prior):
    model = bmb.Model(f"y ~ x + {wrapper}", crossed, family="bernoulli", priors={wrapper: prior})
    model.build()
    backend = model.backend.model
    assert np.isfinite(backend.compile_logp()(backend.initial_point()))


def test_estimated_ar_prior_builds(crossed):
    model = bmb.Model("y ~ ar1(0 + year | subject)", crossed, family="bernoulli")
    model.build()
    backend = model.backend.model
    assert np.isfinite(backend.compile_logp()(backend.initial_point()))
    assert np.isfinite(backend.compile_dlogp()(backend.initial_point())).all()


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("kind", ["ar1", "ou", "toep"])
def test_joint_prediction(crossed, sparse, kind):
    suffix = ", max_lag=5" if kind == "toep" else ""
    wrapper = f"{kind}(0 + year | subject:condition:context{suffix})"
    previous = bmb.config["SPARSE_DOT"]
    try:
        bmb.config["SPARSE_DOT"] = sparse
        model = bmb.Model(f"y ~ {wrapper}", crossed, family="bernoulli")
        model.build()
        with model.backend.model:
            prior = pm.sample_prior_predictive(draws=4, random_seed=137)
        trace = xr.DataTree.from_dict({"posterior": prior["prior"].to_dataset()})
        target = crossed.iloc[[0, 0, 0, 0, 0]].copy()
        target["year"] = [2, 2, 4, 4, 0]
        target.iloc[3, target.columns.get_loc("subject")] = "new"
        model.predict(trace, data=target, random_seed=163)
        probability = trace["predictions"]["p"].values
        np.testing.assert_allclose(probability[..., 0], probability[..., 1])
        assert np.isfinite(probability).all()
        assert not np.allclose(probability[..., 2], probability[..., 3])
        expected = expit(
            trace["posterior"]["Intercept"].values + trace["posterior"][wrapper].values[..., 0, 0]
        )
        np.testing.assert_allclose(probability[..., 4], expected)
        model.predict(trace, data=crossed.iloc[:2], random_seed=163)
    finally:
        bmb.config["SPARSE_DOT"] = previous


@pytest.mark.parametrize("sparse", [False, True])
def test_alias_prior_update_and_exclusion(crossed, sparse):
    wrapper = "ar1(0 + year | subject)"
    previous = bmb.config["SPARSE_DOT"]
    try:
        bmb.config["SPARSE_DOT"] = sparse
        model = bmb.Model(f"y ~ x + (1 | condition) + {wrapper}", crossed, family="bernoulli")
        model.set_priors({wrapper: {"sd": 0.5, "rho": 0.2}})
        model.set_alias({wrapper: "trajectory"})
        model.build()
        assert "trajectory" in model.backend.model
        with model.backend.model:
            prior = pm.sample_prior_predictive(draws=3, random_seed=181)
        trace = xr.DataTree.from_dict({"posterior": prior["prior"].to_dataset()})
        model.predict(trace, data=crossed.iloc[:2][["x"]], include_group_specific=False)
        expected = expit(
            trace["posterior"]["Intercept"].values[..., None]
            + trace["posterior"]["x"].values[..., None] * crossed.x.values[:2]
        )
        np.testing.assert_allclose(trace["predictions"]["p"].values, expected)
    finally:
        bmb.config["SPARSE_DOT"] = previous


def test_unstructured_unknown_numeric_category_rejected(crossed):
    wrapper = "us(0 + C(year) | subject)"
    model = bmb.Model(f"y ~ {wrapper}", crossed, family="bernoulli")
    term = model.parameters["p"].terms[wrapper]
    with pytest.raises(ValueError, match="declared coefficient"):
        term.block.prediction_design(crossed.assign(year=100))


def test_shared_missing_data(crossed):
    data = crossed.copy()
    data.loc[0, "x"] = np.nan
    data.loc[1, "year"] = np.nan
    model = bmb.Model(bmb.Formula("y ~ x", "sigma ~ ar1(0 + year | subject)"), data, dropna=True)
    assert len(model.data) == len(data) - 2
    model.build()


@pytest.mark.parametrize("parameter", ["sd", "rho"])
def test_scalar_hyperprior_contract(crossed, parameter):
    wrapper = "ar1(0 + year | subject)"
    model = bmb.Model(
        f"y ~ {wrapper}",
        crossed,
        family="bernoulli",
        priors={wrapper: {parameter: bmb.Prior("Normal", mu=[0.1, 0.2, 0.3], sigma=0.1)}},
    )
    with pytest.raises(ValueError, match="scalar prior"):
        model.build()


@pytest.mark.parametrize("sparse", [False, True])
def test_novel_three_way_group_keeps_lower_order_effects(crossed, sparse):
    factors = ["subject", "subject:condition", "subject:context", "subject:condition:context"]
    wrappers = [f"ar1(0 + year | {factor})" for factor in factors]
    heldout = (crossed.subject == "s0") & (crossed.condition == "a") & (crossed.context == "u")
    previous = bmb.config["SPARSE_DOT"]
    try:
        bmb.config["SPARSE_DOT"] = sparse
        model = bmb.Model(
            "y ~ " + " + ".join(wrappers),
            crossed.loc[~heldout],
            family="bernoulli",
            priors={wrapper: {"sd": 0.5, "rho": 0.4} for wrapper in wrappers},
        )
        model.build()
        target = crossed.loc[heldout].iloc[[0, 0]]
        _, _, plans = model.backend._build_new_data(target, "prediction", "response_params")
        assert [len(plan.groups_new) for plan in plans] == [0, 0, 0, 1]
        with model.backend.model:
            prior = pm.sample_prior_predictive(draws=4, random_seed=193)
        trace = xr.DataTree.from_dict({"posterior": prior["prior"].to_dataset()})
        model.predict(trace, data=target, random_seed=197)
        values = trace["predictions"]["p"].values
        np.testing.assert_allclose(values[..., 0], values[..., 1])
    finally:
        bmb.config["SPARSE_DOT"] = previous


@pytest.mark.parametrize("family", ["gaussian", "poisson", "negativebinomial"])
@pytest.mark.parametrize("noncentered", [True, False])
def test_scalar_families_and_parameterizations(crossed, family, noncentered):
    model = bmb.Model(
        "y ~ 0 + ar1(0 + year | subject)",
        crossed,
        family=family,
        noncentered=noncentered,
    )
    model.build()
    backend = model.backend.model
    assert np.isfinite(backend.compile_logp()(backend.initial_point()))
    assert np.isfinite(backend.compile_dlogp()(backend.initial_point())).all()


def test_duplicate_and_nonadditive_wrappers():
    for formula in [
        "y ~ ar1(0 + year | subject) + ar1(0 + year | subject)",
        "y ~ x * ar1(0 + year | subject)",
        "y ~ I(ar1(0 + year | subject))",
    ]:
        with pytest.raises(ValueError):
            lower_covariance_formula(formula, {})


def test_backend_conditional_moments(crossed):
    wrapper = "ar1(0 + year | subject)"
    model = bmb.Model(
        f"y ~ {wrapper}",
        crossed,
        family="bernoulli",
        priors={wrapper: {"sd": 0.7, "rho": -0.5}},
    )
    model.build()
    term = model.parameters["p"].terms[wrapper]
    fitted = np.array([[-0.2, 0.3, 0.8], [0.4, -0.1, 0.2]])
    coefficients = prediction_coefficients(
        term, pt.as_tensor_variable(fitted), (2, 5), 1, model.backend.model
    )
    draws = pm.draw(coefficients, draws=4000, random_seed=211)
    covariance = 0.7**2 * correlation_matrix("ar1", [0, 1, 3, 2, 5], rho=-0.5)
    mean, conditional = conditional_gaussian(covariance, [0, 1, 2], [3, 4], fitted)
    for group in range(2):
        values = draws[:, group, 3:]
        np.testing.assert_array_less(
            np.abs(values.mean(axis=0) - mean[group]),
            5 * np.sqrt(np.diag(conditional) / len(values)),
        )
        covariance_mcse = np.sqrt(
            (conditional**2 + np.outer(np.diag(conditional), np.diag(conditional))) / len(values)
        )
        np.testing.assert_array_less(np.abs(np.cov(values.T) - conditional), 5 * covariance_mcse)
    np.testing.assert_allclose(draws[:, :2, :3], np.broadcast_to(fitted, (4000, 2, 3)))
    np.testing.assert_array_less(
        np.abs(draws[:, 2].mean(axis=0)), 5 * np.sqrt(np.diag(covariance) / len(draws))
    )


def test_binomial_aggregation_log_likelihood_constant(crossed):
    from scipy.special import gammaln

    wrapper = "ar1(0 + year | subject)"
    data = crossed.iloc[:6].copy()
    data["n"] = 3
    priors = {wrapper: {"sd": 0.7, "rho": 0.4}}
    aggregated = bmb.Model(f"proportion(y, n) ~ {wrapper}", data, family="binomial", priors=priors)
    expanded = data.loc[data.index.repeat(3)].reset_index(drop=True)
    expanded["y"] = np.concatenate([np.r_[np.ones(y), np.zeros(3 - y)] for y in data.y])
    individual = bmb.Model(f"y ~ {wrapper}", expanded, family="bernoulli", priors=priors)
    aggregated.build()
    individual.build()
    total_a = aggregated.backend.model.compile_logp()(aggregated.backend.model.initial_point())
    total_b = individual.backend.model.compile_logp()(individual.backend.model.initial_point())
    constant = np.sum(gammaln(4) - gammaln(data.y + 1) - gammaln(4 - data.y))
    np.testing.assert_allclose(total_a - total_b, constant)


def test_confounded_design_warning(crossed):
    with pytest.warns(RuntimeWarning, match="identical grouping"):
        bmb.Model(
            "y ~ ar1(0 + year | subject) + ar1(0 + year | subject:condition)",
            crossed.assign(condition="a"),
            family="bernoulli",
        )


def test_default_prior_output_omits_derived_matrices(crossed):
    wrapper = "ar1(0 + year | subject)"
    model = bmb.Model(f"y ~ {wrapper}", crossed, family="bernoulli")
    model.build()
    prior = model.prior_predictive(draws=3, random_seed=223)
    assert f"{wrapper}_cholesky" not in prior["prior"]
    trace = xr.DataTree.from_dict({"posterior": prior["prior"].to_dataset()})
    target = crossed.iloc[[0, 0]].copy()
    target["year"] = [0, 2]
    model.predict(trace, data=target, random_seed=227)
    expected = expit(
        trace["posterior"]["Intercept"].values + trace["posterior"][wrapper].values[..., 0, 0]
    )
    np.testing.assert_allclose(trace["predictions"]["p"].values[..., 0], expected)


@pytest.mark.parametrize(
    "wrapper",
    [
        "ar1(0 + year | subject)",
        "ou(0 + year | subject)",
        "cs(0 + year | subject)",
        "toep(0 + year | subject, max_lag=3)",
        "us(1 + x | subject)",
    ],
)
def test_omitted_offsets_prediction_and_log_likelihood(crossed, wrapper):
    model = bmb.Model(f"y ~ {wrapper}", crossed, family="bernoulli")
    model.build()
    prior = model.prior_predictive(draws=3, random_seed=229)
    trace = xr.DataTree.from_dict({"posterior": prior["prior"].to_dataset()})
    expected = trace["posterior"]["p"].copy()
    model.predict(trace)
    np.testing.assert_allclose(trace["posterior"]["p"].values, expected.values)
    model.compute_log_likelihood(trace)
    np.testing.assert_allclose(
        trace["log_likelihood"]["y"].values,
        np.where(crossed.y.to_numpy(), np.log(expected), np.log1p(-expected)),
    )
