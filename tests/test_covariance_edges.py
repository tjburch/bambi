"""Deterministic boundary checks for covariance formula and prior handling."""

from types import SimpleNamespace

import bambi as bmb
import formulae as fm
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import pytensor.tensor as pt

from bambi.backend.pymc.terms.structured import build_structured_correlation


@pytest.fixture
def panel():
    return pd.DataFrame(
        {
            "y": [0, 1, 1, 0, 0, 1],
            "year": [0, 1, 2, 0, 1, 2],
            "subject": ["a", "a", "a", "b", "b", "b"],
        }
    )


def test_unstructured_namespace_levels(panel):
    wrapper = "us(0 + C(year, levels=visits) | subject)"
    model = bmb.Model(
        f"y ~ {wrapper}",
        panel,
        family="bernoulli",
        extra_namespace={"visits": [2, 1, 0]},
    )
    block = next(iter(model._covariance_blocks.values()))
    assert block.variables == ("year",)
    design, coordinates = block.prediction_design(panel)
    assert design.shape == (6, 3)
    np.testing.assert_array_equal(design[2], [1, 0, 0])
    assert coordinates == ()
    with pytest.raises(ValueError, match="ordered categorical column"):
        block.prediction_design(panel.iloc[:1])


def test_unstructured_transformed_categories(panel):
    wrapper = "us(0 + C(bucket(year)) | subject)"
    model = bmb.Model(
        f"y ~ {wrapper}",
        panel,
        family="bernoulli",
        extra_namespace={"bucket": lambda values: values // 2},
    )
    block = next(iter(model._covariance_blocks.values()))
    actual, _ = block.prediction_design(panel.iloc[:1].assign(year=3))
    np.testing.assert_array_equal(actual, [[0, 1]])
    previous = fm.config["EVAL_UNSEEN_CATEGORIES"]
    try:
        fm.config["EVAL_UNSEEN_CATEGORIES"] = "silent"
        with pytest.raises(ValueError, match="declared coefficient levels"):
            block.prediction_design(panel.iloc[:1].assign(year=4))
    finally:
        fm.config["EVAL_UNSEEN_CATEGORIES"] = previous


def test_unstructured_declared_ordered_categories(panel):
    panel = panel.assign(year=pd.Categorical(panel.year, categories=[0, 1, 2, 3], ordered=True))
    model = bmb.Model("y ~ us(0 + year | subject)", panel, family="bernoulli")
    block = next(iter(model._covariance_blocks.values()))
    actual, _ = block.prediction_design(panel.iloc[:1].assign(year=3))
    np.testing.assert_array_equal(actual, [[0, 0, 0, 1]])


def test_unstructured_data_shadows_namespace(panel):
    model = bmb.Model(
        "y ~ us(1 + year | subject)",
        panel,
        family="bernoulli",
        extra_namespace={"year": 100},
    )
    block = next(iter(model._covariance_blocks.values()))
    np.testing.assert_array_equal(block.design.design_matrix[:, 1], panel.year)


@pytest.mark.parametrize("eta", [0, -1, True, "2", np.nan, np.inf, [2]])
@pytest.mark.parametrize("size", [1, 2])
def test_invalid_lkj_eta(eta, size):
    term = SimpleNamespace(
        block=SimpleNamespace(kind="us", coordinates=np.arange(size)),
        prior={"eta": eta},
        label="coefficient",
    )
    with pytest.raises(ValueError, match="fixed positive scalar"):
        build_structured_correlation(term, pm.Model())


@pytest.mark.parametrize(
    "kind,settings",
    [
        ("ar1", {"rho": -1}),
        ("ar1", {"rho": 1}),
        ("cs", {"rho": -0.5}),
        ("cs", {"rho": 1}),
        ("ou", {"decay": 0}),
        ("toep", {"partial": [0, 1]}),
    ],
)
def test_singular_fixed_parameters_rejected(kind, settings):
    term = SimpleNamespace(
        block=SimpleNamespace(kind=kind, coordinates=np.arange(3), max_lag=2),
        prior=settings,
        label="coefficient",
        hyperprior_alias={},
    )
    with pytest.raises(ValueError):
        build_structured_correlation(term, pm.Model())


def test_distributional_wrapper_namespace(panel):
    model = bmb.Model(
        bmb.Formula("y ~ 1", "sigma ~ us(0 + C(year, levels=visits) | subject)"),
        panel,
        extra_namespace={"visits": [2, 1, 0]},
    )
    block = next(iter(model._covariance_blocks.values()))
    assert block.variables == ("year",)
    assert block.design.design_matrix.shape == (6, 3)


def test_bulk_group_prior_keeps_structured_prior(panel):
    wrapper = "ar1(0 + year | subject)"
    model = bmb.Model(
        f"y ~ (1 | subject) + {wrapper}",
        panel,
        family="bernoulli",
        priors={wrapper: {"sd": 0.7, "rho": -0.4}},
    )
    ordinary_prior = bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1))
    model.set_priors(group_specific=ordinary_prior)
    terms = model.parameters["p"].group_specific_terms
    assert terms[wrapper].prior == {"sd": 0.7, "rho": -0.4}
    assert terms["1|subject"].prior == ordinary_prior


@pytest.mark.parametrize("rho", [0.0, -0.4, 0.7])
def test_ar1_gradient_at_zero_and_signed_rho(rho):
    term = SimpleNamespace(
        block=SimpleNamespace(kind="ar1", coordinates=np.arange(4)),
        prior={"rho": bmb.Prior("Normal", mu=0, sigma=0.5)},
        label="coefficient",
        hyperprior_alias={},
    )
    model = pm.Model()
    corr = build_structured_correlation(term, model)
    parameter = model["coefficient_rho"]
    gradient = pt.stack([pt.grad(corr[0, lag], parameter) for lag in range(4)])
    evaluate = model.compile_fn(gradient, inputs=[parameter], point_fn=False)
    np.testing.assert_allclose(evaluate(rho), [0, 1, 2 * rho, 3 * rho**2])


@pytest.mark.parametrize(
    "wrapper,prior,correlation,scales",
    [
        (
            "ar1(0 + year | subject)",
            {"rho": -0.4, "sd": 0.7},
            [[1, -0.4, 0.16], [-0.4, 1, -0.4], [0.16, -0.4, 1]],
            [0.7] * 3,
        ),
        (
            "ou(0 + year | subject)",
            {"decay": np.log(2), "sd": 0.7},
            [[1, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 1]],
            [0.7] * 3,
        ),
        (
            "cs(0 + year | subject)",
            {"rho": -0.3, "sd": 0.7},
            [[1, -0.3, -0.3], [-0.3, 1, -0.3], [-0.3, -0.3, 1]],
            [0.7] * 3,
        ),
        (
            "toep(0 + year | subject, max_lag=2)",
            {"partial": [0.2, -0.3], "sd": 0.7},
            [[1, 0.2, -0.248], [0.2, 1, 0.2], [-0.248, 0.2, 1]],
            [0.7] * 3,
        ),
        (
            "us(1 + year | subject)",
            {"correlation": [[1, 0.4], [0.4, 1]], "sd": [0.7, 0.3]},
            [[1, 0.4], [0.4, 1]],
            [0.7, 0.3],
        ),
    ],
)
def test_centered_coefficient_density(panel, wrapper, prior, correlation, scales):
    model = bmb.Model(
        f"y ~ {wrapper}", panel, family="bernoulli", priors={wrapper: prior}, noncentered=False
    )
    model.build()
    values = np.linspace(-0.4, 0.8, 2 * len(scales)).reshape(2, -1)
    covariance = np.asarray(scales)[:, None] * correlation * np.asarray(scales)[None, :]
    expected = -0.5 * (
        len(scales) * np.log(2 * np.pi)
        + np.linalg.slogdet(covariance)[1]
        + np.sum(values * np.linalg.solve(covariance, values.T).T, axis=1)
    )
    actual = pm.logp(model.backend.model[wrapper], values).eval()
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)
