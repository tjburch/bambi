"""Structured term designs, inference selection, and prediction integration."""

import json

import bambi as bmb
import numpy as np
import pandas as pd
import pytest

from bambi.structured.inference import resolve_plan

SEED = sum(map(ord, "structured-integration"))


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "y": [0.3, 1.1, -0.2, 0.5, 0.9, -0.4],
            "x": [-1.0, 0.3, 0.8, -0.5, 0.1, 1.0],
            "time": [2, 0, 2, 1, 3, 0],
            "g": ["a", "a", "a", "b", "b", "b"],
            "h": ["z", "w", "z", "w", "z", "w"],
            "visit": pd.Categorical(
                ["a", "b", "a", "a", "b", "a"], categories=["a", "b", "c"], ordered=True
            ),
        }
    )


@pytest.mark.parametrize(
    "wrapper,inner",
    [("ar1", "time"), ("ou", "time"), ("diag", "1+x"), ("cs", "1+x"), ("us", "1+x")],
)
def test_build_prior_and_selection(data, wrapper, inner):
    model = bmb.Model(f"y ~ x + {wrapper}({inner}, by=g)", data)
    model.build()
    assert np.isfinite(model.backend.model.compile_logp()(model.backend.model.initial_point()))
    prior = model.prior_predictive(draws=2, random_seed=SEED)
    term = next(iter(model.parameters["mu"].structured_terms.values()))
    assert term.label in prior.prior
    assert resolve_plan(model, "auto", None)["selected"] == term.qualified_name
    assert resolve_plan(model, "latent", None)["selected"] is None


def test_equivalent_prior_and_alias(data):
    model = bmb.Model(
        'y ~ ar1(time, by="g")', data, priors={"ar1(time, by=g)": {"sd": 0.7, "rho": -0.4}}
    )
    term = next(iter(model.parameters["mu"].structured_terms.values()))
    assert term.prior["rho"] == -0.4
    model.set_alias({'ar1(time, by="g")': "trajectory"})
    assert resolve_plan(model, "marginal", "trajectory")["selected"] == term.qualified_name


@pytest.mark.parametrize("wrapper", ["ar1", "ou", "diag", "cs", "us"])
def test_function_call_nesting_preserves_ordinary_group_terms(data, wrapper):
    inner = "time" if wrapper in ("ar1", "ou") else "1+x"
    model = bmb.Model(f"y ~ (1+x|h) + {wrapper}({inner}, by=g/h)", data)
    component = model.parameters["mu"]
    assert [term.term.groups for term in component.structured_terms.values()] == [
        ("g",),
        ("g", "h"),
    ]
    assert set(component.group_specific_terms) == {"1|h", "x|h"}


@pytest.mark.parametrize("wrapper", ["ar1", "ou", "diag", "cs", "us"])
def test_structured_inner_group_terms_are_rejected(data, wrapper):
    with pytest.raises(ValueError, match="Use 'by='"):
        bmb.Model(f"y ~ {wrapper}(0+time|g)", data)


def test_quoted_coefficient_name_keeps_distinct_design_and_prior(data):
    data["x + time"] = data.x**2
    quoted = "us(`x + time`, by=g)"
    model = bmb.Model(f"y ~ {quoted} + us(x + time, by=g)", data)
    terms = list(model.parameters["mu"].structured_terms.values())
    assert [term.dimension for term in terms] == [2, 3]
    model.set_priors({quoted: {"sd": 0.7}})
    assert terms[0].prior["sd"] == 0.7
    model.set_alias({quoted: "quoted_slopes"})
    assert resolve_plan(model, "marginal", "quoted_slopes")["selected"] == terms[0].qualified_name


def test_additive_and_auxiliary_selection(data):
    model = bmb.Model(
        bmb.Formula(
            "y ~ x + (1|h) + ar1(time, by=g) + diag(1+x, by=h)", "sigma ~ x + ar1(time, by=h)"
        ),
        data,
    )
    assert len(model.parameters["mu"].structured_terms) == 2
    assert resolve_plan(model, "auto", None)["selected"] == "mu.ar1(time, by=g)"
    with pytest.raises(ValueError, match="auxiliary"):
        resolve_plan(model, "marginal", "sigma.ar1(time, by=h)")
    model.build()
    assert np.isfinite(model.backend.model.compile_logp()(model.backend.model.initial_point()))


@pytest.mark.parametrize(
    "formula, message",
    [
        ("ar1(x, by=g)", "integers"),
        ("ar1(g, by=h)", "order"),
        ('ou(time, by=g, time_unit="months")', "time_unit"),
    ],
)
def test_coordinate_errors(data, formula, message):
    with pytest.raises(ValueError, match=message):
        bmb.Model("y ~ " + formula, data)


def test_declared_categories_and_datetime(data):
    model = bmb.Model("y ~ us(0+C(visit), by=g)", data)
    term = next(iter(model.parameters["mu"].structured_terms.values()))
    assert term.dimension == 3
    data["timestamp"] = pd.to_datetime("2020-01-01", utc=True) + pd.to_timedelta(
        data.time, unit="h"
    )
    model = bmb.Model('y ~ ou(timestamp, by=g, time_unit="hours")', data)
    term = next(iter(model.parameters["mu"].structured_terms.values()))
    np.testing.assert_array_equal(term.times, data.time)


def test_shared_missing_mask(data):
    data.loc[0, "h"] = None
    data.loc[1, "x"] = np.nan
    model = bmb.Model(
        bmb.Formula('y ~ ar1(time, by="g")', 'sigma ~ x + diag(1, by="h")'), data, dropna=True
    )
    assert model.response_term.shape[0] == 4
    for component in model.conditional_parameters.values():
        for term in component.structured_terms.values():
            assert len(term.data) == 4


def test_selection_errors_and_fallback(data):
    plain = bmb.Model("y ~ x", data)
    with pytest.raises(ValueError, match="No eligible"):
        resolve_plan(plain, "marginal", None)
    model = bmb.Model("y ~ ar1(time, by=g)", data, family="poisson")
    assert resolve_plan(model, "auto", None)["fallback"]
    with pytest.raises(ValueError, match="Gaussian"):
        resolve_plan(model, "marginal", None)
    with pytest.raises(ValueError, match="incompatible"):
        resolve_plan(model, "latent", "ar1(time, by=g)")
    with pytest.raises(ValueError, match="exactly one"):
        resolve_plan(model, "auto", ["ar1(time, by=g)"])


@pytest.mark.parametrize("wrapper", ["ar1", "ou", "diag", "cs", "us"])
@pytest.mark.parametrize("noncentered", [True, False])
def test_categorical_response_axes(data, wrapper, noncentered):
    data["category"] = pd.Categorical(["a", "b", "c", "b", "c", "a"])
    inner = "time" if wrapper in ("ar1", "ou") else "1+x+I(x**2)"
    model = bmb.Model(
        f"category ~ {wrapper}({inner}, by=g)", data, family="categorical", noncentered=noncentered
    )
    model.build()
    assert np.isfinite(model.backend.model.compile_logp()(model.backend.model.initial_point()))
    prior = model.prior_predictive(draws=2, random_seed=SEED)
    term = next(iter(model.parameters["p"].structured_terms.values()))
    assert prior.prior[term.label].shape[2:] == (tuple(len(v) for v in term.coords.values()) + (2,))
    # Prior draws can exercise the fitted prediction basis without posterior sampling.
    import xarray as xr

    idata = xr.DataTree.from_dict({"posterior": prior.prior.to_dataset()})
    model.predict(idata, data=data.iloc[[2, 0]], random_seed=SEED)
    assert idata.posterior.p.shape == (1, 2, 2, 3)


def test_zero_effect_prediction_and_independent_terms(data):
    model = bmb.Model(
        "y ~ 0 + diag(1, by=g) + diag(1, by=h)",
        data,
        priors={"diag(1, by=g)": {"sd": 1.0}, "diag(1, by=h)": {"sd": 1.0}},
    )
    model.build()
    prior = model.prior_predictive(draws=500, random_seed=SEED)
    import xarray as xr

    idata = xr.DataTree.from_dict({"posterior": prior.prior.to_dataset()})
    model.predict(idata, include_group_specific=False)
    np.testing.assert_array_equal(idata.posterior.mu, 0.0)
    target = data.iloc[[0, 0]].copy()
    target["g"], target["h"] = "new", "new"
    model.predict(idata, data=target, sample_new_groups=True, random_seed=SEED)
    draws = idata.posterior.mu.values[0, :, 0]
    assert 1.5 < draws.var() < 2.5
    np.testing.assert_array_equal(
        idata.posterior.mu.values[:, :, 0], idata.posterior.mu.values[:, :, 1]
    )


def test_prior_update_spelling_and_fixed_sigma(data):
    model = bmb.Model('y ~ ar1(time, by="g")', data, priors={"sigma": 0.5})
    model.set_priors({"ar1(time, by = g)": {"sd": 0.7, "rho": -0.4}})
    assert resolve_plan(model, "marginal", None)["selected"]
    model.structured_plan = resolve_plan(model, "marginal", None)
    model.build()
    assert np.isfinite(model.backend.model.compile_logp()(model.backend.model.initial_point()))


@pytest.mark.parametrize("wrapper", ["ar1", "ou", "diag", "cs", "us"])
def test_joint_prediction_domains_and_reordering(data, wrapper):
    import xarray as xr

    inner = "time" if wrapper in ("ar1", "ou") else "1 + center(x) + C(visit)"
    model = bmb.Model(f"y ~ {wrapper}({inner}, by=g)", data)
    model.set_alias({next(iter(model.parameters["mu"].structured_terms)): "effect"})
    model.build()
    prior = model.prior_predictive(draws=3, random_seed=SEED)
    idata = xr.DataTree.from_dict({"posterior": prior.prior.to_dataset()})
    target = data.iloc[[0, 0, 1, 3]].copy()
    target["time"] = [1, 1, 5, -1]
    target["x"] = [2.0, 2.0, -3.0, 4.0]
    target["visit"] = ["c", "c", "a", "b"]
    first = model.predict(idata, data=target, inplace=False, random_seed=SEED)
    np.testing.assert_array_equal(
        first.posterior.mu.values[:, :, 0], first.posterior.mu.values[:, :, 1]
    )
    permutation = [3, 1, 0, 2]
    reordered = model.predict(idata, data=target.iloc[permutation], inplace=False, random_seed=SEED)
    np.testing.assert_array_equal(
        first.posterior.mu.values[:, :, permutation], reordered.posterior.mu.values
    )
    again = model.predict(idata, data=target, inplace=False, random_seed=SEED)
    np.testing.assert_array_equal(first.posterior.mu, again.posterior.mu)
    empty = model.predict(idata, data=target.iloc[:0], inplace=False, random_seed=SEED)
    assert empty.posterior.mu.sizes["__obs__"] == 0
    model.predict(idata, data=target, random_seed=np.random.RandomState(SEED))
    if wrapper not in ("ar1", "ou"):
        target["visit"] = ["unknown"] * len(target)
        with pytest.raises(ValueError, match="levels|categor"):
            model.predict(idata, data=target, random_seed=SEED)


def test_composite_typed_keys_and_confounding(data):
    data["key"] = pd.Series(
        [("a:b", 1), ("a:b", "1"), ("a:b", 1.0), ("a:b", 1), ("a:b", "1"), ("a:b", 1.0)],
        dtype=object,
    )
    model = bmb.Model("y ~ ar1(time, by=key)", data)
    assert len(next(iter(model.parameters["mu"].structured_terms.values())).groups) == 3
    data["other"] = data.g
    with pytest.warns(UserWarning, match="exactly confounded"):
        bmb.Model("y ~ ar1(time, by=g) + ar1(time, by=other)", data)


def test_potentials_and_custom_likelihood_eligibility(data):
    from copy import deepcopy

    model = bmb.Model("y ~ ar1(time, by=g)", data)
    model.set_alias({"ar1(time, by=g)": "trajectory"})
    model.potentials = [("trajectory_sd", lambda scale: -scale)]
    assert resolve_plan(model, "marginal", None)["selected"]
    model.potentials = [("trajectory", lambda effect: -effect.sum())]
    with pytest.raises(ValueError, match="potential"):
        resolve_plan(model, "marginal", None)
    family = deepcopy(model.family)
    family.transform_backend_kwargs = lambda kwargs: {**kwargs, "mu": 2 * kwargs["mu"]}
    custom = bmb.Model("y ~ ar1(time, by=g)", data, family=family)
    assert resolve_plan(custom, "auto", None)["fallback"]
    with pytest.raises(ValueError, match="Custom likelihood"):
        resolve_plan(custom, "marginal", None)


def test_fit_plan_switching_and_rejected_fit_state(data, monkeypatch):
    from bambi.backend.pymc import PyMCModel

    monkeypatch.setattr(PyMCModel, "run", lambda self, **kwargs: self.model)
    model = bmb.Model("y ~ ar1(time, by=g) + diag(1+x, by=h)", data)
    model.build()
    latent = model.backend
    model.fit()
    assert model.backend is not latent
    assert model.structured_plan["selected"] == "mu.ar1(time, by=g)"
    marginal = model.backend
    model.fit(structured_method="latent")
    assert model.backend is not marginal
    assert model.structured_plan["selected"] is None
    before, plan = model.backend, model.structured_plan.copy()
    with pytest.raises(ValueError, match="NUTS"):
        model.fit(inference_method="vi", structured_method="marginal")
    assert model.backend is before and model.structured_plan == plan and model.built
    model.fit(structured_method="marginal", marginalize="diag(1+x, by=h)")
    assert model.structured_plan["selected"] == "mu.diag(1 + x, by=h)"


@pytest.mark.parametrize("wrapper", ["ar1", "ou", "diag", "cs", "us"])
def test_conditional_observation_log_likelihood_after_recovery(data, wrapper):
    import xarray as xr
    from scipy.stats import norm
    from bambi.structured.inference import recover_effect

    inner = "time" if wrapper in ("ar1", "ou") else "1+x"
    model = bmb.Model(f"y ~ x + (1|h) + {wrapper}({inner}, by=g)", data, priors={"sigma": 0.5})
    model.build()
    prior = model.prior_predictive(draws=3, random_seed=SEED)
    idata = xr.DataTree.from_dict({"posterior": prior.prior.to_dataset()})
    model.structured_plan = resolve_plan(model, "marginal", None)
    model.build()
    recover_effect(model, idata, SEED)
    model.compute_log_likelihood(idata)
    assert idata.log_likelihood.y.shape == (1, 3, len(data))
    expected = norm.logpdf(data.y.to_numpy(), loc=idata.posterior.mu.values, scale=0.5)
    np.testing.assert_allclose(idata.log_likelihood.y.values, expected)
    assert json.loads(idata.posterior.attrs["structured_inference"])["effective"] == "marginal"


@pytest.mark.parametrize(
    "model_setting, override, expected",
    [
        (True, None, True),
        (False, None, False),
        ({"mu": False}, None, False),
        (False, True, True),
        (True, False, False),
    ],
)
def test_structured_parameterization_controls(data, model_setting, override, expected):
    model = bmb.Model(
        "y ~ diag(1+x, by=g)",
        data,
        noncentered=model_setting,
        priors={
            "diag(1+x, by=g)": {"sd": bmb.Prior("HalfNormal", sigma=0.5, noncentered=override)}
        },
    )
    model.build()
    assert ("diag(1 + x, by=g)_offset" in model.backend.model.named_vars) == expected


def test_individual_coefficient_priors_and_shared_prior_errors(data):
    model = bmb.Model(
        "y ~ diag(1+x, by=g)",
        data,
        priors={"diag(1+x, by=g)": {"sd": [bmb.Prior("HalfNormal", sigma=0.5), 0.3]}},
    )
    model.build()
    assert "diag(1 + x, by=g)_sd_0" in model.backend.model.named_vars
    with pytest.raises(ValueError, match="Shared"):
        bmb.Model(
            "y ~ diag(1+x, by=g)",
            data,
            priors={"group_specific": bmb.Prior("Normal", mu=0, sigma=1)},
        )


@pytest.mark.parametrize("family", ["bernoulli", "poisson", "cumulative"])
@pytest.mark.parametrize("wrapper", ["ar1", "ou", "diag", "cs", "us"])
def test_other_families_latent_fallback(data, family, wrapper):
    if family == "bernoulli":
        data["y"] = [0, 1, 0, 1, 1, 0]
    elif family == "poisson":
        data["y"] = [0, 2, 1, 3, 1, 2]
    else:
        data["y"] = pd.Categorical(["a", "b", "c", "b", "c", "a"], ordered=True)
    inner = "time" if wrapper in ("ar1", "ou") else "1+x"
    model = bmb.Model(f"y ~ x + {wrapper}({inner}, by=g)", data, family=family)
    assert resolve_plan(model, "auto", None)["effective"] == "latent"
    model.build()
    assert np.isfinite(model.backend.model.compile_logp()(model.backend.model.initial_point()))


def test_hsgp_additive_marginal_build(data):
    model = bmb.Model("y ~ hsgp(x, m=3, c=1.5) + ar1(time, by=g)", data)
    model.structured_plan = resolve_plan(model, "marginal", None)
    model.build()
    assert np.isfinite(model.backend.model.compile_logp()(model.backend.model.initial_point()))


def test_us_public_correlation_and_latent_factor(data):
    model = bmb.Model(
        "y ~ us(1+x, by=g)",
        data,
        priors={"us(1+x, by=g)": {"sd": [0.7, 0.3]}},
    )
    model.build()
    prior = model.prior_predictive(draws=4, omit_offsets=False, random_seed=SEED)
    term = next(iter(model.parameters["mu"].structured_terms.values()))
    correlation = prior.prior[term.label + "_correlation"].values
    factor = prior.prior[term.label + "_correlation_cholesky"].values
    np.testing.assert_allclose(correlation, factor @ factor.swapaxes(-1, -2), atol=1e-12)
    np.testing.assert_allclose(np.diagonal(correlation, axis1=-2, axis2=-1), 1, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(correlation) > 0)
    offsets = prior.prior[term.label + "_offset"].values
    scaled = np.array([0.7, 0.3])[:, None] * factor
    expected = offsets @ scaled.swapaxes(-1, -2)
    np.testing.assert_allclose(prior.prior[term.label], expected)


def test_ar1_numeric_categories_and_declared_grid(data):
    data["time"] = pd.Categorical([2024, 2026, 2024, 2026, 2024, 2026])
    model = bmb.Model("y ~ ar1(time, by=g)", data)
    term = next(iter(model.parameters["mu"].structured_terms.values()))
    np.testing.assert_array_equal(term.times, np.asarray(data.time, dtype=float))
    np.testing.assert_array_equal(term.coordinate_values(data.time), term.times)
    data["time"] = data.time.cat.as_ordered()
    model = bmb.Model("y ~ ar1(time, by=g)", data)
    term = next(iter(model.parameters["mu"].structured_terms.values()))
    np.testing.assert_array_equal(term.times, data.time.cat.codes)


def test_source_spelling_whitespace(data):
    model = bmb.Model(
        "y ~ ar1(time, by=g)",
        data,
        priors={" ar1 (time,by = g) ": {"sd": 0.7, "rho": -0.4}},
    )
    plan = resolve_plan(model, "marginal", ' ar1 ( time, by = "g" ) ')
    assert plan["selected"] == "mu.ar1(time, by=g)"


def test_nutpie_metadata_collision_and_failure_restoration():
    import pymc as pm
    from bambi.backend.utils import sampler_dimensions

    with pm.Model(coords={"a,b": ["x", "y"], "a%2Cb": ["one"]}) as model:
        pm.Normal("weights", dims=("a,b", "a%2Cb"))
    original_coords = model.coords.copy()
    original_dims = model.named_vars_to_dims.copy()
    original_lengths = model.dim_lengths.copy()
    with pytest.raises(RuntimeError, match="sampling failed"):
        with sampler_dimensions(model, "nutpie") as restore:
            assert len(set(model.coords)) == 2
            assert all("," not in dim for dim in model.coords)
            assert set(restore.values()) == set(original_coords)
            raise RuntimeError("sampling failed")
    assert model.coords == original_coords
    assert model.named_vars_to_dims == original_dims
    assert all(model.dim_lengths[name] is value for name, value in original_lengths.items())


@pytest.mark.parametrize("wrapper", ["ar1", "ou", "diag", "cs", "us"])
def test_single_state_and_one_coefficient(data, wrapper):
    data["time"] = 2
    data["g"] = "one"
    inner = "time" if wrapper in ("ar1", "ou") else "1"
    model = bmb.Model(f"y ~ {wrapper}({inner}, by=g)", data)
    model.build()
    assert np.isfinite(model.backend.model.compile_logp()(model.backend.model.initial_point()))
    prior = model.prior_predictive(draws=2, random_seed=SEED)
    term = next(iter(model.parameters["mu"].structured_terms.values()))
    assert term.dimension == 1
    assert np.isfinite(prior.prior[term.label]).all()
    if wrapper in ("cs", "us"):
        assert term.label + "_rho" not in prior.prior
        assert term.label + "_correlation_cholesky" not in prior.prior


@pytest.mark.parametrize(
    "term,prior,message",
    [
        ("ar1(time, by=g)", {"rho": -1.0}, "strictly between"),
        ("cs(1+x, by=g)", {"rho": 1.0}, "strictly between"),
        ("ou(time, by=g)", {"decay": 0.0}, "positive"),
        ("diag(1+x, by=g)", {"sd": [0.5, 0.2, 0.1]}, "shape"),
        ("us(1+x, by=g)", {"correlation": [[1, 1], [1, 1]]}, "positive definite"),
        ("us(1+x, by=g)", {"eta": 2, "correlation": np.eye(2)}, "either"),
    ],
)
def test_invalid_covariance_parameters(data, term, prior, message):
    with pytest.raises(ValueError, match=message):
        bmb.Model("y ~ " + term, data, priors={term: prior})


def test_all_invalid_rows_and_incompatible_datetime_prediction(data):
    data["timestamp"] = pd.to_datetime("2020-01-01", utc=True) + pd.to_timedelta(
        data.time, unit="h"
    )
    model = bmb.Model("y ~ ou(timestamp, by=g)", data)
    term = next(iter(model.parameters["mu"].structured_terms.values()))
    with pytest.raises(ValueError, match="naive/aware"):
        term.coordinate_values(data.timestamp.dt.tz_localize(None))
    data["g"] = None
    with pytest.raises(ValueError, match="No observations remain"):
        bmb.Model("y ~ ar1(time, by=g)", data, dropna=True)


@pytest.mark.parametrize("wrapper", ["diag", "cs", "us"])
@pytest.mark.parametrize("auto_scale", [False, True])
def test_coefficient_default_scaling(data, wrapper, auto_scale):
    data["x"] *= 10
    formula = f"{wrapper}(1+x, by=g)"
    model = bmb.Model("y ~ " + formula, data, auto_scale=auto_scale)
    term = next(iter(model.parameters["mu"].structured_terms.values()))
    expected = (
        2.5 * np.std(data.y) / np.array([1, np.std(data.x)]) if auto_scale else np.array([2.5, 2.5])
    )
    np.testing.assert_allclose(term.prior["sd"].args["sigma"], expected)
    model.set_priors({formula: {"sd": bmb.Prior("HalfNormal", sigma=[0.7, 0.2])}})
    np.testing.assert_array_equal(term.prior["sd"].args["sigma"], [0.7, 0.2])


@pytest.mark.parametrize("wrapper,inner", [("ar1", "time"), ("cs", "1+x")])
@pytest.mark.parametrize("argument", ["mu", "sigma"])
def test_correlation_prior_must_be_scalar(data, wrapper, inner, argument):
    term = f"{wrapper}({inner}, by=g)"
    parameters = {"mu": 0, "sigma": 0.5, argument: np.ones(6)}
    with pytest.raises(ValueError, match="rho prior parameters must be scalar"):
        bmb.Model("y ~ " + term, data, priors={term: {"rho": bmb.Prior("Normal", **parameters)}})


def test_prior_updates_resolve_structured_aliases(data):
    term = "ar1(time, by=g)"
    model = bmb.Model(bmb.Formula("y ~ " + term, "sigma ~ " + term), data)
    model.set_alias({"mu": {term: "trajectory"}, "sigma": {term: "trajectory"}})
    with pytest.raises(ValueError, match="ambiguous.*Component-qualified"):
        model.set_priors({"trajectory": {"sd": 0.7}})
    model.set_priors({"mu.trajectory": {"sd": 0.7}, "sigma": {"trajectory": {"sd": 0.2}}})
    assert model.parameters["mu"].structured_terms[term].prior["sd"] == 0.7
    assert model.parameters["sigma"].structured_terms[term].prior["sd"] == 0.2


@pytest.mark.parametrize("reverse", [False, True])
def test_prior_alias_collision_and_component_merge(data, reverse):
    term = "ar1(time, by=g)"
    model = bmb.Model(bmb.Formula("y ~ " + term, "sigma ~ x + " + term), data)
    model.set_alias({"mu": {term: "trajectory"}, "sigma": {term: "scale_process"}})
    updates = [("trajectory", {"sd": 0.7}), (term, {"sd": 0.8})]
    with pytest.raises(ValueError, match="Multiple specifications"):
        model.set_priors(dict(updates[::-1] if reverse else updates))
    updates = [
        ("sigma.scale_process", {"sd": 0.2}),
        ("sigma", {"x": bmb.Prior("Normal", sigma=0.1)}),
    ]
    model.set_priors(dict(updates[::-1] if reverse else updates))
    assert model.parameters["sigma"].structured_terms[term].prior["sd"] == 0.2
    assert model.parameters["sigma"].terms["x"].prior.args["sigma"] == 0.1
    model.set_priors({'sigma.ar1(time, by="g")': {"sd": 0.3}})
    assert model.parameters["sigma"].structured_terms[term].prior["sd"] == 0.3


def test_individual_sd_priors_reject_vector_parameters(data):
    term = "diag(1+x, by=g)"
    with pytest.raises(ValueError, match="Each coefficient SD prior must have scalar"):
        bmb.Model(
            "y ~ " + term,
            data,
            priors={term: {"sd": [bmb.Prior("HalfNormal", sigma=[0.5, 0.2]), 0.3]}},
        )


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_lkj_prior_has_exchangeable_pairwise_moments(data, dimension):
    inner = "1+x" + "".join(f"+I(x**{power})" for power in range(2, dimension))
    term = f"us({inner}, by=g)"
    model = bmb.Model("y ~ " + term, data, priors={term: {"sd": 1.0, "eta": 2}})
    model.set_alias({term: "effect"})
    model.build()
    prior = model.prior_predictive(draws=8000, random_seed=SEED)
    correlation = prior.prior.effect_correlation.values.reshape(-1, dimension, dimension)
    i, j = np.tril_indices(dimension, -1)
    second_moment = (correlation[:, i, j] ** 2).mean(axis=0)
    denominator = 4 + dimension - 1
    expected = 1 / denominator
    fourth_moment = 3 / (denominator * (denominator + 2))
    tolerance = 5 * np.sqrt((fourth_moment - expected**2) / len(correlation))
    np.testing.assert_allclose(second_moment, expected, atol=tolerance)


@pytest.mark.parametrize("wrapper", ["ar1", "ou", "diag", "cs", "us"])
def test_ungrouped_process_is_one_group(data, wrapper):
    import xarray as xr

    inner = "time" if wrapper in ("ar1", "ou") else "1+x"
    model = bmb.Model(f"y ~ {wrapper}({inner})", data)
    term = next(iter(model.parameters["mu"].structured_terms.values()))
    assert term.groups == [()]
    np.testing.assert_array_equal(term.group_index, 0)
    model.build()
    assert np.isfinite(model.backend.model.compile_logp()(model.backend.model.initial_point()))

    prior = model.prior_predictive(draws=2, random_seed=SEED)
    idata = xr.DataTree.from_dict({"posterior": prior.prior.to_dataset()})
    target = data.iloc[[0, 0]].copy()
    target["time"] = 5
    model.predict(idata, data=target, random_seed=SEED)
    np.testing.assert_array_equal(
        idata.posterior.mu.values[..., 0], idata.posterior.mu.values[..., 1]
    )
    model.structured_plan = resolve_plan(model, "marginal", None)
    model.build()
    assert np.isfinite(model.backend.model.compile_logp()(model.backend.model.initial_point()))


@pytest.mark.parametrize("family", ["multinomial", "dirichlet_multinomial"])
@pytest.mark.parametrize("wrapper", ["ar1", "ou", "diag", "cs", "us"])
def test_multivariate_count_response_axes(data, family, wrapper):
    import xarray as xr

    data["y1"], data["y2"], data["y3"] = [1, 0, 2, 1, 1, 0], [1, 2, 0, 1, 0, 2], [1, 1, 1, 1, 2, 1]
    inner = "time" if wrapper in ("ar1", "ou") else "1+x"
    model = bmb.Model(f"counts(y1,y2,y3) ~ {wrapper}({inner}, by=g)", data, family=family)
    assert resolve_plan(model, "auto", None)["effective"] == "latent"
    model.build()
    assert np.isfinite(model.backend.model.compile_logp()(model.backend.model.initial_point()))
    prior = model.prior_predictive(draws=2, random_seed=SEED)
    idata = xr.DataTree.from_dict({"posterior": prior.prior.to_dataset()})
    model.predict(idata, data=data.iloc[[0, 1]], kind="response", random_seed=SEED)
    assert idata.posterior[model.family.likelihood.parent].shape == (1, 2, 2, 3)
    response = next(iter(idata.posterior_predictive.data_vars.values()))
    assert response.shape == (1, 2, 2, 3)
    np.testing.assert_array_equal(response.sum(axis=-1), 3)
