import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

import bambi as bmb


def make_model(groups=False, auxiliary=True, bare=False, noncentered=True):
    x = np.linspace(-1, 1, 12)
    data = pd.DataFrame({"y": 1 + x, "x": x, "z": x**2, "g": np.repeat(["u", "v", "w"], 4)})
    suffix = " + (1 | g)" if groups else ""
    formulas = ["y ~ a" if bare else "y ~ a * x", "a ~ 1" + suffix]
    if auxiliary:
        formulas.append("sigma ~ z" + suffix)
    return bmb.Model(
        bmb.Formula(*formulas, nonlinear=True),
        data,
        center_predictors=False,
        noncentered=noncentered,
    )


def aliases(auxiliary=True):
    return {
        "mu": {"mu": "mean"},
        "a": {"a": "baseline", "Intercept": "a0"},
        "sigma": {"sigma": "noise", "Intercept": "s0", "z": "noise_z"} if auxiliary else "noise",
        "y": "response",
    }


@pytest.mark.parametrize("auxiliary", [False, True])
@pytest.mark.parametrize("bare", [False, True])
def test_alias_predictions_likelihood_and_rebuild(auxiliary, bare):
    model = make_model(auxiliary=auxiliary, bare=bare)
    model.build()
    values = {"a_Intercept": 2.0}
    values.update({"sigma_Intercept": -0.5, "sigma_z": 0.3} if auxiliary else {"sigma": 0.8})
    draws = xr.Dataset({name: (("chain", "draw"), [[value]]) for name, value in values.items()})
    original = model.predict(xr.DataTree.from_dict({"posterior": draws}), inplace=False)
    model.set_alias(aliases(auxiliary))
    assert not model.built
    model.build()
    renamed = {"a_Intercept": "a0"}
    renamed.update(
        {"sigma_Intercept": "s0", "sigma_z": "noise_z"} if auxiliary else {"sigma": "noise"}
    )
    idata = xr.DataTree.from_dict({"posterior": draws.rename(renamed)})
    result = model.predict(idata, inplace=False)
    np.testing.assert_allclose(result.posterior["mean"], original.posterior["mu"])
    for data in (model.data, model.data.iloc[:3].assign(x=0.7, z=0.4)):
        result = model.predict(idata, data=data, inplace=False)
        expected = np.full(len(data), 2.0) if bare else 2 * data.x.to_numpy()
        scale = np.exp(-0.5 + 0.3 * data.z) if auxiliary else 0.8
        np.testing.assert_allclose(result.predictions["mean"], [[expected]])
        likelihood = model.compute_log_likelihood(idata, data=data, inplace=False)
        direct = pm.logp(pm.Normal.dist(mu=expected, sigma=scale), data.y).eval()
        np.testing.assert_allclose(likelihood.log_likelihood.response, [[direct]])
    if not bare:
        assert "mean__x_data" in model.backend.model.named_vars
        assert "mu__x_data" not in model.backend.model.named_vars
    assert "baseline" in model.backend.model.named_vars
    assert "a" not in model.backend.model.named_vars
    assert "a0" in str(model)
    model.set_priors({"a": {"Intercept": bmb.Prior("Normal", mu=0.4, sigma=0.2)}})
    model.build()
    assert model.nonlinear_predictors["a"].terms["Intercept"].prior.args["mu"] == 0.4
    assert "a0" in model.backend.model.named_vars


@pytest.mark.parametrize("auxiliary", [False, True])
def test_alias_prior_filtering(auxiliary):
    model = make_model(groups=True, auxiliary=auxiliary)
    mapping = aliases(auxiliary)
    mapping["a"].update({"1|g": "by_group", "sigma": "group_sd"})
    model.set_alias(mapping)
    model.build()
    prior = model.backend.prior_predictive(
        draws=3, prior_only=True, omit_group_specific=True, random_seed=123
    )
    assert "a0" in prior.prior
    assert "by_group_group_sd" in prior.prior
    assert not {"mean", "baseline", "by_group", "sigma_1|g"} & set(prior.prior)
    assert ("noise" in prior.prior) is not auxiliary


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("sparse_dot", [False, True])
@pytest.mark.parametrize("include_response_params", [False, True])
def test_alias_group_sampling_and_unknown_prediction(
    monkeypatch, sparse_dot, include_response_params
):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    model = make_model(groups=True)
    mapping = aliases()
    mapping["a"].update({"1|g": "by_group", "sigma": "group_sd"})
    mapping["sigma"].update({"1|g": "noise_group", "sigma": "noise"})
    model.set_alias(mapping)
    idata = model.fit(draws=3, chains=1, include_response_params=include_response_params)
    assert "baseline" not in idata.posterior
    assert ("mean" in idata.posterior) is include_response_params
    assert ("noise" in idata.posterior) is include_response_params
    assert "by_group_group_sd" in idata.posterior
    for group in ("u", "unseen"):
        data = model.data.iloc[:3].assign(g=group)
        prediction = model.predict(
            idata, data=data, kind="response", inplace=False, random_seed=123
        )
        assert prediction.predictions.response.shape == (1, 3, 3)
        assert prediction.predictions["mean"].shape == (1, 3, 3)
        if group == "unseen":
            with pytest.raises(ValueError, match="Cannot compute log likelihood for new groups"):
                model.compute_log_likelihood(idata, data=data, inplace=False)
        else:
            likelihood = model.compute_log_likelihood(idata, data=data, inplace=False)
            assert likelihood.log_likelihood.response.shape == (1, 3, 3)


def test_alias_unknown_names_types_and_collisions():
    model = make_model()
    with pytest.warns(UserWarning, match="missing, absent"):
        model.set_alias({"a": {"missing": "unused"}, "absent": {"Intercept": "other"}})
    with pytest.raises(AssertionError, match="Alias must be a string"):
        model.set_alias({"mu": {"mu": 5}})
    with pytest.raises(ValueError, match="must be a dictionary"):
        model.set_alias("a")
    model.set_alias({"a": {"a": "mu"}})
    with pytest.raises(ValueError, match="already exists"):
        model.build()


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("omit_offsets", [False, True])
def test_offset_suffix_aliases_remain_in_prior_and_posterior(nonlinear, omit_offsets):
    model = make_model(groups=True, auxiliary=False)
    if nonlinear:
        model.set_alias(
            {"a": {"Intercept": "baseline_offset", "1|g": "group_offset"}, "sigma": "noise_offset"}
        )
    else:
        model = bmb.Model("y ~ x + (1|g)", model.data)
        model.set_alias(
            {"Intercept": "baseline_offset", "1|g": "group_offset", "sigma": "noise_offset"}
        )
    model.build()
    for prior_only in (False, True):
        prior = model.backend.prior_predictive(
            draws=3, prior_only=prior_only, omit_offsets=omit_offsets, random_seed=123
        )
        assert {"baseline_offset", "noise_offset", "group_offset"} <= set(prior.prior)
        assert ("group_offset_offset" in prior.prior) is not omit_offsets
    posterior = model.fit(draws=3, chains=1, omit_offsets=omit_offsets).posterior
    assert {"baseline_offset", "noise_offset", "group_offset"} <= set(posterior)
    assert ("group_offset_offset" in posterior) is not omit_offsets


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("override", [False, True])
def test_offset_inventory_respects_prior_override(override):
    model = make_model(groups=True, auxiliary=False, noncentered=not override)
    model.set_priors(
        {
            "a": {
                "1|g": bmb.Prior(
                    "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1), noncentered=override
                )
            }
        }
    )
    model.build()
    assert ("a_1|g_offset" in model.backend.model.__bambi_attrs__["offset_names"]) is override
    prior = model.backend.prior_predictive(draws=3, omit_offsets=True, random_seed=123)
    assert "a_1|g_offset" not in prior.prior
    assert "a_1|g" in prior.prior
    idata = model.fit(draws=3, chains=1)
    prediction = model.predict(idata, data=model.data.iloc[:3], inplace=False)
    assert prediction.predictions.mu.shape == (1, 3, 3)
    likelihood = model.compute_log_likelihood(idata, inplace=False)
    assert likelihood.log_likelihood.y.shape == (1, 3, 12)


def test_offset_inventory_includes_recursive_hyperpriors():
    model = make_model(groups=True, auxiliary=False)
    model.set_priors(
        {
            "a": {
                "1|g": bmb.Prior(
                    "Normal",
                    mu=0,
                    sigma=bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)),
                )
            }
        }
    )
    model.build()
    assert model.backend.model.__bambi_attrs__["offset_names"] == {
        "a_1|g_offset",
        "a_1|g_sigma_offset",
    }
