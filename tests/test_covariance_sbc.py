"""Cheap checks for independent simulation and fail-closed calibration accounting."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "docs/examples/group_covariance_validation"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reference = load_script("single_block_reference")
sbc = load_script("sbc")
load_script("validation_identity")


@pytest.mark.parametrize("kind", reference.KINDS)
@pytest.mark.parametrize("family", ["bernoulli", "binomial"])
def test_prior_fixture_is_reproducible_and_valid(kind, family):
    first = reference.generate_fixture(kind, family, 271828)
    assert first == reference.generate_fixture(kind, family, 271828)
    rows, fixture, truth = first
    assert all(0 <= row["y"] <= row["trials"] for row in rows)
    assert all((row["x1"] * 8).is_integer() for row in rows)
    assert all(row["trials"] == 1 for row in rows) == (family == "bernoulli")
    assert np.isfinite(list(truth.values())).all()
    assert len(fixture["blocks"]) == (4 if kind == "four-block-ar1" else 1)
    assert not any(
        row["subject"] == "s1" and row["condition"] == "a" and row["context"] == "c" for row in rows
    )
    assert not any(
        row["subject"] == "s2" and row["year"] == fixture["blocks"][0]["times"][1] for row in rows
    )


@pytest.mark.parametrize("kind", reference.KINDS[:-1])
def test_independent_correlation_is_positive_definite(kind):
    rng = np.random.default_rng(161803)
    for _ in range(20):
        correlation, parameters = reference.prior_correlation(kind, [0, 1, 3], rng)
        np.testing.assert_allclose(correlation, correlation.T)
        np.testing.assert_allclose(np.diag(correlation), 1)
        assert np.linalg.eigvalsh(correlation).min() > 0
        assert parameters


def test_lkj_generator_has_correct_marginal_second_moment():
    rng = np.random.default_rng(314159)
    values = [reference.prior_correlation("us", [0, 1, 3], rng)[0][0, 1] for _ in range(4000)]
    # LKJ(eta=2), dimension 3: each correlation has second moment 1 / (2 eta + Q - 1).
    assert abs(np.mean(np.square(values)) - 1 / 6) < 0.015
    assert abs(np.mean(values)) < 0.03


def test_campaign_sizes_and_seeds_are_fixed():
    pilot = sbc.case_specifications("pilot")
    full = sbc.case_specifications("full")
    assert len(pilot) == 60
    assert len(full) == 600
    assert len({case["id"] for case in full}) == 600
    assert len({case["seed"] for case in full}) == 600
    assert {case["seed"] for case in full}.isdisjoint(case["seed"] for case in pilot)
    assert full == sbc.case_specifications("full")


def valid_diagnostics():
    return dict(
        chains=4,
        rhat_max=1,
        ess_bulk_min=500,
        ess_tail_min=500,
        divergences=0,
        bfmi_min=0.8,
        treedepth_hits=0,
    )


@pytest.mark.parametrize(
    "field,bad",
    [
        ("chains", 1),
        ("rhat_max", 1.02),
        ("ess_bulk_min", 399),
        ("ess_tail_min", 399),
        ("divergences", 1),
        ("bfmi_min", 0.2),
        ("treedepth_hits", 1),
        ("rhat_max", float("nan")),
    ],
)
def test_sampler_failure_cannot_be_certified(field, bad):
    diagnostics = valid_diagnostics()
    sbc.validate_diagnostics(diagnostics)
    diagnostics[field] = bad
    with pytest.raises(ValueError, match=field):
        sbc.validate_diagnostics(diagnostics)


def test_finite_rank_ties_are_randomized():
    rng = np.random.default_rng(12345)
    assert sbc.randomized_rank(2, [0, 1, 3], rng) == 2
    ranks = {sbc.randomized_rank(1, [1] * 4, rng) for _ in range(100)}
    assert ranks == set(range(5))
    with pytest.raises(ValueError):
        sbc.randomized_rank(float("nan"), [0], rng)


def test_checker_detects_bias_and_accepts_discrete_uniform_ranks():
    assert sbc.rank_check(list(range(101)), 100, tests=20)["passed"]
    assert not sbc.rank_check([0] * 100, 100, tests=20)["passed"]
    assert not sbc.rank_check([100] * 100, 100, tests=20)["passed"]
    with pytest.raises(ValueError):
        sbc.rank_check([101], 100, tests=1)


def test_coverage_uncertainty_is_not_a_point_estimate():
    low, high = sbc.wilson_interval(94, 100)
    assert low < 0.94 < high
    assert low > 0.8 and high < 1


def test_incomplete_campaign_never_passes(tmp_path):
    manifest = dict(
        schema_version=1,
        phase="pilot",
        source={},
        settings=dict(rank_draws=100),
        cases=sbc.case_specifications("pilot"),
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    result = sbc.aggregate(path)
    assert not result["passed"]
    assert result["completed"] == 0
    assert len(result["failures"]) == 60


def test_removed_or_duplicate_case_is_rejected(tmp_path):
    cases = sbc.case_specifications("pilot")
    for changed in (cases[:-1], cases + [cases[0]]):
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(dict(phase="pilot", cases=changed)))
        with pytest.raises(ValueError, match="specification"):
            sbc.aggregate(path)


@pytest.mark.parametrize("value", [True, 4.5])
def test_chain_count_must_be_an_integer(value):
    diagnostics = valid_diagnostics()
    diagnostics["chains"] = value
    with pytest.raises(ValueError, match="chains"):
        sbc.validate_diagnostics(diagnostics)


def mock_rank_dataset(monkeypatch, original_ess, retained_ess):
    class Minimum:
        def __init__(self, value):
            self.value = value

        def min(self, **kwargs):
            return self.value

    class Dataset:
        sizes = {"chain": 4, "draw": 1000}

        def isel(self, draw):
            return SimpleNamespace(indices=draw)

    def summary(dataset, **kwargs):
        value = retained_ess if hasattr(dataset, "indices") else original_ess
        return SimpleNamespace(ess_bulk=Minimum(value), ess_tail=Minimum(value))

    monkeypatch.setitem(sys.modules, "arviz_stats", SimpleNamespace(summary=summary))
    return Dataset()


def test_rank_selection_spaces_within_chains(monkeypatch):
    dataset = mock_rank_dataset(monkeypatch, 400, 90)
    selected, diagnostics = sbc.select_rank_draws(dataset, 100, 12345)
    assert len(selected.indices) == 25
    assert np.all(np.diff(selected.indices) == 20)
    assert diagnostics["rank_draws"] == 100
    assert selected.indices[-1] < 1000


@pytest.mark.parametrize("original,retained", [(100, 90), (400, 70), (float("nan"), 90)])
def test_rank_selection_rejects_dependent_or_insufficient_draws(monkeypatch, original, retained):
    dataset = mock_rank_dataset(monkeypatch, original, retained)
    with pytest.raises(ValueError):
        sbc.select_rank_draws(dataset, 100, 12345)


@pytest.mark.parametrize("kind", reference.KINDS)
@pytest.mark.parametrize("family", ["bernoulli", "binomial"])
def test_reference_public_model_builds_without_sampling(tmp_path, kind, family):
    path = tmp_path / "fixture"
    reference.write_fixture(kind, family, 271828, path)
    model, data, fixture = reference.build_model(path)
    terms = list(model.parameters[model.family.likelihood.parent].group_specific_terms.values())
    assert len(terms) == len(fixture["blocks"])
    assert model.family.name == family
    assert len(data) > 0
    for term, block in zip(terms, fixture["blocks"]):
        assert term.block.kind == block["kind"]
        assert len(term.groups) == len(block["groups"])
        assert term.label in model.backend.model
        assert term.prior["sd"].name == "HalfNormal"
        assert term.prior["sd"].args["sigma"] == 2.5


@pytest.mark.parametrize("change", ["prior", "group", "family"])
def test_altered_fixture_contract_is_rejected_before_model_build(tmp_path, change):
    path = tmp_path / "fixture"
    reference.write_fixture("ar1", "bernoulli", 271828, path)
    fixture_path = path / "fixture.json"
    fixture = json.loads(fixture_path.read_text())
    if change == "prior":
        fixture["priors"]["sd"] = "HalfNormal(1)"
    elif change == "group":
        fixture["blocks"][0]["group_id"][0] = 2
    else:
        fixture["family"] = "poisson"
    fixture_path.write_text(json.dumps(fixture))
    with pytest.raises(ValueError):
        reference.validate_fixture_identity(path)


@pytest.mark.parametrize("kind", ["ar1", "us", "four-block-ar1"])
def test_reference_export_pipeline_without_mcmc(tmp_path, monkeypatch, kind):
    import bambi as bmb
    import xarray as xr

    helpers = load_script("reference_metrics")
    original_prior = bmb.Model.prior_predictive

    def small_prior(self, **kwargs):
        kwargs["draws"] = 3
        return original_prior(self, **kwargs)

    def forward_draws_as_posterior(self, **kwargs):
        assert kwargs["cores"] == 1
        assert kwargs["inference_method"] == "nutpie"
        prior = small_prior(self, random_seed=271828, omit_offsets=False)
        return xr.DataTree.from_dict({"posterior": prior["prior"].to_dataset()})

    monkeypatch.setattr(bmb.Model, "prior_predictive", small_prior)
    monkeypatch.setattr(bmb.Model, "fit", forward_draws_as_posterior)
    monkeypatch.setattr(helpers, "diagnose_scalar_metrics", lambda *args: valid_diagnostics())
    monkeypatch.setattr(
        helpers,
        "summarize_metrics",
        lambda data: {
            name: {"mean": float(value.mean())} for name, value in data.data_vars.items()
        },
    )
    fixture = tmp_path / "fixture"
    output = tmp_path / "fit"
    reference.write_fixture(kind, "binomial", 271828, fixture)
    reference.bambi_fit(fixture, output, chains=4, warmup=1000, draws=1000, seed=271829)
    summary = json.loads((output / "summary.json").read_text())
    truth = json.loads((fixture / "truth.json").read_text())
    assert set(truth) <= set(summary["metrics"])
    assert "log_likelihood.1" in summary["metrics"]
    assert "predictive_second_moment.1" in summary["metrics"]
    for name in ("prior.nc", "inference.nc", "inference-predictive.nc", "metrics.nc"):
        assert (output / name).is_file()
    with xr.open_dataset(output / "metrics.nc") as metrics:
        assert metrics.sizes["draw"] == 3
        assert set(truth) <= set(metrics.data_vars)
