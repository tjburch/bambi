"""Held-out partitions and predictive scoring without model fitting."""

import importlib.util
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).parents[1] / "docs/examples/group_covariance_validation/heldout_checks.py"
SPEC = importlib.util.spec_from_file_location("heldout_checks", PATH)
CHECKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKS)


@pytest.fixture
def panel():
    return pd.DataFrame(
        product(["s1", "s2"], ["a", "b"], ["c", "d"], [0, 1, 3], [1, 2]),
        columns=["subject", "condition", "context", "year", "replicate"],
    )


@pytest.mark.parametrize("split", ["subject", "future", "three-way"])
def test_heldout_split_has_no_block_leakage(panel, split):
    mask = CHECKS.split_mask(panel, split)
    train, heldout = panel.loc[~mask], panel.loc[mask]
    assert len(train) + len(heldout) == len(panel)
    if split == "subject":
        assert set(train.subject).isdisjoint(heldout.subject)
    elif split == "future":
        assert train.year.max() < heldout.year.min()
    else:
        columns = ["subject", "condition", "context"]
        assert set(map(tuple, train[columns].values)).isdisjoint(
            map(tuple, heldout[columns].values)
        )
        for pair in (["subject", "condition"], ["subject", "context"]):
            assert set(map(tuple, heldout[pair].values)) <= set(map(tuple, train[pair].values))


def test_split_rejects_insufficient_training_time(panel):
    with pytest.raises(ValueError, match="time coordinates"):
        CHECKS.split_mask(panel.loc[panel.year != 3], "future")


def test_predictive_log_score_is_mixture_not_mean_log_score():
    probability = np.array([[[0.1], [0.9]]])
    prediction = np.array([[[0], [1]]])
    metrics, joint = CHECKS.predictive_scores(probability, prediction, [1], [1], 723)
    np.testing.assert_allclose(metrics["probability_mean"], [0.5])
    np.testing.assert_allclose(metrics["log_score"], [np.log(0.5)])
    assert joint == pytest.approx(np.log(0.5))
    assert 0.5 <= metrics["randomized_pit"][0] <= 1


def test_joint_log_score_preserves_shared_draws():
    probability = np.array([[[0.1, 0.1], [0.9, 0.9]]])
    prediction = np.array([[[0, 0], [1, 1]]])
    metrics, joint = CHECKS.predictive_scores(probability, prediction, [1, 1], [1, 1], 727)
    assert joint == pytest.approx(np.log((0.1**2 + 0.9**2) / 2))
    assert joint != pytest.approx(sum(metrics["log_score"]))


def test_invalid_predictive_support_fails():
    with pytest.raises(ValueError, match="counts"):
        CHECKS.predictive_scores(np.array([[[0.5]]]), np.array([[[2]]]), [1], [1], 733)
