"""Independent joint Gaussian checks through the public prediction API."""

import numpy as np
import pytest

from docs.examples.group_covariance_validation.joint_prediction_check import (
    check_joint_predictions,
    conditional_moments,
    fixture,
)


@pytest.mark.parametrize("kind", ["ar1", "ou", "toep"])
@pytest.mark.parametrize("sparse", [False, True])
def test_four_block_joint_prediction(kind, sparse):
    model, inference, target = fixture(kind, sparse)
    point = inference["posterior"].to_dataset().isel(chain=0, draw=0, drop=True)
    mean, covariance = conditional_moments(model, point, target)
    np.testing.assert_allclose(mean[1], mean[2])
    np.testing.assert_allclose(covariance[1], covariance[2])
    np.testing.assert_allclose(covariance[6], covariance[7])
    np.testing.assert_allclose(covariance[4:6, :4], 0, atol=1e-12)
    assert covariance[1, 3] != 0
    assert covariance[1, 6] != 0
    assert covariance[0, 0] > 0
    result = check_joint_predictions(model, inference, target, draws=1024, sparse=sparse)
    assert result["passed"]
