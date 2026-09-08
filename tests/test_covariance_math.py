"""Deterministic checks for structured coefficient covariance."""

import numpy as np
import pytest

from bambi.covariance_math import (
    autocorrelation_from_partial,
    conditional_gaussian,
    correlation_matrix,
    covariance_matrix,
)


@pytest.mark.parametrize("rho", [-0.7, 0.0, 0.4])
def test_ar1_preserves_integer_gaps_and_permutation(rho):
    coordinates = np.array([8, 3, 5])
    expected = np.array([[1, rho**5, rho**3], [rho**5, 1, rho**2], [rho**3, rho**2, 1]])
    np.testing.assert_allclose(correlation_matrix("ar1", coordinates, rho=rho), expected)


def test_ou_uses_continuous_distance():
    actual = correlation_matrix("ou", [0, 0.25, 2.5], decay=2)
    np.testing.assert_allclose(actual[0], np.exp(-2 * np.array([0, 0.25, 2.5])))


def test_exchangeable_uses_level_count():
    actual = correlation_matrix("cs", ["a", "b", "c"], rho=-0.4)
    np.testing.assert_allclose(actual, np.eye(3) * 1.4 - 0.4)
    with pytest.raises(ValueError, match="CS rho"):
        correlation_matrix("cs", ["a", "b", "c", "d"], rho=-0.4)


def test_partial_autocorrelation_recursion():
    first, second, third = 0.3, -0.2, 0.4
    lag_two = second * (1 - first**2) + first**2
    lag_three = first * (1 - second) * lag_two + second * first
    lag_three += third * (1 - first**2) * (1 - second**2)
    np.testing.assert_allclose(
        autocorrelation_from_partial([first, second, third]),
        [1, first, lag_two, lag_three],
    )


def test_toeplitz_reduces_to_ar1():
    coordinates = [6, 2, 3]
    actual = correlation_matrix("toep", coordinates, partial=[-0.6, 0, 0, 0], max_lag=4)
    np.testing.assert_allclose(actual, correlation_matrix("ar1", coordinates, rho=-0.6))


@pytest.mark.parametrize("partial", [[0.8, -0.7, 0.6], [-0.9, -0.8, -0.7]])
def test_toeplitz_positive_definite(partial):
    actual = correlation_matrix("toep", [0, 1, 2, 3], partial=partial, max_lag=3)
    assert np.linalg.eigvalsh(actual).min() > 0
    np.testing.assert_allclose(np.diag(actual), 1)


def test_unstructured_and_marginal_scales():
    corr = correlation_matrix("us", ["Intercept", "x"], correlation=[[1, -0.2], [-0.2, 1]])
    np.testing.assert_allclose(covariance_matrix(corr, [2, 3]), [[4, -1.2], [-1.2, 9]])
    np.testing.assert_allclose(covariance_matrix(corr, 2), corr * 4)


def test_conditional_gaussian_joint_interpolation_and_batches():
    covariance = correlation_matrix("ar1", [0, 1, 2, 3], rho=0.5)
    values = np.array([[1, 2], [3, -1]])
    mean, conditional = conditional_gaussian(covariance, [0, 3], [1, 2], values)
    np.testing.assert_allclose(mean, values @ np.array([[10, 4], [4, 10]]) / 21)
    np.testing.assert_allclose(conditional, np.array([[5, 2], [2, 5]]) / 7)


def test_conditional_gaussian_empty_sets():
    covariance = np.array([[4, 1], [1, 2]])
    mean, conditional = conditional_gaussian(covariance, [], [1, 0], np.empty((3, 0)))
    np.testing.assert_array_equal(mean, np.zeros((3, 2)))
    np.testing.assert_array_equal(conditional, covariance[::-1, ::-1])
    mean, conditional = conditional_gaussian(covariance, [0], [], [2])
    assert mean.shape == (0,)
    assert conditional.shape == (0, 0)


@pytest.mark.parametrize(
    "kind,coordinates,parameters",
    [
        ("unknown", [0, 1], {}),
        ("ar1", [0, 0], {"rho": 0.3}),
        ("ar1", [0, 0.5], {"rho": 0.3}),
        ("ar1", [0, np.inf], {"rho": 0.3}),
        ("ar1", [True, False], {"rho": 0.3}),
        ("ar1", ["0", "1"], {"rho": 0.3}),
        ("ar1", [0, 1], {"rho": 1}),
        ("ar1", [0, 1], {"rho": -1}),
        ("ar1", [0, 1], {"rho": True}),
        ("ar1", [0, 1], {"rho": [0.2]}),
        ("ar1", [0, 1], {"rho": 0.3, "decay": 2}),
        ("ou", [0, 1], {"decay": 0}),
        ("ou", [0, 1], {"decay": np.nan}),
        ("cs", [0, 1, 2], {"rho": -0.5}),
        ("cs", [], {"rho": 0.2}),
        ("toep", [0, 3], {"partial": [0.2, 0.1], "max_lag": 2}),
        ("toep", [0, 1], {"partial": [0.2], "max_lag": 2}),
        ("toep", [0, 1], {"partial": [1], "max_lag": 1}),
        ("toep", [0, 1], {"partial": [0.2], "max_lag": True}),
        ("us", [0, 1], {"correlation": [[1, 1], [1, 1]]}),
        ("us", [0, 1], {"correlation": [[2, 0], [0, 1]]}),
        ("us", [0, 1], {"correlation": [[1, 0.1], [0.2, 1]]}),
    ],
)
def test_invalid_correlation_inputs(kind, coordinates, parameters):
    with pytest.raises(ValueError):
        correlation_matrix(kind, coordinates, **parameters)


@pytest.mark.parametrize("scale", [0, -1, True, [1], [1, np.nan], [[1, 2]]])
def test_invalid_scales(scale):
    with pytest.raises(ValueError):
        covariance_matrix(np.eye(2), scale)


@pytest.mark.parametrize(
    "observed,target,values",
    [
        ([0], [0], [1]),
        ([0, 0], [1], [1, 1]),
        ([0], [2], [1]),
        ([0], [-1], [1]),
        ([0.5], [1], [1]),
        ([True], [1], [1]),
        ([0], [1], [1, 2]),
        ([0], [1], [np.nan]),
    ],
)
def test_invalid_conditioning_inputs(observed, target, values):
    with pytest.raises(ValueError):
        conditional_gaussian(np.eye(2), observed, target, values)
