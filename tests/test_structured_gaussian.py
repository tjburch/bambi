"""Independent dense Gaussian references for filtering and joint recovery."""

import numpy as np
import pytest
import pytensor
import pytensor.tensor as pt
from scipy.stats import multivariate_normal

from bambi.structured.gaussian import (
    transitions,
    temporal_logp,
    temporal_conditional,
    temporal_predict,
    coefficient_logp,
    coefficient_conditional,
)

SEED = sum(map(ord, "structured-gaussian-dense-reference"))


@pytest.mark.parametrize(
    "structure,parameter", [("ar1", -0.65), ("ar1", 0.0), ("ar1", 0.999), ("ou", 0.4)]
)
def test_temporal_likelihood_gradient_and_recovery(structure, parameter):
    times = np.array([0.0, 0.0, 2.0, 5.0])
    residual = np.array([0.4, -0.7, 1.2, 0.1])
    variance = np.array([0.1, 0.5, 0.2, 0.8])
    gaps, starts = np.r_[0.0, np.diff(times)], np.array([True, False, False, False])
    sd, par = pt.dscalar("sd"), pt.dscalar("parameter")
    a, q = transitions(structure, gaps, par, starts, tensor=True)
    value = temporal_logp(residual, variance, sd, a, q)
    evaluate = pytensor.function([sd, par], [value, *pt.grad(value, [sd, par])])

    def dense(s, p):
        distance = np.abs(times[:, None] - times)
        covariance = s**2 * (p**distance if structure == "ar1" else np.exp(-p * distance))
        return multivariate_normal.logpdf(residual, cov=covariance + np.diag(variance))

    actual = evaluate(0.8, parameter)
    assert actual[0] == pytest.approx(dense(0.8, parameter), abs=1e-9)
    delta = 1e-6
    expected_grad = [
        (dense(0.8 + delta, parameter) - dense(0.8 - delta, parameter)) / (2 * delta),
        (dense(0.8, parameter + delta) - dense(0.8, parameter - delta)) / (2 * delta),
    ]
    np.testing.assert_allclose(actual[1:], expected_grad, rtol=1e-5, atol=1e-7)
    grid = np.unique(times)
    distance = np.abs(grid[:, None] - grid)
    covariance = 0.8**2 * (
        parameter**distance if structure == "ar1" else np.exp(-parameter * distance)
    )
    design = np.eye(len(grid))[np.searchsorted(grid, times)]
    observed_cov = design @ covariance @ design.T + np.diag(variance)
    expected_mean = covariance @ design.T @ np.linalg.solve(observed_cov, residual)
    expected_cov = covariance - covariance @ design.T @ np.linalg.solve(
        observed_cov, design @ covariance
    )
    a, q = transitions(structure, gaps, parameter, starts)
    rng = np.random.default_rng(SEED)
    draws = np.array(
        [temporal_conditional(residual, variance, 0.8, a, q, rng)[[1, 2, 3]] for _ in range(6000)]
    )
    np.testing.assert_allclose(draws.mean(0), expected_mean, atol=0.025)
    np.testing.assert_allclose(np.cov(draws.T), expected_cov, atol=0.015)


def test_coefficient_likelihood_and_joint_recovery():
    design = np.array([[1.0, -1.0], [1.0, 0.5], [1.0, 2.0]])
    covariance = np.array([[0.64, -0.2], [-0.2, 0.36]])
    factor = np.linalg.cholesky(covariance)
    variance, residual = np.array([0.2, 0.4, 0.1]), np.array([0.8, -0.3, 0.5])
    value = coefficient_logp(pt.as_tensor_variable(residual), variance, design, factor).eval()
    observed_cov = design @ covariance @ design.T + np.diag(variance)
    assert value == pytest.approx(multivariate_normal.logpdf(residual, cov=observed_cov), abs=1e-10)
    expected_mean = covariance @ design.T @ np.linalg.solve(observed_cov, residual)
    expected_cov = covariance - covariance @ design.T @ np.linalg.solve(
        observed_cov, design @ covariance
    )
    rng = np.random.default_rng(SEED)
    draws = np.array(
        [coefficient_conditional(residual, variance, design, factor, rng) for _ in range(6000)]
    )
    np.testing.assert_allclose(draws.mean(0), expected_mean, atol=0.02)
    np.testing.assert_allclose(np.cov(draws.T), expected_cov, atol=0.01)


def test_joint_interpolation_extrapolation_and_repeats():
    target, observed = np.array([-1.0, 1.0, 3.0, 1.0]), np.array([0.0, 2.0])
    fitted = np.array([0.8, -0.5])
    kernel = lambda x, y: 0.7**2 * np.exp(-0.4 * np.abs(x[:, None] - y))
    cross = kernel(target, observed)
    mean = cross @ np.linalg.solve(kernel(observed, observed), fitted)
    covariance = kernel(target, target) - cross @ np.linalg.solve(
        kernel(observed, observed), cross.T
    )
    rng = np.random.default_rng(SEED)
    draws = np.array(
        [temporal_predict(target, observed, fitted, 0.7, "ou", 0.4, rng) for _ in range(6000)]
    )
    np.testing.assert_array_equal(draws[:, 1], draws[:, 3])
    np.testing.assert_allclose(draws.mean(0), mean, atol=0.025)
    np.testing.assert_allclose(np.cov(draws.T), covariance, atol=0.02)


@pytest.mark.parametrize(
    "structure,homogeneous,rho",
    [
        ("diag", False, 0.0),
        ("cs", False, -0.49),
        ("cs", True, -0.3),
        ("cs", False, 0.999),
        ("us", False, 0.25),
    ],
)
def test_coefficient_covariance_parameter_gradients(structure, homogeneous, rho):
    design = np.array([[1.0, -0.3, 0.2], [1.0, 0.5, -0.7], [1.0, 0.5, -0.7], [1.0, 0.9, 0.4]])
    residual = np.array([0.2, -0.4, 0.7, 1.1])
    variance = np.array([0.2, 0.7, 0.4, 0.1])
    scale, correlation = pt.dscalar("scale"), pt.dscalar("correlation")
    scales = scale * (np.ones(3) if homogeneous else np.array([0.7, 1.0, 1.2]))
    matrix = pt.eye(3) if structure == "diag" else (1 - correlation) * pt.eye(3) + correlation
    factor = scales[:, None] * pt.linalg.cholesky(matrix)
    logp = coefficient_logp(pt.as_tensor_variable(residual), variance, design, factor)
    params = [scale] if structure == "diag" else [scale, correlation]
    evaluate = pytensor.function(params, [logp, *pt.grad(logp, params)])

    def dense(s, r):
        scales = s * (np.ones(3) if homogeneous else np.array([0.7, 1.0, 1.2]))
        matrix = np.eye(3) if structure == "diag" else (1 - r) * np.eye(3) + r
        covariance = scales[:, None] * matrix * scales
        return multivariate_normal.logpdf(
            residual, cov=design @ covariance @ design.T + np.diag(variance)
        )

    result = evaluate(0.8) if structure == "diag" else evaluate(0.8, rho)
    delta = 1e-6
    assert result[0] == pytest.approx(dense(0.8, rho), abs=1e-9)
    expected = [(dense(0.8 + delta, rho) - dense(0.8 - delta, rho)) / (2 * delta)]
    if structure != "diag":
        expected.append((dense(0.8, rho + delta) - dense(0.8, rho - delta)) / (2 * delta))
    np.testing.assert_allclose(result[1:], expected, rtol=1e-5, atol=1e-7)


def test_grouped_filter_replicates_with_another_effect():
    groups = np.array([0, 0, 0, 1, 1, 1])
    times = np.array([0.0, 0.0, 3.0, 2.0, 4.0, 4.0])
    mean = np.array([0.2, -0.3, 0.5, 0.8, -0.1, 0.1])
    y = np.array([0.4, 1.0, -0.2, -0.7, 0.3, 0.2])
    variance = np.array([0.1, 0.5, 0.3, 0.8, 0.2, 0.9])
    starts = np.r_[True, np.diff(groups) != 0]
    gaps = np.r_[0.0, np.diff(times)]
    gaps[starts] = 0
    a, q = transitions("ar1", gaps, -0.6, starts)
    actual = temporal_logp(y - mean, variance, 0.7, a, q).eval()
    covariance = 0.7**2 * (-0.6) ** np.abs(times[:, None] - times)
    covariance *= groups[:, None] == groups
    expected = multivariate_normal.logpdf(y, mean=mean, cov=covariance + np.diag(variance))
    assert actual == pytest.approx(expected, abs=1e-10)
    observation_covariance = covariance + np.diag(variance)
    expected_mean = covariance @ np.linalg.solve(observation_covariance, y - mean)
    expected_covariance = covariance - covariance @ np.linalg.solve(
        observation_covariance, covariance
    )
    rng = np.random.default_rng(SEED)
    draws = np.array(
        [temporal_conditional(y - mean, variance, 0.7, a, q, rng) for _ in range(6000)]
    )
    np.testing.assert_allclose(draws.mean(0), expected_mean, atol=0.025)
    np.testing.assert_allclose(np.cov(draws.T), expected_covariance, atol=0.02)


def test_unstructured_partial_correlation_gradients():
    design = np.array([[1.0, -0.3, 0.2], [1.0, 0.5, -0.7], [1.0, 0.5, -0.7], [1.0, 0.9, 0.4]])
    residual, variance = np.array([0.2, -0.4, 0.7, 1.1]), np.array([0.2, 0.7, 0.4, 0.1])
    partial = pt.dvector("partial")
    a, b, c = partial[0], partial[1], partial[2]
    factor = (
        pt.stack(
            [
                pt.stack([1.0, 0.0, 0.0]),
                pt.stack([a, pt.sqrt(1 - a * a), 0.0]),
                pt.stack([b, c * pt.sqrt(1 - b * b), pt.sqrt((1 - b * b) * (1 - c * c))]),
            ]
        )
        * np.array([0.6, 0.8, 0.4])[:, None]
    )
    density = coefficient_logp(pt.as_tensor_variable(residual), variance, design, factor)
    evaluate = pytensor.function([partial], [density, pt.grad(density, partial)])

    def dense(values):
        a, b, c = values
        factor = np.array(
            [
                [1.0, 0.0, 0.0],
                [a, np.sqrt(1 - a * a), 0.0],
                [b, c * np.sqrt(1 - b * b), np.sqrt((1 - b * b) * (1 - c * c))],
            ]
        )
        factor *= np.array([0.6, 0.8, 0.4])[:, None]
        covariance = design @ factor @ factor.T @ design.T + np.diag(variance)
        return multivariate_normal.logpdf(residual, cov=covariance)

    point = np.array([-0.4, 0.2, -0.3])
    value, gradient = evaluate(point)
    assert value == pytest.approx(dense(point), abs=1e-10)
    finite_difference = np.array(
        [
            (dense(point + 1e-6 * direction) - dense(point - 1e-6 * direction)) / 2e-6
            for direction in np.eye(3)
        ]
    )
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("dimension", [2, 3, 4])
@pytest.mark.parametrize("eta", [0.5, 1.0, 2.0, 5.0])
def test_lkj_density_against_correlation_matrix_jacobian(dimension, eta):
    import bambi as bmb
    import pandas as pd
    import pymc as pm
    from scipy.special import gammaln

    data = pd.DataFrame({"y": [0.1, 0.2, -0.4, 0.3], "x": [-1.0, -0.3, 0.5, 1.2]})
    inner = "1+x" + "".join(f"+I(x**{power})" for power in range(2, dimension))
    model = bmb.Model(f"y ~ us({inner})", data, priors={f"us({inner})": {"eta": eta}})
    model.set_alias({f"us({inner})": "effect"})
    model.build()
    count = dimension * (dimension - 1) // 2
    unit = np.linspace(0.25, 0.7, count)

    def factor(values):
        result = np.eye(dimension)
        index = 0
        for row in range(dimension):
            remainder = 1.0
            for column in range(row):
                partial = 2 * values[index] - 1
                result[row, column] = partial * remainder
                remainder *= np.sqrt(1 - partial**2)
                index += 1
            result[row, row] = remainder
        return result

    def matrix_coordinates(values):
        chol = factor(values)
        return (chol @ chol.T)[np.tril_indices(dimension, -1)]

    identity = np.eye(count)
    jacobian = np.stack(
        [
            (matrix_coordinates(unit + step * 1e-6) - matrix_coordinates(unit - step * 1e-6)) / 2e-6
            for step in identity
        ],
        axis=-1,
    )
    row = np.arange(1, dimension)
    normalizer = np.sum(
        gammaln(eta + (dimension - 1) / 2)
        - gammaln(eta + (dimension - row - 1) / 2)
        - row / 2 * np.log(np.pi)
    )
    chol = factor(unit)
    expected = (
        normalizer
        + (eta - 1) * np.linalg.slogdet(chol @ chol.T)[1]
        + np.linalg.slogdet(jacobian)[1]
    )
    beta = model.backend.model["effect_correlation_beta"]
    actual = pm.logp(beta, unit).sum().eval()
    assert actual == pytest.approx(expected, abs=1e-7)
    evaluate = pytensor.function([beta], model.backend.model["effect_correlation_cholesky"])
    np.testing.assert_allclose(evaluate(unit), chol, atol=1e-12)
