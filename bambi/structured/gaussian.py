"""Exact Gaussian filtering and coefficient-block conditioning.

All likelihoods include normalizing constants. Temporal algorithms operate on
observations sorted within groups; repeated coordinates use a unit transition.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
from scipy.linalg import cho_factor, cho_solve


def transitions(structure, gaps, parameter, starts, tensor=False):
    """Return transitions and innovation variance relative to stationary variance."""
    xp = pt if tensor else np
    if structure == "ar1":
        # Do not differentiate rho**0 at rho=0.
        power = parameter ** xp.maximum(gaps, 1)
        a = xp.where(gaps == 0, 1.0, power)
        q = (1 - a) * (1 + a)
    else:
        a = xp.exp(-parameter * gaps)
        q = -xp.expm1(-2 * parameter * gaps)
    return xp.where(starts, 0.0, a), xp.where(starts, 1.0, q)


def temporal_logp(residual, variance, sd, a, q):
    """Scalar Kalman likelihood with resets at independent group boundaries."""
    residual = pt.as_tensor_variable(residual)
    variance = pt.broadcast_to(pt.as_tensor_variable(variance), residual.shape)
    a, q = pt.as_tensor_variable(a), pt.as_tensor_variable(q) * sd**2

    def step(y, r, transition, innovation, mean, covariance):
        predicted_mean = transition * mean
        predicted_variance = transition**2 * covariance + innovation
        total = predicted_variance + r
        error = y - predicted_mean
        gain = predicted_variance / total
        mean = predicted_mean + gain * error
        covariance = predicted_variance * r / total
        logp = -0.5 * (np.log(2 * np.pi) + pt.log(total) + error**2 / total)
        logp = pt.where((r > 0) & ~pt.isinf(r), logp, -np.inf)
        return mean, covariance, logp

    (_, _, increments), _ = pytensor.scan(
        step,
        sequences=[residual, variance, a, q],
        outputs_info=[pt.zeros((), dtype="float64"), pt.zeros((), dtype="float64"), None],
        n_steps=a.type.shape[0],
        strict=True,
    )
    return increments.sum()


def temporal_conditional(residual, variance, sd, a, q, rng):
    """Draw the complete joint state posterior with backward simulation."""
    n = len(residual)
    variance = np.broadcast_to(variance, (n,))
    means, covariances, predicted = np.empty(n), np.empty(n), np.empty(n)
    mean, covariance = 0.0, 0.0
    for i in range(n):
        mean = a[i] * mean
        covariance = a[i] ** 2 * covariance + sd**2 * q[i]
        predicted[i] = covariance
        total = covariance + variance[i]
        mean += covariance / total * (residual[i] - mean)
        covariance *= variance[i] / total
        means[i], covariances[i] = mean, covariance
    states = np.empty(n)
    for i in range(n - 1, -1, -1):
        mean, covariance = means[i], covariances[i]
        if i < n - 1 and a[i + 1] != 0:
            gain = covariance * a[i + 1] / predicted[i + 1]
            mean += gain * (states[i + 1] - a[i + 1] * means[i])
            covariance -= gain**2 * predicted[i + 1]
        states[i] = mean + np.sqrt(max(covariance, 0.0)) * rng.normal()
    return states


def temporal_prior_random(sd, parameter, rng=None, size=None, *, structure, gaps, starts):
    """Draw stationary Markov states for a centered PyMC distribution."""
    a, q = transitions(structure, gaps, parameter, starts)
    noise = rng.normal(size=size)
    result = np.empty_like(noise)
    previous = np.zeros_like(noise[0])
    for index, innovation in enumerate(noise):
        previous = a[index] * previous + sd * np.sqrt(q[index]) * innovation
        result[index] = previous
    return result


def coefficient_logp(residual, variance, design, cholesky):
    """Use the determinant lemma and Woodbury identity in coefficient space."""
    r = pt.broadcast_to(pt.as_tensor_variable(variance), residual.shape)
    weighted = pt.as_tensor_variable(design) @ cholesky
    precision = pt.eye(design.shape[1]) + (weighted.T / r) @ weighted
    factor = pt.linalg.cholesky(precision)
    score = weighted.T @ (residual / r)
    solved = pt.linalg.solve_triangular(factor, score, lower=True)
    logdet = pt.log(r).sum() + 2 * pt.log(pt.diag(factor)).sum()
    quadratic = (residual**2 / r).sum() - (solved**2).sum()
    value = -0.5 * (design.shape[0] * np.log(2 * np.pi) + logdet + quadratic)
    return pt.where(pt.all((r > 0) & ~pt.isinf(r)), value, -np.inf)


def coefficient_conditional(residual, variance, design, cholesky, rng):
    """Draw coefficients jointly, conditioning in the whitened prior basis."""
    variance = np.broadcast_to(variance, (len(residual),))
    weighted = design @ cholesky
    precision = np.eye(design.shape[1]) + (weighted.T / variance) @ weighted
    factor = cho_factor(precision, lower=True)
    mean = cho_solve(factor, weighted.T @ (residual / variance))
    # If P = L L.T then L.T inverse transforms a standard normal to covariance P inverse.
    noise = np.linalg.solve(np.tril(factor[0]).T, rng.normal(size=len(mean)))
    return cholesky @ (mean + noise)


def temporal_predict(times, observed_times, observed_states, sd, structure, parameter, rng):
    """Joint Markov bridge conditional on exact fitted states, without dense matrices."""
    grid = np.union1d(times, observed_times)
    if not len(grid):
        return np.empty(0)
    known = dict(zip(observed_times, observed_states))
    gaps = np.r_[0.0, np.diff(grid)]
    a, q = transitions(structure, gaps, parameter, np.arange(len(grid)) == 0)
    means, variances, predicted = np.empty(len(grid)), np.empty(len(grid)), np.empty(len(grid))
    mean, variance = 0.0, 0.0
    for i, time in enumerate(grid):
        mean, variance = a[i] * mean, a[i] ** 2 * variance + sd**2 * q[i]
        predicted[i] = variance
        if time in known:
            mean, variance = known[time], 0.0
        means[i], variances[i] = mean, variance
    result = np.empty(len(grid))
    for i in range(len(grid) - 1, -1, -1):
        mean, variance = means[i], variances[i]
        if i < len(grid) - 1 and predicted[i + 1] > 0:
            gain = variance * a[i + 1] / predicted[i + 1]
            mean += gain * (result[i + 1] - a[i + 1] * mean)
            variance -= gain**2 * predicted[i + 1]
        result[i] = mean + np.sqrt(max(variance, 0.0)) * rng.normal()
    return result[np.searchsorted(grid, times)]
