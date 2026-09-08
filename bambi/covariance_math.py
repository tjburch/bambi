"""Numerical covariance and prediction calculations for group-specific effects."""

import numpy as np


def _numeric(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite real numbers.")
    return array.astype(float)


def _scalar(value, name):
    array = _numeric(value, name)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    return float(array)


def _positive_definite(matrix, name):
    matrix = _numeric(matrix, name)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size:
        raise ValueError(f"{name} must be a nonempty square matrix.")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{name} must be symmetric.")
    matrix = (matrix + matrix.T) / 2
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite.") from error
    return matrix


def autocorrelation_from_partial(partial):
    """Convert partial autocorrelations to lags 0..L using Levinson recursion."""
    partial = _numeric(partial, "partial")
    if partial.ndim != 1 or np.any(np.abs(partial) >= 1):
        raise ValueError("partial must be a vector with entries strictly between -1 and 1.")
    correlations = np.ones(len(partial) + 1)
    coefficients = np.empty(0)
    innovation = 1.0
    for lag, value in enumerate(partial, start=1):
        correlations[lag] = value * innovation + coefficients @ correlations[1:lag][::-1]
        coefficients = np.append(coefficients - value * coefficients[::-1], value)
        innovation *= 1 - value**2
    return correlations


def correlation_matrix(
    kind, coordinates, rho=None, decay=None, partial=None, correlation=None, max_lag=None
):
    """Construct a correlation matrix on unique coefficient coordinates.

    AR1 and Toeplitz use actual integer distances, not coordinate positions.
    Exchangeable and unstructured coordinates may be categorical labels.
    Toeplitz support is finite: the largest requested distance cannot exceed
    ``max_lag``, and one partial autocorrelation is required per supported lag.
    """
    if kind not in {"ar1", "ou", "cs", "toep", "us"}:
        raise ValueError(f"Unknown covariance structure: {kind!r}.")
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 1 or not coordinates.size:
        raise ValueError("coordinates must be a nonempty vector.")
    if coordinates.dtype.kind in "iuf":
        _numeric(coordinates, "coordinates")
    elif coordinates.dtype.kind not in "US":
        if coordinates.dtype.kind != "O" or not all(
            isinstance(value, str) for value in coordinates
        ):
            raise ValueError("coordinates must contain real numbers or string labels.")
    if len(np.unique(coordinates)) != len(coordinates):
        raise ValueError("coordinates must be unique.")
    size = len(coordinates)
    supplied = {
        "rho": rho,
        "decay": decay,
        "partial": partial,
        "correlation": correlation,
        "max_lag": max_lag,
    }
    allowed = {
        "ar1": {"rho"},
        "ou": {"decay"},
        "cs": {"rho"},
        "toep": {"partial", "max_lag"},
        "us": {"correlation"},
    }
    for name, value in supplied.items():
        if value is not None and name not in allowed[kind]:
            raise ValueError(f"{name} is not a parameter of {kind}.")
    if kind in {"ar1", "ou", "toep"}:
        coordinates = _numeric(coordinates, "coordinates")
        distances = np.abs(coordinates[:, None] - coordinates[None, :])
        if not np.all(np.isfinite(distances)):
            raise ValueError("Coordinate distances must be finite.")
        if kind != "ou":
            if np.any(coordinates != np.floor(coordinates)):
                raise ValueError(f"{kind} requires integer coordinates.")
            if np.any(distances >= np.iinfo(np.int64).max):
                raise ValueError("Integer coordinate distances are too large.")
            distances = distances.astype(np.int64)
    if kind == "ar1":
        rho = _scalar(rho, "rho")
        if not -1 < rho < 1:
            raise ValueError("AR1 rho must be strictly between -1 and 1.")
        result = rho**distances
    elif kind == "ou":
        decay = _scalar(decay, "decay")
        if decay <= 0:
            raise ValueError("OU decay must be positive.")
        result = np.exp(-decay * distances)
    elif kind == "cs":
        rho = _scalar(rho, "rho")
        lower = -1 / (size - 1) if size > 1 else -1
        if not lower < rho < 1:
            raise ValueError(f"CS rho must be strictly between {lower} and 1.")
        result = np.full((size, size), rho)
        np.fill_diagonal(result, 1)
    elif kind == "toep":
        if isinstance(max_lag, (bool, np.bool_)) or not isinstance(max_lag, (int, np.integer)):
            raise ValueError("max_lag must be a nonnegative integer.")
        if max_lag < 0:
            raise ValueError("max_lag must be a nonnegative integer.")
        lags = autocorrelation_from_partial(partial)
        if len(lags) != max_lag + 1:
            raise ValueError("partial must have exactly max_lag entries.")
        if np.max(distances) > max_lag:
            raise ValueError("Requested coordinates exceed the Toeplitz max_lag horizon.")
        result = lags[distances]
    else:
        result = _positive_definite(correlation, "correlation")
        if result.shape != (size, size) or not np.allclose(np.diag(result), 1, rtol=0, atol=1e-12):
            raise ValueError("correlation must match coordinates and have a unit diagonal.")
    return result


def covariance_matrix(correlation, scale):
    """Apply marginal standard deviations: covariance = D correlation D."""
    correlation = _positive_definite(correlation, "correlation")
    if not np.allclose(np.diag(correlation), 1, rtol=0, atol=1e-12):
        raise ValueError("correlation must have a unit diagonal.")
    scale = _numeric(scale, "scale")
    if scale.ndim > 1 or (scale.ndim == 1 and scale.shape != (len(correlation),)):
        raise ValueError("scale must be scalar or have one entry per coefficient.")
    if np.any(scale <= 0):
        raise ValueError("scale must be positive.")
    scale = np.broadcast_to(scale, (len(correlation),))
    return scale[:, None] * correlation * scale[None, :]


def conditional_gaussian(covariance, observed_indices, target_indices, observed_values):
    """Condition a zero-mean Gaussian using solves, with observations on the last axis.

    Index sets must be disjoint. Empty observed or target sets are supported.
    The returned covariance is shared by all batches of observed values.
    """
    covariance = _positive_definite(covariance, "covariance")
    indices = []
    named_indices = (("observed_indices", observed_indices), ("target_indices", target_indices))
    for name, values in named_indices:
        array = np.asarray(values)
        if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
            raise ValueError(f"{name} must be an integer vector.")
        if np.any(array < 0) or np.any(array >= len(covariance)):
            raise ValueError(f"{name} are out of bounds.")
        if len(np.unique(array)) != len(array):
            raise ValueError(f"{name} must be unique.")
        indices.append(array.astype(np.intp))
    observed, target = indices[0], indices[1]
    if np.intersect1d(observed, target).size:
        raise ValueError("Observed and target indices must be disjoint.")
    values = _numeric(observed_values, "observed_values")
    if values.ndim == 0 or values.shape[-1] != len(observed):
        raise ValueError("The last observed_values axis must match observed_indices.")
    target_covariance = covariance[np.ix_(target, target)]
    if not observed.size:
        return np.zeros(values.shape[:-1] + (len(target),)), target_covariance
    cross_covariance = covariance[np.ix_(observed, target)]
    weights = np.linalg.solve(covariance[np.ix_(observed, observed)], cross_covariance)
    mean = values @ weights
    conditional = target_covariance - cross_covariance.T @ weights
    return mean, (conditional + conditional.T) / 2
