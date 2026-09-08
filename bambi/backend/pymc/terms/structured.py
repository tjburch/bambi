"""Joint Gaussian coefficient priors for structured group-specific effects."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph import ancestors
import xarray as xr

from bambi.backend.pymc.utils import get_distribution_from_prior
from bambi.covariance_math import correlation_matrix
from bambi.priors.prior import Prior


def recover_structured_offsets(term, posterior, model):
    """Invert the fitted Cholesky transform without storing matrices for every draw."""
    coefficient = model[term.label]
    chol = model[f"{term.label}_cholesky"]
    dependencies = set(ancestors([chol]))
    inputs = [coefficient] + [rv for rv in model.free_RVs if rv in dependencies]
    values = [posterior[variable.name] for variable in inputs]
    core_dims = [[dim for dim in value.dims if dim not in {"chain", "draw"}] for value in values]
    inverse = pt.linalg.solve_triangular(chol, coefficient.T, lower=True).T
    function = model.compile_fn(inverse, inputs=inputs, point_fn=False)
    return xr.apply_ufunc(
        function,
        *values,
        input_core_dims=core_dims,
        output_core_dims=[core_dims[0]],
        vectorize=True,
        output_dtypes=[float],
    )


def _parameter(term, key, default, model, lower=None, upper=None, shape=None):
    value = term.prior.get(key, default)
    name = f"{term.label}_{term.hyperprior_alias.get(key, key)}"
    if name in model:
        return model[name]
    if isinstance(value, Prior):
        if any(isinstance(arg, Prior) for arg in value.args.values()):
            raise ValueError("Covariance hyperparameters cannot have nested priors.")
        for arg in value.args.values():
            if isinstance(arg, (list, tuple, np.ndarray)):
                arg_shape = np.shape(arg)
                if shape is None and arg_shape:
                    raise ValueError(f"{key} requires a scalar prior.")
                if shape is not None and arg_shape not in {(), (shape,)}:
                    raise ValueError(f"{key} prior arguments must be scalar or shape ({shape},).")
        distribution = get_distribution_from_prior(value)
        name = f"{term.label}_{term.hyperprior_alias.get(key, key)}"
        with model:
            if lower is not None or upper is not None:
                return pm.Truncated(
                    name, distribution.dist(**value.args), lower=lower, upper=upper, shape=shape
                )
            return distribution(name, **value.args, shape=shape)
    values = np.asarray(value)
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError(f"{key} must contain finite real numbers or be a Prior.")
    if lower is not None and np.any(values <= lower):
        raise ValueError(f"{key} must be greater than {lower}.")
    if upper is not None and np.any(values >= upper):
        raise ValueError(f"{key} must be less than {upper}.")
    if shape is None and values.ndim:
        raise ValueError(f"{key} must be scalar for this covariance structure.")
    if shape is not None and values.ndim and values.shape != (shape,):
        raise ValueError(f"{key} must be scalar or have shape ({shape},).")
    return pt.as_tensor_variable(value)


def build_structured_correlation(
    term, model, coordinates=None
):  # pylint: disable=too-many-return-statements
    block = term.block
    coordinates = block.coordinates if coordinates is None else coordinates
    if block.kind in {"ar1", "ou", "toep"}:
        coordinates = np.asarray(coordinates, dtype=float)
    size = len(coordinates)
    regularizing = Prior("Normal", mu=0, sigma=0.5)
    if block.kind in {"ar1", "cs"}:
        lower = -1 if block.kind == "ar1" else -1 / (size - 1)
        rho = _parameter(term, "rho", regularizing, model, lower, 1)
        if block.kind == "ar1":
            gaps = np.abs(coordinates[:, None] - coordinates[None, :]).astype("int64")
            powers = rho ** np.maximum(gaps, 1)
            return pt.where(gaps == 0, 1.0, powers)
        return pt.eye(size) * (1 - rho) + rho
    if block.kind == "ou":
        decay = _parameter(term, "decay", Prior("Exponential", lam=1), model, 0)
        gaps = np.abs(coordinates[:, None] - coordinates[None, :])
        return pt.exp(-decay * gaps)
    if block.kind == "toep":
        partial = _parameter(term, "partial", regularizing, model, -1, 1, block.max_lag)
        partial = pt.broadcast_to(partial, (block.max_lag,))
        correlations = [pt.as_tensor_variable(1.0)]
        coefficients = pt.zeros((0,))
        innovation = pt.as_tensor_variable(1.0)
        for lag in range(1, block.max_lag + 1):
            value = partial[lag - 1]
            previous = pt.stack(correlations[1:][::-1]) if lag > 1 else pt.zeros((0,))
            correlations.append(value * innovation + pt.dot(coefficients, previous))
            coefficients = pt.concatenate([coefficients - value * coefficients[::-1], value[None]])
            innovation = innovation * (1 - value**2)
        gaps = np.abs(coordinates[:, None] - coordinates[None, :]).astype("int64")
        return pt.stack(correlations)[gaps]
    if "correlation" in term.prior:
        return pt.as_tensor_variable(
            correlation_matrix("us", coordinates, correlation=term.prior["correlation"])
        )
    eta = term.prior.get("eta", 2)
    if (
        not np.isscalar(eta)
        or np.asarray(eta).dtype.kind not in "iuf"
        or not np.isfinite(eta)
        or eta <= 0
    ):
        raise ValueError("LKJ eta must be a fixed positive scalar.")
    if size == 1:
        return pt.ones((1, 1))
    transform = pm.distributions.transforms.CholeskyCorrTransform(n=size, upper=False)
    # PyMC's inverse value-variable naming strips a single underscore-delimited suffix.
    transform.name = "corr"
    with model:
        chol = pm.LKJCorr(
            f"{term.label}_corr_cholesky", n=size, eta=eta, default_transform=transform
        )
    # PyMC 6 represents LKJCorr by its lower Cholesky factor.
    return pt.dot(chol, pt.transpose(chol))


def prediction_coefficients(term, coefficients, coordinates_new, n_new_groups, model):
    """Condition joint trajectories on fitted coefficients, then add population groups."""
    size = len(term.block.coordinates)
    if not coordinates_new:
        chol = model[f"{term.label}_cholesky"]
    else:
        coordinates = np.concatenate([term.block.coordinates, coordinates_new])
        corr = build_structured_correlation(term, model, coordinates)
        sd = _parameter(term, "sd", Prior("HalfNormal", sigma=2.5), model, lower=0)
        covariance = sd**2 * corr
        weights = pt.linalg.solve(covariance[:size, :size], covariance[:size, size:])
        conditional = covariance[size:, size:] - pt.dot(covariance[size:, :size], weights)
        conditional = (conditional + conditional.T) / 2
        offset = pm.Normal.dist(shape=(len(term.groups), len(coordinates_new)))
        draws = pt.dot(coefficients, weights) + pt.dot(offset, pt.linalg.cholesky(conditional).T)
        coefficients = pt.concatenate([coefficients, draws], axis=1)
        chol = pt.linalg.cholesky(covariance)
    if n_new_groups:
        offset = pm.Normal.dist(shape=(n_new_groups, size + len(coordinates_new)))
        coefficients = pt.concatenate([coefficients, pt.dot(offset, chol.T)], axis=0)
    return coefficients


def build_structured_distribution(term_info, param_spec, model):
    term = term_info.term
    if param_spec.ndim:
        raise NotImplementedError("Structured covariance requires a scalar linear predictor.")
    dims = tuple(term_info.factor_coords) + tuple(term_info.expression_coords)
    size = len(term.block.coordinates)
    sd = _parameter(
        term,
        "sd",
        Prior("HalfNormal", sigma=2.5),
        model,
        lower=0,
        shape=size if term.block.kind == "us" else None,
    )
    corr = build_structured_correlation(term, model)
    chol = pt.linalg.cholesky(corr)
    if sd.ndim:
        chol = sd[:, None] * chol
    else:
        chol = sd * chol
    with model:
        pm.Deterministic(f"{term.label}_cholesky", chol)
        if term.noncentered:
            offset = pm.Normal(f"{term.label}_offset", dims=dims)
            return pm.Deterministic(term.label, pt.dot(offset, chol.T), dims=dims)
        return pm.MvNormal(term.label, mu=pt.zeros(size), chol=chol, dims=dims)
