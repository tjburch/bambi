"""PyMC graphs for structured Gaussian effects."""

from copy import deepcopy
from functools import partial

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from bambi.backend.utils import get_distribution_from_prior, noncentered_default_for
from bambi.priors import Prior
from bambi.structured.gaussian import (
    transitions,
    temporal_logp,
    coefficient_logp,
    temporal_prior_random,
)


def parameter(term, key, dims=(), lower=None, upper=None):
    """Create an interpretable covariance parameter, including fixed values."""
    value = term.prior[key]
    label = term.label + "_" + key
    if isinstance(value, (list, tuple)) and any(isinstance(item, Prior) for item in value):
        items = []
        for index, item in enumerate(value):
            item_dims = dims[1:]
            if isinstance(item, Prior):
                distribution = get_distribution_from_prior(item)
                draw = pm.Truncated(
                    f"{label}_{index}", distribution.dist(**item.args), lower=0.0, dims=item_dims
                )
            else:
                shape = tuple(len(pm.modelcontext(None).coords[dim]) for dim in item_dims)
                draw = pt.broadcast_to(float(item), shape)
            items.append(draw)
        result = pm.Deterministic(label, pt.stack(items), dims=dims)
    elif isinstance(value, Prior):
        value = deepcopy(value)
        if key == "sd" and len(dims) > 1:
            for argument, item in value.args.items():
                if np.shape(item) == (term.dimension,):
                    value.args[argument] = np.asarray(item).reshape(
                        (term.dimension,) + (1,) * (len(dims) - 1)
                    )
        distribution = get_distribution_from_prior(value)
        if lower is not None or upper is not None:
            result = pm.Truncated(
                label, distribution.dist(**value.args), lower=lower, upper=upper, dims=dims
            )
        else:
            result = distribution(label, **value.args, dims=dims)
    else:
        shape = tuple(len(pm.modelcontext(None).coords[dim]) for dim in dims)
        value = np.asarray(value, dtype=float)
        if key == "sd" and value.shape == (term.dimension,) and len(dims) > 1:
            value = value.reshape((term.dimension,) + (1,) * (len(dims) - 1))
        result = pm.Deterministic(label, pt.broadcast_to(value, shape), dims=dims)
    return result


def build_covariance(term, response_dims=()):
    """Factor covariance once for all groups and independent response axes."""
    scales_dims = (
        response_dims if term.temporal or term.homogeneous else (term.dims[-1], *response_dims)
    )
    sd = parameter(term, "sd", scales_dims, lower=0.0)
    if term.temporal:
        key = "rho" if term.structure == "ar1" else "decay"
        value = parameter(
            term,
            key,
            response_dims,
            lower=-1.0 if key == "rho" else 0.0,
            upper=1.0 if key == "rho" else None,
        )
        return {"sd": sd, key: value}
    dimension = term.dimension
    correlation = pt.eye(dimension)
    factor = None
    rho = None
    if term.structure == "cs" and dimension > 1:
        rho = parameter(term, "rho", response_dims, lower=-1.0 / (dimension - 1), upper=1.0)
        correlation = (1 - rho[..., None, None]) * pt.eye(dimension) + rho[..., None, None]
    elif term.structure == "us" and dimension > 1:
        correlation_dims = (*response_dims, term.dims[-1], term.dims[-1] + "_aux")
        pm.modelcontext(None).add_coords({correlation_dims[-1]: term.coefficient_labels})
        if "correlation" in term.prior:
            value = np.asarray(term.prior["correlation"], dtype=float)
            correlation = pt.broadcast_to(
                value, tuple(len(pm.modelcontext(None).coords[d]) for d in correlation_dims)
            )
        else:
            # Canonical partial correlations give LKJ without a matrix-valued prior transform.
            rows, columns = np.tril_indices(dimension, -1)
            partial_dim = term.dims[-1] + "_partial"
            pm.modelcontext(None).add_coords(
                {
                    partial_dim: [
                        f"{term.coefficient_labels[i]}:{term.coefficient_labels[j]}"
                        for i, j in zip(rows, columns)
                    ]
                }
            )
            alpha = term.prior["eta"] + (dimension - columns - 2) / 2
            beta_dims = (*response_dims, partial_dim)
            unit = pm.Beta(
                term.label + "_correlation_beta",
                alpha=alpha,
                beta=alpha,
                dims=beta_dims,
                shape=tuple(len(pm.modelcontext(None).coords[d]) for d in beta_dims),
            )
            partial_correlations = 2 * unit - 1
            factor_rows = []
            index = 0
            for row in range(dimension):
                remainder = pt.ones_like(partial_correlations[..., 0])
                entries = []
                for _ in range(row):
                    entries.append(partial_correlations[..., index] * remainder)
                    remainder = remainder * pt.sqrt(
                        (1 - partial_correlations[..., index])
                        * (1 + partial_correlations[..., index])
                    )
                    index += 1
                entries.append(remainder)
                entries.extend([pt.zeros_like(remainder)] * (dimension - row - 1))
                factor_rows.append(pt.stack(entries, axis=-1))
            factor = pm.Deterministic(
                term.label + "_correlation_cholesky",
                pt.stack(factor_rows, axis=-2),
                dims=correlation_dims,
            )
            correlation = factor @ pt.swapaxes(factor, -1, -2)
        correlation = pm.Deterministic(
            term.label + "_correlation", correlation, dims=correlation_dims
        )
    if factor is None:
        factor = pt.linalg.cholesky(correlation)
    scales = (
        sd if term.homogeneous else (sd.transpose(*range(1, sd.ndim), 0) if response_dims else sd)
    )
    factor = factor * (scales[..., None, None] if term.homogeneous else scales[..., :, None])
    return {"sd": sd, "cholesky": factor, "rho": rho}


def build_term(term, backend, spec, marginalized=False):
    model = backend.model
    for dim, values in term.coords.items():
        model.add_coords({dim: values})
    response_dims = tuple(spec.response_term.coords)
    covariance = build_covariance(term, response_dims)
    backend.structured_covariances[term.qualified_name] = covariance
    if marginalized:
        return 0
    dims = (*term.dims, *response_dims)
    noncentered = noncentered_default_for(spec, term.component.name)
    prior_values = [
        item
        for value in term.prior.values()
        for item in (value if isinstance(value, (list, tuple)) else [value])
    ]
    overrides = {
        value.noncentered
        for value in prior_values
        if isinstance(value, Prior) and value.noncentered is not None
    }
    if len(overrides) > 1:
        raise ValueError(
            "Covariance priors specify conflicting parameterizations for one Gaussian vector."
        )
    if overrides:
        noncentered = overrides.pop()
    if term.temporal:
        parameter_value = covariance["rho" if term.structure == "ar1" else "decay"]
        expand = (slice(None),) + (None,) * len(response_dims)
        a, q = transitions(
            term.structure, term.gaps[expand], parameter_value, term.starts[expand], tensor=True
        )
        innovation = covariance["sd"] * pt.sqrt(q)
        if noncentered:
            offsets = pm.Normal(term.label + "_offset", 0, 1, dims=dims)
            offsets = pt.specify_shape(offsets, tuple(len(model.coords[d]) for d in dims))
            states, _ = pytensor.scan(
                lambda z, transition, scale, previous: transition * previous + scale * z,
                sequences=[offsets, a, innovation],
                outputs_info=[pt.zeros_like(offsets[0])],
                n_steps=term.dimension,
                strict=True,
            )
            effect = pm.Deterministic(term.label, states, dims=dims)
        else:

            def logp(value, sd, par):
                transition, variance = transitions(
                    term.structure, term.gaps[expand], par, term.starts[expand], tensor=True
                )
                previous = pt.concatenate([pt.zeros_like(value[:1]), value[:-1]])
                return pm.logp(
                    pm.Normal.dist(transition * previous, sd * pt.sqrt(variance)), value
                ).sum()

            random = partial(
                temporal_prior_random,
                structure=term.structure,
                gaps=term.gaps[expand],
                starts=term.starts[expand],
            )
            effect = pm.CustomDist(
                term.label,
                covariance["sd"],
                parameter_value,
                logp=logp,
                random=random,
                shape=tuple(len(model.coords[d]) for d in dims),
                dims=dims,
            )
        return effect[term.observation_index]
    factor = covariance["cholesky"]
    if noncentered:
        offset = pm.Normal(term.label + "_offset", 0, 1, dims=dims)
        if response_dims:
            value = pt.einsum("...ij,gj...->gi...", factor, offset)
        else:
            value = offset @ factor.T
        effect = pm.Deterministic(term.label, value, dims=dims)
    else:
        if response_dims:
            response_shape = tuple(len(model.coords[d]) for d in response_dims)
            centered = pm.MvNormal(
                term.label + "_centered",
                mu=0,
                chol=factor,
                dims=(term.dims[0], *response_dims, term.dims[1]),
                shape=(len(term.groups), *response_shape, term.dimension),
            )
            axes = (0, len(response_dims) + 1, *range(1, len(response_dims) + 1))
            effect = pm.Deterministic(term.label, pt.transpose(centered, axes), dims=dims)
        else:
            effect = pm.MvNormal(term.label, mu=0, chol=factor, dims=dims)
    predictor = term.data[(slice(None), slice(None)) + (None,) * len(response_dims)]
    return (effect[term.group_index] * predictor).sum(axis=1)


def build_marginal_response(term, backend, kwargs, name):
    """One exact joint Gaussian likelihood conditional on every other effect."""
    covariance = backend.structured_covariances[term.qualified_name]
    observed = kwargs["observed"]
    mu = pt.broadcast_to(kwargs["mu"], observed.shape)
    variance = pt.broadcast_to(kwargs["sigma"] ** 2, observed.shape)
    if term.temporal:
        sd, par = covariance["sd"], covariance["rho" if term.structure == "ar1" else "decay"]

        def temporal_density(value, mean, residual_variance, scale, parameter_value):
            a, q = transitions(
                term.structure,
                term.observation_gaps,
                parameter_value,
                term.observation_starts,
                tensor=True,
            )
            return temporal_logp(
                (value - mean)[term.order], residual_variance[term.order], scale, a, q
            )

        return pm.CustomDist(
            name,
            mu,
            variance,
            sd,
            par,
            logp=temporal_density,
            signature="(n),(n),(),()->(n)",
            observed=observed,
            dims=kwargs["dims"],
        )

    def coefficient_density(value, mean, residual_variance, factor):
        residual = value - mean
        return sum(
            coefficient_logp(residual[rows], residual_variance[rows], term.data[rows], factor)
            for rows in term.group_rows
        )

    return pm.CustomDist(
        name,
        mu,
        variance,
        covariance["cholesky"],
        logp=coefficient_density,
        signature="(n),(n),(k,k)->(n)",
        observed=observed,
        dims=kwargs["dims"],
    )
