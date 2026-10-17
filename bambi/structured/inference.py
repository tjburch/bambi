"""Inference selection and exact conditional effect recovery."""

import json
import warnings
from itertools import combinations

import formulae as fm
import numpy as np
import xarray as xr

from bambi.structured import FORMULA_WRAPPERS
from bambi.structured.gaussian import transitions, temporal_conditional, coefficient_conditional
from bambi.utils import as_dataset


def all_terms(spec):
    return {
        term.qualified_name: term
        for component in spec.conditional_parameters.values()
        for term in component.structured_terms.values()
    }


def warn_confounding(spec):
    """Flag exactly matching observed random-effect designs without removing them."""

    def same_partition(left, right):
        pairs = set(zip(left, right))
        return len(pairs) == len(set(left)) == len(set(right))

    for component in spec.conditional_parameters.values():
        terms = list(component.structured_terms.values())
        for left, right in combinations(terms, 2):
            if left.temporal != right.temporal or not same_partition(
                left.group_index, right.group_index
            ):
                continue
            equivalent = (
                same_partition(left.observation_index, right.observation_index)
                if left.temporal
                else (left.data.shape == right.data.shape and np.array_equal(left.data, right.data))
            )
            if equivalent:
                warnings.warn(
                    f"Structured terms {left.name!r} and {right.name!r} have exactly "
                    "confounded observed designs. Their separate effects need prior information.",
                    UserWarning,
                )
        for term in terms:
            if term.temporal or term.dimension != 1:
                continue
            for ordinary in component.group_specific_terms.values():
                if same_partition(term.group_index, ordinary.group_index) and np.array_equal(
                    term.data[:, 0], np.asarray(ordinary.predictor).reshape(-1)
                ):
                    warnings.warn(
                        f"Random effects {term.name!r} and {ordinary.name!r} have "
                        "exactly confounded observed designs.",
                        UserWarning,
                    )


def canonical_name(name):
    """Normalize a wrapper spelling through Formulae's supported interface."""
    if not isinstance(name, str) or name.split("(", 1)[0].strip() not in FORMULA_WRAPPERS:
        return name
    model = fm.model_description("0 + " + name, formula_wrappers=FORMULA_WRAPPERS)
    structured = [term for term in model.common_terms if term.kind == "structured"]
    if len(structured) != 1:
        raise ValueError("A selector or prior key must identify one expanded structured term.")
    return structured[0].name


def normalize_mapping(mapping):
    """Normalize term keys while preserving component nesting and prior dictionaries."""
    if not isinstance(mapping, dict):
        return mapping
    result = {}
    for name, value in mapping.items():
        normalized = canonical_name(name)
        if (
            isinstance(value, dict)
            and normalized == name
            and name not in FORMULA_WRAPPERS
            and "(" not in name
        ):
            value = normalize_mapping(value)
        if normalized in result:
            raise ValueError(f"Multiple specifications identify the same term: {normalized}")
        result[normalized] = value
    return result


def normalize_prior_updates(spec, mapping):
    """Resolve aliases and qualified structured names without losing component updates."""
    mapping = normalize_mapping(mapping)
    if not isinstance(mapping, dict):
        return mapping
    terms = all_terms(spec)
    result = {}

    def destination(name, component):
        choices = []
        for term in terms.values():
            if component is not None and term.component.name != component:
                continue
            prefix = term.component.name + "."
            qualified = (
                prefix + canonical_name(name[len(prefix) :]) if name.startswith(prefix) else None
            )
            if (
                name == term.alias
                or name == prefix + term.label
                or qualified == term.qualified_name
                or (component is not None and name == term.term.name)
            ):
                choices.append(term)
        if len(choices) > 1:
            raise ValueError(
                f"Prior alias {name!r} is ambiguous. Component-qualified choices: "
                f"{[term.component.name + '.' + term.label for term in choices]}"
            )
        return choices[0] if choices else None

    def insert(target, name, value):
        if name in target:
            raise ValueError(f"Multiple specifications identify the same term: {name}")
        target[name] = value

    # Keep explicit parent-component precedence over bare parent priors.
    for name, value in mapping.items():
        if name in spec.conditional_parameters and isinstance(value, dict):
            target = result.setdefault(name, {})
            for key, prior in value.items():
                term = destination(key, name)
                insert(target, term.term.name if term else key, prior)
    for name, value in mapping.items():
        if name in spec.conditional_parameters and isinstance(value, dict):
            continue
        term = destination(name, None)
        if term is None:
            insert(result, name, value)
        elif term.component.is_parent and not name.startswith(term.component.name + "."):
            insert(result, term.term.name, value)
        else:
            insert(result.setdefault(term.component.name, {}), term.term.name, value)
    return result


def ineligibility(spec, term):  # pylint: disable=too-many-return-statements
    parent = spec.family.likelihood.parent
    if not term.component.is_parent:
        return "The structured term is in an auxiliary parameter predictor."
    if (
        spec.family.name != "gaussian"
        or spec.family.likelihood.name != "Normal"
        or spec.family.likelihood.dist is not None
    ):
        return "Exact marginal inference requires the univariate Gaussian likelihood."
    if any(
        hasattr(spec.family, hook) for hook in ("transform_backend_kwargs", "transform_backend_eta")
    ):
        return "Custom likelihood transformations require latent inference."
    if spec.family.link[parent].name != "identity":
        return "Exact marginal inference requires an identity mean link."
    response = spec.response_term
    if any(
        getattr(response, key)
        for key in ("is_censored", "is_truncated", "is_constrained", "is_weighted")
    ):
        return "The modified observation likelihood requires latent inference."
    if np.asarray(response.data).ndim != 1:
        return "Exact marginal inference requires a univariate response."
    for names, _ in spec.potentials or []:
        names = [names] if isinstance(names, str) else names
        if any(
            name
            in (
                term.name,
                term.label,
                parent,
                term.component.alias,
                term.label + "_offset",
                term.label + "_centered",
            )
            for name in names
        ):
            return "A potential depends on the selected latent effect or its predictor."
    return None


def resolve_plan(spec, method, selector):
    if method not in ("auto", "latent", "marginal"):
        raise ValueError("structured_method must be 'auto', 'latent', or 'marginal'.")
    if selector is not None and not isinstance(selector, str):
        raise ValueError("marginalize must select exactly one structured term by name.")
    if method == "latent" and selector is not None:
        raise ValueError('marginalize is incompatible with structured_method="latent".')
    terms = all_terms(spec)
    reasons = {name: ineligibility(spec, term) for name, term in terms.items()}
    eligible = [name for name in terms if reasons[name] is None]
    selected = None
    if selector is not None:
        normalized = canonical_name(selector)
        choices = []
        for name, term in terms.items():
            component_prefix = term.component.name + "."
            qualified = (
                component_prefix + canonical_name(selector[len(component_prefix) :])
                if selector.startswith(component_prefix)
                else None
            )
            if (
                normalized in (name, term.term.name, term.name, term.label)
                or qualified == name
                or selector == component_prefix + term.label
            ):
                choices.append(name)
        if len(choices) != 1:
            raise ValueError(
                f"Selector {selector!r} is unknown or ambiguous. "
                f"Component-qualified choices: {sorted(choices or terms)}"
            )
        selected = choices[0]
        if reasons[selected]:
            raise ValueError(f"Cannot marginalize {selected}: {reasons[selected]}")
    elif method != "latent" and eligible:
        selected = min(eligible, key=lambda name: (-terms[name].latent_size, name))
    elif method == "marginal":
        raise ValueError(
            f'No eligible structured term: {reasons or "model has no structured terms"}'
        )
    fallback = None
    if method == "auto" and terms and selected is None:
        fallback = "; ".join(f"{name}: {reason}" for name, reason in sorted(reasons.items()))
    return {
        "requested": method,
        "effective": "marginal" if selected else "latent",
        "selected": selected,
        "remaining": sorted(set(terms) - {selected}),
        "fallback": fallback,
    }


def recover_effect(spec, idata, random_seed):
    """Recover a joint draw for each posterior draw, then attach public metadata."""
    selected = spec.structured_plan["selected"]
    posterior = as_dataset(idata.posterior).copy()
    original_variables = set(posterior.data_vars)
    if selected is not None:
        term = all_terms(spec)[selected]
        shape = (
            posterior.sizes["chain"],
            posterior.sizes["draw"],
            *[len(values) for values in term.coords.values()],
        )
        posterior[term.label] = xr.DataArray(
            np.zeros(shape), dims=("chain", "draw", *term.dims), coords=term.coords
        )
        idata["posterior"] = posterior
        spec.predict(idata)
        posterior = as_dataset(idata.posterior).copy()
        mean_name = term.component.alias or term.component.name
        sigma_component = spec.parameters["sigma"]
        sigma_name = sigma_component.alias or "sigma"
        rng = np.random.default_rng(random_seed)
        recovered = np.empty(shape)
        y = spec.response_term.data
        for chain in range(shape[0]):
            for draw in range(shape[1]):
                sample = posterior.isel(chain=chain, draw=draw)
                residual = y - sample[mean_name].values
                sigma = sample[sigma_name].values if sigma_name in sample else sigma_component.prior
                variance = np.asarray(sigma) ** 2
                sd = sample[term.label + "_sd"].values
                if term.temporal:
                    parameter = float(
                        sample[term.label + ("_rho" if term.structure == "ar1" else "_decay")]
                    )
                    a, q = transitions(
                        term.structure, term.observation_gaps, parameter, term.observation_starts
                    )
                    states = temporal_conditional(
                        residual[term.order],
                        np.broadcast_to(variance, residual.shape)[term.order],
                        float(sd),
                        a,
                        q,
                        rng,
                    )
                    # Replicate measurements correspond to the same recovered state.
                    recovered[chain, draw, term.observation_index[term.order]] = states
                else:
                    correlation = np.eye(term.dimension)
                    if term.structure == "us" and term.dimension > 1:
                        correlation = sample[term.label + "_correlation"].values
                    elif term.structure == "cs" and term.dimension > 1:
                        rho = float(sample[term.label + "_rho"])
                        correlation = (1 - rho) * correlation + rho
                    factor = np.broadcast_to(sd, (term.dimension,))[:, None] * np.linalg.cholesky(
                        correlation
                    )
                    for group, rows in enumerate(term.group_rows):
                        recovered[chain, draw, group] = coefficient_conditional(
                            residual[rows],
                            np.broadcast_to(variance, residual.shape)[rows],
                            term.data[rows],
                            factor,
                            rng,
                        )
        posterior[term.label] = xr.DataArray(
            recovered, dims=("chain", "draw", *term.dims), coords=term.coords
        )
        posterior = posterior.drop_vars(
            list(set(posterior.data_vars) - original_variables - {term.label})
        )
        idata["posterior"] = posterior
    metadata = json.dumps(spec.structured_plan, sort_keys=True)
    idata["posterior"].attrs["structured_inference"] = metadata
    for name, term in all_terms(spec).items():
        if term.label in idata["posterior"]:
            idata["posterior"][term.label].attrs.update(
                structure=term.structure, canonical_term=name, recovered=int(name == selected)
            )
            if term.structure == "ou":
                idata["posterior"][term.label].attrs.update(
                    time_unit=term.time_unit,
                    time_origin=str(term.time_origin) if term.time_origin is not None else "",
                )
            if term.coordinate_levels is not None:
                idata["posterior"][term.label].attrs["coordinate_order"] = json.dumps(
                    [repr(level) for level in term.coordinate_levels]
                )
        for suffix, representation in (
            ("_offset", "noncentered Gaussian auxiliary"),
            ("_centered", "centered Gaussian auxiliary"),
            ("_correlation_cholesky", "lower Cholesky factor of correlation"),
            ("_correlation_beta", "LKJ canonical partial correlations: 2 * beta - 1"),
        ):
            variable = term.label + suffix
            if variable in idata["posterior"]:
                idata["posterior"][variable].attrs["structured_representation"] = representation
