"""Fitted indexing and priors for structured Gaussian effects."""

from copy import deepcopy

import numpy as np
import pandas as pd
from formulae.categorical import CategoricalBox

from bambi.priors import Prior
from bambi.structured.gaussian import temporal_predict

TIME_UNITS = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0, "days": 86400.0, "weeks": 604800.0}


def group_keys(frame):
    """Preserve both identifier type and value in composite keys."""
    if len(frame.columns) == 0:
        return [()] * len(frame.index)

    def typed_value(value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, tuple):
            return ("tuple", tuple(typed_value(item) for item in value))
        try:
            hash(value)
        except TypeError as error:
            raise ValueError(
                "Structured group identifiers must be hashable scalars or tuples."
            ) from error
        if pd.isna(value) or (isinstance(value, (float, int)) and not np.isfinite(value)):
            raise ValueError(
                f"Structured grouping columns {list(frame.columns)} require "
                "finite, non-missing values."
            )
        return (type(value).__name__, value)

    keys = []
    for row in frame.itertuples(index=False, name=None):
        keys.append(tuple(typed_value(value) for value in row))
    return keys


class StructuredTerm:  # pylint: disable=too-many-instance-attributes
    """One Gaussian vector or process per observed group, with shared covariance."""

    def __init__(self, term, prior, prefix, component):
        self.term = term
        self.prefix = prefix
        self.component = component
        self.alias = None
        self.prior = prior
        self.structure = term.wrapper
        self.temporal = term.mode == "coordinate"
        self.kind = "structured"
        self.categorical = False
        self.levels = None
        self.options = term.options.copy()
        allowed = (
            {"time_unit"}
            if self.structure == "ou"
            else ({"homogeneous"} if self.structure == "cs" else set())
        )
        unknown = set(self.options) - allowed
        if unknown:
            raise ValueError(f"Unknown options for {self.structure}: {sorted(unknown)}")
        if "homogeneous" in self.options and not isinstance(self.options["homogeneous"], bool):
            raise ValueError("homogeneous must be a boolean.")
        self.homogeneous = self.options.get("homogeneous", False)
        self.time_unit = self.options.get("time_unit")
        if self.time_unit is not None and self.time_unit not in TIME_UNITS:
            raise ValueError(f"time_unit must be one of {list(TIME_UNITS)}.")
        self.coordinate_levels = None
        self.time_origin = None
        self.timezone = None
        keys = group_keys(term.group_values)
        self.groups = sorted(set(keys), key=repr)
        lookup = {key: index for index, key in enumerate(self.groups)}
        self.group_index = np.array([lookup[key] for key in keys], dtype=int)
        sorted_rows = np.argsort(self.group_index, kind="stable")
        splits = np.flatnonzero(np.diff(self.group_index[sorted_rows])) + 1
        self.group_rows = np.split(sorted_rows, splits)
        if self.temporal:
            self.times = self.coordinate_values(term.coordinate, fitting=True)
            if len(self.times) != len(self.group_index):
                raise ValueError(
                    "A structured coordinate expression must return one value per observation."
                )
            self.state_keys = sorted(set(zip(self.group_index, self.times)))
            lookup = {key: i for i, key in enumerate(self.state_keys)}
            self.observation_index = np.array(
                [lookup[key] for key in zip(self.group_index, self.times)]
            )
            self.state_groups = np.array([key[0] for key in self.state_keys])
            self.state_times = np.array([key[1] for key in self.state_keys])
            self.starts = np.r_[True, np.diff(self.state_groups) != 0]
            self.gaps = np.r_[0.0, np.diff(self.state_times)]
            self.gaps[self.starts] = 0
            self.order = np.lexsort((self.times, self.group_index))
            sorted_times, sorted_groups = self.times[self.order], self.group_index[self.order]
            self.observation_starts = np.r_[True, np.diff(sorted_groups) != 0]
            self.observation_gaps = np.r_[0.0, np.diff(sorted_times)]
            self.observation_gaps[self.observation_starts] = 0
            self.dimension = len(self.state_keys)
            self.coefficient_labels = None
            self.data = self.times
        else:
            self.data = np.asarray(term.design.design_matrix, dtype=float)
            self.dimension = self.data.shape[1]
            if not self.dimension or not np.isfinite(self.data).all():
                raise ValueError("Structured coefficient designs must be nonempty and finite.")
            self.coefficient_labels = [
                label for t in term.design.terms.values() for label in t.labels
            ]
        self.shape = (self.latent_size,)

    @property
    def name(self):
        return f"{self.prefix}_{self.term.name}" if self.prefix else self.term.name

    @property
    def label(self):
        return self.alias or self.name

    @property
    def qualified_name(self):
        return f"{self.component.name}.{self.term.name}"

    @property
    def latent_size(self):
        return self.dimension if self.temporal else self.dimension * len(self.groups)

    @property
    def dims(self):
        # Nutpie transports dimension names as comma-separated metadata.
        label = self.label.replace("%", "%25").replace(",", "%2C")
        if self.temporal:
            return (label + "__state",)
        return (label + "__group", label + "__coefficient")

    @property
    def coords(self):
        if self.temporal:
            return {
                self.dims[0]: [
                    repr((self.groups[group], self._coordinate_label(time)))
                    for group, time in self.state_keys
                ]
            }
        return {self.dims[0]: [repr(g) for g in self.groups], self.dims[1]: self.coefficient_labels}

    def _coordinate_label(self, time):
        if self.coordinate_levels is not None:
            return self.coordinate_levels[int(time)]
        if self.time_origin is not None:
            elapsed = pd.to_timedelta(time * TIME_UNITS[self.time_unit], unit="s")
            return (self.time_origin + elapsed).isoformat()
        return time

    def coordinate_values(self, values, fitting=False):
        if isinstance(values, CategoricalBox):
            if values.levels is not None:
                values = pd.Series(
                    pd.Categorical(values.data, categories=values.levels, ordered=True)
                )
            else:
                values = pd.Series(values.data)
        values = pd.Series(values)
        if self.structure == "ar1":
            if fitting and isinstance(values.dtype, pd.CategoricalDtype):
                if not values.cat.ordered:
                    if not pd.api.types.is_numeric_dtype(values.cat.categories):
                        raise ValueError("AR(1) categorical coordinates require a declared order.")
                    values = pd.Series(np.asarray(values))
                else:
                    self.coordinate_levels = values.cat.categories.tolist()
            if (
                self.coordinate_levels is None
                and isinstance(values.dtype, pd.CategoricalDtype)
                and pd.api.types.is_numeric_dtype(values.cat.categories)
            ):
                values = pd.Series(np.asarray(values))
            if self.coordinate_levels is not None:
                codes = pd.Categorical(values, categories=self.coordinate_levels).codes
                if (codes < 0).any():
                    raise ValueError("AR(1) coordinates must use the fitted declared categories.")
                return codes.astype(float)
            if not pd.api.types.is_numeric_dtype(values):
                raise ValueError(
                    "AR(1) requires integer coordinates or an ordered categorical grid."
                )
            result = values.to_numpy(dtype=float)
            if not np.isfinite(result).all() or not np.equal(result, np.floor(result)).all():
                raise ValueError(
                    "AR(1) coordinates must be finite integers; values are not rounded."
                )
            return result
        if isinstance(values.dtype, pd.CategoricalDtype) and pd.api.types.is_numeric_dtype(
            values.cat.categories
        ):
            values = pd.Series(np.asarray(values))
        if pd.api.types.is_datetime64_any_dtype(values):
            dates = pd.DatetimeIndex(values)
            if fitting:
                self.timezone = str(dates.tz) if dates.tz is not None else None
                self.time_origin = dates.min()
                self.time_unit = self.time_unit or "days"
            if self.time_origin is None or (dates.tz is None) != (self.timezone is None):
                raise ValueError(
                    "OU prediction datetimes must match the fitted naive/aware time domain."
                )
            result = (dates - self.time_origin).total_seconds().to_numpy() / TIME_UNITS[
                self.time_unit
            ]
        else:
            if self.time_origin is not None or not pd.api.types.is_numeric_dtype(values):
                raise ValueError("OU requires numeric coordinates or compatible datetimes.")
            result = values.to_numpy(dtype=float)
            if fitting:
                self.time_unit = self.time_unit or "native numeric units"
        if not np.isfinite(result).all():
            raise ValueError("OU coordinates must be finite and non-missing.")
        return result

    def build_priors(self):
        if self.prior is not None and not isinstance(self.prior, dict):
            raise ValueError(f"{self.name} requires a covariance-parameter prior dictionary.")
        supplied = deepcopy(self.prior or {})
        allowed = {"sd"}
        if self.structure in ("ar1", "cs"):
            if self.structure == "ar1" or self.dimension > 1:
                allowed.add("rho")
        elif self.structure == "ou":
            allowed.add("decay")
        elif self.structure == "us":
            allowed |= {"eta", "correlation"}
        if set(supplied) - allowed:
            raise ValueError(
                f"Unknown covariance prior keys for {self.name}: {sorted(set(supplied)-allowed)}"
            )
        if "eta" in supplied and "correlation" in supplied:
            raise ValueError("Specify either eta or correlation, not both.")
        spec = self.component.spec
        scale = 1.0
        if (
            spec.auto_scale
            and self.component.is_parent
            and spec.family.name == "gaussian"
            and spec.family.link[self.component.name].name == "identity"
        ):
            scale = float(np.std(spec.response_term.data)) or 1.0
        sd_scale = 2.5 * scale
        if spec.auto_scale and not self.temporal and not self.homogeneous:
            std = self.data.std(axis=0)
            sd_scale = sd_scale / np.where(std > 0, std, 1.0)
        defaults = {"sd": Prior("HalfNormal", sigma=sd_scale)}
        if "rho" in allowed:
            defaults["rho"] = Prior("Normal", mu=0, sigma=0.5)
        if "decay" in allowed:
            positive = self.gaps[(self.gaps > 0) & ~self.starts]
            if not len(positive):
                positive = np.diff(np.unique(self.times))
            reference = np.median(positive) if len(positive) else 1.0
            defaults["decay"] = Prior("Exponential", lam=reference if spec.auto_scale else 1.0)
        if "eta" in allowed and "correlation" not in supplied:
            defaults["eta"] = 2.0
        # Preserve the original specification so repeated prior preparation never rescales it.
        self.prior = {**defaults, **supplied}
        rho = self.prior.get("rho")
        if isinstance(rho, Prior) and any(np.ndim(value) for value in rho.args.values()):
            raise ValueError(
                "rho prior parameters must be scalar; correlation is shared across groups."
            )
        for key in ("sd", "decay"):
            value = self.prior.get(key)
            expected = (
                () if key == "decay" or self.temporal or self.homogeneous else (self.dimension,)
            )
            if isinstance(value, Prior):
                for argument in value.args.values():
                    if isinstance(argument, (list, tuple, np.ndarray)) and np.shape(
                        argument
                    ) not in ((), expected):
                        raise ValueError(
                            f"{key} prior parameters must be scalar or have shape {expected}."
                        )
                continue
            if value is None:
                continue
            if (
                key == "sd"
                and isinstance(value, (list, tuple))
                and any(isinstance(item, Prior) for item in value)
            ):
                if expected != (len(value),):
                    raise ValueError(f"sd requires {self.dimension} coefficient specifications.")
                for item in value:
                    if isinstance(item, Prior) and any(np.ndim(arg) for arg in item.args.values()):
                        raise ValueError("Each coefficient SD prior must have scalar parameters.")
                    if not isinstance(item, Prior) and (
                        not np.isscalar(item) or not np.isfinite(item) or item <= 0
                    ):
                        raise ValueError("Every fixed coefficient SD must be positive and finite.")
                continue
            array = np.asarray(value, dtype=float)
            if array.shape not in ((), expected) or not np.all(np.isfinite(array) & (array > 0)):
                raise ValueError(f"{key} must be positive and have shape {expected} or be scalar.")
        if "rho" in self.prior and not isinstance(self.prior["rho"], Prior):
            lower = -1.0 if self.structure == "ar1" else -1.0 / (self.dimension - 1)
            rho = np.asarray(self.prior["rho"])
            if rho.ndim or not np.isfinite(rho) or not lower < rho < 1:
                raise ValueError(f"rho must be strictly between {lower} and 1.")
        if "eta" in self.prior:
            eta = self.prior["eta"]
            if not np.isscalar(eta) or not np.isfinite(eta) or eta <= 0:
                raise ValueError("eta must be a positive fixed scalar.")
        if "correlation" in self.prior:
            correlation = np.asarray(self.prior["correlation"], dtype=float)
            if (
                correlation.shape != (self.dimension, self.dimension)
                or not np.isfinite(correlation).all()
                or not np.allclose(correlation, correlation.T)
                or not np.allclose(np.diag(correlation), 1)
            ):
                raise ValueError(
                    "correlation must be symmetric with unit diagonal and the fitted dimension."
                )
            try:
                np.linalg.cholesky(correlation)
            except np.linalg.LinAlgError as error:
                raise ValueError("correlation must be positive definite.") from error

    def predict(self, posterior, data, sample_new_groups, random_seed):
        if data is None:
            values = posterior[self.label].transpose("chain", "draw", *self.dims, ...).values
            result = (
                values[:, :, self.observation_index]
                if self.temporal
                else np.einsum("cdnk...,nk->cdn...", values[:, :, self.group_index], self.data)
            )
            return result
        raw, frame = self.term.evaluate_new_data(data)
        keys = group_keys(frame)
        target_groups = sorted(set(keys), key=repr)
        unknown = set(target_groups) - set(self.groups)
        if unknown and not sample_new_groups:
            raise ValueError(
                "Prediction contains new structured groups. Set sample_new_groups=True."
            )
        times = self.coordinate_values(raw) if self.temporal else None
        if self.temporal and len(times) != len(data):
            raise ValueError(
                "A structured coordinate expression must return one value per observation."
            )
        design = None if self.temporal else np.asarray(raw)
        source = posterior[self.label].transpose("chain", "draw", *self.dims, ...)
        extra_dims = tuple(d for d in source.dims if d not in ("chain", "draw", *self.dims))
        extra_shape = tuple(source.sizes[d] for d in extra_dims)
        result = np.empty((source.sizes["chain"], source.sizes["draw"], len(data), *extra_shape))
        rng = np.random.default_rng(random_seed)
        for chain in range(source.sizes["chain"]):
            for draw in range(source.sizes["draw"]):
                for axis in np.ndindex(extra_shape):

                    def parameter(name, chain=chain, draw=draw, axis=axis):
                        value = posterior[self.label + "_" + name].isel(chain=chain, draw=draw)
                        for dim, index in zip(extra_dims, axis):
                            if dim in value.dims:
                                value = value.isel({dim: index})
                        return value.values

                    sd = parameter("sd")
                    for group in target_groups:
                        rows = np.array(
                            [i for i, key in enumerate(keys) if key == group], dtype=int
                        )
                        index = self.groups.index(group) if group in self.groups else None
                        effects = (
                            source.values[(chain, draw, slice(None), *axis)]
                            if self.temporal
                            else source.values[(chain, draw, slice(None), slice(None), *axis)]
                        )
                        if self.temporal:
                            known = (
                                self.state_groups == index
                                if index is not None
                                else np.zeros(self.dimension, dtype=bool)
                            )
                            prediction = temporal_predict(
                                times[rows],
                                self.state_times[known],
                                effects[known],
                                float(sd),
                                self.structure,
                                float(parameter("rho" if self.structure == "ar1" else "decay")),
                                rng,
                            )
                        else:
                            if index is None:
                                correlation = (
                                    parameter("correlation")
                                    if self.structure == "us" and self.dimension > 1
                                    else np.eye(self.dimension)
                                )
                                if self.structure == "cs" and self.dimension > 1:
                                    rho = float(parameter("rho"))
                                    correlation = (1 - rho) * np.eye(self.dimension) + rho
                                coefficients = np.broadcast_to(sd, (self.dimension,)) * (
                                    np.linalg.cholesky(correlation)
                                    @ rng.normal(size=self.dimension)
                                )
                            else:
                                coefficients = effects[index]
                            prediction = design[rows] @ coefficients
                        result[(chain, draw, rows, *axis)] = prediction
        return result

    def __str__(self):
        units = f", time_unit={self.time_unit!r}" if self.structure == "ou" else ""
        return (
            f"StructuredTerm(name={self.label!r}, structure={self.structure!r}, "
            f"coefficients={self.latent_size}, prior={self.prior}{units})"
        )

    __repr__ = __str__
